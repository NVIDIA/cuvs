/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * Helpers and factories for the dense datasets: the padded row width, views over existing
 * matrices, and owning copies. None of this knows about any particular algorithm; the algorithms
 * that need padded rows ask for a padded dataset and use these to get one.
 */

#include <cuvs/core/dataset/dense.hpp>
#include <cuvs/core/dataset/traits.hpp>
#include <cuvs/core/export.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/memory_type.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>   // get_device_for_address, copy_matrix
#include <raft/util/integer_utils.hpp>  // rounding up

#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <type_traits>

namespace CUVS_EXPORT cuvs {

// -----------------------------------------------------------------------------
// Padded row width in elements (shared by make_*_padded_dataset* and the index layout checks).
// -----------------------------------------------------------------------------

/**
 * @brief Required row width in elements of a padded dataset: the minimum leading dimension (LDA)
 *        per row for the default per-row byte alignment (16 bytes, combined with `sizeof` element
 *        type), given `logical_columns` feature columns.
 */
[[nodiscard]] inline uint32_t padded_row_width(uint32_t logical_columns,
                                               std::size_t sizeof_value,
                                               uint32_t align_bytes = 16)
{
  return static_cast<uint32_t>(
    raft::round_up_safe<std::size_t>(static_cast<std::size_t>(logical_columns) * sizeof_value,
                                     std::lcm(align_bytes, static_cast<uint32_t>(sizeof_value))) /
    sizeof_value);
}

template <typename ValueT>
[[nodiscard]] inline uint32_t padded_row_width(uint32_t logical_columns, uint32_t align_bytes = 16)
{
  return padded_row_width(logical_columns, sizeof(ValueT), align_bytes);
}

/** Actual row width in elements (leading dimension) of a 2D row-major matrix view. */
template <typename T, typename I, typename L>
[[nodiscard]] inline uint32_t matrix_actual_row_width(raft::device_matrix_view<T, I, L> m)
{
  return m.stride(0) > 0 ? static_cast<uint32_t>(m.stride(0)) : static_cast<uint32_t>(m.extent(1));
}

template <typename T, typename I, typename L>
[[nodiscard]] inline uint32_t matrix_actual_row_width(raft::host_matrix_view<T, I, L> m)
{
  return m.stride(0) > 0 ? static_cast<uint32_t>(m.stride(0)) : static_cast<uint32_t>(m.extent(1));
}

/**
 * @brief True if the matrix's row width in elements matches `padded_row_width` for
 *        `m.extent(1)` and element type `T` (the padded row layout is satisfied for this view).
 */
template <typename T, typename I, typename L>
[[nodiscard]] inline bool matrix_has_padded_row_width(raft::device_matrix_view<T, I, L> m,
                                                      uint32_t align_bytes = 16)
{
  using value_type = std::remove_const_t<T>;
  const uint32_t need =
    padded_row_width<value_type>(static_cast<uint32_t>(m.extent(1)), align_bytes);
  return matrix_actual_row_width(m) == need;
}

template <typename T, typename I, typename L>
[[nodiscard]] inline bool matrix_has_padded_row_width(raft::host_matrix_view<T, I, L> m,
                                                      uint32_t align_bytes = 16)
{
  using value_type = std::remove_const_t<T>;
  const uint32_t need =
    padded_row_width<value_type>(static_cast<uint32_t>(m.extent(1)), align_bytes);
  return matrix_actual_row_width(m) == need;
}

namespace detail {

template <typename SrcT>
[[nodiscard]] inline uint32_t mdspan_row_stride_elements(SrcT const& src)
{
  return src.stride(0) > 0 ? static_cast<uint32_t>(src.stride(0))
                           : static_cast<uint32_t>(src.extent(1));
}

}  // namespace detail

/**
 * The mapping of `rows x dim` elements. The dense layouts (contiguous and padded) derive the row
 * stride from the dimension alone; callers that need it read it from the mapping (or from the
 * array allocated with it) and compare it with the stride they expect, if any.
 */
template <typename MappingT>
[[nodiscard]] inline auto make_dense_mapping(int64_t rows, uint32_t dim) -> MappingT
{
  return MappingT{raft::make_extents<int64_t>(rows, dim)};
}

template <typename ViewT>
[[nodiscard]] inline auto make_dense_view(typename ViewT::data_view_type::data_handle_type ptr,
                                          int64_t rows,
                                          uint32_t dim,
                                          uint32_t stride) -> ViewT
{
  using data_view_type = typename ViewT::data_view_type;
  auto mapping         = make_dense_mapping<typename data_view_type::mapping_type>(rows, dim);
  RAFT_EXPECTS(rows == 0 || static_cast<uint32_t>(mapping.stride(0)) == stride,
               "make_dense_view: the row stride %u does not match the stride %u of the layout "
               "(a contiguous dataset has no padding; use a padded dataset for padded rows)",
               stride,
               static_cast<uint32_t>(mapping.stride(0)));
  return ViewT{data_view_type{ptr, mapping}};
}

/**
 * The rows of a dense dataset (or its view) as a plain row-major matrix, as wide as the row stride:
 * `extent(1)` spans the padding too, unlike `data_const_view()`, which spans the logical dimension.
 *
 * The result has `rows * stride` elements. That is within `required_span_size()` of the storage,
 * the only amount of memory a mapping owns: both dense layouts (contiguous and padded) include the
 * padding of every row, including the last one, in their span.
 */
template <typename DatasetT>
[[nodiscard]] inline auto dense_row_major_view(DatasetT const& dataset)
{
  auto v       = dataset.data_const_view();
  using view_t = raft::mdspan<typename decltype(v)::element_type,
                              raft::matrix_extent<int64_t>,
                              raft::row_major,
                              typename decltype(v)::accessor_type>;
  return view_t{v.data_handle(), raft::make_extents<int64_t>(v.extent(0), v.stride(0))};
}

/** A zero-row view of the given dimension, standing in for a dataset that is not there (yet). */
template <typename ViewT>
[[nodiscard]] inline auto make_empty_dataset_view(uint32_t dim = 0) -> ViewT
{
  if constexpr (is_dense_dataset_view_v<ViewT>) {
    return make_dense_view<ViewT>(nullptr, 0, dim, dim);
  } else if constexpr (is_empty_dataset_view_v<ViewT>) {
    return ViewT{typename ViewT::data_view_type{dim}};
  } else {
    return ViewT{};
  }
}

template <typename DatasetT>
[[nodiscard]] inline auto make_dense_array(raft::resources const& res, int64_t rows, uint32_t dim)
  -> typename DatasetT::data_type
{
  using data_type = typename DatasetT::data_type;
  auto mapping    = make_dense_mapping<typename data_type::mapping_type>(rows, dim);
  return data_type{res, mapping, typename data_type::container_policy_type{}};
}

namespace detail {

template <typename ValueT, typename SrcT>
[[nodiscard]] inline ValueT* expect_device_accessible_data_handle(SrcT const& src,
                                                                  char const* error_msg)
{
  // Device and managed allocations (including those owned by another device, which the multi-GPU
  // paths use once the owning device is current), pinned host memory (through its device mapping)
  // and pageable memory on ATS systems are device accessible. HMM pageable memory is not.
  auto* device_ptr = raft::get_pointer_memory_info(src.data_handle()).device_pointer;
  RAFT_EXPECTS(device_ptr != nullptr, "%s", error_msg);
  return const_cast<ValueT*>(device_ptr);
}

template <typename ValueT, typename ViewT, typename SrcT>
[[nodiscard]] inline ViewT make_device_dense_row_major_view_from_src(SrcT const& src,
                                                                     uint32_t logical_dim)
{
  auto* device_ptr = expect_device_accessible_data_handle<ValueT>(
    src, "make_device_*_dataset_view: source must be device-accessible.");
  return make_dense_view<ViewT>(
    device_ptr, src.extent(0), logical_dim, mdspan_row_stride_elements(src));
}

template <typename ValueT, typename ViewT, typename SrcT>
[[nodiscard]] inline ViewT make_host_dense_row_major_view_from_src(SrcT const& src,
                                                                   uint32_t logical_dim)
{
  RAFT_EXPECTS(raft::get_device_for_address(src.data_handle()) == -1,
               "make_host_*_dataset_view: source must be host-accessible.");
  return make_dense_view<ViewT>(
    src.data_handle(), src.extent(0), logical_dim, mdspan_row_stride_elements(src));
}

template <typename DatasetT, typename ValueT, typename SrcT>
auto make_device_dense_row_major_dataset_from_src(raft::resources const& res,
                                                  SrcT const& src,
                                                  uint32_t logical_dim,
                                                  uint32_t target_stride,
                                                  char const* view_factory_name)
  -> std::unique_ptr<DatasetT>
{
  uint32_t const src_stride = mdspan_row_stride_elements(src);
  RAFT_EXPECTS(logical_dim <= target_stride,
               "logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(logical_dim),
               static_cast<unsigned>(target_stride));
  RAFT_EXPECTS(static_cast<uint32_t>(src.extent(1)) <= target_stride,
               "Source row length must not exceed required stride.");
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  bool const device_src =
    (ptr_attrs.type == cudaMemoryTypeDevice) || (ptr_attrs.type == cudaMemoryTypeManaged);
  if (device_src && src_stride == target_stride) {
    RAFT_EXPECTS(false,
                 "source is device and stride is already correct. "
                 "Use %s() to get a view instead.",
                 view_factory_name);
  }
  auto out_array = make_dense_array<DatasetT>(res, src.extent(0), logical_dim);
  // The copy below writes `target_stride` columns per row: that has to be the stride of the layout,
  // or the rows would not fit in `required_span_size()`.
  RAFT_EXPECTS(
    src.extent(0) == 0 || static_cast<uint32_t>(out_array.mapping().stride(0)) == target_stride,
    "the row stride %u does not match the stride %u of the dataset layout",
    target_stride,
    static_cast<uint32_t>(out_array.mapping().stride(0)));
  RAFT_CUDA_TRY(cudaMemsetAsync(out_array.data_handle(),
                                0,
                                out_array.mapping().required_span_size() * sizeof(ValueT),
                                raft::resource::get_cuda_stream(res).get()));
  raft::copy_matrix(out_array.data_handle(),
                    target_stride,
                    src.data_handle(),
                    src_stride,
                    logical_dim,
                    src.extent(0),
                    raft::resource::get_cuda_stream(res));
  return std::make_unique<DatasetT>(std::move(out_array));
}

template <typename DatasetT, typename ValueT, typename SrcT>
auto make_host_dense_row_major_dataset_from_src(raft::resources const& res,
                                                SrcT const& src,
                                                uint32_t logical_dim,
                                                uint32_t target_stride,
                                                char const* view_factory_name)
  -> std::unique_ptr<DatasetT>
{
  uint32_t const src_stride = mdspan_row_stride_elements(src);
  constexpr bool device_src = SrcT::accessor_type::is_device_accessible;
  RAFT_EXPECTS(logical_dim <= target_stride,
               "logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(logical_dim),
               static_cast<unsigned>(target_stride));
  if (!device_src && src_stride == target_stride) {
    RAFT_EXPECTS(false,
                 "source stride is already correct. Use %s() to get a view instead.",
                 view_factory_name);
  }
  RAFT_EXPECTS(static_cast<uint32_t>(src.extent(1)) <= target_stride,
               "Source row length must not exceed required stride.");
  auto out_array = make_dense_array<DatasetT>(res, src.extent(0), logical_dim);
  // The copy below writes `target_stride` columns per row: that has to be the stride of the layout,
  // or the rows would not fit in `required_span_size()`.
  RAFT_EXPECTS(
    src.extent(0) == 0 || static_cast<uint32_t>(out_array.mapping().stride(0)) == target_stride,
    "the row stride %u does not match the stride %u of the dataset layout",
    target_stride,
    static_cast<uint32_t>(out_array.mapping().stride(0)));
  std::memset(
    out_array.data_handle(), 0, out_array.mapping().required_span_size() * sizeof(ValueT));
  raft::copy_matrix(out_array.data_handle(),
                    target_stride,
                    src.data_handle(),
                    src_stride,
                    logical_dim,
                    src.extent(0),
                    raft::resource::get_cuda_stream(res));
  if (device_src) { raft::resource::sync_stream(res); }
  return std::make_unique<DatasetT>(std::move(out_array));
}

}  // namespace detail

template <typename SrcT>
auto make_device_padded_dataset_view(const raft::resources& res, SrcT const& src)
  -> device_padded_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type         = typename SrcT::value_type;
  using index_type         = typename SrcT::index_type;
  uint32_t required_stride = padded_row_width<value_type>(static_cast<uint32_t>(src.extent(1)));
  RAFT_EXPECTS(
    detail::mdspan_row_stride_elements(src) == required_stride,
    "make_device_padded_dataset_view: stride is incorrect (required stride for alignment). "
    "Use make_device_padded_dataset() to get an owning padded copy.");
  return detail::make_device_dense_row_major_view_from_src<
    value_type,
    device_padded_dataset_view<value_type, index_type>>(src, static_cast<uint32_t>(src.extent(1)));
}

template <typename SrcT>
auto make_device_padded_dataset(const raft::resources& res, SrcT const& src)
  -> std::unique_ptr<device_padded_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type               = typename SrcT::value_type;
  using index_type               = typename SrcT::index_type;
  uint32_t const logical_dim     = static_cast<uint32_t>(src.extent(1));
  uint32_t const required_stride = padded_row_width<value_type>(logical_dim);
  return detail::make_device_dense_row_major_dataset_from_src<
    device_padded_dataset<value_type, index_type>,
    value_type>(res, src, logical_dim, required_stride, "make_device_padded_dataset_view");
}

template <typename SrcT>
auto make_host_padded_dataset_view(SrcT const& src)
  -> host_padded_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type         = typename SrcT::value_type;
  using index_type         = typename SrcT::index_type;
  uint32_t required_stride = padded_row_width<value_type>(static_cast<uint32_t>(src.extent(1)));
  RAFT_EXPECTS(
    detail::mdspan_row_stride_elements(src) == required_stride,
    "make_host_padded_dataset_view: stride is incorrect (required stride for alignment). "
    "Use make_host_padded_dataset() to get an owning padded copy.");
  return detail::make_host_dense_row_major_view_from_src<
    value_type,
    host_padded_dataset_view<value_type, index_type>>(src, static_cast<uint32_t>(src.extent(1)));
}

template <typename SrcT>
auto make_host_padded_dataset(const raft::resources& res, SrcT const& src)
  -> std::unique_ptr<host_padded_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type               = typename SrcT::value_type;
  using index_type               = typename SrcT::index_type;
  uint32_t const logical_dim     = static_cast<uint32_t>(src.extent(1));
  uint32_t const required_stride = padded_row_width<value_type>(logical_dim);
  return detail::make_host_dense_row_major_dataset_from_src<
    host_padded_dataset<value_type, index_type>,
    value_type>(res, src, logical_dim, required_stride, "make_host_padded_dataset_view");
}

template <typename SrcT>
auto make_device_contiguous_dataset_view(SrcT const& src)
  -> device_contiguous_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  RAFT_EXPECTS(
    detail::mdspan_row_stride_elements(src) == static_cast<uint32_t>(src.extent(1)),
    "make_device_contiguous_dataset_view: the rows must be densely packed (stride equals "
    "the dimension). Use make_device_padded_dataset_view() or "
    "make_device_padded_dataset() for rows with padding.");
  return detail::make_device_dense_row_major_view_from_src<
    value_type,
    device_contiguous_dataset_view<value_type, index_type>>(src,
                                                            static_cast<uint32_t>(src.extent(1)));
}

/**
 * @brief Create an owning device contiguous (densely packed) dataset as a copy of the source rows.
 *
 * Prefer `make_device_contiguous_dataset_view()` when the source is already densely packed device
 * memory; this function throws in that case.
 */
template <typename SrcT>
auto make_device_contiguous_dataset(const raft::resources& res, SrcT const& src) -> std::unique_ptr<
  device_contiguous_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type           = typename SrcT::value_type;
  using index_type           = typename SrcT::index_type;
  uint32_t const logical_dim = static_cast<uint32_t>(src.extent(1));
  return detail::make_device_dense_row_major_dataset_from_src<
    device_contiguous_dataset<value_type, index_type>,
    value_type>(res, src, logical_dim, logical_dim, "make_device_contiguous_dataset_view");
}

template <typename SrcT>
auto make_host_contiguous_dataset_view(SrcT const& src)
  -> host_contiguous_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  RAFT_EXPECTS(detail::mdspan_row_stride_elements(src) == static_cast<uint32_t>(src.extent(1)),
               "make_host_contiguous_dataset_view: the rows must be densely packed (stride equals "
               "the dimension). Use make_host_padded_dataset_view() or "
               "make_host_padded_dataset() for rows with padding.");
  return detail::make_host_dense_row_major_view_from_src<
    value_type,
    host_contiguous_dataset_view<value_type, index_type>>(src,
                                                          static_cast<uint32_t>(src.extent(1)));
}

}  // namespace CUVS_EXPORT cuvs
