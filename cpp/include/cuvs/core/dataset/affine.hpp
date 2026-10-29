/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/type.hpp>
#include <cuvs/core/export.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace CUVS_EXPORT cuvs {

/**
 * Rows with a per-row `scale` and a per-row `shift`: the building block of the data of affine
 * datasets. The rows are the values themselves for a plain affine dataset, and an encoding of the
 * values for a quantized one (BBQ has one such group, RaBitQ has two); the `scale` and the `shift`
 * are the per-row terms the rows are mapped through.
 *
 * It holds arrays for an owning dataset and spans for a view of it, and it is an aggregate, so it
 * is made as `affine_data{rows, scale, shift}`.
 *
 * @tparam RowsT the type of `rows`: a matrix of `n_rows` rows
 * @tparam VectorT the type of `scale` and `shift`: a vector of `n_rows` elements, in the same
 *   memory space as the rows
 */
template <typename RowsT, typename VectorT>
struct affine_data {
  RowsT rows;
  VectorT scale;
  VectorT shift;

  /** The spans over the arrays. */
  [[nodiscard]] auto view() noexcept
  {
    return affine_data<typename RowsT::view_type, typename VectorT::view_type>{
      rows.view(), scale.view(), shift.view()};
  }

  /** The read-only spans over the arrays or the spans. */
  [[nodiscard]] auto const_view() const noexcept
  {
    return affine_data<decltype(as_const_view(rows)), decltype(as_const_view(scale))>{
      as_const_view(rows), as_const_view(scale), as_const_view(shift)};
  }
};

/**
 * Affine dataset specification: the rows of values (not quantized), each with its own `scale` and
 * `shift`. The data is one `affine_data`; there is no manifest.
 *
 * @tparam MathT the element type of `scale` and `shift`
 * @tparam RowsPolicy the container policy of the rows (elements of the value type); `scale` and
 *   `shift` live in the same memory space
 */
template <typename MathT, typename RowsPolicy>
struct affine_spec {
  using math_type = MathT;

  template <typename T, typename IdxT>
  struct apply {
    using value_type = std::remove_const_t<T>;
    using index_type = std::remove_const_t<IdxT>;

    using rows_type = raft::
      mdarray<value_type, raft::matrix_extent<int64_t>, raft::layout_c_contiguous, RowsPolicy>;
    using vector_type = raft::mdarray<
      math_type,
      raft::vector_extent<int64_t>,
      raft::layout_c_contiguous,
      std::conditional_t<RowsPolicy::is_device_accessible,
                         raft::device_accessor<raft::device_container_policy<math_type>>,
                         raft::host_accessor<raft::host_container_policy<math_type>>>>;

    static constexpr bool is_device_accessible = RowsPolicy::is_device_accessible;

    using data_type      = affine_data<rows_type, vector_type>;
    using data_view_type = decltype(std::declval<data_type&>().view());

    [[nodiscard]] static auto get_data_view(data_type& d) noexcept -> data_view_type
    {
      return d.view();
    }

    // NOTE: the shape accessors take the data in either form - the storage an owning dataset holds,
    // or the view a `dataset_view` holds. There is no manifest, so it is ignored.
    [[nodiscard]] static auto get_n_rows(const auto& d) noexcept -> index_type
    {
      return static_cast<index_type>(d.rows.extent(0));
    }
    [[nodiscard]] static auto get_dim(const auto& d, const auto&) noexcept -> uint32_t
    {
      return static_cast<uint32_t>(d.rows.extent(1));
    }
  };
};

/** The dense values of `T` (not quantized) with the per-row scale and shift. */
template <typename T, typename IdxT, typename MathT, typename RowsPolicy>
using affine_dataset = dataset<T, IdxT, affine_spec<MathT, RowsPolicy>>;

template <typename T, typename IdxT, typename MathT = float>
using device_affine_dataset =
  affine_dataset<T, IdxT, MathT, raft::device_accessor<raft::device_container_policy<T>>>;

template <typename T, typename IdxT, typename MathT = float>
using host_affine_dataset =
  affine_dataset<T, IdxT, MathT, raft::host_accessor<raft::host_container_policy<T>>>;

}  // namespace CUVS_EXPORT cuvs
