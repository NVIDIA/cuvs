/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/cagra.hpp>

#include "cagra_merge_scaffold.cuh"
#include "graph_core.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/refine.hpp>

#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

namespace cuvs::neighbors::cagra::detail {

/** Per-index write offsets into a merged dataset buffer, length `indices.size() + 1`, with the
 *  last entry equal to the final row count. Entry `i` is the row at which caller-concatenated data
 *  for `indices[i]` must start. For `row_filter = none_sample_filter`, offsets are just the
 *  cumulative sizes of `indices` -- callers can compute those directly and do not need this
 *  function. For a bitset `row_filter`, the per-index surviving row counts are not derivable from
 *  public APIs alone, so this function walks the filter's sorted surviving-row list (via
 *  `bitset_view::to_csr`) and locates each index's boundary in it. */
template <class T, class IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
std::vector<int64_t> merged_dataset_offsets(
  raft::resources const& handle,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*> const& indices,
  cuvs::neighbors::filtering::base_filter const& row_filter)
{
  std::vector<int64_t> unfiltered_offsets;
  unfiltered_offsets.reserve(indices.size() + 1);
  unfiltered_offsets.push_back(0);
  for (auto* index : indices) {
    RAFT_EXPECTS(index != nullptr,
                 "Null pointer detected in 'indices'. Ensure all elements are valid before usage.");
    unfiltered_offsets.push_back(unfiltered_offsets.back() + static_cast<int64_t>(index->size()));
  }

  if (row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::None) {
    return unfiltered_offsets;
  }
  RAFT_EXPECTS(row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::Bitset,
               "Only none and bitset filters are supported by cagra::merged_dataset_offsets");

  auto const& actual_filter =
    dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(row_filter);
  int64_t const final_rows = actual_filter.view().count(handle);

  auto surviving_rows = raft::make_device_csr_matrix<uint32_t, int64_t, int64_t, int64_t>(
    handle, 1, static_cast<std::size_t>(unfiltered_offsets.back()));
  surviving_rows.initialize_sparsity(final_rows);
  actual_filter.view().to_csr(handle, surviving_rows);
  auto const csr_indices = surviving_rows.structure_view().get_indices();

  std::vector<int64_t> surviving_rows_host(csr_indices.size());
  raft::copy(surviving_rows_host.data(),
             csr_indices.data(),
             csr_indices.size(),
             raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);

  std::vector<int64_t> filtered_offsets;
  filtered_offsets.reserve(unfiltered_offsets.size());
  for (int64_t boundary : unfiltered_offsets) {
    filtered_offsets.push_back(static_cast<int64_t>(
      std::lower_bound(surviving_rows_host.begin(), surviving_rows_host.end(), boundary) -
      surviving_rows_host.begin()));
  }
  return filtered_offsets;
}

/** Validate that `offsets` is a well-formed length-`indices.size() + 1` boundary vector ending at
 *  `final_rows`: starts at 0, non-decreasing, and the caller-supplied row counts stay in range. */
inline void validate_merge_offsets(std::vector<int64_t> const& offsets,
                                   std::size_t num_indices,
                                   int64_t final_rows)
{
  RAFT_EXPECTS(offsets.size() == num_indices + 1,
               "offsets must have indices.size() + 1 (%zu) entries, got %zu",
               num_indices + 1,
               offsets.size());
  RAFT_EXPECTS(offsets.front() == 0, "offsets[0] must be 0");
  for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
    RAFT_EXPECTS(offsets[i] <= offsets[i + 1], "offsets must be non-decreasing");
  }
  RAFT_EXPECTS(offsets.back() == final_rows,
               "offsets.back() (%ld) must equal merged_dataset's row count (%ld)",
               long(offsets.back()),
               long(final_rows));
}

/** Validate `indices` for concatenation: every index must carry a non-empty, uncompressed,
 *  dense row-major device dataset, and they must all share the same dimension and row stride.
 *  Returns that shared (dim, stride). */
template <class T, class IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
auto validate_indices_for_concat(
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*> const& indices,
  char const* caller) -> std::pair<uint32_t, int64_t>
{
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>;

  uint32_t dim   = 0;
  int64_t stride = -1;
  for (cagra_index_t* index : indices) {
    RAFT_EXPECTS(index != nullptr,
                 "Null pointer detected in 'indices'. Ensure all elements are valid before usage.");
    auto const& dataset = index->dataset();
    if constexpr (cuvs::neighbors::is_dense_row_major_dataset_view_v<
                    std::decay_t<decltype(dataset)>>) {
      RAFT_EXPECTS(dataset.n_rows() != 0,
                   "%s only supports an index to which the dataset is attached. Please check if "
                   "the index has an empty dataset; attach one with update_dataset before "
                   "concatenating.",
                   caller);
      if (dim == 0) {
        dim    = index->dim();
        stride = static_cast<int64_t>(dataset.stride());
      } else {
        RAFT_EXPECTS(dim == index->dim(), "Dimension of datasets in indices must be equal.");
        RAFT_EXPECTS(stride == static_cast<int64_t>(dataset.stride()),
                     "Row stride of datasets in indices must be equal.");
      }
    } else {
      RAFT_FAIL("%s only supports an uncompressed dense device dataset index", caller);
    }
  }
  return {dim, stride};
}

/** Copy every input index's dataset into `dst` (row stride `dst_stride`, in `indices` order),
 *  starting at row 0. `dst` must already be zeroed/allocated for `total_rows * dst_stride`
 *  elements. */
template <class T, class IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
void copy_datasets_into(
  raft::resources const& handle,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*> const& indices,
  T* dst,
  int64_t dst_stride,
  uint32_t dim)
{
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>;

  int64_t row_offset  = 0;
  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();
  for (cagra_index_t* index : indices) {
    auto const& v      = index->dataset();
    const T* src_ptr   = v.view().data_handle();
    std::size_t n_rows = static_cast<std::size_t>(v.n_rows());
    raft::copy_matrix(dst + static_cast<std::size_t>(row_offset) * dst_stride,
                      static_cast<std::size_t>(dst_stride),
                      src_ptr,
                      static_cast<std::size_t>(dst_stride),
                      static_cast<std::size_t>(dim),
                      n_rows,
                      stream);
    row_offset += static_cast<int64_t>(index->size());
  }
}

/** Concatenate every input index's dataset (unfiltered, in `indices` order) into a freshly
 *  allocated, CAGRA-padded, owning device dataset. This is a convenience helper for building
 *  `merge()`'s `merged_dataset` argument in the unfiltered case; the matching `offsets` are simply
 *  each index's cumulative `.size()` (or `merged_dataset_offsets()` with a `none_sample_filter`).
 */
template <class T, class IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
auto concatenate_datasets(
  raft::resources const& handle,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*> const& indices)
  -> std::unique_ptr<cuvs::neighbors::device_padded_dataset<T, int64_t>>
{
  auto [dim, stride] =
    validate_indices_for_concat<T, IdxT, DatasetViewT>(indices, "cagra::concatenate_datasets");

  int64_t total_rows = 0;
  for (auto* index : indices) {
    total_rows += static_cast<int64_t>(index->size());
  }

  auto out_array      = raft::make_device_matrix<T, int64_t>(handle, total_rows, stride);
  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();
  RAFT_CUDA_TRY(cudaMemsetAsync(
    out_array.data_handle(), 0, static_cast<std::size_t>(out_array.size()) * sizeof(T), stream));

  copy_datasets_into<T, IdxT, DatasetViewT>(handle, indices, out_array.data_handle(), stride, dim);

  return std::make_unique<cuvs::neighbors::device_padded_dataset<T, int64_t>>(std::move(out_array),
                                                                              dim);
}

/** Concatenate every input index's dataset (in `indices` order), retaining only the rows selected
 *  by `row_filter`, into a freshly allocated, CAGRA-padded, owning device dataset. This is a
 *  convenience helper for building `merge()`'s `merged_dataset` argument in the bitset-filtered
 *  case; call `merged_dataset_offsets()` with the same `row_filter` to get the matching `offsets`.
 *
 *  Peak device memory is only the output buffer, sized to the *surviving* row count: each index's
 *  surviving rows are gathered directly out of that index's own (already device-resident) dataset
 *  storage, so no unfiltered-sized staging buffer is ever allocated. The per-index row-id lists fed
 *  to the gather are small (bounded by the surviving row count) and round-trip through host memory,
 *  which does not count against device peak.
 */
template <class T, class IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
auto concatenate_and_filter_datasets(
  raft::resources const& handle,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*> const& indices,
  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> const& row_filter)
  -> std::unique_ptr<cuvs::neighbors::device_padded_dataset<T, int64_t>>
{
  auto [dim, stride] = validate_indices_for_concat<T, IdxT, DatasetViewT>(
    indices, "cagra::concatenate_and_filter_datasets");

  std::vector<int64_t> unfiltered_offsets;
  unfiltered_offsets.reserve(indices.size() + 1);
  unfiltered_offsets.push_back(0);
  for (auto* index : indices) {
    unfiltered_offsets.push_back(unfiltered_offsets.back() + static_cast<int64_t>(index->size()));
  }
  int64_t const unfiltered_rows = unfiltered_offsets.back();
  int64_t const final_rows      = row_filter.view().count(handle);

  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();

  // Sorted list of surviving *global* row ids -- sized to `final_rows`, not `unfiltered_rows`.
  auto surviving_rows_csr = raft::make_device_csr_matrix<uint32_t, int64_t, int64_t, int64_t>(
    handle, 1, static_cast<std::size_t>(unfiltered_rows));
  surviving_rows_csr.initialize_sparsity(final_rows);
  row_filter.view().to_csr(handle, surviving_rows_csr);
  auto csr_indices = surviving_rows_csr.structure_view().get_indices();

  std::vector<int64_t> surviving_rows_host(csr_indices.size());
  raft::copy(surviving_rows_host.data(), csr_indices.data(), csr_indices.size(), stream);
  raft::resource::sync_stream(handle);

  // Per-index segment boundaries within `surviving_rows_host` (same `lower_bound` technique as
  // `merged_dataset_offsets()`): index i's surviving rows are surviving_rows_host[offsets[i] ..
  // offsets[i+1]), still expressed as global row ids.
  std::vector<int64_t> offsets;
  offsets.reserve(unfiltered_offsets.size());
  for (int64_t boundary : unfiltered_offsets) {
    offsets.push_back(static_cast<int64_t>(
      std::lower_bound(surviving_rows_host.begin(), surviving_rows_host.end(), boundary) -
      surviving_rows_host.begin()));
  }

  auto out_array = raft::make_device_matrix<T, int64_t>(handle, final_rows, stride);
  RAFT_CUDA_TRY(cudaMemsetAsync(
    out_array.data_handle(), 0, static_cast<std::size_t>(out_array.size()) * sizeof(T), stream));

  for (std::size_t i = 0; i < indices.size(); ++i) {
    int64_t const n_i = offsets[i + 1] - offsets[i];
    if (n_i == 0) { continue; }

    // Shift this index's surviving global row ids down to local (0-based within its own storage)
    // row ids, since the gather source below is that index's own dataset, not a global buffer.
    std::vector<int64_t> local_ids_host(surviving_rows_host.begin() + offsets[i],
                                        surviving_rows_host.begin() + offsets[i + 1]);
    int64_t const base = unfiltered_offsets[i];
    for (auto& id : local_ids_host) {
      id -= base;
    }

    auto local_ids = raft::make_device_vector<int64_t, int64_t>(handle, n_i);
    raft::copy(local_ids.data_handle(), local_ids_host.data(), local_ids_host.size(), stream);
    raft::resource::sync_stream(handle);

    auto const& v = indices[i]->dataset();
    auto src_view = raft::make_device_matrix_view<const T, int64_t>(
      v.view().data_handle(), static_cast<int64_t>(v.n_rows()), stride);
    auto dst_view = raft::make_device_matrix_view<T, int64_t>(
      out_array.data_handle() + static_cast<std::size_t>(offsets[i]) * stride, n_i, stride);

    raft::matrix::copy_rows(handle, src_view, dst_view, raft::make_const_mdspan(local_ids.view()));
  }

  return std::make_unique<cuvs::neighbors::device_padded_dataset<T, int64_t>>(std::move(out_array),
                                                                              dim);
}

/** Build a fresh CAGRA graph over a caller-populated, already-concatenated (and, if applicable,
 *  already-filtered) merged dataset. The caller owns `merged_dataset`; this only merges the graph
 *  and rebinds a view of it, mirroring the `extend()` contract. */
template <class T, class IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT> merge_rebuild(
  raft::resources const& handle,
  const cagra::index_params& params,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*>& indices,
  DatasetViewT merged_dataset,
  std::vector<int64_t> const& offsets,
  const cuvs::neighbors::filtering::base_filter& row_filter)
{
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>;

  uint32_t dim   = 0;
  int64_t stride = -1;

  RAFT_EXPECTS(row_filter.get_filter_type() != cuvs::neighbors::filtering::FilterType::Bitmap,
               "Bitmap filter isn't supported inside cagra::merge");
  RAFT_EXPECTS(row_filter.get_filter_type() != cuvs::neighbors::filtering::FilterType::Bloom,
               "Bloom filter isn't supported inside cagra::merge");
  RAFT_EXPECTS(row_filter.get_filter_type() != cuvs::neighbors::filtering::FilterType::Roaring,
               "Roaring filter isn't supported inside cagra::merge");

  for (cagra_index_t* index : indices) {
    RAFT_EXPECTS(index != nullptr,
                 "Null pointer detected in 'indices'. Ensure all elements are valid before usage.");
    auto const& dataset = index->dataset();
    if constexpr (cuvs::neighbors::is_dense_row_major_dataset_view_v<
                    std::decay_t<decltype(dataset)>>) {
      RAFT_EXPECTS(
        dataset.n_rows() != 0,
        "cagra::merge only supports an index to which the dataset is attached. Please check if "
        "the index has an empty dataset; attach one with update_dataset "
        "before merge.");
      if (dim == 0) {
        dim    = index->dim();
        stride = static_cast<int64_t>(dataset.stride());
      } else {
        RAFT_EXPECTS(dim == index->dim(), "Dimension of datasets in indices must be equal.");
        RAFT_EXPECTS(stride == static_cast<int64_t>(dataset.stride()),
                     "Row stride of datasets in indices must be equal.");
      }
    } else {
      RAFT_FAIL("cagra::merge only supports an uncompressed dense device dataset index");
    }
  }

  validate_merge_offsets(offsets, indices.size(), static_cast<int64_t>(merged_dataset.n_rows()));
  RAFT_EXPECTS(merged_dataset.dim() == dim,
               "merged_dataset dimension (%u) must equal the input dimension (%u)",
               unsigned(merged_dataset.dim()),
               unsigned(dim));
  RAFT_EXPECTS(merged_dataset.stride() == stride,
               "merged_dataset stride (%u) must equal the input stride (%ld)",
               unsigned(merged_dataset.stride()),
               long(stride));

  auto build_params                    = params;
  build_params.attach_dataset_on_build = false;
  auto index = ::cuvs::neighbors::cagra::build(handle, build_params, merged_dataset);
  index      = ::cuvs::neighbors::cagra::update_dataset(handle, std::move(index), merged_dataset);
  return index;
}

struct fastener_preflight_result {
  bool eligible  = false;
  int64_t rows   = 0;
  int64_t dim    = 0;
  int64_t stride = 0;
  std::vector<int64_t> offsets;
  std::string reason;
};

/** Validate every input and option without mutating anything. */
template <typename T, typename IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
auto preflight_fastener(
  raft::resources const& handle,
  cagra::index_params const& params,
  cagra::merge_params const& merge_params,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*> const& indices,
  std::vector<int64_t> const& offsets,
  cuvs::neighbors::filtering::base_filter const& row_filter) -> fastener_preflight_result
{
  fastener_preflight_result result;
  auto reject = [&](std::string reason) {
    result.reason = std::move(reason);
    return result;
  };

  if constexpr (!(std::is_same_v<T, float> || std::is_same_v<T, half> ||
                  std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) ||
                !std::is_same_v<IdxT, uint32_t>) {
    return reject("the scalar or graph index type is unsupported");
  }
  // Fastener reads the dataset densely per row with an explicit stride, so it needs a dense
  // device view; VPQ and host views are rejected here rather than deep inside a kernel.
  if constexpr (!cuvs::neighbors::is_dense_row_major_device_dataset_view_v<DatasetViewT>) {
    return reject("only dense row-major device datasets are supported");
  }
  if (indices.size() < 2) { return reject("at least two input indices are required"); }
  if (row_filter.get_filter_type() != cuvs::neighbors::filtering::FilterType::None) {
    return reject("row filters are not supported");
  }
  if (params.metric != cuvs::distance::DistanceType::L2Expanded) {
    return reject("only L2Expanded is supported");
  }
  if (merge_params.levels == 0) { return reject("levels must be positive"); }
  if (merge_params.root_fanout < 1 || merge_params.root_fanout > merge_scaffold::MAX_FANOUT ||
      merge_params.lower_fanout < 1 || merge_params.lower_fanout > merge_scaffold::MAX_FANOUT) {
    return reject("root_fanout and lower_fanout must be between 1 and " +
                  std::to_string(merge_scaffold::MAX_FANOUT));
  }
  if (!(merge_params.leader_fraction > 0.0 && merge_params.leader_fraction <= 1.0)) {
    return reject("leader_fraction must be in (0, 1]");
  }
  if (merge_params.max_leaders == 0 || merge_params.max_leaders > merge_scaffold::MAX_LEADERS) {
    return reject("max_leaders must be between 1 and " +
                  std::to_string(merge_scaffold::MAX_LEADERS));
  }
  if (merge_params.max_leaders < std::max(merge_params.root_fanout, merge_params.lower_fanout)) {
    return reject("max_leaders must cover both configured fanouts");
  }
  if (merge_params.leaf_size == 0 || merge_params.leaf_size > merge_scaffold::MAX_LEAF_SIZE) {
    return reject("leaf_size must be between 1 and " +
                  std::to_string(merge_scaffold::MAX_LEAF_SIZE));
  }
  if (merge_params.leaf_degree == 0 ||
      merge_params.leaf_degree > static_cast<uint32_t>(merge_scaffold::MAX_LEAF_DEGREE)) {
    return reject("leaf_degree must be between 1 and " +
                  std::to_string(merge_scaffold::MAX_LEAF_DEGREE));
  }

  uint64_t const max_spill = std::numeric_limits<uint8_t>::max() / merge_params.leaf_degree;
  uint64_t spill           = merge_params.root_fanout;
  auto const candidate_width_limit =
    "root_fanout * lower_fanout^(levels - 1) * leaf_degree must not exceed " +
    std::to_string(std::numeric_limits<uint8_t>::max());
  if (spill > max_spill) { return reject(candidate_width_limit); }
  if (merge_params.lower_fanout > 1) {
    for (uint32_t level = 1; level < merge_params.levels; ++level) {
      if (spill > max_spill / merge_params.lower_fanout) { return reject(candidate_width_limit); }
      spill *= merge_params.lower_fanout;
    }
  }
  uint64_t const scaffold_degree = spill * merge_params.leaf_degree;

  uint64_t rows             = 0;
  uint64_t max_input_degree = 0;
  result.offsets.reserve(indices.size() + 1);
  result.offsets.push_back(0);

  for (auto const* index : indices) {
    if (index == nullptr) { return reject("all input index pointers must be non-null"); }
    auto const& dataset = index->dataset();
    if (dataset.n_rows() != static_cast<int64_t>(index->size())) {
      return reject("every input must have an attached, uncompressed dataset");
    }
    if (index->metric() != params.metric) {
      return reject("every input metric must match index_params.metric");
    }
    if (result.offsets.size() == 1) {
      result.dim    = static_cast<int64_t>(index->dim());
      result.stride = static_cast<int64_t>(dataset.stride());
    } else {
      if (result.dim != static_cast<int64_t>(index->dim())) {
        return reject("all input dimensions must match");
      }
      // The merged dataset has a single row pitch, so mixed input strides cannot be consolidated
      // without re-padding each input separately.
      if (result.stride != static_cast<int64_t>(dataset.stride())) {
        return reject("all input row strides must match");
      }
    }
    auto graph = index->graph();
    if (graph.extent(0) <= 0 || graph.extent(1) <= 0 ||
        graph.extent(0) != static_cast<int64_t>(index->size())) {
      return reject("every input must have a nonempty device graph");
    }

    auto const input_rows = static_cast<uint64_t>(index->size());
    if (rows > std::numeric_limits<uint32_t>::max() - input_rows) {
      return reject("the combined row count must fit in uint32_t");
    }
    rows += input_rows;
    max_input_degree = std::max<uint64_t>(max_input_degree, static_cast<uint64_t>(graph.extent(1)));
    result.offsets.push_back(static_cast<int64_t>(rows));
  }

  // Fastener never applies a row filter (rejected above), so the caller-supplied offsets must be
  // exactly the unfiltered per-index cumulative sizes computed above -- merge_dataset_offsets()
  // returns this same vector for an unfiltered merge, so a caller who used it will always match.
  if (offsets != result.offsets) {
    return reject(
      "offsets must equal the cumulative unfiltered row counts of each input index for Fastener");
  }

  if (result.dim <= 0 || result.dim > std::numeric_limits<int>::max()) {
    return reject("dataset dimension must be positive and fit cuBLAS int dimensions");
  }
  if (!merge_scaffold::leaf_gemm_supported(
        result.dim, merge_params.leaf_size, raft::resource::get_workspace_free_bytes(handle))) {
    return reject("dataset dimension exceeds the leaf GEMM workspace limit");
  }
  if (!merge_scaffold::assignment_gemm_supported(
        result.dim,
        static_cast<int64_t>(rows),
        merge_scaffold::split_params{
          .fanout          = std::max(merge_params.root_fanout, merge_params.lower_fanout),
          .leader_fraction = merge_params.leader_fraction,
          .max_leaders     = merge_params.max_leaders},
        raft::resource::get_workspace_free_bytes(handle))) {
    return reject("dataset dimension exceeds the assignment GEMM workspace limit");
  }
  if (rows > std::numeric_limits<uint32_t>::max() / spill) {
    return reject("combined rows times the configured spill width must fit in uint32_t");
  }
  if (params.graph_degree == 0 || static_cast<uint64_t>(params.graph_degree) >= rows) {
    return reject("graph_degree must be positive and smaller than the combined row count");
  }
  if (static_cast<uint64_t>(params.graph_degree) > max_input_degree + scaffold_degree) {
    return reject("graph_degree exceeds the input graph plus scaffold capacity");
  }
  // The appended candidate graph is sorted by launch_sort_knn_graph, whose kernel capacity is
  // kMaxSortDegree. Without this check an input degree at the limit plus any scaffold passes
  // preflight and then fails inside the sorter, after the merge has already started mutating -- and
  // in AUTO that also loses the rebuild fallback.
  if (max_input_degree + scaffold_degree > cagra::detail::graph::kMaxSortDegree) {
    return reject("the widest input graph degree plus the scaffold degree must not exceed " +
                  std::to_string(cagra::detail::graph::kMaxSortDegree));
  }

  result.rows     = static_cast<int64_t>(rows);
  result.eligible = true;
  return result;
}

/** Build a merged CAGRA graph via Fastener over a caller-populated, already-concatenated merged
 *  dataset (Fastener never applies a row filter, so `merged_dataset` always holds the full,
 *  unfiltered concatenation of every input in `indices` order). The caller owns `merged_dataset`;
 *  this only merges the graph and rebinds a view of it. */
template <typename T, typename IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
auto merge_fastener(raft::resources const& handle,
                    cagra::index_params const& params,
                    cagra::merge_params const& merge_params,
                    std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*>& indices,
                    DatasetViewT merged_dataset,
                    fastener_preflight_result const& preflight)
  -> cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>
{
  auto const stride = static_cast<int64_t>(merged_dataset.stride());
  RAFT_EXPECTS(merged_dataset.n_rows() == preflight.rows,
               "merged_dataset rows (%ld) must equal the merged row count (%ld)",
               long(merged_dataset.n_rows()),
               long(preflight.rows));
  RAFT_EXPECTS(static_cast<int64_t>(merged_dataset.dim()) == preflight.dim,
               "merged_dataset dimension (%u) must equal the input dimension (%ld)",
               unsigned(merged_dataset.dim()),
               long(preflight.dim));

  auto const output_const_view = merged_dataset.view();
  // The scaffold and the sorter read the consolidated rows with this pitch; dim stays logical.
  auto dataset_view = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
    output_const_view.data_handle(), preflight.rows, stride);

  merge_scaffold::build_params scaffold_params;
  scaffold_params.levels          = merge_params.levels;
  scaffold_params.root_fanout     = merge_params.root_fanout;
  scaffold_params.lower_fanout    = merge_params.lower_fanout;
  scaffold_params.leader_fraction = merge_params.leader_fraction;
  scaffold_params.max_leaders     = merge_params.max_leaders;
  scaffold_params.leaf_size       = merge_params.leaf_size;
  scaffold_params.leaf_degree     = merge_params.leaf_degree;

  auto merged_graph = [&] {
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> scope("cagra::merge/scaffold");
    int64_t base_degree = 0;
    for (auto const* index : indices) {
      base_degree = std::max<int64_t>(base_degree, index->graph_degree());
    }
    auto graph = merge_scaffold::build<T>(
      handle, dataset_view, preflight.dim, preflight.offsets, scaffold_params, base_degree);
    merge_scaffold::append_to_input_graphs<T, IdxT, DatasetViewT>(
      handle, indices, preflight.offsets, graph.view(), base_degree);
    return graph;
  }();

  {
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> scope("cagra::merge/append/sort");
    // Padding is zeroed above, so passing the padded width as the dimension is exact for the
    // L2Expanded metric that preflight restricts Fastener to.
    cagra::detail::graph::launch_sort_knn_graph(handle,
                                                params.metric,
                                                dataset_view.data_handle(),
                                                static_cast<uint32_t>(dataset_view.extent(0)),
                                                static_cast<uint32_t>(dataset_view.extent(1)),
                                                merged_graph.data_handle(),
                                                static_cast<uint32_t>(merged_graph.extent(1)));
    merged_graph = merge_scaffold::cap_sorted_graph(
      handle, raft::make_const_mdspan(merged_graph.view()), params.graph_degree);
  }

  auto optimized_graph =
    raft::make_device_matrix<uint32_t, int64_t>(handle, preflight.rows, params.graph_degree);
  {
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> scope("cagra::merge/optimize");
    cagra::detail::graph::optimize_device_graph(
      handle, merged_graph.view(), optimized_graph.view(), params.guarantee_connectivity);
  }

  // The caller owns merged_dataset; the returned index holds only a view of it.
  cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT> merged_index(handle, params.metric);
  // Must move: the device_matrix_view overload only stores a view, which would dangle once
  // optimized_graph goes out of scope.
  merged_index.update_graph(handle, std::move(optimized_graph));
  merged_index =
    ::cuvs::neighbors::cagra::update_dataset(handle, std::move(merged_index), merged_dataset);
  return merged_index;
}

template <class T, class IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
auto merge(raft::resources const& handle,
           cagra::index_params const& params,
           std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*>& indices,
           DatasetViewT merged_dataset,
           std::vector<int64_t> const& offsets,
           cagra::merge_params const& merge_params,
           cuvs::neighbors::filtering::base_filter const& row_filter)
  -> cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> merge_scope(
    "cagra::merge(algo=%d,parts=%zu)", static_cast<int>(merge_params.algo), indices.size());

  RAFT_EXPECTS(merge_params.algo == cagra::merge_algo::AUTO ||
                 merge_params.algo == cagra::merge_algo::FASTENER ||
                 merge_params.algo == cagra::merge_algo::REBUILD,
               "Unknown cagra::merge algorithm");
  if (merge_params.algo == cagra::merge_algo::REBUILD) {
    return merge_rebuild<T, IdxT, DatasetViewT>(
      handle, params, indices, merged_dataset, offsets, row_filter);
  }

  auto preflight = preflight_fastener<T, IdxT, DatasetViewT>(
    handle, params, merge_params, indices, offsets, row_filter);
  if (!preflight.eligible) {
    if (merge_params.algo == cagra::merge_algo::AUTO) {
      return merge_rebuild<T, IdxT, DatasetViewT>(
        handle, params, indices, merged_dataset, offsets, row_filter);
    }
    RAFT_FAIL("FASTENER cagra::merge is unsupported: %s", preflight.reason.c_str());
  }

  if (merge_params.algo == cagra::merge_algo::AUTO) {
    // Preflight validates shapes and configured limits, but it cannot know whether the temporary
    // GEMM workspaces and the candidate graph will all fit at run time. Fastener never mutates its
    // inputs -- it only reads them and allocates its own temporaries -- so the rebuild is free to
    // run on the same indices after a failed attempt has unwound. Explicit FASTENER still surfaces
    // the failure rather than silently doing something much slower.
    try {
      return merge_fastener<T, IdxT, DatasetViewT>(
        handle, params, merge_params, indices, merged_dataset, preflight);
    } catch (std::bad_alloc const& failure) {
      RAFT_LOG_WARN("Fastener cagra::merge could not allocate (%s); falling back to rebuild",
                    failure.what());
      return merge_rebuild<T, IdxT, DatasetViewT>(
        handle, params, indices, merged_dataset, offsets, row_filter);
    }
  }

  return merge_fastener<T, IdxT, DatasetViewT>(
    handle, params, merge_params, indices, merged_dataset, preflight);
}

/** AUTO-algorithm convenience overload matching the base `merge` signature. */
template <class T, class IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
auto merge(raft::resources const& handle,
           cagra::index_params const& params,
           std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*>& indices,
           DatasetViewT merged_dataset,
           std::vector<int64_t> const& offsets,
           cuvs::neighbors::filtering::base_filter const& row_filter)
  -> cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>
{
  // Fully qualified: an unqualified call also finds cuvs::neighbors::cagra::merge via ADL on the
  // index arguments, which is ambiguous with this overload.
  return cuvs::neighbors::cagra::detail::merge<T, IdxT, DatasetViewT>(
    handle, params, indices, merged_dataset, offsets, cagra::merge_params{}, row_filter);
}

}  // namespace cuvs::neighbors::cagra::detail
