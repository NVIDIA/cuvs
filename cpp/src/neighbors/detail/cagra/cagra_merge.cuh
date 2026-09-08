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

#include <limits>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

namespace cuvs::neighbors::cagra::detail {

template <class T, class IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
int64_t merged_dataset_size(
  raft::resources const& handle,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*> const& indices,
  cuvs::neighbors::filtering::base_filter const& row_filter)
{
  int64_t merged_rows = 0;
  for (auto* index : indices) {
    RAFT_EXPECTS(index != nullptr,
                 "Null pointer detected in 'indices'. Ensure all elements are valid before usage.");
    merged_rows += static_cast<int64_t>(index->size());
  }
  if (row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::Bitset) {
    auto const& actual_filter =
      dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(row_filter);
    return actual_filter.view().count(handle);
  }
  RAFT_EXPECTS(row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::None,
               "Only none and bitset filters are supported inside cagra::merge");
  return merged_rows;
}

template <class T, class IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT> merge_rebuild(
  raft::resources const& handle,
  const cagra::index_params& params,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*>& indices,
  DatasetViewT merged_dataset,
  const cuvs::neighbors::filtering::base_filter& row_filter)
{
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>;

  int64_t merged_rows = 0;
  uint32_t dim        = 0;
  int64_t stride      = -1;

  RAFT_EXPECTS(row_filter.get_filter_type() != cuvs::neighbors::filtering::FilterType::Bitmap,
               "Bitmap filter isn't supported inside cagra::merge");
  RAFT_EXPECTS(row_filter.get_filter_type() != cuvs::neighbors::filtering::FilterType::Bloom,
               "Bloom filter isn't supported inside cagra::merge");

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
      merged_rows += static_cast<int64_t>(index->size());
    } else {
      RAFT_FAIL("cagra::merge only supports an uncompressed dense device dataset index");
    }
  }

  bool const bitset_filtered =
    row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::Bitset;
  int64_t const final_rows =
    merged_dataset_size<T, IdxT, DatasetViewT>(handle, indices, row_filter);

  RAFT_EXPECTS(merged_dataset.n_rows() == final_rows,
               "merged_dataset rows (%ld) must equal the final merged row count (%ld)",
               long(merged_dataset.n_rows()),
               long(final_rows));
  RAFT_EXPECTS(merged_dataset.dim() == dim,
               "merged_dataset dimension (%u) must equal the input dimension (%u)",
               unsigned(merged_dataset.dim()),
               unsigned(dim));
  RAFT_EXPECTS(merged_dataset.stride() == stride,
               "merged_dataset stride (%u) must equal the input stride (%ld)",
               unsigned(merged_dataset.stride()),
               long(stride));

  auto output_const_view = merged_dataset.view();
  auto output_view       = raft::make_device_matrix_view<T, int64_t>(
    const_cast<T*>(output_const_view.data_handle()), final_rows, stride);

  auto merge_dataset = [&](T* dst, std::size_t dst_ld) {
    IdxT row_offset = 0;
    for (cagra_index_t* index : indices) {
      const T* src_ptr   = nullptr;
      std::size_t n_rows = 0;
      auto const& v      = index->dataset();
      if constexpr (cuvs::neighbors::is_dense_row_major_dataset_view_v<std::decay_t<decltype(v)>>) {
        src_ptr = v.view().data_handle();
        n_rows  = static_cast<std::size_t>(v.n_rows());
      } else {
        RAFT_FAIL("cagra::merge: unexpected dataset type while copying rows");
      }
      raft::copy_matrix(dst + static_cast<std::size_t>(row_offset) * dst_ld,
                        dst_ld,
                        src_ptr,
                        static_cast<std::size_t>(stride),
                        static_cast<std::size_t>(dim),
                        n_rows,
                        raft::resource::get_cuda_stream(handle));

      row_offset += IdxT(index->dataset().n_rows());
    }
  };

  auto build_merged_index = [&] {
    auto build_params                    = params;
    build_params.attach_dataset_on_build = false;
    auto index = ::cuvs::neighbors::cagra::build(handle, build_params, merged_dataset);
    index      = ::cuvs::neighbors::cagra::update_dataset(handle, std::move(index), merged_dataset);
    RAFT_LOG_DEBUG("cagra merge: using device memory for merged dataset");
    return index;
  };

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  if (bitset_filtered) {
    auto staging = raft::make_device_mdarray<T, int64_t>(
      handle,
      raft::resource::get_large_workspace_resource_ref(handle),
      raft::make_extents<int64_t>(merged_rows, stride));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      staging.data_handle(), 0, static_cast<std::size_t>(staging.size()) * sizeof(T), stream));
    merge_dataset(staging.data_handle(), static_cast<std::size_t>(stride));

    auto actual_filter =
      dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(row_filter);

    auto indices_csr = raft::make_device_csr_matrix<uint32_t, int64_t, int64_t, int64_t>(
      handle, 1, static_cast<std::size_t>(merged_rows));
    indices_csr.initialize_sparsity(final_rows);

    actual_filter.view().to_csr(handle, indices_csr);

    auto csr_indices  = indices_csr.structure_view().get_indices();
    auto indices_view = raft::make_device_vector_view<const int64_t, int64_t>(
      csr_indices.data(), static_cast<int64_t>(csr_indices.size()));

    RAFT_CUDA_TRY(cudaMemsetAsync(
      output_view.data_handle(),
      0,
      static_cast<std::size_t>(final_rows) * static_cast<std::size_t>(stride) * sizeof(T),
      stream));

    raft::matrix::copy_rows(
      handle, raft::make_const_mdspan(staging.view()), output_view, indices_view);

    return build_merged_index();
  }

  RAFT_CUDA_TRY(cudaMemsetAsync(
    output_view.data_handle(),
    0,
    static_cast<std::size_t>(final_rows) * static_cast<std::size_t>(stride) * sizeof(T),
    stream));
  merge_dataset(output_view.data_handle(), static_cast<std::size_t>(stride));
  return build_merged_index();
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

/** Copy every input dataset into its row range of the caller-supplied merged dataset. Both sides
 *  carry a row pitch: the inputs share one stride (enforced by preflight) and the destination uses
 *  the merged dataset's own stride. */
template <typename T, typename IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
void copy_input_datasets(
  raft::resources const& handle,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*> const& indices,
  std::vector<int64_t> const& offsets,
  int64_t dim,
  int64_t destination_stride,
  T* destination)
{
  for (std::size_t i = 0; i < indices.size(); ++i) {
    auto const& source = indices[i]->dataset();
    raft::copy_matrix(destination + offsets[i] * destination_stride,
                      static_cast<std::size_t>(destination_stride),
                      source.view().data_handle(),
                      static_cast<std::size_t>(source.stride()),
                      static_cast<std::size_t>(dim),
                      static_cast<std::size_t>(source.n_rows()),
                      raft::resource::get_cuda_stream(handle));
  }
}

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
  auto* destination            = const_cast<T*>(output_const_view.data_handle());
  {
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> scope("cagra::merge/consolidate");
    // The copy below overwrites columns [0, dim), while the sorter computes L2 over [0, stride).
    // Zero the remaining padded columns; when stride == dim, there are none to initialize.
    if (stride > preflight.dim) {
      RAFT_CUDA_TRY(cudaMemset2DAsync(destination + preflight.dim,
                                      static_cast<std::size_t>(stride) * sizeof(T),
                                      0,
                                      static_cast<std::size_t>(stride - preflight.dim) * sizeof(T),
                                      static_cast<std::size_t>(preflight.rows),
                                      raft::resource::get_cuda_stream(handle)));
    }
    copy_input_datasets<T, IdxT, DatasetViewT>(
      handle, indices, preflight.offsets, preflight.dim, stride, destination);
  }
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
      handle, params, indices, merged_dataset, row_filter);
  }

  auto preflight =
    preflight_fastener<T, IdxT, DatasetViewT>(handle, params, merge_params, indices, row_filter);
  if (!preflight.eligible) {
    if (merge_params.algo == cagra::merge_algo::AUTO) {
      return merge_rebuild<T, IdxT, DatasetViewT>(
        handle, params, indices, merged_dataset, row_filter);
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
        handle, params, indices, merged_dataset, row_filter);
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
           cuvs::neighbors::filtering::base_filter const& row_filter)
  -> cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>
{
  // Fully qualified: an unqualified call also finds cuvs::neighbors::cagra::merge via ADL on the
  // index arguments, which is ambiguous with this overload.
  return cuvs::neighbors::cagra::detail::merge<T, IdxT, DatasetViewT>(
    handle, params, indices, merged_dataset, cagra::merge_params{}, row_filter);
}

}  // namespace cuvs::neighbors::cagra::detail
