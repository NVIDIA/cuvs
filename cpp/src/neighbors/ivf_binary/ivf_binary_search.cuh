/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/map_launchers.hpp"
#include "../../core/nvtx.hpp"
#include "../../distance/detail/distance_ops/bitwise_hamming.cuh"
#include "../../distance/detail/pairwise_matrix/dispatch.cuh"
#include "../detail/ann_utils.cuh"
#include "../ivf_common.cuh"
#include "../ivf_flat/ivf_flat_interleaved_scan_ext.cuh"
#include "ivf_binary_index_access.hpp"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_binary.hpp>
#include <cuvs/selection/select_k.hpp>

#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/util/pow2_utils.cuh>

#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <typeinfo>

namespace cuvs::neighbors::ivf_binary::detail {

using namespace cuvs::spatial::knn::detail;  // NOLINT

constexpr auto kMetric = cuvs::distance::DistanceType::BitwiseHamming;

inline auto is_local_topk_feasible(uint32_t k) -> bool
{
  return k <= raft::matrix::detail::select::warpsort::kMaxCapacity;
}

template <typename IvfSampleFilterT>
void search_impl(raft::resources const& handle,
                 const index<int64_t>& idx,
                 const uint8_t* queries,
                 uint32_t n_queries,
                 uint32_t queries_offset,
                 uint32_t k,
                 uint32_t n_probes,
                 uint32_t max_samples,
                 int64_t* neighbors,
                 float* distances,
                 rmm::device_async_resource_ref search_mr,
                 IvfSampleFilterT sample_filter)
{
  using IdxT                   = int64_t;
  constexpr bool select_min    = true;
  auto stream                  = raft::resource::get_cuda_stream(handle);
  const auto& lists            = index_access::lists(idx);
  const uint32_t n_lists       = idx.n_lists();
  const uint32_t dim           = idx.storage_dim();
  std::size_t n_queries_probes = std::size_t(n_queries) * std::size_t(n_probes);

  // The distance value of cluster(list) and queries
  rmm::device_uvector<float> distance_buffer_dev(size_t(n_queries) * n_lists, stream, search_mr);
  // The topk distance value of cluster(list) and queries
  rmm::device_uvector<float> coarse_distances_dev(n_queries_probes, stream, search_mr);
  // The topk  index of cluster(list) and queries
  rmm::device_uvector<uint32_t> coarse_indices_dev(n_queries_probes, stream, search_mr);

  // Optional structures if postprocessing is required
  // The topk distance value of candidate vectors from each cluster(list)
  rmm::device_uvector<float> distances_tmp_dev(0, stream, search_mr);
  // Number of samples for each query
  rmm::device_uvector<uint32_t> num_samples(0, stream, search_mr);
  // Offsets per probe for each query
  rmm::device_uvector<uint32_t> chunk_index(0, stream, search_mr);
  // The topk index of candidate vectors from each cluster(list), local index offset
  // also we might need additional storage for select_k
  rmm::device_uvector<uint32_t> indices_tmp_dev(0, stream, search_mr);
  rmm::device_uvector<uint32_t> neighbors_uint32_buf(0, stream, search_mr);
  auto distance_buffer_dev_view =
    raft::make_device_matrix_view<float, int64_t>(distance_buffer_dev.data(), n_queries, n_lists);

  // Coarse search: Hamming distances to the centers
  {
    cuvs::distance::detail::ops::bitwise_hamming_distance_op<uint8_t, uint32_t, IdxT> distance_op{
      static_cast<IdxT>(dim)};
    rmm::device_uvector<uint32_t> uint32_distances(size_t(n_queries) * n_lists, stream, search_mr);
    cuvs::distance::detail::pairwise_matrix_dispatch<decltype(distance_op),
                                                     uint8_t,
                                                     uint32_t,
                                                     uint32_t,
                                                     raft::identity_op,
                                                     IdxT>(distance_op,
                                                           static_cast<IdxT>(n_queries),
                                                           static_cast<IdxT>(n_lists),
                                                           static_cast<IdxT>(dim),
                                                           queries,
                                                           idx.centers().data_handle(),
                                                           nullptr,
                                                           nullptr,
                                                           uint32_distances.data(),
                                                           raft::identity_op{},
                                                           stream.get(),
                                                           true);
    cuvs::detail::map(
      handle,
      distance_buffer_dev_view,
      raft::cast_op<float>{},
      raft::make_const_mdspan(raft::make_device_matrix_view<const uint32_t, int64_t>(
        uint32_distances.data(), n_queries, n_lists)));
  }

  cuvs::selection::select_k(
    handle,
    raft::make_const_mdspan(distance_buffer_dev_view),
    std::nullopt,
    raft::make_device_matrix_view<float, int64_t>(coarse_distances_dev.data(), n_queries, n_probes),
    raft::make_device_matrix_view<uint32_t, int64_t>(
      coarse_indices_dev.data(), n_queries, n_probes),
    select_min);

  using ScanAccT      = typename utils::config<uint8_t>::value_t;
  uint32_t grid_dim_x = 0;
  if (n_probes > 1) {
    // query the gridDimX size to store probes topK output
    ivf_flat::detail::ivfflat_interleaved_scan<uint8_t, ScanAccT, IdxT, IvfSampleFilterT>(
      lists,
      nullptr,
      nullptr,
      n_queries,
      queries_offset,
      kMetric,
      n_probes,
      k,
      0,
      nullptr,
      select_min,
      sample_filter,
      nullptr,
      nullptr,
      grid_dim_x,
      stream,
      std::nullopt);
  } else {
    grid_dim_x = 1;
  }

  num_samples.resize(n_queries, stream);
  chunk_index.resize(n_queries_probes, stream);

  ivf::detail::calc_chunk_indices::configure(n_probes, n_queries)(lists.list_sizes().data_handle(),
                                                                  coarse_indices_dev.data(),
                                                                  chunk_index.data(),
                                                                  num_samples.data(),
                                                                  stream);

  auto distances_dev_ptr = distances;

  neighbors_uint32_buf.resize(std::size_t(n_queries) * std::size_t(k), stream);
  uint32_t* neighbors_uint32 = neighbors_uint32_buf.data();

  uint32_t* indices_dev_ptr = nullptr;

  bool manage_local_topk = is_local_topk_feasible(k);
  if (!manage_local_topk || grid_dim_x > 1) {
    auto target_size = std::size_t(n_queries) * (manage_local_topk ? grid_dim_x * k : max_samples);

    distances_tmp_dev.resize(target_size, stream);
    if (manage_local_topk) indices_tmp_dev.resize(target_size, stream);

    distances_dev_ptr = distances_tmp_dev.data();
    indices_dev_ptr   = indices_tmp_dev.data();
  } else {
    indices_dev_ptr = neighbors_uint32;
  }

  ivf_flat::detail::ivfflat_interleaved_scan<uint8_t, ScanAccT, IdxT, IvfSampleFilterT>(
    lists,
    queries,
    coarse_indices_dev.data(),
    n_queries,
    queries_offset,
    kMetric,
    n_probes,
    k,
    max_samples,
    chunk_index.data(),
    select_min,
    sample_filter,
    indices_dev_ptr,
    distances_dev_ptr,
    grid_dim_x,
    stream,
    std::nullopt);

  // Merge topk values from different blocks
  if (!manage_local_topk || grid_dim_x > 1) {
    std::optional<raft::device_vector_view<const uint32_t>> num_samples_vector;
    if (!manage_local_topk) {
      num_samples_vector =
        raft::make_device_vector_view<const uint32_t>(num_samples.data(), n_queries);
    }

    auto cols = manage_local_topk ? (k * grid_dim_x) : max_samples;

    cuvs::selection::select_k(
      handle,
      raft::make_device_matrix_view<const float, int64_t>(
        distances_tmp_dev.data(), n_queries, cols),
      raft::make_device_matrix_view<const uint32_t, int64_t>(
        indices_tmp_dev.data(), n_queries, cols),
      raft::make_device_matrix_view<float, int64_t>(distances, n_queries, k),
      raft::make_device_matrix_view<uint32_t, int64_t>(neighbors_uint32, n_queries, k),
      select_min,
      false,
      cuvs::selection::SelectAlgo::kAuto,
      num_samples_vector);
  }
  if (!manage_local_topk) {
    // post process distances && neighbor IDs
    ivf::detail::postprocess_distances(
      handle, distances, distances, kMetric, n_queries, k, 1.0, false);
  }
  ivf::detail::postprocess_neighbors(neighbors,
                                     neighbors_uint32,
                                     lists.inds_ptrs().data_handle(),
                                     coarse_indices_dev.data(),
                                     chunk_index.data(),
                                     n_queries,
                                     n_probes,
                                     k,
                                     stream);
}

template <typename IvfSampleFilterT>
void search_with_filtering(
  raft::resources const& handle,
  const search_params& params,
  const index<int64_t>& idx,
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  IvfSampleFilterT sample_filter)
{
  RAFT_EXPECTS(
    queries.extent(0) == neighbors.extent(0) && queries.extent(0) == distances.extent(0),
    "Number of rows in output neighbors and distances matrices must equal the number of queries.");
  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "Number of columns in output neighbors and distances matrices must be equal");
  RAFT_EXPECTS(queries.extent(1) == idx.storage_dim(),
               "Number of query bytes should equal the number of bytes in the index vectors.");
  RAFT_EXPECTS(params.n_probes > 0,
               "n_probes (number of clusters to probe in the search) must be positive.");
  const uint32_t n_queries = queries.extent(0);
  const uint32_t k         = neighbors.extent(1);
  if (n_queries == 0 || k == 0) { return; }

  common::nvtx::range<common::nvtx::domain::cuvs> fun_scope(
    "ivf_binary::search(k = %u, n_queries = %u, dim = %u)", k, n_queries, idx.dim());

  const auto& lists      = index_access::lists(idx);
  auto n_probes          = std::min<uint32_t>(params.n_probes, idx.n_lists());
  bool manage_local_topk = is_local_topk_feasible(k);

  uint32_t max_samples = 0;
  if (!manage_local_topk) {
    int64_t ms = raft::Pow2<128 / sizeof(float)>::roundUp(
      std::max<int64_t>(lists.accum_sorted_sizes()(n_probes), k));
    RAFT_EXPECTS(ms <= int64_t(std::numeric_limits<uint32_t>::max()),
                 "The maximum sample size is too big.");
    max_samples = ms;
  }

  // a batch size heuristic: try to keep the workspace within the specified size
  constexpr uint64_t kExpectedWsSize = 1024 * 1024 * 1024;
  uint64_t max_ws_size =
    std::min(raft::resource::get_workspace_free_bytes(handle), kExpectedWsSize);

  uint64_t ws_size_per_query = 4ull * (2ull * n_probes + 2ull * idx.n_lists() + 1) +
                               (manage_local_topk ? ((sizeof(int64_t) + 4) * n_probes * k)
                                                  : (4ull * (max_samples + n_probes + 1)));

  const uint32_t max_queries =
    std::min<uint32_t>(n_queries, raft::div_rounding_up_safe(max_ws_size, ws_size_per_query));

  for (uint32_t offset_q = 0; offset_q < n_queries; offset_q += max_queries) {
    uint32_t queries_batch = raft::min(max_queries, n_queries - offset_q);

    search_impl<IvfSampleFilterT>(handle,
                                  idx,
                                  queries.data_handle() + size_t(offset_q) * idx.storage_dim(),
                                  queries_batch,
                                  offset_q,
                                  k,
                                  n_probes,
                                  max_samples,
                                  neighbors.data_handle() + size_t(offset_q) * k,
                                  distances.data_handle() + size_t(offset_q) * k,
                                  raft::resource::get_workspace_resource_ref(handle),
                                  sample_filter);
  }
}

/** Search with the binary-quantized queries; see cuvs::neighbors::ivf_binary::search. */
inline void search(raft::resources const& handle,
                   const search_params& params,
                   const index<int64_t>& idx,
                   raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
                   raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                   raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                   const cuvs::neighbors::filtering::base_filter& sample_filter_ref)
{
  try {
    auto& sample_filter =
      dynamic_cast<const cuvs::neighbors::filtering::none_sample_filter&>(sample_filter_ref);
    return search_with_filtering(handle, params, idx, queries, neighbors, distances, sample_filter);
  } catch (const std::bad_cast&) {
  }

  try {
    auto& sample_filter =
      dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(
        sample_filter_ref);
    return search_with_filtering(handle, params, idx, queries, neighbors, distances, sample_filter);
  } catch (const std::bad_cast&) {
    RAFT_FAIL("Unsupported sample filter type");
  }
}

}  // namespace cuvs::neighbors::ivf_binary::detail
