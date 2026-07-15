/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"
#include "../detail/ann_utils.cuh"
#include "../ivf_common.cuh"                  // cuvs::neighbors::detail::ivf
#include "ivf_flat_interleaved_scan_ext.cuh"  // interleaved_scan
#include <cuvs/neighbors/common.hpp>          // none_sample_filter
#include <cuvs/neighbors/ivf_flat.hpp>        // cuvs::neighbors::ivf_flat::index

#include "../detail/ann_utils.cuh"      // utils::mapping
#include <cuvs/distance/distance.hpp>   // is_min_close, DistanceType
#include <cuvs/selection/select_k.hpp>  // cuvs::selection::select_k
#include <raft/core/error.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>  // raft::resources
#include <raft/linalg/gemm.cuh>     // raft::linalg::gemm
#include <raft/linalg/map.cuh>      // raft::linalg::map
#include <raft/linalg/norm.cuh>     // raft::linalg::norm
#include <raft/matrix/detail/select_warpsort.cuh>

#include <rmm/resource_ref.hpp>

#include <type_traits>

namespace cuvs::neighbors::ivf_flat::detail {

using namespace cuvs::spatial::knn::detail;  // NOLINT

auto RAFT_WEAK_FUNCTION is_local_topk_feasible(uint32_t k) -> bool
{
  return k <= raft::matrix::detail::select::warpsort::kMaxCapacity;
}

template <typename T, typename AccT, typename IdxT, typename IvfSampleFilterT>
void search_impl(raft::resources const& handle,
                 const cuvs::neighbors::ivf_flat::index<T, IdxT>& index,
                 cuvs::distance::DistanceType effective_metric,
                 const search_params& params,
                 const T* queries,
                 uint32_t n_queries,
                 uint32_t queries_offset,
                 uint32_t k,
                 uint32_t n_probes,
                 uint32_t max_samples,
                 bool select_min,
                 IdxT* neighbors,
                 AccT* distances,
                 rmm::device_async_resource_ref search_mr,
                 IvfSampleFilterT sample_filter)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  std::size_t n_queries_probes = std::size_t(n_queries) * std::size_t(n_probes);

  // The norm of query
  rmm::device_uvector<float> query_norm_dev(n_queries, stream, search_mr);
  // The distance value of cluster(list) and queries
  rmm::device_uvector<float> distance_buffer_dev(n_queries * index.n_lists(), stream, search_mr);
  // The topk distance value of cluster(list) and queries
  rmm::device_uvector<float> coarse_distances_dev(n_queries_probes, stream, search_mr);
  // The topk  index of cluster(list) and queries
  rmm::device_uvector<uint32_t> coarse_indices_dev(n_queries_probes, stream, search_mr);

  // Optional structures if postprocessing is required
  // The topk distance value of candidate vectors from each cluster(list)
  rmm::device_uvector<AccT> distances_tmp_dev(0, stream, search_mr);
  // Number of samples for each query
  rmm::device_uvector<uint32_t> num_samples(0, stream, search_mr);
  // Offsets per probe for each query
  rmm::device_uvector<uint32_t> chunk_index(0, stream, search_mr);

  // The topk index of candidate vectors from each cluster(list), local index offset
  // also we might need additional storage for select_k
  rmm::device_uvector<uint32_t> indices_tmp_dev(0, stream, search_mr);
  rmm::device_uvector<uint32_t> neighbors_uint32_buf(0, stream, search_mr);
  auto distance_buffer_dev_view = raft::make_device_matrix_view<AccT, int64_t>(
    distance_buffer_dev.data(), n_queries, index.n_lists());

  size_t float_query_size;
  if constexpr (std::is_same_v<T, float>) {
    float_query_size = 0;
  } else {
    float_query_size = n_queries * index.dim();
  }
  rmm::device_uvector<float> converted_queries_dev(float_query_size, stream, search_mr);
  float* converted_queries_ptr = converted_queries_dev.data();

  if constexpr (std::is_same_v<T, float>) {
    converted_queries_ptr = const_cast<float*>(queries);
  } else {
    raft::linalg::map(
      handle,
      raft::make_device_vector_view<float>(converted_queries_ptr, n_queries * index.dim()),
      utils::mapping<float>{},
      raft::make_const_mdspan(
        raft::make_device_vector_view<const T>(queries, n_queries * index.dim())));
  }

  float alpha = 1.0f;
  float beta  = 0.0f;

  // todo(lsugy): raft distance? (if performance is similar/better than gemm)
  switch (effective_metric) {
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2SqrtExpanded: {
      alpha = -2.0f;
      beta  = 1.0f;
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const float, IdxT, raft::row_major>(
          converted_queries_ptr, static_cast<IdxT>(n_queries), static_cast<IdxT>(index.dim())),
        raft::make_device_vector_view<float, IdxT>(query_norm_dev.data(),
                                                   static_cast<IdxT>(n_queries)));
      utils::outer_add(query_norm_dev.data(),
                       (IdxT)n_queries,
                       index.center_norms()->data_handle(),
                       (IdxT)index.n_lists(),
                       distance_buffer_dev.data(),
                       stream);
      RAFT_LOG_TRACE_VEC(index.center_norms()->data_handle(), std::min<uint32_t>(20, index.dim()));
      RAFT_LOG_TRACE_VEC(distance_buffer_dev.data(), std::min<uint32_t>(20, index.n_lists()));
      break;
    }
    case cuvs::distance::DistanceType::CosineExpanded: {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const float, IdxT, raft::row_major>(
          converted_queries_ptr, static_cast<IdxT>(n_queries), static_cast<IdxT>(index.dim())),
        raft::make_device_vector_view<float, IdxT>(query_norm_dev.data(),
                                                   static_cast<IdxT>(n_queries)),
        raft::sqrt_op{});
      alpha = -1.0f;
      beta  = 0.0f;
      break;
    }
    default: {
      alpha = 1.0f;
      beta  = 0.0f;
    }
  }

  raft::linalg::gemm(handle,
                     true,
                     false,
                     index.n_lists(),
                     n_queries,
                     index.dim(),
                     &alpha,
                     index.centers().data_handle(),
                     index.dim(),
                     converted_queries_ptr,
                     index.dim(),
                     &beta,
                     distance_buffer_dev.data(),
                     index.n_lists(),
                     stream);

  if (effective_metric == cuvs::distance::DistanceType::CosineExpanded) {
    auto n_lists                      = index.n_lists();
    const auto* q_norm_ptr            = query_norm_dev.data();
    const auto* index_center_norm_ptr = index.center_norms()->data_handle();
    raft::linalg::map_offset(
      handle,
      distance_buffer_dev_view,
      [=] __device__(const uint32_t idx, const float dist) {
        const auto query   = idx / n_lists;
        const auto cluster = idx % n_lists;
        return dist / (q_norm_ptr[query] * index_center_norm_ptr[cluster]);
      },
      raft::make_const_mdspan(distance_buffer_dev_view));
  }
  RAFT_LOG_TRACE_VEC(distance_buffer_dev.data(), std::min<uint32_t>(20, index.n_lists()));

  cuvs::selection::select_k(
    handle,
    raft::make_const_mdspan(distance_buffer_dev_view),
    std::nullopt,
    raft::make_device_matrix_view<AccT, int64_t>(coarse_distances_dev.data(), n_queries, n_probes),
    raft::make_device_matrix_view<uint32_t, int64_t>(
      coarse_indices_dev.data(), n_queries, n_probes),
    select_min);

  RAFT_LOG_TRACE_VEC(coarse_indices_dev.data(), n_probes);
  RAFT_LOG_TRACE_VEC(coarse_distances_dev.data(), n_probes);

  uint32_t grid_dim_x = 0;
  if (n_probes > 1) {
    // query the gridDimX size to store probes topK output
    ivfflat_interleaved_scan<T, typename utils::config<T>::value_t, IdxT, IvfSampleFilterT>(
      index,
      nullptr,
      nullptr,
      n_queries,
      queries_offset,
      effective_metric,
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
      params.metric_udf);
  } else {
    grid_dim_x = 1;
  }

  num_samples.resize(n_queries, stream);
  chunk_index.resize(n_queries_probes, stream);

  ivf::detail::calc_chunk_indices::configure(n_probes, n_queries)(index.list_sizes().data_handle(),
                                                                  coarse_indices_dev.data(),
                                                                  chunk_index.data(),
                                                                  num_samples.data(),
                                                                  stream);

  auto distances_dev_ptr = distances;

  uint32_t* neighbors_uint32 = nullptr;
  if constexpr (sizeof(IdxT) == sizeof(uint32_t)) {
    neighbors_uint32 = reinterpret_cast<uint32_t*>(neighbors);
  } else {
    neighbors_uint32_buf.resize(std::size_t(n_queries) * std::size_t(k), stream);
    neighbors_uint32 = neighbors_uint32_buf.data();
  }

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

  ivfflat_interleaved_scan<T, typename utils::config<T>::value_t, IdxT, IvfSampleFilterT>(
    index,
    queries,
    coarse_indices_dev.data(),
    n_queries,
    queries_offset,
    effective_metric,
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
    params.metric_udf);

  RAFT_LOG_TRACE_VEC(distances_dev_ptr, 2 * k);
  if (indices_dev_ptr != nullptr) { RAFT_LOG_TRACE_VEC(indices_dev_ptr, 2 * k); }

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
      raft::make_device_matrix_view<const AccT, int64_t>(distances_tmp_dev.data(), n_queries, cols),
      raft::make_device_matrix_view<const uint32_t, int64_t>(
        indices_tmp_dev.data(), n_queries, cols),
      raft::make_device_matrix_view<AccT, int64_t>(distances, n_queries, k),
      raft::make_device_matrix_view<uint32_t, int64_t>(neighbors_uint32, n_queries, k),
      select_min,
      false,
      cuvs::selection::SelectAlgo::kAuto,
      num_samples_vector);
  }
  if (!manage_local_topk) {
    // post process distances && neighbor IDs
    ivf::detail::postprocess_distances(
      handle, distances, distances, effective_metric, n_queries, k, 1.0, false);
  }
  ivf::detail::postprocess_neighbors(neighbors,
                                     neighbors_uint32,
                                     index.inds_ptrs().data_handle(),
                                     coarse_indices_dev.data(),
                                     chunk_index.data(),
                                     n_queries,
                                     n_probes,
                                     k,
                                     stream);
}

/** See cuvs::neighbors::ivf_flat::search docs */
template <typename T,
          typename IdxT,
          typename IvfSampleFilterT = cuvs::neighbors::filtering::none_sample_filter>
inline void search_with_filtering(raft::resources const& handle,
                                  const search_params& params,
                                  const index<T, IdxT>& index,
                                  cuvs::distance::DistanceType effective_metric,
                                  const T* queries,
                                  uint32_t n_queries,
                                  uint32_t k,
                                  IdxT* neighbors,
                                  float* distances,
                                  IvfSampleFilterT sample_filter = IvfSampleFilterT())
{
  common::nvtx::range<common::nvtx::domain::cuvs> fun_scope(
    "ivf_flat::search(k = %u, n_queries = %u, dim = %zu)", k, n_queries, index.dim());

  RAFT_EXPECTS(params.n_probes > 0,
               "n_probes (number of clusters to probe in the search) must be positive.");
  auto n_probes          = std::min<uint32_t>(params.n_probes, index.n_lists());
  bool manage_local_topk = is_local_topk_feasible(k);

  uint32_t max_samples = 0;
  if (!manage_local_topk) {
    IdxT ms = raft::Pow2<128 / sizeof(float)>::roundUp(
      std::max<IdxT>(index.accum_sorted_sizes()(n_probes), k));
    RAFT_EXPECTS(ms <= IdxT(std::numeric_limits<uint32_t>::max()),
                 "The maximum sample size is too big.");
    max_samples = ms;
  }

  // a batch size heuristic: try to keep the workspace within the specified size
  constexpr uint64_t kExpectedWsSize = 1024 * 1024 * 1024;
  uint64_t max_ws_size =
    std::min(raft::resource::get_workspace_free_bytes(handle), kExpectedWsSize);

  uint64_t ws_size_per_query = 4ull * (2 * n_probes + index.n_lists() + index.dim() + 1) +
                               (manage_local_topk ? ((sizeof(IdxT) + 4) * n_probes * k)
                                                  : (4ull * (max_samples + n_probes + 1)));

  const uint32_t max_queries =
    std::min<uint32_t>(n_queries, raft::div_rounding_up_safe(max_ws_size, ws_size_per_query));

  for (uint32_t offset_q = 0; offset_q < n_queries; offset_q += max_queries) {
    uint32_t queries_batch = raft::min(max_queries, n_queries - offset_q);

    search_impl<T, float, IdxT, IvfSampleFilterT>(
      handle,
      index,
      effective_metric,
      params,
      queries + offset_q * index.dim(),
      queries_batch,
      offset_q,
      k,
      n_probes,
      max_samples,
      cuvs::distance::is_min_close(effective_metric),
      neighbors + offset_q * k,
      distances + offset_q * k,
      raft::resource::get_workspace_resource_ref(handle),
      sample_filter);
  }
}

template <typename T, typename IdxT, typename IvfSampleFilterT>
void search_with_filtering(raft::resources const& handle,
                           const search_params& params,
                           const index<T, IdxT>& index,
                           raft::device_matrix_view<const T, IdxT, raft::row_major> queries,
                           raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,
                           raft::device_matrix_view<float, IdxT, raft::row_major> distances,
                           IvfSampleFilterT sample_filter = IvfSampleFilterT())
{
  RAFT_EXPECTS(
    queries.extent(0) == neighbors.extent(0) && queries.extent(0) == distances.extent(0),
    "Number of rows in output neighbors and distances matrices must equal the number of queries.");

  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "Number of columns in output neighbors and distances matrices must be equal");

  RAFT_EXPECTS(queries.extent(1) == index.dim(),
               "Number of query dimensions should equal number of dimensions in the index.");

  auto effective_metric =
    params.metric_udf.has_value() ? cuvs::distance::DistanceType::CustomUDF : index.metric();

  search_with_filtering(handle,
                        params,
                        index,
                        effective_metric,
                        queries.data_handle(),
                        static_cast<std::uint32_t>(queries.extent(0)),
                        static_cast<std::uint32_t>(neighbors.extent(1)),
                        neighbors.data_handle(),
                        distances.data_handle(),
                        sample_filter);
}

template <typename T, typename IdxT>
void search(raft::resources const& handle,
            const search_params& params,
            const index<T, IdxT>& idx,
            raft::device_matrix_view<const T, IdxT, raft::row_major> queries,
            raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,
            raft::device_matrix_view<float, IdxT, raft::row_major> distances,
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

}  // namespace cuvs::neighbors::ivf_flat::detail
