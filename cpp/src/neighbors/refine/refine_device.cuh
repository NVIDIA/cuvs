/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"
#include "../detail/ann_utils.cuh"
#include "../ivf_flat/ivf_flat_build.cuh"
#include "../ivf_flat/ivf_flat_interleaved_scan_ext.cuh"
#include "refine_common.hpp"
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <optional>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/matrix/init.cuh>
#include <raft/util/cuda_utils.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cuvs::neighbors {

namespace detail {

namespace refine_fast {

enum class metric_kind { kL2, kL2Sqrt, kInnerProduct, kCosine };

/** Accumulator type: exact integer arithmetic for 8-bit data, float otherwise. */
template <typename DataT>
using acc_t = std::
  conditional_t<std::is_same_v<DataT, int8_t> || std::is_same_v<DataT, uint8_t>, int32_t, float>;

/** Shared memory per warp limits the number of candidates the fast path can handle. */
constexpr uint32_t kMaxSmemBytes = 48 * 1024;
constexpr uint32_t kMaxWarps     = 4;

inline auto max_candidates() -> uint32_t { return kMaxSmemBytes / sizeof(float); }

/**
 * One warp per query: computes the distances to all candidates of the query directly from the
 * dataset rows (lanes stride over the dimensions, so every row read is coalesced) and selects the
 * k best ones. Ties are resolved in favor of the candidate that comes first.
 *
 * `keys` holds the selection keys in shared memory (smaller is better). Invalid and already
 * selected candidates get `upper_bound<float>()` (+inf), so selecting it means that no valid
 * candidates are left.
 */
template <metric_kind Metric, typename DataT, typename IdxT, typename DistT>
RAFT_KERNEL refine_warp_kernel(const DataT* dataset,
                               int64_t dim,
                               const DataT* queries,
                               const IdxT* candidates,
                               uint32_t n_candidates,
                               uint32_t k,
                               int64_t n_queries,
                               IdxT* out_indices,
                               DistT* out_distances)
{
  using AccT = acc_t<DataT>;
  extern __shared__ float smem_keys[];
  const uint32_t lane    = threadIdx.x % raft::WarpSize;
  const uint32_t warp_id = threadIdx.x / raft::WarpSize;
  const int64_t query_ix = int64_t(blockIdx.x) * (blockDim.x / raft::WarpSize) + warp_id;
  if (query_ix >= n_queries) { return; }

  float* keys          = smem_keys + size_t(warp_id) * n_candidates;
  const DataT* query   = queries + query_ix * dim;
  const IdxT* cand_row = candidates + query_ix * n_candidates;

  float query_norm = 0;
  if constexpr (Metric == metric_kind::kCosine) {
    AccT qn = 0;
    for (int64_t d = lane; d < dim; d += raft::WarpSize) {
      const auto q = static_cast<AccT>(query[d]);
      qn += q * q;
    }
    query_norm = raft::sqrt(static_cast<float>(raft::warpReduce(qn)));
  }

  for (uint32_t j = 0; j < n_candidates; j++) {
    const IdxT id    = cand_row[j];
    const bool valid = id != ivf::kInvalidRecord<IdxT> && id != raft::upper_bound<IdxT>();
    float key        = raft::upper_bound<float>();
    if (valid) {
      const DataT* row = dataset + int64_t(id) * dim;
      AccT acc         = 0;
      AccT row_norm    = 0;
      for (int64_t d = lane; d < dim; d += raft::WarpSize) {
        const auto x = static_cast<AccT>(row[d]);
        const auto q = static_cast<AccT>(query[d]);
        if constexpr (Metric == metric_kind::kL2 || Metric == metric_kind::kL2Sqrt) {
          const auto t = x - q;
          acc += t * t;
        } else {
          acc += x * q;
        }
        if constexpr (Metric == metric_kind::kCosine) { row_norm += x * x; }
      }
      const auto v = static_cast<float>(raft::warpReduce(acc));
      if constexpr (Metric == metric_kind::kL2 || Metric == metric_kind::kL2Sqrt) {
        key = v;
      } else if constexpr (Metric == metric_kind::kInnerProduct) {
        key = -v;
      } else {
        const auto rn = raft::sqrt(static_cast<float>(raft::warpReduce(row_norm)));
        key           = -v / (query_norm * rn);
      }
    }
    if (lane == 0) { keys[j] = key; }
  }
  __syncwarp();

  for (uint32_t r = 0; r < k; r++) {
    float best_key = raft::upper_bound<float>();
    uint32_t best  = n_candidates;
    for (uint32_t j = lane; j < n_candidates; j += raft::WarpSize) {
      if (keys[j] < best_key) {
        best_key = keys[j];
        best     = j;
      }
    }
    for (int offset = raft::WarpSize / 2; offset > 0; offset /= 2) {
      const float other_key = raft::shfl_xor(best_key, offset);
      const uint32_t other  = raft::shfl_xor(best, offset);
      if (other_key < best_key || (other_key == best_key && other < best)) {
        best_key = other_key;
        best     = other;
      }
    }
    if (lane == 0) {
      const int64_t out_ix = query_ix * k + r;
      if (best >= n_candidates || best_key == raft::upper_bound<float>()) {
        // Not enough valid candidates: mark the slot the same way the IVF-based path does.
        out_indices[out_ix]   = ivf::detail::kOutOfBoundsRecord<IdxT>;
        out_distances[out_ix] = Metric == metric_kind::kInnerProduct ? raft::lower_bound<DistT>()
                                                                     : raft::upper_bound<DistT>();
      } else {
        out_indices[out_ix] = cand_row[best];
        if constexpr (Metric == metric_kind::kL2) {
          out_distances[out_ix] = best_key;
        } else if constexpr (Metric == metric_kind::kL2Sqrt) {
          out_distances[out_ix] = raft::sqrt(best_key);
        } else if constexpr (Metric == metric_kind::kInnerProduct) {
          out_distances[out_ix] = -best_key;
        } else {
          out_distances[out_ix] = 1.0f + best_key;  // 1 - cos
        }
        keys[best] = raft::upper_bound<float>();
      }
    }
    __syncwarp();
  }
}

/** Returns false if the fast path does not support the metric or the number of candidates. */
template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
bool try_refine(raft::resources const& handle,
                raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> dataset,
                raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> queries,
                raft::device_matrix_view<const idx_t, matrix_idx, raft::row_major> candidates,
                raft::device_matrix_view<idx_t, matrix_idx, raft::row_major> indices,
                raft::device_matrix_view<distance_t, matrix_idx, raft::row_major> distances,
                cuvs::distance::DistanceType metric)
{
  const auto n_candidates = static_cast<uint32_t>(candidates.extent(1));
  const auto n_queries    = static_cast<int64_t>(queries.extent(0));
  if (n_candidates == 0 || n_candidates > max_candidates()) { return false; }
  if (n_queries == 0) { return true; }

  const uint32_t n_warps =
    std::max<uint32_t>(1, std::min<uint32_t>(kMaxWarps, max_candidates() / n_candidates));
  const dim3 block(n_warps * raft::WarpSize);
  const dim3 grid(raft::ceildiv<int64_t>(n_queries, n_warps));
  const size_t smem = size_t(n_warps) * n_candidates * sizeof(float);
  auto stream       = raft::resource::get_cuda_stream(handle);

  auto launch = [&](auto kernel) {
    kernel<<<grid, block, smem, stream.get()>>>(dataset.data_handle(),
                                                static_cast<int64_t>(dataset.extent(1)),
                                                queries.data_handle(),
                                                candidates.data_handle(),
                                                n_candidates,
                                                static_cast<uint32_t>(indices.extent(1)),
                                                n_queries,
                                                indices.data_handle(),
                                                distances.data_handle());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    return true;
  };
  using cuvs::distance::DistanceType;
  switch (metric) {
    case DistanceType::L2Expanded:
    case DistanceType::L2Unexpanded:
      return launch(refine_warp_kernel<metric_kind::kL2, data_t, idx_t, distance_t>);
    case DistanceType::L2SqrtExpanded:
    case DistanceType::L2SqrtUnexpanded:
      return launch(refine_warp_kernel<metric_kind::kL2Sqrt, data_t, idx_t, distance_t>);
    case DistanceType::InnerProduct:
      return launch(refine_warp_kernel<metric_kind::kInnerProduct, data_t, idx_t, distance_t>);
    case DistanceType::CosineExpanded:
      return launch(refine_warp_kernel<metric_kind::kCosine, data_t, idx_t, distance_t>);
    default: return false;
  }
}

}  // namespace refine_fast

/**
 * See cuvs::neighbors::refine for docs.
 */
template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine_device(
  raft::resources const& handle,
  raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> dataset,
  raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> queries,
  raft::device_matrix_view<const idx_t, matrix_idx, raft::row_major> neighbor_candidates,
  raft::device_matrix_view<idx_t, matrix_idx, raft::row_major> indices,
  raft::device_matrix_view<distance_t, matrix_idx, raft::row_major> distances,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded)
{
  matrix_idx n_candidates = neighbor_candidates.extent(1);
  matrix_idx n_queries    = queries.extent(0);
  matrix_idx dim          = dataset.extent(1);
  uint32_t k              = static_cast<uint32_t>(indices.extent(1));

  // TODO: this restriction could be lifted with some effort
  RAFT_EXPECTS(k <= raft::matrix::detail::select::warpsort::kMaxCapacity,
               "k must be less than topk::kMaxCapacity (%d).",
               raft::matrix::detail::select::warpsort::kMaxCapacity);

  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "neighbors::refine(%zu, %u)", size_t(n_queries), uint32_t(n_candidates));

  refine_check_input(dataset.extents(),
                     queries.extents(),
                     neighbor_candidates.extents(),
                     indices.extents(),
                     distances.extents(),
                     metric);

  // Fast path: compute the candidate distances directly from the dataset rows, one warp per query.
  if (refine_fast::try_refine(
        handle, dataset, queries, neighbor_candidates, indices, distances, metric)) {
    return;
  }

  // The refinement search can be mapped to an IVF flat search:
  // - We consider that the candidate vectors form a cluster, separately for each query.
  // - In other words, the n_queries * n_candidates vectors form n_queries clusters, each with
  //   n_candidates elements.
  // - We consider that the coarse level search is already performed and assigned a single cluster
  //   to search for each query (the cluster formed from the corresponding candidates).
  // - We run IVF flat search with n_probes=1 to select the best k elements of the candidates.
  rmm::device_uvector<uint32_t> fake_coarse_idx(n_queries, raft::resource::get_cuda_stream(handle));

  raft::linalg::map_offset(handle,
                           raft::make_device_vector_view(fake_coarse_idx.data(), n_queries),
                           raft::cast_op<uint32_t>{});

  cuvs::neighbors::ivf_flat::index<data_t, int64_t> refinement_index(
    handle, metric, n_queries, false, true, dim);

  cuvs::neighbors::ivf_flat::detail::fill_refinement_index<data_t, int64_t>(
    handle,
    &refinement_index,
    dataset.data_handle(),
    neighbor_candidates.data_handle(),
    static_cast<int64_t>(n_queries),
    static_cast<uint32_t>(n_candidates));
  uint32_t grid_dim_x = 1;

  // the neighbor ids will be computed in uint32_t as offset
  rmm::device_uvector<uint32_t> neighbors_uint32_buf(0, raft::resource::get_cuda_stream(handle));
  // Offsets per probe for each query [n_queries] as n_probes = 1
  rmm::device_uvector<uint32_t> chunk_index(n_queries, raft::resource::get_cuda_stream(handle));

  // we know that each cluster has exactly n_candidates entries
  raft::matrix::fill(
    handle, raft::make_device_vector_view(chunk_index.data(), n_queries), uint32_t(n_candidates));

  uint32_t* neighbors_uint32 = nullptr;
  if constexpr (sizeof(idx_t) == sizeof(uint32_t)) {
    neighbors_uint32 = reinterpret_cast<uint32_t*>(indices.data_handle());
  } else {
    neighbors_uint32_buf.resize(std::size_t(n_queries) * std::size_t(k),
                                raft::resource::get_cuda_stream(handle));
    neighbors_uint32 = neighbors_uint32_buf.data();
  }

  using acc_t = typename cuvs::spatial::knn::detail::utils::config<data_t>::value_t;

  cuvs::neighbors::ivf_flat::detail::ivfflat_interleaved_scan<data_t, acc_t, int64_t>(
    refinement_index,
    queries.data_handle(),
    fake_coarse_idx.data(),
    static_cast<uint32_t>(n_queries),
    0,
    metric,
    1,
    k,
    0,
    chunk_index.data(),
    cuvs::distance::is_min_close(metric),
    cuvs::neighbors::filtering::none_sample_filter(),
    neighbors_uint32,
    distances.data_handle(),
    grid_dim_x,
    raft::resource::get_cuda_stream(handle),
    std::nullopt);

  // postprocessing -- neighbors from position to actual id
  cuvs::neighbors::ivf::detail::postprocess_neighbors(indices.data_handle(),
                                                      neighbors_uint32,
                                                      refinement_index.inds_ptrs().data_handle(),
                                                      fake_coarse_idx.data(),
                                                      chunk_index.data(),
                                                      n_queries,
                                                      1,
                                                      k,
                                                      raft::resource::get_cuda_stream(handle));
}

}  // namespace detail

template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine_impl(
  raft::resources const& handle,
  raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> dataset,
  raft::device_matrix_view<const data_t, matrix_idx, raft::row_major> queries,
  raft::device_matrix_view<const idx_t, matrix_idx, raft::row_major> neighbor_candidates,
  raft::device_matrix_view<idx_t, matrix_idx, raft::row_major> indices,
  raft::device_matrix_view<distance_t, matrix_idx, raft::row_major> distances,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded)
{
  detail::refine_device(handle, dataset, queries, neighbor_candidates, indices, distances, metric);
}
}  // namespace cuvs::neighbors
