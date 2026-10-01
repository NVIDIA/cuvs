/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "device_functions.cuh"
#include "kernel_def.hpp"
#include <cuvs/detail/jit_lto/ivf_sq/scan_fragments.hpp>
#include <cuvs/neighbors/ivf_sq.hpp>
#include <neighbors/ivf_common.cuh>

#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/vectorized.cuh>

#include <cstdint>
#include <type_traits>

namespace cuvs::neighbors::ivf_sq::detail {

// block_sort type selection: dispatch the dummy block sort when Capacity == 0
// so the same impl body works for both the fused top-k path (Capacity > 0) and
// the materialize-all path (Capacity == 0).
//
// All metrics are min-close after finalize_distance (IP is negated; cosine
// returns 1 - cos_sim; L2 is squared L2), so the warpsort is hardcoded to
// ascending.
template <int Capacity>
struct sq_block_sort {
  using type = raft::matrix::detail::select::warpsort::block_sort<
    raft::matrix::detail::select::warpsort::warp_sort_filtered,
    Capacity,
    /*Ascending=*/true,
    float,
    uint32_t>;
};

template <>
struct sq_block_sort<0> {
  using type = ivf::detail::dummy_block_sort_t<float, uint32_t, /*Ascending=*/true>;
};

template <int Capacity>
using sq_block_sort_t = typename sq_block_sort<Capacity>::type;

// IVF-SQ scan kernel body with fused in-kernel top-k.
//
// The kernel is metric-agnostic: the four metric-specific pieces are
// linked in at runtime via JIT-LTO as fragments declared in
// device_functions.cuh:
//   - setup_invariant_smem  (Phase 1, once per query)
//   - setup_per_probe_smem  (Phase 2, once per probe)
//   - accumulate_distance   (per-element, inner unrolled loop)
//   - finalize_distance     (per-row, after the dim accumulation; ensures
//                            all metrics are min-close so the warpsort
//                            epilogue and the host select_k can be
//                            hardcoded to ascending / select-min)
//
// The host launcher (ivf_sq_search.cuh) registers the matching metric
// variant of each fragment with the planner. IP scores are negated by
// finalize_distance and undone by postprocess_distances.
//
// Grid layout:
//   kManageLocalTopK (Capacity > 0):
//     grid (grid_dim_x, n_queries) - each block loops over probes
//   otherwise (Capacity == 0):
//     grid (n_probes, n_queries) - one block per (query, probe)
//
// Shared-memory layout (always 3 x dim floats):
//   [s_sq_scale(dim) | s_query_term(dim) | s_aux(dim)]
//
//   s_sq_scale = delta[d]  - SQ dequantization scale, invariant (Phase 1).
//
//   L2 path:
//     Phase 1: s_aux[d] = query[d] - vmin[d]              (invariant)
//     Phase 2: s_query_term[d] = s_aux[d] - centroid[d]    (per-probe)
//     The full SQ reconstruction is centroid + vmin + code*delta, so
//     query - reconstructed = (query - vmin - centroid) - code*delta
//                           = s_query_term - code*s_sq_scale.
//
//   IP/Cosine path:
//     Phase 1: s_query_term[d] = query[d]                  (invariant)
//     Phase 2: s_aux[d] = centroid[d] + vmin[d]            (per-probe)
//     Reconstructed vector component: s_aux[d] + code*s_sq_scale[d].
//
//   After all probes are scanned, the smem is reused for block_sort merge.
template <int Capacity>
__device__ __forceinline__ void ivf_sq_scan_impl(const uint8_t* const* data_ptrs,
                                                 const uint32_t* list_sizes,
                                                 const uint32_t* coarse_indices,
                                                 const float* queries_float,
                                                 const float* centers,
                                                 const float* sq_vmin,
                                                 const float* sq_delta,
                                                 const float* query_norms,
                                                 uint32_t n_probes,
                                                 uint32_t dim,
                                                 uint32_t k,
                                                 uint32_t max_samples,
                                                 const uint32_t* chunk_indices,
                                                 float* out_distances,
                                                 uint32_t* out_indices,
                                                 const int64_t* const* inds_ptrs,
                                                 uint32_t* bitset_ptr,
                                                 int64_t bitset_len,
                                                 int64_t original_nbits)
{
  static_assert(kIndexGroupSize == raft::WarpSize,
                "Warp-coalesced scan requires kIndexGroupSize == WarpSize");

  constexpr int BlockDim          = kSqScanThreads;
  constexpr bool kManageLocalTopK = (Capacity > 0);

  extern __shared__ __align__(256) uint8_t smem_buf[];
  float* smem = reinterpret_cast<float*>(smem_buf);

  float* s_sq_scale   = smem;
  float* s_query_term = smem + dim;
  float* s_aux        = smem + 2 * dim;

  const uint32_t query_ix = blockIdx.y;
  const float* query      = queries_float + query_ix * dim;

  // Hoist the per-query scalar load to a uniform value. The cosine fragment of
  // finalize_distance is the only consumer; for L2/IP, the unused argument is
  // dead-code-eliminated after JIT-LTO inlining and this load disappears.
  const float q_norm = (query_norms != nullptr) ? query_norms[query_ix] : 0.0f;

  if constexpr (kManageLocalTopK) {
    out_distances += uint64_t(query_ix) * k * gridDim.x + blockIdx.x * k;
    out_indices += uint64_t(query_ix) * k * gridDim.x + blockIdx.x * k;
  }

  // Phase 1: load shared memory that is invariant across probes.
  for (uint32_t d = threadIdx.x; d < dim; d += BlockDim) {
    s_sq_scale[d] = sq_delta[d];
  }
  setup_invariant_smem(dim, query, sq_vmin, s_aux, s_query_term);
  __syncthreads();

  using local_topk_t = sq_block_sort_t<Capacity>;
  local_topk_t queue(k);

  const uint32_t* my_coarse = coarse_indices + query_ix * n_probes;
  const uint32_t* my_chunk  = chunk_indices + query_ix * n_probes;

  constexpr uint32_t veclen         = list_spec<uint32_t, uint8_t, int64_t>::kVecLen;
  using code_vec_t                  = raft::TxN_t<uint8_t, veclen>;
  constexpr uint32_t kWarpsPerBlock = BlockDim / raft::WarpSize;
  const uint32_t warp_id            = threadIdx.x / raft::WarpSize;
  const uint32_t lane_id            = threadIdx.x % raft::WarpSize;

  // Phase 2: loop over probes.
  // Synchronization protocol:
  //  (a) __syncthreads after Phase 1 (above) ensures invariant smem arrays
  //      (s_sq_scale, and L2: s_aux / IP-Cosine: s_query_term) are visible
  //      before Phase 2 overwrites the per-probe array.
  //  (b) __syncthreads after per-probe smem writes (L2: s_query_term /
  //      IP-Cosine: s_aux) ensures probe-specific values are visible before
  //      the distance computation.
  //  (c) __syncthreads at the end of each iteration ensures all distance
  //      computation reads are complete before the next iteration overwrites
  //      the per-probe smem region.
  //  When cluster_sz == 0, barrier (c) is skipped because no distance reads
  //  occurred; all threads converge on the same branch uniformly, and the
  //  next iteration's barrier (b) provides the needed ordering.
  for (uint32_t probe_ix = blockIdx.x; probe_ix < n_probes;
       probe_ix += (kManageLocalTopK ? gridDim.x : uint32_t{1})) {
    const uint32_t cluster_id = my_coarse[probe_ix];
    const uint32_t cluster_sz = list_sizes[cluster_id];

    setup_per_probe_smem(dim, centers + cluster_id * dim, sq_vmin, s_aux, s_query_term);
    __syncthreads();  // (b)

    if (cluster_sz == 0) {
      if constexpr (!kManageLocalTopK) break;
      continue;
    }

    const uint8_t* codes   = data_ptrs[cluster_id];
    uint32_t sample_offset = (probe_ix > 0) ? my_chunk[probe_ix - 1] : 0;
    uint32_t padded_dim    = raft::Pow2<veclen>::roundUp(dim);
    uint32_t n_dim_blocks  = padded_dim / veclen;

    for (uint32_t group = warp_id * kIndexGroupSize; group < cluster_sz;
         group += kWarpsPerBlock * kIndexGroupSize) {
      const uint32_t row = group + lane_id;
      const bool valid =
        (row < cluster_sz) &&
        sample_filter<int64_t>(
          inds_ptrs, query_ix, cluster_id, row, bitset_ptr, bitset_len, original_nbits);

      float dist      = 0.0f;
      float v_norm_sq = 0.0f;

      const uint8_t* group_data = codes + size_t(group) * padded_dim;

      for (uint32_t bl = 0; bl < n_dim_blocks; bl++) {
        code_vec_t codes_local;
        codes_local.load(group_data + bl * (veclen * kIndexGroupSize) + lane_id * veclen, 0);

        const uint32_t l = bl * veclen;
#pragma unroll
        for (uint32_t j = 0; j < veclen; j++) {
          if (l + j < dim) {
            accumulate_distance(s_query_term[l + j],
                                s_aux[l + j],
                                s_sq_scale[l + j],
                                codes_local.val.data[j],
                                dist,
                                v_norm_sq);
          }
        }
      }

      dist = finalize_distance(dist, v_norm_sq, q_norm);

      if constexpr (kManageLocalTopK) {
        float val = valid ? dist : local_topk_t::queue_t::kDummy;
        queue.add(val, sample_offset + row);
      } else {
        if (valid) {
          uint32_t out_idx       = query_ix * max_samples + sample_offset + row;
          out_distances[out_idx] = dist;
          out_indices[out_idx]   = sample_offset + row;
        }
      }
    }

    __syncthreads();  // (c)
    if constexpr (!kManageLocalTopK) break;
  }

  if constexpr (kManageLocalTopK) {
    // All probe iterations are done; smem_buf is reused for block_sort merge.
    // The loop's last (b) or (c) barrier ensures all prior smem accesses have
    // completed, so this additional barrier is only needed to synchronize any
    // register-level state across warps before the merge.
    __syncthreads();
    queue.done(smem_buf);
    queue.store(out_distances, out_indices);

    // block_sort initializes unused slots with (kDummy, idx=0). When the
    // probed clusters have fewer than k total valid vectors, those slots
    // survive into the output and share idx=0 with the real first vector,
    // causing duplicates. Mark them with an invalid index so
    // postprocess_neighbors treats them as out-of-bounds.
    // store() is a warp-0-only operation, restrict the fixup to the same warp.
    if (threadIdx.x < raft::WarpSize) {
      constexpr auto kDummyVal = local_topk_t::queue_t::kDummy;
      for (uint32_t i = threadIdx.x; i < k; i += raft::WarpSize) {
        if (out_distances[i] == kDummyVal) { out_indices[i] = uint32_t(0xFFFFFFFF); }
      }
    }
  }
}

}  // namespace cuvs::neighbors::ivf_sq::detail
