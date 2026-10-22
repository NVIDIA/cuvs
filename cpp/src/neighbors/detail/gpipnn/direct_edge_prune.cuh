/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"
#include "hash_prune.cuh"
#include "partition.cuh"
#include "segmented_topk.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

namespace cuvs::neighbors::experimental::gpipnn::detail {

/**
 * Factor by which the occlusion threshold grows between rounds of the alpha ramp.
 */
inline constexpr float kAlphaRampStep = 1.2f;

/**
 * Occlusion factor of candidate `q` seen from a row owner that has just accepted `p_star`:
 * d(owner, q) / d(p_star, q).
 */
__device__ inline auto occlude_factor(float d_owner_q, float d_star_q) -> float
{
  return d_star_q == 0.0f ? __builtin_huge_valf() : d_owner_q / d_star_q;
}

/**
 * Sparsifies one accumulator row into a graph row using the alpha-occlusion rule.
 *
 * The rule is the alpha-RNG test of the reference implementation: candidate q is dropped once some
 * already-accepted neighbour p* satisfies d(owner, q) > cur_alpha * d(p*, q). Both distances are
 * *squared* L2 and `alpha` is applied unscaled (per the paper, scaling the true alpha to ~1.095).
 *
 * The distinguishing property of DirectEdgePrune (GPiPNN Sec 5.4) is that it recomputes no
 * distances: a candidate can only be occluded by an accepted neighbour that already lists it in
 * its own reservoir. Missing witnesses do not occlude, making the graph not exactly match a true
 * RobustPrune due to being potentially denser.

 * Rather than rescan an accepted neighbour's row per candidate, each acceptance does one parallel
 * pass over that row and probes a shared hash table of this row's candidates, turning the cost
 * per row from cubic in the reservoir size into linear.
 *
 * That scan is where the time goes, which is why `slots` and `row_order` are parameters rather than derived here:
 *
 *   `slots` is a power of two so the wrap is a mask rather than a division on the dependency
 *   chain between the id load and the probe. The table maps candidate id to position in the
 *   owner's row and ids are unique, so the slot count cannot change what a lookup returns, only
 *   how long the probe chain is. That is what lets the host size it for occupancy.
 *
 *   `row_order`, when given, is the order blocks take rows in. A row's accepted neighbours are
 *   its own near neighbours, so taking rows in leaf order rather than id order shrinks the set of
 *   accumulator rows the resident blocks touch to something L2 holds. Every row is still handled
 *   by exactly one block writing only its own graph row, so the permutation cannot change the
 *   graph.
 */
RAFT_KERNEL direct_edge_prune_kernel(const id_t* __restrict__ acc_ids,
                                     const float* __restrict__ acc_dists,
                                     const uint32_t* __restrict__ acc_counts,
                                     const id_t* __restrict__ row_order,
                                     int64_t n_rows,
                                     int64_t l_max,
                                     int32_t slots,
                                     int64_t degree,
                                     float alpha,
                                     id_t* __restrict__ graph)
{
  if (blockIdx.x >= static_cast<uint64_t>(n_rows)) { return; }
  const int64_t p    = row_order == nullptr ? static_cast<int64_t>(blockIdx.x)
                                            : static_cast<int64_t>(row_order[blockIdx.x]);
  const int32_t tid  = static_cast<int32_t>(threadIdx.x);
  const int32_t nthr = static_cast<int32_t>(blockDim.x);

  const int32_t c = static_cast<int32_t>(acc_counts[p]);
  // A power of two, so `& slot_mask` is `% slots`. `direct_edge_prune_slots` guarantees it.
  const uint32_t slot_mask = static_cast<uint32_t>(slots) - 1u;

  // The byte-wide array goes last so every 4-byte array stays naturally aligned.
  extern __shared__ char de_smem[];
  id_t* cand_ids     = reinterpret_cast<id_t*>(de_smem);
  float* cand_dists  = reinterpret_cast<float*>(cand_ids + l_max);
  float* occ         = reinterpret_cast<float*>(cand_dists + l_max);
  id_t* table_id     = reinterpret_cast<id_t*>(occ + l_max);
  int32_t* table_idx = reinterpret_cast<int32_t*>(table_id + slots);
  uint8_t* accepted  = reinterpret_cast<uint8_t*>(table_idx + slots);

  for (int32_t i = tid; i < c; i += nthr) {
    cand_ids[i]   = acc_ids[p * l_max + i];
    cand_dists[i] = acc_dists[p * l_max + i];
    occ[i]        = 0.0f;
    accepted[i]   = 0;
  }
  for (int32_t i = tid; i < slots; i += nthr) {
    table_id[i] = kInvalidId;
  }
  __syncthreads();

  // Open-addressed map from candidate id to its position in this row. Ids are unique because
  // HashPrune already deduplicated them, so no slot is ever contested by the same key.
  for (int32_t i = tid; i < c; i += nthr) {
    const id_t key = cand_ids[i];
    int32_t slot   = static_cast<int32_t>((key * 2654435761u) & slot_mask);
    while (atomicCAS(&table_id[slot], kInvalidId, key) != kInvalidId) {
      slot = static_cast<int32_t>(static_cast<uint32_t>(slot + 1) & slot_mask);
    }
    table_idx[slot] = i;
  }
  __syncthreads();

  __shared__ int32_t n_accepted;
  if (tid == 0) { n_accepted = 0; }
  __syncthreads();

  // The ramp: strictest threshold first, so the diverse edges get their slots before the
  // near-duplicates do.
  for (float cur_alpha = 1.0f; cur_alpha <= alpha && n_accepted < degree;
       cur_alpha *= kAlphaRampStep) {
    for (int32_t i = 0; i < c && n_accepted < degree; i++) {
      // Occluded at this threshold, or already accepted in an earlier round.
      if (occ[i] > cur_alpha) { continue; }
      const id_t p_star = cand_ids[i];
      if (tid == 0) {
        graph[p * degree + n_accepted] = p_star;
        n_accepted++;
        accepted[i] = 1;
      }
      __syncthreads();
      if (n_accepted >= degree) { break; }

      // Everything p* witnesses that this row also holds and has not yet reached is a
      // candidate for occlusion.
      const int32_t c_star = static_cast<int32_t>(acc_counts[p_star]);
      for (int32_t j = tid; j < c_star; j += nthr) {
        const id_t q         = acc_ids[static_cast<int64_t>(p_star) * l_max + j];
        const float d_star_q = acc_dists[static_cast<int64_t>(p_star) * l_max + j];
        int32_t slot         = static_cast<int32_t>((q * 2654435761u) & slot_mask);
        int32_t probes       = 0;
        while (probes < slots && table_id[slot] != kInvalidId) {
          if (table_id[slot] == q) {
            const int32_t li = table_idx[slot];
            // Only candidates not yet reached can still be occluded. p*'s row holds each id
            // once, so no two threads reach the same `li` and the maximum needs no atomic.
            if (li > i && occ[li] <= alpha) {
              occ[li] = fmaxf(occ[li], occlude_factor(cand_dists[li], d_star_q));
            }
            break;
          }
          slot = static_cast<int32_t>(static_cast<uint32_t>(slot + 1) & slot_mask);
          probes++;
        }
      }
      // Bars re-acceptance by a later ramp round without disturbing the witnessed factors.
      if (tid == 0) { occ[i] = __builtin_huge_valf(); }
      __syncthreads();
    }
  }

  // Fill under-full accepted lists with previously occluded candidates in ascending order.
  if (tid == 0) {
    for (int32_t i = 0; i < c && n_accepted < degree; i++) {
      if (accepted[i] != 0) { continue; }
      graph[p * degree + n_accepted] = cand_ids[i];
      n_accepted++;
    }
  }
  __syncthreads();

  // CAGRA requires every slot to hold a valid id, so whatever is still empty gets a
  // deterministic ring entry.
  for (int32_t i = n_accepted + tid; i < degree; i += nthr) {
    graph[p * degree + i] = static_cast<id_t>((p + i + 1) % n_rows);
  }
}

/** Shared memory that each block of `direct_edge_prune_kernel` needs. */
inline auto direct_edge_prune_smem(int64_t l_max, int64_t slots) -> size_t
{
  const size_t per_candidate = sizeof(id_t) + 2 * sizeof(float) + sizeof(uint8_t);
  return static_cast<size_t>(l_max) * per_candidate +
         static_cast<size_t>(slots) * (sizeof(id_t) + sizeof(int32_t));
}

/**
 * Slot count for the candidate-position map: a power of two, and as loosely loaded as occupancy
 * allows.
 *
 * The count cannot change the answer since its size only sets the probe chain length. It must be a power of two
 * so the kernel can wrap with a mask instead of a division.
 *
 * Bigger is better until it costs occupancy. The larger table is taken only when the occupancy calculator agrees.
 */
inline auto direct_edge_prune_slots(int64_t l_max) -> int64_t
{
  int64_t small = 1;
  while (small < 2 * l_max) {
    small <<= 1;
  }
  const int64_t large = 2 * small;
  int b_small = 0, b_large = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &b_small, direct_edge_prune_kernel, 128, direct_edge_prune_smem(l_max, small)));
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &b_large, direct_edge_prune_kernel, 128, direct_edge_prune_smem(l_max, large)));
  return b_large >= b_small ? large : small;
}

/** Fills `label` with a value that sorts last and `ids` with the identity. */
RAFT_KERNEL init_prune_order_kernel(uint32_t* __restrict__ label,
                                    id_t* __restrict__ ids,
                                    int64_t n_rows,
                                    uint32_t unlabelled)
{
  const int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (p >= n_rows) { return; }
  label[p] = unlabelled;
  ids[p]   = static_cast<id_t>(p);
}

/** Lowest-numbered leaf each point belongs to, one block per leaf. */
RAFT_KERNEL first_leaf_kernel(const id_t* __restrict__ members,
                              const int64_t* __restrict__ offsets,
                              int64_t n_leaves,
                              uint32_t* __restrict__ label)
{
  const int64_t leaf = blockIdx.x;
  if (leaf >= n_leaves) { return; }
  const int64_t end = offsets[leaf + 1];
  for (int64_t i = offsets[leaf] + threadIdx.x; i < end; i += blockDim.x) {
    atomicMin(&label[members[i]], static_cast<uint32_t>(leaf));
  }
}

/**
 * Order in which `direct_edge_prune`'s blocks should take rows, grouped by leaf.
 *
 * `direct_edge_prune` gathers `acc_ids[p_star * l_max + j]` for each accepted neighbour. 
 * In id order the resident blocks
 * touch rows spread over the whole array and almost nothing hits L2. Points sharing a leaf are
 * near each other, and a point's accepted neighbours are near it, so under this order concurrent
 * blocks gather from overlapping rows.
 *
 * A point belongs to `fanout_level0 * fanout_level1` leaves; the lowest-numbered one groups it
 * with its neighbours and needs no tie-breaking.
 *
 * Must be called while the partition is still alive, i.e. before `leaves.members` is released.
 */
inline auto build_prune_order(raft::resources const& res, const runs& leaves, int64_t n_rows)
  -> rmm::device_uvector<id_t>
{
  auto stream = raft::resource::get_cuda_stream(res).get();
  rmm::device_uvector<id_t> order(n_rows, stream);
  const int64_t n_leaves = leaves.count();
  if (n_leaves <= 0) {
    order.resize(0, stream);
    return order;
  }

  // Points with no leaf keep `n_leaves` and so sort last, which also caps the key width.
  const auto unlabelled = static_cast<uint32_t>(n_leaves);
  int key_bits          = 0;
  while (key_bits < 32 && (uint32_t{1} << key_bits) <= unlabelled) {
    key_bits++;
  }

  rmm::device_uvector<uint32_t> label(n_rows, stream);
  rmm::device_uvector<id_t> ids(n_rows, stream);
  const int64_t threads = 256;
  init_prune_order_kernel<<<(n_rows + threads - 1) / threads, threads, 0, stream>>>(
    label.data(), ids.data(), n_rows, unlabelled);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  first_leaf_kernel<<<n_leaves, 256, 0, stream>>>(
    leaves.members.data(), leaves.offsets.data(), n_leaves, label.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  rmm::device_uvector<uint32_t> label_out(n_rows, stream);
  size_t bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                bytes,
                                                label.data(),
                                                label_out.data(),
                                                ids.data(),
                                                order.data(),
                                                n_rows,
                                                0,
                                                key_bits,
                                                stream));
  rmm::device_uvector<char> workspace(bytes, stream);
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(workspace.data(),
                                                bytes,
                                                label.data(),
                                                label_out.data(),
                                                ids.data(),
                                                order.data(),
                                                n_rows,
                                                0,
                                                key_bits,
                                                stream));
  return order;
}

/**
 * Dispatches the direct_edge_prune kernel.
 */
inline void direct_edge_prune(
  raft::resources const& res,
  const accumulator& acc,
  const index_params& params,
  raft::device_matrix_view<id_t, int64_t, raft::row_major> graph,
  const id_t* row_order = nullptr)
{
  auto stream          = raft::resource::get_cuda_stream(res).get();
  const int64_t n_rows = graph.extent(0);
  const int64_t degree = graph.extent(1);
  const int64_t l_max  = acc.l_max;

  const int64_t slots = direct_edge_prune_slots(l_max);
  const size_t smem   = direct_edge_prune_smem(l_max, slots);
  direct_edge_prune_kernel<<<n_rows, 128, smem, stream>>>(acc.ids.data(),
                                                          acc.dists.data(),
                                                          acc.counts.data(),
                                                          row_order,
                                                          n_rows,
                                                          l_max,
                                                          static_cast<int32_t>(slots),
                                                          degree,
                                                          params.alpha,
                                                          graph.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::neighbors::experimental::gpipnn::detail
