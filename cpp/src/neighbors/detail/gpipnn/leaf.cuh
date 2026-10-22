/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"
#include "partition.cuh"
#include "segmented_topk.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/count.h>
#include <thrust/fill.h>

#include <algorithm>
#include <vector>

namespace cuvs::neighbors::experimental::gpipnn::detail {

/**
 * Candidate edges produced by the leaf phase, in the exact form HashPrune consumes.
 *
 * The hash is deliberately not carried here: it is a pure function of (owner, target) via the
 * LSH sketches, so recomputing it inside the merge kernel is cheaper than widening every record
 * and dragging the extra bytes through the sort.
 */
struct edge_records {
  /** make_sort_key(owner, dist), so sorting groups by owner and orders by distance. */
  rmm::device_uvector<uint64_t> keys;
  rmm::device_uvector<id_t> targets;
  /** Number of leading entries that are real; the rest are kInvalidKey padding. */
  int64_t n_valid;

  explicit edge_records(cuda::stream_ref stream)
    : keys(0, stream), targets(0, stream), n_valid(0)
  {
  }
};

/**
 * Turns each selected leaf-mate into a pair of records, one per direction.
 *
 * Adding reverse edges makes selection asymmetric which more closely matches real-world dataset
 * conditions.
 */
RAFT_KERNEL emit_edge_records_kernel(const id_t* __restrict__ members,
                                     const id_t* __restrict__ sel_ids,
                                     const float* __restrict__ sel_dists,
                                     int64_t n_members,
                                     int64_t k,
                                     uint64_t* __restrict__ keys,
                                     id_t* __restrict__ targets)
{
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_members * k) { return; }

  const id_t owner  = members[i / k];
  const id_t target = sel_ids[i];
  const float dist  = sel_dists[i];

  const int64_t fwd = i * 2;
  const int64_t rev = fwd + 1;
  if (target == kInvalidId || target == owner) {
    keys[fwd]    = kInvalidKey;
    keys[rev]    = kInvalidKey;
    targets[fwd] = kInvalidId;
    targets[rev] = kInvalidId;
    return;
  }
  keys[fwd]    = make_sort_key(owner, dist);
  targets[fwd] = target;
  keys[rev]    = make_sort_key(target, dist);
  targets[rev] = owner;
}

/**
 * Computes the bi-directed 2-NN graph inside leaves `[leaf_begin, leaf_end)` using a dense GEMM.
 *
 * Each leaf is a self-join: its members are both the queries and the candidates, with the
 * diagonal masked. Leaves that were merged can list the same point twice, and masking by global
 * id rather than by position handles those duplicates for free.
 *
 * Restricting to a range of leaves is what bounds the build's peak memory, and since the order
 * of hashprune inputs is irrelevant, the result is the same.
 *
 * The leaves' members are contiguous in `leaves.members`, so the range needs no gather. Reverse
 * records do name owners outside the range, which is why the reservoir is global and merged into
 * rather than built per range.
 */
inline auto leaf_pick_range(raft::resources const& res,
                            raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
                            raft::device_vector_view<const float, int64_t> norms,
                            const runs& leaves,
                            int64_t k,
                            int64_t leaf_begin,
                            int64_t leaf_end) -> edge_records
{
  auto stream = raft::resource::get_cuda_stream(res).get();
  edge_records out(stream);
  if (leaf_begin >= leaf_end) { return out; }

  const int64_t base      = leaves.h_offsets[leaf_begin];
  const int64_t n_members = leaves.h_offsets[leaf_end] - base;
  if (n_members == 0) { return out; }

  // Chunk each leaf's queries so a single segment's operands and score matrix cannot exceed the
  // L2 cache budget. Chunks of a leaf all score against the whole leaf, so the leaf size is both
  // the candidate count and the chunk ceiling.
  const int64_t budget = l2_cache_bytes();
  const int64_t dim    = dataset.extent(1);
  std::vector<segment> segments;
  for (int64_t i = leaf_begin; i < leaf_end; i++) {
    const int64_t begin = leaves.h_offsets[i] - base;
    const int64_t size  = leaves.size_of(i);
    if (size < 2) { continue; }
    const int64_t max_chunk = std::min(max_query_chunk(budget, size, dim), size);
    for (int64_t q = 0; q < size; q += max_chunk) {
      segments.push_back(segment{begin + q, std::min(max_chunk, size - q), begin, size});
    }
  }

  rmm::device_uvector<id_t> sel_ids(n_members * k, stream);
  rmm::device_uvector<float> sel_dists(n_members * k, stream);
  thrust::fill(raft::resource::get_thrust_policy(res),
               sel_ids.data(),
               sel_ids.data() + sel_ids.size(),
               kInvalidId);

  const id_t* members = leaves.members.data() + base;
  auto members_view   = raft::make_device_vector_view<const id_t, int64_t>(members, n_members);
  segmented_topk(res,
                 dataset,
                 norms,
                 members_view,
                 members_view,
                 segments,
                 k,
                 true,
                 raft::make_device_matrix_view<id_t, int64_t>(sel_ids.data(), n_members, k),
                 raft::make_device_matrix_view<float, int64_t>(sel_dists.data(), n_members, k));

  const int64_t n_slots = n_members * k * 2;
  out.keys.resize(n_slots, stream);
  out.targets.resize(n_slots, stream);
  const int64_t threads = 256;
  const int64_t blocks  = (n_members * k + threads - 1) / threads;
  emit_edge_records_kernel<<<blocks, threads, 0, stream>>>(
    members, sel_ids.data(), sel_dists.data(), n_members, k, out.keys.data(), out.targets.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  const int64_t n_invalid = thrust::count(raft::resource::get_thrust_policy(res),
                                          out.keys.data(),
                                          out.keys.data() + n_slots,
                                          kInvalidKey);
  out.n_valid             = n_slots - n_invalid;
  return out;
}

/** Whole-partition leaf join, as the tests and the standalone diagnostics drive it. */
inline auto leaf_pick(raft::resources const& res,
                      raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
                      raft::device_vector_view<const float, int64_t> norms,
                      const runs& leaves,
                      int64_t k) -> edge_records
{
  return leaf_pick_range(res, dataset, norms, leaves, k, 0, leaves.count());
}

}  // namespace cuvs::neighbors::experimental::gpipnn::detail
