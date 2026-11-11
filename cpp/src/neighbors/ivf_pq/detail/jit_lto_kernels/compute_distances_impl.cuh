/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <cub/block/block_reduce.cuh>

#include "block_sort.cuh"
#include "device_functions.cuh"

namespace cuvs::neighbors::ivf_pq::detail {

template <typename OutT, typename LutT, int Capacity>
__device__ void compute_distances_impl(const uint32_t* chunk_indices,
                                       float* query_kths,
                                       uint32_t n_probes,
                                       uint32_t probe_ix,
                                       uint32_t query_ix,
                                       uint32_t max_samples,
                                       uint32_t topk,
                                       uint8_t* lut_end,
                                       uint32_t pq_dim,
                                       const uint8_t* const* pq_dataset,
                                       uint32_t label,
                                       uint32_t queries_offset,
                                       OutT* out_scores,
                                       uint32_t* out_indices,
                                       LutT* lut_scores,
                                       uint8_t* smem_buf,
                                       const int64_t* const* inds_ptrs,
                                       uint32_t* bitset_ptr,
                                       int64_t bitset_len,
                                       int64_t original_nbits,
                                       uint32_t force_n_probes,
                                       OutT forced_score)
{
  constexpr bool kManageLocalTopK = Capacity > 0;

  // Define helper types for efficient access to the pq_dataset, which is stored in an interleaved
  // format. The chunks of PQ data are stored in kIndexGroupVecLen-bytes-long chunks, interleaved
  // in groups of kIndexGroupSize elems (which is normally equal to the warp size) for the fastest
  // possible access by thread warps.
  //
  // Consider one record in the pq_dataset is `pq_dim * pq_bits`-bit-long.
  // Assuming `kIndexGroupVecLen = 16`, one chunk of data read by a thread at once is 128-bits.
  // Then, such a chunk contains `chunk_size = 128 / pq_bits` record elements, and the record
  // consists of `ceildiv(pq_dim, chunk_size)` chunks. The chunks are interleaved in groups of 32,
  // so that the warp can achieve the best coalesced read throughput.
  using group_align  = raft::Pow2<kIndexGroupSize>;
  using vec_align    = raft::Pow2<kIndexGroupVecLen>;
  using local_topk_t = block_sort_t<Capacity, OutT, uint32_t>;

  uint32_t sample_offset = 0;
  if (probe_ix > 0) { sample_offset = chunk_indices[probe_ix - 1]; }
  uint32_t n_samples         = chunk_indices[probe_ix] - sample_offset;
  uint32_t n_samples_aligned = group_align::roundUp(n_samples);
  uint32_t pq_line_width     = get_line_width(pq_dim);
  auto pq_thread_data = pq_dataset[label] + group_align::roundDown(threadIdx.x) * pq_line_width +
                        group_align::mod(threadIdx.x) * vec_align::Value;
  pq_line_width *= blockDim.x;

  constexpr OutT kDummy = raft::upper_bound<OutT>();
  OutT query_kth        = kDummy;
  if constexpr (kManageLocalTopK) { query_kth = OutT(query_kths[query_ix]); }
  OutT early_stop_limit = get_early_stop_limit(query_kth);

  // Ensure lut_scores is written by all threads before using it in ivfpq-compute-score
  __threadfence_block();
  __syncthreads();
  local_topk_t block_topk(topk, lut_end, query_kth);

  // The probes of each query are forced in a cyclic window starting at a pseudo-random probe.
  bool force_probe = probe_ix < force_n_probes;
  if (force_n_probes > 0) {
    uint32_t h  = queries_offset + query_ix;
    h           = (h ^ (h >> 16)) * 0x45d9f3bu;
    h           = (h ^ (h >> 16)) * 0x45d9f3bu;
    h           = h ^ (h >> 16);
    force_probe = (probe_ix + n_probes - h % n_probes) % n_probes < force_n_probes;
  }
  // The best sample of the probe seen by this thread (used when forcing the probe)
  cub::KeyValuePair<uint32_t, OutT> best{0, kDummy};

  // Compute a distance for each sample
  for (uint32_t i = threadIdx.x; i < n_samples_aligned;
       i += blockDim.x, pq_thread_data += pq_line_width) {
    OutT score = kDummy;
    bool valid = i < n_samples;
    // Check bounds and that the sample is acceptable for the query
    if (valid &&
        sample_filter(
          inds_ptrs, queries_offset + query_ix, label, i, bitset_ptr, bitset_len, original_nbits)) {
      score = compute_score<OutT, LutT>(
        pq_dim, reinterpret_cast<const vec_t::io_t*>(pq_thread_data), lut_scores, early_stop_limit);
      score = increment_score<OutT>(score);
      if (force_probe && score < best.value) { best = {i, score}; }
    }
    if constexpr (kManageLocalTopK) {
      block_topk.add(score, sample_offset + i);
    } else {
      if (valid) { out_scores[sample_offset + i] = score; }
    }
  }
  if constexpr (kManageLocalTopK) {
    // sync threads before the topk merging operation, because we reuse smem_buf
    __syncthreads();
    block_topk.done(smem_buf);
    block_topk.store(out_scores, out_indices);
    if (threadIdx.x == 0) { atomicMin(query_kths + query_ix, float(out_scores[topk - 1])); }
  } else {
    // fill in the rest of the out_scores with dummy values
    if (probe_ix + 1 == n_probes) {
      for (uint32_t i = threadIdx.x + sample_offset + n_samples; i < max_samples; i += blockDim.x) {
        out_scores[i] = kDummy;
      }
    }
  }

  if (force_probe) {
    // Make sure the best sample of this probe is selected by the final top-k.
    using block_reduce_t = cub::BlockReduce<cub::KeyValuePair<uint32_t, OutT>, 1024>;
    __shared__ typename block_reduce_t::TempStorage reduce_storage;
    best =
      block_reduce_t(reduce_storage)
        .Reduce(
          best, [](const auto& a, const auto& b) { return b.value < a.value ? b : a; }, blockDim.x);
    if (threadIdx.x == 0) {
      // The head of the queue is the best sample of the probe. If there is no queue, or the queue
      // is empty, use the best sample found above.
      bool in_queue = false;
      if constexpr (kManageLocalTopK) { in_queue = out_scores[0] < kDummy; }
      if (in_queue || best.value < kDummy) {
        // The lowest possible score guarantees the sample is selected
        if constexpr (kManageLocalTopK) {
          out_scores[0] = forced_score;
          if (!in_queue) { out_indices[0] = sample_offset + best.key; }
        } else {
          out_scores[sample_offset + best.key] = forced_score;
        }
      }
    }
  }
}

}  // namespace cuvs::neighbors::ivf_pq::detail
