/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"
#include "leaf.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include <thrust/fill.h>

#include <utility>

namespace cuvs::neighbors::experimental::gpipnn::detail {

/**
 * The bounded per-point candidate reservoir HashPrune fills.
 *
 * Every row is ascending by distance and free of duplicate targets, which is what lets
 * DirectEdgePrune treat it as an ordered candidate list.
 *
 * `hashes` carries the LSH bucket of each held entry, so a later batch can seed its bucket bitmap
 * without recomputing `h(p, t)`. `release_hashes` drops it before DirectEdgePrune runs.
 */
struct accumulator {
  rmm::device_uvector<id_t> ids;
  rmm::device_uvector<float> dists;
  rmm::device_uvector<uint16_t> hashes;
  rmm::device_uvector<uint32_t> counts;
  int64_t l_max;

  accumulator(cuda::stream_ref stream, int64_t n_rows, int64_t l)
    : ids(n_rows * l, stream),
      dists(n_rows * l, stream),
      hashes(n_rows * l, stream),
      counts(n_rows, stream),
      l_max(l)
  {
  }

  void release_hashes(cuda::stream_ref stream)
  {
    hashes.resize(0, stream);
    hashes.shrink_to_fit(stream);
  }
};

/** Allocates an accumulator in which no row holds anything and every slot reads as invalid. */
inline auto make_accumulator(raft::resources const& res, int64_t n_rows, int64_t l_max)
  -> accumulator
{
  auto stream = raft::resource::get_cuda_stream(res).get();
  accumulator acc(stream, n_rows, l_max);
  RAFT_CUDA_TRY(cudaMemsetAsync(acc.counts.data(), 0, n_rows * sizeof(uint32_t), stream));
  thrust::fill(raft::resource::get_thrust_policy(res),
               acc.ids.data(),
               acc.ids.data() + acc.ids.size(),
               kInvalidId);
  return acc;
}

/**
 * Locates each owner's contiguous block in the distance-sorted record array.
 *
 * A binary search rather than a run-length encode because owners that received no records still
 * need a row in the accumulator, and an RLE would silently omit them.
 */
RAFT_KERNEL find_owner_bounds_kernel(const uint64_t* __restrict__ keys,
                                     int64_t n_keys,
                                     int64_t n_rows,
                                     int64_t* __restrict__ row_begin)
{
  const int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (p > n_rows) { return; }
  const uint64_t target = static_cast<uint64_t>(p) << 32;
  int64_t lo = 0, hi = n_keys;
  while (lo < hi) {
    const int64_t mid = lo + (hi - lo) / 2;
    if (keys[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  row_begin[p] = lo;
}

/**
 * Merges one batch of an owner's distance-sorted candidate run into its bounded reservoir,
 * one warp per row.
 *
 * Independent of traversal order, what HashPrune computes is
 *
 *   reservoir(p) = the l_max smallest, by distance, of the per-bucket nearest candidates
 *
 * a pure function of the *set* of records owned by p. The input arrives ascending by distance so
 * the first arrivals to the buckets are assigned, collisions can be skipped, and overflows are
 * simply truncated at capacity.
 *
 * The indepednence of traversal order is what enables batching. Candidates in the reservior
 * just merge back in at the low end of the input stream.
 * Truncating to l_max each time loses nothing since dropping a candidate due to this means there
 * are better candidates present.
 *
 * Ties resolve to the held entry (`<=` in the take-held test), and held entries come from earlier
 * batches, making the result independent of batch size.i
 *
 * There is no separate duplicate-target scan: `h` is a pure function of `(p, t)`, so a repeated
 * target already has its bucket bit set and the hash test rejects it.
 *
 * Both streams are read through a 32-wide warp window. Indexing `acc_dists[row + j]` and friends
 * directly has every lane issue the same address, fetching a 32-byte sector to deliver four or
 * eight bytes, on the loop's dependency chain. The kernel is latency bound regardless -- one
 * 32-thread block per row against a 24-block-per-SM cap -- so loading 32 consecutive entries
 * coalesced and reading entry `j` back with `__shfl_sync` buys a small win.
 */
RAFT_KERNEL hash_prune_merge_kernel(const uint64_t* __restrict__ keys,
                                    const id_t* __restrict__ targets,
                                    const int64_t* __restrict__ row_begin,
                                    const float* __restrict__ sketches,
                                    int64_t n_rows,
                                    int32_t hash_bits,
                                    int64_t l_max,
                                    id_t* __restrict__ acc_ids,
                                    float* __restrict__ acc_dists,
                                    uint16_t* __restrict__ acc_hashes,
                                    uint32_t* __restrict__ acc_counts)
{
  const int64_t p = blockIdx.x;
  if (p >= n_rows) { return; }
  const int32_t lane = static_cast<int32_t>(threadIdx.x);

  const int64_t begin = row_begin[p];
  const int64_t end   = row_begin[p + 1];
  if (begin >= end) { return; }  // this batch brought nothing for this row

  const int64_t row  = p * l_max;
  const int32_t held = static_cast<int32_t>(acc_counts[p]);

  // The inputs are ascending, so if the fartest existing entry is nearer than the batch's nearest
  // candidate, there is no need to merge. This keeps the per-batch cost proportional to the
  // number of records received rather than to the reservoir depth and the batch count.
  if (held >= l_max && !(key_dist(keys[begin]) < acc_dists[row + l_max - 1])) { return; }

  extern __shared__ uint32_t smem[];
  const int32_t n_words = ((1 << hash_bits) + 31) / 32;
  uint32_t* hash_seen   = smem;  // one bit per hash value
  auto* out_ids         = reinterpret_cast<id_t*>(smem + n_words);
  auto* out_dists       = reinterpret_cast<float*>(out_ids + l_max);
  auto* out_hashes      = reinterpret_cast<uint16_t*>(out_dists + l_max);

  for (int32_t i = lane; i < n_words; i += warpSize) {
    hash_seen[i] = 0u;
  }
  __syncwarp();

  const float* p_sketch = sketches + p * hash_bits;
  int32_t count         = 0;
  int32_t j             = 0;      // cursor over the entries the row already holds
  int64_t i             = begin;  // cursor over this batch's run

  // The loaded window of each stream: lane `x` holds the entry at `base + x`. Every read below
  // is guarded by the same bound the unwindowed loads had -- `k < held` against a reservoir of
  // `l_max`, and `k < end` against `end <= records.n_valid` -- so the windows touch nothing the
  // element-at-a-time form did not.
  int32_t held_base  = -1;
  float held_dist    = 0.0f;
  uint32_t held_hash = 0u;
  id_t held_id       = 0u;
  int64_t rec_base   = -1;
  uint64_t rec_key   = 0u;
  id_t rec_target    = 0u;

  while (count < l_max && (j < held || i < end)) {
    if (j < held) {
      const int32_t base = j & ~31;
      if (base != held_base) {
        held_base       = base;
        const int32_t k = base + lane;
        const bool in   = k < held;
        held_dist       = in ? acc_dists[row + k] : 0.0f;
        held_hash       = in ? acc_hashes[row + k] : 0u;
        held_id         = in ? acc_ids[row + k] : kInvalidId;
      }
    }
    if (i < end) {
      const int64_t base = (i - begin) & ~int64_t{31};
      if (base != rec_base) {
        rec_base        = base;
        const int64_t k = begin + base + lane;
        const bool in   = k < end;
        rec_key         = in ? keys[k] : kInvalidKey;
        rec_target      = in ? targets[k] : kInvalidId;
      }
    }
    const float cur_dist = j < held ? __shfl_sync(0xffffffffu, held_dist, j & 31) : 0.0f;
    const uint64_t cur_key =
      i < end ? __shfl_sync(0xffffffffu, rec_key, static_cast<int>((i - begin) & 31)) : 0u;

    const bool take_held = (i >= end) || (j < held && cur_dist <= key_dist(cur_key));

    if (take_held) {
      const uint32_t hash = __shfl_sync(0xffffffffu, held_hash, j & 31);
      const id_t hid      = __shfl_sync(0xffffffffu, held_id, j & 31);
      const uint32_t word = hash >> 5;
      const uint32_t mask = 1u << (hash & 31u);
      j++;
      // First-wins applies to what the row already holds exactly as it does to a new record: a
      // nearer record in the same bucket displaces the held entry rather than joining it. The
      // number of distinct buckets is unchanged by such a swap, so a row's fill still only
      // grows.
      if ((hash_seen[word] & mask) != 0u) { continue; }
      if (lane == 0) {
        hash_seen[word]   = hash_seen[word] | mask;
        out_ids[count]    = hid;
        out_dists[count]  = cur_dist;
        out_hashes[count] = static_cast<uint16_t>(hash);
      }
      __syncwarp();
      count++;
      continue;
    }

    const id_t t  = __shfl_sync(0xffffffffu, rec_target, static_cast<int>((i - begin) & 31));
    const float d = key_dist(cur_key);
    i++;
    if (t == static_cast<id_t>(p)) { continue; }

    // h_p(t): the sign pattern of the residual sketch. Lane j owns bit j.
    const float* t_sketch = sketches + static_cast<int64_t>(t) * hash_bits;
    const bool bit        = lane < hash_bits && (t_sketch[lane] - p_sketch[lane]) > 0.0f;
    const uint32_t hash   = __ballot_sync(0xffffffffu, bit) & ((1u << hash_bits) - 1u);

    const uint32_t word = hash >> 5;
    const uint32_t mask = 1u << (hash & 31u);
    if ((hash_seen[word] & mask) != 0u) { continue; }

    if (lane == 0) {
      hash_seen[word]   = hash_seen[word] | mask;
      out_ids[count]    = t;
      out_dists[count]  = d;
      out_hashes[count] = static_cast<uint16_t>(hash);
    }
    __syncwarp();
    count++;
  }

  for (int32_t x = lane; x < count; x += warpSize) {
    acc_ids[row + x]    = out_ids[x];
    acc_dists[row + x]  = out_dists[x];
    acc_hashes[row + x] = out_hashes[x];
  }
  if (lane == 0) { acc_counts[p] = static_cast<uint32_t>(count); }
}

/** Shared memory one block of `hash_prune_merge_kernel` needs. */
inline auto hash_prune_merge_smem(int32_t hash_bits, int64_t l_max) -> size_t
{
  const size_t bitmap = static_cast<size_t>(((1 << hash_bits) + 31) / 32) * sizeof(uint32_t);
  const size_t staged =
    static_cast<size_t>(l_max) * (sizeof(id_t) + sizeof(float) + sizeof(uint16_t));
  return bitmap + staged;
}

/**
 * Projects the dataset onto `hash_bits` random hyperplanes.
 *
 * HashPrune uses the sign pattern of the difference of two sketches as a locality-sensitive
 * hash of the direction from a point to a candidate, so that candidates clustered in the same
 * direction collapse to one entry and the reservoir spreads over the sphere.
 */
inline auto compute_sketches(raft::resources const& res,
                             raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
                             int32_t hash_bits,
                             uint64_t seed) -> rmm::device_uvector<float>
{
  auto stream          = raft::resource::get_cuda_stream(res).get();
  const int64_t n_rows = dataset.extent(0);
  const int64_t dim    = dataset.extent(1);

  rmm::device_uvector<float> planes_f(hash_bits * dim, stream);
  raft::random::RngState rng(seed);
  raft::random::normal(res, rng, planes_f.data(), planes_f.size(), 0.0f, 1.0f);

  rmm::device_uvector<half> planes(hash_bits * dim, stream);
  const int64_t threads = 256;
  cast_to_half_kernel<<<(planes_f.size() + threads - 1) / threads, threads, 0, stream>>>(
    planes_f.data(), planes.data(), static_cast<int64_t>(planes_f.size()));
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  rmm::device_uvector<float> sketches(n_rows * hash_bits, stream);
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  auto cublas_h     = raft::resource::get_cublas_handle(res);
  RAFT_CUBLAS_TRY(cublasSetStream(cublas_h, stream));
  RAFT_CUBLAS_TRY(cublasGemmEx(cublas_h,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               hash_bits,
                               static_cast<int>(n_rows),
                               static_cast<int>(dim),
                               &alpha,
                               planes.data(),
                               CUDA_R_16F,
                               static_cast<int>(dim),
                               dataset.data_handle(),
                               CUDA_R_16F,
                               static_cast<int>(dim),
                               &beta,
                               sketches.data(),
                               CUDA_R_32F,
                               hash_bits,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT));
  return sketches;
}

/**
 * Highest key bit that can distinguish two records, given the owner range.
 *
 * Owners are below `n_rows`, so every key's bits above the owner's width are zero -- except
 * `kInvalidKey`, whose owner field then reads as the all-ones value that no real owner can
 * take. Sorting only the bits below this point therefore still leaves the padding after every
 * real record, and saves a radix pass or two over an unconditional 64.
 */
inline auto sort_key_bits(int64_t n_rows) -> int
{
  int owner_bits = 0;
  while (owner_bits < 32 && (int64_t{1} << owner_bits) <= n_rows) {
    owner_bits++;
  }
  return 32 + owner_bits;
}

/**
 * Sorts one batch of candidate edge records by (owner, distance) and merges each owner's run
 * into its bounded, deduplicated, distance-ordered reservoir.
 *
 * The batch may be the whole record array or any contiguous slice of it; see
 * `hash_prune_merge_kernel` for why the answer is the same either way.
 *
 * `records` is sorted in place with `cub::DoubleBuffer` rather than the plain overload, which
 * hides a third copy of the largest array the build allocates inside its temporary storage. The
 * ping-pong can finish in either buffer, so the winner is moved back into `records`, keeping the
 * routine idempotent.
 *
 * `row_begin` is caller-owned scratch of `n_rows + 1` entries, so a batch loop allocates it once.
 */
inline void hash_prune_merge(raft::resources const& res,
                             raft::device_vector_view<const float, int64_t> sketches,
                             edge_records& records,
                             const index_params& params,
                             rmm::device_uvector<int64_t>& row_begin,
                             accumulator& acc)
{
  auto stream           = raft::resource::get_cuda_stream(res).get();
  const int64_t n_rows  = static_cast<int64_t>(row_begin.size()) - 1;
  const int64_t l_max   = acc.l_max;
  const auto hash_bits  = static_cast<int32_t>(params.hash_bits);
  const int64_t n_total = static_cast<int64_t>(records.keys.size());
  if (n_total == 0) { return; }

  rmm::device_uvector<uint64_t> keys_alt(n_total, stream);
  rmm::device_uvector<id_t> targets_alt(n_total, stream);
  cub::DoubleBuffer<uint64_t> d_keys(records.keys.data(), keys_alt.data());
  cub::DoubleBuffer<id_t> d_targets(records.targets.data(), targets_alt.data());

  const int end_bit = sort_key_bits(n_rows);
  size_t bytes      = 0;
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
    nullptr, bytes, d_keys, d_targets, n_total, 0, end_bit, stream));
  rmm::device_uvector<char> workspace(bytes, stream);
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
    workspace.data(), bytes, d_keys, d_targets, n_total, 0, end_bit, stream));
  if (d_keys.Current() != records.keys.data()) { records.keys = std::move(keys_alt); }
  if (d_targets.Current() != records.targets.data()) { records.targets = std::move(targets_alt); }

  const int64_t threads = 128;
  find_owner_bounds_kernel<<<(n_rows + threads) / threads, threads, 0, stream>>>(
    records.keys.data(), records.n_valid, n_rows, row_begin.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  const size_t smem = hash_prune_merge_smem(hash_bits, l_max);
  hash_prune_merge_kernel<<<n_rows, 32, smem, stream>>>(records.keys.data(),
                                                        records.targets.data(),
                                                        row_begin.data(),
                                                        sketches.data_handle(),
                                                        n_rows,
                                                        hash_bits,
                                                        l_max,
                                                        acc.ids.data(),
                                                        acc.dists.data(),
                                                        acc.hashes.data(),
                                                        acc.counts.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Whole-array HashPrune: sketch, allocate, merge once.
 *
 * `build_graph` runs the batched form instead; this is the shape the tests drive, and merging a
 * single batch is by construction the same computation.
 */
inline auto hash_prune(raft::resources const& res,
                       raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
                       edge_records& records,
                       const index_params& params) -> accumulator
{
  auto stream          = raft::resource::get_cuda_stream(res).get();
  const int64_t n_rows = dataset.extent(0);
  const auto hash_bits = static_cast<int32_t>(params.hash_bits);

  auto acc      = make_accumulator(res, n_rows, params.max_candidates);
  auto sketches = compute_sketches(res, dataset, hash_bits, params.seed + 7);
  rmm::device_uvector<int64_t> row_begin(n_rows + 1, stream);
  hash_prune_merge(
    res,
    raft::make_device_vector_view<const float, int64_t>(sketches.data(), n_rows * hash_bits),
    records,
    params,
    row_begin,
    acc);
  acc.release_hashes(stream);
  return acc;
}

}  // namespace cuvs::neighbors::experimental::gpipnn::detail
