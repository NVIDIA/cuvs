/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"
#include "segmented_topk.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace cuvs::neighbors::experimental::gpipnn::detail {

/**
 * A set of point runs: `members` holds point ids grouped by run, and run `r` occupies
 * `members[offsets[r] .. offsets[r+1])`. After ball carving finishes, each run is a leaf.
 *
 * Offsets are mirrored on the host because every round has to decide, per run, how many leaders
 * to sample and how to chunk its queries.
 */
struct runs {
  rmm::device_uvector<id_t> members;
  rmm::device_uvector<int64_t> offsets;
  std::vector<int64_t> h_offsets;

  explicit runs(cuda::stream_ref stream) : members(0, stream), offsets(0, stream) {}

  [[nodiscard]] auto count() const -> int64_t
  {
    return h_offsets.empty() ? 0 : static_cast<int64_t>(h_offsets.size()) - 1;
  }
  [[nodiscard]] auto size_of(int64_t r) const -> int64_t { return h_offsets[r + 1] - h_offsets[r]; }
  [[nodiscard]] auto total() const -> int64_t { return h_offsets.empty() ? 0 : h_offsets.back(); }
};

/** Statistics gathered while carving, surfaced for diagnostics and for tests to assert on. */
struct carve_stats {
  int64_t rounds_run     = 0;
  int64_t chopped_runs   = 0;
  int64_t merged_runs    = 0;
  int64_t oversized_left = 0;
};

/**
 * Replaces the host-side offsets and pushes them to the device.
 *
 * The sync is what lets the next call move a fresh vector into `h_offsets`: the copy reads
 * pageable host memory, so the old buffer has to survive until it completes. It costs one stream
 * sync per carve round against a round that is otherwise several GEMMs long.
 */
inline void set_offsets(raft::resources const& res, runs& r, std::vector<int64_t>&& offsets)
{
  auto stream = raft::resource::get_cuda_stream(res).get();
  r.h_offsets = std::move(offsets);
  r.offsets.resize(r.h_offsets.size(), stream);
  raft::update_device(
    r.offsets.data(), r.h_offsets.data(), static_cast<int64_t>(r.h_offsets.size()), stream);
  raft::resource::sync_stream(res);
}

/**
 * One carve round's decision for a single run: where its members sit in `runs::members`, and the
 * range of child slots it was given.
 *
 * A run that passes the round through untouched still reserves one slot, so `is_split` rather
 * than `slot_count` is what tells the two kernels which case they are in.
 */
struct run_plan {
  int64_t member_begin;
  int64_t member_count;
  int64_t slot_begin;
  int64_t slot_count;
  bool is_split;
};

/**
 * Picks `plan[r].slot_count` leaders out of run `r` by striding through it from a random offset.
 *
 * Striding rather than drawing independently is simpler and guarantees distinct picks whenever the
 * leader count does not exceed the run size.
 */
RAFT_KERNEL sample_leaders_kernel(const id_t* __restrict__ members,
                                  const run_plan* __restrict__ plan,
                                  int64_t n_runs,
                                  uint64_t seed,
                                  id_t* __restrict__ leaders)
{
  const int64_t r = blockIdx.x;
  if (r >= n_runs) { return; }
  const run_plan p    = plan[r];
  const int64_t size  = p.member_count;
  const int64_t count = p.slot_count;
  if (count <= 0 || size <= 0) { return; }

  // A splitmix64 finalizer; sampling only needs decorrelation across runs, not stream quality.
  uint64_t h = seed ^ (static_cast<uint64_t>(r) * 0x9e3779b97f4a7c15ULL);
  h ^= h >> 30;
  h *= 0xbf58476d1ce4e5b9ULL;
  h ^= h >> 27;
  const int64_t offset = static_cast<int64_t>(h % static_cast<uint64_t>(size));
  const int64_t stride = size / count > 0 ? size / count : 1;

  for (int64_t j = threadIdx.x; j < count; j += blockDim.x) {
    const int64_t pos         = (offset + j * stride) % size;
    leaders[p.slot_begin + j] = members[p.member_begin + pos];
  }
}

/**
 * Emits one (slot, point) pair per assignment slot, at a fixed position so the result does not
 * depend on scheduling. Unused slots get `kInvalidId`, which sorts to the end of the array and
 * is then trimmed. Runs that were not subdivided send all their members to one reserved slot.
 *
 * Each thread owns a member and writes that member's whole `fanout`-wide group, so a warp's store
 * covers `32 * fanout` elements to land 32 of them.
 *
 * The grid is two-dimensional because runs alone do not supply enough blocks: round 0 carves a
 * single run covering the whole dataset, which would use one block. `blockIdx.y` slices a run's
 * members, and the caller adds slices only when the run count cannot fill the device.
 */
RAFT_KERNEL emit_pairs_kernel(const id_t* __restrict__ members,
                              const run_plan* __restrict__ plan,
                              const id_t* __restrict__ sel_slots,
                              int64_t n_runs,
                              int64_t fanout,
                              id_t* __restrict__ keys,
                              id_t* __restrict__ vals)
{
  const int64_t r = blockIdx.x;
  if (r >= n_runs) { return; }
  const run_plan p    = plan[r];
  const int64_t begin = p.member_begin;
  const int64_t size  = p.member_count;
  const bool split    = p.is_split;
  const id_t own_slot = static_cast<id_t>(p.slot_begin);

  const int64_t slice_stride = static_cast<int64_t>(blockDim.x) * gridDim.y;
  for (int64_t j = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x; j < size;
       j += slice_stride) {
    const int64_t row = begin + j;
    const id_t point  = members[row];
    for (int64_t f = 0; f < fanout; f++) {
      const int64_t out = row * fanout + f;
      keys[out]         = split ? sel_slots[out] : (f == 0 ? own_slot : kInvalidId);
      vals[out]         = point;
    }
  }
}

/**
 * Sorts the first `n_total` (slot, point) pairs by slot, then derives run offsets from the
 * equal-slot runs among the first `n_valid` of them.
 *
 * CUB's radix sort is stable, so identical keys keep their emission order and the resulting
 * partition is reproducible across runs.
 */
inline void group_pairs_into_runs(raft::resources const& res,
                                  rmm::device_uvector<id_t>& keys,
                                  rmm::device_uvector<id_t>& vals,
                                  int64_t n_total,
                                  int64_t n_valid,
                                  int64_t n_slots,
                                  runs& out)
{
  auto stream = raft::resource::get_cuda_stream(res).get();
  if (n_valid == 0) {
    out.members.resize(0, stream);
    set_offsets(res, out, std::vector<int64_t>{0});
    return;
  }

  // Real slots are below `n_slots`, so only those bits can separate two keys; `kInvalidId` keeps
  // its all-ones value and still sorts last.
  radix_sort_pairs_inplace(stream, keys, vals, n_total, std::max(1, key_bits(n_slots)));

  // Where each slot's run starts. A binary search rather than a run-length encode because slots
  // that received no member still need a bound, and an RLE would silently omit them.
  rmm::device_uvector<int64_t> slot_begin(n_slots + 1, stream);
  thrust::lower_bound(raft::resource::get_thrust_policy(res),
                      keys.data(),
                      keys.data() + n_valid,
                      thrust::counting_iterator<id_t>(0),
                      thrust::counting_iterator<id_t>(static_cast<id_t>(n_slots + 1)),
                      slot_begin.data());

  std::vector<int64_t> h_slot_begin(n_slots + 1);
  raft::update_host(h_slot_begin.data(), slot_begin.data(), n_slots + 1, stream);
  raft::resource::sync_stream(res);

  // Runs in increasing slot order, skipping empty slots. Empty slots are zero wide, so the
  // non-empty ones are adjacent and the boundaries are just their start offsets.
  std::vector<int64_t> h_offsets;
  for (int64_t s = 0; s < n_slots; s++) {
    if (h_slot_begin[s + 1] > h_slot_begin[s]) { h_offsets.push_back(h_slot_begin[s]); }
  }
  h_offsets.push_back(n_valid);

  vals.resize(n_valid, stream);
  out.members = std::move(vals);
  set_offsets(res, out, std::move(h_offsets));
}

/** How one assign-then-sort round chooses its leader counts. */
struct round_config {
  /** Neighbours each member is assigned to. Values above 1 create the overlap the graph needs. */
  int64_t fanout;
  /** Runs at or below this size pass through untouched. */
  int64_t split_above;
  /** Desired size of the runs this round produces. */
  int64_t target_size;
  /** Hard cap on leaders per run; bounds the top-level round. */
  int64_t max_leaders;
};

/**
 * Runs one assign-then-sort round: subdivide every run larger than `cfg.split_above` by
 * sampling leaders inside it and assigning each member to its `cfg.fanout` nearest, then
 * regroup all assignments into new runs.
 *
 * This is the flattened form of the CPU algorithm's recursion (GPiPNN Sec 5.1): the whole
 * dataset advances one level per call instead of descending one subtree at a time. With
 * `fanout > 1` the round creates overlap; with `fanout == 1` it purely subdivides, which is how
 * the oversized tail is brought under the size bound.
 *
 * Slot ids are allocated in input-run order, so the sorted output keeps runs descending from
 * the same parent adjacent. Undersized-run merging depends on that.
 */
inline void assign_then_sort_round(
  raft::resources const& res,
  raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
  raft::device_vector_view<const float, int64_t> norms,
  const runs& in,
  const round_config& cfg,
  uint64_t seed,
  runs& out)
{
  auto stream          = raft::resource::get_cuda_stream(res).get();
  const int64_t n_runs = in.count();
  const int64_t fanout = cfg.fanout;
  const int64_t n_rows = in.total();
  if (n_runs == 0) {
    set_offsets(res, out, std::vector<int64_t>{0});
    return;
  }

  // Reserve slots: one per pass-through run, `n_sub` per run being subdivided.
  std::vector<run_plan> h_plan(n_runs);
  int64_t total_slots = 0;
  for (int64_t r = 0; r < n_runs; r++) {
    const int64_t size = in.size_of(r);
    run_plan& p        = h_plan[r];
    p.member_begin     = in.h_offsets[r];
    p.member_count     = size;
    p.slot_begin       = total_slots;
    // A run must offer at least `fanout` leaders, so
    // one too small to do that passes through even if it exceeds the size threshold.
    const int64_t max_sub = std::min(cfg.max_leaders, size);
    if (size <= cfg.split_above || max_sub < fanout) {
      p.is_split   = false;
      p.slot_count = 1;
      total_slots += 1;
      continue;
    }
    // Overlap multiplies the member count by `fanout`
    int64_t n_sub = (size * fanout + cfg.target_size - 1) / cfg.target_size;
    n_sub         = std::clamp<int64_t>(n_sub, fanout, max_sub);
    p.is_split    = true;
    p.slot_count  = n_sub;
    total_slots += n_sub;
  }
  RAFT_EXPECTS(total_slots < static_cast<int64_t>(kInvalidId),
               "gpipnn: partition produced too many buckets to address with a 32-bit id");

  rmm::device_uvector<run_plan> d_plan(n_runs, stream);
  raft::update_device(d_plan.data(), h_plan.data(), n_runs, stream);

  rmm::device_uvector<id_t> leaders(total_slots, stream);
  sample_leaders_kernel<<<n_runs, 128, 0, stream>>>(
    in.members.data(), d_plan.data(), n_runs, seed, leaders.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Chunk each run's queries so no segment's working set overruns the L2 cache budget. All chunks
  // of a run share its leader range, which lets one dense call serve every run at once.
  const int64_t budget = l2_cache_bytes();
  const int64_t dim    = dataset.extent(1);
  std::vector<segment> segments;
  for (const auto& p : h_plan) {
    if (!p.is_split) { continue; }
    const int64_t max_chunk = std::min(max_query_chunk(budget, p.slot_count, dim), p.member_count);
    for (int64_t q = 0; q < p.member_count; q += max_chunk) {
      segments.push_back(segment{
        p.member_begin + q, std::min(max_chunk, p.member_count - q), p.slot_begin, p.slot_count});
    }
  }

  // Selection results are indexed by the query's position in `in.members`, so rows belonging to
  // pass-through runs are never written and never read. Only the assigned slot matters, so no
  // distances are needed.
  const int64_t n_total = n_rows * fanout;
  rmm::device_uvector<id_t> keys(n_total, stream);
  rmm::device_uvector<id_t> vals(n_total, stream);
  {
    rmm::device_uvector<id_t> sel_ids(n_total, stream);
    rmm::device_uvector<id_t> sel_slots(n_total, stream);

    segmented_topk(res,
                   dataset,
                   norms,
                   raft::make_device_vector_view<const id_t, int64_t>(in.members.data(), n_rows),
                   raft::make_device_vector_view<const id_t, int64_t>(leaders.data(), total_slots),
                   segments,
                   fanout,
                   false,
                   raft::make_device_matrix_view<id_t, int64_t>(sel_ids.data(), n_rows, fanout),
                   std::nullopt,
                   raft::make_device_matrix_view<id_t, int64_t>(sel_slots.data(), n_rows, fanout));

    // The id half is never consulted, so it is released before the sort to reduce peak memory
    // usage.
    sel_ids.resize(0, stream);
    sel_ids.shrink_to_fit(stream);

    // Slices per run, so a round with too few runs to fill the device still gets a full grid. The
    // 65535 cap on `gridDim.y` is why the kernel keeps a grid-stride loop.
    int device = 0;
    RAFT_CUDA_TRY(cudaGetDevice(&device));
    int sms = 0;
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const int64_t want_blocks = (sms > 0 ? static_cast<int64_t>(sms) : 64) * 32;
    const int64_t slices =
      std::clamp<int64_t>(want_blocks / std::max<int64_t>(1, n_runs), 1, 65535);
    const dim3 emit_grid(static_cast<unsigned>(n_runs), static_cast<unsigned>(slices));
    emit_pairs_kernel<<<emit_grid, 128, 0, stream>>>(
      in.members.data(), d_plan.data(), sel_slots.data(), n_runs, fanout, keys.data(), vals.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  const int64_t n_invalid = thrust::count(
    raft::resource::get_thrust_policy(res), keys.data(), keys.data() + n_total, kInvalidId);
  group_pairs_into_runs(res, keys, vals, n_total, n_total - n_invalid, total_slots, out);
}

/** Slices runs still above `max_size` into fixed-size chunks. */
inline void chop_oversized(raft::resources const& res, runs& r, int64_t max_size, int64_t& chopped)
{
  std::vector<int64_t> fixed;
  fixed.reserve(r.h_offsets.size());
  fixed.push_back(0);
  for (int64_t i = 0; i < r.count(); i++) {
    const int64_t begin = r.h_offsets[i];
    const int64_t size  = r.size_of(i);
    if (size <= max_size) {
      fixed.push_back(begin + size);
      continue;
    }
    chopped++;
    for (int64_t o = 0; o < size; o += max_size) {
      fixed.push_back(begin + std::min(o + max_size, size));
    }
  }
  set_offsets(res, r, std::move(fixed));
}

/**
 * Merges consecutive runs below `min_size` into their neighbours, up to `max_size`.
 */
inline void merge_undersized(
  raft::resources const& res, runs& r, int64_t min_size, int64_t max_size, int64_t& merged)
{
  std::vector<int64_t> fixed;
  fixed.reserve(r.h_offsets.size());
  fixed.push_back(0);
  int64_t i = 0;
  while (i < r.count()) {
    int64_t end  = i;
    int64_t size = 0;
    while (end < r.count() && size < min_size && size + r.size_of(end) <= max_size) {
      size += r.size_of(end);
      end++;
    }
    if (end == i) {  // A single run already large enough to stand alone.
      end = i + 1;
    }
    if (end > i + 1) { merged += end - i; }
    fixed.push_back(r.h_offsets[end]);
    i = end;
  }
  set_offsets(res, r, std::move(fixed));
}

/**
 * Partitions the dataset into small overlapping leaves by flattened randomized ball carving.
 *
 * Two overlap-creating rounds produce the leaves; further rounds, if the size bound is still
 * violated, subdivide with fanout 1 and touch only the shrinking oversized tail. Chopping is a
 * last-resort backstop because its chunks are arbitrary subsets rather than balls.
 */
inline auto ball_carve(raft::resources const& res,
                       raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
                       raft::device_vector_view<const float, int64_t> norms,
                       const index_params& params,
                       carve_stats& stats) -> runs
{
  auto stream            = raft::resource::get_cuda_stream(res).get();
  const int64_t n_rows   = dataset.extent(0);
  const int64_t max_leaf = params.max_leaf_size;
  const int64_t target =
    (static_cast<int64_t>(params.min_leaf_size) + static_cast<int64_t>(params.max_leaf_size)) / 2;

  // Seed with a single run holding every point in id order.
  runs cur(stream);
  cur.members.resize(n_rows, stream);
  thrust::sequence(raft::resource::get_thrust_policy(res),
                   cur.members.data(),
                   cur.members.data() + n_rows,
                   id_t{0});
  set_offsets(res, cur, std::vector<int64_t>{0, n_rows});

  // A dataset that already fits in one leaf needs no carving.
  if (n_rows <= max_leaf) { return cur; }

  // Round 0: a global leader sample, sized by rate and capped
  const int64_t n_leaders =
    std::clamp<int64_t>(static_cast<int64_t>(params.leader_fraction * static_cast<float>(n_rows)),
                        params.fanout_level0,
                        std::min<int64_t>(params.max_leaders, n_rows));
  runs r0(stream);
  assign_then_sort_round(
    res,
    dataset,
    norms,
    cur,
    round_config{static_cast<int64_t>(params.fanout_level0),
                 /* split_above */ max_leaf,
                 /* target_size */ std::max<int64_t>(1, n_rows * params.fanout_level0 / n_leaders),
                 /* max_leaders */ n_leaders},
    params.seed,
    r0);
  cur              = std::move(r0);
  stats.rounds_run = 1;

  // Round 1: bucket-local sub-leaders, sized to land the resulting leaves near `target`.
  runs r1(stream);
  assign_then_sort_round(res,
                         dataset,
                         norms,
                         cur,
                         round_config{static_cast<int64_t>(params.fanout_level1),
                                      /* split_above */ target,
                                      /* target_size */ target,
                                      /* max_leaders */ params.max_leaders},
                         params.seed + 1,
                         r1);
  cur              = std::move(r1);
  stats.rounds_run = 2;

  // Size-driven subdivision of whatever is still oversized. Fanout 1 so it only subdivides.
  while (stats.rounds_run < static_cast<int64_t>(params.max_rounds)) {
    int64_t oversized = 0;
    for (int64_t i = 0; i < cur.count(); i++) {
      if (cur.size_of(i) > max_leaf) { oversized++; }
    }
    if (oversized == 0) { break; }
    runs extra(stream);
    assign_then_sort_round(res,
                           dataset,
                           norms,
                           cur,
                           round_config{/* fanout */ 1,
                                        /* split_above */ max_leaf,
                                        /* target_size */ target,
                                        /* max_leaders */ params.max_leaders},
                           params.seed + 2 + static_cast<uint64_t>(stats.rounds_run),
                           extra);
    cur = std::move(extra);
    stats.rounds_run++;
  }

  for (int64_t i = 0; i < cur.count(); i++) {
    if (cur.size_of(i) > max_leaf) { stats.oversized_left++; }
  }
  chop_oversized(res, cur, max_leaf, stats.chopped_runs);
  merge_undersized(res, cur, params.min_leaf_size, max_leaf, stats.merged_runs);
  return cur;
}

}  // namespace cuvs::neighbors::experimental::gpipnn::detail
