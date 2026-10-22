/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
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
#include <thrust/count.h>
#include <thrust/execution_policy.h>
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

/** Replaces the host-side offsets and pushes them to the device. */
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
 * Picks `slot_count[r]` leaders out of run `r` by striding through it from a random offset.
 *
 * Striding rather than drawing independently is simpler and guarantees distinct picks whenever the
 * leader count does not exceed the run size.
 */
RAFT_KERNEL sample_leaders_kernel(const id_t* __restrict__ members,
                                  const int64_t* __restrict__ run_begin,
                                  const int64_t* __restrict__ run_size,
                                  const int64_t* __restrict__ slot_begin,
                                  const int64_t* __restrict__ slot_count,
                                  int64_t n_runs,
                                  uint64_t seed,
                                  id_t* __restrict__ leaders)
{
  const int64_t r = blockIdx.x;
  if (r >= n_runs) { return; }
  const int64_t size  = run_size[r];
  const int64_t count = slot_count[r];
  if (count <= 0 || size <= 0) { return; }

  // A splitmix64 finalizer; sampling only needs decorrelation across runs, not stream quality.
  uint64_t h = seed ^ (static_cast<uint64_t>(r) * 0x9e3779b97f4a7c15ULL);
  h ^= h >> 30;
  h *= 0xbf58476d1ce4e5b9ULL;
  h ^= h >> 27;
  const int64_t offset = static_cast<int64_t>(h % static_cast<uint64_t>(size));
  const int64_t stride = size / count > 0 ? size / count : 1;

  for (int64_t j = threadIdx.x; j < count; j += blockDim.x) {
    const int64_t pos          = (offset + j * stride) % size;
    leaders[slot_begin[r] + j] = members[run_begin[r] + pos];
  }
}

/**
 * Emits one (slot, point) pair per assignment slot, at a fixed position so the result does not
 * depend on scheduling. Unused slots get `kInvalidId`, which sorts to the end of the array and
 * is then trimmed. Runs that were not subdivided send all their members to one reserved slot.
 *
 * Each thread owns a member and writes that member's whole `fanout`-wide group, so a warp's store
 * covers `32 * fanout` elements to land 32 of them.
 * Which block emits a pair cannot change the output, since every write lands at a position fixed
 * by the member's own index.
 *
 * The grid is two-dimensional because runs alone do not supply enough blocks (round 0 carves a
 * single run covering the whole dataset, which would use 1 block). `blockIdx.y`
 * slices a run's members, and the caller adds slices only when the run count cannot fill the
 * device, so later rounds keep the grid they had.
 *

 */
RAFT_KERNEL emit_pairs_kernel(const id_t* __restrict__ members,
                              const int64_t* __restrict__ run_begin,
                              const int64_t* __restrict__ run_size,
                              const int64_t* __restrict__ slot_begin,
                              const uint8_t* __restrict__ is_split,
                              const id_t* __restrict__ sel_slots,
                              int64_t n_runs,
                              int64_t fanout,
                              id_t* __restrict__ keys,
                              id_t* __restrict__ vals)
{
  const int64_t r = blockIdx.x;
  if (r >= n_runs) { return; }
  const int64_t begin = run_begin[r];
  const int64_t size  = run_size[r];
  const bool split    = is_split[r] != 0;
  const id_t own_slot = static_cast<id_t>(slot_begin[r]);

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
 * Locates each slot's contiguous block in the slot-sorted assignment array.
 *
 * The same binary search `find_owner_bounds_kernel` does for record owners. Also lets slots with no
 * member find its bound and report itself as empty.
 */
RAFT_KERNEL find_slot_bounds_kernel(const id_t* __restrict__ keys,
                                    int64_t n_keys,
                                    int64_t n_slots,
                                    int64_t* __restrict__ slot_begin)
{
  const int64_t s = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (s > n_slots) { return; }
  const auto target = static_cast<id_t>(s);
  int64_t lo = 0, hi = n_keys;
  while (lo < hi) {
    const int64_t mid = lo + (hi - lo) / 2;
    if (keys[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  slot_begin[s] = lo;
}

/**
 * Highest key bit that can distinguish two slot keys, given how many slots exist.
 *
 * Real slots are below `n_slots`, so every real key's bits above this point are zero, except
 * `kInvalidId`.
 *
 * Same reasoning as `sort_key_bits` in hash_prune.cuh.
 */
inline auto slot_key_bits(int64_t n_slots) -> int
{
  int bits = 1;
  while (bits < 32 && (int64_t{1} << bits) - 1 < n_slots) {
    bits++;
  }
  return bits;
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

  {
    rmm::device_uvector<id_t> keys_alt(n_total, stream);
    rmm::device_uvector<id_t> vals_alt(n_total, stream);
    cub::DoubleBuffer<id_t> d_keys(keys.data(), keys_alt.data());
    cub::DoubleBuffer<id_t> d_vals(vals.data(), vals_alt.data());
    const int end_bit = slot_key_bits(n_slots);
    size_t bytes      = 0;
    RAFT_CUDA_TRY(
      cub::DeviceRadixSort::SortPairs(nullptr, bytes, d_keys, d_vals, n_total, 0, end_bit, stream));
    rmm::device_uvector<char> workspace(bytes, stream);
    RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
      workspace.data(), bytes, d_keys, d_vals, n_total, 0, end_bit, stream));
    if (d_keys.Current() != keys.data()) { keys = std::move(keys_alt); }
    if (d_vals.Current() != vals.data()) { vals = std::move(vals_alt); }
  }

  rmm::device_uvector<int64_t> slot_begin(n_slots + 1, stream);
  const int64_t threads = 128;
  find_slot_bounds_kernel<<<(n_slots + threads) / threads, threads, 0, stream>>>(
    keys.data(), n_valid, n_slots, slot_begin.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

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
  std::vector<int64_t> h_run_begin(n_runs), h_run_size(n_runs), h_slot_begin(n_runs),
    h_slot_count(n_runs);
  std::vector<uint8_t> h_is_split(n_runs);
  int64_t total_slots = 0;
  for (int64_t r = 0; r < n_runs; r++) {
    const int64_t size = in.size_of(r);
    h_run_begin[r]     = in.h_offsets[r];
    h_run_size[r]      = size;
    h_slot_begin[r]    = total_slots;
    // A run must offer at least `fanout` leaders, so
    // one too small to do that passes through even if it exceeds the size threshold.
    const int64_t max_sub = std::min(cfg.max_leaders, size);
    if (size <= cfg.split_above || max_sub < fanout) {
      h_is_split[r]   = 0;
      h_slot_count[r] = 1;
      total_slots += 1;
      continue;
    }
    // Overlap multiplies the member count by `fanout`
    int64_t n_sub   = (size * fanout + cfg.target_size - 1) / cfg.target_size;
    n_sub           = std::clamp<int64_t>(n_sub, fanout, max_sub);
    h_is_split[r]   = 1;
    h_slot_count[r] = n_sub;
    total_slots += n_sub;
  }
  RAFT_EXPECTS(total_slots < static_cast<int64_t>(kInvalidId),
               "gpipnn: partition produced too many buckets to address with a 32-bit id");

  rmm::device_uvector<int64_t> d_run_begin(n_runs, stream);
  rmm::device_uvector<int64_t> d_run_size(n_runs, stream);
  rmm::device_uvector<int64_t> d_slot_begin(n_runs, stream);
  rmm::device_uvector<int64_t> d_slot_count(n_runs, stream);
  rmm::device_uvector<uint8_t> d_is_split(n_runs, stream);
  raft::update_device(d_run_begin.data(), h_run_begin.data(), n_runs, stream);
  raft::update_device(d_run_size.data(), h_run_size.data(), n_runs, stream);
  raft::update_device(d_slot_begin.data(), h_slot_begin.data(), n_runs, stream);
  raft::update_device(d_slot_count.data(), h_slot_count.data(), n_runs, stream);
  raft::update_device(d_is_split.data(), h_is_split.data(), n_runs, stream);

  rmm::device_uvector<id_t> leaders(total_slots, stream);
  sample_leaders_kernel<<<n_runs, 128, 0, stream>>>(in.members.data(),
                                                    d_run_begin.data(),
                                                    d_run_size.data(),
                                                    d_slot_begin.data(),
                                                    d_slot_count.data(),
                                                    n_runs,
                                                    seed,
                                                    leaders.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Chunk each run's queries so no segment's working set overruns the L2 cache budget. All chunks
  // of a run share its leader range, which lets one dense call serve every run at once.
  const int64_t budget = l2_cache_bytes();
  const int64_t dim    = dataset.extent(1);
  std::vector<segment> segments;
  for (int64_t r = 0; r < n_runs; r++) {
    if (h_is_split[r] == 0) { continue; }
    const int64_t cands     = h_slot_count[r];
    const int64_t max_chunk = std::min(max_query_chunk(budget, cands, dim), h_run_size[r]);
    for (int64_t q = 0; q < h_run_size[r]; q += max_chunk) {
      segments.push_back(segment{
        h_run_begin[r] + q, std::min(max_chunk, h_run_size[r] - q), h_slot_begin[r], cands});
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
    const int64_t want_blocks = sm_count() * 32;
    const int64_t slices =
      std::clamp<int64_t>(want_blocks / std::max<int64_t>(1, n_runs), 1, 65535);
    const dim3 emit_grid(static_cast<unsigned>(n_runs), static_cast<unsigned>(slices));
    emit_pairs_kernel<<<emit_grid, 128, 0, stream>>>(in.members.data(),
                                                     d_run_begin.data(),
                                                     d_run_size.data(),
                                                     d_slot_begin.data(),
                                                     d_is_split.data(),
                                                     sel_slots.data(),
                                                     n_runs,
                                                     fanout,
                                                     keys.data(),
                                                     vals.data());
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
 *
 * Because slot ids were allocated in parent-run order, consecutive runs descend from the same
 * parent and are therefore geometrically close. This stands in for the paper's random merge and
 * should be no worse: merging exists only so each leaf offers enough candidates for the prune
 * step, and pairs drawn from a distant leaf would lose top-2 selection anyway.
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
