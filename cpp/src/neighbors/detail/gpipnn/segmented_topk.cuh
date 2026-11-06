/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"
#include "fused_leaf_topk.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <optional>
#include <vector>

namespace cuvs::neighbors::experimental::gpipnn::detail {

/**
 * Gathers rows named by `ids` into a padded slab of `n_seg * pad` rows. Rows past a segment's
 * true count are zero-filled so the GEMM produces finite garbage that selection masks.
 *
 * Not `raft::matrix::gather` because the padded destination layout has no source row to name for
 * the padding slots.
 */
RAFT_KERNEL gather_padded_kernel(const half* __restrict__ dataset,
                                 const id_t* __restrict__ ids,
                                 const segment_dev* __restrict__ segs,
                                 bool use_query_side,
                                 int64_t pad,
                                 int64_t dim,
                                 half* __restrict__ slab)
{
  const int64_t slab_row = blockIdx.x;
  const int64_t seg      = slab_row / pad;
  const int64_t local    = slab_row - seg * pad;

  const segment_dev s = segs[seg];
  const int64_t count = use_query_side ? s.query_count : s.cand_count;
  const int64_t begin = use_query_side ? s.query_begin : s.cand_begin;

  half* dst = slab + slab_row * dim;
  if (local >= count) {
    for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
      dst[i] = __float2half(0.0f);
    }
    return;
  }
  const half* src = dataset + static_cast<int64_t>(ids[begin + local]) * dim;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    dst[i] = src[i];
  }
}

/**
 * The same gather, moving 16 bytes per thread instead of 2.
 *
 * Indexing flat over 16-byte vectors decouples the block shape from `dim`, so a warp moves 512
 * contiguous bytes at any dimension rather than the scalar kernel's 64.
 *
 * Only launched when `dim` is a multiple of `kHalfPerVec` and both buffers are 16-byte aligned,
 * which is what keeps every vector aligned.
 */
RAFT_KERNEL gather_padded_vec_kernel(const uint4* __restrict__ dataset,
                                     const id_t* __restrict__ ids,
                                     const segment_dev* __restrict__ segs,
                                     bool use_query_side,
                                     int64_t pad,
                                     int64_t vecs_per_row,
                                     int64_t n_slab_rows,
                                     uint4* __restrict__ slab)
{
  const int64_t total  = n_slab_rows * vecs_per_row;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int64_t slab_row = idx / vecs_per_row;
    const int64_t v        = idx - slab_row * vecs_per_row;
    const int64_t seg      = slab_row / pad;
    const int64_t local    = slab_row - seg * pad;

    const segment_dev s = segs[seg];
    const int64_t count = use_query_side ? s.query_count : s.cand_count;
    const int64_t begin = use_query_side ? s.query_begin : s.cand_begin;

    uint4* dst = slab + slab_row * vecs_per_row + v;
    if (local >= count) {
      *dst = uint4{0u, 0u, 0u, 0u};
      continue;
    }
    *dst = dataset[static_cast<int64_t>(ids[begin + local]) * vecs_per_row + v];
  }
}

/** Halves one `uint4` of the vectorized gather carries. */
inline constexpr int64_t kHalfPerVec = static_cast<int64_t>(sizeof(uint4) / sizeof(half));

/** Gathers `n_slab_rows` padded rows, vectorized where the shapes allow it. */
inline void launch_gather_padded(const half* dataset,
                                 const id_t* ids,
                                 const segment_dev* segs,
                                 bool use_query_side,
                                 int64_t pad,
                                 int64_t dim,
                                 int64_t n_slab_rows,
                                 half* slab,
                                 cudaStream_t stream)
{
  if (n_slab_rows <= 0) { return; }
  // The 16-byte-per-thread gather needs both buffers aligned and a whole number of vectors a row.
  constexpr auto kAlign = static_cast<uintptr_t>(sizeof(uint4));
  if (dim % kHalfPerVec == 0 && reinterpret_cast<uintptr_t>(dataset) % kAlign == 0 &&
      reinterpret_cast<uintptr_t>(slab) % kAlign == 0) {
    const int64_t vecs_per_row = dim / kHalfPerVec;
    const int64_t total        = n_slab_rows * vecs_per_row;
    const int threads          = 256;
    // Four vectors per thread keeps several loads outstanding at every dimension; the cap stops
    // the grid growing with the slab.
    const int64_t blocks =
      std::clamp<int64_t>((total + threads * 4 - 1) / (threads * 4), 1, 262144);
    gather_padded_vec_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint4*>(dataset),
      ids,
      segs,
      use_query_side,
      pad,
      vecs_per_row,
      n_slab_rows,
      reinterpret_cast<uint4*>(slab));
  } else {
    const int threads = static_cast<int>(std::min<int64_t>(dim, 256));
    gather_padded_kernel<<<n_slab_rows, threads, 0, stream>>>(
      dataset, ids, segs, use_query_side, pad, dim, slab);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Batched extents are rounded up to a multiple of this, to 16-byte-align the score matrix's
 * leading dimension and batch stride.
 */
inline constexpr int64_t kPadMultiple = 8;

/**
 * Reduces one row of the dot-product matrix straight to its `k` nearest candidates.
 *
 * Scores are never materialized. Ordering uses one unsigned 64-bit key per candidate:
 * order-preserving score in the high half, the candidate's column in the low half.
 *
 * One warp per query row. Each lane sweeps a strided slice, so reads coalesce, and keeps its own
 * `K` best in registers; a butterfly minimum reduces the lists to the global best `k`.
 *
 * @tparam K per-lane list length; must be at least `k`.
 */
template <int K>
RAFT_KERNEL fused_topk_kernel(const float* __restrict__ dots,
                              const float* __restrict__ norms,
                              const id_t* __restrict__ query_ids,
                              const id_t* __restrict__ cand_ids,
                              const segment_dev* __restrict__ segs,
                              int64_t rows,
                              int64_t query_pad,
                              int64_t cand_pad,
                              bool exclude_self,
                              int64_t k,
                              id_t* __restrict__ out_ids,
                              float* __restrict__ out_dists,
                              id_t* __restrict__ out_cand_slots)
{
  const int lane    = static_cast<int>(threadIdx.x) & (warpSize - 1);
  const int n_warps = static_cast<int>(blockDim.x) / warpSize;
  const int64_t row =
    static_cast<int64_t>(blockIdx.x) * n_warps + static_cast<int>(threadIdx.x) / warpSize;
  // Warp-uniform, as is every branch below, so all 32 lanes reach the butterfly at the end.
  if (row >= rows) { return; }

  const int64_t seg     = row / query_pad;
  const int64_t local_q = row - seg * query_pad;
  const segment_dev s   = segs[seg];

  // A padded row has no output slot; the emit indexes by `s.query_begin + local_q`.
  if (local_q >= s.query_count) { return; }

  const id_t q_id      = query_ids[s.query_begin + local_q];
  const float q_norm   = norms[q_id];
  const float* dot_row = dots + row * cand_pad;
  const int64_t n_cand = s.cand_count;

  const uint32_t salt = row_salt(q_id);

  uint64_t best[K];
#pragma unroll
  for (int i = 0; i < K; i++) {
    best[i] = kInvalidKey;
  }

  // Columns past the segment's candidate count are batch padding and are not visited.
  for (int64_t c = lane; c < n_cand; c += warpSize) {
    const id_t c_id   = cand_ids[s.cand_begin + c];
    const bool masked = exclude_self && c_id == q_id;
    const float score = masked ? kMaskedScore : q_norm + norms[c_id] - 2.0f * dot_row[c];
    fused_insert_sorted<K>(best,
                           (static_cast<uint64_t>(order_preserving_u32(score)) << 32) |
                             static_cast<uint64_t>(static_cast<uint32_t>(c) ^ salt));
  }

  for (int64_t j = 0; j < k; j++) {
    uint64_t m = best[0];
#pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      const uint64_t v = __shfl_xor_sync(0xffffffffu, m, off);
      m                = v < m ? v : m;
    }
    if (lane == 0) {
      // No real key collides with `kInvalidKey`: the high half tops out at +inf's 0xff800000.
      emit_rank(m,
                salt,
                q_id,
                cand_ids,
                s.cand_begin,
                exclude_self,
                (s.query_begin + local_q) * k + j,
                out_ids,
                out_dists,
                out_cand_slots);
    }
    // Each column lives in exactly one lane, so at most one lane matches and the pop needs no
    // further arbitration.
    if (m != kInvalidKey && best[0] == m) { fused_pop_front(best); }
  }
}

/**
 * Divisor applied to L2 before it becomes the working-set budget.
 *
 * Operands and scores count against the same budget, so at 1 the GEMM slabs and score matrix
 * contend for the same cache.
 */
inline constexpr int64_t kL2BudgetDiv = 2;

/**
 * Marks a buffer as persisting in L2 for as long as the guard lives.
 *
 * `dots` is the one buffer worth holding: the GEMM writes it and selection reads it straight back,
 * while the operand slabs are streamed once each.
 *
 * Purely a hint: it changes residency, never results. Every call that sets it up is therefore
 * best-effort, and a driver that refuses one leaves the window inactive and the error state
 * cleared rather than failing the build.
 *
 * Both settings outlive a plain scope -- the carve-out is device-wide and the access policy
 * belongs to the caller's stream -- so the guard records what it found and puts both back.
 * `cudaCtxResetPersistingL2Cache` is still not called, because `segmented_topk` is asynchronous
 * and the guard dies while its kernels are still in flight; dropping the carve-out is what
 * releases the reservation.
 */
class persisting_l2_window {
 public:
  persisting_l2_window(cudaStream_t stream, const void* ptr, int64_t bytes) : stream_(stream)
  {
    if (ptr == nullptr || bytes <= 0) { return; }
    int device = 0;
    RAFT_CUDA_TRY(cudaGetDevice(&device));
    int max_window = 0;
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxPersistingL2CacheSize, device));
    if (max_window <= 0) { return; }

    // Names the first `max_window` bytes rather than spreading a hit ratio over the whole matrix,
    // which would let later rows evict earlier ones from the reserved lines.
    const int64_t window = std::min<int64_t>(bytes, max_window);

    // The carve-out is device-wide, so a concurrent build may already have raised it. Growing it
    // only when it is too small, and rolling back only a value this guard itself wrote, keeps two
    // builds from cutting each other's window short.
    size_t prior_limit  = 0;
    const auto read_err = cudaDeviceGetLimit(&prior_limit, cudaLimitPersistingL2CacheSize);
    if (read_err != cudaSuccess) {
      (void)cudaGetLastError();
      RAFT_LOG_DEBUG("gpipnn: L2 carve-out unreadable (%s); running without it",
                     cudaGetErrorString(read_err));
      return;
    }
    if (prior_limit < static_cast<size_t>(window)) {
      const auto limit_err =
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, static_cast<size_t>(window));
      if (limit_err != cudaSuccess) {
        (void)cudaGetLastError();
        RAFT_LOG_DEBUG("gpipnn: L2 carve-out of %ld bytes refused (%s); running without it",
                       static_cast<long>(window),
                       cudaGetErrorString(limit_err));
        return;
      }
      prior_limit_ = prior_limit;
      set_limit_   = static_cast<size_t>(window);
      owns_limit_  = true;
    }

    // Saved so a window the caller set for its own reasons survives this guard.
    const auto prior_err =
      cudaStreamGetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &prior_attr_);
    if (prior_err != cudaSuccess) {
      (void)cudaGetLastError();
      RAFT_LOG_DEBUG("gpipnn: stream access policy unreadable (%s); running without it",
                     cudaGetErrorString(prior_err));
      restore_limit();
      return;
    }

    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr  = const_cast<void*>(ptr);
    attr.accessPolicyWindow.num_bytes = static_cast<size_t>(window);
    // Window and carve-out are the same size, so the ratio has nothing to ration.
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp  = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    const auto attr_err =
      cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    if (attr_err != cudaSuccess) {
      (void)cudaGetLastError();
      RAFT_LOG_DEBUG("gpipnn: L2 access-policy window refused (%s); running without it",
                     cudaGetErrorString(attr_err));
      restore_limit();
      return;
    }
    active_ = true;
  }

  ~persisting_l2_window()
  {
    // Unchecked throughout: this can run while an exception from the batch loop propagates, and
    // losing it to report a failed cache hint would be a poor trade.
    if (active_) {
      RAFT_CUDA_TRY_NO_THROW(
        cudaStreamSetAttribute(stream_, cudaStreamAttributeAccessPolicyWindow, &prior_attr_));
    }
    restore_limit();
  }

  persisting_l2_window(const persisting_l2_window&)                    = delete;
  persisting_l2_window(persisting_l2_window&&)                         = delete;
  auto operator=(const persisting_l2_window&) -> persisting_l2_window& = delete;
  auto operator=(persisting_l2_window&&) -> persisting_l2_window&      = delete;

 private:
  /** Puts the device-wide carve-out back, but only if it still holds this guard's value. */
  void restore_limit()
  {
    if (!owns_limit_) { return; }
    owns_limit_    = false;
    size_t current = 0;
    if (cudaDeviceGetLimit(&current, cudaLimitPersistingL2CacheSize) != cudaSuccess) {
      (void)cudaGetLastError();
      return;
    }
    // Anything else means a concurrent build raised it further and is still relying on it.
    if (current != set_limit_) { return; }
    if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prior_limit_) != cudaSuccess) {
      (void)cudaGetLastError();
    }
  }

  cudaStream_t stream_ = nullptr;
  cudaStreamAttrValue prior_attr_{};
  size_t prior_limit_ = 0;
  size_t set_limit_   = 0;
  bool owns_limit_    = false;
  bool active_        = false;
};

/** Returns the working-set budget in bytes, used to bound how many segments run at once. */
inline auto l2_cache_bytes() -> int64_t
{
  int device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  int l2 = 0;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device));
  // Fall back to a conservative 8 MiB if the driver reports nothing useful.
  const int64_t bytes = l2 > 0 ? static_cast<int64_t>(l2) : (8 << 20);
  return bytes / kL2BudgetDiv;
}

/**
 * Largest number of query rows one segment scoring against `cand_count` candidates can carry
 * with its own working set still inside `budget`.
 *
 * Inverts the batch planner's cost formula at one segment, where the candidate slab is fixed and
 * every other term is linear in the query count. Both slabs are counted even where only one will
 * be gathered: whether a call is a self-join depends on whether this bound leaves its queries in
 * a single chunk. Never returns less than 1.
 */
inline auto max_query_chunk(int64_t budget, int64_t cand_count, int64_t dim) -> int64_t
{
  const int64_t per_query =
    cand_count * static_cast<int64_t>(sizeof(float)) + dim * static_cast<int64_t>(sizeof(half));
  const int64_t cand_fixed = cand_count * dim * static_cast<int64_t>(sizeof(half));
  return std::max<int64_t>(1, (budget - cand_fixed) / per_query);
}

/** One batched launch: a window of the size-sorted order, and the extents every member pads to. */
struct batch {
  size_t begin;
  size_t end;
  int64_t query_pad;
  int64_t cand_pad;
};

/** A batched-path schedule, and the buffer sizes that cover every batch in it. */
struct batch_plan {
  std::vector<int64_t> order;  // Segment indices, in the order the batches walk them
  std::vector<batch> batches;
  int64_t query_slab = 0;  // Halves
  int64_t cand_slab  = 0;  // Halves; zero on a self-join, which reuses the query slab
  int64_t dots       = 0;  // Floats
};

/**
 * Groups `order`'s segments into batches that each fit the L2 working-set budget.
 *
 * Every batch is planned before any runs so the device buffers can be sized once for the largest.
 * Sizing them per batch would mean a fresh allocation and copy almost every time, since the
 * size-sorted order makes the widths only grow.
 */
inline auto plan_batches(const std::vector<segment>& segments,
                         std::vector<int64_t>&& order,
                         int64_t dim,
                         bool self_join) -> batch_plan
{
  batch_plan out;
  out.order = std::move(order);

  // Increasing size order keeps each batch's padded width close to its members' true sizes.
  // Padding waste is quadratic in the width.
  std::sort(out.order.begin(), out.order.end(), [&](int64_t a, int64_t b) {
    return segments[a].cand_count < segments[b].cand_count;
  });

  // Budget the in-flight operands and score matrices against L2 so the intermediate matrices
  // stay out of DRAM.
  const int64_t budget = l2_cache_bytes();
  const auto pad_up    = [](int64_t n) {
    return (n + kPadMultiple - 1) / kPadMultiple * kPadMultiple;
  };

  for (size_t pos = 0; pos < out.order.size();) {
    // Grow the batch while the operands and score matrices still fit the budget, recomputing the
    // padded extents as we extend.
    int64_t query_pad = 0;
    int64_t cand_pad  = 0;
    size_t end        = pos;
    while (end < out.order.size()) {
      const auto& s    = segments[out.order[end]];
      const int64_t qp = pad_up(std::max(query_pad, s.query_count));
      const int64_t cp = pad_up(std::max(cand_pad, s.cand_count));
      const int64_t n  = static_cast<int64_t>(end - pos) + 1;
      // Operands and scores are both live across the GEMM, so they add rather than max, and
      // every term is linear in the batch size. A self-join gathers one slab and uses it twice.
      // The operand term is what keeps high-dimensional data inside the budget: it overtakes the
      // score term once `dim` exceeds roughly `1.5 * cp`.
      const int64_t cost =
        n * (qp * cp * static_cast<int64_t>(sizeof(float)) +
             (self_join ? qp : qp + cp) * dim * static_cast<int64_t>(sizeof(half)));
      if (end > pos && cost > budget) { break; }
      query_pad = qp;
      cand_pad  = cp;
      end++;
    }
    const int64_t nb   = static_cast<int64_t>(end - pos);
    const int64_t rows = nb * query_pad;
    out.query_slab     = std::max(out.query_slab, rows * dim);
    out.dots           = std::max(out.dots, rows * cand_pad);
    if (!self_join) { out.cand_slab = std::max(out.cand_slab, nb * cand_pad * dim); }
    out.batches.push_back(batch{pos, end, query_pad, cand_pad});
    pos = end;
  }
  return out;
}

/**
 * For every query row in every segment, finds the `k` nearest candidates within that row's own
 * segment.
 *
 * Output order follows `query_ids`, not the order segments happened to be processed in. Rows with
 * fewer than `k` usable candidates get `kInvalidId` in the surplus slots.
 *
 * @param[in] dataset half-precision scoring operand [n_rows, dim]
 * @param[in] norms fp32 squared row norms of `dataset` [n_rows]
 * @param[in] query_ids concatenated query row ids across all segments
 * @param[in] cand_ids concatenated candidate row ids across all segments
 * @param[in] segments host-side segment descriptors
 * @param[in] k neighbors per query row
 * @param[in] exclude_self drop candidates whose id equals the query's own id
 * @param[out] out_ids [total_queries, k] global candidate ids
 * @param[out] out_dists optional [total_queries, k] fp32 squared L2 distances, in the expanded
 *             form described on `fused_topk_kernel`. Not computed when omitted.
 * @param[out] out_cand_slots optional [total_queries, k] position of the selected candidate
 *             within `cand_ids`. Two segments can name the same point as a candidate, so this
 *             is what distinguishes "leader X of bucket A" from "leader X of bucket B".
 */
inline void segmented_topk(
  raft::resources const& res,
  raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
  raft::device_vector_view<const float, int64_t> norms,
  raft::device_vector_view<const id_t, int64_t> query_ids,
  raft::device_vector_view<const id_t, int64_t> cand_ids,
  const std::vector<segment>& segments,
  int64_t k,
  bool exclude_self,
  raft::device_matrix_view<id_t, int64_t, raft::row_major> out_ids,
  std::optional<raft::device_matrix_view<float, int64_t, raft::row_major>> out_dists,
  std::optional<raft::device_matrix_view<id_t, int64_t, raft::row_major>> out_cand_slots =
    std::nullopt)
{
  if (segments.empty()) { return; }
  const int64_t dim = dataset.extent(1);
  auto stream       = raft::resource::get_cuda_stream(res).get();
  RAFT_EXPECTS(cand_ids.extent(0) < static_cast<int64_t>(kInvalidId),
               "gpipnn: too many candidate slots to address with a 32-bit id");
  // `check_params` bounds the fanouts and `leaf_degree`, which are the only sources of `k`, so a
  // violation here means a caller inside the builder, not a user.
  RAFT_EXPECTS(k >= 1 && k <= kMaxSelectK,
               "gpipnn: segmented_topk supports k in [1, %zu] (got %zu)",
               static_cast<size_t>(kMaxSelectK),
               static_cast<size_t>(k));

  // Both paths need the two id arrays to be one array, which is what `leaf_pick_range` passes.
  // Detected here rather than asserted by the caller, so the equality stays a checked fact.
  const bool same_ids = query_ids.data_handle() == cand_ids.data_handle();

  // The batched path additionally needs *every* segment to score one range against itself, since
  // it gathers a single slab and hands it to the GEMM as both operands. The fused path does not:
  // it takes segments one at a time, so it tests the same ranges per segment and a leaf that had
  // to be chunked costs only itself.
  bool self_join = same_ids;
  for (const auto& s : segments) {
    self_join = self_join && s.query_begin == s.cand_begin && s.query_count == s.cand_count;
  }

  // Segments the fused leaf kernel can take leave the batched path entirely: it works from the
  // dataset itself and shares none of the buffers the batch planner budgets. Partitioning up
  // front keeps the two paths from having to agree about anything.
  auto fused  = plan_fused_segments(dataset.data_handle(), dim, k, same_ids, segments);
  auto& order = fused.batched;
  if (!fused.fused.empty()) {
    launch_fused_leaf_topk(fused.block_m,
                           fused.fused,
                           dataset.data_handle(),
                           norms.data_handle(),
                           query_ids.data_handle(),
                           cand_ids.data_handle(),
                           dim,
                           k,
                           exclude_self,
                           out_ids.data_handle(),
                           out_dists.has_value() ? out_dists->data_handle() : nullptr,
                           out_cand_slots.has_value() ? out_cand_slots->data_handle() : nullptr,
                           stream);
  }
  if (order.empty()) { return; }

  const batch_plan plan = plan_batches(segments, std::move(order), dim, self_join);

  // Every batch's descriptors are a contiguous window of the size-sorted order, so the whole array
  // goes up in one transfer and each batch is handed a pointer into it.
  std::vector<segment_dev> h_segs(plan.order.size());
  for (size_t i = 0; i < plan.order.size(); i++) {
    h_segs[i] = to_device(segments[plan.order[i]]);
  }
  rmm::device_uvector<segment_dev> d_segs(h_segs.size(), stream);
  raft::update_device(d_segs.data(), h_segs.data(), h_segs.size(), stream);

  rmm::device_uvector<half> query_slab(plan.query_slab, stream);
  rmm::device_uvector<half> cand_slab(plan.cand_slab, stream);
  rmm::device_uvector<float> dots(plan.dots, stream);

  // Every batch scores into the same allocation, so one window covers the whole loop.
  const persisting_l2_window dots_window(
    stream, dots.data(), plan.dots * static_cast<int64_t>(sizeof(float)));

  for (const auto& b : plan.batches) {
    const int64_t query_pad  = b.query_pad;
    const int64_t cand_pad   = b.cand_pad;
    const int64_t nb         = static_cast<int64_t>(b.end - b.begin);
    const int64_t rows       = nb * query_pad;
    const segment_dev* b_seg = d_segs.data() + b.begin;
    launch_gather_padded(dataset.data_handle(),
                         query_ids.data_handle(),
                         b_seg,
                         true,
                         query_pad,
                         dim,
                         rows,
                         query_slab.data(),
                         stream);
    // A self-join's two slabs are the same rows in the same padded layout, so the second gather
    // would write the bytes the first just wrote.
    const half* cand_operand = query_slab.data();
    if (!self_join) {
      launch_gather_padded(dataset.data_handle(),
                           cand_ids.data_handle(),
                           b_seg,
                           false,
                           cand_pad,
                           dim,
                           nb * cand_pad,
                           cand_slab.data(),
                           stream);
      cand_operand = cand_slab.data();
    }

    // Row-major out[q, c] = <query[q], cand[c]> expressed column-major, following the same
    // operand swap cuVS uses in unfused_distance_nn.
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    auto cublas_h     = raft::resource::get_cublas_handle(res);
    RAFT_CUBLAS_TRY(cublasSetStream(cublas_h, stream));
    RAFT_CUBLAS_TRY(cublasGemmStridedBatchedEx(cublas_h,
                                               CUBLAS_OP_T,
                                               CUBLAS_OP_N,
                                               static_cast<int>(cand_pad),
                                               static_cast<int>(query_pad),
                                               static_cast<int>(dim),
                                               &alpha,
                                               cand_operand,
                                               CUDA_R_16F,
                                               static_cast<int>(dim),
                                               (self_join ? query_pad : cand_pad) * dim,
                                               query_slab.data(),
                                               CUDA_R_16F,
                                               static_cast<int>(dim),
                                               query_pad * dim,
                                               &beta,
                                               dots.data(),
                                               CUDA_R_32F,
                                               static_cast<int>(cand_pad),
                                               query_pad * cand_pad,
                                               static_cast<int>(nb),
                                               CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT));

    // Select at the smallest instantiated list length that covers `k`; the per-lane register
    // lists make the length a template parameter. `check_params` bounds `k` at `kMaxSelectK`.
    constexpr int kWarpsPerBlock = 8;
    const int threads            = kWarpsPerBlock * 32;
    const int64_t blocks         = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    auto select                  = [&](auto tag) {
      fused_topk_kernel<decltype(tag)::value><<<blocks, threads, 0, stream>>>(
        dots.data(),
        norms.data_handle(),
        query_ids.data_handle(),
        cand_ids.data_handle(),
        b_seg,
        rows,
        query_pad,
        cand_pad,
        exclude_self,
        k,
        out_ids.data_handle(),
        out_dists.has_value() ? out_dists->data_handle() : nullptr,
        out_cand_slots.has_value() ? out_cand_slots->data_handle() : nullptr);
    };
    if (k <= 1) {
      select(std::integral_constant<int, 1>{});
    } else if (k <= 2) {
      select(std::integral_constant<int, 2>{});
    } else if (k <= 4) {
      select(std::integral_constant<int, 4>{});
    } else if (k <= 8) {
      select(std::integral_constant<int, 8>{});
    } else if (k <= 16) {
      select(std::integral_constant<int, 16>{});
    } else {
      select(std::integral_constant<int, 32>{});
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

}  // namespace cuvs::neighbors::experimental::gpipnn::detail
