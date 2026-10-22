/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"
#include "fused_leaf_topk.cuh"

#include <raft/core/device_mdspan.hpp>
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
 * The scalar kernel gives each slab row one thread per `half`, which is coalesced but latency
 * bound: a warp has only 64 bytes in flight, and at low `dim` there is no second loop iteration
 * to hide the round trip with. Indexing flat over 16-byte vectors instead decouples the block
 * shape from `dim`, so a warp moves 512 contiguous bytes at any dimension. The two divisions per
 * element are by runtime values, but this is bandwidth bound, not issue bound.
 *
 * Byte-for-byte identical to the scalar kernel; `__float2half(0.0f)` is a zero bit pattern.
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

/** Whether `dim` and the two buffers admit the 16-byte-per-thread gather. */
inline auto gather_can_vectorize(const half* dataset, const half* slab, int64_t dim) -> bool
{
  constexpr auto kAlign = static_cast<uintptr_t>(sizeof(uint4));
  return dim % kHalfPerVec == 0 && reinterpret_cast<uintptr_t>(dataset) % kAlign == 0 &&
         reinterpret_cast<uintptr_t>(slab) % kAlign == 0;
}

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
  if (gather_can_vectorize(dataset, slab, dim)) {
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
 * Batched extents are rounded up to a multiple of this.
 *
 * Unrounded, a batch's width is the largest segment in it, so the score matrix's leading
 * dimension and batch stride are almost never 16-byte aligned, so we round up to a multiple of 8.
 * The added rows and columns are inert: the gather zero-fills every slab row past its segment's
 * count, and selection visits only the columns its own segment has. Rounding depends on nothing
 * but the width, which keeps a self-join's two extents equal.
 */
inline constexpr int64_t kPadMultiple = 8;

/** Rounds a padded extent up to `kPadMultiple`. */
inline auto pad_up(int64_t n) -> int64_t
{
  return (n + kPadMultiple - 1) / kPadMultiple * kPadMultiple;
}

/**
 * Reduces one row of the dot-product matrix straight to its `k` nearest candidates.
 *
 * Scores are never materialized: only `leaf_degree` of as many as `max_leaf_size` values per row
 * survive. Ordering uses one unsigned 64-bit key per candidate -- order-preserving score in the high half,
 * the candidate's column in the low half -- so an unsigned minimum is exactly "smallest score,
 * then smallest index" and ties resolve identically on every run and launch shape.
 *
 * One warp per query row. Each lane sweeps a strided slice, so reads coalesce, and keeps its own
 * `K` best in registers. The lists reduced to the global best 'k' by a butterfly minimum.
 *
 * The stored distance is the expanded score `q_norm + c_norm - 2*dot` rather than an unexpanded
 * difference, so it is fine where norms are comparable to neighbour distances and loses precision
 * for data far from the origin.
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
    best[i] = kEmptyKey;
  }

  // Columns past the segment's candidate count are batch padding and are not visited. A masked
  // self-match keeps its +inf score: it can still be picked when a segment has no more candidates
  // than ranks, and `emit_rank` discards it there.
  for (int64_t c = lane; c < n_cand; c += warpSize) {
    const id_t c_id   = cand_ids[s.cand_begin + c];
    const bool masked = exclude_self && c_id == q_id;
    const float score = masked ? kMaskedScore : q_norm + norms[c_id] - 2.0f * dot_row[c];
    uint64_t x        = (static_cast<uint64_t>(order_preserving_u32(score)) << 32) |
                 static_cast<uint64_t>(static_cast<uint32_t>(c) ^ salt);
    if (x < best[K - 1]) {
      // Insert into the ascending list and drop what falls off the end.
#pragma unroll
      for (int i = 0; i < K; i++) {
        const uint64_t b = best[i];
        const bool take  = x < b;
        best[i]          = take ? x : b;
        x                = take ? b : x;
      }
    }
  }

  for (int64_t j = 0; j < k; j++) {
    uint64_t m = best[0];
#pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      const uint64_t v = __shfl_xor_sync(0xffffffffu, m, off);
      m                = v < m ? v : m;
    }
    if (lane == 0) {
      // No real key collides with `kEmptyKey`: the high half tops out at +inf's 0xff800000.
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
    if (m != kEmptyKey && best[0] == m) {
#pragma unroll
      for (int i = 0; i < K - 1; i++) {
        best[i] = best[i + 1];
      }
      best[K - 1] = kEmptyKey;
    }
  }
}

/** Launches `fused_topk_kernel` at the smallest instantiated list length that covers `k`. */
inline void launch_fused_topk(int64_t rows,
                              int64_t query_pad,
                              int64_t cand_pad,
                              int64_t k,
                              bool exclude_self,
                              const float* dots,
                              const float* norms,
                              const id_t* query_ids,
                              const id_t* cand_ids,
                              const segment_dev* segs,
                              id_t* out_ids,
                              float* out_dists,
                              id_t* out_cand_slots,
                              cudaStream_t stream)
{
  constexpr int kWarpsPerBlock = 8;
  const int threads            = kWarpsPerBlock * 32;
  const int64_t blocks         = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  auto go                      = [&](auto tag) {
    fused_topk_kernel<decltype(tag)::value><<<blocks, threads, 0, stream>>>(dots,
                                                                            norms,
                                                                            query_ids,
                                                                            cand_ids,
                                                                            segs,
                                                                            rows,
                                                                            query_pad,
                                                                            cand_pad,
                                                                            exclude_self,
                                                                            k,
                                                                            out_ids,
                                                                            out_dists,
                                                                            out_cand_slots);
  };
  if (k <= 1) {
    go(std::integral_constant<int, 1>{});
  } else if (k <= 2) {
    go(std::integral_constant<int, 2>{});
  } else if (k <= 4) {
    go(std::integral_constant<int, 4>{});
  } else if (k <= 8) {
    go(std::integral_constant<int, 8>{});
  } else if (k <= 16) {
    go(std::integral_constant<int, 16>{});
  } else {
    go(std::integral_constant<int, 32>{});
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Divisor applied to L2 before it becomes the working-set budget.
 *
 * Operands and scores count against the same budget, so at 1 the GEMM streams its slabs through
 * the very cache the score matrix needs to survive in and selection finds the early rows evicted.
 * Dividing leaves the scores room between the two launches, paid for in more and smaller batches.
 */
inline constexpr int64_t kL2BudgetDiv = 2;

/**
 * Marks a buffer as persisting in L2 for as long as the guard lives.
 *
 * `dots` is the one buffer worth holding: the GEMM writes it and selection reads it straight back,
 * while the operand slabs are streamed once each. Complementary to `kL2BudgetDiv`, which buys the
 * same residency by shrinking batches -- this one pays where the score term dominates the working
 * set (low dimension).
 *
 * Destruction undoes only the stream attribute. The carve-out limit is left set and
 * `cudaCtxResetPersistingL2Cache` is not called, because `segmented_topk` is asynchronous and the
 * guard dies while its kernels are still in flight.
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
    RAFT_CUDA_TRY(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, static_cast<size_t>(window)));

    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr  = const_cast<void*>(ptr);
    attr.accessPolicyWindow.num_bytes = static_cast<size_t>(window);
    // Window and carve-out are the same size, so the ratio has nothing to ration.
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp  = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    RAFT_CUDA_TRY(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
    active_ = true;
  }

  ~persisting_l2_window()
  {
    if (!active_) { return; }
    // Unchecked: this can run while an exception from the batch loop propagates, and losing it to
    // report a failed cache hint would be a poor trade.
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.num_bytes = 0;
    cudaStreamSetAttribute(stream_, cudaStreamAttributeAccessPolicyWindow, &attr);
  }

  persisting_l2_window(const persisting_l2_window&)                    = delete;
  persisting_l2_window(persisting_l2_window&&)                         = delete;
  auto operator=(const persisting_l2_window&) -> persisting_l2_window& = delete;
  auto operator=(persisting_l2_window&&) -> persisting_l2_window&      = delete;

 private:
  cudaStream_t stream_ = nullptr;
  bool active_         = false;
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

/** Multiprocessor count, for sizing grids that would otherwise leave the device idle. */
inline auto sm_count() -> int64_t
{
  int device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  int sms = 0;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  return sms > 0 ? static_cast<int64_t>(sms) : 64;
}

/**
 * Bytes of device memory a batch of `n_segments` segments keeps in flight, at padded extents
 * `query_pad` by `cand_pad` over `dim` dimensions.
 *
 * Every array the batch loop allocates is `n_segments` times a per-segment term built from the
 * *padded* extents, so the whole cost is linear in the batch size:
 *
 *   - the gathered fp16 operands, `query_pad * dim` and `cand_pad * dim` elements. Both are live
 *     at once because the GEMM reads them together, so they add rather than max. A self-join
 *     gathers one slab and hands it to the GEMM twice, so it pays for one.
 *   - the fp32 dot products, `query_pad * cand_pad`. Selection reads them in place, so there is
 *     no separate score matrix.
 *
 * The operand term is what keeps high-dimensional data inside the budget. Scores grow with
 * `cand_pad` and operands with `dim`, so the slabs overtake the scores once `dim` exceeds roughly
 * `1.5 * cand_pad`.
 */
inline auto batch_working_bytes(
  int64_t n_segments, int64_t query_pad, int64_t cand_pad, int64_t dim, bool self_join) -> int64_t
{
  const int64_t score_bytes = query_pad * cand_pad * static_cast<int64_t>(sizeof(float));
  const int64_t operand_bytes =
    (self_join ? query_pad : query_pad + cand_pad) * dim * static_cast<int64_t>(sizeof(half));
  return n_segments * (score_bytes + operand_bytes);
}

/**
 * Largest number of query rows one segment scoring against `cand_count` candidates can carry
 * with its own working set still inside `budget`.
 *
 * This inverts `batch_working_bytes` at `n_segments == 1`. The candidate slab is fixed once the
 * candidate range is known and every other term is linear in the query count, so the bound is
 *
 *   q * (sizeof(float) * cand_count + sizeof(half) * dim)
 *     + sizeof(half) * cand_count * dim  <=  budget.
 *
 * Both slabs are counted even where only one will be gathered: whether a call is a self-join
 * depends on whether this bound leaves its queries in a single chunk. 
 * Never returns less than 1, so a segment whose candidate slab alone overruns the budget still
 * makes progress a row at a time.
 */
inline auto max_query_chunk(int64_t budget, int64_t cand_count, int64_t dim) -> int64_t
{
  const int64_t per_query =
    cand_count * static_cast<int64_t>(sizeof(float)) + dim * static_cast<int64_t>(sizeof(half));
  const int64_t cand_fixed = cand_count * dim * static_cast<int64_t>(sizeof(half));
  return std::max<int64_t>(1, (budget - cand_fixed) / per_query);
}

/**
 * For every query row in every segment, finds the `k` nearest candidates within that row's own
 * segment.
 *
 * Results are written at the row's position in the global query ordering, so the caller sees
 * outputs in the same order as `query_ids` regardless of the order segments were processed in.
 * Rows with fewer than `k` usable candidates get `kInvalidId` in the surplus slots.
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
 *             form described on `fused_topk_kernel`. Left out by callers that only need the ids,
 *             in which case none are computed.
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

  // A call whose every segment scores one id range against itself gathers a single slab and hands
  // it to the GEMM as both operands -- `leaf_pick` is exactly that. Detected here rather than
  // asserted by the caller, so the equality stays a checked fact.
  bool self_join = query_ids.data_handle() == cand_ids.data_handle();
  for (const auto& s : segments) {
    self_join = self_join && s.query_begin == s.cand_begin && s.query_count == s.cand_count;
  }

  // Segments the fused leaf kernel can take leave the batched path entirely: it works from the
  // dataset itself and shares none of the buffers the batch planner budgets. Partitioning up
  // front keeps the two paths from having to agree about anything.
  const bool fused_leaf = fused_leaf_usable(dim, k, self_join);
  std::vector<segment> fused_segs;
  std::vector<int64_t> order;
  order.reserve(segments.size());
  for (size_t i = 0; i < segments.size(); i++) {
    if (fused_leaf && fused_leaf_eligible(segments[i])) {
      fused_segs.push_back(segments[i]);
    } else {
      order.push_back(static_cast<int64_t>(i));
    }
  }
  if (!fused_segs.empty()) {
    launch_fused_leaf_topk(fused_segs,
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

  // Increasing size order keeps each batch's padded width close to its members' true sizes.
  // Padding waste is quadratic in the width.
  std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return segments[a].cand_count < segments[b].cand_count;
  });

  // Budget the in-flight operands and score matrices against L2 so the intermediate matrices
  // stay out of DRAM.
  const int64_t budget = l2_cache_bytes();

  // Plan every batch before running any, so the device buffers can be sized once for the largest.
  // Resizing per batch means a fresh RMM allocation and copy almost every time, since the
  // size-sorted order makes the widths only grow.
  struct batch {
    size_t begin;
    size_t end;
    int64_t query_pad;
    int64_t cand_pad;
  };
  std::vector<batch> plan;
  for (size_t pos = 0; pos < order.size();) {
    // Grow the batch while the operands and score matrices still fit the budget, recomputing the
    // padded extents as we extend.
    int64_t query_pad = 0;
    int64_t cand_pad  = 0;
    size_t end        = pos;
    while (end < order.size()) {
      const auto& s      = segments[order[end]];
      const int64_t qp   = pad_up(std::max(query_pad, s.query_count));
      const int64_t cp   = pad_up(std::max(cand_pad, s.cand_count));
      const int64_t n    = static_cast<int64_t>(end - pos) + 1;
      const int64_t cost = batch_working_bytes(n, qp, cp, dim, self_join);
      if (end > pos && cost > budget) { break; }
      query_pad = qp;
      cand_pad  = cp;
      end++;
    }
    plan.push_back(batch{pos, end, query_pad, cand_pad});
    pos = end;
  }

  int64_t max_rows = 0, max_slab = 0, max_cand_slab = 0, max_dots = 0;
  for (const auto& b : plan) {
    const int64_t nb   = static_cast<int64_t>(b.end - b.begin);
    const int64_t rows = nb * b.query_pad;
    max_rows           = std::max(max_rows, rows);
    max_slab           = std::max(max_slab, rows * dim);
    max_dots           = std::max(max_dots, rows * b.cand_pad);
    if (!self_join) { max_cand_slab = std::max(max_cand_slab, nb * b.cand_pad * dim); }
  }

  // Every batch's descriptors are a contiguous window of the size-sorted order, so the whole array
  // goes up in one transfer and each batch is handed a pointer into it.
  std::vector<segment_dev> h_segs(order.size());
  for (size_t i = 0; i < order.size(); i++) {
    const auto& s = segments[order[i]];
    h_segs[i]     = segment_dev{s.query_begin,
                            s.cand_begin,
                            static_cast<int32_t>(s.query_count),
                            static_cast<int32_t>(s.cand_count)};
  }
  rmm::device_uvector<segment_dev> d_segs(order.size(), stream);
  raft::update_device(d_segs.data(), h_segs.data(), order.size(), stream);

  rmm::device_uvector<half> query_slab(max_slab, stream);
  rmm::device_uvector<half> cand_slab(max_cand_slab, stream);
  rmm::device_uvector<float> dots(max_dots, stream);

  // Every batch scores into the same allocation, so one window covers the whole loop.
  const persisting_l2_window dots_window(
    stream, dots.data(), max_dots * static_cast<int64_t>(sizeof(float)));

  for (const auto& b : plan) {
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

    launch_fused_topk(rows,
                      query_pad,
                      cand_pad,
                      k,
                      exclude_self,
                      dots.data(),
                      norms.data_handle(),
                      query_ids.data_handle(),
                      cand_ids.data_handle(),
                      b_seg,
                      out_ids.data_handle(),
                      out_dists.has_value() ? out_dists->data_handle() : nullptr,
                      out_cand_slots.has_value() ? out_cand_slots->data_handle() : nullptr,
                      stream);
  }
}

}  // namespace cuvs::neighbors::experimental::gpipnn::detail
