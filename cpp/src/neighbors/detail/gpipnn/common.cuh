/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/gpipnn.hpp>

#include <raft/core/detail/macros.hpp>
#include <raft/core/error.hpp>
#include <raft/core/logger.hpp>
#include <raft/linalg/norm.cuh>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/util_type.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace cuvs::neighbors::experimental::gpipnn::detail {

/** Index type used for point ids and graph entries throughout the build. */
using id_t = uint32_t;

/** Sentinel marking an unused slot. */
inline constexpr id_t kInvalidId = static_cast<id_t>(-1);

/**
 * Maps a float to a uint32_t whose unsigned ordering matches the float's ordering, for the low
 * half of a 64-bit radix sort key.
 */
__host__ __device__ inline auto order_preserving_u32(float x) -> uint32_t
{
  uint32_t bits;
#ifdef __CUDA_ARCH__
  bits = __float_as_uint(x);
#else
  static_assert(sizeof(float) == sizeof(uint32_t));
  __builtin_memcpy(&bits, &x, sizeof(bits));
#endif
  // For positive floats the sign bit is 0 and the remaining bits already sort correctly, so
  // setting the sign bit preserves order. For negative floats the ordering is reversed, so all
  // bits are flipped.
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

/** Inverse of order_preserving_u32, over the whole float range. */
__host__ __device__ inline auto from_order_preserving_u32(uint32_t key) -> float
{
  // The high bit says which branch produced the key.
  const uint32_t bits = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
#ifdef __CUDA_ARCH__
  return __uint_as_float(bits);
#else
  float x;
  __builtin_memcpy(&x, &bits, sizeof(x));
  return x;
#endif
}

/** Packs an owner id and a distance into a single sortable key: owner major, distance minor. */
__host__ __device__ inline auto make_sort_key(id_t owner, float dist) -> uint64_t
{
  return (static_cast<uint64_t>(owner) << 32) | static_cast<uint64_t>(order_preserving_u32(dist));
}

/** Get the distance from a key built by make_sort_key. */
__host__ __device__ inline auto key_dist(uint64_t key) -> float
{
  return from_order_preserving_u32(static_cast<uint32_t>(key & 0xffffffffu));
}

/** Sentinel value to mark an empty slot. Sorts after every real value. */
inline constexpr uint64_t kInvalidKey = ~uint64_t{0};

/** Distance score to mark an empty slot. Sorts after every real value. */
inline constexpr float kMaskedScore = __builtin_huge_valf();

/**
 * Largest `k` `segmented_topk` can select, set by the per-lane register lists in
 * `fused_topk_kernel`. Every `k` the builder asks for is a fanout or `leaf_degree`, so
 * `check_params` bounds those rather than leaving the limit to be discovered at launch.
 */
inline constexpr int64_t kMaxSelectK = 32;

/** One independent top-K subproblem: a block of query rows scored against a block of candidates. */
struct segment {
  int64_t query_begin;
  int64_t query_count;
  int64_t cand_begin;
  int64_t cand_count;
};

/** Device-side mirror of `segment`, with the counts narrowed to what a single launch can cover. */
struct segment_dev {
  int64_t query_begin;
  int64_t cand_begin;
  int32_t query_count;
  int32_t cand_count;
};

/** Narrows a `segment` into the form the kernels read. */
inline auto to_device(const segment& s) -> segment_dev
{
  return segment_dev{s.query_begin,
                     s.cand_begin,
                     static_cast<int32_t>(s.query_count),
                     static_cast<int32_t>(s.cand_count)};
}

/**
 * Per-row tie-break permutation of the candidate column, used by topk selection to avoid
 * duplicate candidates from the same leaf.
 *
 * Uses a hash function using the golden-ratio constant and finalizers to produce a unique
 * value for each query. The result is XOR-ed with the candidate column to break ties.
 */
__host__ __device__ inline auto row_salt(id_t q_id) -> uint32_t
{
  uint32_t salt = q_id * 0x9e3779b9u + 0x85ebca6bu;  // golden-ratio constant
  salt ^= salt >> 16;
  salt *= 0x7feb352du;
  salt ^= salt >> 15;
  return salt;
}

/**
 * Writes one rank's final id, distance and candidate slot, given the key that won it.
 */
__device__ inline void emit_rank(uint64_t key,
                                 uint32_t salt,
                                 id_t q_id,
                                 const id_t* __restrict__ cand_ids,
                                 int64_t cand_begin,
                                 bool exclude_self,
                                 int64_t out_slot,
                                 id_t* __restrict__ out_ids,
                                 float* __restrict__ out_dists,
                                 id_t* __restrict__ out_cand_slots)
{
  id_t global_id = kInvalidId;
  id_t cand_slot = kInvalidId;
  float dist     = kMaskedScore;

  if (key != kInvalidKey) {
    const uint32_t local_c = static_cast<uint32_t>(key & 0xffffffffu) ^ salt;
    const id_t c_id        = cand_ids[cand_begin + local_c];
    if (!(exclude_self && c_id == q_id)) {
      cand_slot = static_cast<id_t>(cand_begin + local_c);
      global_id = c_id;
      // Clamps the small negatives the expansion rounds to for near-identical points. Applied
      // after packing so the ordering is untouched.
      dist = fmaxf(from_order_preserving_u32(static_cast<uint32_t>(key >> 32)), 0.0f);
    }
  }

  out_ids[out_slot] = global_id;
  if (out_dists != nullptr) { out_dists[out_slot] = dist; }
  if (out_cand_slots != nullptr) { out_cand_slots[out_slot] = cand_slot; }
}

/**
 * Reduction operator returning the larger of two magnitudes, treating non-finite values as 0.
 *
 * Dropping them keeps a single stray infinity or NaN in the caller's data from swallowing the
 * maximum and forcing a scale that crushes everything else to zero. The sanitizing step is
 * idempotent, so it is safe to apply on partial results as the reduction combines them.
 */
struct finite_max_abs_op {
  __host__ __device__ auto operator()(float a, float b) const -> float
  {
    const float x = isfinite(a) ? fabsf(a) : 0.0f;
    const float y = isfinite(b) ? fabsf(b) : 0.0f;
    return fmaxf(x, y);
  }
};

/** Largest finite magnitude in `[data, data + n)`, or 0 when there is none. */
inline auto finite_max_abs(const float* data, int64_t n, cudaStream_t stream) -> float
{
  rmm::device_uvector<float> out(1, stream);
  size_t bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(
    nullptr, bytes, data, out.data(), n, finite_max_abs_op{}, 0.0f, stream));
  rmm::device_uvector<char> workspace(bytes, stream);
  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(
    workspace.data(), bytes, data, out.data(), n, finite_max_abs_op{}, 0.0f, stream));

  float max_abs = 0.0f;
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(&max_abs, out.data(), sizeof(float), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return max_abs;
}

/**
 * Power-of-two factor that moves a dataset peaking at `max_abs` into half's useful range, or 1
 * when it is already there.
 *
 * Scaling every element by one positive constant multiplies every squared L2 distance by its
 * square, which leaves the neighbour ordering untouched, and DirectEdgePrune compares ratios of
 * distances, so the occlusion decisions are unaffected too. Restricting the factor to a power of
 * two makes the fp32 accumulation scale exactly, so data already inside the band is left
 * bit-for-bit as it was.
 */
inline auto half_cast_scale(float max_abs) -> float
{
  // Past the upper edge `__float2half` returns infinity rather than saturating, and the expanded
  // distance then evaluates to `inf - inf`. Below the lower edge the largest values sit close
  // enough to half's smallest normal (2^-14) that the rest of each row falls subnormal and loses
  // mantissa bits.
  constexpr float kHi = 32768.0f;       // 2^15
  constexpr float kLo = 0.0009765625f;  // 2^-10
  if (!(max_abs > 0.0f) || (max_abs >= kLo && max_abs <= kHi)) { return 1.0f; }

  int exp = 0;
  std::frexp(max_abs, &exp);  // max_abs lies in [2^(exp-1), 2^exp)
  // Puts max_abs in (2^14, 2^15]. The clamp keeps the factor itself representable when the whole
  // dataset sits near the bottom of fp32, where the unclamped exponent would overflow to inf.
  return std::ldexp(1.0f, std::min(15 - exp, 127));
}

/**
 * Casts a buffer of `T` to half, rescaling by a shared positive factor on the way. Exact for
 * int8 and uint8: half represents every integer up to 2048.
 *
 * `scale` comes from `half_cast_scale`. Multiplying by the default of 1 is exact for every
 * input, so callers that do not need rescaling get the plain conversion.
 */
template <typename T>
RAFT_KERNEL cast_to_half_kernel(const T* __restrict__ src,
                                half* __restrict__ dst,
                                int64_t n,
                                float scale = 1.0f)
{
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) { dst[i] = __float2half(static_cast<float>(src[i]) * scale); }
}

/**
 * Accumulator width for `normalize_rows_to_half_kernel`.
 *
 * Only a float row can push the squared norm out of fp32: squaring halves the exponent headroom,
 * so magnitudes past ~1e18 saturate the sum to inf and magnitudes under ~1e-23 round it to zero,
 * both of which land on the zero-row branch and silently flatten a valid direction to all zeros.
 * The narrower types cannot reach either end -- the largest half squares to 4.3e9, so overflowing
 * would take a row of 1e29 elements, and the smallest nonzero square is 3.6e-15 for half and 1 for
 * the integer types, decades clear of underflow. They therefore stay on fp32 bit-for-bit.
 */
template <typename T>
using norm_acc_t = std::conditional_t<std::is_same_v<T, float>, double, float>;

/** `rsqrt` per accumulator width, keeping the fp32 path on the intrinsic it already used. */
__device__ inline auto norm_rsqrt(float x) -> float { return rsqrtf(x); }
__device__ inline auto norm_rsqrt(double x) -> double { return rsqrt(x); }

/** Applies the row's reciprocal norm to one element and rounds it to half, once. */
__device__ inline auto norm_to_half(float v, float s) -> half { return __float2half(v * s); }
__device__ inline auto norm_to_half(double v, double s) -> half { return __double2half(v * s); }

/**
 * Casts a buffer of `T` to half with each row scaled to unit L2 norm.
 * This normalization enables cosine support and can use the remaining code from the L2 path.
 *
 * One block per row, so `dim` does not bound the block size.
 */
template <typename T>
RAFT_KERNEL normalize_rows_to_half_kernel(const T* __restrict__ src,
                                          half* __restrict__ dst,
                                          int64_t n_rows,
                                          int64_t dim)
{
  using acc_t       = norm_acc_t<T>;
  const int64_t row = blockIdx.x;
  if (row >= n_rows) { return; }
  const T* in = src + row * dim;
  half* out   = dst + row * dim;

  acc_t acc = acc_t{0};
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const acc_t v = static_cast<acc_t>(in[i]);
    acc           = fma(v, v, acc);
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
  }
  __shared__ acc_t partial[32];
  __shared__ acc_t scale;
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  if (lane == 0) { partial[warp] = acc; }
  __syncthreads();
  if (warp == 0) {
    const int n_warps = static_cast<int>(blockDim.x) / warpSize;
    acc               = lane < n_warps ? partial[lane] : acc_t{0};
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    // Zero for a zero row, so the stores below stay finite without a second branch.
    if (lane == 0) { scale = acc > acc_t{0} ? norm_rsqrt(acc) : acc_t{0}; }
  }
  __syncthreads();

  // The scale stays at accumulator width: a row of float subnormals normalizes by ~1e43, which
  // only fp32 inputs can produce and which fp32 cannot hold.
  const acc_t s = scale;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    out[i] = norm_to_half(static_cast<acc_t>(in[i]), s);
  }
}

/**
 * Computes fp32 squared row norms of a half matrix.
 *
 * Taken from the half data rather than the caller's fp32 original so the expanded distance stays
 * self-consistent. `raft::sq_op` has a `half` overload that widens before squaring, so the
 * accumulation is fp32 throughout.
 */
inline void row_norms(const half* data, int64_t n_rows, int64_t dim, float* norms, cudaStream_t s)
{
  raft::linalg::rowNorm<raft::linalg::L2Norm, true>(norms, data, dim, n_rows, s);
}

/**
 * Bits needed to represent every value in `[0, max_value]`, capped at 32.
 *
 * Used to narrow the radix-sort end bit: keys whose bits above this point are all zero need no
 * passes over them, which saves a pass or two over an unconditional width.
 */
inline auto key_bits(int64_t max_value) -> int
{
  int bits = 0;
  while (bits < 32 && (int64_t{1} << bits) <= max_value) {
    bits++;
  }
  return bits;
}

/**
 * `cub::DeviceRadixSort::SortPairs` from `keys_in`/`vals_in` into `keys_out`/`vals_out`.
 *
 * Wraps the two CUB calls, once to compute temporary size and once to sort.
 */
template <typename KeyT, typename ValueT>
void radix_sort_pairs(cudaStream_t stream,
                      const KeyT* keys_in,
                      KeyT* keys_out,
                      const ValueT* vals_in,
                      ValueT* vals_out,
                      int64_t n,
                      int end_bit)
{
  size_t bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
    nullptr, bytes, keys_in, keys_out, vals_in, vals_out, n, 0, end_bit, stream));
  rmm::device_uvector<char> workspace(bytes, stream);
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
    workspace.data(), bytes, keys_in, keys_out, vals_in, vals_out, n, 0, end_bit, stream));
}

/**
 * The same sort in place, over CUB's ping-pong buffers.
 *
 * CUB picks which of the two buffers holds the result, so the alternates are moved back into
 * `keys`/`vals` whenever it finished on the far side.
 */
template <typename KeyT, typename ValueT>
void radix_sort_pairs_inplace(cudaStream_t stream,
                              rmm::device_uvector<KeyT>& keys,
                              rmm::device_uvector<ValueT>& vals,
                              int64_t n,
                              int end_bit)
{
  rmm::device_uvector<KeyT> keys_alt(n, stream);
  rmm::device_uvector<ValueT> vals_alt(n, stream);
  cub::DoubleBuffer<KeyT> d_keys(keys.data(), keys_alt.data());
  cub::DoubleBuffer<ValueT> d_vals(vals.data(), vals_alt.data());
  size_t bytes = 0;
  RAFT_CUDA_TRY(
    cub::DeviceRadixSort::SortPairs(nullptr, bytes, d_keys, d_vals, n, 0, end_bit, stream));
  rmm::device_uvector<char> workspace(bytes, stream);
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
    workspace.data(), bytes, d_keys, d_vals, n, 0, end_bit, stream));
  if (d_keys.Current() != keys.data()) { keys = std::move(keys_alt); }
  if (d_vals.Current() != vals.data()) { vals = std::move(vals_alt); }
}

/** Validates user-provided parameters. */
inline void check_params(const index_params& params, int64_t n_rows, int64_t dim)
{
  RAFT_EXPECTS(n_rows > 0, "gpipnn: dataset must be non-empty");
  RAFT_EXPECTS(dim > 0, "gpipnn: dataset dimension must be positive");
  // Cosine is served by normalizing on the way in; see `normalize_rows_to_half_kernel`.
  RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded ||
                 params.metric == cuvs::distance::DistanceType::CosineExpanded,
               "gpipnn: only L2Expanded and CosineExpanded are supported (got %d)",
               static_cast<int>(params.metric));
  // Odd dimensions bypass the CUTLASS half path for a scalar kernel, so reject rather than
  // quietly lose tensor cores.
  RAFT_EXPECTS(dim % 2 == 0,
               "gpipnn: dataset dimension must be even (got %zu); pad the dataset to an even "
               "dimension before building",
               static_cast<size_t>(dim));
  RAFT_EXPECTS(params.graph_degree > 0, "gpipnn: graph_degree must be positive");
  RAFT_EXPECTS(params.max_candidates >= params.graph_degree,
               "gpipnn: max_candidates (%u) must be >= graph_degree (%u)",
               params.max_candidates,
               params.graph_degree);
  RAFT_EXPECTS(params.max_candidates <= kMaxCandidates,
               "gpipnn: max_candidates (%u) exceeds the supported maximum of %u",
               params.max_candidates,
               kMaxCandidates);
  // Legal but pathological: with as many slots as candidates, DirectEdgePrune's saturation pass
  // takes back everything the occlusion rule rejected and the stage becomes a sort by distance.
  if (params.max_candidates == params.graph_degree) {
    RAFT_LOG_WARN(
      "gpipnn: max_candidates == graph_degree (%u) leaves the alpha-occlusion prune with nothing "
      "to discard; use at least 2 * graph_degree to enable it",
      params.graph_degree);
  }
  RAFT_EXPECTS(params.leaf_degree > 0 && params.leaf_degree <= kMaxSelectK,
               "gpipnn: leaf_degree must be in [1, %zu] (got %u)",
               static_cast<size_t>(kMaxSelectK),
               params.leaf_degree);
  RAFT_EXPECTS(params.hash_bits > 0 && params.hash_bits <= 16,
               "gpipnn: hash_bits must be in [1, 16] (got %u)",
               params.hash_bits);
  RAFT_EXPECTS(params.min_leaf_size <= params.max_leaf_size,
               "gpipnn: min_leaf_size (%u) must be <= max_leaf_size (%u)",
               params.min_leaf_size,
               params.max_leaf_size);
  RAFT_EXPECTS(params.max_leaf_size >= 2, "gpipnn: max_leaf_size must be at least 2");
  // Each round's fanout becomes the `k` of a `segmented_topk` call, so both are capped there.
  RAFT_EXPECTS(params.fanout_level0 > 0 && params.fanout_level0 <= kMaxSelectK &&
                 params.fanout_level1 > 0 && params.fanout_level1 <= kMaxSelectK,
               "gpipnn: fanout values must be in [1, %zu] (got %u, %u)",
               static_cast<size_t>(kMaxSelectK),
               params.fanout_level0,
               params.fanout_level1);
  // Each point is assigned to `fanout` of its bucket's leaders, so there must be that many.
  RAFT_EXPECTS(
    params.max_leaders >= params.fanout_level0 && params.max_leaders >= params.fanout_level1,
    "gpipnn: max_leaders (%u) must be >= both fanout values (%u, %u)",
    params.max_leaders,
    params.fanout_level0,
    params.fanout_level1);
  RAFT_EXPECTS(params.leader_fraction > 0.0f && params.leader_fraction <= 1.0f,
               "gpipnn: leader_fraction must be in (0, 1]");
  RAFT_EXPECTS(params.max_rounds >= 2, "gpipnn: max_rounds must be at least 2");
  // Infinity clears the lower bound, and DirectEdgePrune's alpha ramp then never exits: its
  // `cur_alpha <= alpha` guard stays true once `cur_alpha` saturates, so a row holding fewer
  // candidates than `graph_degree` spins forever. NaN already fails the comparison.
  RAFT_EXPECTS(std::isfinite(params.alpha) && params.alpha >= 1.0f,
               "gpipnn: alpha must be finite and >= 1 (got %f)",
               static_cast<double>(params.alpha));
}

}  // namespace cuvs::neighbors::experimental::gpipnn::detail
