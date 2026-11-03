/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
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
#include <cub/util_type.cuh>

#include <cstdint>

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
  uint32_t salt = q_id * 0x9e3779b9u + 0x85ebca6bu; // golden-ratio constant
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
 * Casts a buffer of `T` to half. Exact for int8 and uint8: half represents every integer up to
 * 2048.
 */
template <typename T>
RAFT_KERNEL cast_to_half_kernel(const T* __restrict__ src, half* __restrict__ dst, int64_t n)
{
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) { dst[i] = __float2half(static_cast<float>(src[i])); }
}

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
  const int64_t row = blockIdx.x;
  if (row >= n_rows) { return; }
  const T* in = src + row * dim;
  half* out   = dst + row * dim;

  float acc = 0.0f;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = static_cast<float>(in[i]);
    acc           = fmaf(v, v, acc);
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
  }
  __shared__ float partial[32];
  __shared__ float scale;
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  if (lane == 0) { partial[warp] = acc; }
  __syncthreads();
  if (warp == 0) {
    const int n_warps = static_cast<int>(blockDim.x) / warpSize;
    acc               = lane < n_warps ? partial[lane] : 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    // Zero for a zero row, so the stores below stay finite without a second branch.
    if (lane == 0) { scale = acc > 0.0f ? rsqrtf(acc) : 0.0f; }
  }
  __syncthreads();

  const float s = scale;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    out[i] = __float2half(static_cast<float>(in[i]) * s);
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
  RAFT_EXPECTS(params.alpha >= 1.0f, "gpipnn: alpha must be >= 1");
}

}  // namespace cuvs::neighbors::experimental::gpipnn::detail
