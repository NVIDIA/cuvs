/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "graph_core.cuh"
#include "graph_shared.cuh"

#include "../../../core/nvtx.hpp"
#include "../ann_utils.cuh"

#include <cuvs/neighbors/brute_force.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/matrix/init.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <new>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cuvs::neighbors::cagra::detail::graph {
namespace {

constexpr uint32_t kNoLink       = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kThreads      = 256;
constexpr uint32_t kPruneWarps   = 4;
constexpr uint32_t kMaxLinks     = 256;
constexpr uint32_t kProjectWarps = 4;
constexpr uint64_t kMaxSample    = helpers::kBridgeMaxSample;
// A pool holds kPoolCoverage times the median number of samples per component.
constexpr uint64_t kPoolCoverage = 8;
constexpr uint64_t kMinPool      = 128;
constexpr uint64_t kMaxPool      = 1024;
constexpr int kMaxPasses         = 3;
// Share of a full row that bridges may replace.
constexpr double kMaxRowShare = 0.5;
// Joining what the passes leave: sampled nodes per component and exact neighbors per node.
constexpr uint32_t kJoinReps      = 8;
constexpr uint64_t kJoinNeighbors = 32;
constexpr int kMaxJoinRounds      = 3;

// all: every pruned link of the sampled layer; cross: only the links between components.
enum class bridge_mode { off, all, cross };

// TODO(cuvs#2955): temporary switch CUVS_CAGRA_BRIDGE=0|cross for A/B tests, remove before merge.
inline auto bridge_mode_from_env() -> bridge_mode
{
  const char* env     = std::getenv("CUVS_CAGRA_BRIDGE");
  const std::string v = env == nullptr ? "" : env;
  if (v == "0" || v == "off") { return bridge_mode::off; }
  if (v == "cross") { return bridge_mode::cross; }
  return bridge_mode::all;
}

inline auto blocks_for(uint64_t count, uint32_t threads) -> uint32_t
{
  return static_cast<uint32_t>(std::max<uint64_t>(1, raft::ceildiv<uint64_t>(count, threads)));
}

struct is_nonzero {
  __device__ auto operator()(uint32_t v) const -> bool { return v != 0; }
};

struct is_set {
  __device__ auto operator()(uint8_t v) const -> bool { return v != 0; }
};

struct key_row {
  __host__ __device__ auto operator()(uint64_t key) const -> uint64_t { return key >> 32; }
};

__host__ __device__ inline auto splitmix64(uint64_t x) -> uint64_t
{
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

__device__ __forceinline__ auto warp_sum(float v) -> float
{
  for (int offset = raft::WarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

/** Smaller is closer: squared L2, or the negated inner product. */
template <bool InnerProduct>
__device__ __forceinline__ auto warp_distance(const float* a,
                                              const float* b,
                                              uint32_t dim,
                                              uint32_t lane) -> float
{
  float acc = 0.f;
  for (uint32_t j = lane; j < dim; j += raft::WarpSize) {
    if constexpr (InnerProduct) {
      acc += a[j] * b[j];
    } else {
      const float d = a[j] - b[j];
      acc += d * d;
    }
  }
  acc = warp_sum(acc);
  return InnerProduct ? -acc : acc;
}

/** Unites the endpoints of every edge in a batch of rows and counts the in-degrees. */
__global__ void kern_union_rows(raft::device_matrix_view<uint32_t, int64_t> rows,
                                uint64_t row_offset,
                                uint64_t n,
                                uint32_t* parent,
                                uint32_t* in_degree)
{
  const uint64_t degree = rows.extent(1);
  const uint64_t tid    = threadIdx.x + static_cast<uint64_t>(blockDim.x) * blockIdx.x;
  if (tid >= rows.extent(0) * degree) { return; }
  const uint64_t src = row_offset + tid / degree;
  const uint64_t dst = rows.data_handle()[tid];
  if (dst >= n || dst == src) { return; }
  atomicAdd(in_degree + dst, 1u);
  uf_unite(parent, static_cast<uint32_t>(src), static_cast<uint32_t>(dst));
}

__global__ void kern_flatten(uint32_t* parent, uint64_t n)
{
  const uint64_t i = threadIdx.x + static_cast<uint64_t>(blockDim.x) * blockIdx.x;
  if (i < n) { parent[i] = uf_find(parent, static_cast<uint32_t>(i)); }
}

/** Takes each node with probability 1 / max(ratio, size / cap) of its component, and every root. */
__global__ void kern_sample(const uint32_t* labels,
                            const uint32_t* sizes,
                            uint64_t n,
                            uint32_t ratio,
                            uint32_t cap,
                            uint64_t seed,
                            uint8_t* flags)
{
  const uint64_t i = threadIdx.x + static_cast<uint64_t>(blockDim.x) * blockIdx.x;
  if (i >= n) { return; }
  const uint32_t root = labels[i];
  const uint32_t rate = max(ratio, raft::ceildiv(sizes[root], cap));
  flags[i]            = root == i || splitmix64(seed ^ splitmix64(i)) % rate == 0;
}

/** One warp per row: out[r] = data[ids[r]] as float, unit length when `normalize`. */
template <typename T>
__global__ void kern_gather_rows(const T* data,
                                 uint64_t stride,
                                 uint32_t dim,
                                 const uint32_t* ids,
                                 uint64_t count,
                                 bool normalize,
                                 float* out)
{
  const uint64_t row =
    (threadIdx.x + static_cast<uint64_t>(blockDim.x) * blockIdx.x) / raft::WarpSize;
  const uint32_t lane = threadIdx.x % raft::WarpSize;
  if (row >= count) { return; }
  const T* x = data + (ids == nullptr ? row : uint64_t(ids[row])) * stride;
  float* y   = out + row * dim;
  float norm = 0.f;
  for (uint32_t j = lane; j < dim; j += raft::WarpSize) {
    const float v = cuvs::spatial::knn::detail::utils::mapping<float>{}(x[j]);
    y[j]          = v;
    norm += v * v;
  }
  if (!normalize) { return; }
  norm = warp_sum(norm);
  if (norm <= 0.f) { return; }
  const float scale = rsqrtf(norm);
  for (uint32_t j = lane; j < dim; j += raft::WarpSize) {
    y[j] *= scale;
  }
}

/**
 * HNSW neighbor selection over a candidate pool, one block per sample row q. Candidates are
 * visited nearest first; c is rejected if an accepted k has dist(c, k) < dist(q, c). If no
 * accepted link leaves q's component, the nearest candidate of another component takes the last
 * slot.
 */
template <bool InnerProduct>
__global__ void kern_prune(const float* __restrict__ vectors,  // [s, dim]
                           uint32_t dim,
                           uint64_t s,
                           const uint32_t* __restrict__ labels,  // [s]
                           const int64_t* __restrict__ pool,     // [rows, pool_size]
                           uint32_t pool_size,
                           uint64_t row_offset,
                           uint32_t m,
                           uint32_t* __restrict__ links,    // [s, m]
                           float* __restrict__ link_dists)  // [s, m]
{
  extern __shared__ float prune_smem[];
  __shared__ uint32_t kept_id[kMaxLinks];
  __shared__ float kept_dist[kMaxLinks];
  __shared__ int64_t cand_id[kPruneWarps];
  __shared__ float cand_dist[kPruneWarps];
  __shared__ int cand_ok[kPruneWarps];
  __shared__ uint32_t n_kept;

  const uint32_t warp = threadIdx.x / raft::WarpSize;
  const uint32_t lane = threadIdx.x % raft::WarpSize;
  const uint64_t q    = row_offset + blockIdx.x;
  float* q_vec        = prune_smem;
  float* cand_vec     = q_vec + dim;
  float* my_vec       = cand_vec + warp * dim;

  for (uint32_t j = threadIdx.x; j < dim; j += blockDim.x) {
    q_vec[j] = vectors[q * dim + j];
  }
  if (threadIdx.x == 0) { n_kept = 0; }
  __syncthreads();

  const int64_t* row_pool = pool + uint64_t(blockIdx.x) * pool_size;
  for (uint32_t base = 0; base < pool_size; base += kPruneWarps) {
    const uint32_t kept_now = n_kept;
    if (kept_now >= m) { break; }
    const uint32_t idx = base + warp;
    const int64_t c    = idx < pool_size ? row_pool[idx] : int64_t{-1};
    bool ok            = uint64_t(c) < s && uint64_t(c) != q;
    float d_qc         = 0.f;
    if (ok) {
      const float* src = vectors + uint64_t(c) * dim;
      for (uint32_t j = lane; j < dim; j += raft::WarpSize) {
        my_vec[j] = src[j];
      }
      __syncwarp();
      d_qc = warp_distance<InnerProduct>(my_vec, q_vec, dim, lane);
      for (uint32_t t = 0; t < kept_now; t++) {
        if (warp_distance<InnerProduct>(my_vec, vectors + uint64_t(kept_id[t]) * dim, dim, lane) <
            d_qc) {
          ok = false;
          break;
        }
      }
    }
    if (lane == 0) {
      cand_id[warp]   = c;
      cand_dist[warp] = d_qc;
      cand_ok[warp]   = ok;
    }
    __syncthreads();

    // Accept the chunk's survivors in pool order, testing the later ones against each.
    for (uint32_t w = 0; w < kPruneWarps; w++) {
      if (!cand_ok[w]) { continue; }
      const uint32_t slot = n_kept;
      if (slot >= m) { break; }
      if (warp > w && cand_ok[warp]) {
        const bool rejected =
          warp_distance<InnerProduct>(my_vec, cand_vec + w * dim, dim, lane) < cand_dist[warp];
        if (rejected && lane == 0) { cand_ok[warp] = 0; }
      }
      if (threadIdx.x == 0) {
        kept_id[slot]   = static_cast<uint32_t>(cand_id[w]);
        kept_dist[slot] = cand_dist[w];
      }
      __syncthreads();
      if (threadIdx.x == 0) { n_kept = slot + 1; }
      __syncthreads();
    }
    __syncthreads();
  }

  if (warp == 0) {
    const uint32_t own = labels[q];
    bool foreign       = false;
    for (uint32_t t = lane; t < n_kept; t += raft::WarpSize) {
      foreign |= labels[kept_id[t]] != own;
    }
    if (!__any_sync(0xffffffffu, foreign)) {
      for (uint32_t base = 0; base < pool_size; base += raft::WarpSize) {
        const uint32_t idx = base + lane;
        const int64_t c    = idx < pool_size ? row_pool[idx] : int64_t{-1};
        const unsigned hit = __ballot_sync(0xffffffffu, uint64_t(c) < s && labels[c] != own);
        if (hit == 0) { continue; }
        const int64_t f = __shfl_sync(0xffffffffu, c, __ffs(hit) - 1);
        const float d = warp_distance<InnerProduct>(vectors + uint64_t(f) * dim, q_vec, dim, lane);
        if (lane == 0) {
          const uint32_t slot = min(n_kept, m - 1);
          kept_id[slot]       = static_cast<uint32_t>(f);
          kept_dist[slot]     = d;
          n_kept              = slot + 1;
        }
        break;
      }
    }
  }
  __syncthreads();

  const uint32_t total = n_kept;
  for (uint32_t t = threadIdx.x; t < m; t += blockDim.x) {
    links[q * m + t]      = t < total ? kept_id[t] : kNoLink;
    link_dists[q * m + t] = t < total ? kept_dist[t] : INFINITY;
  }
}

/** Each link a-b as the keys (a, b) and (b, a) in sample positions. */
__global__ void kern_emit_links(const uint32_t* links,
                                const float* link_dists,
                                const uint32_t* labels,
                                uint64_t s,
                                uint32_t m,
                                bool cross_only,
                                uint64_t* keys,
                                float* dists)
{
  const uint64_t i = threadIdx.x + static_cast<uint64_t>(blockDim.x) * blockIdx.x;
  if (i >= s * m) { return; }
  const uint64_t a = i / m;
  const uint64_t b = links[i];
  if (b == kNoLink || b == a || (cross_only && labels[a] == labels[b])) {
    keys[2 * i]     = std::numeric_limits<uint64_t>::max();
    keys[2 * i + 1] = std::numeric_limits<uint64_t>::max();
    return;
  }
  keys[2 * i]      = (a << 32) | b;
  keys[2 * i + 1]  = (b << 32) | a;
  dists[2 * i]     = link_dists[i];
  dists[2 * i + 1] = link_dists[i];
}

__device__ __forceinline__ auto ordered_bits(float f) -> uint32_t
{
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/** (row, target) keys to (row, distance) keys, so that each row's links sort nearest first. */
__global__ void kern_rank_links(
  const uint64_t* keys, const float* dists, uint64_t count, uint64_t* rank_keys, uint32_t* targets)
{
  const uint64_t i = threadIdx.x + static_cast<uint64_t>(blockDim.x) * blockIdx.x;
  if (i >= count) { return; }
  rank_keys[i] = (keys[i] >> 32 << 32) | ordered_bits(dists[i]);
  targets[i]   = static_cast<uint32_t>(keys[i] & 0xffffffffull);
}

/** Shared memory per warp of kern_project_rows: row, fresh links, usable slots, slot classes. */
__host__ __device__ constexpr auto project_smem_per_warp(uint32_t degree) -> size_t
{
  return (size_t(degree) * (2 * sizeof(uint32_t) + sizeof(uint16_t) + 1) + 15) / 16 * 16;
}

/**
 * Writes the links of each sampled row into its graph row, one warp per row. Links whose target
 * is already in the row are dropped and keep that slot. The others fill the empty slots, then
 * replace up to `max_replace` edges from the end of the row, never the last in-edge of a node.
 * counters: [0] links written, [1] edges replaced, [2] rows changed.
 */
__global__ void kern_project_rows(uint32_t* rows,  // [s, degree]
                                  uint32_t degree,
                                  uint64_t s,
                                  uint64_t n,
                                  const uint32_t* sample_ids,
                                  const uint32_t* link_targets,  // sample positions, by row
                                  const uint64_t* row_start,     // [s + 1]
                                  uint32_t max_replace,
                                  uint32_t* in_degree,
                                  unsigned long long* counters)
{
  extern __shared__ uint8_t project_smem[];
  const uint32_t warp = threadIdx.x / raft::WarpSize;
  const uint32_t lane = threadIdx.x % raft::WarpSize;
  const uint64_t a    = blockIdx.x * uint64_t(kProjectWarps) + warp;
  if (a >= s) { return; }

  uint8_t* base = project_smem + warp * project_smem_per_warp(degree);
  auto* row     = reinterpret_cast<uint32_t*>(base);
  auto* fresh   = row + degree;
  auto* usable  = reinterpret_cast<uint16_t*>(fresh + degree);
  auto* cls     = reinterpret_cast<uint8_t*>(usable + degree);  // 0 empty, 1 replaceable, 2 keep
  uint32_t* g   = rows + a * degree;

  for (uint32_t j = lane; j < degree; j += raft::WarpSize) {
    const uint32_t w = g[j];
    row[j]           = w;
    cls[j]           = w >= n ? 0 : 1;
  }
  __syncwarp();

  uint32_t n_fresh  = 0;
  const uint64_t lo = row_start[a];
  const uint64_t hi = row_start[a + 1];
  for (uint64_t chunk = lo; chunk < hi; chunk += raft::WarpSize) {
    const uint64_t i = chunk + lane;
    const bool has   = i < hi;
    const uint32_t t = has ? sample_ids[link_targets[i]] : kNoLink;
    bool present     = false;
    if (has) {
      for (uint32_t j = 0; j < degree; j++) {
        if (row[j] == t) {
          present = true;
          cls[j]  = 2;
          break;
        }
      }
    }
    const bool is_fresh   = has && !present;
    const unsigned ballot = __ballot_sync(0xffffffffu, is_fresh);
    const uint32_t pos    = n_fresh + __popc(ballot & ((1u << lane) - 1u));
    if (is_fresh && pos < degree) { fresh[pos] = t; }
    n_fresh += __popc(ballot);
  }
  n_fresh = min(n_fresh, degree);
  __syncwarp();

  uint32_t writes = 0;
  if (lane == 0) {
    for (uint32_t j = 0; j < degree && writes < n_fresh; j++) {
      if (cls[j] == 0) { usable[writes++] = static_cast<uint16_t>(j); }
    }
    uint32_t replaced = 0;
    for (uint32_t j = degree; j-- > 0 && writes < n_fresh && replaced < max_replace;) {
      if (cls[j] != 1) { continue; }
      uint32_t* deg = in_degree + row[j];
      uint32_t old  = *deg;
      while (old > 1) {
        const uint32_t prev = atomicCAS(deg, old, old - 1);
        if (prev == old) { break; }
        old = prev;
      }
      if (old > 1) {
        usable[writes++] = static_cast<uint16_t>(j);
        replaced++;
      }
    }
    if (writes > 0) {
      atomicAdd(counters, static_cast<unsigned long long>(writes));
      atomicAdd(counters + 1, static_cast<unsigned long long>(replaced));
      atomicAdd(counters + 2, 1ull);
    }
  }
  writes = __shfl_sync(0xffffffffu, writes, 0);
  __syncwarp();

  for (uint32_t r = lane; r < writes; r += raft::WarpSize) {
    g[usable[r]] = fresh[r];
    atomicAdd(in_degree + fresh[r], 1u);
  }
}

/**
 * Weak components and in-degrees of a host or device graph, streamed to the device in row
 * batches. Each label becomes the smallest node id of its component.
 */
void label_components(raft::resources const& res,
                      const uint32_t* graph,
                      uint64_t n,
                      uint64_t degree,
                      raft::device_vector_view<uint32_t, int64_t> labels,
                      raft::device_vector_view<uint32_t, int64_t> in_degree)
{
  auto stream = raft::resource::get_cuda_stream(res);
  thrust::sequence(
    raft::resource::get_thrust_policy(res), labels.data_handle(), labels.data_handle() + n);
  raft::matrix::fill(res, in_degree, 0u);
  auto batches = cuvs::spatial::knn::detail::utils::make_batch_load_iterator<uint32_t>(
    res, graph, n, degree, helpers::kOptimizeBatchSize, stream);
  for (const auto& batch : batches) {
    kern_union_rows<<<blocks_for(batch.size() * degree, kThreads), kThreads, 0, stream.get()>>>(
      batch.view(), batch.offset(), n, labels.data_handle(), in_degree.data_handle());
  }
  kern_flatten<<<blocks_for(n, kThreads), kThreads, 0, stream.get()>>>(labels.data_handle(), n);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/** Component sizes in ascending order; `sizes` gets the size of each root's component. */
auto component_sizes(raft::resources const& res,
                     raft::device_vector_view<uint32_t, int64_t> labels,
                     raft::device_vector_view<uint32_t, int64_t> sizes) -> std::vector<uint32_t>
{
  const uint64_t n = labels.extent(0);
  auto stream      = raft::resource::get_cuda_stream(res);
  auto policy      = raft::resource::get_thrust_policy(res);
  raft::matrix::fill(res, sizes, 0u);
  kern_component_sizes<uint32_t>
    <<<blocks_for(n, kThreads), kThreads, 0, stream.get()>>>(labels, sizes);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  const uint32_t* first = sizes.data_handle();
  const uint64_t count  = thrust::count_if(policy, first, first + n, is_nonzero{});
  auto d_out            = raft::make_device_vector<uint32_t, int64_t>(res, count);
  thrust::copy_if(policy, first, first + n, d_out.data_handle(), is_nonzero{});
  std::vector<uint32_t> out(count);
  raft::copy(out.data(), d_out.data_handle(), count, stream);
  raft::resource::sync_stream(res);
  std::sort(out.begin(), out.end());
  return out;
}

struct sample_plan {
  uint64_t ratio  = 0;
  uint64_t pool   = 0;
  uint64_t sample = 0;  // expected
};

/**
 * Sampling at 1 / ratio, but at most kMaxPool / kPoolCoverage nodes per component, so that a
 * pool of kPoolCoverage times the median component's samples reaches beyond the component.
 * The ratio doubles until the expected sample is at most kMaxSample.
 */
auto plan_sample(std::vector<uint32_t> const& sizes, uint64_t ratio) -> sample_plan
{
  const uint64_t cap = kMaxPool / kPoolCoverage;
  auto samples       = [cap](uint64_t r, uint64_t size) {
    return 1 + size / std::max(r, raft::ceildiv(size, cap));
  };
  sample_plan plan;
  plan.ratio = ratio;
  for (;;) {
    plan.sample = 0;
    for (auto size : sizes) {
      plan.sample += samples(plan.ratio, size);
    }
    if (plan.sample <= kMaxSample || plan.ratio >= sizes.back()) { break; }
    plan.ratio *= 2;
  }
  const uint64_t median = samples(plan.ratio, sizes[sizes.size() / 2]);
  plan.pool             = std::clamp<uint64_t>(
    raft::round_up_safe<uint64_t>(kPoolCoverage * median, 32), kMinPool, kMaxPool);
  return plan;
}

template <typename T>
struct bridge_input {
  const T* dataset;
  bool on_device;
  // Row pitch and dimension; padding columns are zero.
  uint32_t dim;
  // Inner product, on unit vectors for cosine; squared L2 otherwise.
  bool ip;
  bool cosine;
};

struct node_sample {
  raft::device_vector<uint32_t, int64_t> ids;
  raft::device_vector<uint32_t, int64_t> labels;
  std::vector<uint32_t> h_ids;
};

/** The nodes kern_sample takes, with their component labels. */
auto draw_sample(raft::resources const& res,
                 raft::device_vector_view<uint32_t, int64_t> labels,
                 raft::device_vector_view<uint32_t, int64_t> sizes,
                 raft::device_vector_view<uint8_t, int64_t> flags,
                 uint64_t ratio,
                 uint32_t cap,
                 uint64_t seed) -> node_sample
{
  const uint64_t n = labels.extent(0);
  auto stream      = raft::resource::get_cuda_stream(res);
  auto policy      = raft::resource::get_thrust_policy(res);
  kern_sample<<<blocks_for(n, kThreads), kThreads, 0, stream.get()>>>(
    labels.data_handle(),
    sizes.data_handle(),
    n,
    static_cast<uint32_t>(std::min<uint64_t>(ratio, kNoLink)),
    cap,
    seed,
    flags.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  const uint8_t* f = flags.data_handle();
  const uint64_t s = thrust::count_if(policy, f, f + n, is_set{});
  node_sample out{raft::make_device_vector<uint32_t, int64_t>(res, s),
                  raft::make_device_vector<uint32_t, int64_t>(res, s),
                  std::vector<uint32_t>(s)};
  thrust::copy_if(policy,
                  thrust::counting_iterator<uint32_t>(0),
                  thrust::counting_iterator<uint32_t>(static_cast<uint32_t>(n)),
                  f,
                  out.ids.data_handle(),
                  is_set{});
  thrust::gather(policy,
                 out.ids.data_handle(),
                 out.ids.data_handle() + s,
                 labels.data_handle(),
                 out.labels.data_handle());
  raft::copy(out.h_ids.data(), out.ids.data_handle(), s, stream);
  raft::resource::sync_stream(res);
  return out;
}

/** The sampled vectors as float, unit length for cosine. */
template <typename T>
auto gather_vectors(raft::resources const& res,
                    bridge_input<T> const& in,
                    node_sample const& sample) -> raft::device_matrix<float, int64_t>
{
  const uint64_t s      = sample.h_ids.size();
  const uint32_t dim    = in.dim;
  const uint32_t blocks = blocks_for(s * raft::WarpSize, kThreads);
  auto stream           = raft::resource::get_cuda_stream(res);
  auto out              = raft::make_device_matrix<float, int64_t>(res, s, dim);
  if (in.on_device) {
    kern_gather_rows<T><<<blocks, kThreads, 0, stream.get()>>>(
      in.dataset, dim, dim, sample.ids.data_handle(), s, in.cosine, out.data_handle());
  } else {
    auto h_rows = raft::make_host_matrix<T, int64_t>(s, dim);
#pragma omp parallel for
    for (int64_t r = 0; r < static_cast<int64_t>(s); r++) {
      const T* src = in.dataset + uint64_t(sample.h_ids[r]) * dim;
      std::copy(src, src + dim, &h_rows(r, 0));
    }
    auto d_rows = raft::make_device_matrix<T, int64_t>(res, s, dim);
    raft::copy(d_rows.data_handle(), h_rows.data_handle(), s * dim, stream);
    kern_gather_rows<T><<<blocks, kThreads, 0, stream.get()>>>(
      d_rows.data_handle(), dim, dim, nullptr, s, in.cosine, out.data_handle());
    raft::resource::sync_stream(res);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  return out;
}

/**
 * Writes u -> v unless present, into an empty slot or else over the edge to the target with the
 * largest in-degree (at least 2). Slots in `kept` are not replaced, and the new one joins them.
 */
auto insert_edge(raft::host_matrix_view<uint32_t, int64_t> graph,
                 uint32_t u,
                 uint32_t v,
                 std::vector<uint32_t>& in_degree,
                 std::unordered_set<uint64_t>& kept) -> bool
{
  const uint64_t n      = graph.extent(0);
  const uint64_t degree = graph.extent(1);
  uint32_t* row         = &graph(u, 0);
  if (std::find(row, row + degree, v) != row + degree) { return true; }
  int64_t slot = -1;
  for (uint64_t j = 0; j < degree; j++) {
    if (kept.count(u * degree + j) != 0) { continue; }
    const uint32_t w = row[j];
    if (w >= n) {
      slot = j;
      break;
    }
    if (in_degree[w] > 1 && (slot < 0 || in_degree[w] > in_degree[row[slot]])) { slot = j; }
  }
  if (slot < 0) { return false; }
  if (row[slot] < n) { in_degree[row[slot]]--; }
  row[slot] = v;
  in_degree[v]++;
  kept.insert(u * degree + slot);
  return true;
}

/** Gives each node without in-edges one from its nearest out-neighbor with a slot to spare. */
auto repair_in_degree(raft::host_matrix_view<uint32_t, int64_t> graph,
                      std::vector<uint32_t>& in_degree,
                      std::unordered_set<uint64_t>& kept) -> uint64_t
{
  const uint64_t n      = graph.extent(0);
  const uint64_t degree = graph.extent(1);
  uint64_t repaired     = 0;
  for (uint64_t v = 0; v < n; v++) {
    if (in_degree[v] != 0) { continue; }
    for (uint64_t j = 0; j < degree; j++) {
      const uint32_t u = graph(v, j);
      if (u < n && u != v && insert_edge(graph, u, uint32_t(v), in_degree, kept)) {
        repaired++;
        break;
      }
    }
  }
  return repaired;
}

/**
 * Joins the components left by the passes along a spanning forest of exact nearest edges between
 * up to kJoinReps sampled nodes per component (Kruskal), and chains what that leaves apart, so the
 * graph becomes weakly connected unless a replaced edge splits another component. Components join
 * only once an edge between them is written, so a full row defers to the next nearest pair.
 * Returns the number of joining edges.
 */
template <typename T>
auto join_components(raft::resources const& res,
                     bridge_input<T> const& in,
                     raft::host_matrix_view<uint32_t, int64_t> graph,
                     raft::device_vector_view<uint32_t, int64_t> labels,
                     raft::device_vector_view<uint32_t, int64_t> sizes,
                     raft::device_vector_view<uint32_t, int64_t> d_in_degree,
                     raft::device_vector_view<uint8_t, int64_t> flags,
                     uint64_t num_components,
                     std::unordered_set<uint64_t>& kept,
                     int round) -> uint64_t
{
  const uint64_t n = graph.extent(0);
  auto stream      = raft::resource::get_cuda_stream(res);
  // kern_sample takes each root and about `cap` more nodes per component; roots only when the
  // ratio exceeds every component.
  const bool many_reps = num_components * (kJoinReps + 1) <= kMaxSample;
  auto sample          = draw_sample(res,
                            labels,
                            sizes,
                            flags,
                            many_reps ? 1 : kNoLink,
                            many_reps ? kJoinReps : 1,
                            0x7a11ull + round);
  const uint64_t r     = sample.h_ids.size();
  std::vector<uint32_t> h_labels(r);
  std::vector<uint32_t> in_degree(n);
  raft::copy(h_labels.data(), sample.labels.data_handle(), r, stream);
  raft::copy(in_degree.data(), d_in_degree.data_handle(), n, stream);
  raft::resource::sync_stream(res);

  std::unordered_map<uint32_t, uint32_t> component_of;
  component_of.reserve(num_components);
  std::vector<uint32_t> component(r);
  for (uint64_t a = 0; a < r; a++) {
    const auto next = static_cast<uint32_t>(component_of.size());
    component[a]    = component_of.emplace(h_labels[a], next).first->second;
  }
  std::vector<uint32_t> parent(component_of.size());
  std::iota(parent.begin(), parent.end(), 0u);
  auto find = [&parent](uint32_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x         = parent[x];
    }
    return x;
  };
  uint64_t joined = 0;
  auto join       = [&](uint32_t a, uint32_t b) {
    const uint32_t x = find(component[a]);
    const uint32_t y = find(component[b]);
    if (x == y) { return true; }
    const uint32_t u    = sample.h_ids[a];
    const uint32_t v    = sample.h_ids[b];
    const bool forward  = insert_edge(graph, u, v, in_degree, kept);
    const bool backward = insert_edge(graph, v, u, in_degree, kept);
    if (!forward && !backward) { return false; }
    parent[std::max(x, y)] = std::min(x, y);
    joined++;
    return true;
  };

  if (r <= kMaxSample) {
    const uint64_t k = std::min<uint64_t>(r, kJoinNeighbors + 1);
    auto vectors     = gather_vectors(res, in, sample);
    cuvs::neighbors::brute_force::index_params bf_params;
    bf_params.metric =
      in.ip ? cuvs::distance::DistanceType::InnerProduct : cuvs::distance::DistanceType::L2Expanded;
    auto bf_index =
      cuvs::neighbors::brute_force::build(res, bf_params, raft::make_const_mdspan(vectors.view()));
    auto d_ids   = raft::make_device_matrix<int64_t, int64_t>(res, r, k);
    auto d_dists = raft::make_device_matrix<float, int64_t>(res, r, k);
    cuvs::neighbors::brute_force::search(res,
                                         cuvs::neighbors::brute_force::search_params{},
                                         bf_index,
                                         raft::make_const_mdspan(vectors.view()),
                                         d_ids.view(),
                                         d_dists.view());
    std::vector<int64_t> ids(r * k);
    std::vector<float> dists(r * k);
    raft::copy(ids.data(), d_ids.data_handle(), r * k, stream);
    raft::copy(dists.data(), d_dists.data_handle(), r * k, stream);
    raft::resource::sync_stream(res);

    struct candidate {
      float dist;
      uint32_t a;
      uint32_t b;
    };
    std::vector<candidate> candidates;
    for (uint64_t a = 0; a < r; a++) {
      for (uint64_t j = 0; j < k; j++) {
        const int64_t b = ids[a * k + j];
        if (b < 0 || uint64_t(b) >= r || component[b] == component[a]) { continue; }
        const float d = dists[a * k + j];
        candidates.push_back({in.ip ? -d : d, uint32_t(a), uint32_t(b)});
      }
    }
    std::sort(candidates.begin(), candidates.end(), [](auto const& x, auto const& y) {
      return x.dist < y.dist;
    });
    for (auto const& c : candidates) {
      join(c.a, c.b);
    }
  }

  // Each sample of another group joins the last sample that joined.
  uint32_t anchor = 0;
  for (uint32_t a = 1; a < r; a++) {
    if (find(component[a]) != find(component[anchor]) && join(anchor, a)) { anchor = a; }
  }
  return joined;
}

/**
 * Connects the weak components of a CAGRA graph. Each pass samples every component, finds a
 * candidate pool per sampled node among the samples, prunes it HNSW-style to m = degree / 4 links
 * (with at least one leaving the component), and writes the links both ways into the graph.
 * Then every node without in-edges gets one, and join_components joins the components that
 * remain. Out of device memory, it stops early and leaves a valid graph.
 */
template <typename T>
void bridge_host_graph(raft::resources const& res,
                       cuvs::distance::DistanceType metric,
                       const T* dataset,
                       bool dataset_on_device,
                       int64_t dataset_stride,
                       raft::host_matrix_view<uint32_t, int64_t> graph)
{
  const auto mode       = bridge_mode_from_env();
  const uint64_t n      = graph.extent(0);
  const uint64_t degree = graph.extent(1);
  const auto dim        = static_cast<uint32_t>(dataset_stride);
  if (mode == bridge_mode::off || n < 2) { return; }

  using cuvs::distance::DistanceType;
  const bool cosine = metric == DistanceType::CosineExpanded;
  const bool ip     = metric == DistanceType::InnerProduct || cosine;
  const bool l2 = metric == DistanceType::L2Expanded || metric == DistanceType::L2SqrtExpanded ||
                  metric == DistanceType::L2Unexpanded || metric == DistanceType::L2SqrtUnexpanded;
  if (!ip && !l2) {
    RAFT_LOG_WARN("CAGRA graph bridging does not support metric %d", static_cast<int>(metric));
    return;
  }
  int device     = 0;
  int smem_optin = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  RAFT_CUDA_TRY(
    cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  const size_t prune_smem   = size_t(1 + kPruneWarps) * dim * sizeof(float);
  const size_t project_smem = kProjectWarps * project_smem_per_warp(degree);
  if (prune_smem + 4096 > size_t(smem_optin) || project_smem > size_t(smem_optin)) {
    RAFT_LOG_WARN("CAGRA graph bridging does not support dim %u with degree %lu", dim, degree);
    return;
  }
  auto prune_kernel = ip ? &kern_prune<true> : &kern_prune<false>;
  RAFT_CUDA_TRY(cudaFuncSetAttribute(
    prune_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(prune_smem)));
  RAFT_CUDA_TRY(cudaFuncSetAttribute(
    kern_project_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, int(project_smem)));

  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "cagra::graph::bridge_components(%zu, %zu)", n, degree);
  auto stream            = raft::resource::get_cuda_stream(res);
  auto policy            = raft::resource::get_thrust_policy(res);
  const uint32_t m       = std::clamp<uint32_t>(degree / 4, 1, kMaxLinks);
  const auto pool_metric = ip ? DistanceType::InnerProduct : DistanceType::L2Expanded;
  const bridge_input<T> in{dataset, dataset_on_device, dim, ip, cosine};
  std::vector<uint32_t> sizes;
  try {
    auto d_labels    = raft::make_device_vector<uint32_t, int64_t>(res, n);
    auto d_in_degree = raft::make_device_vector<uint32_t, int64_t>(res, n);
    auto d_sizes     = raft::make_device_vector<uint32_t, int64_t>(res, n);
    auto d_flags     = raft::make_device_vector<uint8_t, int64_t>(res, n);
    auto relabel     = [&]() {
      label_components(res, graph.data_handle(), n, degree, d_labels.view(), d_in_degree.view());
      sizes = component_sizes(res, d_labels.view(), d_sizes.view());
    };
    relabel();

    for (int pass = 0; pass < kMaxPasses && sizes.size() > 1; pass++) {
      const auto start      = std::chrono::steady_clock::now();
      const uint64_t before = sizes.size();
      const auto plan       = plan_sample(sizes, m);
      if (plan.sample > kMaxSample) { break; }
      auto sample      = draw_sample(res,
                                d_labels.view(),
                                d_sizes.view(),
                                d_flags.view(),
                                plan.ratio,
                                kMaxPool / kPoolCoverage,
                                0x2955ull + pass);
      const uint64_t s = sample.h_ids.size();
      if (s < 2) { break; }
      auto d_vectors = gather_vectors(res, in, sample);

      // Candidate pools, pruned in batches; the nearest candidate is the row itself.
      const uint64_t pool = std::min<uint64_t>(plan.pool + 1, s);
      cuvs::neighbors::brute_force::index_params bf_params;
      bf_params.metric = pool_metric;
      auto bf_index    = cuvs::neighbors::brute_force::build(
        res, bf_params, raft::make_const_mdspan(d_vectors.view()));
      const uint64_t batch = std::clamp<uint64_t>(
        helpers::kBridgePoolBatchBytes / (pool * (sizeof(int64_t) + sizeof(float))), 1, s);
      auto d_pool_ids   = raft::make_device_matrix<int64_t, int64_t>(res, batch, pool);
      auto d_pool_dists = raft::make_device_matrix<float, int64_t>(res, batch, pool);
      auto d_links      = raft::make_device_matrix<uint32_t, int64_t>(res, s, m);
      auto d_link_dists = raft::make_device_matrix<float, int64_t>(res, s, m);
      for (uint64_t lo = 0; lo < s; lo += batch) {
        const uint64_t rows = std::min<uint64_t>(batch, s - lo);
        auto queries        = raft::make_device_matrix_view<const float, int64_t>(
          d_vectors.data_handle() + lo * dim, rows, dim);
        auto ids =
          raft::make_device_matrix_view<int64_t, int64_t>(d_pool_ids.data_handle(), rows, pool);
        auto dists =
          raft::make_device_matrix_view<float, int64_t>(d_pool_dists.data_handle(), rows, pool);
        cuvs::neighbors::brute_force::search(
          res, cuvs::neighbors::brute_force::search_params{}, bf_index, queries, ids, dists);
        prune_kernel<<<static_cast<uint32_t>(rows),
                       kPruneWarps * raft::WarpSize,
                       prune_smem,
                       stream.get()>>>(d_vectors.data_handle(),
                                       dim,
                                       s,
                                       sample.labels.data_handle(),
                                       d_pool_ids.data_handle(),
                                       static_cast<uint32_t>(pool),
                                       lo,
                                       m,
                                       d_links.data_handle(),
                                       d_link_dists.data_handle());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }

      // Symmetrize and dedupe the links; order each row's links nearest first.
      const uint64_t e_max = 2 * s * m;
      auto d_keys          = raft::make_device_vector<uint64_t, int64_t>(res, e_max);
      auto d_dists         = raft::make_device_vector<float, int64_t>(res, e_max);
      kern_emit_links<<<blocks_for(s * m, kThreads), kThreads, 0, stream.get()>>>(
        d_links.data_handle(),
        d_link_dists.data_handle(),
        sample.labels.data_handle(),
        s,
        m,
        mode == bridge_mode::cross,
        d_keys.data_handle(),
        d_dists.data_handle());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      uint64_t* keys = d_keys.data_handle();
      thrust::sort_by_key(policy, keys, keys + e_max, d_dists.data_handle());
      const uint64_t e_unique =
        thrust::unique_by_key(policy, keys, keys + e_max, d_dists.data_handle()).first - keys;
      const uint64_t e_valid =
        thrust::lower_bound(policy, keys, keys + e_unique, std::numeric_limits<uint64_t>::max()) -
        keys;
      auto d_rank_keys = raft::make_device_vector<uint64_t, int64_t>(res, e_valid);
      auto d_targets   = raft::make_device_vector<uint32_t, int64_t>(res, e_valid);
      kern_rank_links<<<blocks_for(e_valid, kThreads), kThreads, 0, stream.get()>>>(
        keys, d_dists.data_handle(), e_valid, d_rank_keys.data_handle(), d_targets.data_handle());
      thrust::stable_sort_by_key(policy,
                                 d_rank_keys.data_handle(),
                                 d_rank_keys.data_handle() + e_valid,
                                 d_targets.data_handle());
      auto d_row_start = raft::make_device_vector<uint64_t, int64_t>(res, s + 1);
      auto row_of      = thrust::make_transform_iterator(d_rank_keys.data_handle(), key_row{});
      thrust::lower_bound(policy,
                          row_of,
                          row_of + e_valid,
                          thrust::counting_iterator<uint64_t>(0),
                          thrust::counting_iterator<uint64_t>(s + 1),
                          d_row_start.data_handle());

      // Write the links into the sampled rows.
      auto h_rows = raft::make_host_matrix<uint32_t, int64_t>(s, degree);
#pragma omp parallel for
      for (int64_t a = 0; a < static_cast<int64_t>(s); a++) {
        std::copy(&graph(sample.h_ids[a], 0), &graph(sample.h_ids[a], 0) + degree, &h_rows(a, 0));
      }
      auto d_rows     = raft::make_device_matrix<uint32_t, int64_t>(res, s, degree);
      auto d_counters = raft::make_device_vector<unsigned long long, int64_t>(res, 3);
      raft::copy(d_rows.data_handle(), h_rows.data_handle(), s * degree, stream);
      raft::matrix::fill(res, d_counters.view(), 0ull);
      kern_project_rows<<<blocks_for(s, kProjectWarps),
                          kProjectWarps * raft::WarpSize,
                          project_smem,
                          stream.get()>>>(d_rows.data_handle(),
                                          static_cast<uint32_t>(degree),
                                          s,
                                          n,
                                          sample.ids.data_handle(),
                                          d_targets.data_handle(),
                                          d_row_start.data_handle(),
                                          static_cast<uint32_t>(kMaxRowShare * degree),
                                          d_in_degree.data_handle(),
                                          d_counters.data_handle());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      unsigned long long counters[3];
      raft::copy(h_rows.data_handle(), d_rows.data_handle(), s * degree, stream);
      raft::copy(counters, d_counters.data_handle(), 3, stream);
      raft::resource::sync_stream(res);
#pragma omp parallel for
      for (int64_t a = 0; a < static_cast<int64_t>(s); a++) {
        std::copy(&h_rows(a, 0), &h_rows(a, 0) + degree, &graph(sample.h_ids[a], 0));
      }

      relabel();
      const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      RAFT_LOG_INFO(
        "CAGRA graph bridging pass %d: %lu -> %lu weakly connected components; %lu of %lu nodes "
        "sampled, pool %lu, %llu links written (%llu edges replaced in %llu rows), %.0f ms",
        pass + 1,
        before,
        sizes.size(),
        s,
        n,
        pool - 1,
        counters[0],
        counters[1],
        counters[2],
        ms);
      if (sizes.size() >= before) { break; }
    }

    std::unordered_set<uint64_t> kept;
    {
      std::vector<uint32_t> in_degree(n);
      raft::copy(in_degree.data(), d_in_degree.data_handle(), n, stream);
      raft::resource::sync_stream(res);
      const uint64_t repaired = repair_in_degree(graph, in_degree, kept);
      if (repaired > 0) {
        RAFT_LOG_INFO("CAGRA graph bridging gave %lu nodes without in-edges one", repaired);
        relabel();
      }
    }
    for (int round = 0; round < kMaxJoinRounds && sizes.size() > 1; round++) {
      const auto start      = std::chrono::steady_clock::now();
      const uint64_t before = sizes.size();
      const uint64_t edges  = join_components(res,
                                             in,
                                             graph,
                                             d_labels.view(),
                                             d_sizes.view(),
                                             d_in_degree.view(),
                                             d_flags.view(),
                                             before,
                                             kept,
                                             round);
      relabel();
      const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      RAFT_LOG_INFO(
        "CAGRA graph joining round %d: %lu -> %lu weakly connected components, %lu edges, %.0f ms",
        round + 1,
        before,
        sizes.size(),
        edges,
        ms);
    }
  } catch (std::bad_alloc const& e) {
    RAFT_LOG_WARN("CAGRA graph bridging stopped early, out of memory: %s", e.what());
  }

  if (sizes.size() > 1) {
    RAFT_LOG_WARN(
      "CAGRA graph still has %lu weakly connected components after bridging, the largest has %u "
      "of %lu nodes",
      sizes.size(),
      sizes.back(),
      n);
  }
}

template <typename T>
void bridge_device_graph(raft::resources const& res,
                         cuvs::distance::DistanceType metric,
                         const T* dataset,
                         bool dataset_on_device,
                         int64_t dataset_stride,
                         raft::device_matrix_view<uint32_t, int64_t> graph)
{
  const uint64_t n      = graph.extent(0);
  const uint64_t degree = graph.extent(1);
  if (bridge_mode_from_env() == bridge_mode::off || n < 2) { return; }
  auto stream = raft::resource::get_cuda_stream(res);
  try {
    auto d_labels    = raft::make_device_vector<uint32_t, int64_t>(res, n);
    auto d_in_degree = raft::make_device_vector<uint32_t, int64_t>(res, n);
    label_components(res, graph.data_handle(), n, degree, d_labels.view(), d_in_degree.view());
    if (component_sizes(res, d_labels.view(), d_in_degree.view()).size() <= 1) { return; }
  } catch (std::bad_alloc const& e) {
    RAFT_LOG_WARN("CAGRA graph bridging skipped, out of memory: %s", e.what());
    return;
  }
  auto h_graph = raft::make_host_matrix<uint32_t, int64_t>(n, degree);
  raft::copy(h_graph.data_handle(), graph.data_handle(), n * degree, stream);
  raft::resource::sync_stream(res);
  bridge_host_graph(res, metric, dataset, dataset_on_device, dataset_stride, h_graph.view());
  raft::copy(graph.data_handle(), h_graph.data_handle(), n * degree, stream);
  raft::resource::sync_stream(res);
}

}  // namespace

#define CUVS_DEFINE_CAGRA_GRAPH_BRIDGE(DataT)                                                \
  void bridge_components(raft::resources const& res,                                         \
                         cuvs::distance::DistanceType metric,                                \
                         DataT const* dataset,                                               \
                         bool dataset_on_device,                                             \
                         int64_t dataset_stride,                                             \
                         raft::host_matrix_view<uint32_t, int64_t, raft::row_major> graph)   \
  {                                                                                          \
    bridge_host_graph(res, metric, dataset, dataset_on_device, dataset_stride, graph);       \
  }                                                                                          \
  void bridge_components(raft::resources const& res,                                         \
                         cuvs::distance::DistanceType metric,                                \
                         DataT const* dataset,                                               \
                         bool dataset_on_device,                                             \
                         int64_t dataset_stride,                                             \
                         raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph) \
  {                                                                                          \
    bridge_device_graph(res, metric, dataset, dataset_on_device, dataset_stride, graph);     \
  }

CUVS_DEFINE_CAGRA_GRAPH_BRIDGE(float)
CUVS_DEFINE_CAGRA_GRAPH_BRIDGE(half)
CUVS_DEFINE_CAGRA_GRAPH_BRIDGE(int8_t)
CUVS_DEFINE_CAGRA_GRAPH_BRIDGE(uint8_t)

#undef CUVS_DEFINE_CAGRA_GRAPH_BRIDGE

}  // namespace cuvs::neighbors::cagra::detail::graph

namespace cuvs::neighbors::cagra::helpers {

bool bridge_enabled()
{
  return detail::graph::bridge_mode_from_env() != detail::graph::bridge_mode::off;
}

}  // namespace cuvs::neighbors::cagra::helpers
