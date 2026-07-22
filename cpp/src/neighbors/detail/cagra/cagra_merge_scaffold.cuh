/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../neighbors_device_intrinsics.cuh"

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra::detail::merge_scaffold {

// ----------------------------------------------------------------------------
// Constants and size limits
// ----------------------------------------------------------------------------

inline constexpr uint32_t MAX_FANOUT         = 32;
inline constexpr uint32_t MAX_LEADERS        = 8192;
inline constexpr uint32_t MAX_LEAF_SIZE      = 256;
inline constexpr int MAX_LEAF_DEGREE         = 8;
inline constexpr int ASSIGNMENT_TILE_ROWS    = 2048;
inline constexpr uint64_t DETERMINISTIC_SEED = 0x4c616e6472756d;
inline constexpr size_t GEMM_WORKSPACE_BYTES = size_t{2} * 1024 * 1024 * 1024;
inline constexpr int WARP_SIZE               = 32;
/** Warps per block in the one-row-per-warp kernels; their launch geometry derives from this. */
inline constexpr int ROW_WARPS_PER_BLOCK = 4;
// Centered INT8 dot products accumulate into INT32. Overflow requires more than 131,071
// dimensions even at the worst-case magnitude of 128, which is not realistic.
inline constexpr int64_t MAX_INTEGER_LEAF_DIMENSION =
  std::numeric_limits<int32_t>::max() / (128 * 128);

/** Grid size for the grid-stride kernels: blocks of 256 threads covering `items`, capped so
 * oversized workloads loop within resident threads instead. */
inline auto strided_grid_size(int64_t items) -> int
{
  return static_cast<int>(std::min<int64_t>((items + 255) / 256, 1048576));
}

// ----------------------------------------------------------------------------
// Build parameters and partition data structures
// ----------------------------------------------------------------------------

/** Internal controls for deterministic multi-level ball carving and leaf neighbor construction. */
struct build_params {
  uint32_t levels        = 2;
  uint32_t root_fanout   = 2;
  uint32_t lower_fanout  = 3;
  double leader_fraction = 0.02;
  uint32_t max_leaders   = 1000;
  uint32_t leaf_size     = 256;
  uint32_t leaf_degree   = 4;
};

/** Controls for one invocation of the reusable many-way split boundary. */
struct split_params {
  uint32_t fanout            = 1;
  double leader_fraction     = 0.02;
  uint32_t max_leaders       = 1000;
  uint32_t leaf_size         = 256;
  uint32_t level             = 0;
  uint32_t occurrence_stride = 1;
};

/** Device state and tuning knobs shared by every split level: the precomputed row norms, the
 * tiling and workspace capacities, and the seed feeding the deterministic leader samples. */
struct split_context {
  split_context(rmm::cuda_stream_view stream, int64_t rows)
    : norms(static_cast<size_t>(rows), stream)
  {
  }

  rmm::device_uvector<float> norms;
  int assignment_tile_rows    = ASSIGNMENT_TILE_ROWS;
  size_t gemm_workspace_bytes = GEMM_WORKSPACE_BYTES;
  uint64_t seed               = DETERMINISTIC_SEED;
};

struct partition_membership {
  uint32_t id         = 0;
  uint16_t occurrence = 0;
  uint16_t padding    = 0;
};

struct partition_range {
  uint32_t key  = 0;
  int64_t start = 0;
  int64_t end   = 0;
};

struct partition_set {
  rmm::device_uvector<partition_membership> memberships;
  std::vector<partition_range> ranges;
};

// ----------------------------------------------------------------------------
// Many-way partitioning
// ----------------------------------------------------------------------------

/** Describes a contiguous tile of a parent partition to assign to child leaders. */
struct assignment_tile {
  int64_t input_start     = 0;
  int64_t group_start     = 0;
  int64_t group_size      = 0;
  int64_t output_start    = 0;
  int32_t rows            = 0;
  int32_t leader_count    = 0;
  uint32_t child_key_base = 0;
  uint32_t leader_offset  = 0;
};

/** Describes a completed parent partition copied unchanged into the next split level. */
struct carry_span {
  int64_t input_start  = 0;
  int64_t output_start = 0;
  int32_t rows         = 0;
  uint32_t child_key   = 0;
};

/** Leader count selection logic */
inline int select_leader_count(int64_t rows, split_params const& params)
{
  auto sampled =
    static_cast<int64_t>(std::ceil(params.leader_fraction * static_cast<double>(rows)));
  sampled = std::max<int64_t>(sampled, params.fanout);
  sampled = std::min<int64_t>(sampled, params.max_leaders);
  sampled = std::min<int64_t>(sampled, rows);
  return static_cast<int>(sampled);
}

/** Return the bucket index for a leader count: ceil(log2(leaders)), so that
 * `1 << leader_bucket_index(leaders)` is the smallest power of two covering it. Requires
 * leaders >= 1. */
inline int leader_bucket_index(int leaders)
{
  return static_cast<int>(std::bit_width(static_cast<unsigned>(leaders - 1)));
}

/** Host-side decision for one parent partition: where its children and output rows land. */
struct parent_plan {
  partition_range parent;
  int64_t output_start    = 0;
  uint32_t child_key_base = 0;
  int32_t leader_count    = 0;  // 0 => carried unchanged as one child
  uint32_t leader_offset  = 0;  // deterministic leader sample start; meaningful only for splits

  auto carried() const -> bool { return leader_count == 0; }
  auto size() const -> int64_t { return parent.end - parent.start; }
  auto child_count() const -> uint32_t
  {
    return carried() ? 1 : static_cast<uint32_t>(leader_count);
  }
  auto output_rows(uint32_t fanout) const -> int64_t
  {
    return carried() ? size() : size() * static_cast<int64_t>(fanout);
  }
};

struct split_plan {
  std::vector<parent_plan> parents;
  int64_t output_rows  = 0;
  uint32_t child_count = 0;
};

/**
 * Decide on host how every parent partition maps into the next level.
 *
 * Parents within the leaf size are carried: their memberships pass through unchanged under one
 * child key. Larger parents are split: a deterministic leader sample of `leader_fraction` of
 * their rows, clamped to [fanout, max_leaders], is taken at evenly strided member positions
 * starting from an offset hashed from the seed, level, and parent identity, so reruns select the
 * same leaders. Child keys and output rows are dense in parent order.
 */
inline auto plan_split(std::vector<partition_range> const& ranges,
                       int64_t membership_count,
                       split_params const& params,
                       uint64_t seed) -> split_plan
{
  RAFT_EXPECTS(params.fanout >= 1 && params.fanout <= MAX_FANOUT,
               "Fastener split fanout must be between 1 and %d",
               MAX_FANOUT);
  RAFT_EXPECTS(params.leader_fraction > 0.0 && params.leader_fraction <= 1.0,
               "Fastener leader fraction must be in (0, 1]");
  RAFT_EXPECTS(params.max_leaders >= params.fanout && params.max_leaders <= MAX_LEADERS,
               "Fastener leader cap must cover the fanout and not exceed %d",
               MAX_LEADERS);

  split_plan plan;
  plan.parents.reserve(ranges.size());
  int64_t covered     = 0;
  int64_t output_rows = 0;
  int64_t child_keys  = 0;

  for (size_t parent_index = 0; parent_index < ranges.size(); ++parent_index) {
    auto const& parent = ranges[parent_index];
    RAFT_EXPECTS(parent.start == covered && parent.end > parent.start,
                 "Fastener parent ranges must compactly cover all memberships");
    covered = parent.end;

    parent_plan entry{.parent         = parent,
                      .output_start   = output_rows,
                      .child_key_base = static_cast<uint32_t>(child_keys)};
    if (entry.size() > params.leaf_size) {
      entry.leader_count  = select_leader_count(entry.size(), params);
      entry.leader_offset = static_cast<uint32_t>(
        cuvs::neighbors::detail::device::xorshift64(
          seed ^ (static_cast<uint64_t>(params.level) << 48) ^
          (static_cast<uint64_t>(parent.key) << 1) ^ static_cast<uint64_t>(parent_index)) %
        static_cast<uint64_t>(entry.size()));  // mixing in all the relevant state
    }

    child_keys += entry.child_count();
    output_rows += entry.output_rows(params.fanout);
    plan.parents.push_back(entry);
  }
  RAFT_EXPECTS(covered == membership_count,
               "Fastener parent ranges must compactly cover all memberships");
  RAFT_EXPECTS(output_rows <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
               "Fastener membership count must fit in uint32_t");
  RAFT_EXPECTS(child_keys <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
               "Fastener child key count must fit in uint32_t");
  plan.output_rows = output_rows;
  plan.child_count = static_cast<uint32_t>(child_keys);
  return plan;
}

/** Render the device-facing descriptor for one tile of a split parent. */
inline auto make_tile(parent_plan const& entry, int64_t start, uint32_t fanout, int tile_rows)
  -> assignment_tile
{
  return assignment_tile{
    .input_start = start,
    .group_start = entry.parent.start,
    .group_size  = entry.size(),
    .output_start =
      entry.output_start + (start - entry.parent.start) * static_cast<int64_t>(fanout),
    .rows           = static_cast<int32_t>(std::min<int64_t>(tile_rows, entry.parent.end - start)),
    .leader_count   = entry.leader_count,
    .child_key_base = entry.child_key_base,
    .leader_offset  = entry.leader_offset};
}

/** Group the split parents by padded leader count, so each bucket shares one GEMM shape. */
inline auto bucket_split_parents(split_plan const& plan, split_params const& params)
  -> std::vector<std::vector<parent_plan const*>>
{
  std::vector<std::vector<parent_plan const*>> buckets(
    leader_bucket_index(static_cast<int>(params.max_leaders)) + 1);
  for (auto const& entry : plan.parents) {
    if (!entry.carried()) { buckets[leader_bucket_index(entry.leader_count)].push_back(&entry); }
  }
  return buckets;
}

/** One candidate assignment of a row to a leader. Defaults to the invalid sentinel, which sorts
 * after every real candidate. */
struct leader_candidate {
  float distance  = std::numeric_limits<float>::max();
  uint32_t leader = std::numeric_limits<uint32_t>::max();

  __device__ auto valid() const -> bool { return leader != std::numeric_limits<uint32_t>::max(); }
};

/** Nearer candidates order first. Equal distances order by the lower leader index, keeping the
 * selection deterministic. */
__device__ inline auto operator<(leader_candidate a, leader_candidate b) -> bool
{
  return a.distance < b.distance || (a.distance == b.distance && a.leader < b.leader);
}

/** Replace the worst entry in the best list if the new candidate is better. */
__device__ inline void manyway_insert_local(leader_candidate candidate,
                                            int fanout,
                                            leader_candidate* best)
{
  int worst = 0;
  for (int j = 1; j < fanout; ++j) {
    if (best[worst] < best[j]) { worst = j; }
  }
  if (candidate < best[worst]) { best[worst] = candidate; }
}

/** Sort the selected candidates in place from the nearest to the farthest.
 *
 * This is hand-rolled because thrust::sort has an allocation and I can't get a cuda::std::sort to
 * compile. */
__device__ inline void manyway_sort_selected(int fanout, leader_candidate* selected)
{
  for (int i = 1; i < fanout; ++i) {
    leader_candidate candidate = selected[i];
    int j                      = i;
    while (j > 0 && candidate < selected[j - 1]) {
      selected[j] = selected[j - 1];
      --j;
    }
    selected[j] = candidate;
  }
}

/** Write the identity membership for each dataset row. */
static __global__ void initialize_root_memberships_kernel(partition_membership* memberships,
                                                          int64_t rows)
{
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < rows) { memberships[row] = {static_cast<uint32_t>(row), uint16_t{0}, uint16_t{0}}; }
}

/** Represent the full dataset as one identity partition without copying any vectors. */
inline auto make_root_partition(raft::resources const& res, int64_t rows) -> partition_set
{
  auto stream = raft::resource::get_cuda_stream(res);
  partition_set root{rmm::device_uvector<partition_membership>(static_cast<size_t>(rows), stream),
                     {{uint32_t{0}, int64_t{0}, rows}}};
  int blocks = static_cast<int>((rows + 255) / 256);
  initialize_root_memberships_kernel<<<blocks, 256, 0, stream>>>(root.memberships.data(), rows);
  RAFT_CUDA_TRY(cudaGetLastError());
  return root;
}

/** Copy the memberships of each completed parent to the output and write its one child key. */
static __global__ void carry_completed_parents_kernel(partition_membership const* input,
                                                      carry_span const* spans,
                                                      partition_membership* output,
                                                      uint32_t* output_keys)
{
  auto span = spans[blockIdx.x];
  for (int row = threadIdx.x; row < span.rows; row += blockDim.x) {
    output[span.output_start + row]      = input[span.input_start + row];
    output_keys[span.output_start + row] = span.child_key;
  }
}

/** Copy the vectors of each tile row into a dense float buffer. Unused rows become zero. */
template <typename T>
__global__ void manyway_gather_tile_points_kernel(T const* dataset,
                                                  int64_t dim,
                                                  partition_membership const* memberships,
                                                  assignment_tile const* tiles,
                                                  int batch_size,
                                                  int tile_rows,
                                                  float* output)
{
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t total  = static_cast<int64_t>(batch_size) * tile_rows * dim;
  for (; linear < total; linear += stride) {
    int64_t d   = linear % dim;
    int64_t row = (linear / dim) % tile_rows;
    int batch   = static_cast<int>(linear / (dim * tile_rows));
    auto tile   = tiles[batch];
    float value = 0.0f;
    if (row < tile.rows) {
      uint32_t id = memberships[tile.input_start + row].id;
      value       = static_cast<float>(dataset[static_cast<int64_t>(id) * dim + d]);
    }
    output[linear] = value;
  }
}

/** Copy the leader vectors of each tile into a dense float buffer and record the leader IDs. */
template <typename T>
__global__ void manyway_gather_tile_leaders_kernel(T const* dataset,
                                                   int64_t dim,
                                                   partition_membership const* memberships,
                                                   assignment_tile const* tiles,
                                                   int batch_size,
                                                   int padded_leaders,
                                                   float* output,
                                                   uint32_t* leader_ids)
{
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t total  = static_cast<int64_t>(batch_size) * padded_leaders * dim;
  for (; linear < total; linear += stride) {
    int64_t d   = linear % dim;
    int leader  = static_cast<int>((linear / dim) % padded_leaders);
    int batch   = static_cast<int>(linear / (dim * padded_leaders));
    auto tile   = tiles[batch];
    float value = 0.0f;
    if (leader < tile.leader_count) {
      int64_t relative = (tile.leader_offset +
                          (static_cast<int64_t>(leader) * tile.group_size) / tile.leader_count) %
                         tile.group_size;
      uint32_t id = memberships[tile.group_start + relative].id;
      value       = static_cast<float>(dataset[static_cast<int64_t>(id) * dim + d]);
      if (d == 0) { leader_ids[static_cast<int64_t>(batch) * padded_leaders + leader] = id; }
    }
    output[linear] = value;
  }
}

/** Select the nearest leaders for each tile row from the dot products. Write the child keys and
 *  memberships. One warp does one row. */
static __global__ void manyway_select_tiles_kernel(float const* dots,
                                                   int batch_size,
                                                   int tile_rows,
                                                   int padded_leaders,
                                                   int fanout,
                                                   int occurrence_stride,
                                                   float const* norms,
                                                   uint32_t const* leader_ids,
                                                   partition_membership const* input_memberships,
                                                   assignment_tile const* tiles,
                                                   uint32_t* output_keys,
                                                   partition_membership* output_memberships)
{
  int lane           = threadIdx.x % WARP_SIZE;
  int warp           = threadIdx.x / WARP_SIZE;
  int64_t linear_row = static_cast<int64_t>(blockIdx.x) * ROW_WARPS_PER_BLOCK + warp;
  int64_t total_rows = static_cast<int64_t>(batch_size) * tile_rows;
  if (linear_row >= total_rows) { return; }
  int batch = static_cast<int>(linear_row / tile_rows);
  int row   = static_cast<int>(linear_row % tile_rows);
  auto tile = tiles[batch];
  if (row >= tile.rows) { return; }

  int64_t input_index = tile.input_start + row;
  auto membership     = input_memberships[input_index];
  leader_candidate local_best[MAX_FANOUT];

  for (int leader = lane; leader < tile.leader_count; leader += WARP_SIZE) {
    uint32_t leader_id = leader_ids[static_cast<int64_t>(batch) * padded_leaders + leader];
    float dot = dots[(static_cast<int64_t>(batch) * tile_rows + row) * padded_leaders + leader];
    float distance = fmaxf(0.0f, norms[membership.id] + norms[leader_id] - 2.0f * dot);
    manyway_insert_local(
      {.distance = distance, .leader = static_cast<uint32_t>(leader)}, fanout, local_best);
  }

  extern __shared__ unsigned char shared_bytes[];
  auto* shared_candidates = reinterpret_cast<leader_candidate*>(shared_bytes);
  int shared_base         = (warp * WARP_SIZE + lane) * fanout;
  for (int j = 0; j < fanout; ++j) {
    shared_candidates[shared_base + j] = local_best[j];
  }
  __syncwarp();

  if (lane == 0) {
    leader_candidate selected[MAX_FANOUT];
    int warp_base = warp * WARP_SIZE * fanout;
    for (int candidate = 0; candidate < WARP_SIZE * fanout; ++candidate) {
      if (shared_candidates[warp_base + candidate].valid()) {
        manyway_insert_local(shared_candidates[warp_base + candidate], fanout, selected);
      }
    }
    manyway_sort_selected(fanout, selected);
    int64_t output_base =
      tile.output_start + (input_index - tile.input_start) * static_cast<int64_t>(fanout);
    for (int rank = 0; rank < fanout; ++rank) {
      output_keys[output_base + rank]        = tile.child_key_base + selected[rank].leader;
      output_memberships[output_base + rank] = {
        membership.id,
        static_cast<uint16_t>(membership.occurrence + rank * occurrence_stride),
        uint16_t{0}};
    }
  }
}

/** Read the sorted keys and make one contiguous range for each unique partition key. */
inline auto collect_partition_ranges(raft::resources const& res,
                                     uint32_t const* keys,
                                     int64_t count,
                                     uint32_t key_count) -> std::vector<partition_range>
{
  auto stream          = raft::resource::get_cuda_stream(res);
  auto output_capacity = static_cast<size_t>(std::min<int64_t>(count, key_count));
  rmm::device_uvector<uint32_t> device_unique_keys(output_capacity, stream);
  rmm::device_uvector<uint32_t> device_counts(output_capacity, stream);
  auto reduced_end = thrust::reduce_by_key(thrust::cuda::par.on(stream),
                                           thrust::device_pointer_cast(keys),
                                           thrust::device_pointer_cast(keys + count),
                                           thrust::make_constant_iterator(uint32_t{1}),
                                           thrust::device_pointer_cast(device_unique_keys.data()),
                                           thrust::device_pointer_cast(device_counts.data()));
  auto group_count =
    static_cast<size_t>(reduced_end.first - thrust::device_pointer_cast(device_unique_keys.data()));

  std::vector<uint32_t> unique_keys(group_count);
  std::vector<uint32_t> counts(group_count);
  raft::copy(unique_keys.data(), device_unique_keys.data(), group_count, stream);
  raft::copy(counts.data(), device_counts.data(), group_count, stream);
  raft::resource::sync_stream(res);

  std::vector<partition_range> groups;
  groups.reserve(group_count);
  int64_t cursor = 0;
  for (size_t index = 0; index < group_count; ++index) {
    RAFT_EXPECTS(unique_keys[index] < key_count, "Many-way group key is out of range");
    groups.push_back({unique_keys[index], cursor, cursor + counts[index]});
    cursor += counts[index];
  }
  RAFT_EXPECTS(cursor == count, "Many-way partition membership histogram lost entries");
  return groups;
}

/** Copy every carried parent's memberships unchanged. */
inline void carry_parents(raft::resources const& res,
                          partition_set const& parents,
                          split_plan const& plan,
                          rmm::device_uvector<uint32_t>& keys,
                          rmm::device_uvector<partition_membership>& memberships)
{
  std::vector<carry_span> carries;
  for (auto const& entry : plan.parents) {
    if (entry.carried()) {
      carries.push_back({entry.parent.start,
                         entry.output_start,
                         static_cast<int32_t>(entry.size()),
                         entry.child_key_base});
    }
  }
  if (carries.empty()) { return; }

  auto stream = raft::resource::get_cuda_stream(res);
  rmm::device_uvector<carry_span> device_carries(carries.size(), stream);
  raft::copy(device_carries.data(), carries.data(), carries.size(), stream);
  carry_completed_parents_kernel<<<static_cast<int>(carries.size()), 256, 0, stream>>>(
    parents.memberships.data(), device_carries.data(), memberships.data(), keys.data());
  RAFT_CUDA_TRY(cudaGetLastError());
}

/** Compute every pairwise dot product between the rows of A_i and the rows of B_i for a strided
 * batch of row-major matrices with rows of length `row_width`:
 * out_i[b * a_rows + a] = A_i[a] . B_i[b].
 *
 * Pretty much a wrapper around `cublasGemmStridedBatchedEx`. */
template <typename DataT, typename AccT>
void batched_row_dot_products(raft::resources const& res,
                              DataT const* a,
                              int a_rows,
                              long long a_stride,
                              DataT const* b,
                              int b_rows,
                              long long b_stride,
                              AccT* out,
                              long long out_stride,
                              int row_width,
                              int batch_count)
{
  static_assert((std::is_same_v<DataT, float> && std::is_same_v<AccT, float>) ||
                  (std::is_same_v<DataT, int8_t> && std::is_same_v<AccT, int32_t>),
                "Unsupported batched dot product type combination");
  constexpr auto data_type = std::is_same_v<DataT, float> ? CUDA_R_32F : CUDA_R_8I;
  constexpr auto out_type  = std::is_same_v<AccT, float> ? CUDA_R_32F : CUDA_R_32I;
  constexpr auto compute_type =
    std::is_same_v<AccT, float> ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32I;
  AccT alpha         = 1;
  AccT beta          = 0;
  auto cublas_handle = raft::resource::get_cublas_handle(res);
  RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
  RAFT_CUBLAS_TRY(cublasGemmStridedBatchedEx(cublas_handle,
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             a_rows,
                                             b_rows,
                                             row_width,
                                             &alpha,
                                             a,
                                             data_type,
                                             row_width,
                                             a_stride,
                                             b,
                                             data_type,
                                             row_width,
                                             b_stride,
                                             &beta,
                                             out,
                                             out_type,
                                             a_rows,
                                             out_stride,
                                             batch_count,
                                             compute_type,
                                             CUBLAS_GEMM_DEFAULT));
}

/**
 * Assign every row of one bucket's split parents to its `fanout` nearest leaders.
 *
 * Parents are cut into tiles of `assignment_tile_rows` rows; every tile in a bucket shares the
 * same padded leader count, so one strided batched GEMM per batch produces all point-leader dot
 * products. Tiles are processed in batches sized to the GEMM workspace: each batch gathers its
 * tile vectors and its parents' leader vectors into dense float buffers, and the selection kernel
 * converts dots to distances with the precomputed row norms (|x|^2 + |l|^2 - 2 x.l), keeps the
 * `fanout` nearest leaders per row (ties broken by lower leader index), and writes the child keys
 * and memberships.
 */
template <typename T>
void assign_bucket(raft::resources const& res,
                   raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
                   partition_set const& parents,
                   std::vector<parent_plan const*> const& bucket,
                   int padded_leaders,
                   split_params const& params,
                   split_context& context,
                   rmm::device_uvector<uint32_t>& keys,
                   rmm::device_uvector<partition_membership>& memberships)
{
  auto stream = raft::resource::get_cuda_stream(res);
  auto dim    = dataset.extent(1);

  // Cut every parent in the bucket into fixed-height tiles
  int tile_rows = context.assignment_tile_rows;
  std::vector<assignment_tile> tiles;
  for (auto const* entry : bucket) {
    for (int64_t start = entry->parent.start; start < entry->parent.end; start += tile_rows) {
      tiles.push_back(make_tile(*entry, start, params.fanout, tile_rows));
    }
  }

  // Size the batch so all per-tile buffers fit in the GEMM workspace budget
  size_t point_elements  = static_cast<size_t>(tile_rows) * dim;
  size_t leader_elements = static_cast<size_t>(padded_leaders) * dim;
  size_t dot_elements    = static_cast<size_t>(tile_rows) * padded_leaders;
  size_t bytes_per_batch = (point_elements + leader_elements + dot_elements) * sizeof(float) +
                           static_cast<size_t>(padded_leaders) * sizeof(uint32_t) +
                           sizeof(assignment_tile);
  RAFT_EXPECTS(bytes_per_batch <= context.gemm_workspace_bytes,
               "Fastener assignment workspace is too small for one tile");
  size_t batch_capacity = std::max<size_t>(
    1, std::min<size_t>(tiles.size(), context.gemm_workspace_bytes / bytes_per_batch));

  rmm::device_uvector<assignment_tile> device_tiles(batch_capacity, stream);
  rmm::device_uvector<float> tile_points(batch_capacity * point_elements, stream);
  rmm::device_uvector<float> tile_leaders(batch_capacity * leader_elements, stream);
  rmm::device_uvector<float> tile_dots(batch_capacity * dot_elements, stream);
  rmm::device_uvector<uint32_t> tile_leader_ids(batch_capacity * padded_leaders, stream);

  auto point_stride  = static_cast<int64_t>(point_elements);
  auto leader_stride = static_cast<int64_t>(leader_elements);
  auto dot_stride    = static_cast<int64_t>(dot_elements);
  size_t selection_shared =
    ROW_WARPS_PER_BLOCK * WARP_SIZE * params.fanout * sizeof(leader_candidate);

  for (size_t tile_offset = 0; tile_offset < tiles.size(); tile_offset += batch_capacity) {
    size_t batch_size = std::min(batch_capacity, tiles.size() - tile_offset);
    raft::copy(device_tiles.data(), tiles.data() + tile_offset, batch_size, stream);

    // Gather the batch's tile rows and leader vectors
    int point_blocks = strided_grid_size(static_cast<int64_t>(batch_size * point_elements));
    manyway_gather_tile_points_kernel<<<point_blocks, 256, 0, stream>>>(
      dataset.data_handle(),
      dim,
      parents.memberships.data(),
      device_tiles.data(),
      static_cast<int>(batch_size),
      tile_rows,
      tile_points.data());
    RAFT_CUDA_TRY(cudaGetLastError());

    int leader_blocks = strided_grid_size(static_cast<int64_t>(batch_size * leader_elements));
    manyway_gather_tile_leaders_kernel<<<leader_blocks, 256, 0, stream>>>(
      dataset.data_handle(),
      dim,
      parents.memberships.data(),
      device_tiles.data(),
      static_cast<int>(batch_size),
      padded_leaders,
      tile_leaders.data(),
      tile_leader_ids.data());
    RAFT_CUDA_TRY(cudaGetLastError());

    // point-leader dot products for the batch
    batched_row_dot_products(res,
                             tile_leaders.data(),
                             padded_leaders,
                             leader_stride,
                             tile_points.data(),
                             tile_rows,
                             point_stride,
                             tile_dots.data(),
                             dot_stride,
                             static_cast<int>(dim),
                             static_cast<int>(batch_size));

    // Keep each row's `fanout` nearest leaders and write child keys and memberships
    int selection_blocks =
      static_cast<int>((static_cast<int64_t>(batch_size) * tile_rows + ROW_WARPS_PER_BLOCK - 1) /
                       ROW_WARPS_PER_BLOCK);
    manyway_select_tiles_kernel<<<selection_blocks,
                                  ROW_WARPS_PER_BLOCK * WARP_SIZE,
                                  selection_shared,
                                  stream>>>(tile_dots.data(),
                                            static_cast<int>(batch_size),
                                            tile_rows,
                                            padded_leaders,
                                            static_cast<int>(params.fanout),
                                            static_cast<int>(params.occurrence_stride),
                                            context.norms.data(),
                                            tile_leader_ids.data(),
                                            parents.memberships.data(),
                                            device_tiles.data(),
                                            keys.data(),
                                            memberships.data());
    RAFT_CUDA_TRY(cudaGetLastError());
  }
}

/**
 * Split every oversized parent into overlapping nearest-leader children.
 *
 * All levels, including the root, traverse this boundary. `plan_split` decides on host how every
 * parent maps into the next level, `carry_parents` copies completed parents unchanged, and
 * `assign_bucket` assigns each bucket of split parents with tiled batched GEMMs. One input row of
 * a split parent emits `fanout` memberships, and each membership's occurrence advances by
 * `rank * occurrence_stride` so that across levels every copy of a row lands in a distinct
 * scaffold slot.
 *
 * Child keys are dense and sequential across the whole output, and the final stable sort by key
 * followed by a reduce-by-key compaction yields one contiguous range per child, which is the next
 * level's partition set.
 */
template <typename T>
auto split_manyway(raft::resources const& res,
                   raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
                   partition_set const& parents,
                   split_params const& params,
                   split_context& context) -> partition_set
{
  auto stream = raft::resource::get_cuda_stream(res);
  auto plan   = plan_split(
    parents.ranges, static_cast<int64_t>(parents.memberships.size()), params, context.seed);

  rmm::device_uvector<uint32_t> keys(static_cast<size_t>(plan.output_rows), stream);
  rmm::device_uvector<partition_membership> memberships(static_cast<size_t>(plan.output_rows),
                                                        stream);

  carry_parents(res, parents, plan, keys, memberships);
  auto buckets = bucket_split_parents(plan, params);
  for (size_t bucket_index = 0; bucket_index < buckets.size(); ++bucket_index) {
    if (buckets[bucket_index].empty()) { continue; }
    assign_bucket(res,
                  dataset,
                  parents,
                  buckets[bucket_index],
                  1 << bucket_index,
                  params,
                  context,
                  keys,
                  memberships);
  }

  thrust::stable_sort_by_key(thrust::cuda::par.on(stream),
                             thrust::device_pointer_cast(keys.data()),
                             thrust::device_pointer_cast(keys.data() + keys.size()),
                             thrust::device_pointer_cast(memberships.data()));
  auto ranges = collect_partition_ranges(res, keys.data(), plan.output_rows, plan.child_count);
  return {std::move(memberships), std::move(ranges)};
}

// ----------------------------------------------------------------------------
// Leaf processing: bounded leaf slicing and cross-input leaf KNN
// ----------------------------------------------------------------------------

/** Return true if one leaf of this dimension and leaf size goes into the GEMM workspace. */
template <typename T>
inline auto leaf_gemm_supported(int64_t dimension, uint32_t leaf_size) -> bool
{
  if (dimension <= 0 || dimension > std::numeric_limits<int>::max()) { return false; }

  int64_t storage_dimension  = dimension;
  size_t vector_element_size = sizeof(float);
  size_t gram_element_size   = sizeof(float);
  if constexpr (std::is_integral_v<T>) {
    // The below check is a little paranoid
    if (dimension > MAX_INTEGER_LEAF_DIMENSION) { return false; }
    storage_dimension   = (dimension + 3) & ~int64_t{3};
    vector_element_size = sizeof(int8_t);
    gram_element_size   = sizeof(int32_t);
  }

  size_t vector_elements = static_cast<size_t>(leaf_size) * storage_dimension;
  size_t gram_elements   = static_cast<size_t>(leaf_size) * leaf_size;
  return vector_elements * vector_element_size + gram_elements * gram_element_size <=
         GEMM_WORKSPACE_BYTES;
}

struct leaf_set {
  partition_set const* partitions = nullptr;
  std::vector<uint32_t> starts_host;
  std::vector<uint32_t> ends_host;
  rmm::device_uvector<uint32_t> starts;
  rmm::device_uvector<uint32_t> ends;
};

/** Divide final grouped partitions into consecutive bounded leaves, without geometric resplitting.
 *
 * This is really just a fallback for when the tree isn't deep enough, and will produce obviously
 * inferior leaves but at no additional cost.
 */
inline auto make_leaves(raft::resources const& res,
                        partition_set const& partitions,
                        uint32_t leaf_size) -> leaf_set
{
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<uint32_t> starts_host;
  std::vector<uint32_t> ends_host;
  starts_host.reserve((partitions.memberships.size() + leaf_size - 1) / leaf_size);
  ends_host.reserve(starts_host.capacity());

  int64_t covered = 0;
  for (auto const& range : partitions.ranges) {
    RAFT_EXPECTS(range.start == covered && range.end > range.start &&
                   range.end <= static_cast<int64_t>(partitions.memberships.size()),
                 "Fastener partition ranges must compactly cover all memberships");
    for (int64_t start = range.start; start < range.end; start += leaf_size) {
      starts_host.push_back(static_cast<uint32_t>(start));
      ends_host.push_back(static_cast<uint32_t>(
        std::min<int64_t>(range.end, start + static_cast<int64_t>(leaf_size))));
    }
    covered = range.end;
  }
  RAFT_EXPECTS(
    covered == static_cast<int64_t>(partitions.memberships.size()) && !starts_host.empty(),
    "Fastener partition ranges did not cover all memberships");

  rmm::device_uvector<uint32_t> starts(starts_host.size(), stream);
  rmm::device_uvector<uint32_t> ends(ends_host.size(), stream);
  raft::copy(starts.data(), starts_host.data(), starts.size(), stream);
  raft::copy(ends.data(), ends_host.data(), ends.size(), stream);
  return {
    &partitions, std::move(starts_host), std::move(ends_host), std::move(starts), std::move(ends)};
}

/** Fill every unwritten scaffold slot with its row ID; final prefix deduplication removes it. */
static __global__ void initialize_self_scaffold_kernel(uint32_t* graph,
                                                       int64_t rows,
                                                       int graph_degree)
{
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total  = rows * graph_degree;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; linear < total; linear += stride) {
    graph[linear] = static_cast<uint32_t>(linear / graph_degree);
  }
}

/** Copy the vectors of each leaf into a dense buffer of OutT, zero-padding rows past the leaf
 *  end and dimensions past `input_dim`. Unsigned input values move to a center of zero. */
template <typename T, typename OutT>
__global__ void manyway_gather_leaf_vectors_kernel(T const* dataset,
                                                   int64_t input_dim,
                                                   int64_t output_dim,
                                                   int leaf_size,
                                                   partition_membership const* memberships,
                                                   uint32_t const* leaf_starts,
                                                   uint32_t const* leaf_ends,
                                                   int64_t leaf_offset,
                                                   int64_t leaf_count,
                                                   OutT* leaf_vectors)
{
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t total  = leaf_count * leaf_size * output_dim;
  for (; linear < total; linear += stride) {
    int64_t d          = linear % output_dim;
    int64_t local_row  = (linear / output_dim) % leaf_size;
    int64_t local_leaf = linear / (output_dim * leaf_size);
    int64_t leaf       = leaf_offset + local_leaf;
    int64_t leaf_n     = static_cast<int64_t>(leaf_ends[leaf] - leaf_starts[leaf]);
    OutT value         = 0;
    if (local_row < leaf_n && d < input_dim) {
      uint32_t point = memberships[leaf_starts[leaf] + local_row].id;
      auto input     = dataset[static_cast<int64_t>(point) * input_dim + d];
      if constexpr (std::is_same_v<T, uint8_t>) {
        value = static_cast<OutT>(static_cast<int>(input) - 128);
      } else {
        value = static_cast<OutT>(input);
      }
    }
    leaf_vectors[linear] = value;
  }
}

/** Select the nearest cross-input neighbors of each point from the leaf Gram matrix. One block
 *  does one leaf. One thread does one point. */
template <typename GramT>
static __global__ void manyway_leaf_gram_knn_kernel(GramT const* gram,
                                                    partition_membership const* memberships,
                                                    uint32_t const* origins,
                                                    uint32_t const* leaf_starts,
                                                    uint32_t const* leaf_ends,
                                                    int64_t leaf_offset,
                                                    int64_t leaf_count,
                                                    int leaf_size,
                                                    int leaf_degree,
                                                    int union_degree,
                                                    uint32_t* graph)
{
  int64_t local_leaf = blockIdx.x;
  if (local_leaf >= leaf_count) { return; }
  int64_t leaf   = leaf_offset + local_leaf;
  uint32_t start = leaf_starts[leaf];
  int leaf_n     = static_cast<int>(leaf_ends[leaf] - start);
  if (leaf_n <= 1 || leaf_n > MAX_LEAF_SIZE) { return; }

  // Stage the leaf's memberships and origin labels once per block.
  __shared__ partition_membership records[MAX_LEAF_SIZE];
  __shared__ uint32_t leaf_origins[MAX_LEAF_SIZE];
  for (int i = threadIdx.x; i < leaf_n; i += blockDim.x) {
    records[i]      = memberships[start + i];
    leaf_origins[i] = origins[records[i].id];
  }
  __syncthreads();

  int u = threadIdx.x;
  if (u >= leaf_n) { return; }
  double top_d[MAX_LEAF_DEGREE];
  uint16_t top_v[MAX_LEAF_DEGREE];
  for (int t = 0; t < leaf_degree; ++t) {
    top_d[t] = std::numeric_limits<double>::max();
    top_v[t] = std::numeric_limits<uint16_t>::max();
  }

  // Scan this point's Gram row, keeping the `leaf_degree` nearest cross-origin neighbors.
  int64_t gram_base = local_leaf * leaf_size * leaf_size;
  double norm_u     = static_cast<double>(gram[gram_base + u * leaf_size + u]);
  for (int v = 0; v < leaf_n; ++v) {
    if (u == v || leaf_origins[u] == leaf_origins[v] || records[u].id == records[v].id) {
      continue;
    }
    double norm_v   = static_cast<double>(gram[gram_base + v * leaf_size + v]);
    double dot      = static_cast<double>(gram[gram_base + v * leaf_size + u]);
    double distance = norm_u + norm_v - 2.0 * dot;
    if (isfinite(distance)) { distance = fmax(0.0, distance); }
    int worst = 0;
    for (int t = 1; t < leaf_degree; ++t) {
      if (top_d[t] > top_d[worst] || (top_d[t] == top_d[worst] && top_v[t] > top_v[worst])) {
        worst = t;
      }
    }
    if (distance < top_d[worst] ||
        (distance == top_d[worst] && records[v].id < records[top_v[worst]].id)) {
      top_d[worst] = distance;
      top_v[worst] = static_cast<uint16_t>(v);
    }
  }

  // Pack the valid selections into this occurrence's slots of the scaffold graph.
  int selected = 0;
  for (int t = 0; t < leaf_degree; ++t) {
    bool valid = top_v[t] != std::numeric_limits<uint16_t>::max() && isfinite(top_d[t]);
    if (valid) {
      int64_t output = static_cast<int64_t>(records[u].id) * union_degree +
                       static_cast<int>(records[u].occurrence) * leaf_degree + selected;
      graph[output] = records[top_v[t]].id;
      ++selected;
    }
  }
}

/** Build directed cross-input nearest neighbors for every leaf occurrence. */
template <typename T>
auto build_leaf_neighbors(raft::resources const& res,
                          raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
                          leaf_set const& leaves,
                          uint32_t const* origins,
                          int union_degree,
                          build_params const& params) -> raft::device_matrix<uint32_t, int64_t>
{
  auto stream             = raft::resource::get_cuda_stream(res);
  int64_t rows            = dataset.extent(0);
  int leaf_size           = static_cast<int>(params.leaf_size);
  int leaf_degree         = static_cast<int>(params.leaf_degree);
  auto const& memberships = leaves.partitions->memberships;

  // Prefill every slot with its own row id, leaf KNN overwrites the slots it fills
  auto graph          = raft::make_device_matrix<uint32_t, int64_t>(res, rows, union_degree);
  int scaffold_blocks = strided_grid_size(rows * union_degree);
  initialize_self_scaffold_kernel<<<scaffold_blocks, 256, 0, stream>>>(
    graph.data_handle(), rows, union_degree);
  RAFT_CUDA_TRY(cudaGetLastError());

  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
    // Size the leaf batch so the vector and Gram buffers fit the GEMM workspace budget
    int64_t input_dimension         = dataset.extent(1);
    int dimension                   = static_cast<int>(input_dimension);
    size_t vector_elements_per_leaf = static_cast<size_t>(leaf_size) * dimension;
    size_t gram_elements_per_leaf   = static_cast<size_t>(leaf_size) * leaf_size;
    size_t bytes_per_leaf =
      vector_elements_per_leaf * sizeof(float) + gram_elements_per_leaf * sizeof(float);
    size_t batch_capacity = std::max<size_t>(
      1, std::min<size_t>(leaves.starts_host.size(), GEMM_WORKSPACE_BYTES / bytes_per_leaf));
    rmm::device_uvector<float> leaf_vectors(batch_capacity * vector_elements_per_leaf, stream);
    rmm::device_uvector<float> gram(batch_capacity * gram_elements_per_leaf, stream);
    auto vector_stride = static_cast<int64_t>(vector_elements_per_leaf);
    auto gram_stride   = static_cast<int64_t>(gram_elements_per_leaf);

    for (size_t leaf_offset = 0; leaf_offset < leaves.starts_host.size();
         leaf_offset += batch_capacity) {
      size_t batch_size = std::min(batch_capacity, leaves.starts_host.size() - leaf_offset);
      int gather_blocks =
        strided_grid_size(static_cast<int64_t>(batch_size * vector_elements_per_leaf));
      // Gather this batch's leaves into the dense vector buffer.
      manyway_gather_leaf_vectors_kernel<<<gather_blocks, 256, 0, stream>>>(
        dataset.data_handle(),
        input_dimension,
        input_dimension,
        leaf_size,
        memberships.data(),
        leaves.starts.data(),
        leaves.ends.data(),
        static_cast<int64_t>(leaf_offset),
        static_cast<int64_t>(batch_size),
        leaf_vectors.data());
      RAFT_CUDA_TRY(cudaGetLastError());
      // the Gram matrix of every leaf in the batch
      batched_row_dot_products(res,
                               leaf_vectors.data(),
                               leaf_size,
                               vector_stride,
                               leaf_vectors.data(),
                               leaf_size,
                               vector_stride,
                               gram.data(),
                               gram_stride,
                               dimension,
                               static_cast<int>(batch_size));
      // select each point's nearest cross-input neighbors from the corresponding Gram matrix
      manyway_leaf_gram_knn_kernel<float>
        <<<static_cast<int>(batch_size), leaf_size, 0, stream>>>(gram.data(),
                                                                 memberships.data(),
                                                                 origins,
                                                                 leaves.starts.data(),
                                                                 leaves.ends.data(),
                                                                 static_cast<int64_t>(leaf_offset),
                                                                 static_cast<int64_t>(batch_size),
                                                                 leaf_size,
                                                                 leaf_degree,
                                                                 union_degree,
                                                                 graph.data_handle());
      RAFT_CUDA_TRY(cudaGetLastError());
    }
  } else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
    // Same pipeline as the float path, with rows padded to a multiple of 4 and an INT32 Gram.
    int64_t input_dimension         = dataset.extent(1);
    int64_t padded_dimension        = (input_dimension + 3) & ~int64_t{3};
    size_t vector_elements_per_leaf = static_cast<size_t>(leaf_size) * padded_dimension;
    size_t gram_elements_per_leaf   = static_cast<size_t>(leaf_size) * leaf_size;
    size_t bytes_per_leaf =
      vector_elements_per_leaf * sizeof(int8_t) + gram_elements_per_leaf * sizeof(int32_t);
    size_t batch_capacity = std::max<size_t>(
      1, std::min<size_t>(leaves.starts_host.size(), GEMM_WORKSPACE_BYTES / bytes_per_leaf));
    rmm::device_uvector<int8_t> leaf_vectors(batch_capacity * vector_elements_per_leaf, stream);
    rmm::device_uvector<int32_t> gram(batch_capacity * gram_elements_per_leaf, stream);
    int dimension      = static_cast<int>(padded_dimension);
    auto vector_stride = static_cast<int64_t>(vector_elements_per_leaf);
    auto gram_stride   = static_cast<int64_t>(gram_elements_per_leaf);

    for (size_t leaf_offset = 0; leaf_offset < leaves.starts_host.size();
         leaf_offset += batch_capacity) {
      size_t batch_size = std::min(batch_capacity, leaves.starts_host.size() - leaf_offset);
      int gather_blocks =
        strided_grid_size(static_cast<int64_t>(batch_size * vector_elements_per_leaf));
      manyway_gather_leaf_vectors_kernel<<<gather_blocks, 256, 0, stream>>>(
        dataset.data_handle(),
        input_dimension,
        padded_dimension,
        leaf_size,
        memberships.data(),
        leaves.starts.data(),
        leaves.ends.data(),
        static_cast<int64_t>(leaf_offset),
        static_cast<int64_t>(batch_size),
        leaf_vectors.data());
      RAFT_CUDA_TRY(cudaGetLastError());
      batched_row_dot_products(res,
                               leaf_vectors.data(),
                               leaf_size,
                               vector_stride,
                               leaf_vectors.data(),
                               leaf_size,
                               vector_stride,
                               gram.data(),
                               gram_stride,
                               dimension,
                               static_cast<int>(batch_size));
      manyway_leaf_gram_knn_kernel<int32_t>
        <<<static_cast<int>(batch_size), leaf_size, 0, stream>>>(gram.data(),
                                                                 memberships.data(),
                                                                 origins,
                                                                 leaves.starts.data(),
                                                                 leaves.ends.data(),
                                                                 static_cast<int64_t>(leaf_offset),
                                                                 static_cast<int64_t>(batch_size),
                                                                 leaf_size,
                                                                 leaf_degree,
                                                                 union_degree,
                                                                 graph.data_handle());
      RAFT_CUDA_TRY(cudaGetLastError());
    }
  }

  return graph;
}

// ----------------------------------------------------------------------------
// Scaffold build driver
// ----------------------------------------------------------------------------

/** Initialize the source-index label for every dataset row. */
static __global__ void initialize_origins_kernel(uint32_t* origins,
                                                 int64_t start,
                                                 int64_t rows,
                                                 uint32_t origin)
{
  int64_t local_row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local_row >= rows) { return; }
  origins[start + local_row] = origin;
}

/** Calculate the squared L2 norm of each dataset row.
 *
 * This could be `raft::linalg::norm`, but half precision vectors might square to inf without a
 * wider accumulator.
 */
template <typename T>
__global__ void manyway_l2_norms_kernel(T const* dataset, int64_t rows, int64_t dim, float* norms)
{
  int lane    = threadIdx.x % WARP_SIZE;
  int warp    = threadIdx.x / WARP_SIZE;
  int64_t row = static_cast<int64_t>(blockIdx.x) * ROW_WARPS_PER_BLOCK + warp;
  if (row >= rows) { return; }

  // Each lane accumulates a strided slice of the row's squared values.
  float sum      = 0.0f;
  T const* point = dataset + row * dim;
  for (int64_t d = lane; d < dim; d += WARP_SIZE) {
    float value = static_cast<float>(point[d]);
    sum         = fmaf(value, value, sum);
  }
  // Shuffle-reduce the partial sums; lane 0 holds the total.
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (lane == 0) { norms[row] = sum; }
}

/** Build the many-way scaffold through splitting and leaf construction. */
template <typename T>
auto build(raft::resources const& res,
           raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
           std::vector<int64_t> const& offsets,
           build_params const& params = {}) -> raft::device_matrix<uint32_t, int64_t>
{
  auto stream  = raft::resource::get_cuda_stream(res);
  int64_t rows = dataset.extent(0);

  RAFT_EXPECTS(offsets.size() >= 3, "Fastener requires at least two input datasets");
  RAFT_EXPECTS(rows > 0, "Fastener row count must be positive");
  RAFT_EXPECTS(params.levels > 0, "Fastener levels must be positive");
  RAFT_EXPECTS(params.root_fanout >= 1 && params.root_fanout <= MAX_FANOUT &&
                 params.lower_fanout >= 1 && params.lower_fanout <= MAX_FANOUT,
               "Fastener fanouts must be between 1 and 32");
  RAFT_EXPECTS(params.leader_fraction > 0.0 && params.leader_fraction <= 1.0,
               "Fastener leader fraction must be in (0, 1]");
  RAFT_EXPECTS(params.max_leaders >= std::max(params.root_fanout, params.lower_fanout) &&
                 params.max_leaders <= MAX_LEADERS,
               "Fastener leader cap must cover both fanouts and not exceed 8192");
  RAFT_EXPECTS(params.leaf_size >= 1 && params.leaf_size <= MAX_LEAF_SIZE,
               "Fastener leaf size must be between 1 and %d",
               MAX_LEAF_SIZE);
  RAFT_EXPECTS(params.leaf_degree >= 1 && params.leaf_degree <= MAX_LEAF_DEGREE,
               "Fastener leaf degree must be between 1 and %d",
               MAX_LEAF_DEGREE);
  RAFT_EXPECTS(leaf_gemm_supported<T>(dataset.extent(1), params.leaf_size),
               "Fastener dataset dimension exceeds the leaf GEMM limits");

  // the number of leaf partitions that a given point will end up in
  uint64_t spill = params.root_fanout;
  for (uint32_t level = 1; level < params.levels; ++level) {
    RAFT_EXPECTS(spill <= std::numeric_limits<uint64_t>::max() / params.lower_fanout,
                 "Fastener spill width overflow");
    spill *= params.lower_fanout;
  }
  // this is because the degree of the candidate list is stored in uint8_t
  RAFT_EXPECTS(spill * params.leaf_degree <= std::numeric_limits<uint8_t>::max(),
               "Fastener candidate width must not exceed 255");
  RAFT_EXPECTS(static_cast<uint64_t>(rows) <= std::numeric_limits<uint32_t>::max() / spill,
               "Fastener total partition memberships (rows=%ld * spill=%lu) must fit in uint32_t",
               static_cast<long>(rows),
               spill);
  int union_degree = static_cast<int>(spill * params.leaf_degree);

  // Per-row index of the input partition this point came from, used to skip same-origin pairs
  rmm::device_uvector<uint32_t> origins(static_cast<size_t>(rows), stream);

  for (size_t part = 0; part + 1 < offsets.size(); ++part) {
    int64_t part_rows = offsets[part + 1] - offsets[part];
    int blocks        = static_cast<int>((part_rows + 255) / 256);
    initialize_origins_kernel<<<blocks, 256, 0, stream>>>(
      origins.data(), offsets[part], part_rows, static_cast<uint32_t>(part));
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  split_context context(stream, rows);
  int norm_blocks = static_cast<int>((rows + ROW_WARPS_PER_BLOCK - 1) / ROW_WARPS_PER_BLOCK);
  manyway_l2_norms_kernel<<<norm_blocks, ROW_WARPS_PER_BLOCK * WARP_SIZE, 0, stream>>>(
    dataset.data_handle(), rows, dataset.extent(1), context.norms.data());
  RAFT_CUDA_TRY(cudaGetLastError());

  // the actual splitting
  auto partitions            = make_root_partition(res, rows);
  uint32_t occurrence_stride = 1;
  for (uint32_t level = 0; level < params.levels; ++level) {
    uint32_t fanout = level == 0 ? params.root_fanout : params.lower_fanout;

    partitions = split_manyway(res,
                               dataset,
                               partitions,
                               split_params{.fanout            = fanout,
                                            .leader_fraction   = params.leader_fraction,
                                            .max_leaders       = params.max_leaders,
                                            .leaf_size         = params.leaf_size,
                                            .level             = level,
                                            .occurrence_stride = occurrence_stride},
                               context);
    occurrence_stride *= fanout;
  }

  // Leaf construction: only consecutive range slicing occurs after configured geometric depth.
  auto leaves   = make_leaves(res, partitions, params.leaf_size);
  auto scaffold = build_leaf_neighbors(res, dataset, leaves, origins.data(), union_degree, params);
  raft::resource::sync_stream(res);
  return scaffold;
}

// ----------------------------------------------------------------------------
// Candidate graph assembly: combine input graphs with the scaffold
// ----------------------------------------------------------------------------

/**
 * Offset-copy one input partition graph and append its global scaffold neighbors.
 */
static __global__ void copy_partition_with_scaffold_kernel(uint32_t const* source,
                                                           uint32_t const* scaffold,
                                                           int64_t source_rows,
                                                           int64_t source_degree,
                                                           uint32_t* destination,
                                                           int64_t destination_degree,
                                                           int64_t base_degree,
                                                           int64_t scaffold_degree,
                                                           uint32_t offset)
{
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= source_rows) { return; }
  int64_t source_base      = row * source_degree;
  int64_t global_row       = row + offset;
  int64_t destination_base = global_row * destination_degree;

  // Rows from lower-degree input graphs are cyclically repeated to the common base width, which is
  // not ideal (adds work to the sorting by distance) but practical applications for mismatched
  // degrees seem hard to imagine.
  for (int64_t j = 0; j < base_degree; ++j) {
    destination[destination_base + j] = source[source_base + (j % source_degree)] + offset;
  }
  for (int64_t j = 0; j < scaffold_degree; ++j) {
    destination[destination_base + base_degree + j] = scaffold[global_row * scaffold_degree + j];
  }
}

/**
 * @brief Form the pre-optimization candidate graph from input graphs and scaffold edges.
 *
 * The maximum input graph degree defines the base width, allowing partitions with mixed degrees.
 */
template <typename T, typename IdxT>
auto append_to_input_graphs(
  raft::resources const& res,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> const& indices,
  std::vector<int64_t> const& offsets,
  raft::device_matrix_view<const uint32_t, int64_t, raft::row_major> scaffold)
  -> raft::device_matrix<uint32_t, int64_t>
{
  auto stream         = raft::resource::get_cuda_stream(res);
  int64_t base_degree = 0;
  for (auto const* index : indices) {
    base_degree = std::max<int64_t>(base_degree, index->graph_degree());
  }
  int64_t scaffold_degree = scaffold.extent(1);
  int64_t graph_degree    = base_degree + scaffold_degree;
  auto graph = raft::make_device_matrix<uint32_t, int64_t>(res, scaffold.extent(0), graph_degree);

  for (size_t part = 0; part < indices.size(); ++part) {
    auto source = indices[part]->graph();
    RAFT_EXPECTS(source.extent(1) > 0, "Input CAGRA graphs must have nonzero degree");
    int blocks = static_cast<int>((source.extent(0) + 255) / 256);
    copy_partition_with_scaffold_kernel<<<blocks, 256, 0, stream>>>(
      source.data_handle(),
      scaffold.data_handle(),
      source.extent(0),
      source.extent(1),
      graph.data_handle(),
      graph_degree,
      base_degree,
      scaffold_degree,
      static_cast<uint32_t>(offsets[part]));
    RAFT_CUDA_TRY(cudaGetLastError());
  }
  raft::resource::sync_stream(res);
  return graph;
}

/**
 * Remove invalid, self, and duplicate candidates from each sorted row, retain the nearest prefix,
 * and cyclically pad short rows to the requested output width.
 *
 * Scaffold padding is measured to be negligible compared to the input graph edges.
 */
static __global__ void deduplicate_graph_prefix_kernel(uint32_t const* input,
                                                       int64_t rows,
                                                       int64_t input_degree,
                                                       uint32_t* output,
                                                       int64_t output_degree)
{
  constexpr int WARPS_PER_BLOCK = 256 / WARP_SIZE;
  int lane                      = threadIdx.x % WARP_SIZE;
  int warp                      = threadIdx.x / WARP_SIZE;
  int64_t row                   = static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (row >= rows) { return; }

  int64_t input_base  = row * input_degree;
  int64_t output_base = row * output_degree;
  int selected        = 0;
  for (int64_t tile = 0; tile < input_degree && selected < output_degree; tile += WARP_SIZE) {
    int64_t column     = tile + lane;
    bool first         = column < input_degree;
    uint32_t candidate = first ? input[input_base + column] : uint32_t{0};
    first              = first && candidate < rows && candidate != static_cast<uint32_t>(row);
    for (int64_t prior = 0; prior < column && first; ++prior) {
      if (input[input_base + prior] == candidate) { first = false; }
    }

    unsigned first_mask = __ballot_sync(0xffffffffu, first);
    unsigned lower_mask = lane == 0 ? 0u : (0xffffffffu >> (WARP_SIZE - lane));
    int output_column   = selected + __popc(first_mask & lower_mask);
    if (first && output_column < output_degree) { output[output_base + output_column] = candidate; }
    selected += __popc(first_mask);
  }

  if (selected == 0) {
    if (lane == 0) { output[output_base] = static_cast<uint32_t>((row + 1) % rows); }
    selected = 1;
  }
  if (selected > output_degree) { selected = static_cast<int>(output_degree); }
  for (int64_t column = selected + lane; column < output_degree; column += WARP_SIZE) {
    output[output_base + column] = output[output_base + (column % selected)];
  }
}

/** Deduplicate a metric-sorted graph and retain a fixed-width nearest-candidate prefix. */
inline auto cap_sorted_graph(
  raft::resources const& res,
  raft::device_matrix_view<const uint32_t, int64_t, raft::row_major> graph,
  int64_t output_degree) -> raft::device_matrix<uint32_t, int64_t>
{
  RAFT_EXPECTS(output_degree > 0 && output_degree <= graph.extent(1),
               "Pre-optimize graph degree cap must be within the sorted graph degree");
  auto output = raft::make_device_matrix<uint32_t, int64_t>(res, graph.extent(0), output_degree);
  constexpr int THREADS_PER_BLOCK = 256;
  constexpr int WARPS_PER_BLOCK   = THREADS_PER_BLOCK / WARP_SIZE;
  int blocks = static_cast<int>((graph.extent(0) + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
  deduplicate_graph_prefix_kernel<<<blocks,
                                    THREADS_PER_BLOCK,
                                    0,
                                    raft::resource::get_cuda_stream(res)>>>(
    graph.data_handle(), graph.extent(0), graph.extent(1), output.data_handle(), output_degree);
  RAFT_CUDA_TRY(cudaGetLastError());
  return output;
}

}  // namespace cuvs::neighbors::cagra::detail::merge_scaffold
