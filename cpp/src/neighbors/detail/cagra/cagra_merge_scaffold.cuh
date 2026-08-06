/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../neighbors_device_intrinsics.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/selection/select_k.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cublaslt_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/aligned.hpp>
#include <rmm/exec_policy.hpp>

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
#include <memory>
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
inline constexpr int THREADS_PER_BLOCK       = 256;
inline constexpr int MAX_STRIDED_GRID_BLOCKS = 1 << 20;
/** Warps per block in the one-row-per-warp kernels; their launch geometry derives from this. */
inline constexpr int ROW_WARPS_PER_BLOCK = 4;

/** Grid size for the grid-stride kernels: blocks of THREADS_PER_BLOCK covering `items`, capped so
 * oversized workloads loop within resident threads instead. */
inline auto strided_grid_size(int64_t items) -> int
{
  return static_cast<int>(std::min<int64_t>(
    raft::div_rounding_up_safe<int64_t>(items, THREADS_PER_BLOCK), MAX_STRIDED_GRID_BLOCKS));
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
  uint32_t max_leaders   = 1024;
  uint32_t leaf_size     = MAX_LEAF_SIZE;
  uint32_t leaf_degree   = 4;
};

/** Controls for one invocation of the reusable many-way split boundary. */
struct split_params {
  uint32_t fanout            = 1;
  double leader_fraction     = 0.02;
  uint32_t max_leaders       = 1024;
  uint32_t leaf_size         = MAX_LEAF_SIZE;
  uint32_t level             = 0;
  uint32_t occurrence_stride = 1;
};

/** Device state and tuning knobs shared by every split level: the precomputed row norms, the
 * tiling and workspace capacities, and the seed feeding the deterministic leader samples. */
struct split_context {
  /** Norms are dataset-sized, so they come from the large workspace resource rather than the
   *  bounded one. Taking `raft::resources` lets the caller control where this memory lives. */
  split_context(raft::resources const& res, int64_t rows, int64_t dim)
    : norms(raft::make_device_mdarray<float, int64_t>(
        res,
        raft::resource::get_large_workspace_resource_ref(res),
        raft::make_extents<int64_t>(rows))),
      logical_dim(dim)
  {
  }

  raft::device_vector<float, int64_t> norms;
  /** Logical dimension of the dataset. The dataset view's extent(1) is its row pitch, which
   *  may exceed this when the consolidated dataset is padded for CAGRA row alignment. */
  int64_t logical_dim;
  int assignment_tile_rows = ASSIGNMENT_TILE_ROWS;
  uint64_t seed            = DETERMINISTIC_SEED;
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

/** Memberships live on the device; their grouping into contiguous ranges is host-side metadata.
 *  Both are fixed-size once built, so raft mdarrays are a good fit -- but note they are neither
 *  default-constructible nor resizable, so every producer sizes them exactly at construction. */
struct partition_set {
  raft::device_vector<partition_membership, int64_t> memberships;
  raft::host_vector<partition_range, int64_t> ranges;
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
inline auto plan_split(raft::host_vector<partition_range, int64_t> const& ranges,
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
  plan.parents.reserve(static_cast<size_t>(ranges.size()));
  int64_t covered     = 0;
  int64_t output_rows = 0;
  int64_t child_keys  = 0;

  for (size_t parent_index = 0; parent_index < static_cast<size_t>(ranges.size()); ++parent_index) {
    auto const& parent = ranges(static_cast<int64_t>(parent_index));
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
  // Memberships are dataset-sized, so they come from the large workspace rather than the bounded
  // one; the single root range is host metadata.
  auto memberships = raft::make_device_mdarray<partition_membership, int64_t>(
    res, raft::resource::get_large_workspace_resource_ref(res), raft::make_extents<int64_t>(rows));
  auto ranges = raft::make_host_vector<partition_range, int64_t>(res, 1);
  ranges(0)   = partition_range{uint32_t{0}, int64_t{0}, rows};
  partition_set root{std::move(memberships), std::move(ranges)};
  auto blocks = static_cast<int>(raft::div_rounding_up_safe<int64_t>(rows, THREADS_PER_BLOCK));
  initialize_root_memberships_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    root.memberships.data_handle(), rows);
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
                                                  int64_t row_stride,
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
      value       = static_cast<float>(dataset[static_cast<int64_t>(id) * row_stride + d]);
    }
    output[linear] = value;
  }
}

/** Copy the leader vectors of each tile into a dense float buffer and record the leader IDs. */
template <typename T>
__global__ void manyway_gather_tile_leaders_kernel(T const* dataset,
                                                   int64_t dim,
                                                   int64_t row_stride,
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
      value       = static_cast<float>(dataset[static_cast<int64_t>(id) * row_stride + d]);
      if (d == 0) { leader_ids[static_cast<int64_t>(batch) * padded_leaders + leader] = id; }
    }
    output[linear] = value;
  }
}

/** Convert the batched point-leader dot products into squared L2 distances in place. Invalid
 * padded rows and leaders become infinity so the generic selection primitive ignores them. */
static __global__ void manyway_materialize_tile_distances_kernel(
  float* dots,
  int batch_size,
  int tile_rows,
  int padded_leaders,
  float const* norms,
  uint32_t const* leader_ids,
  partition_membership const* input_memberships,
  assignment_tile const* tiles)
{
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t total  = static_cast<int64_t>(batch_size) * tile_rows * padded_leaders;
  for (; linear < total; linear += stride) {
    int leader     = static_cast<int>(linear % padded_leaders);
    int row        = static_cast<int>((linear / padded_leaders) % tile_rows);
    int batch      = static_cast<int>(linear / (static_cast<int64_t>(padded_leaders) * tile_rows));
    auto tile      = tiles[batch];
    float distance = std::numeric_limits<float>::infinity();
    if (row < tile.rows && leader < tile.leader_count) {
      auto membership = input_memberships[tile.input_start + row];
      auto leader_id =
        leader_ids[static_cast<int64_t>(batch) * padded_leaders + static_cast<int64_t>(leader)];
      // This could omit the point norm because it is constant across every leader in a row and
      // select_k only depends on their relative ordering. Doing so would also require removing the
      // nonnegative clamp because the resulting ranking scores may legitimately be negative.
      // I'm not doing this for the sake of simplicity and because it would be unlikely to have a
      // performance impact; we still need those norms for sorting the edges by distance.
      distance = fmaxf(0.0f, norms[membership.id] + norms[leader_id] - 2.0f * dots[linear]);
    }
    dots[linear] = distance;
  }
}

/** Write selected leader offsets as child keys and copy their point memberships. */
static __global__ void manyway_emit_tile_assignments_kernel(
  int const* selected_leaders,
  int batch_size,
  int tile_rows,
  int fanout,
  int occurrence_stride,
  partition_membership const* input_memberships,
  assignment_tile const* tiles,
  uint32_t* output_keys,
  partition_membership* output_memberships)
{
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t total  = static_cast<int64_t>(batch_size) * tile_rows;
  for (; linear < total; linear += stride) {
    int row   = static_cast<int>(linear % tile_rows);
    int batch = static_cast<int>(linear / tile_rows);
    auto tile = tiles[batch];
    if (row >= tile.rows) { continue; }

    auto membership       = input_memberships[tile.input_start + row];
    int64_t output_base   = tile.output_start + static_cast<int64_t>(row) * fanout;
    int64_t selected_base = linear * fanout;
    for (int rank = 0; rank < fanout; ++rank) {
      auto leader = static_cast<uint32_t>(selected_leaders[selected_base + rank]);
      output_keys[output_base + rank]        = tile.child_key_base + leader;
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
                                     uint32_t key_count)
  -> raft::host_vector<partition_range, int64_t>
{
  auto stream             = raft::resource::get_cuda_stream(res);
  auto output_capacity    = std::min<int64_t>(count, key_count);
  auto large_mr           = raft::resource::get_large_workspace_resource_ref(res);
  auto device_unique_keys = raft::make_device_mdarray<uint32_t, int64_t>(
    res, large_mr, raft::make_extents<int64_t>(output_capacity));
  auto device_counts = raft::make_device_mdarray<uint32_t, int64_t>(
    res, large_mr, raft::make_extents<int64_t>(output_capacity));
  rmm::exec_policy_nosync thrust_policy{stream, large_mr};
  auto reduced_end =
    thrust::reduce_by_key(thrust_policy,
                          thrust::device_pointer_cast(keys),
                          thrust::device_pointer_cast(keys + count),
                          thrust::make_constant_iterator(uint32_t{1}),
                          thrust::device_pointer_cast(device_unique_keys.data_handle()),
                          thrust::device_pointer_cast(device_counts.data_handle()));
  auto group_count = static_cast<int64_t>(
    reduced_end.first - thrust::device_pointer_cast(device_unique_keys.data_handle()));

  auto unique_keys = raft::make_host_vector<uint32_t, int64_t>(res, group_count);
  auto counts      = raft::make_host_vector<uint32_t, int64_t>(res, group_count);
  raft::copy(unique_keys.data_handle(), device_unique_keys.data_handle(), group_count, stream);
  raft::copy(counts.data_handle(), device_counts.data_handle(), group_count, stream);
  raft::resource::sync_stream(res);

  auto groups    = raft::make_host_vector<partition_range, int64_t>(res, group_count);
  int64_t cursor = 0;
  for (int64_t index = 0; index < group_count; ++index) {
    RAFT_EXPECTS(unique_keys(index) < key_count, "Many-way group key is out of range");
    groups(index) = {unique_keys(index), cursor, cursor + counts(index)};
    cursor += counts(index);
  }
  RAFT_EXPECTS(cursor == count, "Many-way partition membership histogram lost entries");
  return groups;
}

/** Copy every carried parent's memberships unchanged. */
inline void carry_parents(raft::resources const& res,
                          partition_set const& parents,
                          split_plan const& plan,
                          raft::device_vector<uint32_t, int64_t>& keys,
                          raft::device_vector<partition_membership, int64_t>& memberships)
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

  auto stream         = raft::resource::get_cuda_stream(res);
  auto device_carries = raft::make_device_mdarray<carry_span, int64_t>(
    res,
    raft::resource::get_large_workspace_resource_ref(res),
    raft::make_extents<int64_t>(static_cast<int64_t>(carries.size())));
  raft::copy(device_carries.data_handle(), carries.data(), carries.size(), stream);
  carry_completed_parents_kernel<<<static_cast<int>(carries.size()),
                                   THREADS_PER_BLOCK,
                                   0,
                                   stream>>>(parents.memberships.data_handle(),
                                             device_carries.data_handle(),
                                             memberships.data_handle(),
                                             keys.data_handle());
  RAFT_CUDA_TRY(cudaGetLastError());
}

/**
 * Compute type for both scaffold GEMMs.
 *
 * TF32 inputs cost nothing for three of the four supported dataset types: int8_t, uint8_t and half
 * values are all exactly representable in TF32's 11-bit significand, and the accumulator stays
 * float, so only float datasets lose input precision. Both consumers form
 * |u|^2 + |v|^2 - 2 u.v, which cancels for near-duplicate points, but they only rank merge
 * candidates; every surviving edge is re-scored at full precision by launch_sort_knn_graph before
 * the graph is optimized.
 *
 * Wider tensor-core modes are not worth their precision. These shapes are bandwidth-bound once off
 * the FP32 path, so CUBLAS_COMPUTE_32F_FAST_16BF measures no faster than TF32 despite roughly
 * double the arithmetic peak.
 */
inline constexpr cublasComputeType_t GEMM_COMPUTE_TYPE = CUBLAS_COMPUTE_32F_FAST_TF32;

/**
 * Compute every pairwise dot product between the rows of A_i and the rows of B_i for a strided
 * batch of row-major matrices with rows of length `row_width`:
 * out_i[b * a_rows + a] = A_i[a] . B_i[b].
 *
 * Pretty much a wrapper around strided-batched `cublasLtMatmul`.
 *
 * Every dataset scalar type is gathered to float before this call. Native INT8 cuBLAS paths are
 * not portable across architectures (e.g. Ada returns CUBLAS_STATUS_NOT_SUPPORTED for
 * CUDA_R_8I / CUBLAS_COMPUTE_32I strided-batched GEMM), and would gain little here in any case:
 * these shapes run at the memory roofline, where shrinking the operands without shrinking the
 * float output recovers well under the type's arithmetic advantage.
 */
inline void batched_row_dot_products(raft::resources const& res,
                                     float const* a,
                                     int a_rows,
                                     long long a_stride,
                                     float const* b,
                                     int b_rows,
                                     long long b_stride,
                                     float* out,
                                     long long out_stride,
                                     int row_width,
                                     int batch_count)
{
  float alpha = 1.0f;
  float beta  = 0.0f;

  using matmul_descriptor = std::unique_ptr<std::remove_pointer_t<cublasLtMatmulDesc_t>,
                                            decltype(&cublasLtMatmulDescDestroy)>;
  using matrix_layout     = std::unique_ptr<std::remove_pointer_t<cublasLtMatrixLayout_t>,
                                            decltype(&cublasLtMatrixLayoutDestroy)>;

  cublasLtMatmulDesc_t operation_raw = nullptr;
  RAFT_CUBLAS_TRY(cublasLtMatmulDescCreate(&operation_raw, GEMM_COMPUTE_TYPE, CUDA_R_32F));
  matmul_descriptor operation{operation_raw, &cublasLtMatmulDescDestroy};
  cublasOperation_t transpose = CUBLAS_OP_T;
  RAFT_CUBLAS_TRY(cublasLtMatmulDescSetAttribute(
    operation.get(), CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));

  auto make_layout = [batch_count](uint64_t rows,
                                   uint64_t columns,
                                   int64_t leading_dimension,
                                   int64_t batch_stride) {
    cublasLtMatrixLayout_t raw = nullptr;
    RAFT_CUBLAS_TRY(cublasLtMatrixLayoutCreate(&raw, CUDA_R_32F, rows, columns, leading_dimension));
    matrix_layout layout{raw, &cublasLtMatrixLayoutDestroy};
    auto count = static_cast<int32_t>(batch_count);
    RAFT_CUBLAS_TRY(cublasLtMatrixLayoutSetAttribute(
      raw, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)));
    RAFT_CUBLAS_TRY(cublasLtMatrixLayoutSetAttribute(
      raw, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride, sizeof(batch_stride)));
    return layout;
  };
  auto a_layout = make_layout(
    static_cast<uint64_t>(row_width), static_cast<uint64_t>(a_rows), row_width, a_stride);
  auto b_layout = make_layout(
    static_cast<uint64_t>(row_width), static_cast<uint64_t>(b_rows), row_width, b_stride);
  auto out_layout =
    make_layout(static_cast<uint64_t>(a_rows), static_cast<uint64_t>(b_rows), a_rows, out_stride);

  RAFT_CUBLAS_TRY(cublasLtMatmul(raft::resource::get_cublaslt_handle(res),
                                 operation.get(),
                                 &alpha,
                                 a,
                                 a_layout.get(),
                                 b,
                                 b_layout.get(),
                                 &beta,
                                 out,
                                 out_layout.get(),
                                 out,
                                 out_layout.get(),
                                 nullptr,
                                 nullptr,
                                 0,
                                 raft::resource::get_cuda_stream(res)));
}

/**
 * Assign every row of one bucket's split parents to its `fanout` nearest leaders.
 *
 * Parents are cut into tiles of `assignment_tile_rows` rows; every tile in a bucket shares the
 * same padded leader count, so one strided batched GEMM per batch produces all point-leader dot
 * products. Tiles are processed in batches sized to the GEMM workspace: each batch gathers its
 * tile vectors and its parents' leader vectors into dense float buffers, converts dots to
 * distances with the precomputed row norms (|x|^2 + |l|^2 - 2 x.l), uses `select_k` to keep the
 * `fanout` nearest leaders per row, and writes the child keys and memberships.
 */
template <typename T>
void assign_bucket(raft::resources const& res,
                   raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
                   partition_set const& parents,
                   std::vector<parent_plan const*> const& bucket,
                   int padded_leaders,
                   split_params const& params,
                   split_context& context,
                   raft::device_vector<uint32_t, int64_t>& keys,
                   raft::device_vector<partition_membership, int64_t>& memberships)
{
  auto stream           = raft::resource::get_cuda_stream(res);
  auto const row_stride = dataset.extent(1);
  auto dim              = context.logical_dim;
  auto workspace_mr     = raft::resource::get_workspace_resource_ref(res);
  auto workspace_bytes  = raft::resource::get_workspace_free_bytes(res);

  // RMM's limiting workspace accounts for every allocation rounded to the CUDA allocation
  // alignment. Model the seven explicit buffers independently so a raw-byte sum cannot overrun a
  // small caller-configured workspace through alignment alone. RAFT select-k also allocates value
  // and index scratch; each is no larger than its input matrix, so reserve two aligned dot-matrix
  // buffers as a conservative upper bound.
  auto assignment_workspace_bytes = [&](size_t capacity, size_t rows_per_tile) {
    auto aligned = [](size_t bytes) {
      return rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    };
    size_t point_elements    = rows_per_tile * static_cast<size_t>(dim);
    size_t leader_elements   = static_cast<size_t>(padded_leaders) * static_cast<size_t>(dim);
    size_t dot_elements      = rows_per_tile * static_cast<size_t>(padded_leaders);
    size_t selected_elements = rows_per_tile * static_cast<size_t>(params.fanout);
    return aligned(capacity * sizeof(assignment_tile)) +
           aligned(capacity * point_elements * sizeof(float)) +
           aligned(capacity * leader_elements * sizeof(float)) +
           aligned(capacity * dot_elements * sizeof(float)) +
           aligned(capacity * static_cast<size_t>(padded_leaders) * sizeof(uint32_t)) +
           aligned(capacity * selected_elements * sizeof(float)) +
           aligned(capacity * selected_elements * sizeof(int)) +
           aligned(capacity * dot_elements * sizeof(float)) +
           aligned(capacity * dot_elements * sizeof(int));
  };

  RAFT_EXPECTS(assignment_workspace_bytes(1, 1) <= workspace_bytes,
               "Fastener assignment workspace cannot fit the leader matrix and a single point row");
  size_t min_tile_rows = 1;
  size_t max_tile_rows = static_cast<size_t>(context.assignment_tile_rows);
  while (min_tile_rows < max_tile_rows) {
    size_t candidate = min_tile_rows + (max_tile_rows - min_tile_rows + 1) / 2;
    if (assignment_workspace_bytes(1, candidate) <= workspace_bytes) {
      min_tile_rows = candidate;
    } else {
      max_tile_rows = candidate - 1;
    }
  }
  int tile_rows = static_cast<int>(min_tile_rows);

  // Cut every parent in the bucket into fixed-height tiles
  std::vector<assignment_tile> tiles;
  for (auto const* entry : bucket) {
    for (int64_t start = entry->parent.start; start < entry->parent.end; start += tile_rows) {
      tiles.push_back(make_tile(*entry, start, params.fanout, tile_rows));
    }
  }

  // Size the batch so all per-tile buffers fit in the GEMM workspace budget
  size_t point_elements     = static_cast<size_t>(tile_rows) * dim;
  size_t leader_elements    = static_cast<size_t>(padded_leaders) * dim;
  size_t dot_elements       = static_cast<size_t>(tile_rows) * padded_leaders;
  size_t selected_elements  = static_cast<size_t>(tile_rows) * params.fanout;
  size_t min_batch_capacity = 1;
  size_t max_batch_capacity = tiles.size();
  while (min_batch_capacity < max_batch_capacity) {
    size_t candidate = min_batch_capacity + (max_batch_capacity - min_batch_capacity + 1) / 2;
    if (assignment_workspace_bytes(candidate, static_cast<size_t>(tile_rows)) <= workspace_bytes) {
      min_batch_capacity = candidate;
    } else {
      max_batch_capacity = candidate - 1;
    }
  }
  size_t batch_capacity = min_batch_capacity;

  auto device_tiles = raft::make_device_mdarray<assignment_tile, int64_t>(
    res, workspace_mr, raft::make_extents<int64_t>(static_cast<int64_t>(batch_capacity)));
  auto tile_points = raft::make_device_mdarray<float, int64_t>(
    res,
    workspace_mr,
    raft::make_extents<int64_t>(static_cast<int64_t>(batch_capacity * point_elements)));
  auto tile_leaders = raft::make_device_mdarray<float, int64_t>(
    res,
    workspace_mr,
    raft::make_extents<int64_t>(static_cast<int64_t>(batch_capacity * leader_elements)));
  auto tile_dots = raft::make_device_mdarray<float, int64_t>(
    res,
    workspace_mr,
    raft::make_extents<int64_t>(static_cast<int64_t>(batch_capacity * dot_elements)));
  auto tile_leader_ids = raft::make_device_mdarray<uint32_t, int64_t>(
    res,
    workspace_mr,
    raft::make_extents<int64_t>(static_cast<int64_t>(batch_capacity) * padded_leaders));
  auto selected_distances = raft::make_device_mdarray<float, int64_t>(
    res,
    workspace_mr,
    raft::make_extents<int64_t>(static_cast<int64_t>(batch_capacity * selected_elements)));
  auto selected_leaders = raft::make_device_mdarray<int, int64_t>(
    res,
    workspace_mr,
    raft::make_extents<int64_t>(static_cast<int64_t>(batch_capacity * selected_elements)));

  auto point_stride  = static_cast<int64_t>(point_elements);
  auto leader_stride = static_cast<int64_t>(leader_elements);
  auto dot_stride    = static_cast<int64_t>(dot_elements);

  for (size_t tile_offset = 0; tile_offset < tiles.size(); tile_offset += batch_capacity) {
    size_t batch_size = std::min(batch_capacity, tiles.size() - tile_offset);
    raft::copy(device_tiles.data_handle(), tiles.data() + tile_offset, batch_size, stream);

    // Gather the batch's tile rows and leader vectors
    int point_blocks = strided_grid_size(static_cast<int64_t>(batch_size * point_elements));
    manyway_gather_tile_points_kernel<<<point_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      dataset.data_handle(),
      dim,
      row_stride,
      parents.memberships.data_handle(),
      device_tiles.data_handle(),
      static_cast<int>(batch_size),
      tile_rows,
      tile_points.data_handle());
    RAFT_CUDA_TRY(cudaGetLastError());

    int leader_blocks = strided_grid_size(static_cast<int64_t>(batch_size * leader_elements));
    manyway_gather_tile_leaders_kernel<<<leader_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      dataset.data_handle(),
      dim,
      row_stride,
      parents.memberships.data_handle(),
      device_tiles.data_handle(),
      static_cast<int>(batch_size),
      padded_leaders,
      tile_leaders.data_handle(),
      tile_leader_ids.data_handle());
    RAFT_CUDA_TRY(cudaGetLastError());

    // point-leader dot products for the batch
    batched_row_dot_products(res,
                             tile_leaders.data_handle(),
                             padded_leaders,
                             leader_stride,
                             tile_points.data_handle(),
                             tile_rows,
                             point_stride,
                             tile_dots.data_handle(),
                             dot_stride,
                             static_cast<int>(dim),
                             static_cast<int>(batch_size));

    // Materialize distances, keep each row's nearest leaders, and emit their memberships
    int64_t selection_rows = static_cast<int64_t>(batch_size) * tile_rows;
    int distance_blocks    = strided_grid_size(selection_rows * padded_leaders);
    manyway_materialize_tile_distances_kernel<<<distance_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      tile_dots.data_handle(),
      static_cast<int>(batch_size),
      tile_rows,
      padded_leaders,
      context.norms.data_handle(),
      tile_leader_ids.data_handle(),
      parents.memberships.data_handle(),
      device_tiles.data_handle());
    RAFT_CUDA_TRY(cudaGetLastError());

    cuvs::selection::select_k(res,
                              raft::make_device_matrix_view<const float, int64_t>(
                                tile_dots.data_handle(), selection_rows, padded_leaders),
                              std::nullopt,
                              raft::make_device_matrix_view<float, int64_t>(
                                selected_distances.data_handle(), selection_rows, params.fanout),
                              raft::make_device_matrix_view<int, int64_t>(
                                selected_leaders.data_handle(), selection_rows, params.fanout),
                              true,
                              true);

    int assignment_blocks = strided_grid_size(selection_rows);
    manyway_emit_tile_assignments_kernel<<<assignment_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      selected_leaders.data_handle(),
      static_cast<int>(batch_size),
      tile_rows,
      static_cast<int>(params.fanout),
      static_cast<int>(params.occurrence_stride),
      parents.memberships.data_handle(),
      device_tiles.data_handle(),
      keys.data_handle(),
      memberships.data_handle());
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

  // Both scale with the dataset, so they come from the large workspace.
  auto const large_mr = raft::resource::get_large_workspace_resource_ref(res);
  auto keys           = raft::make_device_mdarray<uint32_t, int64_t>(
    res, large_mr, raft::make_extents<int64_t>(plan.output_rows));
  auto memberships = raft::make_device_mdarray<partition_membership, int64_t>(
    res, large_mr, raft::make_extents<int64_t>(plan.output_rows));

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

  rmm::exec_policy_nosync thrust_policy{stream, large_mr};
  thrust::stable_sort_by_key(thrust_policy,
                             thrust::device_pointer_cast(keys.data_handle()),
                             thrust::device_pointer_cast(keys.data_handle() + keys.size()),
                             thrust::device_pointer_cast(memberships.data_handle()));
  auto ranges =
    collect_partition_ranges(res, keys.data_handle(), plan.output_rows, plan.child_count);
  return {std::move(memberships), std::move(ranges)};
}

// ----------------------------------------------------------------------------
// Leaf processing: bounded leaf slicing and cross-input leaf KNN
// ----------------------------------------------------------------------------

/** Return true if one leaf of this dimension and leaf size fits the float GEMM workspace.
 *
 * The limit does not depend on the dataset scalar type: every type, integers included, is gathered
 * into float leaf vectors and produces a float Gram matrix. */
inline auto leaf_gemm_supported(int64_t dimension, uint32_t leaf_size, size_t workspace_bytes)
  -> bool
{
  if (dimension <= 0 || dimension > std::numeric_limits<int>::max()) { return false; }

  size_t vector_elements = static_cast<size_t>(leaf_size) * static_cast<size_t>(dimension);
  size_t gram_elements   = static_cast<size_t>(leaf_size) * leaf_size;
  return rmm::align_up(vector_elements * sizeof(float), rmm::CUDA_ALLOCATION_ALIGNMENT) +
           rmm::align_up(gram_elements * sizeof(float), rmm::CUDA_ALLOCATION_ALIGNMENT) <=
         workspace_bytes;
}

/** Leaves as strided views into `partitions->memberships`: leaf `i` holds the `counts[i]` records
 * at `starts[i] + k * strides[i]`. A unit stride is a plain contiguous slice. */
struct leaf_set {
  partition_set const* partitions = nullptr;
  std::vector<uint32_t> starts_host;
  std::vector<uint32_t> counts_host;
  std::vector<uint32_t> strides_host;
  raft::device_vector<uint32_t, int64_t> starts;
  raft::device_vector<uint32_t, int64_t> counts;
  raft::device_vector<uint32_t, int64_t> strides;
};

/** Divide final grouped partitions into bounded leaves, without geometric resplitting.
 *
 * This is really just a fallback for when the tree isn't deep enough, and will produce obviously
 * inferior leaves but at no additional cost.
 *
 * A partition's memberships are ascending in consolidated row id (the root is the identity and
 * every regroup is a stable sort), and origins are contiguous row-id blocks, so a partition is
 * always origin-sorted. Slicing it into consecutive chunks can therefore hand the leaf kernel a
 * single-origin leaf, and that kernel skips every same-origin pair -- such a leaf contributes no
 * cross-input edges at all. Dealing the members round-robin across the same number of leaves
 * spreads every origin block of length >= the leaf count over every leaf instead. A range that
 * already fits in one leaf yields a unit stride, i.e. exactly the consecutive slice.
 */
inline auto make_leaves(raft::resources const& res,
                        partition_set const& partitions,
                        uint32_t leaf_size) -> leaf_set
{
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<uint32_t> starts_host;
  std::vector<uint32_t> counts_host;
  std::vector<uint32_t> strides_host;
  starts_host.reserve((partitions.memberships.size() + leaf_size - 1) / leaf_size);
  counts_host.reserve(starts_host.capacity());
  strides_host.reserve(starts_host.capacity());

  int64_t covered = 0;
  for (int64_t range_index = 0; range_index < partitions.ranges.extent(0); ++range_index) {
    auto const& range = partitions.ranges(range_index);
    RAFT_EXPECTS(range.start == covered && range.end > range.start &&
                   range.end <= static_cast<int64_t>(partitions.memberships.size()),
                 "Fastener partition ranges must compactly cover all memberships");
    // Deal round-robin: leaf j takes local positions j, j + leaves, j + 2 * leaves, ... Leaf sizes
    // differ by at most one and none exceeds leaf_size, so this also avoids the tiny trailing leaf
    // that consecutive slicing leaves behind (a one-row leaf contributes nothing at all).
    int64_t const size   = range.end - range.start;
    int64_t const leaves = (size + static_cast<int64_t>(leaf_size) - 1) / leaf_size;
    for (int64_t leaf = 0; leaf < leaves; ++leaf) {
      starts_host.push_back(static_cast<uint32_t>(range.start + leaf));
      counts_host.push_back(static_cast<uint32_t>((size - leaf + leaves - 1) / leaves));
      strides_host.push_back(static_cast<uint32_t>(leaves));
    }
    covered = range.end;
  }
  RAFT_EXPECTS(
    covered == static_cast<int64_t>(partitions.memberships.size()) && !starts_host.empty(),
    "Fastener partition ranges did not cover all memberships");

  auto large_mr = raft::resource::get_large_workspace_resource_ref(res);
  auto starts   = raft::make_device_mdarray<uint32_t, int64_t>(
    res, large_mr, raft::make_extents<int64_t>(static_cast<int64_t>(starts_host.size())));
  auto counts = raft::make_device_mdarray<uint32_t, int64_t>(
    res, large_mr, raft::make_extents<int64_t>(static_cast<int64_t>(counts_host.size())));
  auto strides = raft::make_device_mdarray<uint32_t, int64_t>(
    res, large_mr, raft::make_extents<int64_t>(static_cast<int64_t>(strides_host.size())));
  raft::copy(starts.data_handle(), starts_host.data(), starts.size(), stream);
  raft::copy(counts.data_handle(), counts_host.data(), counts.size(), stream);
  raft::copy(strides.data_handle(), strides_host.data(), strides.size(), stream);
  return {&partitions,
          std::move(starts_host),
          std::move(counts_host),
          std::move(strides_host),
          std::move(starts),
          std::move(counts),
          std::move(strides)};
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
 *  end and dimensions past `input_dim`. Every scalar type is promoted to OutT (float) as-is. */
template <typename T, typename OutT>
__global__ void manyway_gather_leaf_vectors_kernel(T const* dataset,
                                                   int64_t input_dim,
                                                   int64_t row_stride,
                                                   int64_t output_dim,
                                                   int leaf_size,
                                                   partition_membership const* memberships,
                                                   uint32_t const* leaf_starts,
                                                   uint32_t const* leaf_counts,
                                                   uint32_t const* leaf_strides,
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
    int64_t leaf_n     = static_cast<int64_t>(leaf_counts[leaf]);
    OutT value         = 0;
    if (local_row < leaf_n && d < input_dim) {
      uint32_t point =
        memberships[leaf_starts[leaf] + local_row * static_cast<int64_t>(leaf_strides[leaf])].id;
      auto input = dataset[static_cast<int64_t>(point) * row_stride + d];
      value      = static_cast<OutT>(input);
    }
    leaf_vectors[linear] = value;
  }
}

/** Select the nearest cross-input neighbors of each point from the leaf Gram matrix. One block
 *  does one leaf. One thread does one point. */
static __global__ void manyway_leaf_gram_knn_kernel(float const* gram,
                                                    partition_membership const* memberships,
                                                    uint32_t const* origins,
                                                    uint32_t const* leaf_starts,
                                                    uint32_t const* leaf_counts,
                                                    uint32_t const* leaf_strides,
                                                    int64_t leaf_offset,
                                                    int64_t leaf_count,
                                                    int leaf_size,
                                                    int leaf_degree,
                                                    int union_degree,
                                                    uint32_t* graph)
{
  int64_t local_leaf = blockIdx.x;
  if (local_leaf >= leaf_count) { return; }
  int64_t leaf         = leaf_offset + local_leaf;
  uint32_t start       = leaf_starts[leaf];
  uint32_t leaf_stride = leaf_strides[leaf];
  int leaf_n           = static_cast<int>(leaf_counts[leaf]);
  if (leaf_n <= 1 || leaf_n > MAX_LEAF_SIZE) { return; }

  // Stage the leaf's memberships and origin labels once per block.
  __shared__ partition_membership records[MAX_LEAF_SIZE];
  __shared__ uint32_t leaf_origins[MAX_LEAF_SIZE];
  for (int i = threadIdx.x; i < leaf_n; i += blockDim.x) {
    records[i]      = memberships[start + static_cast<uint32_t>(i) * leaf_stride];
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
                          int64_t logical_dim,
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
  auto graph = raft::make_device_mdarray<uint32_t, int64_t>(
    res,
    raft::resource::get_large_workspace_resource_ref(res),
    raft::make_extents<int64_t>(rows, union_degree));
  int scaffold_blocks = strided_grid_size(rows * union_degree);
  initialize_self_scaffold_kernel<<<scaffold_blocks, THREADS_PER_BLOCK, 0, stream>>>(
    graph.data_handle(), rows, union_degree);
  RAFT_CUDA_TRY(cudaGetLastError());

  // Gather every scalar type into float leaf buffers. Native INT8 cuBLAS is not portable across
  // architectures (Ada returns CUBLAS_STATUS_NOT_SUPPORTED for the strided-batched INT8 path).
  int64_t const row_stride        = dataset.extent(1);
  int64_t input_dimension         = logical_dim;
  int dimension                   = static_cast<int>(input_dimension);
  size_t vector_elements_per_leaf = static_cast<size_t>(leaf_size) * dimension;
  size_t gram_elements_per_leaf   = static_cast<size_t>(leaf_size) * leaf_size;
  auto workspace_mr               = raft::resource::get_workspace_resource_ref(res);
  auto workspace_bytes            = raft::resource::get_workspace_free_bytes(res);
  auto leaf_workspace_bytes       = [&](size_t capacity) {
    return rmm::align_up(capacity * vector_elements_per_leaf * sizeof(float),
                         rmm::CUDA_ALLOCATION_ALIGNMENT) +
           rmm::align_up(capacity * gram_elements_per_leaf * sizeof(float),
                         rmm::CUDA_ALLOCATION_ALIGNMENT);
  };
  RAFT_EXPECTS(leaf_workspace_bytes(1) <= workspace_bytes,
               "Fastener leaf workspace cannot fit one leaf");
  size_t min_batch_capacity = 1;
  size_t max_batch_capacity = leaves.starts_host.size();
  while (min_batch_capacity < max_batch_capacity) {
    size_t candidate = min_batch_capacity + (max_batch_capacity - min_batch_capacity + 1) / 2;
    if (leaf_workspace_bytes(candidate) <= workspace_bytes) {
      min_batch_capacity = candidate;
    } else {
      max_batch_capacity = candidate - 1;
    }
  }
  size_t batch_capacity = min_batch_capacity;
  auto leaf_vectors     = raft::make_device_mdarray<float, int64_t>(
    res,
    workspace_mr,
    raft::make_extents<int64_t>(static_cast<int64_t>(batch_capacity * vector_elements_per_leaf)));
  auto gram = raft::make_device_mdarray<float, int64_t>(
    res,
    workspace_mr,
    raft::make_extents<int64_t>(static_cast<int64_t>(batch_capacity * gram_elements_per_leaf)));
  auto vector_stride = static_cast<int64_t>(vector_elements_per_leaf);
  auto gram_stride   = static_cast<int64_t>(gram_elements_per_leaf);

  for (size_t leaf_offset = 0; leaf_offset < leaves.starts_host.size();
       leaf_offset += batch_capacity) {
    size_t batch_size = std::min(batch_capacity, leaves.starts_host.size() - leaf_offset);
    int gather_blocks =
      strided_grid_size(static_cast<int64_t>(batch_size * vector_elements_per_leaf));
    manyway_gather_leaf_vectors_kernel<<<gather_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      dataset.data_handle(),
      input_dimension,
      row_stride,
      input_dimension,
      leaf_size,
      memberships.data_handle(),
      leaves.starts.data_handle(),
      leaves.counts.data_handle(),
      leaves.strides.data_handle(),
      static_cast<int64_t>(leaf_offset),
      static_cast<int64_t>(batch_size),
      leaf_vectors.data_handle());
    RAFT_CUDA_TRY(cudaGetLastError());
    batched_row_dot_products(res,
                             leaf_vectors.data_handle(),
                             leaf_size,
                             vector_stride,
                             leaf_vectors.data_handle(),
                             leaf_size,
                             vector_stride,
                             gram.data_handle(),
                             gram_stride,
                             dimension,
                             static_cast<int>(batch_size));
    manyway_leaf_gram_knn_kernel<<<static_cast<int>(batch_size), leaf_size, 0, stream>>>(
      gram.data_handle(),
      memberships.data_handle(),
      origins,
      leaves.starts.data_handle(),
      leaves.counts.data_handle(),
      leaves.strides.data_handle(),
      static_cast<int64_t>(leaf_offset),
      static_cast<int64_t>(batch_size),
      leaf_size,
      leaf_degree,
      union_degree,
      graph.data_handle());
    RAFT_CUDA_TRY(cudaGetLastError());
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
__global__ void manyway_l2_norms_kernel(
  T const* dataset, int64_t rows, int64_t dim, int64_t row_stride, float* norms)
{
  int lane    = threadIdx.x % raft::WarpSize;
  int warp    = threadIdx.x / raft::WarpSize;
  int64_t row = static_cast<int64_t>(blockIdx.x) * ROW_WARPS_PER_BLOCK + warp;
  if (row >= rows) { return; }

  // Each lane accumulates a strided slice of the row's squared values.
  float sum      = 0.0f;
  T const* point = dataset + row * row_stride;
  for (int64_t d = lane; d < dim; d += raft::WarpSize) {
    float value = static_cast<float>(point[d]);
    sum         = fmaf(value, value, sum);
  }
  // Shuffle-reduce the partial sums; lane 0 holds the total.
  for (int offset = raft::WarpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (lane == 0) { norms[row] = sum; }
}

/** Build the many-way scaffold through splitting and leaf construction. */
template <typename T>
auto build(raft::resources const& res,
           raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
           int64_t dim,
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
               "Fastener fanouts must be between 1 and %u",
               MAX_FANOUT);
  RAFT_EXPECTS(params.leader_fraction > 0.0 && params.leader_fraction <= 1.0,
               "Fastener leader fraction must be in (0, 1]");
  RAFT_EXPECTS(params.max_leaders >= std::max(params.root_fanout, params.lower_fanout) &&
                 params.max_leaders <= MAX_LEADERS,
               "Fastener leader cap must cover both fanouts and not exceed %u",
               MAX_LEADERS);
  RAFT_EXPECTS(params.leaf_size >= 1 && params.leaf_size <= MAX_LEAF_SIZE,
               "Fastener leaf size must be between 1 and %d",
               MAX_LEAF_SIZE);
  RAFT_EXPECTS(params.leaf_degree >= 1 && params.leaf_degree <= MAX_LEAF_DEGREE,
               "Fastener leaf degree must be between 1 and %d",
               MAX_LEAF_DEGREE);
  RAFT_EXPECTS(
    leaf_gemm_supported(dim, params.leaf_size, raft::resource::get_workspace_free_bytes(res)),
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
               "Fastener candidate width must not exceed %u",
               static_cast<unsigned>(std::numeric_limits<uint8_t>::max()));
  RAFT_EXPECTS(static_cast<uint64_t>(rows) <= std::numeric_limits<uint32_t>::max() / spill,
               "Fastener total partition memberships (rows=%ld * spill=%lu) must fit in uint32_t",
               static_cast<long>(rows),
               spill);
  int union_degree = static_cast<int>(spill * params.leaf_degree);

  // Per-row index of the input partition this point came from, used to skip same-origin pairs
  auto origins = raft::make_device_mdarray<uint32_t, int64_t>(
    res, raft::resource::get_large_workspace_resource_ref(res), raft::make_extents<int64_t>(rows));

  for (size_t part = 0; part + 1 < offsets.size(); ++part) {
    int64_t part_rows = offsets[part + 1] - offsets[part];
    int blocks        = static_cast<int>((part_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    initialize_origins_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      origins.data_handle(), offsets[part], part_rows, static_cast<uint32_t>(part));
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  split_context context(res, rows, dim);
  int norm_blocks = static_cast<int>((rows + ROW_WARPS_PER_BLOCK - 1) / ROW_WARPS_PER_BLOCK);
  manyway_l2_norms_kernel<<<norm_blocks, ROW_WARPS_PER_BLOCK * raft::WarpSize, 0, stream>>>(
    dataset.data_handle(), rows, dim, dataset.extent(1), context.norms.data_handle());
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

  // Leaf construction: only range slicing occurs after configured geometric depth.
  auto leaves = make_leaves(res, partitions, params.leaf_size);
  auto scaffold =
    build_leaf_neighbors(res, dataset, dim, leaves, origins.data_handle(), union_degree, params);
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
template <typename T, typename IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
auto append_to_input_graphs(
  raft::resources const& res,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*> const& indices,
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
  auto graph              = raft::make_device_mdarray<uint32_t, int64_t>(
    res,
    raft::resource::get_large_workspace_resource_ref(res),
    raft::make_extents<int64_t>(scaffold.extent(0), graph_degree));

  for (size_t part = 0; part < indices.size(); ++part) {
    auto source = indices[part]->graph();
    RAFT_EXPECTS(source.extent(1) > 0, "Input CAGRA graphs must have nonzero degree");
    int blocks = static_cast<int>((source.extent(0) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    copy_partition_with_scaffold_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
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
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / raft::WarpSize;
  int lane                      = threadIdx.x % raft::WarpSize;
  int warp                      = threadIdx.x / raft::WarpSize;
  int64_t row                   = static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (row >= rows) { return; }

  int64_t input_base  = row * input_degree;
  int64_t output_base = row * output_degree;
  int selected        = 0;
  for (int64_t tile = 0; tile < input_degree && selected < output_degree; tile += raft::WarpSize) {
    int64_t column     = tile + lane;
    bool first         = column < input_degree;
    uint32_t candidate = first ? input[input_base + column] : uint32_t{0};
    first              = first && candidate < rows && candidate != static_cast<uint32_t>(row);
    for (int64_t prior = 0; prior < column && first; ++prior) {
      if (input[input_base + prior] == candidate) { first = false; }
    }

    unsigned first_mask = __ballot_sync(0xffffffffu, first);
    unsigned lower_mask = lane == 0 ? 0u : (0xffffffffu >> (raft::WarpSize - lane));
    int output_column   = selected + __popc(first_mask & lower_mask);
    if (first && output_column < output_degree) { output[output_base + output_column] = candidate; }
    selected += __popc(first_mask);
  }

  if (selected == 0) {
    if (lane == 0) { output[output_base] = static_cast<uint32_t>((row + 1) % rows); }
    selected = 1;
  }
  if (selected > output_degree) { selected = static_cast<int>(output_degree); }
  for (int64_t column = selected + lane; column < output_degree; column += raft::WarpSize) {
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
  auto output = raft::make_device_mdarray<uint32_t, int64_t>(
    res,
    raft::resource::get_large_workspace_resource_ref(res),
    raft::make_extents<int64_t>(graph.extent(0), output_degree));
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / raft::WarpSize;
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
