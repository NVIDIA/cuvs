/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../neighbors_device_intrinsics.cuh"

#include <cuvs/core/export.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/aligned.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
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

// Type-independent scaffold operations are compiled once instead of being emitted into every
// dtype-specific merge translation unit.
CUVS_EXPORT void launch_initialize_root_memberships(raft::resources const& res,
                                                    partition_membership* memberships,
                                                    int64_t rows);
CUVS_EXPORT void launch_carry_completed_parents(raft::resources const& res,
                                                partition_membership const* input,
                                                carry_span const* spans,
                                                int span_count,
                                                partition_membership* output,
                                                uint32_t* output_keys);
CUVS_EXPORT void launch_materialize_tile_distances(raft::resources const& res,
                                                   float* dots,
                                                   int batch_size,
                                                   int tile_rows,
                                                   int padded_leaders,
                                                   float const* norms,
                                                   uint32_t const* leader_ids,
                                                   partition_membership const* input_memberships,
                                                   assignment_tile const* tiles);
CUVS_EXPORT void launch_emit_tile_assignments(raft::resources const& res,
                                              int const* selected_leaders,
                                              int batch_size,
                                              int tile_rows,
                                              int fanout,
                                              int occurrence_stride,
                                              partition_membership const* input_memberships,
                                              assignment_tile const* tiles,
                                              uint32_t* output_keys,
                                              partition_membership* output_memberships);
CUVS_EXPORT void select_nearest_leaders(raft::resources const& res,
                                        float const* distances,
                                        int64_t rows,
                                        int leaders,
                                        int fanout,
                                        float* selected_distances,
                                        int* selected_leaders);
CUVS_EXPORT void batched_row_dot_products(raft::resources const& res,
                                          float* a,
                                          int a_rows,
                                          long long a_stride,
                                          float* b,
                                          int b_rows,
                                          long long b_stride,
                                          float* out,
                                          long long out_stride,
                                          int row_width,
                                          int batch_count);
CUVS_EXPORT auto sort_memberships_and_collect_ranges(raft::resources const& res,
                                                     uint32_t* keys,
                                                     partition_membership* memberships,
                                                     int64_t count,
                                                     uint32_t key_count)
  -> raft::host_vector<partition_range, int64_t>;
CUVS_EXPORT void launch_initialize_self_scaffold(raft::resources const& res,
                                                 uint32_t* graph,
                                                 int64_t rows,
                                                 int64_t graph_degree,
                                                 int64_t scaffold_offset,
                                                 int64_t scaffold_degree);
CUVS_EXPORT void launch_leaf_gram_knn(raft::resources const& res,
                                      float const* gram,
                                      partition_membership const* memberships,
                                      uint32_t const* origins,
                                      uint32_t const* leaf_starts,
                                      uint32_t const* leaf_counts,
                                      uint32_t const* leaf_strides,
                                      int64_t leaf_offset,
                                      int64_t leaf_count,
                                      int leaf_size,
                                      int leaf_degree,
                                      int64_t graph_degree,
                                      int64_t scaffold_offset,
                                      uint32_t* graph);
CUVS_EXPORT void launch_initialize_origins(
  raft::resources const& res, uint32_t* origins, int64_t start, int64_t rows, uint32_t origin);
CUVS_EXPORT void launch_copy_partition_graph(raft::resources const& res,
                                             uint32_t const* source,
                                             int64_t source_rows,
                                             int64_t source_degree,
                                             uint32_t* destination,
                                             int64_t destination_degree,
                                             int64_t base_degree,
                                             uint32_t offset);
CUVS_EXPORT void launch_deduplicate_graph_prefix(raft::resources const& res,
                                                 uint32_t const* input,
                                                 int64_t rows,
                                                 int64_t input_degree,
                                                 uint32_t* output,
                                                 int64_t output_degree);

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

/** Represent the full dataset as one identity partition without copying any vectors. */
inline auto make_root_partition(raft::resources const& res, int64_t rows) -> partition_set
{
  // Memberships are dataset-sized, so they come from the large workspace rather than the bounded
  // one; the single root range is host metadata.
  auto memberships = raft::make_device_mdarray<partition_membership, int64_t>(
    res, raft::resource::get_large_workspace_resource_ref(res), raft::make_extents<int64_t>(rows));
  auto ranges = raft::make_host_vector<partition_range, int64_t>(res, 1);
  ranges(0)   = partition_range{uint32_t{0}, int64_t{0}, rows};
  partition_set root{std::move(memberships), std::move(ranges)};
  launch_initialize_root_memberships(res, root.memberships.data_handle(), rows);
  return root;
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
  launch_carry_completed_parents(res,
                                 parents.memberships.data_handle(),
                                 device_carries.data_handle(),
                                 static_cast<int>(carries.size()),
                                 memberships.data_handle(),
                                 keys.data_handle());
}

/** Workspace bytes `assign_bucket` needs for `capacity` tiles of `rows_per_tile` rows each */
inline auto assignment_workspace_bytes(
  size_t capacity, size_t rows_per_tile, size_t padded_leaders, size_t dim, size_t fanout) -> size_t
{
  auto aligned = [](size_t bytes) { return rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT); };
  size_t point_elements    = rows_per_tile * dim;
  size_t leader_elements   = padded_leaders * dim;
  size_t dot_elements      = rows_per_tile * padded_leaders;
  size_t selected_elements = rows_per_tile * fanout;
  return aligned(capacity * sizeof(assignment_tile)) +
         aligned(capacity * point_elements * sizeof(float)) +
         aligned(capacity * leader_elements * sizeof(float)) +
         aligned(capacity * dot_elements * sizeof(float)) +
         aligned(capacity * padded_leaders * sizeof(uint32_t)) +
         aligned(capacity * selected_elements * sizeof(float)) +
         aligned(capacity * selected_elements * sizeof(int)) +
         aligned(capacity * dot_elements * sizeof(float)) +
         aligned(capacity * dot_elements * sizeof(int));
}

/** Return true if the widest padded leader matrix + a single point row fits the workspace.
 *
 * `assign_bucket` shrinks its tile height and batch capacity to whatever the workspace allows, but
 * the leader matrix can't get any smaller than one tile of one row.
 */
inline auto assignment_gemm_supported(int64_t dimension,
                                      int64_t rows,
                                      split_params const& worst_case,
                                      size_t workspace_bytes) -> bool
{
  if (dimension <= 0 || dimension > std::numeric_limits<int>::max() || rows <= 0) { return false; }
  auto const leaders = select_leader_count(rows, worst_case);
  return assignment_workspace_bytes(1,
                                    1,
                                    std::bit_ceil(static_cast<unsigned>(leaders)),
                                    static_cast<size_t>(dimension),
                                    static_cast<size_t>(worst_case.fanout)) <= workspace_bytes;
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

  auto tile_bytes = [&](size_t capacity, size_t rows_per_tile) {
    return assignment_workspace_bytes(capacity,
                                      rows_per_tile,
                                      static_cast<size_t>(padded_leaders),
                                      static_cast<size_t>(dim),
                                      static_cast<size_t>(params.fanout));
  };

  // Preflight rejects any configuration whose widest leader matrix cannot host a single point row,
  // so this only fires for callers that drive the scaffold directly.
  RAFT_EXPECTS(tile_bytes(1, 1) <= workspace_bytes,
               "Fastener assignment workspace cannot fit the leader matrix and a single point row");
  size_t min_tile_rows = 1;
  size_t max_tile_rows = static_cast<size_t>(context.assignment_tile_rows);
  while (min_tile_rows < max_tile_rows) {
    size_t candidate = min_tile_rows + (max_tile_rows - min_tile_rows + 1) / 2;
    if (tile_bytes(1, candidate) <= workspace_bytes) {
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
    if (tile_bytes(candidate, static_cast<size_t>(tile_rows)) <= workspace_bytes) {
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
    launch_materialize_tile_distances(res,
                                      tile_dots.data_handle(),
                                      static_cast<int>(batch_size),
                                      tile_rows,
                                      padded_leaders,
                                      context.norms.data_handle(),
                                      tile_leader_ids.data_handle(),
                                      parents.memberships.data_handle(),
                                      device_tiles.data_handle());

    select_nearest_leaders(res,
                           tile_dots.data_handle(),
                           selection_rows,
                           padded_leaders,
                           static_cast<int>(params.fanout),
                           selected_distances.data_handle(),
                           selected_leaders.data_handle());

    launch_emit_tile_assignments(res,
                                 selected_leaders.data_handle(),
                                 static_cast<int>(batch_size),
                                 tile_rows,
                                 static_cast<int>(params.fanout),
                                 static_cast<int>(params.occurrence_stride),
                                 parents.memberships.data_handle(),
                                 device_tiles.data_handle(),
                                 keys.data_handle(),
                                 memberships.data_handle());
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
  auto plan = plan_split(
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

  auto ranges = sort_memberships_and_collect_ranges(
    res, keys.data_handle(), memberships.data_handle(), plan.output_rows, plan.child_count);
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

/** Build directed cross-input nearest neighbors for every leaf occurrence. */
template <typename T>
auto build_leaf_neighbors(raft::resources const& res,
                          raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
                          int64_t logical_dim,
                          leaf_set const& leaves,
                          uint32_t const* origins,
                          int union_degree,
                          int64_t base_degree,
                          build_params const& params) -> raft::device_matrix<uint32_t, int64_t>
{
  auto stream             = raft::resource::get_cuda_stream(res);
  int64_t rows            = dataset.extent(0);
  int leaf_size           = static_cast<int>(params.leaf_size);
  int leaf_degree         = static_cast<int>(params.leaf_degree);
  auto const& memberships = leaves.partitions->memberships;

  // Prefill every scaffold slot with its own row id; leaf KNN overwrites the slots it fills.
  auto graph = raft::make_device_mdarray<uint32_t, int64_t>(
    res,
    raft::resource::get_large_workspace_resource_ref(res),
    raft::make_extents<int64_t>(rows, base_degree + union_degree));
  launch_initialize_self_scaffold(
    res, graph.data_handle(), rows, graph.extent(1), base_degree, union_degree);

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
    launch_leaf_gram_knn(res,
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
                         graph.extent(1),
                         base_degree,
                         graph.data_handle());
  }
  return graph;
}

// ----------------------------------------------------------------------------
// Scaffold build driver
// ----------------------------------------------------------------------------

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

/** Build the many-way scaffold as the suffix of a candidate graph with `base_degree` columns. */
template <typename T>
auto build(raft::resources const& res,
           raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
           int64_t dim,
           std::vector<int64_t> const& offsets,
           build_params const& params = {},
           int64_t base_degree        = 0) -> raft::device_matrix<uint32_t, int64_t>
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
  RAFT_EXPECTS(base_degree >= 0, "Fastener base graph degree must be nonnegative");
  RAFT_EXPECTS(
    leaf_gemm_supported(dim, params.leaf_size, raft::resource::get_workspace_free_bytes(res)),
    "Fastener dataset dimension exceeds the leaf GEMM limits");
  RAFT_EXPECTS(assignment_gemm_supported(
                 dim,
                 rows,
                 split_params{.fanout          = std::max(params.root_fanout, params.lower_fanout),
                              .leader_fraction = params.leader_fraction,
                              .max_leaders     = params.max_leaders},
                 raft::resource::get_workspace_free_bytes(res)),
               "Fastener dataset dimension exceeds the assignment GEMM limits");

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
    launch_initialize_origins(
      res, origins.data_handle(), offsets[part], part_rows, static_cast<uint32_t>(part));
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
  auto graph  = build_leaf_neighbors(
    res, dataset, dim, leaves, origins.data_handle(), union_degree, base_degree, params);
  raft::resource::sync_stream(res);
  return graph;
}

// ----------------------------------------------------------------------------
// Candidate graph assembly: combine input graphs with the scaffold
// ----------------------------------------------------------------------------

/**
 * @brief Populate the input-graph prefix of the pre-optimization candidate graph.
 *
 * The maximum input graph degree defines the base width, allowing partitions with mixed degrees.
 */
template <typename T, typename IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
void append_to_input_graphs(
  raft::resources const& res,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*> const& indices,
  std::vector<int64_t> const& offsets,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> graph,
  int64_t base_degree)
{
  RAFT_EXPECTS(graph.extent(0) == offsets.back() && graph.extent(1) >= base_degree,
               "Candidate graph shape must cover the merged rows and base degree");

  for (size_t part = 0; part < indices.size(); ++part) {
    auto source = indices[part]->graph();
    RAFT_EXPECTS(source.extent(1) > 0, "Input CAGRA graphs must have nonzero degree");
    launch_copy_partition_graph(res,
                                source.data_handle(),
                                source.extent(0),
                                source.extent(1),
                                graph.data_handle(),
                                graph.extent(1),
                                base_degree,
                                static_cast<uint32_t>(offsets[part]));
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
  launch_deduplicate_graph_prefix(res,
                                  graph.data_handle(),
                                  graph.extent(0),
                                  graph.extent(1),
                                  output.data_handle(),
                                  output_degree);
  return output;
}

}  // namespace cuvs::neighbors::cagra::detail::merge_scaffold
