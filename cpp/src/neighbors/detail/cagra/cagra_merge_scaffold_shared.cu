/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra_merge_scaffold.cuh"

#include <cuvs/selection/select_k.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/array>

#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace cuvs::neighbors::cagra::detail::merge_scaffold {
namespace {

// All dataset types are gathered to float; TF32 only reduces precision for native float inputs.
inline constexpr cublasComputeType_t GEMM_COMPUTE_TYPE = CUBLAS_COMPUTE_32F_FAST_TF32;

__global__ void initialize_root_memberships_kernel(partition_membership* memberships, int64_t rows)
{
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < rows) { memberships[row] = {static_cast<uint32_t>(row), uint16_t{0}, uint16_t{0}}; }
}

__global__ void carry_completed_parents_kernel(partition_membership const* input,
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

__global__ void materialize_tile_distances_kernel(float* dots,
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
      distance = fmaxf(0.0f, norms[membership.id] + norms[leader_id] - 2.0f * dots[linear]);
    }
    dots[linear] = distance;
  }
}

__global__ void emit_tile_assignments_kernel(int const* selected_leaders,
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

__global__ void initialize_self_scaffold_kernel(uint32_t* graph,
                                                int64_t rows,
                                                int64_t graph_degree,
                                                int64_t scaffold_offset,
                                                int64_t scaffold_degree)
{
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total  = rows * scaffold_degree;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; linear < total; linear += stride) {
    int64_t row                                          = linear / scaffold_degree;
    int64_t column                                       = linear % scaffold_degree;
    graph[row * graph_degree + scaffold_offset + column] = static_cast<uint32_t>(row);
  }
}

__global__ void leaf_gram_knn_kernel(float const* gram,
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
                                     uint32_t* graph)
{
  int64_t local_leaf = blockIdx.x;
  if (local_leaf >= leaf_count) { return; }
  int64_t leaf         = leaf_offset + local_leaf;
  uint32_t start       = leaf_starts[leaf];
  uint32_t leaf_stride = leaf_strides[leaf];
  int leaf_n           = static_cast<int>(leaf_counts[leaf]);
  if (leaf_n <= 1 || leaf_n > MAX_LEAF_SIZE) { return; }

  __shared__ partition_membership records[MAX_LEAF_SIZE];
  __shared__ uint32_t leaf_origins[MAX_LEAF_SIZE];
  for (int i = threadIdx.x; i < leaf_n; i += blockDim.x) {
    records[i]      = memberships[start + static_cast<uint32_t>(i) * leaf_stride];
    leaf_origins[i] = origins[records[i].id];
  }
  __syncthreads();

  int u = threadIdx.x;
  if (u >= leaf_n) { return; }
  float top_d[MAX_LEAF_DEGREE];
  uint16_t top_v[MAX_LEAF_DEGREE];
  for (int t = 0; t < leaf_degree; ++t) {
    top_d[t] = std::numeric_limits<float>::max();
    top_v[t] = std::numeric_limits<uint16_t>::max();
  }

  int64_t gram_base = local_leaf * leaf_size * leaf_size;
  float norm_u      = gram[gram_base + u * leaf_size + u];
  for (int v = 0; v < leaf_n; ++v) {
    if (u == v || leaf_origins[u] == leaf_origins[v] || records[u].id == records[v].id) {
      continue;
    }
    float norm_v   = gram[gram_base + v * leaf_size + v];
    float dot      = gram[gram_base + v * leaf_size + u];
    float distance = norm_u + norm_v - 2.0f * dot;
    if (isfinite(distance)) { distance = fmaxf(0.0f, distance); }
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

  int selected = 0;
  for (int t = 0; t < leaf_degree; ++t) {
    bool valid = top_v[t] != std::numeric_limits<uint16_t>::max() && isfinite(top_d[t]);
    if (valid) {
      int64_t output = static_cast<int64_t>(records[u].id) * graph_degree + scaffold_offset +
                       static_cast<int>(records[u].occurrence) * leaf_degree + selected;
      graph[output] = records[top_v[t]].id;
      ++selected;
    }
  }
}

__global__ void initialize_origins_kernel(uint32_t* origins,
                                          int64_t start,
                                          int64_t rows,
                                          uint32_t origin)
{
  int64_t local_row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local_row < rows) { origins[start + local_row] = origin; }
}

__global__ void copy_partition_graph_kernel(uint32_t const* source,
                                            int64_t source_rows,
                                            int64_t source_degree,
                                            uint32_t* destination,
                                            int64_t destination_degree,
                                            int64_t base_degree,
                                            uint32_t offset)
{
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / raft::WarpSize;
  int lane                      = threadIdx.x % raft::WarpSize;
  int warp                      = threadIdx.x / raft::WarpSize;
  int64_t row                   = static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (row >= source_rows) { return; }
  int64_t source_base      = row * source_degree;
  int64_t global_row       = row + offset;
  int64_t destination_base = global_row * destination_degree;

  for (int64_t j = lane; j < base_degree; j += raft::WarpSize) {
    destination[destination_base + j] = source[source_base + (j % source_degree)] + offset;
  }
}

__global__ void deduplicate_graph_prefix_kernel(uint32_t const* input,
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

}  // namespace

void select_nearest_leaders(raft::resources const& res,
                            float const* distances,
                            int64_t rows,
                            int leaders,
                            int fanout,
                            float* selected_distances,
                            int* selected_leaders)
{
  cuvs::selection::select_k(
    res,
    raft::make_device_matrix_view<const float, int64_t>(distances, rows, leaders),
    std::nullopt,
    raft::make_device_matrix_view<float, int64_t>(selected_distances, rows, fanout),
    raft::make_device_matrix_view<int, int64_t>(selected_leaders, rows, fanout),
    true,
    true);
}

void batched_row_dot_products(raft::resources const& res,
                              float* a,
                              int a_rows,
                              long long a_stride,
                              float* b,
                              int b_rows,
                              long long b_stride,
                              float* out,
                              long long out_stride,
                              int row_width,
                              int batch_count)
{
  using index_type = int64_t;
  auto a_extents   = raft::extent_3d<index_type>{batch_count, a_rows, row_width};
  auto b_extents   = raft::extent_3d<index_type>{batch_count, row_width, b_rows};
  auto out_extents = raft::extent_3d<index_type>{batch_count, a_rows, b_rows};

  // A is row-major. B's row-major [b_rows, row_width] storage is a column-major
  // [row_width, b_rows] matrix, and the output preserves its existing column-major layout.
  auto a_layout =
    raft::make_strided_layout(a_extents, cuda::std::array<index_type, 3>{a_stride, row_width, 1});
  auto b_layout =
    raft::make_strided_layout(b_extents, cuda::std::array<index_type, 3>{b_stride, 1, row_width});
  auto out_layout =
    raft::make_strided_layout(out_extents, cuda::std::array<index_type, 3>{out_stride, 1, a_rows});

  auto a_view = raft::device_mdspan<float, decltype(a_extents), raft::layout_stride>{a, a_layout};
  auto b_view = raft::device_mdspan<float, decltype(b_extents), raft::layout_stride>{b, b_layout};
  auto out_view =
    raft::device_mdspan<float, decltype(out_extents), raft::layout_stride>{out, out_layout};

  std::optional<raft::host_scalar_view<float>> alpha;
  std::optional<raft::host_scalar_view<float>> beta;
  raft::linalg::gemm_batched(res, a_view, b_view, out_view, alpha, beta, GEMM_COMPUTE_TYPE);
}

void launch_initialize_root_memberships(raft::resources const& res,
                                        partition_membership* memberships,
                                        int64_t rows)
{
  auto blocks = static_cast<int>(raft::div_rounding_up_safe<int64_t>(rows, THREADS_PER_BLOCK));
  initialize_root_memberships_kernel<<<blocks,
                                       THREADS_PER_BLOCK,
                                       0,
                                       raft::resource::get_cuda_stream(res)>>>(memberships, rows);
  RAFT_CUDA_TRY(cudaGetLastError());
}

void launch_carry_completed_parents(raft::resources const& res,
                                    partition_membership const* input,
                                    carry_span const* spans,
                                    int span_count,
                                    partition_membership* output,
                                    uint32_t* output_keys)
{
  carry_completed_parents_kernel<<<span_count,
                                   THREADS_PER_BLOCK,
                                   0,
                                   raft::resource::get_cuda_stream(res)>>>(
    input, spans, output, output_keys);
  RAFT_CUDA_TRY(cudaGetLastError());
}

void launch_materialize_tile_distances(raft::resources const& res,
                                       float* dots,
                                       int batch_size,
                                       int tile_rows,
                                       int padded_leaders,
                                       float const* norms,
                                       uint32_t const* leader_ids,
                                       partition_membership const* input_memberships,
                                       assignment_tile const* tiles)
{
  auto blocks = strided_grid_size(static_cast<int64_t>(batch_size) * tile_rows * padded_leaders);
  materialize_tile_distances_kernel<<<blocks,
                                      THREADS_PER_BLOCK,
                                      0,
                                      raft::resource::get_cuda_stream(res)>>>(
    dots, batch_size, tile_rows, padded_leaders, norms, leader_ids, input_memberships, tiles);
  RAFT_CUDA_TRY(cudaGetLastError());
}

void launch_emit_tile_assignments(raft::resources const& res,
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
  auto blocks = strided_grid_size(static_cast<int64_t>(batch_size) * tile_rows);
  emit_tile_assignments_kernel<<<blocks,
                                 THREADS_PER_BLOCK,
                                 0,
                                 raft::resource::get_cuda_stream(res)>>>(selected_leaders,
                                                                         batch_size,
                                                                         tile_rows,
                                                                         fanout,
                                                                         occurrence_stride,
                                                                         input_memberships,
                                                                         tiles,
                                                                         output_keys,
                                                                         output_memberships);
  RAFT_CUDA_TRY(cudaGetLastError());
}

auto sort_memberships_and_collect_ranges(raft::resources const& res,
                                         uint32_t* keys,
                                         partition_membership* memberships,
                                         int64_t count,
                                         uint32_t key_count)
  -> raft::host_vector<partition_range, int64_t>
{
  auto stream   = raft::resource::get_cuda_stream(res);
  auto large_mr = raft::resource::get_large_workspace_resource_ref(res);
  rmm::exec_policy_nosync thrust_policy{stream, large_mr};
  thrust::stable_sort_by_key(thrust_policy,
                             thrust::device_pointer_cast(keys),
                             thrust::device_pointer_cast(keys + count),
                             thrust::device_pointer_cast(memberships));

  auto output_capacity    = std::min<int64_t>(count, key_count);
  auto device_unique_keys = raft::make_device_mdarray<uint32_t, int64_t>(
    res, large_mr, raft::make_extents<int64_t>(output_capacity));
  auto device_counts = raft::make_device_mdarray<uint32_t, int64_t>(
    res, large_mr, raft::make_extents<int64_t>(output_capacity));
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

void launch_initialize_self_scaffold(raft::resources const& res,
                                     uint32_t* graph,
                                     int64_t rows,
                                     int64_t graph_degree,
                                     int64_t scaffold_offset,
                                     int64_t scaffold_degree)
{
  auto blocks = strided_grid_size(rows * scaffold_degree);
  initialize_self_scaffold_kernel<<<blocks,
                                    THREADS_PER_BLOCK,
                                    0,
                                    raft::resource::get_cuda_stream(res)>>>(
    graph, rows, graph_degree, scaffold_offset, scaffold_degree);
  RAFT_CUDA_TRY(cudaGetLastError());
}

void launch_leaf_gram_knn(raft::resources const& res,
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
                          uint32_t* graph)
{
  leaf_gram_knn_kernel<<<static_cast<int>(leaf_count),
                         leaf_size,
                         0,
                         raft::resource::get_cuda_stream(res)>>>(gram,
                                                                 memberships,
                                                                 origins,
                                                                 leaf_starts,
                                                                 leaf_counts,
                                                                 leaf_strides,
                                                                 leaf_offset,
                                                                 leaf_count,
                                                                 leaf_size,
                                                                 leaf_degree,
                                                                 graph_degree,
                                                                 scaffold_offset,
                                                                 graph);
  RAFT_CUDA_TRY(cudaGetLastError());
}

void launch_initialize_origins(
  raft::resources const& res, uint32_t* origins, int64_t start, int64_t rows, uint32_t origin)
{
  auto blocks = static_cast<int>(raft::div_rounding_up_safe<int64_t>(rows, THREADS_PER_BLOCK));
  initialize_origins_kernel<<<blocks, THREADS_PER_BLOCK, 0, raft::resource::get_cuda_stream(res)>>>(
    origins, start, rows, origin);
  RAFT_CUDA_TRY(cudaGetLastError());
}

void launch_copy_partition_graph(raft::resources const& res,
                                 uint32_t const* source,
                                 int64_t source_rows,
                                 int64_t source_degree,
                                 uint32_t* destination,
                                 int64_t destination_degree,
                                 int64_t base_degree,
                                 uint32_t offset)
{
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / raft::WarpSize;
  auto blocks = static_cast<int>(raft::div_rounding_up_safe<int64_t>(source_rows, WARPS_PER_BLOCK));
  copy_partition_graph_kernel<<<blocks,
                                THREADS_PER_BLOCK,
                                0,
                                raft::resource::get_cuda_stream(res)>>>(
    source, source_rows, source_degree, destination, destination_degree, base_degree, offset);
  RAFT_CUDA_TRY(cudaGetLastError());
}

void launch_deduplicate_graph_prefix(raft::resources const& res,
                                     uint32_t const* input,
                                     int64_t rows,
                                     int64_t input_degree,
                                     uint32_t* output,
                                     int64_t output_degree)
{
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / raft::WarpSize;
  auto blocks = static_cast<int>(raft::div_rounding_up_safe<int64_t>(rows, WARPS_PER_BLOCK));
  deduplicate_graph_prefix_kernel<<<blocks,
                                    THREADS_PER_BLOCK,
                                    0,
                                    raft::resource::get_cuda_stream(res)>>>(
    input, rows, input_degree, output, output_degree);
  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace cuvs::neighbors::cagra::detail::merge_scaffold
