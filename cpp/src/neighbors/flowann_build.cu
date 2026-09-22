/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/flowann_build.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>

#include <neighbors/detail/flowann/build.hpp>
#include <neighbors/detail/flowann/grouping.hpp>

#include "detail/cagra/cagra_build.cuh"

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra::experimental::flowann::detail {

namespace {

using graph_index_type = std::uint32_t;
using host_graph_view  = raft::host_matrix_view<const graph_index_type, int64_t, raft::row_major>;
using host_edge_order_view = raft::host_matrix_view<const std::uint8_t, int64_t, raft::row_major>;

constexpr std::uint16_t max_nodes_per_packed_row = 64;
constexpr std::uint32_t max_partitions           = 1u << 16;

template <typename StreamT>
auto cuda_stream_handle(StreamT const& stream) -> cudaStream_t
{
  if constexpr (requires { stream.get(); }) {
    return stream.get();
  } else if constexpr (requires { stream.value(); }) {
    return stream.value();
  } else {
    return static_cast<cudaStream_t>(stream);
  }
}

struct conversion_result {
  raft::host_matrix<std::uint8_t, int64_t, raft::row_major> inner_graph;
  raft::host_matrix<graph_index_type, int64_t, raft::row_major> cross_graph;
  std::vector<graph_index_type> old_to_new;
  std::vector<graph_index_type> new_to_old;
  std::vector<std::uint32_t> subgraph_offsets;
  std::vector<std::uint32_t> subgraph_ids;
};

struct device_conversion_result {
  raft::device_matrix<std::uint8_t, int64_t, raft::row_major> inner_graph;
  raft::host_matrix<graph_index_type, int64_t, raft::row_major> cross_graph;
  std::vector<graph_index_type> old_to_new;
  std::vector<graph_index_type> new_to_old;
  std::vector<std::uint32_t> subgraph_offsets;
  std::vector<std::uint32_t> subgraph_ids;
  bool validated;
};

struct uniform_conversion_result {
  raft::device_matrix<std::uint8_t, int64_t, raft::row_major> inner_graph;
  raft::host_matrix<graph_index_type, int64_t, raft::row_major> cross_graph;
  uniform_layout_plan layout;
};

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION auto rank_value(T value) -> float
{
  return static_cast<float>(value);
}

RAFT_DEVICE_INLINE_FUNCTION void set_packed_bits(std::uint8_t* row,
                                                 std::uint64_t bit_offset,
                                                 std::uint32_t value,
                                                 std::uint16_t n_bits)
{
  for (std::uint16_t bit = 0; bit < n_bits; ++bit) {
    auto const position = bit_offset + bit;
    auto const mask     = static_cast<std::uint8_t>(1u << (position & 7u));
    if ((value >> bit) & 1u) { row[position >> 3] |= mask; }
  }
}

template <typename T, bool StagedRows>
__global__ void rank_and_split_kernel(T const* dataset,
                                      std::uint32_t dataset_stride,
                                      T const* staged_rows,
                                      graph_index_type const* graph,
                                      std::uint64_t batch_offset,
                                      std::uint32_t batch_rows,
                                      std::uint32_t dim,
                                      std::uint32_t graph_degree,
                                      std::uint32_t resident_degree,
                                      std::uint32_t cross_degree,
                                      std::uint16_t node_per_cacheline,
                                      std::uint16_t n_bits,
                                      std::uint32_t inner_row_bytes,
                                      std::uint8_t* inner_graph,
                                      graph_index_type* cross_graph,
                                      cuvs::distance::DistanceType metric)
{
  extern __shared__ std::uint8_t shared_bytes[];
  auto* scores = reinterpret_cast<float*>(shared_bytes);
  auto* selected =
    shared_bytes + static_cast<std::size_t>(node_per_cacheline) * graph_degree * sizeof(float);
  auto resident_offset = static_cast<std::size_t>(node_per_cacheline) * graph_degree *
                         (sizeof(float) + sizeof(std::uint8_t));
  resident_offset =
    (resident_offset + alignof(graph_index_type) - 1) & ~(alignof(graph_index_type) - 1);
  auto* resident = reinterpret_cast<graph_index_type*>(shared_bytes + resident_offset);

  auto const lane       = threadIdx.x % raft::WarpSize;
  auto const packed_col = threadIdx.x / raft::WarpSize;
  auto const local_node = static_cast<std::uint64_t>(blockIdx.x) * node_per_cacheline + packed_col;
  auto const valid_node = packed_col < node_per_cacheline && local_node < batch_rows;

  if (valid_node && lane == 0) {
    auto* node_selected = selected + static_cast<std::size_t>(packed_col) * graph_degree;
    for (std::uint32_t edge = 0; edge < graph_degree; ++edge) {
      node_selected[edge] = 0;
    }
  }
  __syncthreads();

  if (valid_node) {
    auto const global_node = batch_offset + local_node;
    auto const* source =
      StagedRows ? staged_rows + static_cast<std::size_t>(local_node) * (graph_degree + 1) * dim
                 : dataset + global_node * dataset_stride;
    for (std::uint32_t edge = 0; edge < graph_degree; ++edge) {
      auto const neighbor = graph[local_node * graph_degree + edge];
      auto const* target =
        StagedRows ? staged_rows +
                       (static_cast<std::size_t>(local_node) * (graph_degree + 1) + edge + 1) * dim
                   : dataset + static_cast<std::uint64_t>(neighbor) * dataset_stride;
      float primary   = 0.0f;
      float source_l2 = 0.0f;
      float target_l2 = 0.0f;
      for (std::uint32_t column = lane; column < dim; column += raft::WarpSize) {
        auto const lhs = rank_value(source[column]);
        auto const rhs = rank_value(target[column]);
        switch (metric) {
          case cuvs::distance::DistanceType::L2Expanded: {
            auto const diff = lhs - rhs;
            primary += diff * diff;
            break;
          }
          case cuvs::distance::DistanceType::InnerProduct: primary -= lhs * rhs; break;
          case cuvs::distance::DistanceType::CosineExpanded:
            primary -= lhs * rhs;
            source_l2 += lhs * lhs;
            target_l2 += rhs * rhs;
            break;
          case cuvs::distance::DistanceType::L1: primary += fabsf(lhs - rhs); break;
          case cuvs::distance::DistanceType::BitwiseHamming:
            if constexpr (std::is_integral_v<T>) {
              primary += __popc((static_cast<std::uint32_t>(source[column]) ^
                                 static_cast<std::uint32_t>(target[column])) &
                                0xffu);
            }
            break;
          default: primary = std::numeric_limits<float>::infinity(); break;
        }
      }
      for (auto offset = raft::WarpSize / 2; offset > 0; offset /= 2) {
        primary += __shfl_down_sync(0xffffffffu, primary, offset);
        source_l2 += __shfl_down_sync(0xffffffffu, source_l2, offset);
        target_l2 += __shfl_down_sync(0xffffffffu, target_l2, offset);
      }
      if (lane == 0) {
        if (metric == cuvs::distance::DistanceType::CosineExpanded) {
          auto const denominator = sqrtf(source_l2 * target_l2);
          primary                = denominator > 0.0f ? 1.0f + primary / denominator
                                                      : std::numeric_limits<float>::infinity();
        }
        if (isnan(primary)) { primary = std::numeric_limits<float>::infinity(); }
        scores[static_cast<std::size_t>(packed_col) * graph_degree + edge] = primary;
      }
    }
  }
  __syncthreads();

  if (valid_node && lane == 0) {
    auto const* node_graph = graph + local_node * graph_degree;
    auto* node_scores      = scores + static_cast<std::size_t>(packed_col) * graph_degree;
    auto* node_selected    = selected + static_cast<std::size_t>(packed_col) * graph_degree;
    auto* node_resident    = resident + static_cast<std::size_t>(packed_col) * resident_degree;

    for (std::uint32_t rank = 0; rank < resident_degree; ++rank) {
      auto best_edge  = graph_degree;
      auto best_id    = std::numeric_limits<graph_index_type>::max();
      auto best_score = std::numeric_limits<float>::infinity();
      for (std::uint32_t edge = 0; edge < graph_degree; ++edge) {
        if (node_selected[edge] != 0) { continue; }
        auto const score    = node_scores[edge];
        auto const neighbor = node_graph[edge];
        if (score < best_score || (score == best_score && neighbor < best_id)) {
          best_edge  = edge;
          best_id    = neighbor;
          best_score = score;
        }
      }
      if (best_edge < graph_degree) {
        node_selected[best_edge] = 1;
        node_resident[rank]      = best_id;
      }
    }

    auto* node_cross = cross_graph + local_node * (cross_degree + 1);
    node_cross[0]    = cross_degree;
    auto output      = std::uint32_t{0};
    for (std::uint32_t edge = 0; edge < graph_degree; ++edge) {
      if (node_selected[edge] == 0) { node_cross[++output] = node_graph[edge]; }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && resident_degree > 0) {
    auto const global_first_node =
      batch_offset + static_cast<std::uint64_t>(blockIdx.x) * node_per_cacheline;
    auto* inner_row = inner_graph + (global_first_node / node_per_cacheline) * inner_row_bytes;
    for (std::uint32_t byte = 0; byte < inner_row_bytes; ++byte) {
      inner_row[byte] = 0;
    }
    auto bit_offset = static_cast<std::uint64_t>(node_per_cacheline) * 8;
    for (std::uint16_t column = 0; column < node_per_cacheline; ++column) {
      auto const node   = static_cast<std::uint64_t>(blockIdx.x) * node_per_cacheline + column;
      auto const count  = node < batch_rows ? resident_degree : 0;
      inner_row[column] = static_cast<std::uint8_t>(count);
      for (std::uint32_t rank = 0; rank < count; ++rank) {
        set_packed_bits(inner_row,
                        bit_offset + static_cast<std::uint64_t>(rank) * n_bits,
                        resident[static_cast<std::size_t>(column) * resident_degree + rank],
                        n_bits);
      }
      bit_offset += static_cast<std::uint64_t>(count) * n_bits;
    }
  }
}

template <typename T, bool StagedRows>
__global__ void rank_graph_rows_kernel(T const* dataset,
                                       std::uint32_t dataset_stride,
                                       T const* staged_rows,
                                       graph_index_type const* input_graph,
                                       std::uint32_t const* labels,
                                       std::uint8_t* output_order,
                                       std::uint64_t batch_offset,
                                       std::uint32_t batch_rows,
                                       std::uint32_t dim,
                                       std::uint32_t graph_degree,
                                       cuvs::distance::DistanceType metric)
{
  extern __shared__ std::uint8_t shared_bytes[];
  auto const warp       = threadIdx.x / raft::WarpSize;
  auto const lane       = threadIdx.x % raft::WarpSize;
  auto const warps      = blockDim.x / raft::WarpSize;
  auto const local_node = static_cast<std::uint64_t>(blockIdx.x) * warps + warp;
  auto* scores          = reinterpret_cast<float*>(shared_bytes) + warp * graph_degree;
  auto* selected = shared_bytes + static_cast<std::size_t>(warps) * graph_degree * sizeof(float) +
                   warp * graph_degree;
  if (local_node >= batch_rows) { return; }

  if (lane == 0) {
    for (std::uint32_t edge = 0; edge < graph_degree; ++edge) {
      selected[edge]                                 = 0;
      output_order[local_node * graph_degree + edge] = std::uint8_t{255};
    }
  }
  auto const global_node = batch_offset + local_node;
  auto const* source     = StagedRows ? staged_rows + local_node * (graph_degree + 1) * dim
                                      : dataset + global_node * dataset_stride;
  for (std::uint32_t edge = 0; edge < graph_degree; ++edge) {
    auto const neighbor = input_graph[local_node * graph_degree + edge];
    auto const* target  = StagedRows
                            ? staged_rows + (local_node * (graph_degree + 1) + edge + 1) * dim
                            : dataset + static_cast<std::uint64_t>(neighbor) * dataset_stride;
    auto score          = 0.0f;
    auto source_l2      = 0.0f;
    auto target_l2      = 0.0f;
    for (std::uint32_t column = lane; column < dim; column += raft::WarpSize) {
      auto const lhs = rank_value(source[column]);
      auto const rhs = rank_value(target[column]);
      switch (metric) {
        case cuvs::distance::DistanceType::L2Expanded: {
          auto const diff = lhs - rhs;
          score += diff * diff;
          break;
        }
        case cuvs::distance::DistanceType::InnerProduct: score -= lhs * rhs; break;
        case cuvs::distance::DistanceType::CosineExpanded:
          score -= lhs * rhs;
          source_l2 += lhs * lhs;
          target_l2 += rhs * rhs;
          break;
        default: score = std::numeric_limits<float>::infinity(); break;
      }
    }
    for (auto offset = raft::WarpSize / 2; offset > 0; offset /= 2) {
      score += __shfl_down_sync(0xffffffffu, score, offset);
      source_l2 += __shfl_down_sync(0xffffffffu, source_l2, offset);
      target_l2 += __shfl_down_sync(0xffffffffu, target_l2, offset);
    }
    if (lane == 0) {
      if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        auto const denominator = sqrtf(source_l2 * target_l2);
        score =
          denominator > 0.0f ? 1.0f + score / denominator : std::numeric_limits<float>::infinity();
      }
      scores[edge] = isnan(score) ? std::numeric_limits<float>::infinity() : score;
    }
  }
  __syncwarp();

  if (lane == 0) {
    auto* output = output_order + local_node * graph_degree;
    for (std::uint32_t rank = 0; rank < graph_degree; ++rank) {
      auto best_edge  = graph_degree;
      auto best_id    = std::numeric_limits<graph_index_type>::max();
      auto best_score = std::numeric_limits<float>::infinity();
      for (std::uint32_t edge = 0; edge < graph_degree; ++edge) {
        if (selected[edge] != 0) { continue; }
        auto const neighbor = input_graph[local_node * graph_degree + edge];
        if (labels[neighbor] != labels[global_node]) { continue; }
        if (scores[edge] < best_score || (scores[edge] == best_score && neighbor < best_id)) {
          best_edge  = edge;
          best_id    = neighbor;
          best_score = scores[edge];
        }
      }
      if (best_edge == graph_degree) { break; }
      selected[best_edge] = 1;
      output[best_edge]   = static_cast<std::uint8_t>(rank);
    }
  }
}

template <typename T>
__global__ void gather_float_rows_kernel(float* output,
                                         T const* input,
                                         std::uint64_t const* row_ids,
                                         std::uint64_t row_offset,
                                         std::uint64_t n_rows,
                                         std::uint32_t dim,
                                         std::uint32_t stride)
{
  auto index  = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto length = n_rows * dim;
  auto step   = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (; index < length; index += step) {
    auto const output_row = index / dim;
    auto const column     = index % dim;
    auto const input_row  = row_ids == nullptr ? row_offset + output_row : row_ids[output_row];
    output[index]         = rank_value(input[input_row * stride + column]);
  }
}

template <typename DatasetViewT>
auto rank_graph_edge_order(raft::resources const& res,
                           cuvs::distance::DistanceType metric,
                           host_graph_view graph,
                           raft::host_vector_view<const std::uint32_t, int64_t> labels,
                           DatasetViewT const& dataset)
  -> raft::host_matrix<std::uint8_t, int64_t, raft::row_major>
{
  using value_type =
    std::remove_const_t<typename decltype(std::declval<DatasetViewT const&>().view())::value_type>;
  constexpr bool host_dataset = cuvs::neighbors::is_host_dataset_view_v<DatasetViewT>;

  auto const n_rows = static_cast<std::uint64_t>(graph.extent(0));
  auto const degree = static_cast<std::uint32_t>(graph.extent(1));
  auto const dim    = dataset.dim();
  RAFT_EXPECTS(n_rows > 0 && degree > 0 && dim > 0,
               "FlowANN graph ranking requires a non-empty graph and dataset");
  RAFT_EXPECTS(degree <= std::numeric_limits<std::uint8_t>::max(),
               "FlowANN graph ranking requires degree <= 255");
  RAFT_EXPECTS(dataset.n_rows() == graph.extent(0),
               "FlowANN graph and full-precision dataset row counts differ");
  RAFT_EXPECTS(labels.extent(0) == graph.extent(0),
               "FlowANN graph ranking requires one group label per row");
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2Expanded ||
                 metric == cuvs::distance::DistanceType::InnerProduct ||
                 metric == cuvs::distance::DistanceType::CosineExpanded,
               "FlowANN grouped graph ranking supports L2, inner product, and cosine metrics");

  auto first_invalid = std::atomic<std::uint64_t>{std::numeric_limits<std::uint64_t>::max()};
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t node = 0; node < graph.extent(0); ++node) {
    for (int64_t edge = 0; edge < graph.extent(1); ++edge) {
      if (graph(node, edge) >= n_rows) {
        auto const position = static_cast<std::uint64_t>(node) * degree + edge;
        auto expected       = first_invalid.load(std::memory_order_relaxed);
        while (position < expected &&
               !first_invalid.compare_exchange_weak(
                 expected, position, std::memory_order_relaxed, std::memory_order_relaxed)) {}
      }
    }
  }
  auto const invalid = first_invalid.load(std::memory_order_relaxed);
  RAFT_EXPECTS(invalid == std::numeric_limits<std::uint64_t>::max(),
               "FlowANN graph contains an out-of-range neighbor at row %llu, column %llu",
               static_cast<unsigned long long>(invalid / degree),
               static_cast<unsigned long long>(invalid % degree));

  constexpr auto target_staging_bytes = std::uint64_t{256} << 20;
  auto const bytes_per_staged_row =
    static_cast<std::uint64_t>(degree + 1) * dim * sizeof(value_type);
  auto batch_rows = host_dataset
                      ? std::max<std::uint64_t>(1, target_staging_bytes / bytes_per_staged_row)
                      : std::uint64_t{1} << 20;
  batch_rows      = std::min<std::uint64_t>(batch_rows, n_rows);
  RAFT_EXPECTS(batch_rows > 0 && batch_rows <= std::numeric_limits<int64_t>::max(),
               "FlowANN could not derive a valid graph-ranking batch size");

  auto edge_order = raft::make_host_matrix<std::uint8_t, int64_t>(graph.extent(0), graph.extent(1));
  auto device_input = raft::make_device_matrix<graph_index_type, int64_t>(
    res, static_cast<int64_t>(batch_rows), graph.extent(1));
  auto device_output = raft::make_device_matrix<std::uint8_t, int64_t>(
    res, static_cast<int64_t>(batch_rows), graph.extent(1));
  auto device_labels = raft::make_device_vector<std::uint32_t, int64_t>(res, graph.extent(0));

  std::optional<raft::device_matrix<value_type, int64_t>> device_staging;
  std::optional<raft::pinned_vector<value_type, int64_t>> host_staging;
  if constexpr (host_dataset) {
    RAFT_EXPECTS(batch_rows <= std::numeric_limits<std::size_t>::max() /
                                 (static_cast<std::size_t>(degree) + 1) / dim,
                 "FlowANN graph-ranking staging size overflows size_t");
    auto const elements = static_cast<std::size_t>(batch_rows) * (degree + 1) * dim;
    host_staging.emplace(raft::make_pinned_vector<value_type, int64_t>(res, elements));
    device_staging.emplace(raft::make_device_matrix<value_type, int64_t>(
      res, static_cast<int64_t>(batch_rows) * (degree + 1), dim));
  }

  constexpr auto warps_per_block = std::uint32_t{4};
  constexpr auto block_size      = warps_per_block * raft::WarpSize;
  auto const shared_bytes =
    static_cast<std::size_t>(warps_per_block) * degree * (sizeof(float) + sizeof(std::uint8_t));
  auto device_id = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device_id));
  auto shared_memory_limit = 0;
  RAFT_CUDA_TRY(
    cudaDeviceGetAttribute(&shared_memory_limit, cudaDevAttrMaxSharedMemoryPerBlock, device_id));
  RAFT_EXPECTS(shared_bytes <= static_cast<std::size_t>(shared_memory_limit),
               "FlowANN graph-ranking kernel requires %zu bytes of shared memory, but the device "
               "limit is %d bytes",
               shared_bytes,
               shared_memory_limit);

  auto const stream         = raft::resource::get_cuda_stream(res);
  auto const raw_stream     = cuda_stream_handle(stream);
  auto const dataset_rows   = dataset.view();
  auto const dataset_stride = dataset.stride();
  raft::copy(device_labels.data_handle(), labels.data_handle(), labels.extent(0), stream);
  for (std::uint64_t offset = 0; offset < n_rows; offset += batch_rows) {
    auto const count =
      static_cast<std::uint32_t>(std::min<std::uint64_t>(batch_rows, n_rows - offset));
    raft::copy(device_input.data_handle(),
               graph.data_handle() + offset * degree,
               static_cast<std::size_t>(count) * degree,
               stream);

    auto const blocks = raft::div_rounding_up_safe<std::uint32_t>(count, warps_per_block);
    if constexpr (host_dataset) {
      auto* staging = host_staging->data_handle();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (std::uint32_t local = 0; local < count; ++local) {
        auto const global = offset + local;
        auto* output      = staging + static_cast<std::size_t>(local) * (degree + 1) * dim;
        std::copy_n(dataset_rows.data_handle() + global * dataset_stride, dim, output);
        auto const* node_graph = graph.data_handle() + global * degree;
        for (std::uint32_t edge = 0; edge < degree; ++edge) {
          std::copy_n(dataset_rows.data_handle() +
                        static_cast<std::uint64_t>(node_graph[edge]) * dataset_stride,
                      dim,
                      output + static_cast<std::size_t>(edge + 1) * dim);
        }
      }
      raft::copy(device_staging->data_handle(),
                 host_staging->data_handle(),
                 static_cast<std::size_t>(count) * (degree + 1) * dim,
                 stream);
      rank_graph_rows_kernel<value_type, true>
        <<<blocks, block_size, shared_bytes, raw_stream>>>(nullptr,
                                                           0,
                                                           device_staging->data_handle(),
                                                           device_input.data_handle(),
                                                           device_labels.data_handle(),
                                                           device_output.data_handle(),
                                                           offset,
                                                           count,
                                                           dim,
                                                           degree,
                                                           metric);
    } else {
      rank_graph_rows_kernel<value_type, false>
        <<<blocks, block_size, shared_bytes, raw_stream>>>(dataset_rows.data_handle(),
                                                           dataset_stride,
                                                           nullptr,
                                                           device_input.data_handle(),
                                                           device_labels.data_handle(),
                                                           device_output.data_handle(),
                                                           offset,
                                                           count,
                                                           dim,
                                                           degree,
                                                           metric);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::copy(edge_order.data_handle() + offset * degree,
               device_output.data_handle(),
               static_cast<std::size_t>(count) * degree,
               stream);
  }
  raft::resource::sync_stream(res);
  return edge_order;
}

template <typename DatasetViewT>
auto convert_uniform_ranked_graph(
  raft::resources const& res,
  index_params const& params,
  raft::host_matrix_view<const graph_index_type, int64_t, raft::row_major> graph,
  DatasetViewT const& dataset) -> uniform_conversion_result
{
  using value_type =
    std::remove_const_t<typename decltype(std::declval<DatasetViewT const&>().view())::value_type>;
  constexpr bool host_dataset = cuvs::neighbors::is_host_dataset_view_v<DatasetViewT>;

  auto const n_rows = static_cast<std::uint64_t>(graph.extent(0));
  auto const degree = static_cast<std::uint32_t>(graph.extent(1));
  auto const dim    = dataset.dim();
  RAFT_EXPECTS(n_rows > 0 && degree > 0 && dim > 0,
               "FlowANN integrated build requires a non-empty graph and dataset");
  RAFT_EXPECTS(dataset.n_rows() == graph.extent(0),
               "FlowANN final graph and full-precision dataset row counts differ");
  RAFT_EXPECTS(params.node_per_cacheline > 0 && params.node_per_cacheline <= 32,
               "FlowANN uniform build requires node_per_cacheline in [1, 32]");
  RAFT_EXPECTS(params.cagra_params.metric == cuvs::distance::DistanceType::L2Expanded ||
                 params.cagra_params.metric == cuvs::distance::DistanceType::InnerProduct ||
                 params.cagra_params.metric == cuvs::distance::DistanceType::CosineExpanded ||
                 params.cagra_params.metric == cuvs::distance::DistanceType::L1 ||
                 params.cagra_params.metric == cuvs::distance::DistanceType::BitwiseHamming,
               "FlowANN uniform build does not support the requested CAGRA metric");

  auto first_invalid = std::atomic<std::uint64_t>{std::numeric_limits<std::uint64_t>::max()};
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t node = 0; node < graph.extent(0); ++node) {
    for (int64_t edge = 0; edge < graph.extent(1); ++edge) {
      if (graph(node, edge) >= n_rows) {
        auto const position = static_cast<std::uint64_t>(node) * degree + edge;
        auto expected       = first_invalid.load(std::memory_order_relaxed);
        while (position < expected &&
               !first_invalid.compare_exchange_weak(
                 expected, position, std::memory_order_relaxed, std::memory_order_relaxed)) {}
      }
    }
  }
  auto const invalid = first_invalid.load(std::memory_order_relaxed);
  RAFT_EXPECTS(invalid == std::numeric_limits<std::uint64_t>::max(),
               "FlowANN final graph contains an out-of-range neighbor at row %llu, column %llu",
               static_cast<unsigned long long>(invalid / degree),
               static_cast<unsigned long long>(invalid % degree));

  auto const layout = plan_uniform_layout(
    n_rows, degree, params.node_per_cacheline, params.device_graph_budget_bytes);
  auto inner_graph = layout.resident_degree == 0
                       ? raft::make_device_matrix<std::uint8_t, int64_t>(res, 0, 0)
                       : raft::make_device_matrix<std::uint8_t, int64_t>(
                           res,
                           raft::div_rounding_up_safe<int64_t>(static_cast<int64_t>(n_rows),
                                                               params.node_per_cacheline),
                           layout.inner_row_bytes);
  auto cross_graph = raft::make_host_matrix<graph_index_type, int64_t>(
    static_cast<int64_t>(n_rows), static_cast<int64_t>(layout.cross_degree) + 1);

  if (layout.resident_degree == 0) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t node = 0; node < graph.extent(0); ++node) {
      cross_graph(node, 0) = degree;
      std::copy_n(graph.data_handle() + static_cast<std::size_t>(node) * degree,
                  degree,
                  cross_graph.data_handle() +
                    static_cast<std::size_t>(node) * (static_cast<std::size_t>(degree) + 1) + 1);
    }
    return {std::move(inner_graph), std::move(cross_graph), layout};
  }

  constexpr auto target_staging_bytes = std::uint64_t{256} << 20;
  auto bytes_per_staged_row = static_cast<std::uint64_t>(degree + 1) * dim * sizeof(value_type);
  auto batch_rows           = host_dataset
                                ? std::max<std::uint64_t>(params.node_per_cacheline,
                                                target_staging_bytes / bytes_per_staged_row)
                                : std::uint64_t{1} << 20;
  batch_rows -= batch_rows % params.node_per_cacheline;
  batch_rows =
    std::max<std::uint64_t>(params.node_per_cacheline, std::min<std::uint64_t>(batch_rows, n_rows));
  if (batch_rows < n_rows && batch_rows % params.node_per_cacheline != 0) {
    batch_rows -= batch_rows % params.node_per_cacheline;
  }
  RAFT_EXPECTS(batch_rows > 0 && batch_rows <= std::numeric_limits<int64_t>::max(),
               "FlowANN could not derive a valid rank batch size");

  auto device_graph = raft::make_device_matrix<graph_index_type, int64_t>(
    res, static_cast<int64_t>(batch_rows), degree);
  auto device_cross = raft::make_device_matrix<graph_index_type, int64_t>(
    res, static_cast<int64_t>(batch_rows), static_cast<int64_t>(layout.cross_degree) + 1);

  std::optional<raft::device_matrix<value_type, int64_t>> device_staging;
  std::optional<raft::pinned_vector<value_type, int64_t>> host_staging;
  if constexpr (host_dataset) {
    RAFT_EXPECTS(batch_rows <= std::numeric_limits<std::size_t>::max() /
                                 (static_cast<std::size_t>(degree) + 1) / dim,
                 "FlowANN host rank staging size overflows size_t");
    auto const elements = static_cast<std::size_t>(batch_rows) * (degree + 1) * dim;
    host_staging.emplace(raft::make_pinned_vector<value_type, int64_t>(res, elements));
    device_staging.emplace(raft::make_device_matrix<value_type, int64_t>(
      res, static_cast<int64_t>(batch_rows) * (degree + 1), dim));
  }

  auto const stream     = raft::resource::get_cuda_stream(res);
  auto const raw_stream = cuda_stream_handle(stream);
  if (inner_graph.size() > 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(
      inner_graph.data_handle(), 0, inner_graph.size() * sizeof(std::uint8_t), raw_stream));
  }
  auto const block_size = static_cast<std::uint32_t>(params.node_per_cacheline) * raft::WarpSize;
  auto shared_bytes     = static_cast<std::size_t>(params.node_per_cacheline) * degree *
                      (sizeof(float) + sizeof(std::uint8_t));
  shared_bytes = (shared_bytes + alignof(graph_index_type) - 1) & ~(alignof(graph_index_type) - 1);
  shared_bytes += static_cast<std::size_t>(params.node_per_cacheline) * layout.resident_degree *
                  sizeof(graph_index_type);
  auto device_id = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device_id));
  auto shared_memory_limit = 0;
  RAFT_CUDA_TRY(
    cudaDeviceGetAttribute(&shared_memory_limit, cudaDevAttrMaxSharedMemoryPerBlock, device_id));
  RAFT_EXPECTS(shared_bytes <= static_cast<std::size_t>(shared_memory_limit),
               "FlowANN rank kernel requires %zu bytes of shared memory, but the device limit is "
               "%d bytes",
               shared_bytes,
               shared_memory_limit);

  auto const dataset_rows   = dataset.view();
  auto const dataset_stride = dataset.stride();
  for (std::uint64_t offset = 0; offset < n_rows; offset += batch_rows) {
    auto const count =
      static_cast<std::uint32_t>(std::min<std::uint64_t>(batch_rows, n_rows - offset));
    raft::copy(device_graph.data_handle(),
               graph.data_handle() + offset * degree,
               static_cast<std::size_t>(count) * degree,
               stream);

    if constexpr (host_dataset) {
      auto* staging = host_staging->data_handle();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (std::uint32_t local = 0; local < count; ++local) {
        auto const global = offset + local;
        auto* output      = staging + static_cast<std::size_t>(local) * (degree + 1) * dim;
        std::copy_n(dataset_rows.data_handle() + global * dataset_stride, dim, output);
        auto const* node_graph = graph.data_handle() + global * degree;
        for (std::uint32_t edge = 0; edge < degree; ++edge) {
          std::copy_n(dataset_rows.data_handle() +
                        static_cast<std::uint64_t>(node_graph[edge]) * dataset_stride,
                      dim,
                      output + static_cast<std::size_t>(edge + 1) * dim);
        }
      }
      raft::copy(device_staging->data_handle(),
                 host_staging->data_handle(),
                 static_cast<std::size_t>(count) * (degree + 1) * dim,
                 stream);
      auto const blocks =
        raft::div_rounding_up_safe<std::uint32_t>(count, params.node_per_cacheline);
      rank_and_split_kernel<value_type, true>
        <<<blocks, block_size, shared_bytes, raw_stream>>>(nullptr,
                                                           0,
                                                           device_staging->data_handle(),
                                                           device_graph.data_handle(),
                                                           offset,
                                                           count,
                                                           dim,
                                                           degree,
                                                           layout.resident_degree,
                                                           layout.cross_degree,
                                                           params.node_per_cacheline,
                                                           layout.n_bits,
                                                           layout.inner_row_bytes,
                                                           inner_graph.data_handle(),
                                                           device_cross.data_handle(),
                                                           params.cagra_params.metric);
    } else {
      auto const blocks =
        raft::div_rounding_up_safe<std::uint32_t>(count, params.node_per_cacheline);
      rank_and_split_kernel<value_type, false>
        <<<blocks, block_size, shared_bytes, raw_stream>>>(dataset_rows.data_handle(),
                                                           dataset_stride,
                                                           nullptr,
                                                           device_graph.data_handle(),
                                                           offset,
                                                           count,
                                                           dim,
                                                           degree,
                                                           layout.resident_degree,
                                                           layout.cross_degree,
                                                           params.node_per_cacheline,
                                                           layout.n_bits,
                                                           layout.inner_row_bytes,
                                                           inner_graph.data_handle(),
                                                           device_cross.data_handle(),
                                                           params.cagra_params.metric);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::copy(
      cross_graph.data_handle() + offset * (static_cast<std::uint64_t>(layout.cross_degree) + 1),
      device_cross.data_handle(),
      static_cast<std::size_t>(count) * (layout.cross_degree + 1),
      stream);
    raft::resource::sync_stream(res);
  }

  return {std::move(inner_graph), std::move(cross_graph), layout};
}

template <typename DatasetViewT>
auto gather_grouping_rows(raft::resources const& res,
                          DatasetViewT const& dataset,
                          std::vector<std::uint64_t> const& ids)
  -> raft::device_matrix<float, int64_t>
{
  auto const dim    = dataset.dim();
  auto const stream = raft::resource::get_cuda_stream(res);
  auto result       = raft::make_device_matrix<float, int64_t>(res, ids.size(), dim);
  if constexpr (cuvs::neighbors::is_host_dataset_view_v<DatasetViewT>) {
    auto staging = raft::make_pinned_vector<float, int64_t>(res, ids.size() * dim);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < ids.size(); ++i) {
      for (int64_t col = 0; col < dim; ++col) {
        staging(i * dim + col) =
          static_cast<float>(dataset.view().data_handle()[ids[i] * dataset.stride() + col]);
      }
    }
    raft::copy(result.data_handle(), staging.data_handle(), result.size(), stream);
    raft::resource::sync_stream(res);
  } else {
    auto device_ids = raft::make_device_vector<std::uint64_t, int64_t>(res, ids.size());
    raft::copy(device_ids.data_handle(), ids.data(), ids.size(), stream);
    auto const blocks =
      static_cast<unsigned>(std::min<std::uint64_t>((result.size() + 255) / 256, 65535));
    gather_float_rows_kernel<<<blocks, 256, 0, cuda_stream_handle(stream)>>>(
      result.data_handle(),
      dataset.view().data_handle(),
      device_ids.data_handle(),
      0,
      ids.size(),
      dim,
      dataset.stride());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::resource::sync_stream(res);
  }
  return result;
}

// Compute nearest-child labels beside the distance matrix, before copying it to the host.
// This avoids another full CPU pass over N * fanout distances solely for argmin/validation.
__global__ void label_group_distances(float* costs,
                                      std::uint8_t* labels,
                                      std::uint64_t rows,
                                      std::uint32_t branches,
                                      bool inner_product,
                                      int* invalid)
{
  for (std::uint64_t row = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x; row < rows;
       row += std::uint64_t{blockDim.x} * gridDim.x) {
    auto* values      = costs + row * branches;
    std::uint8_t best = 0;
    bool finite       = true;
    for (std::uint32_t group = 0; group < branches; ++group) {
      auto value = values[group];
      if (inner_product) {
        value         = -value;
        values[group] = value;
      }
      finite = finite && isfinite(value);
      if (value < values[best]) { best = group; }
    }
    labels[row] = best;
    if (!finite) { atomicExch(invalid, 1); }
  }
}

template <typename DatasetViewT>
auto generate_group_labels(raft::resources const& res,
                           index_params const& index_config,
                           host_graph_view graph,
                           DatasetViewT const& dataset) -> grouping_result
{
  auto const& params = index_config.grouping;
  auto const metric  = index_config.cagra_params.metric;
  RAFT_EXPECTS(graph.extent(0) == dataset.n_rows() && graph.extent(1) > 0 && graph.extent(1) <= 255,
               "FlowANN grouping graph must match dataset rows and have degree in [1, 255]");
  RAFT_EXPECTS(dataset.n_rows() > 0 && static_cast<std::uint64_t>(dataset.n_rows()) <=
                                         std::numeric_limits<std::uint32_t>::max(),
               "FlowANN grouping dataset row count must fit uint32 graph IDs");
  RAFT_EXPECTS(dataset.dim() > 0 && params.kmeans_n_iters > 0,
               "FlowANN grouping requires positive dimension and training iterations");
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2Expanded ||
                 metric == cuvs::distance::DistanceType::InnerProduct ||
                 metric == cuvs::distance::DistanceType::CosineExpanded,
               "FlowANN grouping requires L2Expanded, InnerProduct, or CosineExpanded");
  auto const n_rows = static_cast<std::uint64_t>(dataset.n_rows());
  auto const dim    = dataset.dim();
  auto const plan = plan_grouping(n_rows, params.n_bits, params.n_groups, params.balance_tolerance);
  RAFT_EXPECTS(plan.n_groups <= max_partitions,
               "FlowANN group metadata supports at most %u groups",
               max_partitions);
  auto labels = raft::make_host_vector<std::uint32_t, int64_t>(n_rows);
  if (plan.n_groups == 1) {
    std::fill_n(labels.data_handle(), n_rows, std::uint32_t{0});
    return {std::move(labels), plan.n_bits, plan.n_groups};
  }
  auto const stream = raft::resource::get_cuda_stream(res);
  std::vector<std::uint32_t> members(plan.n_groups > 16 ? n_rows : 0), scratch(members.size());
  std::iota(members.begin(), members.end(), std::uint32_t{0});
  auto source_id = [&](std::uint64_t position) {
    return members.empty() ? static_cast<std::uint32_t>(position) : members[position];
  };
  double training_seconds = 0, assignment_seconds = 0, rebalance_seconds = 0;
  auto elapsed = [](auto begin) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  };
  auto partition = [&](auto&& self,
                       std::uint64_t begin,
                       std::uint64_t count,
                       std::uint32_t first_leaf,
                       std::uint32_t leaves) -> void {
    RAFT_EXPECTS(count >= leaves * plan.lower && count <= leaves * plan.upper,
                 "Subtree violates descendant capacity bounds");
    if (leaves == 1) {
      for (std::uint64_t i = begin; i < begin + count; ++i) {
        labels(members[i]) = first_leaf;
      }
      return;
    }
    auto const children = grouping_children(leaves);
    auto const branches = children.size();
    std::vector<std::uint64_t> sizes;
    std::vector<std::uint64_t> starts(branches + 1, begin);
    {
      auto stage_begin = std::chrono::steady_clock::now();
      // Share the automatic sample budget across subtrees instead of multiplying a full
      // budget by the number of nodes. Keep a minimum sample per centroid for deep trees.
      auto const automatic_rows = std::max<std::uint64_t>(
        256ull * branches,
        std::min<std::uint64_t>(10'000ull * branches,
                                (160'000ull * leaves + plan.n_groups - 1) / plan.n_groups));
      auto const training_rows = std::min<std::uint64_t>(
        count, params.training_rows == 0 ? automatic_rows : params.training_rows);
      RAFT_EXPECTS(training_rows >= branches,
                   "Grouping training sample is smaller than node fanout");
      std::vector<std::uint64_t> sample_ids(training_rows);
      for (std::uint64_t i = 0; i < training_rows; ++i) {
        sample_ids[i] = source_id(begin + i * count / training_rows);
      }
      auto sample    = gather_grouping_rows(res, dataset, sample_ids);
      auto centroids = raft::make_device_matrix<float, int64_t>(res, branches, dim);
      cuvs::cluster::kmeans::balanced_params kmeans_params;
      kmeans_params.n_iters                 = params.kmeans_n_iters;
      kmeans_params.metric                  = metric;
      kmeans_params.balance_lower_tolerance = 1.0f - params.balance_tolerance;
      kmeans_params.balance_upper_tolerance = 1.0f + params.balance_tolerance;
      cuvs::cluster::kmeans::fit(
        res, kmeans_params, raft::make_const_mdspan(sample.view()), centroids.view());
      raft::resource::sync_stream(res);
      training_seconds += elapsed(stage_begin);
      stage_begin = std::chrono::steady_clock::now();
      // Every distance is written by the GPU copy; avoid an extra full-matrix zero fill.
      auto costs = std::unique_ptr<float[]>(new float[count * branches]);
      std::vector<std::uint8_t> node_labels(count);
      auto const batch_rows = std::min<std::uint64_t>(
        count,
        params.assignment_batch_rows == 0
          ? std::max<std::uint64_t>(1, (256ull << 20) / (dim * sizeof(float)))
          : params.assignment_batch_rows);
      auto distances    = raft::make_device_matrix<float, int64_t>(res, batch_rows, branches);
      auto batch_labels = raft::make_device_vector<std::uint8_t, int64_t>(res, batch_rows);
      auto invalid      = raft::make_device_vector<int, int64_t>(res, 1);
      RAFT_CUDA_TRY(
        cudaMemsetAsync(invalid.data_handle(), 0, sizeof(int), cuda_stream_handle(stream)));
      auto batch = raft::make_device_matrix<float, int64_t>(res, batch_rows, dim);
      std::optional<raft::pinned_vector<float, int64_t>> staging;
      std::optional<raft::device_vector<std::uint64_t, int64_t>> device_ids;
      std::vector<std::uint64_t> ids;
      if constexpr (cuvs::neighbors::is_host_dataset_view_v<DatasetViewT>) {
        staging.emplace(raft::make_pinned_vector<float, int64_t>(res, batch_rows * dim));
      } else {
        device_ids.emplace(raft::make_device_vector<std::uint64_t, int64_t>(res, batch_rows));
        ids.resize(batch_rows);
      }
      for (std::uint64_t offset = 0; offset < count; offset += batch_rows) {
        auto const size = std::min(batch_rows, count - offset);
        if constexpr (cuvs::neighbors::is_host_dataset_view_v<DatasetViewT>) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
          for (std::uint64_t row = 0; row < size; ++row) {
            auto const source = source_id(begin + offset + row);
            for (int64_t col = 0; col < dim; ++col) {
              (*staging)(row* dim + col) = static_cast<float>(
                dataset.view().data_handle()[std::uint64_t{source} * dataset.stride() + col]);
            }
          }
          raft::copy(batch.data_handle(), staging->data_handle(), size * dim, stream);
        } else {
          for (std::uint64_t row = 0; row < size; ++row) {
            ids[row] = source_id(begin + offset + row);
          }
          raft::copy(device_ids->data_handle(), ids.data(), size, stream);
          auto const blocks =
            static_cast<unsigned>(std::min<std::uint64_t>((size * dim + 255) / 256, 65535));
          gather_float_rows_kernel<<<blocks, 256, 0, cuda_stream_handle(stream)>>>(
            batch.data_handle(),
            dataset.view().data_handle(),
            device_ids->data_handle(),
            0,
            size,
            dim,
            dataset.stride());
          RAFT_CUDA_TRY(cudaPeekAtLastError());
        }
        auto batch_view =
          raft::make_device_matrix_view<const float, int64_t>(batch.data_handle(), size, dim);
        auto distance_view =
          raft::make_device_matrix_view<float, int64_t>(distances.data_handle(), size, branches);
        cuvs::distance::pairwise_distance(
          res, batch_view, raft::make_const_mdspan(centroids.view()), distance_view, metric);
        auto const blocks =
          static_cast<unsigned>(std::min<std::uint64_t>((size + 255) / 256, 65535));
        label_group_distances<<<blocks, 256, 0, cuda_stream_handle(stream)>>>(
          distances.data_handle(),
          batch_labels.data_handle(),
          size,
          branches,
          metric == cuvs::distance::DistanceType::InnerProduct,
          invalid.data_handle());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
        raft::copy(node_labels.data() + offset, batch_labels.data_handle(), size, stream);
        raft::copy(
          costs.get() + offset * branches, distances.data_handle(), size * branches, stream);
        raft::resource::sync_stream(res);
      }
      int invalid_host = 0;
      raft::copy(&invalid_host, invalid.data_handle(), 1, stream);
      raft::resource::sync_stream(res);
      RAFT_EXPECTS(invalid_host == 0, "Grouping distances must be finite");
      assignment_seconds += elapsed(stage_begin);
      stage_begin = std::chrono::steady_clock::now();
      std::vector<std::uint64_t> lower(branches), upper(branches);
      for (std::size_t child = 0; child < branches; ++child) {
        lower[child] = children[child] * plan.lower;
        upper[child] = children[child] * plan.upper;
      }
      sizes = rebalance_group_distances(costs.get(), node_labels, lower, upper);
      if (leaves <= 16) {
        // The children are final leaves. Write labels directly, avoiding a full permutation
        // and another pass over all IDs for the common single-level case.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::uint64_t row = 0; row < count; ++row) {
          labels(source_id(begin + row)) = first_leaf + node_labels[row];
        }
        rebalance_seconds += elapsed(stage_begin);
        return;
      }
      for (std::size_t child = 0; child < branches; ++child) {
        starts[child + 1] = starts[child] + sizes[child];
      }
      auto next = starts;
      for (std::uint64_t i = 0; i < count; ++i) {
        scratch[next[node_labels[i]]++] = members[begin + i];
      }
      std::copy_n(scratch.data() + begin, count, members.data() + begin);
      // Do not retain per-row distances at ancestor levels while processing a child.
      costs.reset();
      std::vector<std::uint8_t>().swap(node_labels);
      rebalance_seconds += elapsed(stage_begin);
    }
    for (std::size_t child = 0; child < branches; ++child) {
      self(self, starts[child], sizes[child], first_leaf, children[child]);
      first_leaf += children[child];
    }
  };
  partition(partition, 0, n_rows, 0, plan.n_groups);
  RAFT_LOG_INFO(
    "FlowANN hierarchical grouping: bits=%u groups=%u max_fanout=16 "
    "training=%.6f assignment=%.6f rebalance=%.6f seconds",
    plan.n_bits,
    plan.n_groups,
    training_seconds,
    assignment_seconds,
    rebalance_seconds);
  return {std::move(labels), plan.n_bits, plan.n_groups};
}

template <typename DatasetViewT>
auto generate_medoid_seeds(raft::resources const& res,
                           index_params const& params,
                           DatasetViewT const& dataset)
  -> raft::host_vector<graph_index_type, int64_t>
{
  using source_type =
    std::remove_const_t<typename decltype(std::declval<DatasetViewT const&>().view())::value_type>;
  constexpr bool host_dataset = cuvs::neighbors::is_host_dataset_view_v<DatasetViewT>;

  RAFT_EXPECTS(params.num_seeds > 0 && params.num_seeds <= dataset.n_rows(),
               "FlowANN seed count must be in [1, dataset size]");
  RAFT_EXPECTS(params.cagra_params.metric == cuvs::distance::DistanceType::L2Expanded ||
                 params.cagra_params.metric == cuvs::distance::DistanceType::InnerProduct ||
                 params.cagra_params.metric == cuvs::distance::DistanceType::CosineExpanded,
               "FlowANN balanced-kmeans seeds require L2Expanded, InnerProduct, or CosineExpanded");

  auto const n_rows = static_cast<std::uint64_t>(dataset.n_rows());
  auto const dim    = dataset.dim();
  auto sample_rows =
    params.seed_training_rows == 0
      ? std::min<std::uint64_t>(
          n_rows,
          std::max<std::uint64_t>(params.num_seeds, std::uint64_t{10'000} * params.num_seeds))
      : std::min<std::uint64_t>(n_rows, params.seed_training_rows);
  RAFT_EXPECTS(sample_rows >= params.num_seeds,
               "FlowANN seed training sample must contain at least num_seeds rows");
  RAFT_EXPECTS(
    dim > 0 && sample_rows <= static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()) / dim,
    "FlowANN seed training sample size exceeds int64");

  auto sample_ids_host = raft::make_host_vector<std::uint64_t, int64_t>(sample_rows);
  auto const rotation  = params.seed % n_rows;
  for (std::uint64_t row = 0; row < sample_rows; ++row) {
    sample_ids_host(row) =
      (rotation + (static_cast<unsigned __int128>(row) * n_rows) / sample_rows) % n_rows;
  }

  auto sample           = raft::make_device_matrix<float, int64_t>(res, sample_rows, dim);
  auto const rows       = dataset.view();
  auto const stream     = raft::resource::get_cuda_stream(res);
  auto const raw_stream = cuda_stream_handle(stream);
  if constexpr (host_dataset) {
    auto sample_host =
      raft::make_pinned_vector<float, int64_t>(res, static_cast<int64_t>(sample_rows * dim));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::uint64_t row = 0; row < sample_rows; ++row) {
      auto const* input = rows.data_handle() + sample_ids_host(row) * dataset.stride();
      auto* output      = sample_host.data_handle() + row * dim;
      for (std::uint32_t column = 0; column < dim; ++column) {
        output[column] = static_cast<float>(input[column]);
      }
    }
    raft::copy(sample.data_handle(),
               sample_host.data_handle(),
               static_cast<std::size_t>(sample_rows) * dim,
               stream);
  } else {
    auto sample_ids_device = raft::make_device_vector<std::uint64_t, int64_t>(res, sample_rows);
    raft::copy(sample_ids_device.data_handle(), sample_ids_host.data_handle(), sample_rows, stream);
    constexpr auto threads = std::uint32_t{256};
    auto const blocks      = static_cast<std::uint32_t>(
      std::min<std::uint64_t>((sample_rows * dim + threads - 1) / threads, std::uint64_t{65535}));
    gather_float_rows_kernel<<<blocks, threads, 0, raw_stream>>>(sample.data_handle(),
                                                                 rows.data_handle(),
                                                                 sample_ids_device.data_handle(),
                                                                 0,
                                                                 sample_rows,
                                                                 dim,
                                                                 dataset.stride());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  auto centroids =
    raft::make_device_matrix<float, int64_t>(res, static_cast<int64_t>(params.num_seeds), dim);
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.n_iters = 20;
  kmeans_params.metric  = params.cagra_params.metric;
  cuvs::cluster::kmeans::fit(
    res, kmeans_params, raft::make_const_mdspan(sample.view()), centroids.view());
  raft::resource::sync_stream(res);

  constexpr auto target_batch_bytes = std::uint64_t{256} << 20;
  auto batch_rows                   = std::max<std::uint64_t>(
    params.num_seeds, target_batch_bytes / (static_cast<std::uint64_t>(dim) * sizeof(float)));
  batch_rows                = std::min(batch_rows, n_rows);
  auto const batch_capacity = std::min<std::uint64_t>(n_rows, batch_rows + params.num_seeds);
  RAFT_EXPECTS(
    batch_capacity <= static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()) / dim,
    "FlowANN seed scan batch size exceeds int64");
  auto batch = raft::make_device_matrix<float, int64_t>(res, batch_capacity, dim);
  auto nearest =
    raft::make_device_matrix<int64_t, int64_t>(res, params.num_seeds, params.num_seeds);
  auto nearest_distances =
    raft::make_device_matrix<float, int64_t>(res, params.num_seeds, params.num_seeds);
  auto nearest_host = raft::make_host_matrix<int64_t, int64_t>(params.num_seeds, params.num_seeds);
  auto distances_host = raft::make_host_matrix<float, int64_t>(params.num_seeds, params.num_seeds);
  std::optional<raft::pinned_vector<float, int64_t>> batch_host;
  if constexpr (host_dataset) {
    batch_host.emplace(
      raft::make_pinned_vector<float, int64_t>(res, static_cast<int64_t>(batch_capacity * dim)));
  }

  using candidate = std::pair<float, graph_index_type>;
  auto candidates = std::vector<std::vector<candidate>>(params.num_seeds);
  for (auto& list : candidates) {
    list.reserve(params.num_seeds * 2);
  }
  cuvs::neighbors::brute_force::index_params brute_params;
  brute_params.metric = params.cagra_params.metric;
  cuvs::neighbors::brute_force::search_params search_params;
  for (std::uint64_t offset = 0; offset < n_rows;) {
    auto count           = std::min<std::uint64_t>(batch_rows, n_rows - offset);
    auto const remainder = n_rows - offset - count;
    if (remainder > 0 && remainder < params.num_seeds) { count += remainder; }

    if constexpr (host_dataset) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (std::uint64_t row = 0; row < count; ++row) {
        auto const* input = rows.data_handle() + (offset + row) * dataset.stride();
        auto* output      = batch_host->data_handle() + row * dim;
        for (std::uint32_t column = 0; column < dim; ++column) {
          output[column] = static_cast<float>(input[column]);
        }
      }
      raft::copy(batch.data_handle(),
                 batch_host->data_handle(),
                 static_cast<std::size_t>(count) * dim,
                 stream);
    } else {
      constexpr auto threads = std::uint32_t{256};
      auto const blocks      = static_cast<std::uint32_t>(
        std::min<std::uint64_t>((count * dim + threads - 1) / threads, std::uint64_t{65535}));
      gather_float_rows_kernel<<<blocks, threads, 0, raw_stream>>>(
        batch.data_handle(), rows.data_handle(), nullptr, offset, count, dim, dataset.stride());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    auto batch_view = raft::make_device_matrix_view<const float, int64_t>(
      batch.data_handle(), static_cast<int64_t>(count), dim);
    auto brute_index = cuvs::neighbors::brute_force::build(res, brute_params, batch_view);
    cuvs::neighbors::brute_force::search(res,
                                         search_params,
                                         brute_index,
                                         raft::make_const_mdspan(centroids.view()),
                                         nearest.view(),
                                         nearest_distances.view());
    raft::copy(nearest_host.data_handle(), nearest.data_handle(), nearest.size(), stream);
    raft::copy(distances_host.data_handle(),
               nearest_distances.data_handle(),
               nearest_distances.size(),
               stream);
    raft::resource::sync_stream(res);

    for (std::uint32_t center = 0; center < params.num_seeds; ++center) {
      auto& list = candidates[center];
      for (std::uint32_t rank = 0; rank < params.num_seeds; ++rank) {
        list.emplace_back(distances_host(center, rank),
                          static_cast<graph_index_type>(offset + nearest_host(center, rank)));
      }
      std::sort(list.begin(), list.end());
      if (list.size() > params.num_seeds) { list.resize(params.num_seeds); }
    }
    offset += count;
  }

  struct assignment_candidate {
    float distance;
    graph_index_type row;
    std::uint32_t center;
  };
  auto assignments = std::vector<assignment_candidate>();
  assignments.reserve(static_cast<std::size_t>(params.num_seeds) * params.num_seeds);
  for (std::uint32_t center = 0; center < params.num_seeds; ++center) {
    for (auto const& [distance, row] : candidates[center]) {
      assignments.push_back({distance, row, center});
    }
  }
  std::sort(assignments.begin(), assignments.end(), [](auto const& lhs, auto const& rhs) {
    if (lhs.distance != rhs.distance) { return lhs.distance < rhs.distance; }
    if (lhs.row != rhs.row) { return lhs.row < rhs.row; }
    return lhs.center < rhs.center;
  });

  auto result         = raft::make_host_vector<graph_index_type, int64_t>(params.num_seeds);
  auto assigned       = std::vector<bool>(params.num_seeds, false);
  auto used           = std::set<graph_index_type>();
  auto assigned_count = std::uint32_t{0};
  for (auto const& candidate : assignments) {
    if (assigned[candidate.center] || used.contains(candidate.row)) { continue; }
    result(candidate.center)   = candidate.row;
    assigned[candidate.center] = true;
    used.insert(candidate.row);
    if (++assigned_count == params.num_seeds) { break; }
  }
  RAFT_EXPECTS(assigned_count == params.num_seeds,
               "FlowANN could not select a unique medoid for every seed centroid");
  return result;
}

void validate_build_params(build_params const& params, std::uint32_t graph_degree)
{
  RAFT_EXPECTS(
    params.node_per_cacheline > 0 && params.node_per_cacheline <= max_nodes_per_packed_row,
    "FlowANN node_per_cacheline must be in [1, %u]",
    static_cast<unsigned>(max_nodes_per_packed_row));
  RAFT_EXPECTS(params.n_bits > 0 && params.n_bits <= 32, "FlowANN n_bits must be in [1, 32]");
  RAFT_EXPECTS(graph_degree > 0, "FlowANN cannot build from an empty CAGRA graph");
  RAFT_EXPECTS(graph_degree <= std::numeric_limits<std::uint8_t>::max(),
               "FlowANN graph degree must fit in the packed uint8 neighbor count");
  if (params.inner_graph_row_bytes != 0) {
    RAFT_EXPECTS(params.inner_graph_row_bytes >= params.node_per_cacheline,
                 "FlowANN inner_graph_row_bytes is too small for its count header");
  }
}

auto copy_graph_to_host(
  raft::resources const& res,
  raft::device_matrix_view<const graph_index_type, int64_t, raft::row_major> graph)
  -> raft::host_matrix<graph_index_type, int64_t, raft::row_major>
{
  auto result = raft::make_host_matrix<graph_index_type, int64_t>(graph.extent(0), graph.extent(1));
  raft::copy(
    result.data_handle(), graph.data_handle(), graph.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return result;
}

auto validate_and_count_partitions(
  raft::host_vector_view<const std::uint32_t, int64_t> partition_labels, std::uint16_t n_bits)
  -> std::vector<std::uint64_t>
{
  RAFT_EXPECTS(partition_labels.extent(0) > 0, "FlowANN partition labels must not be empty");

  auto max_label = std::uint32_t{0};
  for (int64_t node = 0; node < partition_labels.extent(0); ++node) {
    max_label = std::max(max_label, partition_labels(node));
  }
  RAFT_EXPECTS(max_label < max_partitions,
               "FlowANN supports at most %u compact partition identifiers",
               max_partitions);

  auto partition_sizes = std::vector<std::uint64_t>(static_cast<std::size_t>(max_label) + 1, 0);
  for (int64_t node = 0; node < partition_labels.extent(0); ++node) {
    ++partition_sizes[partition_labels(node)];
  }

  auto const max_partition_size =
    n_bits == 32 ? (std::uint64_t{1} << 32) : (std::uint64_t{1} << n_bits);
  for (std::size_t partition = 0; partition < partition_sizes.size(); ++partition) {
    RAFT_EXPECTS(partition_sizes[partition] > 0,
                 "FlowANN partition identifiers must be compact; partition %zu is empty",
                 partition);
    RAFT_EXPECTS(partition_sizes[partition] <= max_partition_size,
                 "FlowANN partition %zu contains too many nodes for %u-bit local IDs",
                 partition,
                 static_cast<unsigned>(n_bits));
  }
  return partition_sizes;
}

auto count_local_neighbors(host_graph_view graph,
                           raft::host_vector_view<const std::uint32_t, int64_t> partition_labels)
  -> std::pair<std::vector<std::uint8_t>, std::uint64_t>
{
  auto const n_rows  = graph.extent(0);
  auto counts        = std::vector<std::uint8_t>(static_cast<std::size_t>(n_rows));
  auto first_invalid = std::atomic<std::uint64_t>{std::numeric_limits<std::uint64_t>::max()};
  auto total_local   = std::uint64_t{0};

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : total_local)
#endif
  for (int64_t node = 0; node < n_rows; ++node) {
    auto local_count = std::uint32_t{0};
    for (int64_t edge = 0; edge < graph.extent(1); ++edge) {
      auto const neighbor = graph(node, edge);
      if (neighbor >= static_cast<std::uint64_t>(n_rows)) {
        auto const position = static_cast<std::uint64_t>(node) * graph.extent(1) + edge;
        auto expected       = first_invalid.load(std::memory_order_relaxed);
        while (position < expected &&
               !first_invalid.compare_exchange_weak(
                 expected, position, std::memory_order_relaxed, std::memory_order_relaxed)) {}
        continue;
      }
      local_count += partition_labels(neighbor) == partition_labels(node);
    }
    counts[static_cast<std::size_t>(node)] = static_cast<std::uint8_t>(local_count);
    total_local += local_count;
  }

  auto const invalid = first_invalid.load(std::memory_order_relaxed);
  RAFT_EXPECTS(invalid == std::numeric_limits<std::uint64_t>::max(),
               "CAGRA graph contains an out-of-range neighbor at row %llu, column %llu",
               static_cast<unsigned long long>(invalid / graph.extent(1)),
               static_cast<unsigned long long>(invalid % graph.extent(1)));
  return {std::move(counts), total_local};
}

auto make_permutation(std::vector<std::uint8_t> const& local_counts,
                      raft::host_vector_view<const std::uint32_t, int64_t> partition_labels,
                      std::vector<std::uint64_t> const& partition_sizes,
                      std::uint16_t node_per_cacheline)
  -> std::pair<std::vector<graph_index_type>, std::vector<graph_index_type>>
{
  auto groups = std::vector<std::vector<graph_index_type>>(partition_sizes.size());
  for (std::size_t partition = 0; partition < groups.size(); ++partition) {
    groups[partition].reserve(static_cast<std::size_t>(partition_sizes[partition]));
  }
  for (int64_t node = 0; node < partition_labels.extent(0); ++node) {
    groups[partition_labels(node)].push_back(static_cast<graph_index_type>(node));
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int64_t partition = 0; partition < static_cast<int64_t>(groups.size()); ++partition) {
    auto& group = groups[static_cast<std::size_t>(partition)];
    std::sort(group.begin(), group.end(), [&](auto lhs, auto rhs) {
      auto const lhs_count = local_counts[lhs];
      auto const rhs_count = local_counts[rhs];
      return lhs_count != rhs_count ? lhs_count > rhs_count : lhs < rhs;
    });
  }

  auto const n_rows = static_cast<std::size_t>(partition_labels.extent(0));
  auto old_to_new   = std::vector<graph_index_type>(n_rows);
  auto new_to_old   = std::vector<graph_index_type>();
  new_to_old.reserve(n_rows);

  for (auto& group : groups) {
    if (node_per_cacheline == 2 && new_to_old.size() % 2 != 0 && group.size() > 1) {
      auto const middle = group.begin() + static_cast<std::ptrdiff_t>(group.size() / 2);
      std::rotate(group.begin(), middle, middle + 1);
    }

    std::size_t left  = 0;
    std::size_t right = group.size();
    bool take_left    = true;
    while (left < right) {
      auto const old_node = take_left ? group[left++] : group[--right];
      auto const new_node = static_cast<graph_index_type>(new_to_old.size());
      new_to_old.push_back(old_node);
      old_to_new[old_node] = new_node;
      take_left            = !take_left;
    }
  }

  RAFT_EXPECTS(new_to_old.size() == n_rows, "FlowANN node permutation is incomplete");
  return {std::move(old_to_new), std::move(new_to_old)};
}

auto compute_inner_row_bytes(build_params const& params,
                             std::uint64_t n_rows,
                             std::uint64_t total_local_neighbors) -> std::uint32_t
{
  if (params.inner_graph_row_bytes != 0) { return params.inner_graph_row_bytes; }

  auto const packed_rows = (n_rows + params.node_per_cacheline - 1) / params.node_per_cacheline;
  auto const header_bits = static_cast<std::uint64_t>(params.node_per_cacheline) * 8;
  auto const total_bits  = static_cast<unsigned __int128>(header_bits) * packed_rows +
                          static_cast<unsigned __int128>(total_local_neighbors) * params.n_bits;
  auto const denominator = static_cast<unsigned __int128>(packed_rows) * 8;
  auto const row_bytes   = (total_bits + denominator - 1) / denominator;
  RAFT_EXPECTS(row_bytes <= std::numeric_limits<std::uint32_t>::max(),
               "FlowANN inner graph row length exceeds uint32");
  return std::max<std::uint32_t>(params.node_per_cacheline, static_cast<std::uint32_t>(row_bytes));
}

void set_bits(std::uint8_t* destination,
              std::uint64_t bit_offset,
              std::uint32_t value,
              std::uint16_t n_bits)
{
  auto const byte_offset = bit_offset / 8;
  auto const shift       = static_cast<std::uint32_t>(bit_offset % 8);
  auto const bytes       = static_cast<std::size_t>((shift + n_bits + 7) / 8);
  auto current           = std::uint64_t{0};
  std::memcpy(&current, destination + byte_offset, bytes);
  auto const mask = n_bits == 32 ? std::uint64_t{0xffffffffu} : ((std::uint64_t{1} << n_bits) - 1);
  current &= ~(mask << shift);
  current |= (static_cast<std::uint64_t>(value) & mask) << shift;
  std::memcpy(destination + byte_offset, &current, bytes);
}

auto get_bits(std::uint8_t const* source, std::uint64_t bit_offset, std::uint16_t n_bits)
  -> std::uint32_t
{
  auto const byte_offset = bit_offset / 8;
  auto const shift       = static_cast<std::uint32_t>(bit_offset % 8);
  auto const bytes       = static_cast<std::size_t>((shift + n_bits + 7) / 8);
  auto current           = std::uint64_t{0};
  std::memcpy(&current, source + byte_offset, bytes);
  auto const mask = n_bits == 32 ? std::uint64_t{0xffffffffu} : ((std::uint64_t{1} << n_bits) - 1);
  return static_cast<std::uint32_t>((current >> shift) & mask);
}

void assign_local_quotas(std::array<std::uint16_t, max_nodes_per_packed_row>& assigned,
                         std::array<std::uint16_t, max_nodes_per_packed_row> const& available,
                         std::uint16_t group_size,
                         std::uint32_t row_capacity,
                         std::uint16_t node_per_cacheline)
{
  auto remaining  = row_capacity;
  auto const base = row_capacity / node_per_cacheline;
  for (std::uint16_t i = 0; i < group_size; ++i) {
    assigned[i] = static_cast<std::uint16_t>(std::min<std::uint32_t>(base, available[i]));
    remaining -= assigned[i];
  }

  while (remaining > 0) {
    auto selected = group_size;
    for (std::uint16_t i = 0; i < group_size; ++i) {
      if (assigned[i] >= available[i]) { continue; }
      if (selected == group_size || assigned[i] < assigned[selected]) { selected = i; }
    }
    if (selected == group_size) { break; }
    ++assigned[selected];
    --remaining;
  }
}

__global__ void make_group_sort_keys_kernel(std::uint64_t* keys,
                                            graph_index_type* values,
                                            std::uint8_t* local_counts,
                                            std::uint64_t* total_local,
                                            graph_index_type const* graph,
                                            std::uint32_t const* labels,
                                            std::uint64_t n_rows,
                                            std::uint32_t graph_degree)
{
  auto node   = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (; node < n_rows; node += stride) {
    auto const label = labels[node];
    auto count       = std::uint32_t{0};
    auto const* row  = graph + node * graph_degree;
    for (std::uint32_t edge = 0; edge < graph_degree; ++edge) {
      auto const neighbor = row[edge];
      if (neighbor < n_rows) { count += labels[neighbor] == label; }
    }
    local_counts[node] = static_cast<std::uint8_t>(count);
    atomicAdd(reinterpret_cast<unsigned long long*>(total_local),
              static_cast<unsigned long long>(count));
    keys[node] = (static_cast<std::uint64_t>(label) << 40) |
                 (static_cast<std::uint64_t>(std::uint8_t{255} - count) << 32) | node;
    values[node] = static_cast<graph_index_type>(node);
  }
}

__global__ void make_interleaved_permutation_kernel(graph_index_type* new_to_old,
                                                    graph_index_type* old_to_new,
                                                    std::uint64_t const* sorted_keys,
                                                    graph_index_type const* sorted_nodes,
                                                    std::uint32_t const* group_offsets,
                                                    std::uint32_t const* group_sizes,
                                                    std::uint64_t n_rows,
                                                    std::uint16_t node_per_cacheline)
{
  auto new_node = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto stride   = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (; new_node < n_rows; new_node += stride) {
    auto const group  = static_cast<std::uint32_t>(sorted_keys[new_node] >> 40);
    auto const begin  = static_cast<std::uint64_t>(group_offsets[group]);
    auto const size   = static_cast<std::uint64_t>(group_sizes[group]);
    auto const local  = new_node - begin;
    auto source_local = (local & 1u) == 0 ? local / 2 : size - 1 - local / 2;

    // Preserve the reference rearrangement's group-boundary alignment. If a group begins in the
    // middle of a packed row, rotate a few median-degree nodes to the front before alternating
    // high- and low-degree nodes. This avoids pairing two high-degree nodes at a group boundary.
    auto const misaligned = begin % node_per_cacheline;
    auto const middle     = size / 2;
    auto const rotated    = min(misaligned, size - middle);
    if (source_local < rotated) {
      source_local += middle;
    } else if (source_local < middle + rotated) {
      source_local -= rotated;
    }
    auto const old_node  = sorted_nodes[begin + source_local];
    new_to_old[new_node] = old_node;
    old_to_new[old_node] = static_cast<graph_index_type>(new_node);
  }
}

__global__ void pack_grouped_graph_kernel(std::uint8_t* inner_graph,
                                          graph_index_type* cross_graph,
                                          graph_index_type const* graph,
                                          std::uint8_t const* edge_order,
                                          std::uint32_t const* labels,
                                          std::uint8_t const* local_counts,
                                          graph_index_type const* new_to_old,
                                          graph_index_type const* old_to_new,
                                          std::uint32_t const* group_offsets,
                                          std::uint64_t n_rows,
                                          std::uint32_t graph_degree,
                                          std::uint16_t node_per_cacheline,
                                          std::uint16_t n_bits,
                                          std::uint32_t inner_row_bytes,
                                          std::uint32_t row_capacity)
{
  if (threadIdx.x != 0) { return; }
  auto const packed_row = static_cast<std::uint64_t>(blockIdx.x);
  auto const first_node = packed_row * node_per_cacheline;
  if (first_node >= n_rows) { return; }
  auto const group_size = static_cast<std::uint16_t>(
    min(static_cast<std::uint64_t>(node_per_cacheline), n_rows - first_node));
  std::uint16_t available[max_nodes_per_packed_row]{};
  std::uint16_t assigned[max_nodes_per_packed_row]{};
  for (std::uint16_t col = 0; col < group_size; ++col) {
    available[col] = local_counts[new_to_old[first_node + col]];
  }

  auto remaining  = row_capacity;
  auto const base = row_capacity / node_per_cacheline;
  for (std::uint16_t col = 0; col < group_size; ++col) {
    assigned[col] =
      static_cast<std::uint16_t>(min(base, static_cast<std::uint32_t>(available[col])));
    remaining -= assigned[col];
  }
  while (remaining > 0) {
    auto selected = group_size;
    for (std::uint16_t col = 0; col < group_size; ++col) {
      if (assigned[col] >= available[col]) { continue; }
      if (selected == group_size || assigned[col] < assigned[selected]) { selected = col; }
    }
    if (selected == group_size) { break; }
    ++assigned[selected];
    --remaining;
  }

  auto* inner_row = inner_graph == nullptr ? nullptr : inner_graph + packed_row * inner_row_bytes;
  auto bit_offset = static_cast<std::uint64_t>(node_per_cacheline) * 8;
  for (std::uint16_t col = 0; col < group_size; ++col) {
    auto const new_node = first_node + col;
    auto const old_node = new_to_old[new_node];
    auto const group    = labels[old_node];
    auto const* old_row = graph + static_cast<std::uint64_t>(old_node) * graph_degree;
    auto* cross_row     = cross_graph + new_node * (static_cast<std::uint64_t>(graph_degree) + 1);
    auto local_position = std::uint32_t{0};
    auto cross_position = std::uint32_t{0};
    if (inner_row != nullptr) { inner_row[col] = static_cast<std::uint8_t>(assigned[col]); }

    for (std::uint32_t edge = 0; edge < graph_degree; ++edge) {
      auto const old_neighbor = old_row[edge];
      auto const new_neighbor = old_to_new[old_neighbor];
      auto const local_rank =
        edge_order[static_cast<std::uint64_t>(old_node) * graph_degree + edge];
      if (labels[old_neighbor] == group && local_rank < assigned[col]) {
        auto const local_offset = new_neighbor - group_offsets[group];
        set_packed_bits(inner_row,
                        bit_offset + static_cast<std::uint64_t>(local_rank) * n_bits,
                        local_offset,
                        n_bits);
        ++local_position;
      } else {
        cross_row[++cross_position] = new_neighbor;
      }
    }
    cross_row[0] = cross_position;
    for (auto edge = cross_position + 1; edge <= graph_degree; ++edge) {
      cross_row[edge] = 0;
    }
    bit_offset += static_cast<std::uint64_t>(assigned[col]) * n_bits;
  }
}

void verify_conversion(
  host_graph_view graph,
  std::uint8_t const* edge_order,
  raft::host_vector_view<const std::uint32_t, int64_t> partition_labels,
  build_params const& params,
  raft::host_matrix_view<const std::uint8_t, int64_t, raft::row_major> inner_graph,
  raft::host_matrix_view<const graph_index_type, int64_t, raft::row_major> cross_graph,
  std::vector<graph_index_type> const& old_to_new,
  std::vector<graph_index_type> const& new_to_old,
  std::vector<std::uint32_t> const& subgraph_offsets)
{
  auto errors = std::uint64_t{0};

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : errors)
#endif
  for (int64_t new_node = 0; new_node < graph.extent(0); ++new_node) {
    auto const packed_row = new_node / params.node_per_cacheline;
    auto const packed_col = new_node % params.node_per_cacheline;
    auto const* inner_row =
      inner_graph.size() == 0
        ? nullptr
        : inner_graph.data_handle() + static_cast<std::size_t>(packed_row) * inner_graph.extent(1);
    auto const inner_count =
      inner_row == nullptr ? 0u : static_cast<std::uint32_t>(inner_row[packed_col]);
    auto inner_bit_offset = static_cast<std::uint64_t>(params.node_per_cacheline) * 8;
    for (int64_t col = 0; inner_row != nullptr && col < packed_col; ++col) {
      inner_bit_offset += static_cast<std::uint64_t>(inner_row[col]) * params.n_bits;
    }

    auto const old_node  = new_to_old[static_cast<std::size_t>(new_node)];
    auto const partition = partition_labels(old_node);
    auto const* cross_row =
      cross_graph.data_handle() + static_cast<std::size_t>(new_node) * cross_graph.extent(1);
    auto const cross_count = cross_row[0];
    auto local_position    = std::uint32_t{0};
    auto cross_position    = std::uint32_t{0};

    for (int64_t edge = 0; edge < graph.extent(1); ++edge) {
      auto const old_neighbor = graph(old_node, edge);
      auto const new_neighbor = old_to_new[old_neighbor];
      auto local_rank         = std::uint32_t{255};
      if (partition_labels(old_neighbor) == partition) {
        if (edge_order != nullptr) {
          local_rank = edge_order[static_cast<std::size_t>(old_node) * graph.extent(1) + edge];
        } else {
          local_rank = 0;
          for (int64_t candidate = 0; candidate < edge; ++candidate) {
            local_rank += partition_labels(graph(old_node, candidate)) == partition;
          }
        }
      }
      if (partition_labels(old_neighbor) == partition && local_rank < inner_count) {
        auto const local_offset =
          get_bits(inner_row,
                   inner_bit_offset + static_cast<std::uint64_t>(local_rank) * params.n_bits,
                   params.n_bits);
        auto const decoded = static_cast<std::uint64_t>(subgraph_offsets[partition]) + local_offset;
        errors += decoded != new_neighbor;
        ++local_position;
      } else {
        errors += cross_position >= cross_count || cross_row[cross_position + 1] != new_neighbor;
        ++cross_position;
      }
    }
    errors += local_position != inner_count || cross_position != cross_count;
  }

  RAFT_EXPECTS(errors == 0,
               "FlowANN split verification found %llu mismatched rows or edges",
               static_cast<unsigned long long>(errors));
}

auto convert_graph(host_graph_view graph,
                   raft::host_vector_view<const std::uint32_t, int64_t> partition_labels,
                   build_params const& params) -> conversion_result
{
  auto const n_rows = graph.extent(0);
  RAFT_EXPECTS(graph.extent(1) > 0 && static_cast<std::uint64_t>(graph.extent(1)) <=
                                        std::numeric_limits<std::uint8_t>::max(),
               "FlowANN graph degree must be in [1, 255]");
  auto const graph_degree = static_cast<std::uint32_t>(graph.extent(1));
  validate_build_params(params, graph_degree);
  RAFT_EXPECTS(n_rows > 0 &&
                 static_cast<std::uint64_t>(n_rows) <= std::numeric_limits<graph_index_type>::max(),
               "FlowANN graph size must fit uint32");
  RAFT_EXPECTS(partition_labels.extent(0) == n_rows,
               "FlowANN requires one partition label per CAGRA graph row");

  auto partition_sizes             = validate_and_count_partitions(partition_labels, params.n_bits);
  auto [local_counts, total_local] = count_local_neighbors(graph, partition_labels);
  auto [old_to_new, new_to_old] =
    make_permutation(local_counts, partition_labels, partition_sizes, params.node_per_cacheline);

  auto subgraph_offsets = std::vector<std::uint32_t>(partition_sizes.size());
  auto offset           = std::uint64_t{0};
  for (std::size_t partition = 0; partition < partition_sizes.size(); ++partition) {
    subgraph_offsets[partition] = static_cast<std::uint32_t>(offset);
    offset += partition_sizes[partition];
  }
  RAFT_EXPECTS(offset == static_cast<std::uint64_t>(n_rows),
               "FlowANN partition sizes do not cover the graph");

  auto subgraph_ids = std::vector<std::uint32_t>(static_cast<std::size_t>(n_rows));
  for (int64_t new_node = 0; new_node < n_rows; ++new_node) {
    subgraph_ids[static_cast<std::size_t>(new_node)] = partition_labels(new_to_old[new_node]);
  }

  auto const inner_row_bytes =
    compute_inner_row_bytes(params, static_cast<std::uint64_t>(n_rows), total_local);
  auto const header_bits = static_cast<std::uint64_t>(params.node_per_cacheline) * 8;
  RAFT_EXPECTS(static_cast<std::uint64_t>(inner_row_bytes) * 8 >= header_bits,
               "FlowANN inner graph row cannot hold its count header");
  auto const encoded_capacity =
    (static_cast<std::uint64_t>(inner_row_bytes) * 8 - header_bits) / params.n_bits;
  auto const row_capacity = static_cast<std::uint32_t>(std::min<std::uint64_t>(
    encoded_capacity, static_cast<std::uint64_t>(params.node_per_cacheline) * graph_degree));

  auto const packed_rows = raft::div_rounding_up_safe<int64_t>(n_rows, params.node_per_cacheline);
  auto inner_graph = raft::make_host_matrix<std::uint8_t, int64_t>(packed_rows, inner_row_bytes);
  auto cross_graph = raft::make_host_matrix<graph_index_type, int64_t>(n_rows, graph_degree + 1);
  std::fill_n(inner_graph.data_handle(), inner_graph.size(), std::uint8_t{0});

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t packed_row = 0; packed_row < packed_rows; ++packed_row) {
    auto const first_node = packed_row * params.node_per_cacheline;
    auto const group_size =
      static_cast<std::uint16_t>(std::min<int64_t>(params.node_per_cacheline, n_rows - first_node));
    auto available = std::array<std::uint16_t, max_nodes_per_packed_row>{};
    auto assigned  = std::array<std::uint16_t, max_nodes_per_packed_row>{};
    for (std::uint16_t col = 0; col < group_size; ++col) {
      available[col] = local_counts[new_to_old[static_cast<std::size_t>(first_node + col)]];
    }
    assign_local_quotas(assigned, available, group_size, row_capacity, params.node_per_cacheline);

    auto* inner_row =
      inner_graph.data_handle() + static_cast<std::size_t>(packed_row) * inner_graph.extent(1);
    auto bit_offset = header_bits;
    for (std::uint16_t col = 0; col < group_size; ++col) {
      auto const new_node  = first_node + col;
      auto const old_node  = new_to_old[static_cast<std::size_t>(new_node)];
      auto const partition = partition_labels(old_node);
      auto* cross_row =
        cross_graph.data_handle() + static_cast<std::size_t>(new_node) * cross_graph.extent(1);
      auto local_position = std::uint32_t{0};
      auto cross_position = std::uint32_t{0};

      inner_row[col] = static_cast<std::uint8_t>(assigned[col]);
      for (std::uint32_t edge = 0; edge < graph_degree; ++edge) {
        auto const old_neighbor = graph(old_node, edge);
        auto const new_neighbor = old_to_new[old_neighbor];
        if (partition_labels(old_neighbor) == partition && local_position < assigned[col]) {
          auto const local_offset =
            static_cast<std::uint64_t>(new_neighbor) - subgraph_offsets[partition];
          set_bits(inner_row,
                   bit_offset + static_cast<std::uint64_t>(local_position) * params.n_bits,
                   static_cast<std::uint32_t>(local_offset),
                   params.n_bits);
          ++local_position;
        } else {
          cross_row[++cross_position] = new_neighbor;
        }
      }
      cross_row[0] = cross_position;
      std::fill(cross_row + cross_position + 1,
                cross_row + static_cast<std::size_t>(graph_degree) + 1,
                graph_index_type{0});
      bit_offset += static_cast<std::uint64_t>(assigned[col]) * params.n_bits;
    }
  }

  conversion_result result{std::move(inner_graph),
                           std::move(cross_graph),
                           std::move(old_to_new),
                           std::move(new_to_old),
                           std::move(subgraph_offsets),
                           std::move(subgraph_ids)};
  if (params.validate) {
    verify_conversion(graph,
                      nullptr,
                      partition_labels,
                      params,
                      raft::make_const_mdspan(result.inner_graph.view()),
                      raft::make_const_mdspan(result.cross_graph.view()),
                      result.old_to_new,
                      result.new_to_old,
                      result.subgraph_offsets);
  }
  return result;
}

auto convert_graph_on_device(raft::resources const& res,
                             host_graph_view graph,
                             host_edge_order_view edge_order,
                             raft::host_vector_view<const std::uint32_t, int64_t> partition_labels,
                             build_params const& params,
                             std::size_t device_graph_budget_bytes) -> device_conversion_result
{
  auto const n_rows = static_cast<std::uint64_t>(graph.extent(0));
  RAFT_EXPECTS(n_rows > 0 && n_rows <= std::numeric_limits<graph_index_type>::max(),
               "FlowANN graph size must fit uint32");
  RAFT_EXPECTS(graph.extent(1) > 0 && static_cast<std::uint64_t>(graph.extent(1)) <=
                                        std::numeric_limits<std::uint8_t>::max(),
               "FlowANN graph degree must be in [1, 255]");
  auto const graph_degree = static_cast<std::uint32_t>(graph.extent(1));
  validate_build_params(params, graph_degree);
  RAFT_EXPECTS(partition_labels.extent(0) == graph.extent(0),
               "FlowANN requires one partition label per graph row");
  RAFT_EXPECTS(edge_order.extent(0) == graph.extent(0) && edge_order.extent(1) == graph.extent(1),
               "FlowANN edge-order matrix must match the graph shape");

  auto const partition_sizes = validate_and_count_partitions(partition_labels, params.n_bits);
  auto subgraph_offsets      = std::vector<std::uint32_t>(partition_sizes.size());
  auto group_sizes           = std::vector<std::uint32_t>(partition_sizes.size());
  auto offset                = std::uint64_t{0};
  for (std::size_t group = 0; group < partition_sizes.size(); ++group) {
    subgraph_offsets[group] = static_cast<std::uint32_t>(offset);
    group_sizes[group]      = static_cast<std::uint32_t>(partition_sizes[group]);
    offset += partition_sizes[group];
  }
  RAFT_EXPECTS(offset == n_rows, "FlowANN partition sizes do not cover the graph");

  auto const stream     = raft::resource::get_cuda_stream(res);
  auto const raw_stream = cuda_stream_handle(stream);
  auto device_graph =
    raft::make_device_matrix<graph_index_type, int64_t>(res, graph.extent(0), graph.extent(1));
  auto device_edge_order =
    raft::make_device_matrix<std::uint8_t, int64_t>(res, graph.extent(0), graph.extent(1));
  auto device_labels = raft::make_device_vector<std::uint32_t, int64_t>(res, graph.extent(0));
  raft::copy(device_graph.data_handle(), graph.data_handle(), graph.size(), stream);
  raft::copy(device_edge_order.data_handle(), edge_order.data_handle(), edge_order.size(), stream);
  raft::copy(device_labels.data_handle(),
             partition_labels.data_handle(),
             partition_labels.extent(0),
             stream);

  auto keys_in      = raft::make_device_vector<std::uint64_t, int64_t>(res, graph.extent(0));
  auto keys_out     = raft::make_device_vector<std::uint64_t, int64_t>(res, graph.extent(0));
  auto nodes_in     = raft::make_device_vector<graph_index_type, int64_t>(res, graph.extent(0));
  auto nodes_sorted = raft::make_device_vector<graph_index_type, int64_t>(res, graph.extent(0));
  auto local_counts = raft::make_device_vector<std::uint8_t, int64_t>(res, graph.extent(0));
  auto total_local  = raft::make_device_vector<std::uint64_t, int64_t>(res, 1);
  RAFT_CUDA_TRY(cudaMemsetAsync(total_local.data_handle(), 0, sizeof(std::uint64_t), raw_stream));
  constexpr auto threads = std::uint32_t{256};
  auto const blocks =
    static_cast<std::uint32_t>(std::min<std::uint64_t>((n_rows + threads - 1) / threads, 65535));
  make_group_sort_keys_kernel<<<blocks, threads, 0, raw_stream>>>(keys_in.data_handle(),
                                                                  nodes_in.data_handle(),
                                                                  local_counts.data_handle(),
                                                                  total_local.data_handle(),
                                                                  device_graph.data_handle(),
                                                                  device_labels.data_handle(),
                                                                  n_rows,
                                                                  graph_degree);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  auto sort_workspace_bytes = std::size_t{0};
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                sort_workspace_bytes,
                                                keys_in.data_handle(),
                                                keys_out.data_handle(),
                                                nodes_in.data_handle(),
                                                nodes_sorted.data_handle(),
                                                graph.extent(0),
                                                0,
                                                56,
                                                raw_stream));
  rmm::device_buffer sort_workspace(sort_workspace_bytes, stream);
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(sort_workspace.data(),
                                                sort_workspace_bytes,
                                                keys_in.data_handle(),
                                                keys_out.data_handle(),
                                                nodes_in.data_handle(),
                                                nodes_sorted.data_handle(),
                                                graph.extent(0),
                                                0,
                                                56,
                                                raw_stream));

  auto device_offsets =
    raft::make_device_vector<std::uint32_t, int64_t>(res, subgraph_offsets.size());
  auto device_sizes = raft::make_device_vector<std::uint32_t, int64_t>(res, group_sizes.size());
  auto device_new_to_old =
    raft::make_device_vector<graph_index_type, int64_t>(res, graph.extent(0));
  auto device_old_to_new =
    raft::make_device_vector<graph_index_type, int64_t>(res, graph.extent(0));
  raft::copy(
    device_offsets.data_handle(), subgraph_offsets.data(), subgraph_offsets.size(), stream);
  raft::copy(device_sizes.data_handle(), group_sizes.data(), group_sizes.size(), stream);
  make_interleaved_permutation_kernel<<<blocks, threads, 0, raw_stream>>>(
    device_new_to_old.data_handle(),
    device_old_to_new.data_handle(),
    keys_out.data_handle(),
    nodes_sorted.data_handle(),
    device_offsets.data_handle(),
    device_sizes.data_handle(),
    n_rows,
    params.node_per_cacheline);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  auto old_to_new       = std::vector<graph_index_type>(n_rows);
  auto new_to_old       = std::vector<graph_index_type>(n_rows);
  auto total_local_host = std::uint64_t{0};
  raft::copy(old_to_new.data(), device_old_to_new.data_handle(), n_rows, stream);
  raft::copy(new_to_old.data(), device_new_to_old.data_handle(), n_rows, stream);
  raft::copy(&total_local_host, total_local.data_handle(), 1, stream);
  raft::resource::sync_stream(res);

  auto const packed_rows = raft::div_rounding_up_safe<std::uint64_t>(
    n_rows, static_cast<std::uint64_t>(params.node_per_cacheline));
  auto inner_row_bytes = budgeted_grouped_row_bytes(
    n_rows, graph_degree, params.node_per_cacheline, params.n_bits, device_graph_budget_bytes);
  if (params.inner_graph_row_bytes != 0) {
    inner_row_bytes = std::min(inner_row_bytes, params.inner_graph_row_bytes);
  }
  auto const header_bits = static_cast<std::uint64_t>(params.node_per_cacheline) * 8;
  auto row_capacity      = std::uint32_t{0};
  if (inner_row_bytes >= params.node_per_cacheline) {
    row_capacity = static_cast<std::uint32_t>(std::min<std::uint64_t>(
      (static_cast<std::uint64_t>(inner_row_bytes) * 8 - header_bits) / params.n_bits,
      static_cast<std::uint64_t>(params.node_per_cacheline) * graph_degree));
  }
  if (row_capacity == 0) { inner_row_bytes = 0; }

  auto inner_graph  = inner_row_bytes == 0
                        ? raft::make_device_matrix<std::uint8_t, int64_t>(res, 0, 0)
                        : raft::make_device_matrix<std::uint8_t, int64_t>(
                           res, static_cast<int64_t>(packed_rows), inner_row_bytes);
  auto device_cross = raft::make_device_matrix<graph_index_type, int64_t>(
    res, graph.extent(0), static_cast<int64_t>(graph_degree) + 1);
  if (inner_graph.size() > 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(
      inner_graph.data_handle(), 0, inner_graph.size() * sizeof(std::uint8_t), raw_stream));
  }
  RAFT_EXPECTS(packed_rows <= std::numeric_limits<std::uint32_t>::max(),
               "FlowANN packed row count exceeds the CUDA grid limit");
  pack_grouped_graph_kernel<<<static_cast<std::uint32_t>(packed_rows), 32, 0, raw_stream>>>(
    inner_graph.size() == 0 ? nullptr : inner_graph.data_handle(),
    device_cross.data_handle(),
    device_graph.data_handle(),
    device_edge_order.data_handle(),
    device_labels.data_handle(),
    local_counts.data_handle(),
    device_new_to_old.data_handle(),
    device_old_to_new.data_handle(),
    device_offsets.data_handle(),
    n_rows,
    graph_degree,
    params.node_per_cacheline,
    params.n_bits,
    inner_row_bytes,
    row_capacity);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  auto cross_graph = raft::make_host_matrix<graph_index_type, int64_t>(
    graph.extent(0), static_cast<int64_t>(graph_degree) + 1);
  raft::copy(cross_graph.data_handle(), device_cross.data_handle(), device_cross.size(), stream);
  raft::resource::sync_stream(res);

  auto subgraph_ids = std::vector<std::uint32_t>(n_rows);
  for (std::uint64_t new_node = 0; new_node < n_rows; ++new_node) {
    subgraph_ids[new_node] = partition_labels(new_to_old[new_node]);
  }

  if (params.validate) {
    auto host_inner =
      raft::make_host_matrix<std::uint8_t, int64_t>(inner_graph.extent(0), inner_graph.extent(1));
    if (inner_graph.size() > 0) {
      raft::copy(host_inner.data_handle(), inner_graph.data_handle(), inner_graph.size(), stream);
      raft::resource::sync_stream(res);
    }
    verify_conversion(graph,
                      edge_order.data_handle(),
                      partition_labels,
                      params,
                      raft::make_const_mdspan(host_inner.view()),
                      raft::make_const_mdspan(cross_graph.view()),
                      old_to_new,
                      new_to_old,
                      subgraph_offsets);
  }

  RAFT_LOG_INFO("FlowANN grouped layout: groups=%zu local_edges=%llu row_bytes=%u budget_bytes=%zu",
                partition_sizes.size(),
                static_cast<unsigned long long>(total_local_host),
                inner_row_bytes,
                device_graph_budget_bytes);
  return {std::move(inner_graph),
          std::move(cross_graph),
          std::move(old_to_new),
          std::move(new_to_old),
          std::move(subgraph_offsets),
          std::move(subgraph_ids),
          params.validate};
}

template <typename T>
__global__ void gather_rows_kernel(T* destination,
                                   T const* source,
                                   graph_index_type const* new_to_old,
                                   std::uint64_t elements,
                                   std::uint32_t row_length)
{
  auto offset = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (; offset < elements; offset += stride) {
    auto const new_row = offset / row_length;
    auto const column  = offset % row_length;
    destination[offset] =
      source[static_cast<std::uint64_t>(new_to_old[new_row]) * row_length + column];
  }
}

template <typename T>
void gather_device_rows(raft::resources const& res,
                        T* destination,
                        T const* source,
                        std::uint64_t n_rows,
                        std::uint32_t row_length,
                        std::vector<graph_index_type> const& new_to_old)
{
  auto device_mapping =
    raft::make_device_vector<graph_index_type, int64_t>(res, static_cast<int64_t>(n_rows));
  auto const stream     = raft::resource::get_cuda_stream(res);
  auto const raw_stream = cuda_stream_handle(stream);
  raft::copy(device_mapping.data_handle(), new_to_old.data(), n_rows, stream);

  auto const elements       = n_rows * row_length;
  constexpr auto block_size = std::uint32_t{256};
  auto const grid_size      = static_cast<std::uint32_t>(
    std::min<std::uint64_t>((elements + block_size - 1) / block_size, 65535));
  gather_rows_kernel<<<grid_size, block_size, 0, raw_stream>>>(
    destination, source, device_mapping.data_handle(), elements, row_length);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::resource::sync_stream(res);
}

template <typename T>
auto reorder_dense_dataset(raft::resources const& res,
                           cuvs::neighbors::device_padded_dataset_view<T, int64_t> const& source,
                           std::vector<graph_index_type> const& new_to_old)
  -> std::unique_ptr<cuvs::neighbors::device_padded_dataset<T, int64_t>>
{
  auto const source_data = source.view();
  auto reordered =
    raft::make_device_matrix<T, int64_t>(res, source_data.extent(0), source_data.extent(1));
  gather_device_rows(res,
                     reordered.data_handle(),
                     source_data.data_handle(),
                     source_data.extent(0),
                     static_cast<std::uint32_t>(source_data.extent(1)),
                     new_to_old);
  return std::make_unique<cuvs::neighbors::device_padded_dataset<T, int64_t>>(std::move(reordered),
                                                                              source.dim());
}

auto reorder_vpq_dataset(raft::resources const& res,
                         cuvs::neighbors::device_vpq_dataset_view<half, int64_t> const& source,
                         std::vector<graph_index_type> const& new_to_old)
  -> std::unique_ptr<cuvs::neighbors::device_vpq_dataset<half, int64_t>>
{
  auto const& source_data = source.dset();
  auto vq_code_book       = raft::make_device_matrix<half, std::uint32_t>(
    res, source_data.vq_code_book.extent(0), source_data.vq_code_book.extent(1));
  auto pq_code_book = raft::make_device_matrix<half, std::uint32_t>(
    res, source_data.pq_code_book.extent(0), source_data.pq_code_book.extent(1));
  auto encoded = raft::make_device_matrix<std::uint8_t, int64_t>(
    res, source_data.data.extent(0), source_data.data.extent(1));
  auto const stream = raft::resource::get_cuda_stream(res);
  raft::copy(vq_code_book.data_handle(),
             source_data.vq_code_book.data_handle(),
             source_data.vq_code_book.size(),
             stream);
  raft::copy(pq_code_book.data_handle(),
             source_data.pq_code_book.data_handle(),
             source_data.pq_code_book.size(),
             stream);
  gather_device_rows(res,
                     encoded.data_handle(),
                     source_data.data.data_handle(),
                     source_data.data.extent(0),
                     static_cast<std::uint32_t>(source_data.data.extent(1)),
                     new_to_old);
  return std::make_unique<cuvs::neighbors::device_vpq_dataset<half, int64_t>>(
    std::move(vq_code_book), std::move(pq_code_book), std::move(encoded));
}

auto reorder_vpq_dataset_consuming(
  raft::resources const& res,
  std::unique_ptr<cuvs::neighbors::device_vpq_dataset<half, int64_t>> source,
  std::vector<graph_index_type> const& new_to_old)
  -> std::unique_ptr<cuvs::neighbors::device_vpq_dataset<half, int64_t>>
{
  RAFT_EXPECTS(source != nullptr, "FlowANN VPQ conversion requires a dataset owner");
  auto const n_rows     = source->data.extent(0);
  auto const row_length = source->data.extent(1);
  RAFT_EXPECTS(n_rows == static_cast<int64_t>(new_to_old.size()),
               "FlowANN VPQ dataset and graph row counts differ");

  auto host_source  = raft::make_host_matrix<std::uint8_t, int64_t>(n_rows, row_length);
  auto const stream = raft::resource::get_cuda_stream(res);
  raft::copy(host_source.data_handle(), source->data.data_handle(), source->data.size(), stream);
  raft::resource::sync_stream(res);

  auto vq_code_book = std::move(source->vq_code_book);
  auto pq_code_book = std::move(source->pq_code_book);
  source.reset();

  auto host_reordered = raft::make_host_matrix<std::uint8_t, int64_t>(n_rows, row_length);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t new_row = 0; new_row < n_rows; ++new_row) {
    std::copy_n(
      host_source.data_handle() +
        static_cast<std::size_t>(new_to_old[static_cast<std::size_t>(new_row)]) * row_length,
      row_length,
      host_reordered.data_handle() + static_cast<std::size_t>(new_row) * row_length);
  }
  host_source = raft::make_host_matrix<std::uint8_t, int64_t>(0, 0);

  auto encoded = raft::make_device_matrix<std::uint8_t, int64_t>(res, n_rows, row_length);
  raft::copy(encoded.data_handle(), host_reordered.data_handle(), host_reordered.size(), stream);
  raft::resource::sync_stream(res);
  return std::make_unique<cuvs::neighbors::device_vpq_dataset<half, int64_t>>(
    std::move(vq_code_book), std::move(pq_code_book), std::move(encoded));
}

template <typename CagraIndexT>
auto make_source_indices(raft::resources const& res,
                         CagraIndexT const& source,
                         std::vector<graph_index_type> const& new_to_old)
  -> std::vector<graph_index_type>
{
  auto result = std::vector<graph_index_type>(new_to_old.size());
  if (source.source_indices().has_value()) {
    auto const source_view = source.source_indices().value();
    RAFT_EXPECTS(source_view.extent(0) == static_cast<int64_t>(new_to_old.size()),
                 "FlowANN source-index mapping and graph row counts differ");
    auto source_ids = std::vector<graph_index_type>(new_to_old.size());
    raft::copy(source_ids.data(),
               source_view.data_handle(),
               source_view.extent(0),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    for (std::size_t new_node = 0; new_node < new_to_old.size(); ++new_node) {
      result[new_node] = source_ids[new_to_old[new_node]];
    }
  } else {
    result = new_to_old;
  }
  return result;
}

template <typename FlowIndexT>
void populate_index(raft::resources const& res,
                    FlowIndexT& index,
                    conversion_result&& conversion,
                    std::vector<graph_index_type> const& source_indices,
                    std::optional<raft::host_vector_view<const graph_index_type, int64_t>> seeds)
{
  auto const n_rows = conversion.old_to_new.size();
  index.update_graph(res, std::move(conversion.inner_graph), std::move(conversion.cross_graph));
  index.update_source_indices(
    res,
    raft::make_host_vector_view<const graph_index_type, int64_t>(source_indices.data(), n_rows));
  index.update_subgraph_layout(
    res,
    raft::make_host_vector_view<const std::uint32_t, int64_t>(conversion.subgraph_offsets.data(),
                                                              conversion.subgraph_offsets.size()),
    raft::make_host_vector_view<const std::uint32_t, int64_t>(conversion.subgraph_ids.data(),
                                                              conversion.subgraph_ids.size()));

  if (seeds.has_value()) {
    RAFT_EXPECTS(seeds->extent(0) > 0, "FlowANN seed list must not be empty");
    auto remapped = std::vector<graph_index_type>(static_cast<std::size_t>(seeds->extent(0)));
    for (int64_t seed = 0; seed < seeds->extent(0); ++seed) {
      RAFT_EXPECTS((*seeds)(seed) < n_rows,
                   "FlowANN input seed at position %ld is out of range",
                   static_cast<long>(seed));
      remapped[static_cast<std::size_t>(seed)] = conversion.old_to_new[(*seeds)(seed)];
    }
    index.update_seeds(res,
                       raft::make_host_vector_view<const graph_index_type, int64_t>(
                         remapped.data(), remapped.size()));
  }
  raft::resource::sync_stream(res);
}

template <typename FlowIndexT>
void populate_index(raft::resources const& res,
                    FlowIndexT& index,
                    device_conversion_result&& conversion,
                    std::vector<graph_index_type> const& source_indices,
                    std::optional<raft::host_vector_view<const graph_index_type, int64_t>> seeds)
{
  auto const n_rows    = conversion.old_to_new.size();
  auto const validated = conversion.validated;
  index.update_graph(
    res, std::move(conversion.inner_graph), std::move(conversion.cross_graph), validated);
  index.update_source_indices(
    res,
    raft::make_host_vector_view<const graph_index_type, int64_t>(source_indices.data(), n_rows));
  index.update_subgraph_layout(
    res,
    raft::make_host_vector_view<const std::uint32_t, int64_t>(conversion.subgraph_offsets.data(),
                                                              conversion.subgraph_offsets.size()),
    raft::make_host_vector_view<const std::uint32_t, int64_t>(conversion.subgraph_ids.data(),
                                                              conversion.subgraph_ids.size()));

  if (seeds.has_value()) {
    auto remapped = std::vector<graph_index_type>(static_cast<std::size_t>(seeds->extent(0)));
    for (int64_t seed = 0; seed < seeds->extent(0); ++seed) {
      RAFT_EXPECTS((*seeds)(seed) < n_rows,
                   "FlowANN input seed at position %ld is out of range",
                   static_cast<long>(seed));
      remapped[static_cast<std::size_t>(seed)] = conversion.old_to_new[(*seeds)(seed)];
    }
    index.update_seeds(res,
                       raft::make_host_vector_view<const graph_index_type, int64_t>(
                         remapped.data(), remapped.size()));
  }
  raft::resource::sync_stream(res);
}

template <typename FlowIndexT>
void populate_uniform_index(raft::resources const& res,
                            FlowIndexT& index,
                            uniform_conversion_result&& conversion)
{
  auto const n_rows = conversion.cross_graph.extent(0);
  auto const layout = conversion.layout;
  index.update_graph(res,
                     std::move(conversion.inner_graph),
                     std::move(conversion.cross_graph),
                     layout.resident_degree + layout.cross_degree,
                     layout.resident_degree);

  auto offsets      = raft::make_host_vector<std::uint32_t, int64_t>(1);
  offsets(0)        = 0;
  auto subgraph_ids = raft::make_host_vector<std::uint32_t, int64_t>(n_rows);
  std::fill_n(subgraph_ids.data_handle(), subgraph_ids.size(), std::uint32_t{0});
  index.update_subgraph_layout(
    res, raft::make_const_mdspan(offsets.view()), raft::make_const_mdspan(subgraph_ids.view()));
  raft::resource::sync_stream(res);
}

inline void reject_ace(index_params const& params)
{
  RAFT_EXPECTS(!std::holds_alternative<cuvs::neighbors::cagra::graph_build_params::ace_params>(
                 params.cagra_params.graph_build_params),
               "FlowANN integrated rank/budget build does not support ACE graph construction");
}

template <typename CagraIndexT>
auto convert_cagra_graph(raft::resources const& res,
                         build_params const& params,
                         CagraIndexT const& source,
                         raft::host_vector_view<const std::uint32_t, int64_t> partition_labels)
  -> conversion_result
{
  RAFT_EXPECTS(source.graph().extent(0) > 0 && source.graph().extent(1) > 0,
               "FlowANN build requires an in-memory CAGRA graph");
  RAFT_EXPECTS(source.dataset().n_rows() == source.graph().extent(0),
               "FlowANN build requires an in-memory dataset matching the CAGRA graph");
  auto host_graph = copy_graph_to_host(res, source.graph());
  return convert_graph(raft::make_const_mdspan(host_graph.view()), partition_labels, params);
}

}  // namespace

}  // namespace cuvs::neighbors::cagra::experimental::flowann::detail

namespace cuvs::neighbors::cagra::experimental::flowann {

template <typename T>
auto build_dense_integrated(raft::resources const& res,
                            index_params const& params,
                            cuvs::neighbors::device_padded_dataset_view<T, int64_t> const& dataset)
  -> device_padded_index_bundle<T, std::uint32_t>
{
  detail::reject_ace(params);

  auto const build_begin = std::chrono::steady_clock::now();
  auto graph             = cuvs::neighbors::cagra::detail::build_graph<T, std::uint32_t>(
    res, params.cagra_params, dataset);
  auto const graph_end = std::chrono::steady_clock::now();
  if (params.grouping.enabled) {
    auto grouping =
      detail::generate_group_labels(res, params, raft::make_const_mdspan(graph.view()), dataset);
    auto& labels            = grouping.labels;
    auto const grouping_end = std::chrono::steady_clock::now();
    auto edge_order         = detail::rank_graph_edge_order(res,
                                                    params.cagra_params.metric,
                                                    raft::make_const_mdspan(graph.view()),
                                                    raft::make_const_mdspan(labels.view()),
                                                    dataset);
    auto const rank_end     = std::chrono::steady_clock::now();
    build_params grouped_params;
    grouped_params.node_per_cacheline = params.node_per_cacheline;
    grouped_params.n_bits             = grouping.n_bits;
    grouped_params.validate           = params.grouping.validate;
    auto conversion                   = detail::convert_graph_on_device(res,
                                                      raft::make_const_mdspan(graph.view()),
                                                      raft::make_const_mdspan(edge_order.view()),
                                                      raft::make_const_mdspan(labels.view()),
                                                      grouped_params,
                                                      params.device_graph_budget_bytes);
    auto const split_end              = std::chrono::steady_clock::now();
    graph                             = raft::make_host_matrix<std::uint32_t, int64_t>(0, 0);
    std::optional<raft::host_vector<std::uint32_t, int64_t>> seeds;
    if (params.num_seeds > 0) {
      seeds.emplace(detail::generate_medoid_seeds(res, params, dataset));
    }
    auto const seeds_end = std::chrono::steady_clock::now();

    auto source_indices = conversion.new_to_old;
    auto dataset_owner  = detail::reorder_dense_dataset(res, dataset, conversion.new_to_old);
    flowann::device_padded_index<T, std::uint32_t> index(res,
                                                         params.cagra_params.metric,
                                                         dataset_owner->as_dataset_view(),
                                                         params.node_per_cacheline,
                                                         grouping.n_bits);
    std::optional<raft::host_vector_view<const std::uint32_t, int64_t>> seed_view;
    if (seeds.has_value()) { seed_view.emplace(raft::make_const_mdspan(seeds->view())); }
    detail::populate_index(res, index, std::move(conversion), source_indices, seed_view);
    auto const finish = std::chrono::steady_clock::now();
    RAFT_LOG_INFO(
      "FlowANN build stages: cagra=%.3f s, grouping=%.3f s, rank=%.3f s, rearrange=%.3f s, "
      "seeds=%.3f s, dataset=%.3f s",
      std::chrono::duration<double>(graph_end - build_begin).count(),
      std::chrono::duration<double>(grouping_end - graph_end).count(),
      std::chrono::duration<double>(rank_end - grouping_end).count(),
      std::chrono::duration<double>(split_end - rank_end).count(),
      std::chrono::duration<double>(seeds_end - split_end).count(),
      std::chrono::duration<double>(finish - seeds_end).count());
    return {std::move(dataset_owner), std::move(index)};
  }
  auto conversion = detail::convert_uniform_ranked_graph(
    res, params, raft::make_const_mdspan(graph.view()), dataset);
  auto const split_end = std::chrono::steady_clock::now();
  graph                = raft::make_host_matrix<std::uint32_t, int64_t>(0, 0);
  std::optional<raft::host_vector<std::uint32_t, int64_t>> seeds;
  if (params.num_seeds > 0) { seeds.emplace(detail::generate_medoid_seeds(res, params, dataset)); }
  auto const seeds_end = std::chrono::steady_clock::now();

  auto source_rows = dataset.view();
  auto owned_rows =
    raft::make_device_matrix<T, int64_t>(res, source_rows.extent(0), source_rows.extent(1));
  raft::copy(owned_rows.data_handle(),
             source_rows.data_handle(),
             source_rows.size(),
             raft::resource::get_cuda_stream(res));
  auto dataset_owner = std::make_unique<cuvs::neighbors::device_padded_dataset<T, int64_t>>(
    std::move(owned_rows), dataset.dim());
  flowann::device_padded_index<T, std::uint32_t> index(res,
                                                       params.cagra_params.metric,
                                                       dataset_owner->as_dataset_view(),
                                                       params.node_per_cacheline,
                                                       conversion.layout.n_bits);
  detail::populate_uniform_index(res, index, std::move(conversion));
  if (seeds.has_value()) { index.update_seeds(res, raft::make_const_mdspan(seeds->view())); }
  auto const finish = std::chrono::steady_clock::now();
  RAFT_LOG_INFO(
    "FlowANN build stages: cagra=%.3f s, rank_split=%.3f s, seeds=%.3f s, dataset=%.3f s",
    std::chrono::duration<double>(graph_end - build_begin).count(),
    std::chrono::duration<double>(split_end - graph_end).count(),
    std::chrono::duration<double>(seeds_end - split_end).count(),
    std::chrono::duration<double>(finish - seeds_end).count());
  return {std::move(dataset_owner), std::move(index)};
}

template <typename T>
auto build_vpq_integrated(raft::resources const& res,
                          index_params const& params,
                          cuvs::neighbors::vpq_params const& vpq_params,
                          cuvs::neighbors::host_standard_dataset_view<T, int64_t> const& dataset)
  -> vpq_f16_index_bundle<T, std::uint32_t>
{
  detail::reject_ace(params);
  RAFT_EXPECTS(params.cagra_params.metric == cuvs::distance::DistanceType::L2Expanded,
               "FlowANN VPQ-F16 integrated build requires L2Expanded");

  auto const build_begin = std::chrono::steady_clock::now();
  auto graph             = cuvs::neighbors::cagra::detail::build_graph<T, std::uint32_t>(
    res, params.cagra_params, dataset);
  auto const graph_end = std::chrono::steady_clock::now();
  if (params.grouping.enabled) {
    auto grouping =
      detail::generate_group_labels(res, params, raft::make_const_mdspan(graph.view()), dataset);
    auto& labels            = grouping.labels;
    auto const grouping_end = std::chrono::steady_clock::now();
    auto edge_order         = detail::rank_graph_edge_order(res,
                                                    params.cagra_params.metric,
                                                    raft::make_const_mdspan(graph.view()),
                                                    raft::make_const_mdspan(labels.view()),
                                                    dataset);
    auto const rank_end     = std::chrono::steady_clock::now();
    build_params grouped_params;
    grouped_params.node_per_cacheline = params.node_per_cacheline;
    grouped_params.n_bits             = grouping.n_bits;
    grouped_params.validate           = params.grouping.validate;
    auto conversion                   = detail::convert_graph_on_device(res,
                                                      raft::make_const_mdspan(graph.view()),
                                                      raft::make_const_mdspan(edge_order.view()),
                                                      raft::make_const_mdspan(labels.view()),
                                                      grouped_params,
                                                      params.device_graph_budget_bytes);
    auto const split_end              = std::chrono::steady_clock::now();
    graph                             = raft::make_host_matrix<std::uint32_t, int64_t>(0, 0);
    std::optional<raft::host_vector<std::uint32_t, int64_t>> seeds;
    if (params.num_seeds > 0) {
      seeds.emplace(detail::generate_medoid_seeds(res, params, dataset));
    }
    auto const seeds_end = std::chrono::steady_clock::now();

    auto unordered_dataset = std::make_unique<cuvs::neighbors::device_vpq_dataset<half, int64_t>>(
      cuvs::preprocessing::quantize::pq::make_vpq_dataset(res, vpq_params, dataset));
    auto source_indices = conversion.new_to_old;
    auto dataset_owner =
      detail::reorder_vpq_dataset(res, unordered_dataset->as_dataset_view(), conversion.new_to_old);
    unordered_dataset.reset();
    flowann::vpq_f16_index<T, std::uint32_t> index(res,
                                                   params.cagra_params.metric,
                                                   dataset_owner->as_dataset_view(),
                                                   params.node_per_cacheline,
                                                   grouping.n_bits);
    std::optional<raft::host_vector_view<const std::uint32_t, int64_t>> seed_view;
    if (seeds.has_value()) { seed_view.emplace(raft::make_const_mdspan(seeds->view())); }
    detail::populate_index(res, index, std::move(conversion), source_indices, seed_view);
    auto const finish = std::chrono::steady_clock::now();
    RAFT_LOG_INFO(
      "FlowANN build stages: cagra=%.3f s, grouping=%.3f s, rank=%.3f s, rearrange=%.3f s, "
      "seeds=%.3f s, vpq=%.3f s",
      std::chrono::duration<double>(graph_end - build_begin).count(),
      std::chrono::duration<double>(grouping_end - graph_end).count(),
      std::chrono::duration<double>(rank_end - grouping_end).count(),
      std::chrono::duration<double>(split_end - rank_end).count(),
      std::chrono::duration<double>(seeds_end - split_end).count(),
      std::chrono::duration<double>(finish - seeds_end).count());
    return {std::move(dataset_owner), std::move(index)};
  }
  auto conversion = detail::convert_uniform_ranked_graph(
    res, params, raft::make_const_mdspan(graph.view()), dataset);
  auto const split_end = std::chrono::steady_clock::now();
  graph                = raft::make_host_matrix<std::uint32_t, int64_t>(0, 0);
  std::optional<raft::host_vector<std::uint32_t, int64_t>> seeds;
  if (params.num_seeds > 0) { seeds.emplace(detail::generate_medoid_seeds(res, params, dataset)); }
  auto const seeds_end = std::chrono::steady_clock::now();

  auto dataset_owner = std::make_unique<cuvs::neighbors::device_vpq_dataset<half, int64_t>>(
    cuvs::preprocessing::quantize::pq::make_vpq_dataset(res, vpq_params, dataset));
  flowann::vpq_f16_index<T, std::uint32_t> index(res,
                                                 params.cagra_params.metric,
                                                 dataset_owner->as_dataset_view(),
                                                 params.node_per_cacheline,
                                                 conversion.layout.n_bits);
  detail::populate_uniform_index(res, index, std::move(conversion));
  if (seeds.has_value()) { index.update_seeds(res, raft::make_const_mdspan(seeds->view())); }
  auto const finish = std::chrono::steady_clock::now();
  RAFT_LOG_INFO("FlowANN build stages: cagra=%.3f s, rank_split=%.3f s, seeds=%.3f s, vpq=%.3f s",
                std::chrono::duration<double>(graph_end - build_begin).count(),
                std::chrono::duration<double>(split_end - graph_end).count(),
                std::chrono::duration<double>(seeds_end - split_end).count(),
                std::chrono::duration<double>(finish - seeds_end).count());
  return {std::move(dataset_owner), std::move(index)};
}

#define CUVS_FLOWANN_DEFINE_INTEGRATED_BUILD(T)                                      \
  auto build(raft::resources const& res,                                             \
             index_params const& params,                                             \
             cuvs::neighbors::device_padded_dataset_view<T, int64_t> const& dataset) \
    -> device_padded_index_bundle<T, std::uint32_t>                                  \
  {                                                                                  \
    return build_dense_integrated<T>(res, params, dataset);                          \
  }                                                                                  \
  auto build(raft::resources const& res,                                             \
             index_params const& params,                                             \
             cuvs::neighbors::vpq_params const& vpq_params,                          \
             cuvs::neighbors::host_standard_dataset_view<T, int64_t> const& dataset) \
    -> vpq_f16_index_bundle<T, std::uint32_t>                                        \
  {                                                                                  \
    return build_vpq_integrated<T>(res, params, vpq_params, dataset);                \
  }

CUVS_FLOWANN_DEFINE_INTEGRATED_BUILD(float)
CUVS_FLOWANN_DEFINE_INTEGRATED_BUILD(half)
CUVS_FLOWANN_DEFINE_INTEGRATED_BUILD(std::int8_t)
CUVS_FLOWANN_DEFINE_INTEGRATED_BUILD(std::uint8_t)

#undef CUVS_FLOWANN_DEFINE_INTEGRATED_BUILD

template <typename T>
auto build_device_padded_from_cagra(
  raft::resources const& res,
  build_params const& params,
  cuvs::neighbors::cagra::device_padded_index<T, std::uint32_t> const& cagra_index,
  raft::host_vector_view<const std::uint32_t, int64_t> partition_labels,
  std::optional<raft::host_vector_view<const std::uint32_t, int64_t>> seeds)
  -> device_padded_index_bundle<T, std::uint32_t>
{
  auto conversion     = detail::convert_cagra_graph(res, params, cagra_index, partition_labels);
  auto source_indices = detail::make_source_indices(res, cagra_index, conversion.new_to_old);
  auto dataset = detail::reorder_dense_dataset(res, cagra_index.dataset(), conversion.new_to_old);
  flowann::device_padded_index<T, std::uint32_t> index(res,
                                                       cagra_index.metric(),
                                                       dataset->as_dataset_view(),
                                                       params.node_per_cacheline,
                                                       params.n_bits);
  detail::populate_index(res, index, std::move(conversion), source_indices, seeds);
  return {std::move(dataset), std::move(index)};
}

template <typename T>
auto build_vpq_from_cagra(
  raft::resources const& res,
  build_params const& params,
  cuvs::neighbors::cagra::device_pq_index<T, std::uint32_t> const& cagra_index,
  raft::host_vector_view<const std::uint32_t, int64_t> partition_labels,
  std::optional<raft::host_vector_view<const std::uint32_t, int64_t>> seeds)
  -> vpq_f16_index_bundle<T, std::uint32_t>
{
  auto conversion     = detail::convert_cagra_graph(res, params, cagra_index, partition_labels);
  auto source_indices = detail::make_source_indices(res, cagra_index, conversion.new_to_old);
  auto dataset = detail::reorder_vpq_dataset(res, cagra_index.dataset(), conversion.new_to_old);
  flowann::vpq_f16_index<T, std::uint32_t> index(res,
                                                 cagra_index.metric(),
                                                 dataset->as_dataset_view(),
                                                 params.node_per_cacheline,
                                                 params.n_bits);
  detail::populate_index(res, index, std::move(conversion), source_indices, seeds);
  return {std::move(dataset), std::move(index)};
}

template <typename T>
auto build_vpq_from_cagra_consuming(
  raft::resources const& res,
  build_params const& params,
  cuvs::neighbors::cagra::device_pq_index<T, std::uint32_t>&& cagra_index,
  std::unique_ptr<cuvs::neighbors::device_vpq_dataset<half, int64_t>>&& vpq_dataset,
  raft::host_vector_view<const std::uint32_t, int64_t> partition_labels,
  std::optional<raft::host_vector_view<const std::uint32_t, int64_t>> seeds)
  -> vpq_f16_index_bundle<T, std::uint32_t>
{
  RAFT_EXPECTS(vpq_dataset != nullptr, "FlowANN VPQ conversion requires a dataset owner");
  RAFT_EXPECTS(vpq_dataset->n_rows() > 0, "FlowANN VPQ dataset must not be empty");
  RAFT_EXPECTS(cagra_index.dataset().n_rows() == vpq_dataset->n_rows(),
               "FlowANN CAGRA index does not view the supplied VPQ dataset");
  RAFT_EXPECTS(&cagra_index.dataset().dset() == vpq_dataset.get(),
               "FlowANN CAGRA index does not view the supplied VPQ dataset owner");

  auto const metric   = cagra_index.metric();
  auto conversion     = detail::convert_cagra_graph(res, params, cagra_index, partition_labels);
  auto source_indices = detail::make_source_indices(res, cagra_index, conversion.new_to_old);
  cagra_index         = cuvs::neighbors::cagra::device_pq_index<T, std::uint32_t>(res, metric);

  auto dataset =
    detail::reorder_vpq_dataset_consuming(res, std::move(vpq_dataset), conversion.new_to_old);
  flowann::vpq_f16_index<T, std::uint32_t> index(
    res, metric, dataset->as_dataset_view(), params.node_per_cacheline, params.n_bits);
  detail::populate_index(res, index, std::move(conversion), source_indices, seeds);
  return {std::move(dataset), std::move(index)};
}

#define CUVS_FLOWANN_DEFINE_GROUP_GRAPH(T)                                                      \
  auto group_graph(raft::resources const& res,                                                  \
                   index_params const& params,                                                  \
                   raft::host_matrix_view<const std::uint32_t, int64_t, raft::row_major> graph, \
                   cuvs::neighbors::host_standard_dataset_view<T, int64_t> const& dataset)      \
    -> grouping_result                                                                          \
  {                                                                                             \
    return detail::generate_group_labels(res, params, graph, dataset);                          \
  }
CUVS_FLOWANN_DEFINE_GROUP_GRAPH(float)
CUVS_FLOWANN_DEFINE_GROUP_GRAPH(half)
CUVS_FLOWANN_DEFINE_GROUP_GRAPH(std::int8_t)
CUVS_FLOWANN_DEFINE_GROUP_GRAPH(std::uint8_t)
#undef CUVS_FLOWANN_DEFINE_GROUP_GRAPH

#define CUVS_FLOWANN_DEFINE_BUILD(T)                                                             \
  auto build_from_cagra(                                                                         \
    raft::resources const& res,                                                                  \
    build_params const& params,                                                                  \
    cuvs::neighbors::cagra::device_padded_index<T, std::uint32_t> const& cagra_index,            \
    raft::host_vector_view<const std::uint32_t, int64_t> partition_labels,                       \
    std::optional<raft::host_vector_view<const std::uint32_t, int64_t>> seeds)                   \
    -> device_padded_index_bundle<T, std::uint32_t>                                              \
  {                                                                                              \
    return build_device_padded_from_cagra<T>(res, params, cagra_index, partition_labels, seeds); \
  }                                                                                              \
  auto build_from_cagra(                                                                         \
    raft::resources const& res,                                                                  \
    build_params const& params,                                                                  \
    cuvs::neighbors::cagra::device_pq_index<T, std::uint32_t> const& cagra_index,                \
    raft::host_vector_view<const std::uint32_t, int64_t> partition_labels,                       \
    std::optional<raft::host_vector_view<const std::uint32_t, int64_t>> seeds)                   \
    -> vpq_f16_index_bundle<T, std::uint32_t>                                                    \
  {                                                                                              \
    return build_vpq_from_cagra<T>(res, params, cagra_index, partition_labels, seeds);           \
  }                                                                                              \
  auto build_from_cagra(                                                                         \
    raft::resources const& res,                                                                  \
    build_params const& params,                                                                  \
    cuvs::neighbors::cagra::device_pq_index<T, std::uint32_t>&& cagra_index,                     \
    std::unique_ptr<cuvs::neighbors::device_vpq_dataset<half, int64_t>>&& vpq_dataset,           \
    raft::host_vector_view<const std::uint32_t, int64_t> partition_labels,                       \
    std::optional<raft::host_vector_view<const std::uint32_t, int64_t>> seeds)                   \
    -> vpq_f16_index_bundle<T, std::uint32_t>                                                    \
  {                                                                                              \
    return build_vpq_from_cagra_consuming<T>(                                                    \
      res, params, std::move(cagra_index), std::move(vpq_dataset), partition_labels, seeds);     \
  }

CUVS_FLOWANN_DEFINE_BUILD(float)
CUVS_FLOWANN_DEFINE_BUILD(half)
CUVS_FLOWANN_DEFINE_BUILD(std::int8_t)
CUVS_FLOWANN_DEFINE_BUILD(std::uint8_t)

#undef CUVS_FLOWANN_DEFINE_BUILD

}  // namespace cuvs::neighbors::cagra::experimental::flowann
