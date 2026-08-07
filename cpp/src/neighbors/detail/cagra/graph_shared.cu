/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "graph_core.cuh"
#include "graph_shared.cuh"
#include "utils.hpp"

// TODO: This shouldn't be invoking anything from spatial/knn
#include "../ann_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/bitonic_sort.cuh>
#include <raft/util/cuda_rt_essentials.hpp>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail::graph {
namespace {

template <class DATA_T, int numElementsPerThread>
__global__ void kern_sort(const DATA_T* const dataset,  // [dataset_chunk_size, dataset_dim]
                          const uint32_t dataset_dim,
                          uint32_t* const knn_graph,  // [graph_chunk_size, graph_degree]
                          const uint32_t graph_size,
                          const uint32_t graph_degree,
                          const cuvs::distance::DistanceType metric)
{
  const uint32_t srcNode = (blockDim.x * blockIdx.x + threadIdx.x) / raft::WarpSize;
  if (srcNode >= graph_size) { return; }

  const uint32_t lane_id = threadIdx.x % raft::WarpSize;

  float my_keys[numElementsPerThread];
  uint32_t my_vals[numElementsPerThread];

  // Compute distance from a src node to its neighbors
  for (int k = 0; k < graph_degree; k++) {
    const uint32_t dstNode = knn_graph[k + static_cast<uint64_t>(graph_degree) * srcNode];
    float dist             = 0;
    float norm2_dst        = 0;
    if (metric == cuvs::distance::DistanceType::InnerProduct ||
        metric == cuvs::distance::DistanceType::CosineExpanded) {
      for (int d = lane_id; d < dataset_dim; d += raft::WarpSize) {
        auto elem_b = cuvs::spatial::knn::detail::utils::mapping<float>{}(
          dataset[d + static_cast<uint64_t>(dataset_dim) * dstNode]);
        dist -= cuvs::spatial::knn::detail::utils::mapping<float>{}(
                  dataset[d + static_cast<uint64_t>(dataset_dim) * srcNode]) *
                elem_b;

        if (metric == cuvs::distance::DistanceType::CosineExpanded) {
          norm2_dst += elem_b * elem_b;
        }
      }
    } else if (metric == cuvs::distance::DistanceType::L2Expanded) {
      for (int d = lane_id; d < dataset_dim; d += raft::WarpSize) {
        float diff = cuvs::spatial::knn::detail::utils::mapping<float>{}(
                       dataset[d + static_cast<uint64_t>(dataset_dim) * srcNode]) -
                     cuvs::spatial::knn::detail::utils::mapping<float>{}(
                       dataset[d + static_cast<uint64_t>(dataset_dim) * dstNode]);
        dist += diff * diff;
      }
    } else if (metric == cuvs::distance::DistanceType::L1) {
      for (int d = lane_id; d < dataset_dim; d += raft::WarpSize) {
        float diff = cuvs::spatial::knn::detail::utils::mapping<float>{}(
                       dataset[d + static_cast<uint64_t>(dataset_dim) * srcNode]) -
                     cuvs::spatial::knn::detail::utils::mapping<float>{}(
                       dataset[d + static_cast<uint64_t>(dataset_dim) * dstNode]);
        dist += raft::abs(diff);
      }
    } else if (metric == cuvs::distance::DistanceType::BitwiseHamming) {
      if constexpr (std::is_integral_v<DATA_T>) {
        for (int d = lane_id; d < dataset_dim; d += raft::WarpSize) {
          dist += __popc(
            static_cast<uint32_t>(dataset[d + static_cast<uint64_t>(dataset_dim) * srcNode] ^
                                  dataset[d + static_cast<uint64_t>(dataset_dim) * dstNode]) &
            0xffu);
        }
      }
    }
    dist += __shfl_xor_sync(0xffffffff, dist, 1);
    dist += __shfl_xor_sync(0xffffffff, dist, 2);
    dist += __shfl_xor_sync(0xffffffff, dist, 4);
    dist += __shfl_xor_sync(0xffffffff, dist, 8);
    dist += __shfl_xor_sync(0xffffffff, dist, 16);

    if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 1);
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 2);
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 4);
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 8);
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 16);
      if (lane_id == (k % raft::WarpSize)) { dist /= sqrt(norm2_dst); }
    }

    if (lane_id == (k % raft::WarpSize)) {
      my_keys[k / raft::WarpSize] = dist;
      my_vals[k / raft::WarpSize] = dstNode;
    }
  }
  for (int k = graph_degree; k < raft::WarpSize * numElementsPerThread; k++) {
    if (lane_id == k % raft::WarpSize) {
      my_keys[k / raft::WarpSize] = utils::get_max_value<float>();
      my_vals[k / raft::WarpSize] = utils::get_max_value<uint32_t>();
    }
  }

  raft::util::bitonic<numElementsPerThread>(true).sort(my_keys, my_vals);

  for (int i = 0; i < numElementsPerThread; i++) {
    const int k = i * raft::WarpSize + lane_id;
    if (k < graph_degree) {
      knn_graph[k + (static_cast<uint64_t>(graph_degree) * srcNode)] = my_vals[i];
    }
  }
}

constexpr int kMaxSortElementsPerThread = 32;

template <typename DataT>
using sort_kernel_type =
  void (*)(DataT const*, uint32_t, uint32_t*, uint32_t, uint32_t, cuvs::distance::DistanceType);

template <typename DataT>
auto select_sort_kernel(uint32_t degree) -> sort_kernel_type<DataT>
{
  if (degree <= raft::WarpSize * 1) { return kern_sort<DataT, 1>; }
  if (degree <= raft::WarpSize * 2) { return kern_sort<DataT, 2>; }
  if (degree <= raft::WarpSize * 4) { return kern_sort<DataT, 4>; }
  if (degree <= raft::WarpSize * 8) { return kern_sort<DataT, 8>; }
  if (degree <= raft::WarpSize * 16) { return kern_sort<DataT, 16>; }
  if (degree <= kMaxSortDegree) { return kern_sort<DataT, kMaxSortElementsPerThread>; }
  RAFT_FAIL(
    "The degree of input knn graph is too large (%u). It must be equal to or smaller than %lu.",
    degree,
    kMaxSortDegree);
}

template <typename DataT>
void launch_sort_knn_graph_impl(raft::resources const& res,
                                cuvs::distance::DistanceType metric,
                                DataT const* dataset,
                                uint32_t dataset_size,
                                uint32_t dataset_dim,
                                uint32_t* knn_graph,
                                uint32_t graph_degree)
{
  auto kernel = select_sort_kernel<DataT>(graph_degree);

  constexpr uint32_t block_size = 256;
  auto const warps              = block_size / raft::WarpSize;
  auto const blocks             = (dataset_size + warps - 1) / warps;
  kernel<<<blocks, block_size, 0, raft::resource::get_cuda_stream(res)>>>(
    dataset, dataset_dim, knn_graph, dataset_size, graph_degree, metric);
  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace

#define CUVS_DEFINE_CAGRA_GRAPH_SORT(DataT)                                      \
  void launch_sort_knn_graph(raft::resources const& res,                         \
                             cuvs::distance::DistanceType metric,                \
                             DataT const* dataset,                               \
                             uint32_t dataset_size,                              \
                             uint32_t dataset_dim,                               \
                             uint32_t* knn_graph,                                \
                             uint32_t graph_degree)                              \
  {                                                                              \
    launch_sort_knn_graph_impl(                                                  \
      res, metric, dataset, dataset_size, dataset_dim, knn_graph, graph_degree); \
  }

CUVS_DEFINE_CAGRA_GRAPH_SORT(float)
CUVS_DEFINE_CAGRA_GRAPH_SORT(half)
CUVS_DEFINE_CAGRA_GRAPH_SORT(int8_t)
CUVS_DEFINE_CAGRA_GRAPH_SORT(uint8_t)

#undef CUVS_DEFINE_CAGRA_GRAPH_SORT

void optimize_device_graph(
  raft::resources const& res,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> output_graph,
  bool guarantee_connectivity)
{
  optimize(res, knn_graph, output_graph, guarantee_connectivity);
}

}  // namespace cuvs::neighbors::cagra::detail::graph
