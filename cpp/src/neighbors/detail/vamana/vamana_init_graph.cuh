/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../core/nvtx.hpp"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <cuvs/neighbors/vamana.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/bitonic_sort.cuh>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cuvs::neighbors::vamana::detail {

template <typename T>
__device__ __forceinline__ auto init_graph_to_float(T x) -> float
{
  if constexpr (std::is_same_v<T, half>) {
    return __half2float(x);
  } else {
    return static_cast<float>(x);
  }
}

/**
 * Sorts every neighbor list of a graph by squared L2 distance to the list's own vector, with one
 * warp per list. Ties are broken by neighbor id, so the order is deterministic. Ids that are not
 * valid rows move to the end of the list as raft::upper_bound<IdxT>().
 *
 * With drop_repeats set, self links and repeated ids are removed as well, so each list holds
 * distinct neighbors, sorted, followed by invalid ids. This is the form the Vamana insertion
 * expects of an existing graph.
 *
 * kCapacity is the number of list entries each lane holds: degree <= kCapacity * WarpSize.
 */
template <int kCapacity, typename T, typename IdxT, typename Accessor>
__global__ void sort_graph_by_distance_kernel(
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  IdxT* graph,
  int64_t n_rows,
  int degree,
  bool drop_repeats)
{
  constexpr uint64_t kInvalidKey = std::numeric_limits<uint64_t>::max();
  constexpr uint32_t kFullMask   = 0xffffffffu;

  const int64_t row =
    (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / raft::WarpSize;
  if (row >= n_rows) { return; }
  const int lane    = threadIdx.x % raft::WarpSize;
  const int64_t dim = dataset.extent(1);
  const T* data     = dataset.data_handle();
  const T* src      = data + row * dim;
  IdxT* list        = graph + row * degree;

  // Entry j of the list goes to lane j % WarpSize, slot j / WarpSize. The key holds the distance
  // bits (non-negative floats order like their bit patterns) above the neighbor id.
  uint64_t keys[kCapacity];
#pragma unroll
  for (int i = 0; i < kCapacity; i++) {
    keys[i] = kInvalidKey;
    for (int l = 0; l < raft::WarpSize; l++) {
      const int j = i * raft::WarpSize + l;
      if (j >= degree) { break; }
      // Every lane reads the same id, so these branches are uniform across the warp.
      const IdxT nbr = list[j];
      if (static_cast<uint64_t>(nbr) >= static_cast<uint64_t>(n_rows)) { continue; }
      if (drop_repeats && static_cast<int64_t>(nbr) == row) { continue; }
      const T* dst = data + static_cast<int64_t>(nbr) * dim;
      float dist   = 0.0f;
      for (int64_t k = lane; k < dim; k += raft::WarpSize) {
        const float diff = init_graph_to_float(src[k]) - init_graph_to_float(dst[k]);
        dist             = fmaf(diff, diff, dist);
      }
      for (int offset = raft::WarpSize / 2; offset > 0; offset /= 2) {
        dist += __shfl_xor_sync(kFullMask, dist, offset);
      }
      if (lane == l) {
        keys[i] = (static_cast<uint64_t>(__float_as_uint(dist)) << 32) | static_cast<uint32_t>(nbr);
      }
    }
  }

  raft::util::bitonic<kCapacity>(true).sort(keys);
  __syncwarp();

  // After the sort, slot i of lane l holds the entry of rank i * WarpSize + l. Write the kept
  // entries back in rank order. A repeated id sorts next to its first copy, since both copies have
  // the same key.
  int n_out          = 0;
  uint32_t prev_last = std::numeric_limits<uint32_t>::max();
#pragma unroll
  for (int i = 0; i < kCapacity; i++) {
    const uint32_t id = static_cast<uint32_t>(keys[i]);
    uint32_t prev     = __shfl_up_sync(kFullMask, id, 1);
    if (lane == 0) { prev = prev_last; }
    const bool keep       = keys[i] != kInvalidKey && !(drop_repeats && id == prev);
    const uint32_t ballot = __ballot_sync(kFullMask, keep);
    if (keep) { list[n_out + __popc(ballot & ((1u << lane) - 1u))] = static_cast<IdxT>(id); }
    n_out += __popc(ballot);
    prev_last = __shfl_sync(kFullMask, id, raft::WarpSize - 1);
  }
  for (int j = n_out + lane; j < degree; j += raft::WarpSize) {
    list[j] = raft::upper_bound<IdxT>();
  }
}

template <typename T, typename IdxT, typename Accessor>
void sort_graph_by_distance(
  raft::resources const& res,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  raft::device_matrix_view<IdxT, int64_t> graph,
  bool drop_repeats)
{
  const int64_t n_rows            = graph.extent(0);
  const int degree                = static_cast<int>(graph.extent(1));
  constexpr int kBlockSize        = 256;
  constexpr int64_t kRowsPerBlock = kBlockSize / raft::WarpSize;
  const auto n_blocks =
    static_cast<unsigned int>(raft::div_rounding_up_safe<int64_t>(n_rows, kRowsPerBlock));
  if (n_blocks == 0) { return; }
  auto stream = raft::resource::get_cuda_stream(res);

  auto launch = [&](auto kernel) {
    kernel<<<n_blocks, kBlockSize, 0, stream.get()>>>(
      dataset, graph.data_handle(), n_rows, degree, drop_repeats);
  };
  if (degree <= raft::WarpSize) {
    launch(sort_graph_by_distance_kernel<1, T, IdxT, Accessor>);
  } else if (degree <= 2 * raft::WarpSize) {
    launch(sort_graph_by_distance_kernel<2, T, IdxT, Accessor>);
  } else if (degree <= 4 * raft::WarpSize) {
    launch(sort_graph_by_distance_kernel<4, T, IdxT, Accessor>);
  } else if (degree <= 8 * raft::WarpSize) {
    launch(sort_graph_by_distance_kernel<8, T, IdxT, Accessor>);
  } else if (degree <= 16 * raft::WarpSize) {
    launch(sort_graph_by_distance_kernel<16, T, IdxT, Accessor>);
  } else {
    RAFT_FAIL(
      "Cannot sort neighbor lists longer than %d entries, got %d", 16 * raft::WarpSize, degree);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/** Sums each column over one chunk of rows. Block (x, y) covers columns of x and rows of chunk y.
 */
template <typename T, typename Accessor>
__global__ void column_chunk_sums_kernel(
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  int64_t rows_per_chunk,
  float* chunk_sums)
{
  const int64_t n_rows = dataset.extent(0);
  const int64_t dim    = dataset.extent(1);
  const int64_t col    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= dim) { return; }
  const int64_t chunk = blockIdx.y;
  const int64_t begin = chunk * rows_per_chunk;
  const int64_t end   = std::min(n_rows, begin + rows_per_chunk);
  const T* data       = dataset.data_handle();
  float sum           = 0.0f;
  for (int64_t r = begin; r < end; r++) {
    sum += init_graph_to_float(data[r * dim + col]);
  }
  chunk_sums[chunk * dim + col] = sum;
}

/**
 * Adds up the chunk sums of each column in a fixed order, so the result is deterministic, and
 * divides by the row count. SumT is the accumulator type.
 */
template <typename SumT>
__global__ void column_means_kernel(
  const float* chunk_sums, int64_t n_chunks, int64_t dim, int64_t n_rows, float* means)
{
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= dim) { return; }
  SumT sum = 0;
  for (int64_t c = 0; c < n_chunks; c++) {
    sum += static_cast<SumT>(chunk_sums[c * dim + col]);
  }
  means[col] = static_cast<float>(sum / static_cast<SumT>(n_rows));
}

/** Squared L2 distance from each row to `point`, one warp per row. */
template <typename T, typename Accessor>
__global__ void distance_to_point_kernel(
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  const float* point,
  float* distances)
{
  const int64_t n_rows = dataset.extent(0);
  const int64_t dim    = dataset.extent(1);
  const int64_t row =
    (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / raft::WarpSize;
  if (row >= n_rows) { return; }
  const int lane = threadIdx.x % raft::WarpSize;
  const T* src   = dataset.data_handle() + row * dim;
  float dist     = 0.0f;
  for (int64_t k = lane; k < dim; k += raft::WarpSize) {
    const float diff = init_graph_to_float(src[k]) - point[k];
    dist             = fmaf(diff, diff, dist);
  }
  for (int offset = raft::WarpSize / 2; offset > 0; offset /= 2) {
    dist += __shfl_xor_sync(0xffffffffu, dist, offset);
  }
  if (lane == 0) { distances[row] = dist; }
}

/** Returns the row nearest the mean of the dataset (the lowest such row on ties). */
template <typename T, typename IdxT, typename Accessor>
auto nearest_to_mean(
  raft::resources const& res,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset) -> IdxT
{
  const int64_t n_rows = dataset.extent(0);
  const int64_t dim    = dataset.extent(1);
  auto stream          = raft::resource::get_cuda_stream(res);
  auto workspace       = raft::resource::get_workspace_resource_ref(res);

  // Grid dimension y is limited to 65535 chunks.
  constexpr int64_t kMaxChunks    = 65535;
  const int64_t rows_per_chunk    = std::max<int64_t>(1024, (n_rows + kMaxChunks - 1) / kMaxChunks);
  const int64_t n_chunks          = (n_rows + rows_per_chunk - 1) / rows_per_chunk;
  constexpr int kBlockSize        = 256;
  const auto col_blocks           = static_cast<unsigned int>((dim + kBlockSize - 1) / kBlockSize);
  constexpr int64_t kRowsPerBlock = kBlockSize / raft::WarpSize;

  auto chunk_sums =
    raft::make_device_mdarray<float>(res, workspace, raft::make_extents<int64_t>(n_chunks * dim));
  auto means = raft::make_device_mdarray<float>(res, workspace, raft::make_extents<int64_t>(dim));
  auto distances =
    raft::make_device_mdarray<float>(res,
                                     raft::resource::get_large_workspace_resource_ref(res),
                                     raft::make_extents<int64_t>(n_rows));

  column_chunk_sums_kernel<T, Accessor>
    <<<dim3(col_blocks, static_cast<unsigned int>(n_chunks)), kBlockSize, 0, stream.get()>>>(
      dataset, rows_per_chunk, chunk_sums.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  column_means_kernel<double><<<col_blocks, kBlockSize, 0, stream.get()>>>(
    chunk_sums.data_handle(), n_chunks, dim, n_rows, means.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  distance_to_point_kernel<T, Accessor>
    <<<static_cast<unsigned int>((n_rows + kRowsPerBlock - 1) / kRowsPerBlock),
       kBlockSize,
       0,
       stream.get()>>>(dataset, means.data_handle(), distances.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  auto first   = thrust::device_pointer_cast(distances.data_handle());
  auto nearest = thrust::min_element(rmm::exec_policy(stream, workspace), first, first + n_rows);
  return static_cast<IdxT>(nearest - first);
}

/**
 * Builds the NN-descent kNN graph that seeds the Vamana graph, in host memory. NN-descent runs
 * before the Vamana build allocates its buffers, and its device memory is released when it
 * returns.
 */
template <typename T, typename Accessor>
auto build_init_knn_graph(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  int64_t graph_degree) -> raft::host_matrix<uint32_t, int64_t>
{
  // NN-descent supports every data type and metric that Vamana does. Check here as well, so that
  // a future Vamana data type or metric fails clearly instead of being seeded incorrectly.
  if constexpr (!(std::is_same_v<T, float> || std::is_same_v<T, half> ||
                  std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)) {
    RAFT_FAIL(
      "NN_DESCENT graph initialization supports float, half, int8_t and uint8_t datasets only");
  } else {
    const int64_t n_rows = dataset.extent(0);
    const int64_t dim    = dataset.extent(1);
    RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded ||
                   params.metric == cuvs::distance::DistanceType::L2SqrtExpanded,
                 "NN_DESCENT graph initialization supports the L2Expanded and L2SqrtExpanded "
                 "metrics only");
    RAFT_EXPECTS(n_rows > graph_degree,
                 "NN_DESCENT graph initialization needs more dataset rows (%ld) than graph_degree "
                 "(%ld)",
                 static_cast<long>(n_rows),
                 static_cast<long>(graph_degree));

    // The kNN graph is twice the Vamana degree (fewer on tiny datasets), and CAGRA's optimize step
    // prunes it to the Vamana degree.
    const int64_t knn_degree = std::min<int64_t>(2 * graph_degree, n_rows - 1);
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
      "vamana::build_init_knn_graph<NN-DESCENT>(%zu, %zu, %zu)",
      static_cast<size_t>(n_rows),
      static_cast<size_t>(dim),
      static_cast<size_t>(knn_degree));
    RAFT_LOG_DEBUG("Building the NN-descent kNN graph of degree %ld to seed the Vamana graph",
                   static_cast<long>(knn_degree));

    cuvs::neighbors::nn_descent::index_params nnd_params(static_cast<size_t>(knn_degree),
                                                         params.metric);
    nnd_params.return_distances = false;
    auto knn_graph              = raft::make_host_matrix<uint32_t, int64_t>(n_rows, knn_degree);
    if constexpr (Accessor::is_device_accessible) {
      cuvs::neighbors::nn_descent::build(
        res,
        nnd_params,
        raft::make_device_matrix_view<const T, int64_t>(dataset.data_handle(), n_rows, dim),
        knn_graph.view());
    } else {
      cuvs::neighbors::nn_descent::build(
        res,
        nnd_params,
        raft::make_host_matrix_view<const T, int64_t>(dataset.data_handle(), n_rows, dim),
        knn_graph.view());
    }
    return knn_graph;
  }
}

/**
 * Fills the Vamana graph from the NN-descent kNN graph: sorts the kNN lists by distance, prunes
 * them to the graph degree with CAGRA's optimize step, and sorts the pruned lists by distance
 * without self links or repeats. Returns the medoid, the row nearest the dataset mean.
 */
template <typename T, typename IdxT, typename Accessor>
auto init_graph_from_knn_graph(
  raft::resources const& res,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  raft::host_matrix_view<uint32_t, int64_t> knn_graph,
  raft::device_matrix_view<IdxT, int64_t> graph) -> IdxT
{
  if constexpr (!std::is_same_v<IdxT, uint32_t>) {
    RAFT_FAIL("NN_DESCENT graph initialization requires 32-bit graph indices");
  } else {
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
      "vamana::init_graph_from_knn_graph(%zu, %zu)",
      static_cast<size_t>(graph.extent(0)),
      static_cast<size_t>(graph.extent(1)));
    auto stream = raft::resource::get_cuda_stream(res);
    {
      auto d_knn_graph = raft::make_device_mdarray<uint32_t>(
        res,
        raft::resource::get_large_workspace_resource_ref(res),
        raft::make_extents<int64_t>(knn_graph.extent(0), knn_graph.extent(1)));
      raft::copy(d_knn_graph.data_handle(), knn_graph.data_handle(), knn_graph.size(), stream);
      // CAGRA's optimize step ranks the kNN neighbors by their position in each list.
      sort_graph_by_distance<T, uint32_t, Accessor>(res, dataset, d_knn_graph.view(), false);
      cuvs::neighbors::cagra::helpers::optimize(res, d_knn_graph.view(), graph, false);
    }
    sort_graph_by_distance<T, IdxT, Accessor>(res, dataset, graph, true);
    return nearest_to_mean<T, IdxT, Accessor>(res, dataset);
  }
}

}  // namespace cuvs::neighbors::vamana::detail
