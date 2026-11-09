/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "graph_core.cuh"
#include "graph_shared.cuh"
#include "utils.hpp"

// TODO: This shouldn't be invoking anything from spatial/knn
#include "../../../preprocessing/quantize/detail/bbq_distance.cuh"
#include "../ann_utils.cuh"

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/util/bitonic_sort.cuh>
#include <raft/util/cuda_rt_essentials.hpp>

#include <type_traits>
#include <utility>

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
  kernel<<<blocks, block_size, 0, raft::resource::get_cuda_stream(res).get()>>>(
    dataset, dataset_dim, knn_graph, dataset_size, graph_degree, metric);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT, typename IdxT>
using bbq_quantizer_view_t = cuvs::preprocessing::quantize::bbq::quantizer_view<DataT, IdxT>;

/**
 * Distance between two dataset rows in quantized space
 *
 */
template <typename DataT>
__device__ __forceinline__ float bbq_row_distance(
  const bbq_quantizer_view_t<DataT, int64_t>& quantizer_document,
  const bbq_quantizer_view_t<DataT, int64_t>& quantizer_query,
  cuvs::distance::DistanceType metric,
  int64_t row_document,
  int64_t row_query)
{
  namespace bbq_detail = cuvs::preprocessing::quantize::bbq::detail;
  const uint32_t raw =
    bbq_detail::code_inner_product(quantizer_document, quantizer_query, row_document, row_query);
  return bbq_detail::bbq_calculate_metric(
    raw,
    bbq_detail::get_dequant_factors(quantizer_document, row_document),
    bbq_detail::get_dequant_factors(quantizer_query, row_query),
    quantizer_document,
    quantizer_query,
    metric,
    raft::identity_op{},
    row_document,
    row_query);
}

template <typename DataT, int numElementsPerThread>
__global__ void kern_sort_bbq(const bbq_quantizer_view_t<DataT, int64_t> quantizer_document,
                              const bbq_quantizer_view_t<DataT, int64_t> quantizer_query,
                              uint32_t* const knn_graph,  // [graph_size, graph_degree]
                              const uint32_t graph_size,
                              const uint32_t graph_degree,
                              const cuvs::distance::DistanceType metric)
{
  const uint32_t src_node = (blockDim.x * blockIdx.x + threadIdx.x) / raft::WarpSize;
  if (src_node >= graph_size) { return; }

  const uint32_t lane_id = threadIdx.x % raft::WarpSize;

  float my_keys[numElementsPerThread];
  uint32_t my_vals[numElementsPerThread];

  for (int i = 0; i < numElementsPerThread; i++) {
    const uint32_t k = i * raft::WarpSize + lane_id;
    if (k >= graph_degree) {
      my_keys[i] = utils::get_max_value<float>();
      my_vals[i] = utils::get_max_value<uint32_t>();
      continue;
    }
    const uint32_t dst_node = knn_graph[k + static_cast<uint64_t>(graph_degree) * src_node];
    my_keys[i] = bbq_row_distance(quantizer_document, quantizer_query, metric, src_node, dst_node);
    my_vals[i] = dst_node;
  }

  raft::util::bitonic<numElementsPerThread>(true).sort(my_keys, my_vals);

  for (int i = 0; i < numElementsPerThread; i++) {
    const uint32_t k = i * raft::WarpSize + lane_id;
    if (k < graph_degree) {
      knn_graph[k + (static_cast<uint64_t>(graph_degree) * src_node)] = my_vals[i];
    }
  }
}

template <typename DataT>
using sort_bbq_kernel_type = void (*)(bbq_quantizer_view_t<DataT, int64_t>,
                                      bbq_quantizer_view_t<DataT, int64_t>,
                                      uint32_t*,
                                      uint32_t,
                                      uint32_t,
                                      cuvs::distance::DistanceType);

template <typename DataT>
auto select_sort_bbq_kernel(uint32_t degree) -> sort_bbq_kernel_type<DataT>
{
  if (degree <= raft::WarpSize * 1) { return kern_sort_bbq<DataT, 1>; }
  if (degree <= raft::WarpSize * 2) { return kern_sort_bbq<DataT, 2>; }
  if (degree <= raft::WarpSize * 4) { return kern_sort_bbq<DataT, 4>; }
  if (degree <= raft::WarpSize * 8) { return kern_sort_bbq<DataT, 8>; }
  if (degree <= raft::WarpSize * 16) { return kern_sort_bbq<DataT, 16>; }
  if (degree <= kMaxSortDegree) { return kern_sort_bbq<DataT, kMaxSortElementsPerThread>; }
  RAFT_FAIL(
    "The degree of input knn graph is too large (%u). It must be equal to or smaller than %lu.",
    degree,
    kMaxSortDegree);
}

template <typename DataT>
auto select_sort_quantizers(cuvs::neighbors::device_bbq_dataset_view<DataT, int64_t> const& dataset)
  -> std::pair<bbq_quantizer_view_t<DataT, int64_t>, bbq_quantizer_view_t<DataT, int64_t>>
{
  using bbq_code_layout = cuvs::preprocessing::quantize::bbq::bbq_code_layout;

  if (dataset.quantizers.size() == 1) { return {dataset.quantizers[0], dataset.quantizers[0]}; }

  const bool has_1b  = dataset.has_layout(bbq_code_layout::packed_1b);
  const bool has_4b  = dataset.has_layout(bbq_code_layout::packed_4b);
  const bool has_2bt = dataset.has_layout(bbq_code_layout::transposed_2b);
  const bool has_4bt = dataset.has_layout(bbq_code_layout::transposed_4b);

  const bool tc_pair   = has_1b && has_4b;
  const bool simt_pair = (has_4bt && (has_1b || has_2bt)) || (has_2bt && has_1b);
  RAFT_EXPECTS(tc_pair || simt_pair,
               "Unsupported BBQ layout pair for an asymmetric dataset. Supported (document, "
               "query) pairs: (packed_1b, packed_4b), (packed_1b, transposed_2b), "
               "(packed_1b, transposed_4b), (transposed_2b, transposed_4b).");
  return {has_1b ? dataset.get_quantizer(bbq_code_layout::packed_1b)
                 : dataset.get_quantizer(bbq_code_layout::transposed_2b),
          tc_pair   ? dataset.get_quantizer(bbq_code_layout::packed_4b)
          : has_4bt ? dataset.get_quantizer(bbq_code_layout::transposed_4b)
                    : dataset.get_quantizer(bbq_code_layout::transposed_2b)};
}

template <typename DataT>
void sort_knn_graph_bbq_impl(raft::resources const& res,
                             cuvs::distance::DistanceType metric,
                             cuvs::neighbors::device_bbq_dataset_view<DataT, int64_t> dataset,
                             raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph)
{
  namespace bbq_detail = cuvs::preprocessing::quantize::bbq::detail;

  RAFT_EXPECTS(!dataset.quantizers.empty(), "the BBQ dataset holds no quantizer");
  RAFT_EXPECTS(dataset.n_rows() == knn_graph.extent(0),
               "dataset size is expected to have the same number of graph index size");
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2Expanded ||
                 metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                 metric == cuvs::distance::DistanceType::InnerProduct ||
                 metric == cuvs::distance::DistanceType::CosineExpanded,
               "Unsupported metric. Only L2Expanded, L2SqrtExpanded, InnerProduct and "
               "CosineExpanded are supported for a BBQ-quantized dataset");

  auto const graph_size                            = static_cast<uint32_t>(knn_graph.extent(0));
  auto const graph_degree                          = static_cast<uint32_t>(knn_graph.extent(1));
  auto kernel                                      = select_sort_bbq_kernel<DataT>(graph_degree);
  auto const [quantizer_document, quantizer_query] = select_sort_quantizers(dataset);
  // The code inner products read both rows as uint32_t words, and a bit-sliced layout starts
  // every plane at a multiple of the plane stride, so each plane must be 4-byte aligned.
  for (const auto& quantizer : {quantizer_document, quantizer_query}) {
    const auto row_length = bbq_detail::get_encoded_row_length(quantizer);
    const auto planes     = static_cast<uint32_t>(bbq_detail::get_code_planes(quantizer.layout));
    RAFT_EXPECTS(row_length % (4u * planes) == 0,
                 "Sorting a BBQ-quantized kNN graph requires the encoded row length to be a "
                 "multiple of 4*n_planes for 32-bit aligned plane loads, got %u with n_planes = %u",
                 row_length,
                 planes);
  }
  // A packed_1b document is promoted to 4-bit width one 32-dimension word at a time, which only
  // covers the packed_4b query row exactly when the dimensionality is a multiple of 32.
  RAFT_EXPECTS(
    quantizer_query.layout != cuvs::preprocessing::quantize::bbq::bbq_code_layout::packed_4b ||
      quantizer_document.layout == quantizer_query.layout || quantizer_document.dim() % 32 == 0,
    "Sorting a BBQ-quantized kNN graph with packed_1b codes against packed_4b ones "
    "requires the dataset dim to be a multiple of 32, got %u",
    quantizer_document.dim());

  const double time_sort_start = cur_time();
  RAFT_LOG_DEBUG("# Sorting kNN Graph on GPUs\n");

  auto large_tmp_mr  = raft::resource::get_large_workspace_resource_ref(res);
  auto d_input_graph = raft::make_device_mdarray<uint32_t>(
    res, large_tmp_mr, raft::make_extents<int64_t>(graph_size, graph_degree));
  raft::copy(res, d_input_graph.view(), knn_graph);

  constexpr uint32_t block_size = 256;
  auto const warps              = block_size / raft::WarpSize;
  auto const blocks             = (graph_size + warps - 1) / warps;
  kernel<<<blocks, block_size, 0, raft::resource::get_cuda_stream(res).get()>>>(
    quantizer_document,
    quantizer_query,
    d_input_graph.data_handle(),
    graph_size,
    graph_degree,
    metric);
  RAFT_CUDA_TRY(cudaGetLastError());
  raft::resource::sync_stream(res);
  raft::copy(res, knn_graph, raft::make_const_mdspan(d_input_graph.view()));

  const double time_sort_end = cur_time();
  RAFT_LOG_DEBUG("# Sorting kNN graph time: %.1lf sec\n", time_sort_end - time_sort_start);
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

#define CUVS_DEFINE_CAGRA_GRAPH_SORT_BBQ(DataT)                                                 \
  void sort_knn_graph_bbq(raft::resources const& res,                                           \
                          cuvs::distance::DistanceType metric,                                  \
                          cuvs::neighbors::device_bbq_dataset_view<DataT, int64_t> dataset,     \
                          raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph) \
  {                                                                                             \
    sort_knn_graph_bbq_impl(res, metric, dataset, knn_graph);                                   \
  }

CUVS_DEFINE_CAGRA_GRAPH_SORT_BBQ(float)
CUVS_DEFINE_CAGRA_GRAPH_SORT_BBQ(half)
CUVS_DEFINE_CAGRA_GRAPH_SORT_BBQ(int8_t)
CUVS_DEFINE_CAGRA_GRAPH_SORT_BBQ(uint8_t)

#undef CUVS_DEFINE_CAGRA_GRAPH_SORT_BBQ

void optimize_device_graph(
  raft::resources const& res,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> output_graph,
  bool guarantee_connectivity)
{
  optimize(res, knn_graph, output_graph, guarantee_connectivity);
}

}  // namespace cuvs::neighbors::cagra::detail::graph
