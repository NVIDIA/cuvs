/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra_helpers.hpp"

#include <algorithm>
#include <cstdint>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <utility>

namespace cuvs::neighbors::cagra::helpers {
namespace {
// Size in bytes of a single element of the given CUDA data type.
size_t cuda_data_type_size(cudaDataType_t dtype)
{
  switch (dtype) {
    case CUDA_R_32F: return 4;
    case CUDA_R_16F: return 2;
    case CUDA_R_8I:
    case CUDA_R_8U: return 1;
    default:
      RAFT_FAIL("cagra_build_mem_usage: unsupported dataset element type %d",
                static_cast<int>(dtype));
  }
}
}  // namespace

// Calculate CAGRA optimize workspace memory requirements.
// This is the working memory on top of the input/output memory usage.
std::tuple<size_t, size_t, size_t, size_t> optimize_workspace_size(size_t n_rows,
                                                                   size_t graph_degree,
                                                                   size_t intermediate_degree,
                                                                   size_t index_size,
                                                                   bool mst_optimize)
{
  RAFT_EXPECTS(graph_degree > 0, "graph_degree must be greater than 0");
  RAFT_EXPECTS(intermediate_degree >= graph_degree,
               "intermediate_degree must be greater than or equal to graph_degree");

  // MST optimization memory (host only)
  size_t mst_host       = 0;
  size_t mst_host_fixed = 0;
  if (mst_optimize) {
    mst_host = n_rows * index_size;                  // mst_graph_num_edges
    mst_host += n_rows * graph_degree * index_size;  // mst_graph allocated in optimize
    mst_host += n_rows * graph_degree * index_size;  // mst_graph allocated in mst_optimize
    mst_host +=
      n_rows * index_size * 7;  // Five vectors _edges suffix, and label, cluster_size vectors.
    mst_host_fixed += (graph_degree - 1) * (graph_degree - 1) * index_size;  // iB_candidates
    mst_host += mst_host_fixed;
  }

  // batchsize for both prune and combine stages
  size_t batch_size = std::min(kOptimizeBatchSize, n_rows);

  // Prune stage memory
  // We neglect 8 bytes (both on host and device) for stats
  size_t prune_dev_fixed = batch_size * intermediate_degree;      // detour count (uint8_t)
  prune_dev_fixed += batch_size * sizeof(uint32_t);               // d_num_detour_edges
  prune_dev_fixed += 2 * batch_size * graph_degree * index_size;  // d_output_graph(2*batch)

  size_t prune_dev = n_rows * intermediate_degree * index_size;  // d_input_graph
  prune_dev += prune_dev_fixed;

  // Reverse graph stage memory
  size_t rev_dev = n_rows * graph_degree * index_size;  // d_rev_graph
  rev_dev += n_rows * sizeof(uint32_t);                 // d_rev_graph_count
  rev_dev += n_rows * index_size;                       // d_dest_nodes

  // Memory for merging graphs (host only optional)
  size_t combine_host_fixed = graph_degree * sizeof(uint32_t);  // histogram
  size_t combine_host       = n_rows * sizeof(uint32_t);        // n_edge_count
  combine_host += combine_host_fixed;

  // additional memory for combine stage on device (3 batches)
  size_t combine_dev_fixed = 2 * batch_size * graph_degree * index_size;  // d_output_graph(2*batch)
  if (mst_optimize) {
    combine_dev_fixed += 2 * batch_size * graph_degree * index_size;  // d_mst_graph(2*batch)
    combine_dev_fixed += 2 * batch_size * sizeof(uint32_t);  // d_mst_graph_num_edges(2*batch)
  }
  size_t combine_dev = combine_dev_fixed;

  size_t debug_host_size = 0;
  if (raft::default_logger().should_log(rapids_logger::level_enum::debug)) {
    // cagra::detail::graph::optimize() allocates extra memory to calculate
    // graph metrics when debug logging is enabled
    debug_host_size = n_rows * graph_degree * sizeof(uint32_t)  // host_copy_output_graph
                      + n_rows * sizeof(uint32_t)               // in_edge_count
                      + graph_degree * sizeof(uint32_t);        // hist
  }

  size_t total_host       = mst_host + combine_host + debug_host_size;
  size_t total_host_fixed = mst_host_fixed + combine_host_fixed;
  size_t total_dev        = std::max(prune_dev, rev_dev + combine_dev);
  size_t total_dev_fixed  = std::max(prune_dev_fixed, combine_dev_fixed);

  return std::make_tuple(total_host, total_dev, total_host_fixed, total_dev_fixed);
}

// All sizes are in bytes
inline std::pair<size_t, size_t> ivf_pq_build_mem_usage(
  raft::resources const& res,
  raft::matrix_extent<int64_t> dataset,
  cudaDataType_t dtype,
  cuvs::neighbors::graph_build_params::ivf_pq_params params,
  size_t graph_degree,
  size_t intermediate_graph_degree,
  bool guarantee_connectivity)
{
  size_t dtype_size   = cuda_data_type_size(dtype);
  bool input_is_float = (dtype == CUDA_R_32F);

  size_t n_rows = dataset.extent(0);
  size_t dim    = dataset.extent(1);

  size_t dataset_gpu_mem =
    cuvs::neighbors::ivf_pq::helpers::compressed_dataset_size(res, dataset, params.build_params);
  size_t graph_host_mem = n_rows * (graph_degree + intermediate_graph_degree) * sizeof(uint32_t);

  auto [host_workspace_size,
        gpu_workspace_size,
        host_workspace_size_fixed,
        gpu_workspace_size_fixed] =
    cuvs::neighbors::cagra::helpers::optimize_workspace_size(
      n_rows, graph_degree, intermediate_graph_degree, sizeof(uint32_t), guarantee_connectivity);

  size_t kmeans_trainset_ratio = std::max<size_t>(
    1,
    n_rows / std::max<size_t>(params.build_params.kmeans_trainset_fraction * n_rows,
                              params.build_params.n_lists));
  size_t kmeans_n_rows  = n_rows / kmeans_trainset_ratio;
  size_t kmeans_gpu_mem = kmeans_n_rows * dim * sizeof(float);
  if (dtype != CUDA_R_32F) {
    // kmeans trainset tmp allocation
    kmeans_gpu_mem += kmeans_n_rows * dim * dtype_size;
  }

  // For non-float input, ivf_pq::build first samples into a temporary trainset of type T
  if (!input_is_float) { kmeans_gpu_mem += kmeans_n_rows * dim * dtype_size; }

  // Trainset sampling (raft::matrix::sample_rows, raft::matrix::detail::gather)
  size_t kmeans_indices_host          = kmeans_n_rows * sizeof(int64_t);
  constexpr size_t kGatherBufferElems = 32768ul * 1024ul;  // matches raft gather buffer_size
  size_t pinned_rows =
    std::min<size_t>(kmeans_n_rows, kGatherBufferElems / std::max<size_t>(dim, 1));
  size_t kmeans_pinned_host = 2 * pinned_rows * dim * dtype_size;  // two staging double-buffers
  size_t kmeans_host_mem    = kmeans_indices_host + kmeans_pinned_host;

  // Add graph to index on GPU
  size_t create_index_gpu_mem = n_rows * graph_degree * sizeof(uint32_t);

  // Note: We omit attached dataset size since we have a fallback path when its allocation fails

  // Search phase (build_knn_graph):
  constexpr size_t kWorkspaceRatio = 5;
  size_t top_k                     = intermediate_graph_degree + 1;
  size_t gpu_top_k   = static_cast<size_t>(intermediate_graph_degree * params.refinement_rate);
  gpu_top_k          = std::min<size_t>(std::max<size_t>(gpu_top_k, top_k), n_rows);
  size_t max_queries = params.search_params.max_internal_batch_size;
  size_t search_io_dev =
    max_queries * (dtype_size * dim                                 // query batch
                   + (sizeof(float) + sizeof(int64_t)) * gpu_top_k  // distances + neighbors
                   + (sizeof(float) + sizeof(int64_t)) * top_k);    // refined distances + neighbors
  size_t search_phase_dev = dataset_gpu_mem + kWorkspaceRatio * search_io_dev;

  // Host-side I/O buffers for the search phase (mirrors build_knn_graph<IVF-PQ>).
  size_t search_io_host = max_queries * (dtype_size * dim               // queries_host
                                         + sizeof(int64_t) * gpu_top_k  // neighbors_host
                                         + (sizeof(float) + sizeof(int64_t)) * top_k);  // refined_*

  // Phases run sequentially (train/extend -> search -> optimize)
  size_t total_dev =
    std::max({kmeans_gpu_mem, search_phase_dev, gpu_workspace_size, create_index_gpu_mem});

  // The graph (and its optimize workspace) stays resident across phases
  size_t total_host =
    graph_host_mem + host_workspace_size + std::max(kmeans_host_mem, search_io_host);

  return std::make_pair(total_host, total_dev);
}

// All sizes are in bytes
inline std::pair<size_t, size_t> nn_descent_build_mem_usage(raft::resources const& res,
                                                            raft::matrix_extent<int64_t> dataset,
                                                            size_t graph_degree,
                                                            size_t intermediate_graph_degree,
                                                            bool guarantee_connectivity)
{
  auto [nnd_host, nnd_dev] = cuvs::neighbors::nn_descent::build_mem_usage(
    res, dataset, intermediate_graph_degree, sizeof(uint32_t));

  auto [host_workspace_size, gpu_workspace_size, host_ws_fixed, gpu_ws_fixed] =
    cuvs::neighbors::cagra::helpers::optimize_workspace_size(dataset.extent(0),
                                                             graph_degree,
                                                             intermediate_graph_degree,
                                                             sizeof(uint32_t),
                                                             guarantee_connectivity);

  size_t graph_host_mem =
    dataset.extent(0) * (graph_degree + intermediate_graph_degree) * sizeof(uint32_t);

  size_t total_host = nnd_host + graph_host_mem + host_workspace_size;
  size_t total_dev  = std::max(nnd_dev, gpu_workspace_size);
  return std::make_pair(total_host, total_dev);
}

std::pair<size_t, size_t> cagra_build_mem_usage(raft::resources const& res,
                                                raft::matrix_extent<int64_t> dataset,
                                                cudaDataType_t dtype,
                                                cuvs::neighbors::cagra::index_params cparams)
{
  using namespace cuvs::neighbors;

  size_t total_host = 0;
  size_t total_dev  = 0;

  if (std::holds_alternative<graph_build_params::ivf_pq_params>(cparams.graph_build_params)) {
    RAFT_LOG_INFO("Considering CAGRA in memory build with IVF-PQ");
    graph_build_params::ivf_pq_params pq_params =
      std::get<graph_build_params::ivf_pq_params>(cparams.graph_build_params);
    std::tie(total_host, total_dev) = ivf_pq_build_mem_usage(res,
                                                             dataset,
                                                             dtype,
                                                             pq_params,
                                                             cparams.graph_degree,
                                                             cparams.intermediate_graph_degree,
                                                             cparams.guarantee_connectivity);
  } else if (std::holds_alternative<graph_build_params::nn_descent_params>(
               cparams.graph_build_params)) {
    RAFT_LOG_INFO("Considering CAGRA in memory build with NN-descent");
    std::tie(total_host, total_dev) = nn_descent_build_mem_usage(res,
                                                                 dataset,
                                                                 cparams.graph_degree,
                                                                 cparams.intermediate_graph_degree,
                                                                 cparams.guarantee_connectivity);
  } else {
    // iterative build
    // TODO(tfeher): proper estimate
    total_host = dataset.extent(0) * dataset.extent(1) * cuda_data_type_size(dtype) +
                 dataset.extent(0) * (cparams.graph_degree + cparams.intermediate_graph_degree) *
                   sizeof(uint32_t);
    total_dev = total_host;
  }

  size_t extra_gpu_workspace_size = raft::resource::get_workspace_total_bytes(res);
  return std::make_pair(total_host + static_cast<size_t>(1e9),
                        total_dev + extra_gpu_workspace_size);
}

}  // namespace cuvs::neighbors::cagra::helpers
