/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <optional>
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
                                                                   bool mst_optimize,
                                                                   bool device_resident_graphs)
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
  size_t prune_dev_fixed = batch_size * intermediate_degree;  // detour count (uint8_t)
  prune_dev_fixed += batch_size * sizeof(uint32_t);           // d_num_detour_edges

  // Buffers that only exist to stage host-resident graphs to the device. When the caller already
  // owns device graphs, batch_load_iterator passes them through and the kernels read and write
  // them in place.
  size_t prune_dev = 0;
  if (!device_resident_graphs) {
    prune_dev_fixed += 2 * batch_size * graph_degree * index_size;  // d_output_graph(2*batch)
    prune_dev += n_rows * intermediate_degree * index_size;         // d_input_graph
  }
  prune_dev += prune_dev_fixed;

  // Reverse graph stage memory
  size_t rev_dev = n_rows * graph_degree * index_size;  // d_rev_graph
  rev_dev += n_rows * sizeof(uint32_t);                 // d_rev_graph_count
  if (!device_resident_graphs) {
    rev_dev += n_rows * index_size;  // d_dest_nodes
  }

  // Memory for merging graphs (host only optional)
  size_t combine_host_fixed = graph_degree * sizeof(uint32_t);  // histogram
  size_t combine_host       = n_rows * sizeof(uint32_t);        // n_edge_count
  combine_host += combine_host_fixed;

  // additional memory for combine stage on device (3 batches)
  size_t combine_dev_fixed = 0;
  if (!device_resident_graphs) {
    combine_dev_fixed += 2 * batch_size * graph_degree * index_size;  // d_output_graph(2*batch)
  }
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

size_t vpq_dataset_size(raft::matrix_extent<int64_t> dataset,
                        cuvs::neighbors::vpq_params params,
                        size_t codebook_element_size)
{
  const size_t n_rows = dataset.extent(0);
  const size_t dim    = dataset.extent(1);

  // Mirror detail::fill_missing_params_heuristics for the fields that affect the footprint.
  const size_t pq_bits = params.pq_bits == 0 ? 8 : params.pq_bits;
  const size_t pq_dim =
    params.pq_dim == 0 ? raft::div_rounding_up_safe(dim, size_t{4}) : params.pq_dim;
  const size_t vq_n_centers =
    params.vq_n_centers == 0
      ? raft::round_up_safe<size_t>(static_cast<size_t>(std::sqrt(static_cast<double>(n_rows))), 8)
      : params.vq_n_centers;

  const size_t pq_len       = raft::div_rounding_up_safe(dim, pq_dim);
  const size_t pq_n_centers = size_t{1} << pq_bits;

  // Every encoded row starts with its inlined VQ label (vpq_build always requests inline labels)
  // and is followed by the bit-packed PQ codes.
  using label_type            = uint32_t;
  constexpr size_t kLabelBits = 8 * sizeof(label_type);
  const size_t encoded_row_length =
    sizeof(label_type) * (1 + raft::div_rounding_up_safe(pq_dim * pq_bits, kLabelBits));

  return vq_n_centers * dim * codebook_element_size       // vq_code_book
         + pq_n_centers * pq_len * codebook_element_size  // pq_code_book
         + n_rows * encoded_row_length;                   // encoded rows
}

namespace {

/**
 * Device memory held by a single CAGRA search plan, in bytes.
 *
 * Mirrors `search_plan_impl_base` (algorithm selection), `search_plan_impl::adjust_search_params`
 * and `search_plan_impl::calc_hashmap_params` together with the per-algorithm buffers of
 * detail/cagra/search_single_cta.cuh and detail/cagra/search_multi_cta.cuh.
 */
size_t search_plan_mem_usage(cuvs::neighbors::cagra::search_params params,
                             size_t max_queries,
                             size_t itopk_size,
                             size_t graph_degree,
                             size_t dataset_size,
                             size_t index_size)
{
  // The iterative build always searches with max_queries = kIterativeBuildChunkSize, which is far
  // above the occupancy threshold of the AUTO heuristic, so the algorithm is decided by itopk
  // alone.
  const bool multi_cta =
    params.algo == cuvs::neighbors::cagra::search_algo::MULTI_CTA ||
    (params.algo == cuvs::neighbors::cagra::search_algo::AUTO && itopk_size > 512);

  constexpr size_t kMultiCtaItopkSize = 32;

  const size_t search_width = std::max<size_t>(1, params.search_width);

  size_t max_iterations = params.max_iterations;
  if (params.max_iterations == 0) {
    max_iterations         = multi_cta ? kMultiCtaItopkSize : itopk_size / search_width;
    size_t reachable_nodes = 1;
    while (reachable_nodes < dataset_size) {
      reachable_nodes *= std::max<size_t>(2, graph_degree / 2);
      max_iterations += 1;
    }
  }
  if (params.max_iterations < params.min_iterations) { max_iterations = params.min_iterations; }
  max_iterations = std::max(max_iterations, params.max_iterations);

  // The internal topk is rounded up to a multiple of 32.
  if (itopk_size % 32 != 0) { itopk_size += 32 - (itopk_size % 32); }

  // Smallest hash table that keeps the expected number of entries under the maximum fill rate.
  auto hash_bitlen = [&](size_t min_bitlen, size_t expected_nodes) {
    size_t bitlen = std::max<size_t>(min_bitlen, params.hashmap_min_bitlen);
    while (static_cast<double>(expected_nodes) >
           static_cast<double>(size_t{1} << bitlen) * params.hashmap_max_fill_rate) {
      bitlen += 1;
    }
    return bitlen;
  };

  // num_executed_iterations, allocated for every non-persistent plan.
  size_t dev = max_queries * sizeof(uint32_t);

  if (!multi_cta) {
    // AUTO and SMALL hash modes keep the visited-node table in shared memory, so nothing is
    // allocated globally. AUTO falls back to a global table once the small one would exceed 8K
    // entries.
    if (params.hashmap_mode != cuvs::neighbors::cagra::hash_mode::HASH) {
      const size_t small_bitlen = hash_bitlen(8, itopk_size + search_width * graph_degree);
      if (small_bitlen <= 13) { return dev; }
    }
    const size_t bitlen =
      hash_bitlen(11, itopk_size + search_width * graph_degree * max_iterations);
    return dev + index_size * max_queries * (size_t{1} << bitlen);  // hashmap
  }

  // Multi-CTA keeps the per-CTA visited table in shared memory but shares the traversed-node
  // table across the CTAs of a query, so that one is always global.
  const size_t num_cta_per_query =
    std::max(search_width, raft::div_rounding_up_safe(itopk_size, kMultiCtaItopkSize));
  const size_t num_intermediate = num_cta_per_query * kMultiCtaItopkSize;
  const size_t bitlen =
    hash_bitlen(11, num_cta_per_query * std::max(kMultiCtaItopkSize, max_iterations));

  dev += index_size * max_queries * (size_t{1} << bitlen);  // hashmap
  // intermediate_indices / intermediate_distances
  dev += num_intermediate * max_queries * (index_size + sizeof(float));
  // topk_workspace: one state byte per 8 candidates per thread of a 1024-thread block
  // (_cuann_find_topk_bufferSize).
  constexpr size_t kTopkThreads   = 1024;
  constexpr size_t kTopkStateBits = 8;
  dev += raft::div_rounding_up_safe(
           raft::div_rounding_up_safe(num_intermediate, kTopkThreads), kTopkStateBits) *
         kTopkThreads * max_queries;
  return dev;
}

}  // namespace

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

// All sizes are in bytes
inline std::pair<size_t, size_t> iterative_build_mem_usage(
  raft::matrix_extent<int64_t> dataset,
  cudaDataType_t dtype,
  cuvs::neighbors::graph_build_params::iterative_search_params params,
  size_t graph_degree,
  size_t intermediate_graph_degree,
  bool guarantee_connectivity,
  std::optional<cuvs::neighbors::vpq_params> compression)
{
  // Mirrors detail::iterative_build_graph and detail::search_and_optimize. The build grows the
  // graph by repeatedly searching the graph it has so far and optimizing the result; every
  // allocation peaks on the final iteration, where the query set, the kNN graph and the output
  // graph all span the whole dataset.
  constexpr size_t kIndexSize = sizeof(uint32_t);  // IdxT

  const size_t n_rows     = dataset.extent(0);
  const size_t dim        = dataset.extent(1);
  const size_t chunk      = kIterativeBuildChunkSize;
  const size_t dtype_size = cuda_data_type_size(dtype);
  // The search may return the query node itself, hence the extra column.
  const size_t topk = intermediate_graph_degree + 1;

  // The dataset stays resident on the device for the whole build: either VPQ-compressed, or padded
  // to CAGRA's row alignment.
  size_t dataset_dev;
  size_t query_scratch;
  if (compression.has_value()) {
    dataset_dev = vpq_dataset_size(dataset, compression.value());
    // Queries are reconstructed from the codes one chunk at a time rather than materialized for
    // the whole dataset.
    query_scratch = chunk * dim * dtype_size;
  } else {
    const size_t stride =
      cuvs::neighbors::cagra_required_row_width(static_cast<uint32_t>(dim), dtype_size);
    dataset_dev = n_rows * stride * dtype_size;
    // Padded rows are depadded into a per-chunk scratch buffer before being used as queries.
    query_scratch = stride == dim ? 0 : chunk * dim * dtype_size;
  }

  // Search results for one chunk, live for the whole loop.
  size_t results_dev = chunk * topk * kIndexSize;  // dev_neighbors
  results_dev += chunk * topk * sizeof(float);     // dev_distances

  // The graph produced by the previous iteration is still alive while the current search fills the
  // kNN graph, and the kNN graph is still alive while optimize writes the output graph. So one
  // full-size graph and one full-size kNN graph always coexist. search_and_optimize releases the
  // previous graph before allocating the output graph, so a third full-size buffer never appears.
  const size_t graph_dev = n_rows * graph_degree * kIndexSize;  // dev_graph / dev_output_graph
  const size_t knn_dev   = n_rows * topk * kIndexSize;          // dev_knn_graph

  // The final iteration searches a graph_degree graph, requests topk neighbors and derives its
  // internal topk from that.
  cuvs::neighbors::cagra::search_params search_params = params;
  search_params.max_queries                           = chunk;
  const size_t search_dev =
    search_plan_mem_usage(search_params, chunk, topk + 32, graph_degree, n_rows, kIndexSize);

  auto [host_workspace_size, gpu_workspace_size, host_ws_fixed, gpu_ws_fixed] =
    optimize_workspace_size(n_rows,
                            graph_degree,
                            std::max(topk, graph_degree),
                            kIndexSize,
                            guarantee_connectivity,
                            /* device_resident_graphs = */ true);

  // Searching and optimizing run sequentially within an iteration, so the search plan and the
  // optimize workspace never coexist and are combined with max() rather than summed. The query
  // scratch is not part of that: search_and_optimize holds it at function scope, so it is still
  // alive while optimize runs.
  //
  // Two transients are left out. The dataset copy made by make_device_padded_dataset briefly
  // coexists with its source, and cagra::search re-pads a query chunk when its rows are not
  // CAGRA-aligned; both are caller-owned or chunk-sized, and counting them would inflate the
  // estimate enough to push callers to an out-of-core build unnecessarily.
  size_t total_dev = dataset_dev + results_dev + graph_dev + knn_dev + query_scratch +
                     std::max(search_dev, gpu_workspace_size);

  // On the host the optimize workspace of the last iteration and the returned graph are also
  // sequential: the workspace is released before the device graph is copied back.
  size_t total_host = std::max(host_workspace_size, n_rows * graph_degree * kIndexSize);

  return std::make_pair(total_host, total_dev);
}

std::pair<size_t, size_t> cagra_build_mem_usage(
  raft::resources const& res,
  raft::matrix_extent<int64_t> dataset,
  cudaDataType_t dtype,
  cuvs::neighbors::cagra::index_params cparams,
  std::optional<cuvs::neighbors::vpq_params> compression)
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
  } else if (std::holds_alternative<graph_build_params::iterative_search_params>(
               cparams.graph_build_params)) {
    RAFT_LOG_INFO("Considering CAGRA in memory build with iterative CAGRA search");
    std::tie(total_host, total_dev) = iterative_build_mem_usage(
      dataset,
      dtype,
      std::get<graph_build_params::iterative_search_params>(cparams.graph_build_params),
      cparams.graph_degree,
      cparams.intermediate_graph_degree,
      cparams.guarantee_connectivity,
      compression);
  } else {
    // No graph build algorithm selected yet (std::monostate) or an out-of-core (ACE) build, whose
    // requirements are modelled by check_ace_memory_requirements instead. Fall back to the size of
    // the dataset plus the graphs.
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
