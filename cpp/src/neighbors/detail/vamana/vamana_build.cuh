/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../sparse/neighbors/cross_component_nn.cuh"
#include "../../detail/ann_utils.cuh"
#include "greedy_search.cuh"
#include "robust_prune.cuh"
#include "vamana_structs.cuh"
#include <cuvs/neighbors/vamana.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/init.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>

#include <cuvs/distance/distance.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::vamana::detail {

/* @defgroup vamana_build_detail vamana build
 * @{
 */

static const int blockD    = 32;
static const int maxBlocks = 10000;

// generate random permutation of inserts - TODO do this on GPU / faster
template <typename IdxT>
void create_insert_permutation(std::vector<IdxT>& insert_order, uint32_t N)
{
  insert_order.resize(N);
  for (uint32_t i = 0; i < N; i++) {
    insert_order[i] = (IdxT)i;
  }
  for (uint32_t i = 0; i < N; i++) {
    uint32_t temp;
    uint32_t rand_idx      = rand() % N;
    temp                   = insert_order[i];
    insert_order[i]        = insert_order[rand_idx];
    insert_order[rand_idx] = temp;
  }
}

template <typename IdxT>
__global__ void print_mtx(raft::device_vector_view<IdxT, int64_t> vec)
{
  printf("extents:%ld\n", vec.extent(0));
  for (int i = 0; i < vec.extent(0); i++) {
    printf("%d, ", vec(i));
  }
  printf("\n");
}

template <typename IdxT, typename accT>
__global__ void print_queryIds(void* query_list_ptr)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = 0; i < 50; i++) {
    printf("queryId:%d\n", query_list[i].queryId);
  }
}

#define KERNEL_TIMING (RAFT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_DEBUG)

/********************************************************************************************
 * Main Vamana building function - insert vectors into empty graph in batches
 * Pre - dataset contains the vector data, host matrix allocated to store the graph
 * Post - graph matrix contains the graph edges of the final Vamana graph
 *******************************************************************************************/
template <typename T,
          typename accT,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
void batched_insert_vamana(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t> graph,
  IdxT* medoid_id,
  cuvs::distance::DistanceType metric)
{
  auto stream = raft::resource::get_cuda_stream(res);
  int N       = dataset.extent(0);
  int dim     = dataset.extent(1);
  int degree  = graph.extent(1);

  // Algorithm params
  int max_batchsize = (int)(params.max_fraction * (float)N);

  if (max_batchsize > (int)dataset.extent(0)) {
    RAFT_LOG_WARN(
      "Max fraction is the fraction of the total dataset, so it cannot be larger 1.0, reducing it "
      "to 1.0");
    max_batchsize = (int)dataset.extent(0);
  }
  float insert_iters = (float)(params.vamana_iters);
  double base        = (double)(params.batch_base);
  float alpha        = (float)(params.alpha);
  int visited_size   = params.visited_size;
  int queue_size     = params.queue_size;
  int reverse_batch  = params.reverse_batchsize;

  if ((visited_size & (visited_size - 1)) != 0) {
    RAFT_LOG_WARN("visited_size must be a power of 2, rounding up.");
    int power = params.graph_degree;
    while (power < visited_size)
      power <<= 1;
    visited_size = power;
  }

#if KERNEL_TIMING
  auto start_t = std::chrono::system_clock::now();
#endif

  // Initialize graph with invalid neighbor indices (raft::upper_bound<IdxT>()).
  auto d_graph = raft::make_device_matrix<IdxT, int64_t>(res, graph.extent(0), graph.extent(1));
  raft::linalg::map(res, d_graph.view(), raft::const_op<IdxT>{raft::upper_bound<IdxT>()});

  // Temp storage about each batch of inserts being performed
  auto query_ids      = raft::make_device_vector<IdxT>(res, max_batchsize);
  auto query_list_ptr = raft::make_device_mdarray<QueryCandidates<IdxT, accT>>(
    res,
    raft::resource::get_large_workspace_resource_ref(res),
    raft::make_extents<int64_t>(max_batchsize + 1));
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr.data_handle());

  // Results of each batch of inserts during build - Memory is used by query_list structure
  auto visited_ids =
    raft::make_device_mdarray<IdxT>(res,
                                    raft::resource::get_large_workspace_resource_ref(res),
                                    raft::make_extents<int64_t>(max_batchsize, visited_size));
  auto visited_dists =
    raft::make_device_mdarray<accT>(res,
                                    raft::resource::get_large_workspace_resource_ref(res),
                                    raft::make_extents<int64_t>(max_batchsize, visited_size));

  // Assign memory to query_list structures and initialize
  init_query_candidate_list<IdxT, accT><<<256, blockD, 0, stream>>>(query_list,
                                                                    visited_ids.data_handle(),
                                                                    visited_dists.data_handle(),
                                                                    (int)max_batchsize,
                                                                    visited_size,
                                                                    1);
  auto topk_pq_mem =
    raft::make_device_mdarray<Node<accT>>(res,
                                          raft::resource::get_large_workspace_resource_ref(res),
                                          raft::make_extents<int64_t>(max_batchsize, visited_size));

  int align_padding = raft::alignTo(dim, 16) - dim;

  auto s_coords_mem = raft::make_device_mdarray<T>(
    res,
    raft::resource::get_large_workspace_resource_ref(res),
    raft::make_extents<int64_t>(min(maxBlocks, max(max_batchsize, reverse_batch)),
                                dim + align_padding));

  // Create random permutation for order of node inserts into graph
  std::vector<IdxT> insert_order;
  create_insert_permutation<IdxT>(insert_order, (uint32_t)N);

  // Calculate the shared memory sizes of each kernel
  int sort_smem_size = 0;
  SELECT_SORT_SMEM_SIZE(degree, visited_size);  // Sets sort_smem_size based on dataset

  // Total dynamic shared memory used by GreedySearch
  int search_smem_total_size =
    static_cast<int>((dim + align_padding) * sizeof(T) +  // visited_size * sizeof(Node<accT>) +
                     degree * sizeof(int) + queue_size * sizeof(DistPair<IdxT, accT>));

  // Total dynamic shared memory size needed by both RobustPrune calls
  int prune_smem_total_size = (degree + visited_size) * sizeof(float) +  // Occlusion list
                              (degree + visited_size) * sizeof(DistPair<IdxT, accT>);

  RAFT_LOG_DEBUG(
    "Dynamic shared memory usage (bytes): GreedySearch: %d, Segment Sort: %d, Robust Prune: %d",
    search_smem_total_size,
    sort_smem_size,
    prune_smem_total_size);

#if KERNEL_TIMING
  auto end_t                                    = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_t - start_t;
  double alloc_time                             = elapsed_seconds.count();

  double search_time       = 0.0;
  double segment_sort_time = 0.0;
  double prune1_time       = 0.0;
  double write1_time       = 0.0;
  double rev_time          = 0.0;
  double batch_prune       = 0.0;
#endif

  // Random medoid has minor impact on recall
  // TODO: use heuristic for better medoid selection, issue:
  // https://github.com/rapidsai/cuvs/issues/355
  *medoid_id = rand() % N;

  // size of current batch of inserts, increases logarithmically until max_batchsize
  int step_size = 1;
  // Loop through batches and call the insert and prune kernels - can insert > N times based on
  // iters parameter
  for (int start = 0; start < (int)(insert_iters * (float)N);) {
#if KERNEL_TIMING
    start_t = std::chrono::system_clock::now();
#endif

    if (start + step_size > (int)(insert_iters * (float)N)) {
      step_size = (int)(insert_iters * (float)N) - start;
    }
    if (start + step_size > N) { step_size = N - start; }
    RAFT_LOG_DEBUG("Starting batch of inserts indices_start:%d, batch_size:%d", start, step_size);

    int num_blocks = min(maxBlocks, step_size);

    // Copy ids to be inserted for this batch
    raft::copy(
      res,
      raft::make_device_vector_view(query_ids.data_handle(), int64_t(step_size)),
      raft::make_host_vector_view<const IdxT>(insert_order.data() + start, int64_t(step_size)));
    set_query_ids<IdxT, accT><<<num_blocks, blockD, 0, stream>>>(
      query_list_ptr.data_handle(), query_ids.data_handle(), step_size);

    // Call greedy search to get candidates for every vector being inserted
    GreedySearchKernel<T, accT, IdxT, Accessor>
      <<<num_blocks, blockD, search_smem_total_size, stream>>>(d_graph.view(),
                                                               dataset,
                                                               query_list_ptr.data_handle(),
                                                               step_size,
                                                               *medoid_id,
                                                               visited_size,
                                                               metric,
                                                               queue_size,
                                                               topk_pq_mem.data_handle());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

#if KERNEL_TIMING
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    end_t           = std::chrono::system_clock::now();
    elapsed_seconds = end_t - start_t;
    search_time += elapsed_seconds.count();
    start_t = std::chrono::system_clock::now();
#endif

    // Segmented sort on query list
    SortPairsKernel<T, accT, IdxT><<<num_blocks, blockD, sort_smem_size, stream>>>(
      query_list_ptr.data_handle(), step_size, visited_size);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

#if KERNEL_TIMING
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    end_t           = std::chrono::system_clock::now();
    elapsed_seconds = end_t - start_t;
    segment_sort_time += elapsed_seconds.count();
    start_t = std::chrono::system_clock::now();
#endif

    // Run on candidates of vectors being inserted
    RobustPruneKernel<T, accT, IdxT>
      <<<num_blocks, blockD, prune_smem_total_size, stream>>>(d_graph.view(),
                                                              dataset,
                                                              query_list_ptr.data_handle(),
                                                              step_size,
                                                              visited_size,
                                                              metric,
                                                              alpha,
                                                              s_coords_mem.data_handle());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // Segmented sort on query list
    SortPairsKernel<T, accT, IdxT><<<num_blocks, blockD, sort_smem_size, stream>>>(
      query_list_ptr.data_handle(), step_size, degree);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

#if KERNEL_TIMING
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    end_t           = std::chrono::system_clock::now();
    elapsed_seconds = end_t - start_t;
    prune1_time += elapsed_seconds.count();
    start_t = std::chrono::system_clock::now();
#endif

    // Write results from first prune to graph edge list
    write_graph_edges_kernel<accT, IdxT><<<num_blocks, blockD, 0, stream>>>(
      d_graph.view(), query_list_ptr.data_handle(), degree, step_size);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

#if KERNEL_TIMING
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    end_t           = std::chrono::system_clock::now();
    elapsed_seconds = end_t - start_t;
    write1_time += elapsed_seconds.count();
    start_t = std::chrono::system_clock::now();
#endif

    // compute prefix sums of query_list sizes - TODO parallelize prefix sums
    //    auto d_total_edges = raft::make_device_mdarray<int>(
    //      res, raft::resource::get_workspace_resource_ref(res), raft::make_extents<int64_t>(1));
    rmm::device_scalar<int> d_total_edges(stream);
    prefix_sums_sizes<accT, IdxT><<<1, 1, 0, stream>>>(query_list, step_size, d_total_edges.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    int total_edges = d_total_edges.value(stream);
    //    raft::copy(&total_edges, d_total_edges.data_handle(), 1, stream);
    //    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    auto edge_dist_pair = raft::make_device_mdarray<DistPair<IdxT, accT>>(
      res,
      raft::resource::get_large_workspace_resource_ref(res),
      raft::make_extents<int64_t>(total_edges));

    auto edge_dest =
      raft::make_device_mdarray<IdxT>(res,
                                      raft::resource::get_large_workspace_resource_ref(res),
                                      raft::make_extents<int64_t>(total_edges));
    auto edge_src =
      raft::make_device_mdarray<IdxT>(res,
                                      raft::resource::get_large_workspace_resource_ref(res),
                                      raft::make_extents<int64_t>(total_edges));

    // Create reverse edge list
    create_reverse_edge_list<accT, IdxT>
      <<<num_blocks, blockD, 0, stream>>>(query_list_ptr.data_handle(),
                                          step_size,
                                          degree,
                                          edge_src.data_handle(),
                                          edge_dist_pair.data_handle());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    {
      // Sort by dists first so final edge lists are each sorted by dist
      void* d_temp_storage      = nullptr;
      size_t temp_storage_bytes = 0;

      cub::DeviceMergeSort::SortPairs(d_temp_storage,
                                      temp_storage_bytes,
                                      edge_dist_pair.data_handle(),
                                      edge_src.data_handle(),
                                      total_edges,
                                      CmpDist<IdxT, accT>(),
                                      stream);

      RAFT_LOG_DEBUG("Temp storage needed for sorting dist (bytes): %lu", temp_storage_bytes);

      auto temp_sort_storage = raft::make_device_mdarray<IdxT>(
        res,
        raft::resource::get_large_workspace_resource_ref(res),
        raft::make_extents<int64_t>(temp_storage_bytes / sizeof(IdxT)));

      // Sort to group reverse edges by destination
      cub::DeviceMergeSort::SortPairs(temp_sort_storage.data_handle(),
                                      temp_storage_bytes,
                                      edge_dist_pair.data_handle(),
                                      edge_src.data_handle(),
                                      total_edges,
                                      CmpDist<IdxT, accT>(),
                                      stream);
    }

    /*
    DistPair<IdxT, accT>* temp_ptr = edge_dist_pair.data_handle();
    raft::linalg::map_offset(
      res, edge_dest.view(), [temp_ptr] __device__(size_t i) { return temp_ptr[i].idx; });
      */
    raft::linalg::map(
      res,
      edge_dest.view(),
      [] __device__(auto x) { return x.idx; },
      raft::make_const_mdspan(edge_dist_pair.view()));

    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceMergeSort::SortPairs(d_temp_storage,
                                    temp_storage_bytes,
                                    edge_dest.data_handle(),
                                    edge_src.data_handle(),
                                    total_edges,
                                    CmpEdge<IdxT>(),
                                    stream);

    RAFT_LOG_DEBUG("Temp storage needed for sorting (bytes): %lu", temp_storage_bytes);

    auto temp_sort_storage = raft::make_device_mdarray<IdxT>(
      res,
      raft::resource::get_large_workspace_resource_ref(res),
      raft::make_extents<int64_t>(temp_storage_bytes / sizeof(IdxT)));

    // Sort to group reverse edges by destination
    cub::DeviceMergeSort::SortPairs(temp_sort_storage.data_handle(),
                                    temp_storage_bytes,
                                    edge_dest.data_handle(),
                                    edge_src.data_handle(),
                                    total_edges,
                                    CmpEdge<IdxT>(),
                                    stream);

    // Get number of unique node destinations
    IdxT unique_dests =
      cuvs::sparse::neighbors::get_n_components(edge_dest.data_handle(), total_edges, stream);

    // Find which node IDs have reverse edges and their indices in the reverse edge list
    thrust::device_vector<IdxT> edge_dest_vec(edge_dest.data_handle(),
                                              edge_dest.data_handle() + total_edges);
    auto unique_indices = raft::make_device_vector<int>(res, total_edges);
    raft::linalg::map_offset(res, unique_indices.view(), raft::identity_op{});

    thrust::unique_by_key(edge_dest_vec.begin(), edge_dest_vec.end(), unique_indices.data_handle());

    edge_dest_vec.clear();
    edge_dest_vec.shrink_to_fit();

#if KERNEL_TIMING
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    end_t           = std::chrono::system_clock::now();
    elapsed_seconds = end_t - start_t;
    rev_time += elapsed_seconds.count();
    start_t = std::chrono::system_clock::now();
#endif

    // Batch execution of reverse edge creation/application
    reverse_batch = params.reverse_batchsize;
    for (int rev_start = 0; rev_start < (int)unique_dests; rev_start += reverse_batch) {
      if (rev_start + reverse_batch > (int)unique_dests) {
        reverse_batch = (int)unique_dests - rev_start;
      }

      // Allocate reverse QueryCandidate list based on number of unique destinations
      auto reverse_list_ptr = raft::make_device_mdarray<QueryCandidates<IdxT, accT>>(
        res,
        raft::resource::get_large_workspace_resource_ref(res),
        raft::make_extents<int64_t>(reverse_batch));
      auto rev_ids =
        raft::make_device_mdarray<IdxT>(res,
                                        raft::resource::get_large_workspace_resource_ref(res),
                                        raft::make_extents<int64_t>(reverse_batch, visited_size));

      auto rev_dists =
        raft::make_device_mdarray<accT>(res,
                                        raft::resource::get_large_workspace_resource_ref(res),
                                        raft::make_extents<int64_t>(reverse_batch, visited_size));

      QueryCandidates<IdxT, accT>* reverse_list =
        static_cast<QueryCandidates<IdxT, accT>*>(reverse_list_ptr.data_handle());

      init_query_candidate_list<IdxT, accT><<<256, blockD, 0, stream>>>(reverse_list,
                                                                        rev_ids.data_handle(),
                                                                        rev_dists.data_handle(),
                                                                        (int)reverse_batch,
                                                                        visited_size);

      // May need more blocks for reverse list
      num_blocks = min(maxBlocks, reverse_batch);

      // Populate reverse list ids and candidate lists from edge_src and edge_dest
      populate_reverse_list_struct<T, accT, IdxT>
        <<<num_blocks, blockD, 0, stream>>>(reverse_list,
                                            edge_src.data_handle(),
                                            edge_dest.data_handle(),
                                            unique_indices.data_handle(),
                                            unique_dests,
                                            total_edges,
                                            dataset.extent(0),
                                            rev_start,
                                            reverse_batch);
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      // Recompute distances (avoided keeping it during sorting)
      recompute_reverse_dists<T, accT, IdxT>
        <<<num_blocks, blockD, 0, stream>>>(reverse_list, dataset, reverse_batch, metric);
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      // Call 2nd RobustPrune on reverse query_list
      RobustPruneKernel<T, accT, IdxT>
        <<<num_blocks, blockD, prune_smem_total_size, stream>>>(d_graph.view(),
                                                                raft::make_const_mdspan(dataset),
                                                                reverse_list_ptr.data_handle(),
                                                                reverse_batch,
                                                                visited_size,
                                                                metric,
                                                                alpha,
                                                                s_coords_mem.data_handle());
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      // Segmented sort on reverse_list
      SortPairsKernel<T, accT, IdxT><<<num_blocks, blockD, sort_smem_size, stream>>>(
        reverse_list_ptr.data_handle(), reverse_batch, degree);
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      // Write new edge lists to graph
      write_graph_edges_kernel<accT, IdxT><<<num_blocks, blockD, 0, stream>>>(
        d_graph.view(), reverse_list_ptr.data_handle(), degree, reverse_batch);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

#if KERNEL_TIMING
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    end_t           = std::chrono::system_clock::now();
    elapsed_seconds = end_t - start_t;
    batch_prune += elapsed_seconds.count();
#endif

    start += step_size;
    if (start >= N) {
      start = 0;
      insert_iters -= 1.0;
      step_size = max_batchsize;
    }
    step_size *= base;
    step_size = min(step_size, max_batchsize);

  }  // Batch of inserts

#if KERNEL_TIMING
  printf("intro:%lf\ngreedy:%lf\nseg_sort:%lf\nprune1:%lf\nwrite1:%lf\nrev:%lf\nbatch_prune:%lf\n",
         alloc_time,
         search_time,
         segment_sort_time,
         prune1_time,
         write1_time,
         rev_time,
         batch_prune);
#endif

  raft::copy(res, graph, d_graph.view());

  RAFT_CHECK_CUDA(stream);
}

template <typename T,
          typename IdxT = uint64_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
index<T, IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  uint32_t graph_degree = params.graph_degree;

  RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded,
               "Currently only L2Expanded metric is supported");

  const int* deg_size = std::find(std::begin(DEGREE_SIZES), std::end(DEGREE_SIZES), graph_degree);
  RAFT_EXPECTS(deg_size != std::end(DEGREE_SIZES), "Provided graph_degree not currently supported");

  RAFT_EXPECTS(params.visited_size > graph_degree, "visited_size must be > graph_degree");

  RAFT_EXPECTS(params.vamana_iters >= 1.0,
               "vamana_iters must be at least 1.0 to insert the entire input dataset");

  int dim = dataset.extent(1);

  RAFT_LOG_DEBUG("Creating empty graph structure");
  auto vamana_graph = raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), graph_degree);

  RAFT_LOG_DEBUG("Running Vamana batched insert algorithm");

  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
  IdxT medoid_id;
  batched_insert_vamana<T, float, IdxT, Accessor>(
    res, params, dataset, vamana_graph.view(), &medoid_id, metric);

  std::optional<raft::device_matrix<uint8_t, int64_t, raft::row_major>> quantized_vectors;
  if (params.codebooks) {
    // Full codebook should be a raft::matrix of dimension [2^PQ_BITS * PQ_DIM, VEC_DIM / PQ_DIM]
    // Every row is (VEC_DIM/PQ_DIM) floats representing a group of cluster centroids.
    // Every consecutive [PQ_DIM] rows is a set.

    // short-hand
    auto& codebook_params = params.codebooks.value();
    int pq_codebook_size  = codebook_params.pq_codebook_size;
    int pq_dim            = codebook_params.pq_dim;

    auto pq_params          = cuvs::preprocessing::quantize::pq::params{};
    pq_params.pq_bits       = raft::log2(pq_codebook_size);
    pq_params.pq_dim        = pq_dim;
    pq_params.use_subspaces = true;
    pq_params.use_vq        = false;

    // transform pq_encoding_table (dimensions: pq_codebook_size x dim_per_subspace * pq_dim ) to
    // pq_codebook (dimensions: pq_codebook_size * pq_dim, dim_per_subspace)
    auto pq_encoding_table_device_vec = raft::make_device_vector<float, uint32_t>(
      res,
      codebook_params.pq_encoding_table.size());  // logically a 2D matrix with dimensions
                                                  // pq_codebook_size x dim_per_subspace * pq_dim
    raft::copy(res,
               pq_encoding_table_device_vec.view(),
               raft::make_host_vector_view<const float>(codebook_params.pq_encoding_table.data(),
                                                        pq_encoding_table_device_vec.extent(0)));
    int dim_per_subspace = dim / pq_dim;
    auto pq_codebook =
      raft::make_device_matrix<float, uint32_t>(res, pq_codebook_size * pq_dim, dim_per_subspace);
    auto pq_encoding_table_device_vec_view = pq_encoding_table_device_vec.view();
    raft::linalg::map_offset(
      res,
      pq_codebook.view(),
      [pq_encoding_table_device_vec_view,
       pq_dim,
       pq_codebook_size,
       dim_per_subspace,
       dim] __device__(size_t i) {
        int row_idx        = i / dim_per_subspace;
        int subspace_id    = row_idx / pq_codebook_size;  // idx_pq_dim
        int codebook_id    = row_idx % pq_codebook_size;  // idx_pq_codebook_size
        int id_in_subspace = i % dim_per_subspace;        // idx_dim_per_subspace

        return pq_encoding_table_device_vec_view[codebook_id * pq_dim * dim_per_subspace +
                                                 subspace_id * dim_per_subspace + id_in_subspace];
      });

    // prepare rotation matrix
    auto rotation_matrix_device = raft::make_device_matrix<float, int64_t>(res, dim, dim);
    raft::copy(
      res,
      raft::make_device_vector_view(rotation_matrix_device.data_handle(),
                                    int64_t(codebook_params.rotation_matrix.size())),
      raft::make_host_vector_view<const float>(codebook_params.rotation_matrix.data(),
                                               int64_t(codebook_params.rotation_matrix.size())));

    // process in batches
    const uint32_t n_rows = dataset.extent(0);

    auto quantizer = cuvs::preprocessing::quantize::pq::quantizer<float>(
      pq_params,
      cuvs::neighbors::vpq_dataset<float, int64_t>{
        raft::make_device_matrix<float, uint32_t, raft::row_major>(res, 0, 0),
        std::move(pq_codebook),
        raft::make_device_matrix<uint8_t, int64_t, raft::row_major>(res, 0, 0)});
    const int64_t codes_rowlen = cuvs::preprocessing::quantize::pq::get_quantized_dim(pq_params);
    quantized_vectors =
      raft::make_device_matrix<uint8_t, int64_t, raft::row_major>(res, n_rows, codes_rowlen);
    // TODO: with scaling workspace we could choose the batch size dynamically
    constexpr uint32_t kReasonableMaxBatchSize = 65536;
    const uint32_t max_batch_size              = std::min(n_rows, kReasonableMaxBatchSize);
    auto _vamana_batches = cuvs::spatial::knn::detail::utils::make_batch_load_iterator<T>(
      res,
      dataset.data_handle(),
      static_cast<int64_t>(n_rows),
      static_cast<int64_t>(dim),
      static_cast<size_t>(max_batch_size),
      raft::resource::get_cuda_stream(res),
      raft::resource::get_workspace_resource_ref(res));
    for (const auto& batch : _vamana_batches) {
      // perform rotation
      auto dataset_rotated = raft::make_device_matrix<float, int64_t>(res, batch.size(), dim);
      if constexpr (std::is_same_v<T, float>) {
        auto dataset_view = raft::make_device_matrix_view(const_cast<T*>(batch.data()),
                                                          static_cast<int64_t>(batch.size()),
                                                          static_cast<int64_t>(dim));
        raft::linalg::gemm(
          res, dataset_view, rotation_matrix_device.view(), dataset_rotated.view());
      } else {
        // convert dataset to float
        auto dataset_float = raft::make_device_matrix<float, int64_t>(res, batch.size(), dim);
        auto dataset_view  = raft::make_device_matrix_view(
          batch.data(), static_cast<int64_t>(batch.size()), static_cast<int64_t>(dim));
        raft::linalg::map_offset(
          res, dataset_float.view(), [dataset_view, dim] __device__(size_t i) {
            int row_idx = i / dim;
            int col_idx = i % dim;
            return static_cast<float>(dataset_view(row_idx, col_idx));
          });
        raft::linalg::gemm(
          res, dataset_float.view(), rotation_matrix_device.view(), dataset_rotated.view());
      }

      cuvs::preprocessing::quantize::pq::transform(
        res,
        quantizer,
        raft::make_const_mdspan(dataset_rotated.view()),
        raft::make_device_matrix_view<uint8_t, int64_t>(
          quantized_vectors.value().data_handle() +
            batch.offset() * quantized_vectors.value().extent(1),
          batch.size(),
          quantized_vectors.value().extent(1)));
    }
  }

  try {
    auto idx = index<T, IdxT>(
      res, params.metric, dataset, raft::make_const_mdspan(vamana_graph.view()), medoid_id);
    if (quantized_vectors)
      idx.update_quantized_dataset(res, raft::make_const_mdspan(quantized_vectors.value().view()));
    return idx;
  } catch (std::bad_alloc& e) {
    RAFT_LOG_DEBUG("Insufficient GPU memory to construct VAMANA index with dataset on GPU");
    // We just add the graph. User is expected to update dataset separately (e.g allocating in
    // managed memory).
  } catch (raft::logic_error& e) {
    // The memory error can also manifest as logic_error.
    RAFT_LOG_DEBUG("Insufficient GPU memory to construct VAMANA index with dataset on GPU");
  }
  index<T, IdxT> idx(res, params.metric);
  RAFT_LOG_WARN("Constructor not called, returning empty index");
  return idx;
}

/**
 * @}
 */

}  // namespace cuvs::neighbors::vamana::detail
