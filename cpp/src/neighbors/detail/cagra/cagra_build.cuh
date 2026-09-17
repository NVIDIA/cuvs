/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../core/nvtx.hpp"
#include "../../ivf_pq/ivf_pq_fp16_overflow.cuh"
#include "graph_core.cuh"
#include <cuvs/preprocessing/quantize/pq.hpp>

#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/integer_utils.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <cuvs/neighbors/refine.hpp>
#include <cuvs/util/file_io.hpp>

// TODO: This shouldn't be calling spatial/knn APIs
#include "../ann_utils.cuh"

#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <omp.h>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <sys/mman.h>

namespace cuvs::neighbors::cagra::detail {

template <typename T, typename IdxT>
void check_graph_degree(size_t& intermediate_degree, size_t& graph_degree, size_t dataset_size)
{
  if (intermediate_degree >= static_cast<size_t>(dataset_size)) {
    RAFT_LOG_WARN(
      "Intermediate graph degree cannot be larger than dataset size, reducing it to %lu",
      dataset_size);
    intermediate_degree = dataset_size - 1;
  }
  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }
}

template <typename DataT, typename IdxT, typename accessor>
void build_knn_graph(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cuvs::neighbors::cagra::graph_build_params::ivf_pq_params pq);

// Exact kNN construction shared by ordinary CAGRA and ACE. Request one extra
// neighbor and remove self by ID, including when duplicate vectors tie with self.
template <typename T, typename IdxT, typename Layout, typename Accessor>
void build_knn_graph(raft::resources const& res,
                     raft::mdspan<const T, raft::matrix_extent<int64_t>, Layout, Accessor> dataset,
                     raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
                     graph_build_params::brute_force_params params)
{
  auto const partition_size            = static_cast<size_t>(dataset.extent(0));
  auto const intermediate_graph_degree = static_cast<size_t>(knn_graph.extent(1));
  RAFT_EXPECTS(knn_graph.extent(0) == dataset.extent(0),
               "CAGRA: kNN graph and dataset must have the same number of rows");
  RAFT_EXPECTS(intermediate_graph_degree > 0 && partition_size > intermediate_graph_degree,
               "CAGRA: kNN graph degree must be positive and smaller than the dataset size");
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
    auto device_queries =
      raft::make_device_matrix<T, int64_t>(res, partition_size, dataset.extent(1));
    raft::copy(res, device_queries.view(), dataset);
    auto brute_force_index = cuvs::neighbors::brute_force::build(
      res, params.build_params, raft::make_const_mdspan(device_queries.view()));
    auto device_neighbors = raft::make_device_matrix<int64_t, int64_t>(
      res, partition_size, intermediate_graph_degree + 1);
    auto device_distances =
      raft::make_device_matrix<float, int64_t>(res, partition_size, intermediate_graph_degree + 1);
    cuvs::neighbors::brute_force::search(res,
                                         params.search_params,
                                         brute_force_index,
                                         raft::make_const_mdspan(device_queries.view()),
                                         device_neighbors.view(),
                                         device_distances.view());

    auto host_neighbors =
      raft::make_host_matrix<int64_t, int64_t>(partition_size, intermediate_graph_degree + 1);
    raft::copy(res, host_neighbors.view(), device_neighbors.view());
    raft::resource::sync_stream(res);
    for (size_t row = 0; row < partition_size; ++row) {
      size_t output_rank = 0;
      for (size_t rank = 0; rank < intermediate_graph_degree + 1; ++rank) {
        auto const neighbor = host_neighbors(row, rank);
        if (neighbor == static_cast<int64_t>(row)) { continue; }
        RAFT_EXPECTS(neighbor >= 0 && static_cast<size_t>(neighbor) < partition_size,
                     "CAGRA: brute-force kNN returned an invalid neighbor");
        knn_graph(row, output_rank++) = static_cast<IdxT>(neighbor);
        if (output_rank == intermediate_graph_degree) { break; }
      }
      RAFT_EXPECTS(output_rank == intermediate_graph_degree,
                   "CAGRA: brute-force kNN did not return enough non-self neighbors");
    }
  } else {
    RAFT_FAIL("CAGRA: brute-force kNN build supports only float and half datasets");
  }
}

template <typename IdxT>
void write_to_graph(raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
                    raft::host_matrix_view<int64_t, int64_t, raft::row_major> neighbors_host_view,
                    size_t& num_self_included,
                    size_t batch_size,
                    size_t batch_offset)
{
  uint32_t node_degree = knn_graph.extent(1);
  size_t top_k         = neighbors_host_view.extent(1);
  // omit itself & write out
  for (std::size_t i = 0; i < batch_size; i++) {
    size_t vec_idx = i + batch_offset;
    for (std::size_t j = 0, num_added = 0; j < top_k && num_added < node_degree; j++) {
      const auto v = neighbors_host_view(i, j);
      if (static_cast<size_t>(v) == vec_idx) {
        num_self_included++;
        continue;
      }
      knn_graph(vec_idx, num_added) = v;
      num_added++;
    }
  }
}

template <typename DataT, typename IdxT, typename accessor>
void refine_host_and_write_graph(
  raft::resources const& res,
  raft::host_matrix<DataT, int64_t>& queries_host,
  raft::host_matrix<int64_t, int64_t>& neighbors_host,
  raft::host_matrix<int64_t, int64_t>& refined_neighbors_host,
  raft::host_matrix<float, int64_t>& refined_distances_host,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cuvs::distance::DistanceType metric,
  size_t& num_self_included,
  size_t batch_size,
  size_t batch_offset,
  int top_k,
  int gpu_top_k)
{
  bool do_refine = top_k != gpu_top_k;

  auto refined_neighbors_host_view = raft::make_host_matrix_view<int64_t, int64_t>(
    do_refine ? refined_neighbors_host.data_handle() : neighbors_host.data_handle(),
    batch_size,
    top_k);

  if (do_refine) {
    // needed for compilation as this routine will also be run for device data with !do_refine
    if constexpr (raft::is_host_mdspan_v<decltype(dataset)>) {
      auto queries_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
        queries_host.data_handle(), batch_size, dataset.extent(1));
      auto neighbors_host_view = raft::make_host_matrix_view<const int64_t, int64_t>(
        neighbors_host.data_handle(), batch_size, neighbors_host.extent(1));
      auto refined_distances_host_view = raft::make_host_matrix_view<float, int64_t>(
        refined_distances_host.data_handle(), batch_size, top_k);
      cuvs::neighbors::refine(res,
                              dataset,
                              queries_host_view,
                              neighbors_host_view,
                              refined_neighbors_host_view,
                              refined_distances_host_view,
                              metric);
    }
  }

  write_to_graph(
    knn_graph, refined_neighbors_host_view, num_self_included, batch_size, batch_offset);
}

template <typename DataT, typename IdxT, typename accessor>
void build_knn_graph(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cuvs::neighbors::cagra::graph_build_params::ivf_pq_params pq)
{
  RAFT_EXPECTS(pq.build_params.metric == cuvs::distance::DistanceType::L2Expanded ||
                 pq.build_params.metric == cuvs::distance::DistanceType::InnerProduct ||
                 pq.build_params.metric == cuvs::distance::DistanceType::CosineExpanded,
               "Currently only L2Expanded, InnerProduct and CosineExpanded metrics are supported");

  uint32_t node_degree = knn_graph.extent(1);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "cagra::build_knn_graph<IVF-PQ>(%zu, %zu, %u)",
    size_t(dataset.extent(0)),
    size_t(dataset.extent(1)),
    node_degree);

  // Make model name
  const std::string model_name = [&]() {
    char model_name[1024];
    sprintf(model_name,
            "%s-%lux%lu.cluster_%u.pq_%u.%ubit.itr_%u.metric_%d.pqcenter_%u",
            "IVF-PQ",
            static_cast<size_t>(dataset.extent(0)),
            static_cast<size_t>(dataset.extent(1)),
            pq.build_params.n_lists,
            pq.build_params.pq_dim,
            pq.build_params.pq_bits,
            pq.build_params.kmeans_n_iters,
            static_cast<int>(pq.build_params.metric),
            static_cast<uint32_t>(pq.build_params.codebook_kind));
    return std::string(model_name);
  }();

  RAFT_LOG_DEBUG("# Building IVF-PQ index %s", model_name.c_str());
  auto index = cuvs::neighbors::ivf_pq::build(res, pq.build_params, dataset);

  // Empirically detect FP16 distance overflow on the just-built index: run a small FP16 probe
  // search and downgrade the internal/coarse dtypes to FP32 if any distance comes back non-finite.
  // This observes the actual computation, so it is agnostic of the selected distance type.
  if (pq.search_params.internal_distance_dtype == CUDA_R_16F ||
      pq.search_params.coarse_search_dtype == CUDA_R_16F) {
    {
      const bool fp16_overflow = cuvs::neighbors::ivf_pq::helpers::detect_fp16_overflow(
        res, index, pq.search_params, dataset, node_degree + 1);
      if (fp16_overflow) {
        RAFT_LOG_WARN(
          "IVF-PQ FP16 distance produced non-finite results on a probe search for this dataset -> "
          "switching 'internal_distance_dtype' and 'coarse_search_dtype' to FP32");
        pq.search_params.internal_distance_dtype = CUDA_R_32F;
        pq.search_params.coarse_search_dtype     = CUDA_R_32F;
      }
    }
  }

  //
  // search top (k + 1) neighbors
  //

  const auto top_k       = node_degree + 1;
  uint32_t gpu_top_k     = node_degree * pq.refinement_rate;
  gpu_top_k              = std::min<IdxT>(std::max(gpu_top_k, top_k), dataset.extent(0));
  const auto num_queries = dataset.extent(0);

  // Use the same maximum batch size as the ivf_pq::search to avoid allocating more than needed.
  uint32_t max_queries = pq.search_params.max_internal_batch_size;

  // Heuristic: the build_knn_graph code should use only a fraction of the workspace memory; the
  // rest should be used by the ivf_pq::search. Here we say that the workspace size should be a good
  // multiple of what is required for the I/O batching below.
  constexpr size_t kMinWorkspaceRatio = 5;
  constexpr size_t kMinLargeBatchSize = 512;
  auto desired_workspace_size =
    max_queries * (sizeof(DataT) * dataset.extent(1)  // queries (dataset batch)
                   + sizeof(float) * gpu_top_k        // distances
                   + sizeof(int64_t) * gpu_top_k      // neighbors
                   + sizeof(float) * top_k            // refined_distances
                   + sizeof(int64_t) * top_k          // refined_neighbors
                  );
  auto free_space_ratio    = raft::resource::get_workspace_free_bytes(res) / desired_workspace_size;
  bool use_large_workspace = false;
  if (free_space_ratio < kMinWorkspaceRatio) {
    auto adjusted_max_queries =
      static_cast<uint32_t>(max_queries * free_space_ratio / kMinWorkspaceRatio);
    if (adjusted_max_queries >= kMinLargeBatchSize) {
      // adjust max_queries, so that the ratio free_space_ratio gets not larger than
      // kMinWorkspaceRatio.
      RAFT_LOG_INFO(
        "CAGRA graph build: reducing IVF-PQ search max_internal_batch_size from %u -> %u to fit "
        "the workspace",
        max_queries,
        adjusted_max_queries);
      max_queries                              = adjusted_max_queries;
      pq.search_params.max_internal_batch_size = adjusted_max_queries;
    } else {
      // adjusting max_queries to a very small value isn't practical, so we use the large workspace
      // instead.
      use_large_workspace = true;
      RAFT_LOG_WARN(
        "Using large workspace memory for IVF-PQ search during CAGRA graph build. Desired "
        "workspace size: %zu, free workspace size: %zu",
        desired_workspace_size * kMinWorkspaceRatio,
        raft::resource::get_workspace_free_bytes(res));
    }
  }

  // If the workspace is smaller than desired, put the I/O buffers into the large workspace.
  rmm::device_async_resource_ref workspace_mr =
    use_large_workspace ? raft::resource::get_large_workspace_resource_ref(res)
                        : raft::resource::get_workspace_resource_ref(res);

  RAFT_LOG_DEBUG(
    "IVF-PQ search node_degree: %d, top_k: %d,  gpu_top_k: %d,  max_batch_size:: %d, n_probes: %u",
    node_degree,
    top_k,
    gpu_top_k,
    max_queries,
    pq.search_params.n_probes);

  auto distances = raft::make_device_mdarray<float>(
    res, workspace_mr, raft::make_extents<int64_t>(max_queries, gpu_top_k));
  auto neighbors = raft::make_device_mdarray<int64_t>(
    res, workspace_mr, raft::make_extents<int64_t>(max_queries, gpu_top_k));
  auto refined_distances = raft::make_device_mdarray<float>(
    res, workspace_mr, raft::make_extents<int64_t>(max_queries, top_k));
  auto refined_neighbors = raft::make_device_mdarray<int64_t>(
    res, workspace_mr, raft::make_extents<int64_t>(max_queries, top_k));
  auto neighbors_host = raft::make_host_matrix<int64_t, int64_t>(max_queries, gpu_top_k);
  auto queries_host   = raft::make_host_matrix<DataT, int64_t>(max_queries, dataset.extent(1));
  auto refined_neighbors_host = raft::make_host_matrix<int64_t, int64_t>(max_queries, top_k);
  auto refined_distances_host = raft::make_host_matrix<float, int64_t>(max_queries, top_k);

  // TODO(tfeher): batched search with multiple GPUs
  std::size_t num_self_included = 0;
  bool first                    = true;
  const auto start_clock        = std::chrono::system_clock::now();
  auto last_tick                = start_clock;

  auto vec_batches = cuvs::spatial::knn::detail::utils::make_batch_load_iterator<DataT>(
    res,
    dataset.data_handle(),
    static_cast<int64_t>(dataset.extent(0)),
    static_cast<int64_t>(dataset.extent(1)),
    static_cast<size_t>(max_queries),
    raft::resource::get_cuda_stream(res),
    workspace_mr);

  size_t next_report_offset = 0;
  size_t d_report_offset    = dataset.extent(0) / 100;  // Report progress in 1% steps.

  bool async_host_processing   = raft::is_host_mdspan_v<decltype(dataset)> || top_k == gpu_top_k;
  size_t previous_batch_size   = 0;
  size_t previous_batch_offset = 0;

  for (const auto& batch : vec_batches) {
    auto queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
      batch.data(), batch.size(), batch.row_width());
    auto neighbors_view = raft::make_device_matrix_view<int64_t, int64_t>(
      neighbors.data_handle(), batch.size(), neighbors.extent(1));
    auto distances_view = raft::make_device_matrix_view<float, int64_t>(
      distances.data_handle(), batch.size(), distances.extent(1));

    cuvs::neighbors::ivf_pq::search(
      res, pq.search_params, index, queries_view, neighbors_view, distances_view);

    if (async_host_processing) {
      // process previous batch async on host
      // NOTE: the async path also covers disabled refinement (top_k == gpu_top_k)
      if (previous_batch_size > 0) {
        refine_host_and_write_graph(res,
                                    queries_host,
                                    neighbors_host,
                                    refined_neighbors_host,
                                    refined_distances_host,
                                    dataset,
                                    knn_graph,
                                    pq.build_params.metric,
                                    num_self_included,
                                    previous_batch_size,
                                    previous_batch_offset,
                                    top_k,
                                    gpu_top_k);
      }

      // copy next batch to host
      raft::copy(res,
                 raft::make_host_vector_view(neighbors_host.data_handle(), neighbors_view.size()),
                 raft::make_device_vector_view(neighbors.data_handle(), neighbors_view.size()));
      if (top_k != gpu_top_k) {
        // can be skipped for disabled refinement
        raft::copy(res,
                   raft::make_host_vector_view(queries_host.data_handle(), queries_view.size()),
                   raft::make_device_vector_view(batch.data(), queries_view.size()));
      }

      previous_batch_size   = batch.size();
      previous_batch_offset = batch.offset();

      // we need to ensure the copy operations are done prior using the host data
      raft::resource::sync_stream(res);

      // process last batch
      if (previous_batch_offset + previous_batch_size == (size_t)num_queries) {
        refine_host_and_write_graph(res,
                                    queries_host,
                                    neighbors_host,
                                    refined_neighbors_host,
                                    refined_distances_host,
                                    dataset,
                                    knn_graph,
                                    pq.build_params.metric,
                                    num_self_included,
                                    previous_batch_size,
                                    previous_batch_offset,
                                    top_k,
                                    gpu_top_k);
      }
    } else {
      auto neighbor_candidates_view = raft::make_device_matrix_view<const int64_t, int64_t>(
        neighbors.data_handle(), batch.size(), gpu_top_k);
      auto refined_neighbors_view = raft::make_device_matrix_view<int64_t, int64_t>(
        refined_neighbors.data_handle(), batch.size(), top_k);
      auto refined_distances_view = raft::make_device_matrix_view<float, int64_t>(
        refined_distances.data_handle(), batch.size(), top_k);

      auto dataset_view = raft::make_device_matrix_view<const DataT, int64_t>(
        dataset.data_handle(), dataset.extent(0), dataset.extent(1));
      cuvs::neighbors::refine(res,
                              dataset_view,
                              queries_view,
                              neighbor_candidates_view,
                              refined_neighbors_view,
                              refined_distances_view,
                              pq.build_params.metric);
      raft::copy(res,
                 raft::make_host_vector_view(refined_neighbors_host.data_handle(),
                                             refined_neighbors_view.size()),
                 raft::make_device_vector_view(refined_neighbors_view.data_handle(),
                                               refined_neighbors_view.size()));
      raft::resource::sync_stream(res);

      auto refined_neighbors_host_view = raft::make_host_matrix_view<int64_t, int64_t>(
        refined_neighbors_host.data_handle(), batch.size(), top_k);
      write_to_graph(
        knn_graph, refined_neighbors_host_view, num_self_included, batch.size(), batch.offset());
    }

    size_t num_queries_done = batch.offset() + batch.size();
    const auto end_clock    = std::chrono::system_clock::now();
    if (batch.offset() > next_report_offset &&
        std::chrono::duration_cast<std::chrono::seconds>(end_clock - last_tick) >
          std::chrono::seconds(10)) {
      next_report_offset += d_report_offset;
      const auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() *
        1e-6;
      const auto throughput = num_queries_done / time;
      last_tick             = end_clock;

      RAFT_LOG_DEBUG(
        "# Search %12lu / %12lu (%3.2f %%), %e queries/sec, %.2f minutes ETA, self included = "
        "%3.2f %%    \r",
        num_queries_done,
        dataset.extent(0),
        num_queries_done / static_cast<double>(dataset.extent(0)) * 100,
        throughput,
        (num_queries - num_queries_done) / throughput / 60,
        static_cast<double>(num_self_included) / num_queries_done * 100.);
    }
    first = false;
  }

  if (!first) RAFT_LOG_DEBUG("# Finished building kNN graph");
  if (static_cast<double>(num_self_included) / dataset.extent(0) * 100. < 5) {
    RAFT_LOG_WARN(
      "Self-included ratio is low: %2.2f %%. This can lead to poor recall. "
      "Consider using a different configuration for the IVF-PQ index, "
      "increasing the refinement rate, or using higher-precision data type for "
      "LUT/Internal Distance.",
      static_cast<double>(num_self_included) / dataset.extent(0) * 100.);
  }
}

template <typename DataT, typename IdxT, typename accessor>
void build_knn_graph(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cuvs::neighbors::nn_descent::index_params build_params)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "cagra::build_knn_graph<NN-DESCENT>(%zu, %zu, %u)",
    size_t(dataset.extent(0)),
    size_t(dataset.extent(1)),
    size_t(knn_graph.extent(1)));

  std::optional<raft::host_matrix_view<IdxT, int64_t, row_major>> graph_view = knn_graph;
  auto nn_descent_idx = cuvs::neighbors::nn_descent::build(res, build_params, dataset, graph_view);

  using internal_IdxT = typename std::make_unsigned<IdxT>::type;
  using g_accessor    = typename decltype(nn_descent_idx.graph())::accessor_type;
  using g_accessor_internal =
    raft::host_device_accessor<cuda::std::default_accessor<internal_IdxT>, g_accessor::mem_type>;

  auto knn_graph_internal =
    raft::mdspan<internal_IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor_internal>(
      reinterpret_cast<internal_IdxT*>(nn_descent_idx.graph().data_handle()),
      nn_descent_idx.graph().extent(0),
      nn_descent_idx.graph().extent(1));

  cuvs::neighbors::cagra::detail::graph::sort_knn_graph(
    res, build_params.metric, dataset, knn_graph_internal);
}

template <typename IdxT = uint32_t,
          typename g_accessor =
            raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::host>>
void optimize(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor> knn_graph,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> new_graph,
  const bool guarantee_connectivity = false)
{
  using internal_IdxT = typename std::make_unsigned<IdxT>::type;

  auto new_graph_internal = raft::make_host_matrix_view<internal_IdxT, int64_t>(
    reinterpret_cast<internal_IdxT*>(new_graph.data_handle()),
    new_graph.extent(0),
    new_graph.extent(1));

  using g_accessor_internal =
    raft::host_device_accessor<cuda::std::default_accessor<internal_IdxT>, raft::memory_type::host>;
  auto knn_graph_internal =
    raft::mdspan<internal_IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor_internal>(
      reinterpret_cast<internal_IdxT*>(knn_graph.data_handle()),
      knn_graph.extent(0),
      knn_graph.extent(1));

  cagra::detail::graph::optimize(
    res, knn_graph_internal, new_graph_internal, guarantee_connectivity);
}

// RAII wrapper for allocating memory with Transparent HugePage
struct mmap_owner {
  // Allocate a new memory (not backed by a file)
  mmap_owner(size_t size) : size_{size}
  {
    int flags = MAP_ANONYMOUS | MAP_PRIVATE;
    ptr_      = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr_ == MAP_FAILED) {
      ptr_ = nullptr;
      throw std::runtime_error("cuvs::mmap_owner error");
    }
    if (madvise(ptr_, size, MADV_HUGEPAGE) != 0) {
      munmap(ptr_, size);
      ptr_ = nullptr;
      throw std::runtime_error("cuvs::mmap_owner error");
    }
  }

  ~mmap_owner() noexcept
  {
    if (ptr_ != nullptr) { munmap(ptr_, size_); }
  }

  // No copies for owning struct
  mmap_owner(const mmap_owner& res)                      = delete;
  auto operator=(const mmap_owner& other) -> mmap_owner& = delete;
  // Moving is fine
  mmap_owner(mmap_owner&& other)
    : ptr_{std::exchange(other.ptr_, nullptr)}, size_{std::exchange(other.size_, 0)}
  {
  }
  auto operator=(mmap_owner&& other) -> mmap_owner&
  {
    std::swap(this->ptr_, other.ptr_);
    std::swap(this->size_, other.size_);
    return *this;
  }

  [[nodiscard]] auto data() const -> void* { return ptr_; }
  [[nodiscard]] auto size() const -> size_t { return size_; }

 private:
  void* ptr_;
  size_t size_;
};

/** Upload and/or pad `dataset` to a device-resident CAGRA-aligned view for iterative internal
 * search. */
template <typename T, typename DatasetViewT>
  requires cuvs::neighbors::is_dense_row_major_dataset_view_v<DatasetViewT>
auto ensure_device_padded_for_iterative_search(
  raft::resources const& res,
  DatasetViewT const& dataset,
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<T, int64_t>>& padded_own)
  -> cuvs::neighbors::device_padded_dataset_view<T, int64_t>
{
  if constexpr (cuvs::neighbors::is_device_padded_dataset_view_v<DatasetViewT>) {
    return dataset;
  } else {
    padded_own = cuvs::neighbors::make_device_padded_dataset(res, dataset.view());
    return padded_own->as_dataset_view();
  }
}

template <typename T, typename IdxT = uint32_t, typename DatasetViewT>
  requires cuvs::neighbors::is_dense_row_major_dataset_view_v<DatasetViewT>
auto iterative_build_graph(raft::resources const& res,
                           const index_params& params,
                           DatasetViewT const& dataset) -> raft::host_matrix<IdxT, int64_t>
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;

  auto cagra_graph = raft::make_host_matrix<IdxT, int64_t>(0, 0);

  // Iteratively improve the accuracy of the graph by repeatedly running
  // CAGRA's search() and optimize(). Host or non-CAGRA-aligned device inputs are uploaded
  // and padded here only for the internal search loop — same role as main's
  // make_aligned_dataset() inside iterative_build_graph. IVF-PQ / NN-descent never take this path.
  RAFT_LOG_INFO("Iteratively creating/improving graph index using CAGRA's search() and optimize()");

  std::unique_ptr<cuvs::neighbors::device_padded_dataset<T, int64_t>> padded_own;
  auto search_dataset = ensure_device_padded_for_iterative_search<T>(res, dataset, padded_own);

  auto dev_dataset     = search_dataset.view();
  uint32_t logical_dim = search_dataset.dim();

  // Determine initial graph size.
  uint64_t final_graph_size   = (uint64_t)search_dataset.n_rows();
  uint64_t initial_graph_size = (final_graph_size + 1) / 2;
  while (initial_graph_size > graph_degree * 64) {
    initial_graph_size = (initial_graph_size + 1) / 2;
  }
  RAFT_LOG_DEBUG("# initial graph size = %lu", (uint64_t)initial_graph_size);

  // Allocate memory for search results.
  constexpr uint64_t max_chunk_size = 8192;
  // +1 because the search may return the query node itself as a neighbor;
  // this is consistent with the per-iteration curr_topk = next_graph_degree + 1
  auto topk          = intermediate_degree + 1;
  auto dev_neighbors = raft::make_device_matrix<IdxT, int64_t>(res, max_chunk_size, topk);
  auto dev_distances = raft::make_device_matrix<float, int64_t>(res, max_chunk_size, topk);

  std::optional<raft::device_matrix<T, int64_t>> query_contiguous;
  if (static_cast<int64_t>(logical_dim) != dev_dataset.extent(1)) {
    query_contiguous.emplace(
      raft::make_device_matrix<T, int64_t>(res, max_chunk_size, logical_dim));
  }

  // Determine graph degree and number of search results while increasing
  // graph size.
  auto small_graph_degree = std::max(graph_degree / 2, std::min(graph_degree, (uint64_t)24));
  RAFT_LOG_DEBUG("# small_graph_degree = %lu", (uint64_t)small_graph_degree);
  RAFT_LOG_DEBUG("# graph_degree = %lu", (uint64_t)graph_degree);
  RAFT_LOG_DEBUG("# topk = %lu", (uint64_t)topk);

  // Create an initial graph. The initial graph created here is not suitable for
  // searching, but connectivity is guaranteed.
  auto offset = raft::make_host_vector<IdxT, int64_t>(small_graph_degree);
  for (uint64_t j = 0; j < small_graph_degree; j++) {
    if (j == 0) {
      offset(j) = 1;
    } else {
      offset(j) = offset(j - 1) + 1;
    }
    IdxT ofst = pow((double)(initial_graph_size - 1) / 2, (double)(j + 1) / small_graph_degree);
    if (offset(j) < ofst) { offset(j) = ofst; }
    RAFT_LOG_DEBUG("# offset(%lu) = %lu", (uint64_t)j, (uint64_t)offset(j));
  }
  cagra_graph = raft::make_host_matrix<IdxT, int64_t>(initial_graph_size, small_graph_degree);
  for (uint64_t i = 0; i < initial_graph_size; i++) {
    for (uint64_t j = 0; j < small_graph_degree; j++) {
      cagra_graph(i, j) = (i + offset(j)) % initial_graph_size;
    }
  }

  // Allocate memory for neighbors list using Transparent HugePage
  constexpr size_t thp_size = 2 * 1024 * 1024;
  size_t byte_size          = sizeof(IdxT) * final_graph_size * topk;
  if (byte_size % thp_size) { byte_size += thp_size - (byte_size % thp_size); }
  mmap_owner neighbors_list(byte_size);
  IdxT* neighbors_ptr = (IdxT*)neighbors_list.data();
  memset(neighbors_ptr, 0, byte_size);

  bool flag_last       = false;
  auto curr_graph_size = initial_graph_size;
  while (true) {
    auto start           = std::chrono::high_resolution_clock::now();
    auto curr_query_size = std::min(2 * curr_graph_size, final_graph_size);

    auto next_graph_degree = small_graph_degree;
    if (curr_graph_size == final_graph_size) { next_graph_degree = graph_degree; }

    // The search count (topk) is set to the next graph degree + 1, because
    // pruning is not used except in the last iteration.
    // (*) The appropriate setting for itopk_size requires careful consideration.
    auto curr_topk       = next_graph_degree + 1;
    auto curr_itopk_size = next_graph_degree + 32;
    if (flag_last) {
      curr_topk       = topk;
      curr_itopk_size = curr_topk + 32;
    }

    RAFT_LOG_DEBUG(
      "# graph_size = %lu (%.3lf), graph_degree = %lu, query_size = %lu, itopk = %lu, topk = %lu",
      (uint64_t)cagra_graph.extent(0),
      (double)cagra_graph.extent(0) / final_graph_size,
      (uint64_t)cagra_graph.extent(1),
      (uint64_t)curr_query_size,
      (uint64_t)curr_itopk_size,
      (uint64_t)curr_topk);

    cuvs::neighbors::cagra::search_params search_params;
    search_params.algo        = cuvs::neighbors::cagra::search_algo::AUTO;
    search_params.max_queries = max_chunk_size;
    search_params.itopk_size  = curr_itopk_size;

    // Create an index (idx), a query view (dev_query_view), and a mdarray for
    // search results (neighbors).
    auto dev_dataset_view = raft::make_device_matrix_view<const T, int64_t>(
      dev_dataset.data_handle(), (int64_t)curr_graph_size, dev_dataset.extent(1));
    cuvs::neighbors::device_padded_dataset_view<T, int64_t> sub_padded(dev_dataset_view,
                                                                       logical_dim);

    auto idx = cuvs::neighbors::cagra::device_padded_index<T, IdxT>(
      res, params.metric, sub_padded, raft::make_const_mdspan(cagra_graph.view()));

    auto dev_query_view = raft::make_device_matrix_view<const T, int64_t>(
      dev_dataset.data_handle(), (int64_t)curr_query_size, dev_dataset.extent(1));

    auto neighbors_view =
      raft::make_host_matrix_view<IdxT, int64_t>(neighbors_ptr, curr_query_size, curr_topk);

    // Search.
    // Since there are many queries, divide them into batches and search them.
    auto query_batch = cuvs::spatial::knn::detail::utils::make_batch_load_iterator<T>(
      res,
      dev_query_view.data_handle(),
      static_cast<int64_t>(curr_query_size),
      static_cast<int64_t>(dev_query_view.extent(1)),
      max_chunk_size,
      raft::resource::get_cuda_stream(res),
      raft::resource::get_workspace_resource_ref(res));
    for (const auto& batch : query_batch) {
      raft::device_matrix_view<const T, int64_t> batch_dev_query_view;
      if (query_contiguous) {
        raft::copy_matrix(query_contiguous->data_handle(),
                          static_cast<int64_t>(logical_dim),
                          batch.data(),
                          dev_query_view.extent(1),
                          static_cast<int64_t>(logical_dim),
                          batch.size(),
                          raft::resource::get_cuda_stream(res));
        batch_dev_query_view = raft::make_device_matrix_view<const T, int64_t>(
          query_contiguous->data_handle(), batch.size(), static_cast<int64_t>(logical_dim));
      } else {
        batch_dev_query_view = raft::make_device_matrix_view<const T, int64_t>(
          batch.data(), batch.size(), dev_query_view.extent(1));
      }
      auto batch_dev_neighbors_view = raft::make_device_matrix_view<IdxT, int64_t>(
        dev_neighbors.data_handle(), batch.size(), curr_topk);
      auto batch_dev_distances_view = raft::make_device_matrix_view<float, int64_t>(
        dev_distances.data_handle(), batch.size(), curr_topk);

      cuvs::neighbors::cagra::search(res,
                                     search_params,
                                     idx,
                                     batch_dev_query_view,
                                     batch_dev_neighbors_view,
                                     batch_dev_distances_view);

      auto batch_neighbors_view = raft::make_host_matrix_view<IdxT, int64_t>(
        neighbors_view.data_handle() + batch.offset() * curr_topk, batch.size(), curr_topk);
      raft::copy(res, batch_neighbors_view, batch_dev_neighbors_view);
    }

    // Optimize graph
    auto next_graph_size = curr_query_size;
    cagra_graph          = raft::make_host_matrix<IdxT, int64_t>(0, 0);  // delete existing grahp
    cagra_graph = raft::make_host_matrix<IdxT, int64_t>(next_graph_size, next_graph_degree);
    optimize<IdxT>(
      res, neighbors_view, cagra_graph.view(), flag_last ? params.guarantee_connectivity : 0);

    auto end        = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    RAFT_LOG_DEBUG("# elapsed time: %.3lf sec", (double)elapsed_ms / 1000);

    if (flag_last) { break; }
    flag_last       = (curr_graph_size == final_graph_size);
    curr_graph_size = next_graph_size;
  }

  return cagra_graph;
}

template <typename IdxT>
[[nodiscard]] inline auto resolve_cagra_default_knn_graph_build_params(
  raft::resources const& res,
  index_params const& params,
  raft::matrix_extent<int64_t> dataset_extents,
  size_t intermediate_degree)
{
  auto knn_build_params = params.graph_build_params;
  if (std::holds_alternative<std::monostate>(params.graph_build_params)) {
    if (cuvs::neighbors::nn_descent::has_enough_device_memory(res, dataset_extents, sizeof(IdxT))) {
      RAFT_LOG_DEBUG("NN descent solver");
      knn_build_params =
        cagra::graph_build_params::nn_descent_params(intermediate_degree, params.metric);
    } else {
      RAFT_LOG_DEBUG("Selecting IVF-PQ solver");
      knn_build_params = cagra::graph_build_params::ivf_pq_params(dataset_extents, params.metric);
    }
  }
  return knn_build_params;
}

template <typename T, typename KnnParamsVariant>
inline void validate_cagra_knn_graph_build_constraints(index_params const& params,
                                                       KnnParamsVariant const& knn_build_params)
{
  RAFT_EXPECTS(
    params.metric != cuvs::distance::DistanceType::BitwiseHamming ||
      std::holds_alternative<cagra::graph_build_params::iterative_search_params>(
        knn_build_params) ||
      std::holds_alternative<cagra::graph_build_params::nn_descent_params>(knn_build_params),
    "The selected CAGRA graph builder does not support BitwiseHamming as a metric. Please "
    "use nn-descent or the iterative CAGRA search build.");
  RAFT_EXPECTS(
    params.metric != cuvs::distance::DistanceType::CosineExpanded ||
      std::holds_alternative<cagra::graph_build_params::ivf_pq_params>(knn_build_params) ||
      std::holds_alternative<cagra::graph_build_params::brute_force_params>(knn_build_params) ||
      std::holds_alternative<cagra::graph_build_params::nn_descent_params>(knn_build_params),
    "CosineExpanded distance is not supported for iterative CAGRA graph build.");

  RAFT_EXPECTS(params.metric != cuvs::distance::DistanceType::BitwiseHamming ||
                 (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>),
               "BitwiseHamming distance is only supported for int8_t and uint8_t data types. "
               "Current data type is not supported.");
}

/**
 * Iterative / IVF-PQ / NN-descent / brute-force KNN graph construction and `optimize` → final host
 * CAGRA graph.
 *
 * @param knn_graph_dataset  mdspan passed to IVF-PQ / NN-descent `build_knn_graph` (any stride).
 */
template <typename T, typename IdxT, typename KnnParamsVariant, typename KnnGraphDatasetMdspan>
auto build_cagra_host_graph_from_knn_params(raft::resources const& res,
                                            index_params const& params,
                                            KnnParamsVariant const& knn_build_params,
                                            int64_t n_rows,
                                            size_t intermediate_degree,
                                            size_t graph_degree,
                                            KnnGraphDatasetMdspan&& knn_graph_dataset)
  -> raft::host_matrix<IdxT, int64_t>
{
  std::optional<raft::host_matrix<IdxT, int64_t>> knn_graph(
    raft::make_host_matrix<IdxT, int64_t>(n_rows, intermediate_degree));

  if (std::holds_alternative<cagra::graph_build_params::ivf_pq_params>(knn_build_params)) {
    auto ivf_pq_params =
      std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(knn_build_params);
    if (ivf_pq_params.build_params.metric != params.metric) {
      RAFT_LOG_WARN(
        "Metric (%lu) for IVF-PQ needs to match cagra metric (%lu), "
        "aligning IVF-PQ metric.",
        ivf_pq_params.build_params.metric,
        params.metric);
      ivf_pq_params.build_params.metric = params.metric;
    }
    build_knn_graph(res, knn_graph_dataset, knn_graph->view(), ivf_pq_params);
  } else if (std::holds_alternative<graph_build_params::brute_force_params>(knn_build_params)) {
    auto brute_force_params = std::get<graph_build_params::brute_force_params>(knn_build_params);
    brute_force_params.build_params.metric = params.metric;
    build_knn_graph(res, knn_graph_dataset, knn_graph->view(), brute_force_params);
  } else {
    auto nn_descent_params =
      std::get<cagra::graph_build_params::nn_descent_params>(knn_build_params);

    if (nn_descent_params.metric != params.metric) {
      RAFT_LOG_WARN(
        "Metric (%lu) for nn-descent needs to match cagra metric (%lu), "
        "aligning nn-descent metric.",
        nn_descent_params.metric,
        params.metric);
      nn_descent_params.metric = params.metric;
    }
    if (nn_descent_params.graph_degree != intermediate_degree) {
      RAFT_LOG_WARN(
        "Graph degree (%lu) for nn-descent needs to match cagra intermediate graph degree (%lu), "
        "aligning "
        "nn-descent graph_degree.",
        nn_descent_params.graph_degree,
        intermediate_degree);
      nn_descent_params =
        cagra::graph_build_params::nn_descent_params(intermediate_degree, params.metric);
    }

    nn_descent_params.return_distances = false;
    build_knn_graph<T, IdxT>(res, knn_graph_dataset, knn_graph->view(), nn_descent_params);
  }

  auto cagra_graph = raft::make_host_matrix<IdxT, int64_t>(n_rows, graph_degree);

  RAFT_LOG_TRACE("optimizing graph");
  optimize<IdxT>(res, knn_graph->view(), cagra_graph.view(), params.guarantee_connectivity);

  knn_graph.reset();
  return cagra_graph;
}

/**
 * Build from a host row-major matrix without uploading the full dataset early when IVF-PQ graph
 * construction can consume host batches directly. The iterative path uploads and pads inside
 * `iterative_build_graph`. When requested, the returned index retains the input host dataset as a
 * non-owning view; it still requires a device dataset before search.
 */
template <typename T, typename IdxT = uint32_t, typename DatasetViewT>
  requires cuvs::neighbors::is_host_dataset_view_v<DatasetViewT>
auto build_from_host_matrix(raft::resources const& res,
                            const index_params& params,
                            DatasetViewT const& dataset)
  -> cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>
{
  size_t const n_rows = static_cast<size_t>(dataset.n_rows());
  size_t const dim    = static_cast<size_t>(dataset.dim());

  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;
  common::nvtx::range<common::nvtx::domain::cuvs> function_scope(
    "cagra::detail::build_from_host_matrix(%zu, %zu)", intermediate_degree, graph_degree);
  check_graph_degree<T, IdxT>(intermediate_degree, graph_degree, n_rows);

  auto dataset_extents =
    raft::matrix_extent<int64_t>(static_cast<int64_t>(n_rows), static_cast<int64_t>(dim));

  auto knn_build_params = resolve_cagra_default_knn_graph_build_params<IdxT>(
    res, params, dataset_extents, intermediate_degree);
  validate_cagra_knn_graph_build_constraints<T>(params, knn_build_params);

  auto cagra_graph = [&]() -> raft::host_matrix<IdxT, int64_t> {
    if (std::holds_alternative<cagra::graph_build_params::iterative_search_params>(
          knn_build_params)) {
      return iterative_build_graph<T, IdxT>(res, params, dataset);
    }
    return build_cagra_host_graph_from_knn_params<T, IdxT>(res,
                                                           params,
                                                           knn_build_params,
                                                           static_cast<int64_t>(n_rows),
                                                           intermediate_degree,
                                                           graph_degree,
                                                           dataset.view());
  }();

  RAFT_LOG_TRACE("Graph optimized, creating index");

  if (params.attach_dataset_on_build) {
    return cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>(
      res, params.metric, dataset, raft::make_const_mdspan(cagra_graph.view()));
  }
  cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT> out(res, params.metric);
  out.update_graph(res, raft::make_const_mdspan(cagra_graph.view()));
  return out;
}

/**
 * Build from a dense device `dataset_view` (padded or standard). VPQ views are rejected by
 * `cagra::build()` before this entry point is reached. Also used from ACE sub-builds and merge.
 * The returned index contains only the optimized graph; call
 * `cagra::update_dataset` before search.
 */
template <typename T, typename IdxT, typename DatasetViewT>
  requires cuvs::neighbors::is_dense_row_major_device_dataset_view_v<DatasetViewT>
auto build_from_device_matrix(raft::resources const& res,
                              const index_params& params,
                              DatasetViewT const& device_dataset)
  -> cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;
  common::nvtx::range<common::nvtx::domain::cuvs> function_scope(
    "cagra::detail::build_from_device_matrix(%zu, %zu)", intermediate_degree, graph_degree);
  check_graph_degree<T, IdxT>(
    intermediate_degree, graph_degree, static_cast<size_t>(device_dataset.n_rows()));

  auto dataset_extents =
    raft::matrix_extent<int64_t>(device_dataset.n_rows(), device_dataset.dim());

  auto knn_build_params = resolve_cagra_default_knn_graph_build_params<IdxT>(
    res, params, dataset_extents, intermediate_degree);
  validate_cagra_knn_graph_build_constraints<T>(params, knn_build_params);

  auto cagra_graph = [&]() -> raft::host_matrix<IdxT, int64_t> {
    if (std::holds_alternative<cagra::graph_build_params::iterative_search_params>(
          knn_build_params)) {
      return iterative_build_graph<T, IdxT>(res, params, device_dataset);
    }
    return build_cagra_host_graph_from_knn_params<T, IdxT>(res,
                                                           params,
                                                           knn_build_params,
                                                           device_dataset.n_rows(),
                                                           intermediate_degree,
                                                           graph_degree,
                                                           device_dataset.view());
  }();

  RAFT_LOG_TRACE("Graph optimized, creating index");

  cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT> idx(res, params.metric);
  idx.update_graph(res, raft::make_const_mdspan(cagra_graph.view()));
  return idx;
}
}  // namespace cuvs::neighbors::cagra::detail
