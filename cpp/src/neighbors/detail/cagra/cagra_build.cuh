/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../core/nvtx.hpp"
#include "../../ivf_pq/ivf_pq_fp16_overflow.cuh"
#include "cagra_search.cuh"
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
#include <raft/matrix/init.cuh>
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

#include <sys/stat.h>

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

template <typename DataT, typename IdxT>
void build_knn_graph(raft::resources const& res,
                     cuvs::neighbors::device_bbq_dataset_view<DataT, int64_t> dataset,
                     raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
                     cuvs::neighbors::nn_descent::index_params build_params)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "cagra::build_knn_graph<NN-DESCENT,BBQ>(%zu, %zu, %u)",
    size_t(dataset.n_rows()),
    size_t(dataset.dim()),
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

  cuvs::neighbors::cagra::detail::graph::sort_knn_graph_bbq(
    res, build_params.metric, dataset, knn_graph_internal);
}

template <typename IdxT = uint32_t,
          typename g_accessor =
            raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::host>,
          typename n_accessor =
            raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::host>>
void optimize(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor> knn_graph,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, n_accessor> new_graph,
  const bool guarantee_connectivity = false)
{
  using internal_IdxT = typename std::make_unsigned<IdxT>::type;

  using g_accessor_internal =
    raft::host_device_accessor<cuda::std::default_accessor<internal_IdxT>, g_accessor::mem_type>;
  using n_accessor_internal =
    raft::host_device_accessor<cuda::std::default_accessor<internal_IdxT>, n_accessor::mem_type>;

  auto new_graph_internal =
    raft::mdspan<internal_IdxT, raft::matrix_extent<int64_t>, raft::row_major, n_accessor_internal>(
      reinterpret_cast<internal_IdxT*>(new_graph.data_handle()),
      new_graph.extent(0),
      new_graph.extent(1));

  auto knn_graph_internal =
    raft::mdspan<internal_IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor_internal>(
      reinterpret_cast<internal_IdxT*>(knn_graph.data_handle()),
      knn_graph.extent(0),
      knn_graph.extent(1));

  cagra::detail::graph::optimize(
    res, knn_graph_internal, new_graph_internal, guarantee_connectivity);
}

template <typename T, typename MathT>
__global__ void kern_reconstruct_vpq_queries(const uint8_t* encoded_data,
                                             uint32_t encoded_row_len,
                                             const MathT* vq_codebook,
                                             const MathT* pq_codebook,
                                             uint32_t dim,
                                             uint32_t pq_len,
                                             uint64_t offset,
                                             uint32_t batch_size,
                                             uint32_t output_ld,
                                             T* output)
{
  const uint64_t batch_idx = blockIdx.x;
  if (batch_idx >= batch_size) return;
  const uint64_t vec_idx       = offset + batch_idx;
  const uint8_t* vec_data      = encoded_data + vec_idx * encoded_row_len;
  const uint32_t vq_code       = *reinterpret_cast<const uint32_t*>(vec_data);
  const uint8_t* pq_codes      = vec_data + sizeof(uint32_t);
  const MathT* vq_centroid_ptr = vq_codebook + static_cast<uint64_t>(vq_code) * dim;

  for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
    uint32_t j = d / pq_len;
    uint32_t k = d % pq_len;
    float val  = static_cast<float>(vq_centroid_ptr[d]) +
                static_cast<float>(pq_codebook[static_cast<uint32_t>(pq_codes[j]) * pq_len + k]);
    output[batch_idx * output_ld + d] = static_cast<T>(val);
  }
}

template <typename T, typename MathT, typename IdxT>
void reconstruct_vpq_queries(raft::resources const& res,
                             cuvs::neighbors::device_vpq_dataset_view<MathT, IdxT> const& vpq_view,
                             uint64_t offset,
                             uint32_t batch_size,
                             raft::device_matrix_view<T, int64_t> output)
{
  auto const& vpq_dset     = vpq_view.dset();
  const uint32_t dim       = vpq_dset.dim();
  const uint32_t pq_len    = vpq_dset.pq_len();
  const uint32_t output_ld = static_cast<uint32_t>(output.extent(1));
  const uint32_t threads   = std::min(dim, 256u);
  RAFT_EXPECTS(output_ld >= dim,
               "VPQ query reconstruct output row width (%u) must be >= logical dim (%u)",
               output_ld,
               dim);

  kern_reconstruct_vpq_queries<T, MathT>
    <<<batch_size, threads, 0, raft::resource::get_cuda_stream(res).get()>>>(
      vpq_dset.data.data_handle(),
      vpq_dset.encoded_row_length(),
      vpq_dset.vq_code_book.data_handle(),
      vpq_dset.pq_code_book.data_handle(),
      dim,
      pq_len,
      offset,
      batch_size,
      output_ld,
      output.data_handle());
}

// Runs CAGRA search for `knn_graph.extent(0)` queries against `idx` in chunks of `max_chunk_size`,
// stacks the results into `knn_graph`, and optimizes them into a newly allocated output graph.
//
// `knn_graph` is allocated by the caller. VPQ builds additionally provide a reconstruction
// buffer and the compressed query rows. The previous-iteration
// graph is passed in so it can be released after search and before the (often larger) output graph
// is allocated, so the two owned graphs never coexist.
//
// Query source:
//   - omitted VPQ arguments: queries are read directly from `dev_query_view`
//     (uncompressed build; the view is a slice of the resident padded device dataset, including
//     CAGRA row padding). `cagra::detail::search_main` accepts that padded row width so search
//     does not depad/re-pad the chunk.
//   - VPQ arguments present: `dev_query_view` is ignored and each chunk of queries is reconstructed
//     on the fly from the VPQ codes into `reconstructed_batch_queries` with CAGRA row padding,
//     so we never materialize the whole (up to N x stride) reconstructed dataset.
template <typename T, typename IdxT, typename DatasetViewT>
auto search_and_optimize(
  raft::resources const& res,
  const cuvs::neighbors::cagra::search_params& search_params,
  const cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>& idx,
  raft::device_matrix_view<const T, int64_t> dev_query_view,
  raft::device_matrix_view<IdxT, int64_t> dev_neighbors,
  raft::device_matrix_view<float, int64_t> dev_distances,
  raft::device_matrix<IdxT, int64_t> prev_graph,
  raft::device_matrix_view<IdxT, int64_t> knn_graph,
  size_t next_graph_degree,
  uint64_t max_chunk_size,
  bool guarantee_connectivity,
  std::optional<raft::device_matrix_view<T, int64_t>> reconstructed_batch_queries    = std::nullopt,
  std::optional<cuvs::neighbors::device_vpq_dataset_view<half, int64_t>> vpq_queries = std::nullopt)
  -> raft::device_matrix<IdxT, int64_t>
{
  auto stream                = raft::resource::get_cuda_stream(res);
  auto const curr_query_size = knn_graph.extent(0);
  auto const curr_topk       = knn_graph.extent(1);

  RAFT_EXPECTS(reconstructed_batch_queries.has_value() == vpq_queries.has_value(),
               "VPQ queries and their reconstruction buffer must be provided together");

  auto run_batch = [&](int64_t offset,
                       int64_t batch_size,
                       raft::device_matrix_view<const T, int64_t> batch_query_view) {
    auto batch_dev_neighbors_view = raft::make_device_matrix_view<IdxT, int64_t>(
      dev_neighbors.data_handle(), batch_size, curr_topk);
    auto batch_dev_distances_view = raft::make_device_matrix_view<float, int64_t>(
      dev_distances.data_handle(), batch_size, curr_topk);

    cuvs::neighbors::cagra::detail::search_main(res,
                                                search_params,
                                                idx,
                                                batch_query_view,
                                                batch_dev_neighbors_view,
                                                batch_dev_distances_view,
                                                cuvs::neighbors::filtering::none_sample_filter{});

    raft::copy(knn_graph.data_handle() + offset * curr_topk,
               batch_dev_neighbors_view.data_handle(),
               batch_size * curr_topk,
               stream);
  };

  if (vpq_queries.has_value()) {
    auto const query_dim = static_cast<int64_t>(idx.dim());

    // Reconstruct-and-search one chunk at a time: reconstruct source rows [offset, offset+bs) into
    // the CAGRA-padded scratch, then search that chunk without a second pad copy.
    const int64_t query_ld = reconstructed_batch_queries->extent(1);
    RAFT_EXPECTS(query_ld >= query_dim,
                 "VPQ query scratch row width (%ld) must be >= logical dim (%ld)",
                 static_cast<long>(query_ld),
                 static_cast<long>(query_dim));
    for (int64_t offset = 0; offset < curr_query_size;
         offset += static_cast<int64_t>(max_chunk_size)) {
      const int64_t batch_size =
        std::min<int64_t>(static_cast<int64_t>(max_chunk_size), curr_query_size - offset);
      auto batch_query_view = raft::make_device_matrix_view<T, int64_t>(
        reconstructed_batch_queries->data_handle(), batch_size, query_ld);
      reconstruct_vpq_queries<T, half, int64_t>(res,
                                                *vpq_queries,
                                                static_cast<uint64_t>(offset),
                                                static_cast<uint32_t>(batch_size),
                                                batch_query_view);
      run_batch(offset, batch_size, batch_query_view);
    }
  } else {
    const int64_t source_row_width = dev_query_view.extent(1);
    auto query_batch               = cuvs::spatial::knn::detail::utils::make_batch_load_iterator<T>(
      res,
      dev_query_view.data_handle(),
      curr_query_size,
      source_row_width,
      max_chunk_size,
      stream,
      raft::resource::get_workspace_resource_ref(res));
    for (const auto& batch : query_batch) {
      auto batch_query_view = raft::make_device_matrix_view<const T, int64_t>(
        batch.data(), static_cast<int64_t>(batch.size()), source_row_width);
      run_batch(
        static_cast<int64_t>(batch.offset()), static_cast<int64_t>(batch.size()), batch_query_view);
    }
  }

  // Search has finished, so the previous-iteration graph that `idx` viewed is no longer needed.
  // Release it before allocating the output graph: the new graph often has more rows than prev.
  prev_graph        = raft::make_device_matrix<IdxT, int64_t>(res, 0, 0);
  auto output_graph = raft::make_device_matrix<IdxT, int64_t>(
    res, curr_query_size, static_cast<int64_t>(next_graph_degree));
  graph::optimize<IdxT>(res, knn_graph, output_graph.view(), guarantee_connectivity);
  return output_graph;
}

template <typename T, typename IdxT = uint32_t, typename DatasetViewT>
  requires(cuvs::neighbors::is_dense_row_major_device_dataset_view_v<DatasetViewT> ||
           cuvs::neighbors::is_device_vpq_f16_dataset_view_v<DatasetViewT>)
auto iterative_build_graph(raft::resources const& res,
                           const index_params& params,
                           DatasetViewT const& dataset) -> raft::device_matrix<IdxT, int64_t>
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;

  const auto& iter_params =
    std::get<cagra::graph_build_params::iterative_search_params>(params.graph_build_params);
  RAFT_LOG_INFO("Build search params: search_width=%zu, max_iterations=%zu",
                iter_params.search_width,
                iter_params.max_iterations);

  // Iteratively improve the graph by repeatedly running CAGRA search and optimize. Dense inputs
  // are searched in-place (CAGRA-aligned device storage; no copy of the caller's rows). VPQ
  // inputs are searched directly and reconstructed per query batch.
  RAFT_LOG_INFO("Iteratively creating/improving graph index using CAGRA's search() and optimize()");

  auto dev_dataset =
    raft::make_device_matrix_view<const T, int64_t>(static_cast<const T*>(nullptr), 0, 0);
  uint32_t logical_dim = dataset.dim();
  uint64_t final_graph_size;
  auto vpq_dataset = cuvs::neighbors::device_vpq_dataset_view<half, int64_t>{};

  if constexpr (cuvs::neighbors::is_device_vpq_f16_dataset_view_v<DatasetViewT>) {
    final_graph_size = static_cast<uint64_t>(dataset.n_rows());
    vpq_dataset      = dataset;
  } else {
    auto const required_stride = cuvs::neighbors::cagra_required_row_width<T>(dataset.dim());
    RAFT_EXPECTS(dataset.stride() == required_stride,
                 "iterative CAGRA build requires a CAGRA-aligned device dataset "
                 "(stride %u, required %u). Pass a device_padded_dataset_view, or a "
                 "device_standard_dataset_view whose row width already matches "
                 "cagra_required_row_width.",
                 dataset.stride(),
                 required_stride);
    dev_dataset      = dataset.view();
    logical_dim      = dataset.dim();
    final_graph_size = static_cast<uint64_t>(dataset.n_rows());
  }

  // Determine initial graph size.
  uint64_t initial_graph_size = (final_graph_size + 1) / 2;
  while (initial_graph_size > graph_degree * 64) {
    initial_graph_size = (initial_graph_size + 1) / 2;
  }
  RAFT_LOG_DEBUG("# initial graph size = %lu", (uint64_t)initial_graph_size);

  // Preallocate the kNN graph at last-iteration size (N × (intermediate_degree+1)) from the large
  // workspace so a too-small pool fails here. The owned search graph is allocated each iteration
  // at the current size and released after search, before the (often larger) output graph is
  // allocated. Per-chunk search I/O is ordinary device memory.
  constexpr uint64_t max_chunk_size = helpers::kIterativeBuildChunkSize;
  // +1 because the search may return the query node itself as a neighbor;
  // this is consistent with the per-iteration curr_topk = next_graph_degree + 1
  auto topk             = intermediate_degree + 1;
  auto large_mr         = raft::resource::get_large_workspace_resource_ref(res);
  auto const n_rows_i64 = static_cast<int64_t>(final_graph_size);
  auto const topk_i64   = static_cast<int64_t>(topk);
  auto const chunk_i64  = static_cast<int64_t>(max_chunk_size);

  auto dev_neighbors = raft::make_device_matrix<IdxT, int64_t>(res, chunk_i64, topk_i64);
  auto dev_distances = raft::make_device_matrix<float, int64_t>(res, chunk_i64, topk_i64);
  auto dev_knn_graph = raft::make_device_mdarray<IdxT, int64_t>(
    res, large_mr, raft::make_extents<int64_t>(n_rows_i64, topk_i64));

  std::optional<raft::device_matrix<T, int64_t>> reconstructed_batch_queries;
  if (vpq_dataset.n_rows() > 0) {
    auto const query_stride_i64 = static_cast<int64_t>(
      cuvs::neighbors::cagra_required_row_width<T>(static_cast<uint32_t>(logical_dim)));
    reconstructed_batch_queries.emplace(
      raft::make_device_matrix<T, int64_t>(res, chunk_i64, query_stride_i64));
    // Padding columns must be zero: search_main cosine post-process reduces over the full row
    // width, and reconstruct only writes the logical dim.
    raft::matrix::fill(res, reconstructed_batch_queries->view(), T(0));
  }

  // Determine graph degree and number of search results while increasing
  // graph size.
  auto small_graph_degree = std::max(graph_degree / 2, std::min(graph_degree, (uint64_t)24));
  RAFT_LOG_DEBUG("# small_graph_degree = %lu", (uint64_t)small_graph_degree);
  RAFT_LOG_DEBUG("# graph_degree = %lu", (uint64_t)graph_degree);
  RAFT_LOG_DEBUG("# topk = %lu", (uint64_t)topk);

  // A fixed itopk_size (0 = auto) governs the growing iterations, which build graphs of degree
  // ~graph_degree/2 and thus request topk ~= graph_degree/2 + 1; the search planner requires
  // topk <= itopk_size. (The full-size iterations override itopk internally, so they are not
  // constrained by this value.)
  RAFT_EXPECTS(iter_params.itopk_size == 0 || iter_params.itopk_size >= graph_degree / 2 + 1,
               "iterative build search itopk_size (%zu) must be 0 (auto) or >= "
               "graph_degree / 2 + 1 (%zu)",
               (size_t)iter_params.itopk_size,
               (size_t)(graph_degree / 2 + 1));

  // Create an initial graph and copy it to the device. It is not suitable for
  // searching, but connectivity is guaranteed. Every iteration, including the first,
  // then searches this device graph.
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
  auto initial_graph =
    raft::make_host_matrix<IdxT, int64_t>(initial_graph_size, small_graph_degree);
  for (uint64_t i = 0; i < initial_graph_size; i++) {
    for (uint64_t j = 0; j < small_graph_degree; j++) {
      initial_graph(i, j) = (i + offset(j)) % initial_graph_size;
    }
  }
  auto stream = raft::resource::get_cuda_stream(res);
  auto dev_graph =
    raft::make_device_matrix<IdxT, int64_t>(res, initial_graph.extent(0), initial_graph.extent(1));
  raft::copy(dev_graph.data_handle(),
             initial_graph.data_handle(),
             initial_graph.extent(0) * initial_graph.extent(1),
             stream);

  bool flag_last       = false;
  auto curr_graph_size = initial_graph_size;

  auto knn_prefix = [&](int64_t rows, int64_t degree) {
    return raft::make_device_matrix_view<IdxT, int64_t>(dev_knn_graph.data_handle(), rows, degree);
  };

  while (true) {
    auto start           = std::chrono::high_resolution_clock::now();
    auto curr_query_size = std::min(2 * curr_graph_size, final_graph_size);

    auto next_graph_degree = small_graph_degree;
    if (curr_graph_size == final_graph_size) { next_graph_degree = graph_degree; }
    RAFT_LOG_INFO("Current graph size %lu: # current graph degree = %lu",
                  (uint64_t)curr_graph_size,
                  (uint64_t)next_graph_degree);

    // The search count (topk) is set to the next graph degree + 1, because
    // pruning is not used except in the last iteration.
    // (*) The appropriate setting for itopk_size requires careful consideration.
    auto curr_topk = next_graph_degree + 1;
    // The configurable itopk (iter_params.itopk_size, 0 = auto) applies only to the true growing
    // iterations, where the degree being built is small_graph_degree. When the graph reaches its
    // full size the search builds a graph_degree-degree graph (topk = graph_degree + 1); that
    // iteration needs a larger itopk, so it overrides the configured value with the auto formula.
    // The final iteration (flag_last) uses a fixed itopk tied to the output topk.
    auto curr_itopk_size = (iter_params.itopk_size > 0 && next_graph_degree == small_graph_degree)
                             ? (uint64_t)iter_params.itopk_size
                             : std::max(next_graph_degree + 32, (uint64_t)128);
    if (flag_last) {
      curr_topk       = topk;
      curr_itopk_size = curr_topk + 32;
    }

    RAFT_LOG_DEBUG(
      "# graph_size = %lu (%.3lf), graph_degree = %lu, query_size = %lu, itopk = %lu, topk = %lu",
      (uint64_t)dev_graph.extent(0),
      (double)dev_graph.extent(0) / final_graph_size,
      (uint64_t)dev_graph.extent(1),
      (uint64_t)curr_query_size,
      (uint64_t)curr_itopk_size,
      (uint64_t)curr_topk);

    cuvs::neighbors::cagra::search_params search_params = iter_params;
    search_params.max_queries                           = max_chunk_size;
    search_params.itopk_size                            = curr_itopk_size;

    auto knn_view =
      knn_prefix(static_cast<int64_t>(curr_query_size), static_cast<int64_t>(curr_topk));

    // Each index holds non-owning dataset and graph views. The local dataset owner and `dev_graph`
    // keep those views alive for the duration of the search.
    if (vpq_dataset.n_rows() > 0) {
      auto idx = cuvs::neighbors::cagra::update_dataset(
        res,
        cuvs::neighbors::cagra::device_pq_index<T, IdxT, half>(res, params.metric),
        vpq_dataset);
      idx.update_graph(res, raft::make_const_mdspan(dev_graph.view()));

      auto empty_query_view =
        raft::make_device_matrix_view<const T, int64_t>(static_cast<const T*>(nullptr), 0, 0);
      dev_graph = search_and_optimize(res,
                                      search_params,
                                      idx,
                                      empty_query_view,
                                      dev_neighbors.view(),
                                      dev_distances.view(),
                                      std::move(dev_graph),
                                      knn_view,
                                      next_graph_degree,
                                      max_chunk_size,
                                      flag_last && params.guarantee_connectivity,
                                      std::optional{reconstructed_batch_queries->view()},
                                      std::optional{vpq_dataset});
    } else {
      auto dev_dataset_view = raft::make_device_matrix_view<const T, int64_t>(
        dev_dataset.data_handle(), static_cast<int64_t>(curr_graph_size), dev_dataset.extent(1));
      cuvs::neighbors::device_padded_dataset_view<T, int64_t> sub_padded(dev_dataset_view,
                                                                         logical_dim);
      auto idx = cuvs::neighbors::cagra::update_dataset(
        res, cuvs::neighbors::cagra::device_padded_index<T, IdxT>(res, params.metric), sub_padded);
      idx.update_graph(res, raft::make_const_mdspan(dev_graph.view()));

      auto dev_query_view = raft::make_device_matrix_view<const T, int64_t>(
        dev_dataset.data_handle(), static_cast<int64_t>(curr_query_size), dev_dataset.extent(1));
      dev_graph = search_and_optimize(res,
                                      search_params,
                                      idx,
                                      dev_query_view,
                                      dev_neighbors.view(),
                                      dev_distances.view(),
                                      std::move(dev_graph),
                                      knn_view,
                                      next_graph_degree,
                                      max_chunk_size,
                                      flag_last && params.guarantee_connectivity);
    }

    auto end = std::chrono::high_resolution_clock::now();
    [[maybe_unused]] auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    RAFT_LOG_DEBUG("# elapsed time: %.3lf sec", (double)elapsed_ms / 1000);

    if (flag_last) { break; }
    flag_last            = (curr_graph_size == final_graph_size);
    auto next_graph_size = curr_query_size;
    curr_graph_size      = next_graph_size;
  }

  return dev_graph;
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
 * construction can consume host batches directly. Iterative CAGRA search needs the rows on device
 * in CAGRA-padded layout and does not copy them here; pass a device dataset to `cagra::build`
 * instead. When requested, the returned index retains the input host dataset as a non-owning view;
 * it still requires a device dataset before search.
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
      RAFT_FAIL(
        "iterative CAGRA build requires a device-resident CAGRA-padded dataset; "
        "pass a device_padded_dataset_view (or an already-aligned "
        "device_standard_dataset_view). Host datasets can use IVF-PQ or NN-descent "
        "graph construction.");
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

  cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT> idx(res, params.metric);
  if (std::holds_alternative<cagra::graph_build_params::iterative_search_params>(
        knn_build_params)) {
    auto cagra_graph = iterative_build_graph<T, IdxT>(res, params, device_dataset);
    idx.update_graph(res, std::move(cagra_graph));
  } else {
    auto cagra_graph = build_cagra_host_graph_from_knn_params<T, IdxT>(res,
                                                                       params,
                                                                       knn_build_params,
                                                                       device_dataset.n_rows(),
                                                                       intermediate_degree,
                                                                       graph_degree,
                                                                       device_dataset.view());
    idx.update_graph(res, raft::make_const_mdspan(cagra_graph.view()));
  }

  RAFT_LOG_TRACE("Graph optimized, creating index");
  return idx;
}

[[nodiscard]] inline auto resolve_bbq_knn_graph_build_params(index_params const& params,
                                                             size_t intermediate_degree)
  -> cuvs::neighbors::nn_descent::index_params
{
  if (std::holds_alternative<std::monostate>(params.graph_build_params)) {
    return cagra::graph_build_params::nn_descent_params(intermediate_degree, params.metric);
  }

  auto nn_descent_params =
    std::get<cagra::graph_build_params::nn_descent_params>(params.graph_build_params);
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
      "aligning nn-descent graph_degree.",
      nn_descent_params.graph_degree,
      intermediate_degree);
    nn_descent_params =
      cagra::graph_build_params::nn_descent_params(intermediate_degree, params.metric);
  }
  return nn_descent_params;
}

/**
 * Build from a device-resident BBQ-quantized dataset: the whole graph construction runs on the
 * compressed codes.
 *
 * The returned index cannot be searched, because CAGRA has no BBQ search kernels. Pass an
 * uncompressed device-padded dataset to `cagra::update_dataset` to obtain a searchable index over
 * the same graph.
 */
template <typename T, typename IdxT, typename DatasetViewT>
  requires cuvs::neighbors::is_device_bbq_dataset_view_v<DatasetViewT>
auto build_from_bbq_dataset(raft::resources const& res,
                            const index_params& params,
                            DatasetViewT const& dataset)
  -> cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>
{
  RAFT_EXPECTS(!dataset.quantizers.empty(), "cagra::build: the BBQ dataset is empty.");
  RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded ||
                 params.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                 params.metric == cuvs::distance::DistanceType::CosineExpanded ||
                 params.metric == cuvs::distance::DistanceType::InnerProduct,
               "cagra::build: a BBQ-quantized dataset supports L2Expanded, L2SqrtExpanded, "
               "CosineExpanded, and InnerProduct.");
  RAFT_EXPECTS(std::holds_alternative<std::monostate>(params.graph_build_params) ||
                 std::holds_alternative<cagra::graph_build_params::nn_descent_params>(
                   params.graph_build_params),
               "cagra::build: a BBQ-quantized dataset requires nn-descent graph construction.");

  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;
  common::nvtx::range<common::nvtx::domain::cuvs> function_scope(
    "cagra::detail::build_from_bbq_dataset(%zu, %zu)", intermediate_degree, graph_degree);
  auto const n_rows = static_cast<int64_t>(dataset.n_rows());
  check_graph_degree<T, IdxT>(intermediate_degree, graph_degree, static_cast<size_t>(n_rows));

  auto nn_descent_params = resolve_bbq_knn_graph_build_params(params, intermediate_degree);
  nn_descent_params.return_distances = false;

  auto cagra_graph = [&]() -> raft::host_matrix<IdxT, int64_t> {
    std::optional<raft::host_matrix<IdxT, int64_t>> knn_graph(
      raft::make_host_matrix<IdxT, int64_t>(n_rows, intermediate_degree));
    build_knn_graph<T, IdxT>(res, dataset, knn_graph->view(), nn_descent_params);

    auto optimized = raft::make_host_matrix<IdxT, int64_t>(n_rows, graph_degree);
    RAFT_LOG_TRACE("optimizing graph");
    optimize<IdxT>(res, knn_graph->view(), optimized.view(), params.guarantee_connectivity);
    knn_graph.reset();
    return optimized;
  }();

  RAFT_LOG_TRACE("Graph optimized, creating index");

  if (params.attach_dataset_on_build) {
    return cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>(
      res, params.metric, dataset, raft::make_const_mdspan(cagra_graph.view()));
  }
  cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT> idx(res, params.metric);
  idx.update_graph(res, raft::make_const_mdspan(cagra_graph.view()));
  return idx;
}
}  // namespace cuvs::neighbors::cagra::detail
