/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../detail/knn_brute_force.cuh"
#include "../detail/nn_descent_gnnd.hpp"
#include "../detail/reachability.cuh"
#include "all_neighbors_merge.cuh"

#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <cuvs/neighbors/refine.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/managed_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/gather.cuh>
#include <raft/util/cudart_utils.hpp>

#include <variant>

namespace cuvs::neighbors::all_neighbors::detail {
using namespace cuvs::neighbors;

/**
 * The final destination for the all-neighbors graph a builder produces. Its memory location selects
 * the strategy:
 *  - direct_t:       full build (no batching). The builder writes the full [num_rows x k] result
 *                    straight into device-resident output (optional distances).
 *  - merge_device:   batched, plain-device global graph (single-GPU device output, merged into the
 *                    caller's device arrays). Merge happens on the GPU.
 *  - merge_managed:  batched, managed global graph for multi-GPU setting. Merge happens on the GPU.
 *  - merge_host:     batched, host-resident global graph. Merged happens on the host.
 */
template <typename T, typename IdxT>
struct global_graph_view {
  using direct_t = std::pair<raft::device_matrix_view<IdxT, IdxT, row_major>,
                             std::optional<raft::device_matrix_view<T, IdxT, row_major>>>;
  using merge_device =
    std::pair<raft::device_matrix_view<IdxT, IdxT>, raft::device_matrix_view<T, IdxT>>;
  using merge_managed =
    std::pair<raft::managed_matrix_view<IdxT, IdxT>, raft::managed_matrix_view<T, IdxT>>;
  using merge_host = std::pair<raft::host_matrix_view<IdxT, IdxT>, raft::host_matrix_view<T, IdxT>>;

  std::variant<direct_t, merge_device, merge_managed, merge_host> dest;

  // Used for merge_host type.
  std::mutex* row_locks = nullptr;
  size_t num_row_locks  = 0;

  // full-build (no batching): write directly to device output
  global_graph_view(raft::device_matrix_view<IdxT, IdxT, row_major> indices,
                    std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances)
    : dest{direct_t{indices, distances}}
  {
  }
  // batched, device global graph (single-GPU device output)
  global_graph_view(raft::device_matrix_view<IdxT, IdxT> neighbors,
                    raft::device_matrix_view<T, IdxT> distances)
    : dest{merge_device{neighbors, distances}}
  {
  }
  // batched, managed global graph (multi-GPU device output)
  global_graph_view(raft::managed_matrix_view<IdxT, IdxT> neighbors,
                    raft::managed_matrix_view<T, IdxT> distances)
    : dest{merge_managed{neighbors, distances}}
  {
  }
  // batched, host-resident global graph
  global_graph_view(raft::host_matrix_view<IdxT, IdxT> neighbors,
                    raft::host_matrix_view<T, IdxT> distances)
    : dest{merge_host{neighbors, distances}}
  {
  }

  [[nodiscard]] bool is_full_build() const { return std::holds_alternative<direct_t>(dest); }
};

template <typename T, typename IdxT = int64_t>
struct all_neighbors_builder {
  all_neighbors_builder(raft::resources const& res,
                        size_t n_clusters,
                        size_t min_cluster_size,
                        size_t max_cluster_size,
                        size_t k)
    : res{res},
      n_clusters{n_clusters},
      min_cluster_size{min_cluster_size},
      max_cluster_size{max_cluster_size},
      k{k}
  {
    if (n_clusters > 1) {  // allocating additional space needed for batching
      inverted_indices_d.emplace(raft::make_device_vector<IdxT, IdxT>(res, max_cluster_size));
      batch_neighbors_h.emplace(raft::make_host_matrix<IdxT, IdxT>(max_cluster_size, k));
      batch_neighbors_d.emplace(raft::make_device_matrix<IdxT, IdxT>(res, max_cluster_size, k));
      batch_distances_d.emplace(raft::make_device_matrix<T, IdxT>(res, max_cluster_size, k));
    }
  }

  /**
   * Merge this cluster's knn results into the global graph. The merge strategy is selected by the
   * memory location of the global_graph_view:
   *  - device / managed: the global [num_rows x k] graph is device-accessible; merge runs on the
   *    GPU (merge_subgraphs_kernel).
   *  - host: the global graph lives in host memory; the per-cluster distances are staged
   * device->host and the merge runs on the host.
   */
  template <typename BeforeRemapT, bool SweepAll = false>
  void do_merge(raft::host_matrix_view<BeforeRemapT, IdxT> indices_for_remap_h,
                raft::host_vector_view<IdxT, IdxT> inverted_indices,
                const global_graph_view<T, IdxT>& global,
                size_t num_data_in_cluster,
                bool select_min)
  {
    using merge_device  = typename global_graph_view<T, IdxT>::merge_device;
    using merge_managed = typename global_graph_view<T, IdxT>::merge_managed;
    using merge_host    = typename global_graph_view<T, IdxT>::merge_host;

    // GPU merge for a device-accessible global graph (device or managed).
    auto gpu_merge = [&](auto global_neighbors, auto global_distances) {
      remap_and_merge_subgraphs<T, IdxT, BeforeRemapT, SweepAll>(res,
                                                                 inverted_indices_d.value().view(),
                                                                 inverted_indices,
                                                                 indices_for_remap_h,
                                                                 batch_neighbors_h.value().view(),
                                                                 batch_neighbors_d.value().view(),
                                                                 batch_distances_d.value().view(),
                                                                 global_neighbors,
                                                                 global_distances,
                                                                 num_data_in_cluster,
                                                                 k,
                                                                 select_min);
    };

    if (const auto* g = std::get_if<merge_device>(&global.dest)) {
      gpu_merge(g->first, g->second);
    } else if (const auto* g = std::get_if<merge_managed>(&global.dest)) {
      gpu_merge(g->first, g->second);
    } else if (const auto* g = std::get_if<merge_host>(&global.dest)) {
      if (!batch_distances_h.has_value()) {
        batch_distances_h.emplace(raft::make_host_matrix<T, IdxT>(max_cluster_size, k));
      }
      // stage this cluster's distances from device to host
      auto batch_distances_h_view = raft::make_host_matrix_view<T, IdxT>(
        batch_distances_h.value().data_handle(), num_data_in_cluster, k);
      raft::copy(res,
                 batch_distances_h_view,
                 raft::make_device_matrix_view<const T, IdxT>(
                   batch_distances_d.value().data_handle(), num_data_in_cluster, k));
      raft::resource::sync_stream(res);

      remap_and_merge_subgraphs<T, IdxT, BeforeRemapT>(res,
                                                       inverted_indices,
                                                       indices_for_remap_h,
                                                       batch_distances_h_view,
                                                       g->first,
                                                       g->second,
                                                       num_data_in_cluster,
                                                       k,
                                                       select_min,
                                                       global.row_locks,
                                                       global.num_row_locks);
    } else {
      RAFT_FAIL("do_merge requires a device, managed, or host global graph destination");
    }
  }

  /**
   * Some memory-heavy allocations that can be used over multiple clusters should be allocated here
   * Arguments:
   * - [in] dataset: host_matrix_view or device_matrix_view of the the ENTIRE dataset
   */
  virtual void prepare_build(raft::host_matrix_view<const T, IdxT, row_major> dataset) {}
  virtual void prepare_build(raft::device_matrix_view<const T, IdxT, row_major> dataset) {}

  /**
   * Running the ann algorithm on the given cluster, and merging it into the global result
   * Arguments:
   * - [in] dataset: host_matrix_view or device_matrix_view of the cluster dataset
   * - [in] inverted_indices (optional): global data indices for the data points in the current
   * cluster of size [num_data_in_cluster]. Only needed when using the batching algorithm.
   * - [out] global: the destination for the all-neighbors graph (global_graph_view).
   */
  virtual void build_knn(raft::host_matrix_view<const T, IdxT, row_major> dataset,
                         std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices,
                         global_graph_view<T, IdxT> global)
  {
  }

  // device dataset is full-build only (no batching), so inverted_indices is always empty.
  virtual void build_knn(raft::device_matrix_view<const T, IdxT, row_major> dataset,
                         std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices,
                         global_graph_view<T, IdxT> global)
  {
  }

  virtual ~all_neighbors_builder() = default;

  raft::resources const& res;
  size_t n_clusters, min_cluster_size, max_cluster_size, k;

  // these are optional types since we do not know the size at time of all_neighbors_builder
  // construction
  std::optional<raft::device_vector<IdxT, IdxT>> inverted_indices_d;
  std::optional<raft::host_matrix<IdxT, IdxT>> batch_neighbors_h;
  std::optional<raft::device_matrix<IdxT, IdxT>> batch_neighbors_d;
  std::optional<raft::device_matrix<T, IdxT>> batch_distances_d;
  // host staging buffer for per-cluster distances feeding the host-side merge
  std::optional<raft::host_matrix<T, IdxT>> batch_distances_h;
};

template <typename T, typename IdxT = int64_t>
struct all_neighbors_builder_ivfpq : public all_neighbors_builder<T, IdxT> {
  all_neighbors_builder_ivfpq(raft::resources const& res,
                              size_t n_clusters,
                              size_t min_cluster_size,
                              size_t max_cluster_size,
                              size_t k,
                              graph_build_params::ivf_pq_params& params)
    : all_neighbors_builder<T, IdxT>(res, n_clusters, min_cluster_size, max_cluster_size, k),
      all_ivf_pq_params{params}
  {
  }

  void prepare_build_common(size_t num_cols)
  {
    candidate_k = std::min<IdxT>(
      std::max(static_cast<size_t>(this->k * all_ivf_pq_params.refinement_rate), this->k),
      this->min_cluster_size);

    candidate_distances_d.emplace(
      raft::make_device_matrix<T, IdxT, row_major>(this->res, this->max_cluster_size, candidate_k));
    candidate_neighbors_d.emplace(raft::make_device_matrix<IdxT, IdxT, row_major>(
      this->res, this->max_cluster_size, candidate_k));
    candidate_neighbors_h.emplace(
      raft::make_host_matrix<IdxT, IdxT, row_major>(this->max_cluster_size, candidate_k));

    refined_neighbors_h.emplace(
      raft::make_host_matrix<IdxT, IdxT, row_major>(this->max_cluster_size, this->k));
    refined_distances_h.emplace(
      raft::make_host_matrix<T, IdxT, row_major>(this->max_cluster_size, this->k));
  }

  void prepare_build(raft::host_matrix_view<const T, IdxT, row_major> dataset) override
  {
    // if dataset is on host, then allocate space for device data because ivfpq requires data to be
    // on device
    size_t num_cols = dataset.extent(1);
    data_d.emplace(
      raft::make_device_matrix<T, IdxT, row_major>(this->res, this->max_cluster_size, num_cols));
    prepare_build_common(num_cols);
  }

  void prepare_build(raft::device_matrix_view<const T, IdxT, row_major> dataset) override
  {
    prepare_build_common(dataset.extent(1));
  }

  // Actual build logic using ivfpq.
  // need device and host views of the dataset because ivfpq build and search uses the device view,
  // and refine uses the host view
  void build_knn_common(raft::device_matrix_view<const T, IdxT, row_major> dataset_d,
                        raft::host_matrix_view<const T, IdxT, row_major> dataset_h,
                        std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices,
                        global_graph_view<T, IdxT> global)
  {
    RAFT_EXPECTS(
      global.is_full_build() || inverted_indices.has_value(),
      "need valid inverted_indices for a batched (managed/host) global graph destination");

    size_t num_data_in_cluster = dataset_d.extent(0);
    size_t num_cols            = dataset_d.extent(1);

    auto index_ivfpq = ivf_pq::build(this->res, all_ivf_pq_params.build_params, dataset_d);

    auto candidate_distances_view = raft::make_device_matrix_view<T, IdxT>(
      candidate_distances_d.value().data_handle(), num_data_in_cluster, candidate_k);
    auto candidate_neighbors_view = raft::make_device_matrix_view<IdxT, IdxT>(
      candidate_neighbors_d.value().data_handle(), num_data_in_cluster, candidate_k);
    cuvs::neighbors::ivf_pq::search(this->res,
                                    all_ivf_pq_params.search_params,
                                    index_ivfpq,
                                    dataset_d,
                                    candidate_neighbors_view,
                                    candidate_distances_view);

    // copy candidate neighbors to host
    raft::copy(this->res,
               raft::make_host_vector_view(candidate_neighbors_h.value().data_handle(),
                                           num_data_in_cluster * candidate_k),
               raft::make_device_vector_view<const IdxT>(candidate_neighbors_view.data_handle(),
                                                         num_data_in_cluster * candidate_k));
    auto candidate_neighbors_h_view = raft::make_host_matrix_view<IdxT, IdxT>(
      candidate_neighbors_h.value().data_handle(), num_data_in_cluster, candidate_k);
    auto refined_distances_h_view = raft::make_host_matrix_view<T, IdxT>(
      refined_distances_h.value().data_handle(), num_data_in_cluster, this->k);
    auto refined_neighbors_h_view = raft::make_host_matrix_view<IdxT, IdxT>(
      refined_neighbors_h.value().data_handle(), num_data_in_cluster, this->k);

    refine(this->res,
           dataset_h,
           dataset_h,
           raft::make_const_mdspan(candidate_neighbors_h_view),
           refined_neighbors_h_view,
           refined_distances_h_view,
           all_ivf_pq_params.build_params.metric);

    if (!global.is_full_build()) {  // batched: merge this cluster into the global graph
      raft::copy(this->res,
                 raft::make_device_vector_view(this->batch_distances_d.value().data_handle(),
                                               num_data_in_cluster * this->k),
                 raft::make_host_vector_view<const T>(refined_distances_h_view.data_handle(),
                                                      num_data_in_cluster * this->k));

      this->template do_merge<IdxT>(
        refined_neighbors_h.value().view(),
        inverted_indices.value(),
        global,
        num_data_in_cluster,
        cuvs::distance::is_min_close(all_ivf_pq_params.build_params.metric));
    } else {  // full build: write directly to the device output in the sink
      const auto& direct = std::get<typename global_graph_view<T, IdxT>::direct_t>(global.dest);
      size_t num_rows    = num_data_in_cluster;
      raft::copy(this->res,
                 raft::make_device_vector_view(direct.first.data_handle(), num_rows * this->k),
                 raft::make_host_vector_view<const IdxT>(refined_neighbors_h_view.data_handle(),
                                                         num_rows * this->k));
      if (direct.second.has_value()) {
        raft::copy(
          this->res,
          raft::make_device_vector_view(direct.second.value().data_handle(), num_rows * this->k),
          raft::make_host_vector_view<const T>(refined_distances_h_view.data_handle(),
                                               num_rows * this->k));
      }
    }
  }

  void build_knn(raft::host_matrix_view<const T, IdxT, row_major> dataset,
                 std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices,
                 global_graph_view<T, IdxT> global) override
  {
    // we need data on device for ivfpq build and search.
    raft::copy(this->res,
               raft::make_device_vector_view(data_d.value().data_handle(), dataset.size()),
               raft::make_host_vector_view<const T>(dataset.data_handle(), dataset.size()));

    build_knn_common(raft::make_device_matrix_view<const T, IdxT, row_major>(
                       data_d.value().data_handle(), dataset.extent(0), dataset.extent(1)),
                     dataset,
                     inverted_indices,
                     global);
  }

  void build_knn(raft::device_matrix_view<const T, IdxT, row_major> dataset,
                 std::optional<raft::host_vector_view<IdxT, IdxT>> /*inverted_indices*/,
                 global_graph_view<T, IdxT> global) override
  {
    RAFT_EXPECTS(this->n_clusters <= 1,
                 "building all-neighbors knn graph with dataset on device is not supported with "
                 "batching (n_clusters > 1)");

    // we allocate host memory here and not in the prepare_build function because this function is
    // not called for batching
    auto dataset_h = raft::make_host_matrix<T, IdxT>(dataset.extent(0), dataset.extent(1));

    // we need data on host for refining
    raft::copy(this->res,
               raft::make_host_vector_view(dataset_h.data_handle(), dataset.size()),
               raft::make_device_vector_view<const T>(dataset.data_handle(), dataset.size()));

    build_knn_common(dataset,
                     raft::make_host_matrix_view<const T, IdxT, row_major>(
                       dataset_h.data_handle(), dataset.extent(0), dataset.extent(1)),
                     std::nullopt,
                     global);
  }

  graph_build_params::ivf_pq_params all_ivf_pq_params;
  size_t candidate_k;

  std::optional<raft::device_matrix<T, IdxT>> data_d;

  std::optional<raft::device_matrix<T, IdxT>> candidate_distances_d;
  std::optional<raft::device_matrix<IdxT, IdxT>> candidate_neighbors_d;
  std::optional<raft::host_matrix<IdxT, IdxT>> candidate_neighbors_h;

  std::optional<raft::host_matrix<IdxT, IdxT>> refined_neighbors_h;
  std::optional<raft::host_matrix<T, IdxT>> refined_distances_h;
};

template <typename T, typename IdxT = int64_t, typename DistEpilogueT = raft::identity_op>
struct all_neighbors_builder_nn_descent : public all_neighbors_builder<T, IdxT> {
  all_neighbors_builder_nn_descent(raft::resources const& res,
                                   size_t n_clusters,
                                   size_t min_cluster_size,
                                   size_t max_cluster_size,
                                   size_t k,
                                   graph_build_params::nn_descent_params& params,
                                   DistEpilogueT dist_epilogue = DistEpilogueT{})
    : all_neighbors_builder<T, IdxT>(res, n_clusters, min_cluster_size, max_cluster_size, k),
      nnd_params{params},
      dist_epilogue{dist_epilogue}
  {
  }

  template <typename Accessor>
  void prepare_build_common(mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> dataset)
  {
    if (nnd_params.graph_degree < this->k) {
      RAFT_LOG_WARN(
        "NN Descent's graph degree (%lu) has to be larger than or equal to k. Setting graph_degree "
        "to k (%lu).",
        nnd_params.graph_degree,
        this->k);
      nnd_params.graph_degree = this->k;
    }

    size_t extended_graph_degree, graph_degree;

    auto build_config                = nn_descent::detail::get_build_config(this->res,
                                                             nnd_params,
                                                             this->max_cluster_size,
                                                             static_cast<size_t>(dataset.extent(1)),
                                                             nnd_params.metric,
                                                             extended_graph_degree,
                                                             graph_degree);
    build_config.output_graph_degree = this->k;
    nnd_builder.emplace(this->res, build_config);
    int_graph.emplace(raft::make_host_matrix<int, IdxT, row_major>(
      this->max_cluster_size, static_cast<IdxT>(extended_graph_degree)));

    if constexpr (std::is_same_v<
                    DistEpilogueT,
                    cuvs::neighbors::detail::reachability::ReachabilityPostProcess<IdxT, T>>) {
      batch_core_distances.emplace(
        raft::make_device_vector<T, IdxT>(this->res, this->max_cluster_size));
    }
  }

  void prepare_build(raft::device_matrix_view<const T, IdxT, row_major> dataset) override
  {
    prepare_build_common(dataset);
  }

  void prepare_build(raft::host_matrix_view<const T, IdxT, row_major> dataset) override
  {
    prepare_build_common(dataset);
  }

  template <typename Accessor>
  void build_knn_common(mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> dataset,
                        std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices,
                        global_graph_view<T, IdxT> global)
  {
    RAFT_EXPECTS(global.is_full_build() || inverted_indices.has_value(),
                 "need valid inverted_indices for a batched global graph destination");

    using ReachabilityPP = cuvs::neighbors::detail::reachability::ReachabilityPostProcess<IdxT, T>;

    if (!global.is_full_build()) {
      bool return_distances      = true;
      size_t num_data_in_cluster = dataset.extent(0);
      if constexpr (std::is_same_v<DistEpilogueT, ReachabilityPP>) {
        // gather core dists
        raft::copy(this->res,
                   raft::make_device_vector_view(this->inverted_indices_d.value().data_handle(),
                                                 num_data_in_cluster),
                   raft::make_host_vector_view<const IdxT>(inverted_indices.value().data_handle(),
                                                           num_data_in_cluster));

        raft::matrix::gather(this->res,
                             raft::make_device_matrix_view<const T, IdxT>(
                               dist_epilogue.core_dists, dist_epilogue.n, 1),
                             raft::make_device_vector_view<const IdxT, IdxT>(
                               this->inverted_indices_d.value().data_handle(), num_data_in_cluster),
                             raft::make_device_matrix_view<T, IdxT>(
                               batch_core_distances.value().data_handle(), num_data_in_cluster, 1));

        nnd_builder.value().build(
          dataset.data_handle(),
          static_cast<int>(num_data_in_cluster),
          int_graph.value().data_handle(),
          return_distances,
          this->batch_distances_d.value().data_handle(),
          cuvs::neighbors::detail::reachability::ReachabilityPostProcess<int, T>{
            batch_core_distances.value().data_handle(), dist_epilogue.alpha, num_data_in_cluster});
      } else {
        nnd_builder.value().build(dataset.data_handle(),
                                  static_cast<int>(num_data_in_cluster),
                                  int_graph.value().data_handle(),
                                  return_distances,
                                  this->batch_distances_d.value().data_handle());
      }

      this->template do_merge<int, std::is_same_v<DistEpilogueT, ReachabilityPP>>(
        int_graph.value().view(),
        inverted_indices.value(),
        global,
        num_data_in_cluster,
        cuvs::distance::is_min_close(nnd_params.metric));
    } else {  // full build: write directly to the device output
      const auto& direct = std::get<typename global_graph_view<T, IdxT>::direct_t>(global.dest);
      size_t num_rows    = dataset.extent(0);

      if constexpr (std::is_same_v<DistEpilogueT, ReachabilityPP>) {
        nnd_builder.value().build(
          dataset.data_handle(),
          static_cast<int>(num_rows),
          int_graph.value().data_handle(),
          direct.second.has_value(),
          direct.second.value_or(raft::make_device_matrix<T, IdxT>(this->res, 0, 0).view())
            .data_handle(),
          cuvs::neighbors::detail::reachability::ReachabilityPostProcess<int, T>{
            dist_epilogue.core_dists, dist_epilogue.alpha, dist_epilogue.n});
      } else {
        nnd_builder.value().build(
          dataset.data_handle(),
          static_cast<int>(num_rows),
          int_graph.value().data_handle(),
          direct.second.has_value(),
          direct.second.value_or(raft::make_device_matrix<T, IdxT>(this->res, 0, 0).view())
            .data_handle(),
          dist_epilogue);
      }

      auto tmp_indices = raft::make_host_matrix<IdxT, IdxT>(int_graph.value().extent(0), this->k);

      // host slice
#pragma omp parallel for
      for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < this->k; j++) {
          tmp_indices(i, j) = static_cast<IdxT>(int_graph.value()(i, j));
        }
      }

      // copy to final device output
      raft::copy(
        this->res,
        raft::make_device_vector_view(direct.first.data_handle(), tmp_indices.extent(0) * this->k),
        raft::make_host_vector_view<const IdxT>(tmp_indices.data_handle(),
                                                tmp_indices.extent(0) * this->k));
    }
  }

  void build_knn(raft::host_matrix_view<const T, IdxT> dataset,
                 std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices,
                 global_graph_view<T, IdxT> global) override
  {
    build_knn_common(dataset, inverted_indices, global);
  }

  void build_knn(raft::device_matrix_view<const T, IdxT> dataset,
                 std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices,
                 global_graph_view<T, IdxT> global) override
  {
    RAFT_EXPECTS(this->n_clusters <= 1,
                 "building all-neighbors knn graph with dataset on device is not supported with "
                 "batching (n_clusters > 1)");
    build_knn_common(dataset, inverted_indices, global);
  }

  nn_descent::index_params nnd_params;
  nn_descent::detail::BuildConfig build_config;

  std::optional<nn_descent::detail::GNND<const T, int>> nnd_builder;
  std::optional<raft::host_matrix<int, IdxT>> int_graph;

  DistEpilogueT dist_epilogue;
  std::optional<raft::device_vector<T, IdxT>> batch_core_distances;
};

template <typename T, typename IdxT = int64_t, typename DistEpilogueT = raft::identity_op>
struct all_neighbors_builder_brute_force : public all_neighbors_builder<T, IdxT> {
  all_neighbors_builder_brute_force(raft::resources const& res,
                                    size_t n_clusters,
                                    size_t min_cluster_size,
                                    size_t max_cluster_size,
                                    size_t k,
                                    graph_build_params::brute_force_params& params,
                                    DistEpilogueT dist_epilogue = DistEpilogueT{})
    : all_neighbors_builder<T, IdxT>(res, n_clusters, min_cluster_size, max_cluster_size, k),
      bf_params{params},
      dist_epilogue{dist_epilogue}
  {
  }

  void prepare_build(raft::device_matrix_view<const T, IdxT, row_major> dataset) override {}

  void prepare_build(raft::host_matrix_view<const T, IdxT, row_major> dataset) override
  {
    // data needs to be on device for build - search
    size_t num_cols = dataset.extent(1);
    data_d.emplace(
      raft::make_device_matrix<T, IdxT, row_major>(this->res, this->max_cluster_size, num_cols));
    if constexpr (std::is_same_v<
                    DistEpilogueT,
                    cuvs::neighbors::detail::reachability::ReachabilityPostProcess<IdxT, T>>) {
      batch_core_distances.emplace(
        raft::make_device_vector<T, IdxT>(this->res, this->max_cluster_size));
    }
  }

  void build_knn_common(raft::device_matrix_view<const T, IdxT> dataset,
                        std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices,
                        global_graph_view<T, IdxT> global)
  {
    RAFT_EXPECTS(global.is_full_build() || inverted_indices.has_value(),
                 "need valid inverted_indices for a batched global graph destination");

    if (!global.is_full_build()) {
      size_t num_data_in_cluster = dataset.extent(0);
      using ReachabilityPP =
        cuvs::neighbors::detail::reachability::ReachabilityPostProcess<IdxT, T>;

      if constexpr (std::is_same_v<DistEpilogueT, ReachabilityPP>) {
        // gather core dists
        raft::copy(this->res,
                   raft::make_device_vector_view(this->inverted_indices_d.value().data_handle(),
                                                 num_data_in_cluster),
                   raft::make_host_vector_view<const IdxT>(inverted_indices.value().data_handle(),
                                                           num_data_in_cluster));

        raft::matrix::gather(this->res,
                             raft::make_device_matrix_view<const T, IdxT>(
                               dist_epilogue.core_dists, dist_epilogue.n, 1),
                             raft::make_device_vector_view<const IdxT, IdxT>(
                               this->inverted_indices_d.value().data_handle(), num_data_in_cluster),
                             raft::make_device_matrix_view<T, IdxT>(
                               batch_core_distances.value().data_handle(), num_data_in_cluster, 1));

        cuvs::neighbors::detail::tiled_brute_force_knn<T, IdxT, T, DistEpilogueT>(
          this->res,
          dataset.data_handle(),
          dataset.data_handle(),
          dataset.extent(0),
          dataset.extent(0),
          dataset.extent(1),
          this->k,
          this->batch_distances_d.value().data_handle(),
          this->batch_neighbors_d.value().data_handle(),
          bf_params.build_params.metric,
          2.0,
          0,
          0,
          nullptr,
          nullptr,
          nullptr,
          ReachabilityPP{
            batch_core_distances.value().data_handle(), dist_epilogue.alpha, num_data_in_cluster});
      } else {
        auto idx = cuvs::neighbors::brute_force::build(this->res, bf_params.build_params, dataset);

        cuvs::neighbors::brute_force::search(
          this->res,
          bf_params.search_params,
          idx,
          dataset,
          raft::make_device_matrix_view<IdxT, IdxT>(
            this->batch_neighbors_d.value().data_handle(), num_data_in_cluster, this->k),
          raft::make_device_matrix_view<T, IdxT>(
            this->batch_distances_d.value().data_handle(), num_data_in_cluster, this->k));
      }
      raft::copy(this->res,
                 raft::make_host_vector_view(this->batch_neighbors_h.value().data_handle(),
                                             num_data_in_cluster * this->k),
                 raft::make_device_vector_view<const IdxT>(
                   this->batch_neighbors_d.value().data_handle(), num_data_in_cluster * this->k));

      this->template do_merge<IdxT, std::is_same_v<DistEpilogueT, ReachabilityPP>>(
        this->batch_neighbors_h.value().view(),
        inverted_indices.value(),
        global,
        num_data_in_cluster,
        cuvs::distance::is_min_close(bf_params.build_params.metric));
    } else {  // full build: write directly to the device output in the sink
      const auto& direct = std::get<typename global_graph_view<T, IdxT>::direct_t>(global.dest);
      auto distances_view =
        direct.second.has_value()
          ? direct.second.value()
          : raft::make_device_matrix<T, IdxT>(this->res, dataset.extent(0), this->k).view();
      if constexpr (std::is_same_v<DistEpilogueT, raft::identity_op>) {
        auto idx = cuvs::neighbors::brute_force::build(this->res, bf_params.build_params, dataset);

        cuvs::neighbors::brute_force::search(
          this->res, bf_params.search_params, idx, dataset, direct.first, distances_view);
      } else {
        cuvs::neighbors::detail::tiled_brute_force_knn<T, IdxT, T, DistEpilogueT>(
          this->res,
          dataset.data_handle(),
          dataset.data_handle(),
          dataset.extent(0),
          dataset.extent(0),
          dataset.extent(1),
          this->k,
          distances_view.data_handle(),
          direct.first.data_handle(),
          bf_params.build_params.metric,
          2.0,
          0,
          0,
          nullptr,
          nullptr,
          nullptr,
          dist_epilogue);
      }
    }
  }

  void build_knn(raft::host_matrix_view<const T, IdxT> dataset,
                 std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices,
                 global_graph_view<T, IdxT> global) override
  {
    raft::copy(this->res,
               raft::make_device_vector_view(data_d.value().data_handle(), dataset.size()),
               raft::make_host_vector_view<const T>(dataset.data_handle(), dataset.size()));

    build_knn_common(raft::make_device_matrix_view<const T, IdxT, row_major>(
                       data_d.value().data_handle(), dataset.extent(0), dataset.extent(1)),
                     inverted_indices,
                     global);
  }

  void build_knn(raft::device_matrix_view<const T, IdxT> dataset,
                 std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices,
                 global_graph_view<T, IdxT> global) override
  {
    RAFT_EXPECTS(this->n_clusters <= 1,
                 "building all-neighbors knn graph with dataset on device is not supported with "
                 "batching (n_clusters > 1)");
    build_knn_common(dataset, inverted_indices, global);
  }

  graph_build_params::brute_force_params bf_params;
  DistEpilogueT dist_epilogue;

  std::optional<raft::device_matrix<T, IdxT, row_major>> data_d;
  std::optional<raft::device_vector<T, IdxT>> batch_core_distances;
};

template <typename T, typename IdxT, typename DistEpilogueT = raft::identity_op>
std::unique_ptr<all_neighbors_builder<T, IdxT>> get_knn_builder(
  const raft::resources& handle,
  const all_neighbors_params& params,
  size_t min_cluster_size,
  size_t max_cluster_size,
  size_t k,
  DistEpilogueT dist_epilogue = DistEpilogueT{})
{
  if (std::holds_alternative<graph_build_params::brute_force_params>(params.graph_build_params)) {
    auto brute_force_params =
      std::get<graph_build_params::brute_force_params>(params.graph_build_params);
    if (brute_force_params.build_params.metric != params.metric) {
      RAFT_LOG_WARN("Setting brute_force_params metric to metric given for batching algorithm");
      brute_force_params.build_params.metric = params.metric;
    }
    return std::make_unique<all_neighbors_builder_brute_force<T, IdxT, DistEpilogueT>>(
      handle,
      params.n_clusters,
      min_cluster_size,
      max_cluster_size,
      k,
      brute_force_params,
      dist_epilogue);

  } else if (std::holds_alternative<graph_build_params::nn_descent_params>(
               params.graph_build_params)) {
    auto nn_descent_params =
      std::get<graph_build_params::nn_descent_params>(params.graph_build_params);
    if (nn_descent_params.metric != params.metric) {
      RAFT_LOG_WARN("Setting nnd_params metric to metric given for batching algorithm");
      nn_descent_params.metric = params.metric;
    }
    return std::make_unique<all_neighbors_builder_nn_descent<T, IdxT, DistEpilogueT>>(
      handle,
      params.n_clusters,
      min_cluster_size,
      max_cluster_size,
      k,
      nn_descent_params,
      dist_epilogue);
  } else if (std::holds_alternative<graph_build_params::ivf_pq_params>(params.graph_build_params)) {
    auto ivf_pq_params = std::get<graph_build_params::ivf_pq_params>(params.graph_build_params);
    if (ivf_pq_params.build_params.metric != params.metric) {
      RAFT_LOG_WARN("Setting ivfpq_params metric to metric given for batching algorithm");
      ivf_pq_params.build_params.metric = params.metric;
    }
    return std::make_unique<all_neighbors_builder_ivfpq<T, IdxT>>(
      handle, params.n_clusters, min_cluster_size, max_cluster_size, k, ivf_pq_params);
  } else {
    RAFT_FAIL("Batch KNN build algos only supporting Brute Force, NN Descent, and IVFPQ");
  }
}

}  // namespace cuvs::neighbors::all_neighbors::detail
