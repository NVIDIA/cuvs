/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "../../core/omp_wrapper.hpp"
#include "../detail/reachability.cuh"
#include "all_neighbors_builder.cuh"
#include "raft/core/logger_macros.hpp"
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/refine.hpp>
#include <optional>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/managed_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/matrix/init.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/util/cudart_utils.hpp>
#include <variant>

namespace cuvs::neighbors::all_neighbors::detail {
using namespace cuvs::neighbors;

template <typename T, typename IdxT>
void reset_global_matrices(raft::resources const& res,
                           cuvs::distance::DistanceType metric,
                           raft::managed_matrix_view<IdxT, IdxT> global_neighbors,
                           raft::managed_matrix_view<T, IdxT> global_distances)
{
  size_t num_rows = static_cast<size_t>(global_neighbors.extent(0));
  size_t k        = static_cast<size_t>(global_neighbors.extent(1));

  bool select_min = cuvs::distance::is_min_close(metric);
  IdxT global_neighbors_fill_value =
    select_min ? std::numeric_limits<IdxT>::max() : std::numeric_limits<IdxT>::min();
  T global_distances_fill_value =
    select_min ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();

  raft::matrix::fill(
    res,
    raft::make_device_matrix_view<IdxT, IdxT>(global_neighbors.data_handle(), num_rows, k),
    global_neighbors_fill_value);
  raft::matrix::fill(
    res,
    raft::make_device_matrix_view<T, IdxT>(global_distances.data_handle(), num_rows, k),
    global_distances_fill_value);
}

/**
 * Run balanced kmeans on a subsample of the dataset to get centroids.
 * Arguments:
 * - [in] res: raft resource
 * - [in] metric
 * - [in] dataset [num_rows x num_cols]: entire dataset on host
 * - [out] centroids [n_clusters x num_cols] : centroid vectors
 */
template <typename T, typename IdxT>
void get_centroids_on_data_subsample(raft::resources const& res,
                                     cuvs::distance::DistanceType metric,
                                     raft::host_matrix_view<const T, IdxT, row_major> dataset,
                                     raft::device_matrix_view<T, IdxT> centroids)
{
  size_t num_rows       = static_cast<size_t>(dataset.extent(0));
  size_t num_cols       = static_cast<size_t>(dataset.extent(1));
  size_t n_clusters     = centroids.extent(0);
  size_t num_subsamples = std::min(static_cast<size_t>(num_rows / n_clusters), 50000lu);

  if (num_subsamples <= 1000) {
    // heuristically running kmeans for something this small doesn't work well
    num_subsamples = std::min(num_rows, 5000lu);
  }
  auto dataset_subsample_d = raft::make_device_matrix<T, IdxT>(res, num_subsamples, num_cols);

  raft::matrix::sample_rows<T, IdxT>(
    res, raft::random::RngState{0}, dataset, dataset_subsample_d.view());
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.metric = metric;

  cuvs::cluster::kmeans::fit(
    res, kmeans_params, raft::make_const_mdspan(dataset_subsample_d.view()), centroids);
}

template <typename T, typename IdxT>
void single_gpu_assign_clusters(
  raft::resources const& res,
  size_t overlap_factor,
  size_t n_clusters,
  size_t n_rows_per_batch,
  size_t base_row_offset,
  raft::host_matrix_view<const T, IdxT, row_major> dataset,
  raft::device_matrix_view<T, IdxT, raft::row_major> centroids,
  cuvs::distance::DistanceType metric,
  raft::host_matrix_view<IdxT, IdxT, raft::row_major> global_nearest_cluster)
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));

  // number of batches for this GPU to process
  size_t num_batches = n_clusters;

  auto dataset_batch_d =
    raft::make_device_matrix<T, IdxT, raft::row_major>(res, n_rows_per_batch, num_cols);

  auto nearest_clusters_idx_d =
    raft::make_device_matrix<IdxT, int64_t, raft::row_major>(res, n_rows_per_batch, overlap_factor);
  auto nearest_clusters_dist_d =
    raft::make_device_matrix<T, int64_t, raft::row_major>(res, n_rows_per_batch, overlap_factor);

  std::optional<raft::device_vector_view<const T, int64_t>> norms_view;
  cuvs::neighbors::brute_force::index<T> brute_force_index(res, centroids, norms_view, metric);

  auto stream = resource::get_cuda_stream(res);

  for (size_t i = 0; i < num_batches; i++) {
    size_t row_offset              = n_rows_per_batch * i + base_row_offset;
    size_t n_rows_of_current_batch = std::min(n_rows_per_batch, num_rows - row_offset);
    raft::copy(res,
               raft::make_device_vector_view(dataset_batch_d.data_handle(),
                                             n_rows_of_current_batch * num_cols),
               raft::make_host_vector_view<const T>(dataset.data_handle() + row_offset * num_cols,
                                                    n_rows_of_current_batch * num_cols));

    // n_clusters is usually not large, so okay to do this brute-force
    cuvs::neighbors::brute_force::search(res,
                                         brute_force_index,
                                         raft::make_const_mdspan(dataset_batch_d.view()),
                                         nearest_clusters_idx_d.view(),
                                         nearest_clusters_dist_d.view());
    raft::copy(res,
               raft::make_host_vector_view(
                 global_nearest_cluster.data_handle() + row_offset * overlap_factor,
                 n_rows_of_current_batch * overlap_factor),
               raft::make_device_vector_view<const IdxT>(nearest_clusters_idx_d.data_handle(),
                                                         n_rows_of_current_batch * overlap_factor));
  }
}

/**
 * Assign each data point to top overlap_factor number of clusters. Loads the data in batches
 * onto device for efficiency. Arguments:
 * - [in] res: raft resource
 * - [in] params: params for graph building
 * - [in] dataset [num_rows x num_cols]: entire dataset located on host memory
 * - [in] centroids [n_clusters x num_cols] : centroid vectors
 * - [out] global_nearest_cluster [num_rows X overlap_factor] : top overlap_factor closest
 * clusters for each data point
 */
template <typename T, typename IdxT>
void assign_clusters(raft::resources const& res,
                     const all_neighbors_params& params,
                     raft::host_matrix_view<const T, IdxT, row_major> dataset,
                     raft::device_matrix_view<T, IdxT, raft::row_major> centroids,
                     raft::host_matrix_view<IdxT, IdxT, raft::row_major> global_nearest_cluster)
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));

  auto centroids_h = raft::make_host_matrix<T, IdxT>(params.n_clusters, num_cols);
  raft::copy(
    res,
    raft::make_host_vector_view(centroids_h.data_handle(), params.n_clusters * num_cols),
    raft::make_device_vector_view<const T>(centroids.data_handle(), params.n_clusters * num_cols));

  size_t n_rows_per_cluster = (num_rows + params.n_clusters - 1) / params.n_clusters;

  if (raft::resource::is_multi_gpu(res)) {
    int num_ranks = raft::resource::get_num_ranks(res);
    // multi-gpu assign clusters
    size_t clusters_per_rank = params.n_clusters / num_ranks;
    size_t rem               = params.n_clusters - clusters_per_rank * num_ranks;

#pragma omp parallel for num_threads(num_ranks)
    for (int rank = 0; rank < num_ranks; rank++) {
      auto dev_res = raft::resource::set_current_device_to_rank(res, rank);

      auto centroids_matrix =
        raft::make_device_matrix<T, IdxT>(dev_res, params.n_clusters, num_cols);
      raft::copy(
        dev_res,
        raft::make_device_vector_view(centroids_matrix.data_handle(), params.n_clusters * num_cols),
        raft::make_host_vector_view<const T>(centroids_h.data_handle(),
                                             params.n_clusters * num_cols));

      size_t base_cluster_idx = rank * clusters_per_rank + std::min((size_t)rank, rem);

      size_t n_clusters_for_this_rank =
        (size_t)rank < rem ? clusters_per_rank + 1 : clusters_per_rank;
      size_t base_row_offset_for_this_rank = n_rows_per_cluster * base_cluster_idx;

      single_gpu_assign_clusters(dev_res,
                                 params.overlap_factor,
                                 n_clusters_for_this_rank,
                                 n_rows_per_cluster,
                                 base_row_offset_for_this_rank,
                                 dataset,
                                 centroids_matrix.view(),
                                 params.metric,
                                 global_nearest_cluster);
    }
  } else {
    single_gpu_assign_clusters(res,
                               params.overlap_factor,
                               params.n_clusters,
                               n_rows_per_cluster,
                               0,
                               dataset,
                               centroids,
                               params.metric,
                               global_nearest_cluster);
  }
}

/**
 * Getting data indices that belong to cluster
 * Arguments:
 * - [in] res: raft resource
 * - [in] global_nearest_cluster [num_rows X overlap_factor] : top overlap_factor closest
 * clusters for each data point
 * - [out] inverted_indices [num_rows x overlap_factor sized vector] : vector for data indices
 * for each cluster
 * - [out] cluster_sizes [n_cluster] : cluster size for each cluster
 * - [out] cluster_offsets [n_cluster] : offset in inverted_indices for each cluster
 */
template <typename IdxT = int64_t>
void get_inverted_indices(raft::resources const& res,
                          raft::host_matrix_view<IdxT, IdxT> global_nearest_cluster,
                          raft::host_vector_view<IdxT, IdxT> inverted_indices,
                          raft::host_vector_view<IdxT, IdxT> cluster_sizes,
                          raft::host_vector_view<IdxT, IdxT> cluster_offsets)
{
  // build sparse inverted indices and get number of data points for each cluster
  size_t num_rows       = global_nearest_cluster.extent(0);
  size_t overlap_factor = global_nearest_cluster.extent(1);
  size_t n_clusters     = cluster_sizes.extent(0);

  auto local_offsets = raft::make_host_vector<IdxT>(n_clusters);

  std::fill(cluster_sizes.data_handle(), cluster_sizes.data_handle() + n_clusters, 0);
  std::fill(local_offsets.data_handle(), local_offsets.data_handle() + n_clusters, 0);

  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < overlap_factor; j++) {
      IdxT cluster_id = global_nearest_cluster(i, j);
      cluster_sizes(cluster_id) += 1;
    }
  }

  cluster_offsets(0) = 0;
  for (size_t i = 1; i < n_clusters; i++) {
    cluster_offsets(i) = cluster_offsets(i - 1) + cluster_sizes(i - 1);
  }
  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < overlap_factor; j++) {
      IdxT cluster_id = global_nearest_cluster(i, j);
      inverted_indices(cluster_offsets(cluster_id) + local_offsets(cluster_id)) = i;
      local_offsets(cluster_id) += 1;
    }
  }
}

template <typename IdxT = int64_t>
void get_min_max_cluster_size(size_t k,
                              size_t& max_cluster_size,
                              size_t& min_cluster_size,
                              raft::host_vector_view<IdxT, IdxT> cluster_sizes)
{
  max_cluster_size = 0;
  min_cluster_size = std::numeric_limits<size_t>::max();

  size_t n_clusters = cluster_sizes.extent(0);
  max_cluster_size  = static_cast<size_t>(
    *std::max_element(cluster_sizes.data_handle(), cluster_sizes.data_handle() + n_clusters));
  min_cluster_size = static_cast<size_t>(*std::min_element(
    cluster_sizes.data_handle(), cluster_sizes.data_handle() + n_clusters, [k](size_t a, size_t b) {
      return (a > k && (b <= k || a < b));  // Compare only elements larger than k
    }));
}

template <typename T, typename IdxT>
void single_gpu_batch_build(const raft::resources& handle,
                            raft::host_matrix_view<const T, IdxT, row_major> dataset,
                            detail::all_neighbors_builder<T, IdxT>& knn_builder,
                            size_t n_clusters,
                            raft::managed_matrix_view<IdxT, IdxT> global_neighbors,
                            raft::managed_matrix_view<T, IdxT> global_distances,
                            raft::host_vector_view<IdxT, IdxT, row_major> cluster_sizes,
                            raft::host_vector_view<IdxT, IdxT, row_major> cluster_offsets,
                            raft::host_vector_view<IdxT, IdxT, row_major> inverted_indices)
{
  size_t num_rows = dataset.extent(0);
  size_t num_cols = dataset.extent(1);

  auto cluster_data =
    raft::make_host_matrix<T, IdxT, row_major>(knn_builder.max_cluster_size, num_cols);

  knn_builder.prepare_build(dataset);

  for (size_t cluster_id = 0; cluster_id < n_clusters; cluster_id++) {
    size_t num_data_in_cluster = cluster_sizes(cluster_id);
    size_t offset              = cluster_offsets(cluster_id);
    if (num_data_in_cluster < knn_builder.k) {
      // for the unlikely event where clustering was done lopsidedly and this cluster has less than
      // k data
      continue;
    }

    // gathering dataset from host to host memory
    // cluster_data is on host for now because NN Descent allocates a new device matrix for data of
    // half fp regardless of whether data is on host or device
#pragma omp parallel for
    for (size_t i = 0; i < num_data_in_cluster; i++) {
      for (size_t j = 0; j < num_cols; j++) {
        size_t global_row  = inverted_indices(offset + i);
        cluster_data(i, j) = dataset(global_row, j);
      }
    }

    auto cluster_data_view = raft::make_host_matrix_view<const T, IdxT>(
      cluster_data.data_handle(), num_data_in_cluster, num_cols);
    auto inverted_indices_view = raft::make_host_vector_view<IdxT, IdxT>(
      inverted_indices.data_handle() + offset, num_data_in_cluster);

    knn_builder.build_knn(
      cluster_data_view, inverted_indices_view, global_neighbors, global_distances);
  }
}

template <typename T, typename IdxT, typename DistEpilogueT = raft::identity_op>
void multi_gpu_batch_build(const raft::resources& handle,
                           const all_neighbors_params& params,
                           raft::host_matrix_view<const T, IdxT, row_major> dataset,
                           raft::managed_matrix_view<IdxT, IdxT> global_neighbors,
                           raft::managed_matrix_view<T, IdxT> global_distances,
                           raft::host_vector_view<IdxT, IdxT, row_major> cluster_sizes,
                           raft::host_vector_view<const IdxT, IdxT, row_major> cluster_offsets_c,
                           raft::host_vector_view<IdxT, IdxT, row_major> inverted_indices,
                           DistEpilogueT dist_epilogue = DistEpilogueT{})
{
  size_t num_rows = dataset.extent(0);
  size_t num_cols = dataset.extent(1);
  size_t k        = global_neighbors.extent(1);

  int num_ranks = raft::resource::get_num_ranks(handle);

  size_t clusters_per_rank = params.n_clusters / num_ranks;
  size_t rem               = params.n_clusters - clusters_per_rank * num_ranks;

  auto cluster_offsets = raft::make_host_vector<IdxT, IdxT>(cluster_offsets_c.size());
  raft::copy(handle, cluster_offsets.view(), cluster_offsets_c);

  using ReachabilityPP = cuvs::neighbors::detail::reachability::ReachabilityPostProcess<IdxT, T>;
  const bool mutual_reach_dist = std::is_same_v<DistEpilogueT, ReachabilityPP>;
  std::optional<raft::host_vector<T, IdxT>> core_distances_h;
  if constexpr (mutual_reach_dist) {
    core_distances_h.emplace(raft::make_host_vector<T, IdxT>(num_rows));
    raft::copy(handle,
               core_distances_h.value().view(),
               raft::make_device_vector_view<const T, IdxT>(dist_epilogue.core_dists, num_rows));
  }

  // Ensure all async copies complete before starting parallel region
  raft::resource::sync_stream(handle);

#pragma omp parallel for num_threads(num_ranks)
  for (int rank = 0; rank < num_ranks; rank++) {
    auto dev_res = raft::resource::set_current_device_to_rank(handle, rank);

    // This part is to distribute clusters across ranks as equally as possible
    // E.g. if we had 5 clusters and 3 ranks, instead of splitting into 1, 1, 3
    // we split into 2, 2, 1
    size_t num_data_for_this_rank = 0;
    size_t base_cluster_idx       = rank * clusters_per_rank + std::min((size_t)rank, rem);
    size_t num_clusters_for_this_rank =
      (size_t)rank < rem ? clusters_per_rank + 1 : clusters_per_rank;
    if (num_clusters_for_this_rank == 0) {
      // Happens in the case when num_ranks > n_clusters.
      RAFT_LOG_WARN(
        "Rank %d is not used for computation. This happens because the total number of ranks (%d) "
        "> n_clusters (%lu). Consider increasing n_clusters or reduce the number of GPUs for "
        "better utilization.",
        rank,
        num_ranks,
        params.n_clusters);
      continue;
    }
    for (size_t p = 0; p < num_clusters_for_this_rank; p++) {
      num_data_for_this_rank += cluster_sizes(base_cluster_idx + p);
    }

    size_t rank_offset = cluster_offsets(base_cluster_idx);

    // remap offsets for each rank
    for (size_t p = 0; p < num_clusters_for_this_rank; p++) {
      cluster_offsets(base_cluster_idx + p) -= rank_offset;
    }

    auto cluster_sizes_for_this_rank = raft::make_host_vector_view<IdxT, IdxT>(
      cluster_sizes.data_handle() + base_cluster_idx, num_clusters_for_this_rank);
    auto cluster_offsets_for_this_rank = raft::make_host_vector_view<IdxT, IdxT>(
      cluster_offsets.data_handle() + base_cluster_idx, num_clusters_for_this_rank);
    auto inverted_indices_for_this_rank = raft::make_host_vector_view<IdxT, IdxT>(
      inverted_indices.data_handle() + rank_offset, num_data_for_this_rank);

    size_t max_cluster_size, min_cluster_size;
    get_min_max_cluster_size(k, max_cluster_size, min_cluster_size, cluster_sizes_for_this_rank);

    std::optional<raft::device_vector<T, IdxT>> core_distances_d_for_rank;
    auto dist_epilgogue_for_rank = [&]() {
      if constexpr (mutual_reach_dist) {
        core_distances_d_for_rank.emplace(raft::make_device_vector<T, IdxT>(dev_res, num_rows));
        raft::copy(dev_res,
                   core_distances_d_for_rank.value().view(),
                   raft::make_const_mdspan(core_distances_h.value().view()));
        return ReachabilityPP{
          core_distances_d_for_rank.value().data_handle(), dist_epilogue.alpha, num_rows};
      } else {
        return dist_epilogue;
      }
    }();

    std::unique_ptr<all_neighbors_builder<T, IdxT>> knn_builder =
      get_knn_builder<T, IdxT>(dev_res,
                               params,
                               min_cluster_size,
                               max_cluster_size,
                               k,
                               std::nullopt,
                               std::nullopt,
                               dist_epilgogue_for_rank);

    single_gpu_batch_build(dev_res,
                           dataset,
                           *knn_builder,
                           num_clusters_for_this_rank,
                           global_neighbors,
                           global_distances,
                           cluster_sizes_for_this_rank,
                           cluster_offsets_for_this_rank,
                           inverted_indices_for_this_rank);
  }
}

/* Holds necessary vectors to avoid recomputation when calculating mutual rechability distances */
template <typename IdxT>
struct BatchBuildAux {
  raft::host_vector<IdxT, IdxT> cluster_sizes;
  raft::host_vector<IdxT, IdxT> cluster_offsets;
  raft::host_vector<IdxT, IdxT> inverted_indices;
  bool is_computed = false;

  BatchBuildAux(IdxT n_clusters, IdxT num_rows, IdxT overlap_factor)
    : cluster_sizes(raft::make_host_vector<IdxT, IdxT>(n_clusters)),
      cluster_offsets(raft::make_host_vector<IdxT, IdxT>(n_clusters)),
      inverted_indices(raft::make_host_vector<IdxT, IdxT>(num_rows * overlap_factor))
  {
  }
};

template <typename T,
          typename IdxT,
          typename IndicesMdspan,
          typename DistancesMdspan = raft::device_matrix_view<T, IdxT, row_major>,
          typename DistEpilogueT   = raft::identity_op>
void batch_build(const raft::resources& handle,
                 const all_neighbors_params& params,
                 raft::host_matrix_view<const T, IdxT, row_major> dataset,
                 IndicesMdspan indices,
                 std::optional<DistancesMdspan> distances = std::nullopt,
                 BatchBuildAux<IdxT>* aux_vectors         = nullptr,
                 DistEpilogueT dist_epilogue              = DistEpilogueT{})
{
  if (raft::resource::is_multi_gpu(handle)) {
    // For efficient CPU-computation of omp parallel for regions per GPU
    cuvs::core::omp::set_nested(1);
  }

  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));
  size_t k        = indices.extent(1);

  RAFT_EXPECTS(params.n_clusters > params.overlap_factor,
               "overlap_factor should be smaller than n_clusters. We recommend starting from "
               "overlap_factor=2 and gradually increase it for better knn graph recall.");

  std::optional<raft::host_vector<IdxT, IdxT>> local_inverted_indices;
  std::optional<raft::host_vector<IdxT, IdxT>> local_cluster_sizes;
  std::optional<raft::host_vector<IdxT, IdxT>> local_cluster_offsets;

  auto inverted_indices_view =
    aux_vectors != nullptr
      ? aux_vectors->inverted_indices.view()
      : local_inverted_indices
          .emplace(raft::make_host_vector<IdxT, IdxT>(num_rows * params.overlap_factor))
          .view();

  auto cluster_sizes_view =
    aux_vectors != nullptr
      ? aux_vectors->cluster_sizes.view()
      : local_cluster_sizes.emplace(raft::make_host_vector<IdxT, IdxT>(params.n_clusters)).view();

  auto cluster_offsets_view =
    aux_vectors != nullptr
      ? aux_vectors->cluster_offsets.view()
      : local_cluster_offsets.emplace(raft::make_host_vector<IdxT, IdxT>(params.n_clusters)).view();

  if (aux_vectors == nullptr || !aux_vectors->is_computed) {
    // clustering step is only computed when needed
    auto centroids = raft::make_device_matrix<T, IdxT>(handle, params.n_clusters, num_cols);
    get_centroids_on_data_subsample<T, IdxT>(handle, params.metric, dataset, centroids.view());

    auto global_nearest_cluster =
      raft::make_host_matrix<IdxT, IdxT>(num_rows, params.overlap_factor);
    assign_clusters<T, IdxT>(
      handle, params, dataset, centroids.view(), global_nearest_cluster.view());

    get_inverted_indices(handle,
                         global_nearest_cluster.view(),
                         inverted_indices_view,
                         cluster_sizes_view,
                         cluster_offsets_view);
    if (aux_vectors != nullptr) { aux_vectors->is_computed = true; }
  }

  auto global_neighbors = raft::make_managed_matrix<IdxT, IdxT>(handle, num_rows, k);
  auto global_distances = raft::make_managed_matrix<T, IdxT>(handle, num_rows, k);

  if (raft::resource::is_multi_gpu(handle)) {
    // Check if any GPU is Turing (SM 7.5) or older. These architectures have issues with
    // multi-GPU managed memory coherence for concurrent writes. Force CPU-resident memory
    // to ensure all GPUs access through host memory, avoiding page migration issues.
    // Ampere (SM 8.0+) and newer architectures handle this correctly.
    int num_ranks         = raft::resource::get_num_ranks(handle);
    bool needs_workaround = false;
    int original_device;
    RAFT_CUDA_TRY(cudaGetDevice(&original_device));
    for (int rank = 0; rank < num_ranks; rank++) {
      raft::resource::set_current_device_to_rank(handle, rank);
      int device_id;
      RAFT_CUDA_TRY(cudaGetDevice(&device_id));
      int major = 0;
      RAFT_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
      if (major < 8) {
        needs_workaround = true;
        break;
      }  // Turing is 7.x, Ampere is 8.x
    }
    RAFT_CUDA_TRY(cudaSetDevice(original_device));

    if (needs_workaround) {
      RAFT_LOG_DEBUG("Applying managed memory workaround for pre-Ampere GPU architecture");
      cudaMemLocation cpu_location = {cudaMemLocationTypeHost, 0};
      RAFT_CUDA_TRY(cudaMemAdvise(global_neighbors.data_handle(),
                                  num_rows * k * sizeof(IdxT),
                                  cudaMemAdviseSetPreferredLocation,
                                  cpu_location));
      RAFT_CUDA_TRY(cudaMemAdvise(global_distances.data_handle(),
                                  num_rows * k * sizeof(T),
                                  cudaMemAdviseSetPreferredLocation,
                                  cpu_location));
    }
  }

  reset_global_matrices(handle, params.metric, global_neighbors.view(), global_distances.view());

  if (raft::resource::is_multi_gpu(handle)) {
    // For multi-GPU: sync the stream to ensure fill completes before other GPUs access
    // the managed memory.
    raft::resource::sync_stream(handle);

    multi_gpu_batch_build(handle,
                          params,
                          dataset,
                          global_neighbors.view(),
                          global_distances.view(),
                          cluster_sizes_view,
                          raft::make_const_mdspan(cluster_offsets_view),
                          inverted_indices_view,
                          dist_epilogue);
  } else {
    size_t max_cluster_size, min_cluster_size;
    get_min_max_cluster_size(k, max_cluster_size, min_cluster_size, cluster_sizes_view);
    std::unique_ptr<all_neighbors_builder<T, IdxT>> knn_builder =
      get_knn_builder<T, IdxT>(handle,
                               params,
                               min_cluster_size,
                               max_cluster_size,
                               k,
                               std::nullopt,
                               std::nullopt,
                               dist_epilogue);
    single_gpu_batch_build(handle,
                           dataset,
                           *knn_builder,
                           params.n_clusters,
                           global_neighbors.view(),
                           global_distances.view(),
                           cluster_sizes_view,
                           cluster_offsets_view,
                           inverted_indices_view);
  }

  raft::copy(indices.data_handle(),
             global_neighbors.data_handle(),
             num_rows * k,
             raft::resource::get_cuda_stream(handle));
  if (distances.has_value()) {
    raft::copy(distances.value().data_handle(),
               global_distances.data_handle(),
               num_rows * k,
               raft::resource::get_cuda_stream(handle));
  }
}

}  // namespace cuvs::neighbors::all_neighbors::detail
