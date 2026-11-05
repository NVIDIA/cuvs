/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Approximate nearest-centroid assignment for Lloyd's k-means (cuvs::cluster::kmeans::fit/predict).
 *
 * A CAGRA index is built over the current centroids at the start of EVERY Lloyd iteration (it is
 * never reused across iterations, because the centroids move each iteration), and
 * each sample is assigned to the nearest centroid found by a top-1 search of that index, instead of
 * the brute-force top-1 NN. The index always matches the centroids used in the iteration.
 *
 * Only float data with the L2Expanded / L2SqrtExpanded metrics is supported.
 */
#pragma once

#include "kmeans_common.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map.cuh>

#include <rmm/device_uvector.hpp>

#include <raft/util/cuda_utils.cuh>

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <type_traits>

namespace cuvs::cluster::kmeans::detail {

/** Throws if the ANN-related fields of `params` are invalid. Only checked when ANN is enabled. */
inline void validate_ann_params(cuvs::cluster::kmeans::params const& params)
{
  RAFT_EXPECTS(params.ann_graph_degree > 0, "ann_graph_degree must be > 0");
  RAFT_EXPECTS(params.ann_intermediate_graph_degree > 0,
               "ann_intermediate_graph_degree must be > 0");
  RAFT_EXPECTS(params.ann_itopk_size > 0, "ann_itopk_size must be > 0");
  RAFT_EXPECTS(params.ann_search_width > 0, "ann_search_width must be > 0");
}

/**
 * @brief Whether ANN assignment should be used for this call. Validates the ANN params when the
 * flag is set; unsupported configurations log a warning and return false (brute force is used).
 */
template <typename DataT>
bool use_ann_assignment(cuvs::cluster::kmeans::params const& params, int64_t n_clusters)
{
  if (!params.use_ann_for_assignment) { return false; }
  validate_ann_params(params);
  if constexpr (!std::is_same_v<DataT, float>) {
    RAFT_LOG_WARN("kmeans: use_ann_for_assignment requires float data; using brute force.");
    return false;
  } else {
    if (params.metric != cuvs::distance::DistanceType::L2Expanded &&
        params.metric != cuvs::distance::DistanceType::L2SqrtExpanded) {
      RAFT_LOG_WARN("kmeans: use_ann_for_assignment requires an L2 metric; using brute force.");
      return false;
    }
    if (n_clusters < 2) {
      RAFT_LOG_WARN("kmeans: use_ann_for_assignment requires n_clusters >= 2; using brute force.");
      return false;
    }
    return true;
  }
}

/**
 * @brief A CAGRA index over a copy of the centroids.
 *
 * `cagra::index` only stores a *view* of its dataset, and CAGRA needs padded rows, so the copy is
 * owned here to keep graph and vectors consistent.
 */
class centroid_ann_index {
 public:
  using index_type = cuvs::neighbors::cagra::device_padded_index<float, uint32_t>;

  centroid_ann_index() = default;

  /** Build (or rebuild) the index over the given centroids. */
  void build(raft::resources const& handle,
             cuvs::cluster::kmeans::params const& params,
             raft::device_matrix_view<const float, int64_t, raft::row_major> centroids)
  {
    int64_t n_clusters = centroids.extent(0);
    size_t max_degree  = static_cast<size_t>(std::max<int64_t>(1, n_clusters - 1));

    cuvs::neighbors::cagra::index_params build_params;
    build_params.metric       = cuvs::distance::DistanceType::L2Expanded;  // squared L2 distances
    build_params.graph_degree = std::min(params.ann_graph_degree, max_degree);
    build_params.intermediate_graph_degree =
      std::min(params.ann_intermediate_graph_degree, max_degree);
    build_params.attach_dataset_on_build = true;

    index_.reset();  // drop the old index before replacing the dataset it views
    snapshot_.reset();
    aligned_snapshot_.reset();
    const int64_t dim = centroids.extent(1);
    if (cuvs::neighbors::cagra_required_row_width<float>(static_cast<uint32_t>(dim)) ==
        static_cast<uint32_t>(dim)) {
      // Rows already have the stride CAGRA needs: make_device_padded_dataset refuses to copy in
      // this case, so take the snapshot ourselves and index a view of it.
      aligned_snapshot_.emplace(raft::make_device_matrix<float, int64_t>(handle, n_clusters, dim));
      raft::copy(aligned_snapshot_->data_handle(),
                 centroids.data_handle(),
                 static_cast<size_t>(n_clusters) * static_cast<size_t>(dim),
                 raft::resource::get_cuda_stream(handle));
      index_.emplace(cuvs::neighbors::cagra::build(
        handle,
        build_params,
        cuvs::neighbors::make_device_padded_dataset_view(handle, aligned_snapshot_->view())));
    } else {
      snapshot_ = cuvs::neighbors::make_device_padded_dataset(handle, centroids);
      index_.emplace(cuvs::neighbors::cagra::build(handle, build_params, snapshot_->view()));
    }
  }

  /**
   * Approximate top-1 nearest-centroid search (the ANN counterpart of `top_1_nn`): for each query,
   * writes the id of the nearest indexed centroid into `neighbors` [n_queries, 1] and its squared
   * L2 distance into `distances` [n_queries, 1]. The distance is computed from the centroid's
   * vector, so it is exact for the centroid that was found; only the choice of centroid is
   * approximate. An id may be invalid (>= n_clusters) if the search found nothing.
   */
  void top_1_nn_ann(raft::resources const& handle,
                    cuvs::cluster::kmeans::params const& params,
                    raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
                    raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
                    raft::device_matrix_view<float, int64_t, raft::row_major> distances) const
  {
    RAFT_EXPECTS(index_.has_value(), "centroid_ann_index::top_1_nn_ann called before build");

    cuvs::neighbors::cagra::search_params search_params;
    search_params.itopk_size   = params.ann_itopk_size;
    search_params.search_width = params.ann_search_width;
    search_params.max_queries  = 0;  // auto

    cuvs::neighbors::cagra::search(handle, search_params, *index_, queries, neighbors, distances);
  }

 private:
  // Owns the centroid snapshot the index views: either a padded copy, or (when the rows already
  // have the required stride) a plain device matrix.
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<float, int64_t>> snapshot_;
  std::optional<raft::device_matrix<float, int64_t>> aligned_snapshot_;
  std::optional<index_type> index_;
};

/**
 * @brief ANN counterpart of `minClusterAndDistanceCompute`: assigns every sample to its nearest
 * centroid using a top-1 search of the CAGRA index.
 *
 * @param[out] labels   [n] nearest centroid id
 * @param[out] sq_dist  [n] squared L2 distance to the assigned centroid
 */
template <typename IndexT>
void minClusterAndDistanceComputeANN(raft::resources const& handle,
                                     cuvs::cluster::kmeans::params const& params,
                                     centroid_ann_index const& ann_index,
                                     raft::device_matrix_view<const float, IndexT> X,
                                     IndexT* labels,
                                     float* sq_dist)
{
  const int64_t n   = X.extent(0);
  const int64_t dim = X.extent(1);

  auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(handle, n, 1);
  ann_index.top_1_nn_ann(
    handle,
    params,
    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(X.data_handle(), n, dim),
    neighbors.view(),
    raft::make_device_matrix_view<float, int64_t, raft::row_major>(sq_dist, n, 1));

  raft::linalg::map(
    handle,
    raft::make_device_vector_view<IndexT, int64_t>(labels, n),
    raft::cast_op<IndexT>{},
    raft::make_device_vector_view<const uint32_t, int64_t>(neighbors.data_handle(), n));
}

/**
 * @brief (Re)builds @p ann_index over `n_clusters` x `n_features` row-major centroids at
 * @p centroids. Only called when use_ann_assignment<DataT>() is true, which implies DataT = float.
 */
template <typename DataT>
void build_ann_index(raft::resources const& handle,
                     cuvs::cluster::kmeans::params const& params,
                     centroid_ann_index& ann_index,
                     const DataT* centroids,
                     int64_t n_clusters,
                     int64_t n_features)
{
  if constexpr (std::is_same_v<DataT, float>) {
    ann_index.build(handle,
                    params,
                    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
                      centroids, n_clusters, n_features));
  } else {
    RAFT_FAIL("ANN centroid assignment only supports float data");
  }
}

/**
 * @brief ANN counterpart of `process_batch` (kmeans_common.cuh) for one Lloyd batch: assigns via
 * the CAGRA index, accumulates weighted centroid sums / counts, and accumulates the weighted
 * inertia. The index must already be built over the current centroids.
 */
template <typename DataT, typename IndexT>
void process_batch_ann(raft::resources const& handle,
                       cuvs::cluster::kmeans::params const& params,
                       centroid_ann_index const& ann_index,
                       raft::device_matrix_view<const DataT, IndexT> batch_data,
                       raft::device_vector_view<const DataT, IndexT> batch_weights,
                       rmm::device_uvector<IndexT>& labels_buf,
                       rmm::device_uvector<DataT>& dist_buf,
                       rmm::device_uvector<char>& workspace,
                       raft::device_matrix_view<DataT, IndexT> centroid_sums,
                       raft::device_vector_view<DataT, IndexT> weight_per_cluster,
                       raft::device_scalar_view<DataT> clustering_cost,
                       rmm::device_uvector<char>& batch_workspace,
                       raft::device_scalar_view<DataT> batch_cost)
{
  if constexpr (std::is_same_v<DataT, float>) {
    cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();
    const IndexT n      = batch_data.extent(0);
    labels_buf.resize(n, stream);
    dist_buf.resize(n, stream);

    minClusterAndDistanceComputeANN<IndexT>(
      handle, params, ann_index, batch_data, labels_buf.data(), dist_buf.data());

    compute_centroid_adjustments(handle,
                                 batch_data,
                                 batch_weights,
                                 labels_buf.data(),
                                 static_cast<IndexT>(centroid_sums.extent(0)),
                                 centroid_sums,
                                 weight_per_cluster,
                                 batch_workspace,
                                 /*reset_sums=*/false);

    auto dist_view = raft::make_device_vector_view<float, IndexT>(dist_buf.data(), n);
    raft::linalg::map(handle,
                      dist_view,
                      raft::mul_op{},
                      raft::make_const_mdspan(dist_view),
                      batch_weights);  // weighted squared distance
    computeClusterCost(
      handle, dist_buf.data(), n, workspace, batch_cost, raft::identity_op{}, raft::add_op{});
    raft::linalg::add(clustering_cost.data_handle(),
                      clustering_cost.data_handle(),
                      batch_cost.data_handle(),
                      1,
                      stream);
  } else {
    RAFT_FAIL("ANN centroid assignment only supports float data");
  }
}

/**
 * @brief ANN counterpart of the brute-force body of `kmeans_predict`: builds an index over the
 * (fixed) centroids, writes each sample's nearest-centroid label, and the weighted inertia.
 */
template <typename DataT, typename IndexT>
void kmeans_predict_ann(raft::resources const& handle,
                        cuvs::cluster::kmeans::params const& params,
                        raft::device_matrix_view<const DataT, IndexT> X,
                        raft::device_vector_view<const DataT, IndexT> weight,
                        raft::device_matrix_view<const DataT, IndexT> centroids,
                        IndexT* labels,
                        rmm::device_uvector<char>& workspace,
                        DataT& inertia)
{
  if constexpr (std::is_same_v<DataT, float>) {
    cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();
    const IndexT n      = X.extent(0);
    centroid_ann_index ann_index;
    build_ann_index<DataT>(
      handle, params, ann_index, centroids.data_handle(), centroids.extent(0), X.extent(1));
    rmm::device_uvector<float> sq_dist(n, stream);
    minClusterAndDistanceComputeANN<IndexT>(handle, params, ann_index, X, labels, sq_dist.data());
    auto dist_view = raft::make_device_vector_view<float, IndexT>(sq_dist.data(), n);
    raft::linalg::map(
      handle, dist_view, raft::mul_op{}, raft::make_const_mdspan(dist_view), weight);
    rmm::device_scalar<float> cost(stream);
    computeClusterCost(handle,
                       sq_dist.data(),
                       n,
                       workspace,
                       raft::make_device_scalar_view(cost.data()),
                       raft::identity_op{},
                       raft::add_op{});
    inertia = cost.value(stream);
  } else {
    RAFT_FAIL("ANN centroid assignment only supports float data");
  }
}

}  // namespace cuvs::cluster::kmeans::detail
