/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "detail/kmeans.cuh"
#include "kmeans_mg.hpp"
#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/norm.cuh>

#include <optional>

namespace cuvs::cluster::kmeans {

/**
 * Functor used for sampling centroids
 */
template <typename DataT, typename IndexT>
using SamplingOp = cuvs::cluster::kmeans::detail::SamplingOp<DataT, IndexT>;

/**
 * Functor used to extract the index from a KeyValue pair
 * storing both index and a distance.
 */
template <typename IndexT, typename DataT>
using KeyValueIndexOp = cuvs::cluster::kmeans::detail::KeyValueIndexOp<IndexT, DataT>;

/*
 * @brief Main function used to fit KMeans (after cluster initialization)
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     [in] Initial cluster centers.
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 * @param[in]     workspace     Temporary workspace buffer which can get resized
 */
template <typename DataT, typename IndexT>
void fit_main(raft::resources const& handle,
              const kmeans::params& params,
              raft::device_matrix_view<const DataT, IndexT> X,
              raft::device_vector_view<const DataT, IndexT> sample_weights,
              raft::device_matrix_view<DataT, IndexT> centroids,
              raft::host_scalar_view<DataT> inertia,
              raft::host_scalar_view<IndexT> n_iter,
              rmm::device_uvector<char>& workspace);

#define EXTERN_TEMPLATE_FIT_MAIN(DataT, IndexT)                   \
  extern template void fit_main<DataT, IndexT>(                   \
    raft::resources const& handle,                                \
    const kmeans::params& params,                                 \
    raft::device_matrix_view<const DataT, IndexT> X,              \
    raft::device_vector_view<const DataT, IndexT> sample_weights, \
    raft::device_matrix_view<DataT, IndexT> centroids,            \
    raft::host_scalar_view<DataT> inertia,                        \
    raft::host_scalar_view<IndexT> n_iter,                        \
    rmm::device_uvector<char>& workspace);

EXTERN_TEMPLATE_FIT_MAIN(double, int)
EXTERN_TEMPLATE_FIT_MAIN(double, int64_t)
EXTERN_TEMPLATE_FIT_MAIN(float, int64_t)
EXTERN_TEMPLATE_FIT_MAIN(float, int)

#undef EXTERN_TEMPLATE_FIT_MAIN
/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.cuh>
 *   #include <cuvs/cluster/kmeans_types.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *    cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               std::nullopt,
 *               centroids,
 *               raft::make_scalar_view(&inertia),
 *               raft::make_scalar_view(&n_iter));
 * @endcode
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     [in] When init is InitMethod::Array, use
 *                              centroids as the initial cluster centers.
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
template <typename DataT, typename IndexT>
void fit(raft::resources const& handle,
         const kmeans::params& params,
         raft::device_matrix_view<const DataT, IndexT> X,
         std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
         raft::device_matrix_view<DataT, IndexT> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<IndexT> n_iter);

#define EXTERN_TEMPLATE_FIT(DataT, IndexT)                                      \
  extern template void fit<DataT, IndexT>(                                      \
    raft::resources const& handle,                                              \
    const kmeans::params& params,                                               \
    raft::device_matrix_view<const DataT, IndexT> X,                            \
    std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight, \
    raft::device_matrix_view<DataT, IndexT> centroids,                          \
    raft::host_scalar_view<DataT> inertia,                                      \
    raft::host_scalar_view<IndexT> n_iter);

EXTERN_TEMPLATE_FIT(double, int)
EXTERN_TEMPLATE_FIT(double, int64_t)
EXTERN_TEMPLATE_FIT(float, int)
EXTERN_TEMPLATE_FIT(float, int64_t)

#undef EXTERN_TEMPLATE_FIT
/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.cuh>
 *   #include <cuvs/cluster/kmeans_types.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *    cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               std::nullopt,
 *               centroids.view(),
 *               raft::make_scalar_view(&inertia),
 *               raft::make_scalar_view(&n_iter));
 *   ...
 *   auto labels = raft::make_device_vector<int, int>(handle, X.extent(0));
 *
 *   kmeans::predict(handle,
 *                   params,
 *                   X,
 *                   std::nullopt,
 *                   centroids.view(),
 *                   false,
 *                   labels.view(),
 *                   raft::make_scalar_view(&ineratia));
 * @endcode
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 * @param[in]     handle           The raft handle.
 * @param[in]     params           Parameters for KMeans model.
 * @param[in]     X                New data to predict.
 *                                 [dim = n_samples x n_features]
 * @param[in]     sample_weight    Optional weights for each observation in X.
 *                                 [len = n_samples]
 * @param[in]     centroids        Cluster centroids. The data must be in
 *                                 row-major format.
 *                                 [dim = n_clusters x n_features]
 * @param[in]     normalize_weight True if the weights should be normalized
 * @param[out]    labels           Index of the cluster each sample in X
 *                                 belongs to.
 *                                 [len = n_samples]
 * @param[out]    inertia          Sum of squared distances of samples to
 *                                 their closest cluster center.
 */
template <typename DataT, typename IndexT>
void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const DataT, IndexT> X,
             std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
             raft::device_matrix_view<const DataT, IndexT> centroids,
             raft::device_vector_view<IndexT, IndexT> labels,
             bool normalize_weight,
             raft::host_scalar_view<DataT> inertia);

#define EXTERN_TEMPLATE_PREDICT(DataT, IndexT)                                  \
  extern template void predict<DataT, IndexT>(                                  \
    raft::resources const& handle,                                              \
    const kmeans::params& params,                                               \
    raft::device_matrix_view<const DataT, IndexT> X,                            \
    std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight, \
    raft::device_matrix_view<const DataT, IndexT> centroids,                    \
    raft::device_vector_view<IndexT, IndexT> labels,                            \
    bool normalize_weight,                                                      \
    raft::host_scalar_view<DataT> inertia);

EXTERN_TEMPLATE_PREDICT(double, int)
EXTERN_TEMPLATE_PREDICT(double, int64_t)
EXTERN_TEMPLATE_PREDICT(float, int)
EXTERN_TEMPLATE_PREDICT(float, int64_t)

#undef EXTERN_TEMPLATE_PREDICT

/**
 * @brief Transform X to a cluster-distance space.
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
                                              const kmeans::params& params,
                                              raft::device_matrix_view<const double, int64_t> X,
                                              std::optional<raft::device_vector_view<const double,
 int64_t>> sample_weight, raft::device_matrix_view<const double, int64_t> centroids,
                                              raft::device_vector_view<int64_t, int64_t> labels,
                                              bool normalize_weight,
                                              raft::host_scalar_view<double, int64_t> inertia);
 *                              be in row-major format
 *                              [dim = n_samples x n_features]
 * @param[in]     centroids     Cluster centroids. The data must be in row-major format.
 *                              [dim = n_clusters x n_features]
 * @param[out]    X_new         X transformed in the new space.
 *                              [dim = n_samples x n_features]
 */
template <typename DataT, typename IndexT>
void transform(raft::resources const& handle,
               const kmeans::params& params,
               raft::device_matrix_view<const DataT, IndexT> X,
               raft::device_matrix_view<const DataT, IndexT> centroids,
               raft::device_matrix_view<DataT, IndexT> X_new)
{
  cuvs::cluster::kmeans::detail::kmeans_transform<DataT, IndexT>(
    handle, params, X, centroids, X_new);
}

/**
 * @brief Select centroids according to a sampling operation
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle             The raft handle
 * @param[in]  X                  The data in row-major format
 *                                [dim = n_samples x n_features]
 * @param[in]  minClusterDistance Distance for every sample to it's nearest centroid
 *                                [dim = n_samples]
 * @param[in]  isSampleCentroid   Flag the sample chosen as initial centroid
 *                                [dim = n_samples]
 * @param[in]  select_op          The sampling operation used to select the centroids
 * @param[out] inRankCp           The sampled centroids
 *                                [dim = n_selected_centroids x n_features]
 * @param[in]  workspace          Temporary workspace buffer which can get resized
 *
 */
template <typename DataT, typename IndexT>
void sample_centroids(raft::resources const& handle,
                      raft::device_matrix_view<const DataT, IndexT> X,
                      raft::device_vector_view<DataT, IndexT> minClusterDistance,
                      raft::device_vector_view<std::uint8_t, IndexT> isSampleCentroid,
                      SamplingOp<DataT, IndexT>& select_op,
                      rmm::device_uvector<DataT>& inRankCp,
                      rmm::device_uvector<char>& workspace)
{
  cuvs::cluster::kmeans::detail::sampleCentroids<DataT, IndexT>(
    handle, X, minClusterDistance, isSampleCentroid, select_op, inRankCp, workspace);
}

/**
 * @brief Compute cluster cost
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam ReductionOpT the type of data used for the reduction operation.
 *
 * @param[in]  handle             The raft handle
 * @param[in]  minClusterDistance Distance for every sample to it's nearest centroid
 *                                [dim = n_samples]
 * @param[in]  workspace          Temporary workspace buffer which can get resized
 * @param[out] clusterCost        Resulting cluster cost
 * @param[in]  reduction_op       The reduction operation used for the cost
 *
 */
template <typename DataT, typename IndexT, typename ReductionOpT>
void cluster_cost(raft::resources const& handle,
                  raft::device_vector_view<DataT, IndexT> minClusterDistance,
                  rmm::device_uvector<char>& workspace,
                  raft::device_scalar_view<DataT> clusterCost,
                  ReductionOpT reduction_op)
{
  cuvs::cluster::kmeans::detail::computeClusterCost(
    handle, minClusterDistance, workspace, clusterCost, raft::identity_op{}, reduction_op);
}

/**
 * @brief Compute distance for every sample to it's nearest centroid
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle               The raft handle
 * @param[in]  X                    The data in row-major format
 *                                  [dim = n_samples x n_features]
 * @param[in]  centroids            Centroids data
 *                                  [dim = n_cluster x n_features]
 * @param[out] minClusterDistance   Distance for every sample to it's nearest centroid
 *                                  [dim = n_samples]
 * @param[in]  L2NormX              L2 norm of X : ||x||^2
 *                                  [dim = n_samples]
 * @param[out] L2NormBuf_OR_DistBuf Resizable buffer to store L2 norm of centroids or distance
 *                                  matrix
 * @param[in]  metric               Distance metric to use
 * @param[in]  batch_samples        batch size for input data samples
 * @param[in]  batch_centroids      batch size for input centroids
 * @param[in]  workspace            Temporary workspace buffer which can get resized
 *
 */
template <typename DataT, typename IndexT>
void min_cluster_distance(raft::resources const& handle,
                          raft::device_matrix_view<const DataT, IndexT> X,
                          raft::device_matrix_view<DataT, IndexT> centroids,
                          raft::device_vector_view<DataT, IndexT> minClusterDistance,
                          raft::device_vector_view<DataT, IndexT> L2NormX,
                          rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
                          cuvs::distance::DistanceType metric,
                          int batch_samples,
                          int batch_centroids,
                          rmm::device_uvector<char>& workspace)
{
  cuvs::cluster::kmeans::detail::minClusterDistanceCompute<DataT, IndexT>(handle,
                                                                          X,
                                                                          centroids,
                                                                          minClusterDistance,
                                                                          L2NormX,
                                                                          L2NormBuf_OR_DistBuf,
                                                                          metric,
                                                                          batch_samples,
                                                                          batch_centroids,
                                                                          workspace);
}

/**
 * @brief Compute (optionally weighted) cluster cost (inertia)
 *
 * @tparam DataT  float or double
 * @tparam IndexT Index type
 *
 * @param[in]  handle         The raft handle
 * @param[in]  X              Input data [n_samples x n_features]
 * @param[in]  centroids      Cluster centroids [n_clusters x n_features]
 * @param[out] cost           Sum of squared distances to nearest centroid (device)
 * @param[in]  sample_weight  Optional per-sample weights [n_samples]
 */
template <typename DataT, typename IndexT>
void cluster_cost(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<const DataT, IndexT> centroids,
  raft::device_scalar_view<DataT> cost,
  std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight = std::nullopt)
{
  auto stream     = raft::resource::get_cuda_stream(handle);
  auto n_clusters = centroids.extent(0);
  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);

  rmm::device_uvector<char> workspace(n_samples * sizeof(IndexT), stream);

  auto x_norms = raft::make_device_vector<DataT>(handle, n_samples);
  raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(handle, X, x_norms.view());

  auto min_cluster_distance = raft::make_device_vector<DataT>(handle, n_samples);
  rmm::device_uvector<DataT> l2_norm_or_distance_buffer(0, stream);

  auto metric = cuvs::distance::DistanceType::L2Expanded;

  cuvs::cluster::kmeans::min_cluster_distance<DataT, IndexT>(
    handle,
    X,
    raft::make_device_matrix_view<DataT, IndexT>(
      const_cast<DataT*>(centroids.data_handle()), n_clusters, n_features),
    min_cluster_distance.view(),
    x_norms.view(),
    l2_norm_or_distance_buffer,
    metric,
    n_samples,
    n_clusters,
    workspace);

  if (sample_weight.has_value()) {
    raft::linalg::map(handle,
                      min_cluster_distance.view(),
                      raft::mul_op{},
                      raft::make_const_mdspan(min_cluster_distance.view()),
                      sample_weight.value());
  }

  cuvs::cluster::kmeans::cluster_cost(
    handle, min_cluster_distance.view(), workspace, cost, raft::add_op{});
}

/**
 * @brief Compute (optionally weighted) cluster cost (inertia) — host-scalar output.
 *
 * Convenience wrapper that copies the result to host and synchronizes.
 *
 * @tparam DataT  float or double
 * @tparam IndexT Index type
 *
 * @param[in]  handle         The raft handle
 * @param[in]  X              Input data [n_samples x n_features]
 * @param[in]  centroids      Cluster centroids [n_clusters x n_features]
 * @param[out] cost           Sum of squared distances to nearest centroid (host)
 * @param[in]  sample_weight  Optional per-sample weights [n_samples]
 */
template <typename DataT, typename IndexT>
void cluster_cost(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<const DataT, IndexT> centroids,
  raft::host_scalar_view<DataT> cost,
  std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight = std::nullopt)
{
  auto device_cost = raft::make_device_scalar<DataT>(handle, DataT(0));
  cuvs::cluster::kmeans::cluster_cost(handle, X, centroids, device_cost.view(), sample_weight);
  raft::copy(handle, cost, raft::make_const_mdspan(device_cost.view()));
  raft::resource::sync_stream(handle);
}

/**
 * @brief Calculates a <key, value> pair for every sample in input 'X' where key is an
 * index of one of the 'centroids' (index of the nearest centroid) and 'value'
 * is the distance between the sample and the 'centroid[key]'
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle                The raft handle
 * @param[in]  X                     The data in row-major format
 *                                   [dim = n_samples x n_features]
 * @param[in]  centroids             Centroids data
 *                                   [dim = n_cluster x n_features]
 * @param[out] minClusterAndDistance Distance vector that contains for every sample, the nearest
 *                                   centroid and it's distance
 *                                   [dim = n_samples]
 * @param[in]  L2NormX               L2 norm of X : ||x||^2
 *                                   [dim = n_samples]
 * @param[out] L2NormBuf_OR_DistBuf  Resizable buffer to store L2 norm of centroids or distance
 *                                   matrix
 * @param[in] metric                 distance metric
 * @param[in] batch_samples          batch size of data samples
 * @param[in] batch_centroids        batch size of centroids
 * @param[in] workspace              Temporary workspace buffer which can get resized
 *
 */
template <typename DataT, typename IndexT>
void min_cluster_and_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<const DataT, IndexT> centroids,
  raft::device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT> minClusterAndDistance,
  raft::device_vector_view<DataT, IndexT> L2NormX,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  cuvs::distance::DistanceType metric,
  int batch_samples,
  int batch_centroids,
  rmm::device_uvector<char>& workspace)
{
  cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<DataT, IndexT>(handle,
                                                                             X,
                                                                             centroids,
                                                                             minClusterAndDistance,
                                                                             L2NormX,
                                                                             L2NormBuf_OR_DistBuf,
                                                                             metric,
                                                                             batch_samples,
                                                                             batch_centroids,
                                                                             workspace);
}

/**
 * @brief Shuffle and randomly select 'n_samples_to_gather' from input 'in' and stores
 * in 'out' does not modify the input
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle              The raft handle
 * @param[in]  in                  The data to shuffle and gather
 *                                 [dim = n_samples x n_features]
 * @param[out] out                 The sampled data
 *                                 [dim = n_samples_to_gather x n_features]
 * @param[in]  n_samples_to_gather Number of sample to gather
 * @param[in]  seed                Seed for the shuffle
 *
 */
template <typename DataT, typename IndexT>
void shuffle_and_gather(raft::resources const& handle,
                        raft::device_matrix_view<const DataT, IndexT> in,
                        raft::device_matrix_view<DataT, IndexT> out,
                        uint32_t n_samples_to_gather,
                        uint64_t seed)
{
  cuvs::cluster::kmeans::detail::shuffleAndGather<DataT, IndexT>(
    handle, in, out, n_samples_to_gather, seed);
}

/**
 * @brief Count the number of samples in each cluster
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle               The raft handle
 * @param[in]  params               The parameters for KMeans
 * @param[in]  X                    The data in row-major format
 *                                  [dim = n_samples x n_features]
 * @param[in]  L2NormX              L2 norm of X : ||x||^2
 *                                  [dim = n_samples]
 * @param[in]  centroids            Centroids data
 *                                  [dim = n_cluster x n_features]
 * @param[in]  workspace            Temporary workspace buffer which can get resized
 * @param[out] sampleCountInCluster The count for each centroid
 *                                  [dim = n_cluster]
 *
 */
template <typename DataT, typename IndexT>
void count_samples_in_cluster(raft::resources const& handle,
                              const kmeans::params& params,
                              raft::device_matrix_view<const DataT, IndexT> X,
                              raft::device_vector_view<DataT, IndexT> L2NormX,
                              raft::device_matrix_view<DataT, IndexT> centroids,
                              rmm::device_uvector<char>& workspace,
                              raft::device_vector_view<DataT, IndexT> sampleCountInCluster)
{
  cuvs::cluster::kmeans::detail::countSamplesInCluster<DataT, IndexT>(
    handle, params, X, L2NormX, centroids, workspace, sampleCountInCluster);
}

/**
 * @brief Selects 'n_clusters' samples from the input X using kmeans++ algorithm.
 *
 * @see "k-means++: the advantages of careful seeding". 2007, Arthur, D. and Vassilvitskii, S.
 *        ACM-SIAM symposium on Discrete algorithms.
 *
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 *
 * @param[in]  handle                The raft handle
 * @param[in]  params                The parameters for KMeans
 * @param[in]  X                     The data in row-major format
 *                                   [dim = n_samples x n_features]
 * @param[out] centroids             Centroids data
 *                                   [dim = n_cluster x n_features]
 * @param[in]  workspace             Temporary workspace buffer which can get resized
 */
template <typename DataT, typename IndexT>
void init_plus_plus(raft::resources const& handle,
                    const kmeans::params& params,
                    raft::device_matrix_view<const DataT, IndexT> X,
                    raft::device_matrix_view<DataT, IndexT> centroids,
                    rmm::device_uvector<char>& workspace)
{
  cuvs::cluster::kmeans::detail::kmeansPlusPlus<DataT, IndexT>(
    handle, params, X, centroids, workspace);
}

};  // end namespace  cuvs::cluster::kmeans
