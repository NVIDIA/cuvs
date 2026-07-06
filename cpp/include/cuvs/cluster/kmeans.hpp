/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng_state.hpp>
#include <rapids_logger/logger.hpp>

#include <cuvs/core/export.hpp>
#include <optional>
#include <vector>

namespace CUVS_EXPORT cuvs {
namespace cluster {
namespace kmeans {

/** Base structure for parameters that are common to all k-means algorithms */
struct base_params {
  /**
   * Metric to use for distance computation. The supported metrics can vary per algorithm.
   */
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
};

/**
 * @defgroup kmeans_params k-means hyperparameters
 * @{
 */

/**
 * Simple object to specify hyper-parameters to the kmeans algorithm.
 */
struct params : base_params {
  enum InitMethod {

    /**
     * Sample the centroids using the kmeans++ strategy
     */
    KMeansPlusPlus,

    /**
     * Sample the centroids uniformly at random
     */
    Random,

    /**
     * User provides the array of initial centroids
     */
    Array
  };

  /**
   * The number of clusters to form as well as the number of centroids to generate (default:8).
   */
  int n_clusters = 8;

  /**
   * Method for initialization, defaults to k-means++:
   *  - InitMethod::KMeansPlusPlus (k-means++): Use scalable k-means++ algorithm
   * to select the initial cluster centers.
   *  - InitMethod::Random (random): Choose 'n_clusters' observations (rows) at
   * random from the input data for the initial centroids.
   *  - InitMethod::Array (ndarray): Use 'centroids' as initial cluster centers.
   */
  InitMethod init = KMeansPlusPlus;

  /**
   * Maximum number of iterations of the k-means algorithm for a single run.
   */
  int max_iter = 300;

  /**
   * Relative tolerance with regards to inertia to declare convergence.
   */
  double tol = 1e-4;

  /**
   * verbosity level.
   */
  rapids_logger::level_enum verbosity = rapids_logger::level_enum::info;

  /**
   * Seed to the random number generator.
   */
  raft::random::RngState rng_state{0};

  /**
   * Number of instance k-means algorithm will be run with different seeds.
   */
  int n_init = 1;

  /**
   * Oversampling factor for use in the k-means|| algorithm.
   *
   * In the single-GPU path the value `0` is overloaded as an algorithm switch
   * that selects the classic sequential k-means++ instead of the scalable
   * variant. Any value `> 0` is used as-is.
   *
   * In the multi-GPU path (@ref cuvs::cluster::kmeans::mg::fit with device-
   * resident inputs) any value `< 1.0` (including `0`) is internally clamped to `1.0`.
   * Values `>= 1.0` are passed through unchanged.
   */
  double oversampling_factor = 2.0;

  /**
   * batch_samples and batch_centroids are used to tile 1NN computation which is
   * useful to optimize/control the memory footprint
   * Default tile is [batch_samples x n_clusters] i.e. when batch_centroids is 0
   * then don't tile the centroids
   *
   * NB: These parameters are unrelated to streaming_batch_size, which controls how many
   * samples to transfer from host to device per batch when processing out-of-core
   * data.
   */
  int batch_samples = 1 << 15;

  /**
   * if 0 then batch_centroids = n_clusters
   */
  int batch_centroids = 0;

  /**
   * Number of samples to randomly draw for the KMeansPlusPlus initialization
   * step. A random subset of this size is used for centroid seeding.
   *
   * Only applies when dataset is on host; for device data the full dataset
   * is always used for seeding and this parameter is ignored.
   *
   * When set to 0 (default) with host data uses `min(3 * n_clusters, n_samples)`
   * as a default.
   *
   * In Batched multi-GPU host-data fits, the effective KMeansPlusPlus initialization
   * sample is materialized on device on every rank. Every rank must have enough
   * GPU memory for this sample, and rank 0 must also have enough GPU memory for
   * the seeding workspace.
   *
   * Default: 0.
   */
  int64_t init_size = 0;

  /**
   * Number of samples to process per GPU batch when fitting with host data.
   * When set to 0, defaults to n_samples (process all at once).
   * Only used by the batched (host-data) code path and ignored by
   * device-data overloads.
   *
   * In multi-GPU mode this is a per-rank batch size: each rank processes up
   * to this many local samples per batch, clamped to that rank's local sample
   * count. When MG is invoked with device partitions, the runtime ignores
   * `streaming_batch_size` and processes each partition in full.
   * Default: 0 (process all data at once).
   */
  int64_t streaming_batch_size = 0;
};

/**
 * Simple object to specify hyper-parameters to the balanced k-means algorithm.
 *
 * The following metrics are currently supported in k-means balanced:
 *  - CosineExpanded
 *  - InnerProduct
 *  - L2Expanded
 *  - L2SqrtExpanded
 */
struct balanced_params : base_params {
  /**
   * Number of training iterations
   */
  uint32_t n_iters = 20;
};

/**
 * @brief Type of k-means algorithm.
 */
enum class kmeans_type { KMeans = 0, KMeansBalanced = 1 };

/**
 * @}
 */

/**
 * @defgroup kmeans k-means clustering APIs
 * @{
 */

/**
 * @brief Find clusters with k-means algorithm using batched processing of host data.
 *        Single-GPU only.
 *
 * Multi-GPU migration (breaking change in cuVS 26.08): earlier releases
 * silently dispatched this single-GPU overload to a multi-GPU implementation
 * when the supplied RAFT handle had RAFT comms or an SNMG clique attached. That
 * implicit dispatch has been removed: this overload is now strictly
 * single-GPU. If `handle` carries communications/clique state it is ignored and the call falls back
 * to the single-GPU path. To run on multiple GPUs, call `cuvs::cluster::kmeans::mg::fit`
 * explicitly.
 *
 * TODO: Evaluate replacing the extent type with int64_t. Reference issue:
 * https://github.com/rapidsai/cuvs/issues/1961
 *
 * This overload supports out-of-core computation where the dataset resides
 * on the host. Data is processed in batches, streaming from host to
 * device. The batch size is controlled by `params.streaming_batch_size`.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   params.n_clusters = 100;
 *   params.streaming_batch_size = 100000;
 *   float inertia;
 *   int64_t n_iter;
 *
 *   // Data on host
 *   std::vector<float> h_X(n_samples * n_features);
 *   auto X = raft::make_host_matrix_view<const float, int64_t>(h_X.data(), n_samples, n_features);
 *
 *   // Centroids on device
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, params.n_clusters,
 * n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               std::nullopt,
 *               centroids.view(),
 *               raft::make_host_scalar_view(&inertia),
 *               raft::make_host_scalar_view(&n_iter));
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model. Batch size is read from
 *                              params.streaming_batch_size.
 * @param[in]     X             Training instances on HOST memory. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X (on host).
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
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const float, int64_t> X,
         std::optional<raft::host_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Find clusters with k-means algorithm using batched processing of host data.
 */
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const double, int64_t> X,
         std::optional<raft::host_vector_view<const double, int64_t>> sample_weight,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
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
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int> X,
         std::optional<raft::device_vector_view<const float, int>> sample_weight,
         raft::device_matrix_view<float, int> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int> n_iter);

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int64_t n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, params.n_clusters,
 * n_features);
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
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int64_t> X,
         std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<double, int>(handle, params.n_clusters, n_features);
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
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const double, int> X,
         std::optional<raft::device_vector_view<const double, int>> sample_weight,
         raft::device_matrix_view<double, int> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int> n_iter);

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int64_t n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<double, int64_t>(handle, params.n_clusters,
 * n_features);
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
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const double, int64_t> X,
         std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
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
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const int8_t, int> X,
         std::optional<raft::device_vector_view<const int8_t, int>> sample_weight,
         raft::device_matrix_view<int8_t, int> centroids,
         raft::host_scalar_view<int8_t> inertia,
         raft::host_scalar_view<int> n_iter);

/**
 * @brief Find balanced clusters with k-means algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int64_t n_features = 15;
 *   int64_t n_clusters = 8;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               centroids);
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[out]  centroids       [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 */
void fit(const raft::resources& handle,
         cuvs::cluster::kmeans::balanced_params const& params,
         raft::device_matrix_view<const float, int64_t> X,
         raft::device_matrix_view<float, int64_t> centroids,
         std::optional<raft::host_scalar_view<float>> inertia = std::nullopt);

/**
 * @brief Find balanced clusters with k-means algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int64_t n_features = 15, n_clusters = 8;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               centroids);
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[inout]  centroids     [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 */
void fit(const raft::resources& handle,
         cuvs::cluster::kmeans::balanced_params const& params,
         raft::device_matrix_view<const int8_t, int64_t> X,
         raft::device_matrix_view<float, int64_t> centroids,
         std::optional<raft::host_scalar_view<float>> inertia = std::nullopt);

/**
 * @brief Find balanced clusters with k-means algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int64_t n_features = 15, n_clusters = 8;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               centroids);
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[inout]  centroids     [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 */
void fit(const raft::resources& handle,
         cuvs::cluster::kmeans::balanced_params const& params,
         raft::device_matrix_view<const half, int64_t> X,
         raft::device_matrix_view<float, int64_t> centroids,
         std::optional<raft::host_scalar_view<float>> inertia = std::nullopt);

/**
 * @brief Find balanced clusters with k-means algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int64_t n_features = 15, n_clusters = 8;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               centroids);
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[inout]  centroids     [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 */
void fit(const raft::resources& handle,
         cuvs::cluster::kmeans::balanced_params const& params,
         raft::device_matrix_view<const uint8_t, int64_t> X,
         raft::device_matrix_view<float, int64_t> centroids,
         std::optional<raft::host_scalar_view<float>> inertia = std::nullopt);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
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
 *                   raft::make_scalar_view(&inertia));
 * @endcode
 *
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
void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const float, int> X,
             std::optional<raft::device_vector_view<const float, int>> sample_weight,
             raft::device_matrix_view<const float, int> centroids,
             raft::device_vector_view<int, int> labels,
             bool normalize_weight,
             raft::host_scalar_view<float> inertia);

void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const float, int64_t> X,
             std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<int64_t, int64_t> labels,
             bool normalize_weight,
             raft::host_scalar_view<float> inertia);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
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
 *   auto labels = raft::make_device_vector<int64_t, int>(handle, X.extent(0));
 *
 *   kmeans::predict(handle,
 *                   params,
 *                   X,
 *                   std::nullopt,
 *                   centroids.view(),
 *                   false,
 *                   labels.view(),
 *                   raft::make_scalar_view(&inertia));
 * @endcode
 *
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
void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const float, int64_t> X,
             std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<int64_t, int64_t> labels,
             bool normalize_weight,
             raft::host_scalar_view<float> inertia);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<double, int>(handle, params.n_clusters, n_features);
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
 *                   raft::make_scalar_view(&inertia));
 * @endcode
 *
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
void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const double, int> X,
             std::optional<raft::device_vector_view<const double, int>> sample_weight,
             raft::device_matrix_view<const double, int> centroids,
             raft::device_vector_view<int, int> labels,
             bool normalize_weight,
             raft::host_scalar_view<double> inertia);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<double, int>(handle, params.n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               std::nullopt,
 *               centroids.view(),
 *               raft::make_scalar_view(&inertia),
 *               raft::make_scalar_view(&n_iter));
 *   ...
 *   auto labels = raft::make_device_vector<int64_t, int>(handle, X.extent(0));
 *
 *   kmeans::predict(handle,
 *                   params,
 *                   X,
 *                   std::nullopt,
 *                   centroids.view(),
 *                   false,
 *                   labels.view(),
 *                   raft::make_scalar_view(&inertia));
 * @endcode
 *
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
void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const double, int64_t> X,
             std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
             raft::device_matrix_view<const double, int64_t> centroids,
             raft::device_vector_view<int64_t, int64_t> labels,
             bool normalize_weight,
             raft::host_scalar_view<double> inertia);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int64_t n_features = 15, n_clusters = 8;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               centroids.view());
 *   ...
 *   auto labels = raft::make_device_vector<uint32_t, int64_t>(handle, X.extent(0));
 *
 *   kmeans::predict(handle,
 *                   params,
 *                   X,
 *                   centroids.view(),
 *                   labels.view());
 * @endcode
 *
 * @param[in]     handle           The raft handle.
 * @param[in]     params           Parameters for KMeans model.
 * @param[in]     X                New data to predict.
 *                                 [dim = n_samples x n_features]
 * @param[in]     centroids        Cluster centroids. The data must be in
 *                                 row-major format.
 *                                 [dim = n_clusters x n_features]
 * @param[out]    labels           Index of the cluster each sample in X
 *                                 belongs to.
 *                                 [len = n_samples]
 */
void predict(const raft::resources& handle,
             cuvs::cluster::kmeans::balanced_params const& params,
             raft::device_matrix_view<const int8_t, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<uint32_t, int64_t> labels);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int64_t n_features = 15, n_clusters = 8;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               centroids.view());
 *   ...
 *   auto labels = raft::make_device_vector<int, int64_t>(handle, X.extent(0));
 *
 *   kmeans::predict(handle,
 *                   params,
 *                   X,
 *                   centroids.view(),
 *                   labels.view());
 * @endcode
 *
 * @param[in]     handle           The raft handle.
 * @param[in]     params           Parameters for KMeans model.
 * @param[in]     X                New data to predict.
 *                                 [dim = n_samples x n_features]
 * @param[in]     centroids        Cluster centroids. The data must be in
 *                                 row-major format.
 *                                 [dim = n_clusters x n_features]
 * @param[out]    labels           Index of the cluster each sample in X
 *                                 belongs to.
 *                                 [len = n_samples]
 */
void predict(const raft::resources& handle,
             cuvs::cluster::kmeans::balanced_params const& params,
             raft::device_matrix_view<const int8_t, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<int, int64_t> labels);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int64_t n_features = 15, n_clusters = 8;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               centroids.view());
 *   ...
 *   auto labels = raft::make_device_vector<int, int64_t>(handle, X.extent(0));
 *
 *   kmeans::predict(handle,
 *                   params,
 *                   X,
 *                   centroids.view(),
 *                   labels.view());
 * @endcode
 *
 * @param[in]     handle           The raft handle.
 * @param[in]     params           Parameters for KMeans model.
 * @param[in]     X                New data to predict.
 *                                 [dim = n_samples x n_features]
 * @param[in]     centroids        Cluster centroids. The data must be in
 *                                 row-major format.
 *                                 [dim = n_clusters x n_features]
 * @param[out]    labels           Index of the cluster each sample in X
 *                                 belongs to.
 *                                 [len = n_samples]
 */
void predict(const raft::resources& handle,
             cuvs::cluster::kmeans::balanced_params const& params,
             raft::device_matrix_view<const float, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<int, int64_t> labels);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int64_t n_features = 15, n_clusters = 8;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               centroids.view());
 *   ...
 *   auto labels = raft::make_device_vector<uint32_t, int64_t>(handle, X.extent(0));
 *
 *   kmeans::predict(handle,
 *                   params,
 *                   X,
 *                   centroids.view(),
 *                   labels.view());
 * @endcode
 *
 * @param[in]     handle           The raft handle.
 * @param[in]     params           Parameters for KMeans model.
 * @param[in]     X                New data to predict.
 *                                 [dim = n_samples x n_features]
 * @param[in]     centroids        Cluster centroids. The data must be in
 *                                 row-major format.
 *                                 [dim = n_clusters x n_features]
 * @param[out]    labels           Index of the cluster each sample in X
 *                                 belongs to.
 *                                 [len = n_samples]
 */
void predict(const raft::resources& handle,
             cuvs::cluster::kmeans::balanced_params const& params,
             raft::device_matrix_view<const float, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<uint32_t, int64_t> labels);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int64_t n_features = 15, n_clusters = 8;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               centroids.view());
 *   ...
 *   auto labels = raft::make_device_vector<uint32_t, int64_t>(handle, X.extent(0));
 *
 *   kmeans::predict(handle,
 *                   params,
 *                   X,
 *                   centroids.view(),
 *                   labels.view());
 * @endcode
 *
 * @param[in]     handle           The raft handle.
 * @param[in]     params           Parameters for KMeans model.
 * @param[in]     X                New data to predict.
 *                                 [dim = n_samples x n_features]
 * @param[in]     centroids        Cluster centroids. The data must be in
 *                                 row-major format.
 *                                 [dim = n_clusters x n_features]
 * @param[out]    labels           Index of the cluster each sample in X
 *                                 belongs to.
 *                                 [len = n_samples]
 */
void predict(const raft::resources& handle,
             cuvs::cluster::kmeans::balanced_params const& params,
             raft::device_matrix_view<const half, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<uint32_t, int64_t> labels);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int64_t n_features = 15, n_clusters = 8;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               centroids.view());
 *   ...
 *   auto labels = raft::make_device_vector<uint32_t, int64_t>(handle, X.extent(0));
 *
 *   kmeans::predict(handle,
 *                   params,
 *                   X,
 *                   centroids.view(),
 *                   labels.view());
 * @endcode
 *
 * @param[in]     handle           The raft handle.
 * @param[in]     params           Parameters for KMeans model.
 * @param[in]     X                New data to predict.
 *                                 [dim = n_samples x n_features]
 * @param[in]     centroids        Cluster centroids. The data must be in
 *                                 row-major format.
 *                                 [dim = n_clusters x n_features]
 * @param[out]    labels           Index of the cluster each sample in X
 *                                 belongs to.
 *                                 [len = n_samples]
 */
void predict(const raft::resources& handle,
             cuvs::cluster::kmeans::balanced_params const& params,
             raft::device_matrix_view<const uint8_t, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<uint32_t, int64_t> labels);

/**
 * @brief Compute k-means clustering and predicts cluster index for each sample
 * in the input.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *   auto labels = raft::make_device_vector<int, int>(handle, X.extent(0));
 *
 *   kmeans::fit_predict(handle,
 *                       params,
 *                       X,
 *                       std::nullopt,
 *                       centroids.view(),
 *                       labels.view(),
 *                       raft::make_scalar_view(&inertia),
 *                       raft::make_scalar_view(&n_iter));
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must be
 *                              in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     Optional
 *                              [in] When init is InitMethod::Array, use
 *                              centroids  as the initial cluster centers
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    labels        Index of the cluster each sample in X belongs
 *                              to.
 *                              [len = n_samples]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const float, int> X,
                 std::optional<raft::device_vector_view<const float, int>> sample_weight,
                 std::optional<raft::device_matrix_view<float, int>> centroids,
                 raft::device_vector_view<int, int> labels,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int> n_iter);

/**
 * @brief Compute k-means clustering and predicts cluster index for each sample
 * in the input.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int64_t n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, params.n_clusters,
 * n_features); auto labels = raft::make_device_vector<int64_t, int64_t>(handle, X.extent(0));
 *
 *   kmeans::fit_predict(handle,
 *                       params,
 *                       X,
 *                       std::nullopt,
 *                       centroids.view(),
 *                       labels.view(),
 *                       raft::make_scalar_view(&inertia),
 *                       raft::make_scalar_view(&n_iter));
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must be
 *                              in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     Optional
 *                              [in] When init is InitMethod::Array, use
 *                              centroids  as the initial cluster centers
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    labels        Index of the cluster each sample in X belongs
 *                              to.
 *                              [len = n_samples]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const float, int64_t> X,
                 std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
                 std::optional<raft::device_matrix_view<float, int64_t>> centroids,
                 raft::device_vector_view<int64_t, int64_t> labels,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Compute k-means clustering and predicts cluster index for each sample
 * in the input.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<double, int>(handle, params.n_clusters, n_features);
 *   auto labels = raft::make_device_vector<int, int>(handle, X.extent(0));
 *
 *   kmeans::fit_predict(handle,
 *                       params,
 *                       X,
 *                       std::nullopt,
 *                       centroids.view(),
 *                       labels.view(),
 *                       raft::make_scalar_view(&inertia),
 *                       raft::make_scalar_view(&n_iter));
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must be
 *                              in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     Optional
 *                              [in] When init is InitMethod::Array, use
 *                              centroids  as the initial cluster centers
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    labels        Index of the cluster each sample in X belongs
 *                              to.
 *                              [len = n_samples]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const double, int> X,
                 std::optional<raft::device_vector_view<const double, int>> sample_weight,
                 std::optional<raft::device_matrix_view<double, int>> centroids,
                 raft::device_vector_view<int, int> labels,
                 raft::host_scalar_view<double> inertia,
                 raft::host_scalar_view<int> n_iter);

/**
 * @brief Compute k-means clustering and predicts cluster index for each sample
 * in the input.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int64_t n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<double, int64_t>(handle, params.n_clusters,
 * n_features); auto labels = raft::make_device_vector<int64_t, int64_t>(handle, X.extent(0));
 *
 *   kmeans::fit_predict(handle,
 *                       params,
 *                       X,
 *                       std::nullopt,
 *                       centroids.view(),
 *                       labels.view(),
 *                       raft::make_scalar_view(&inertia),
 *                       raft::make_scalar_view(&n_iter));
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must be
 *                              in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     Optional
 *                              [in] When init is InitMethod::Array, use
 *                              centroids  as the initial cluster centers
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    labels        Index of the cluster each sample in X belongs
 *                              to.
 *                              [len = n_samples]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const double, int64_t> X,
                 std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
                 std::optional<raft::device_matrix_view<double, int64_t>> centroids,
                 raft::device_vector_view<int64_t, int64_t> labels,
                 raft::host_scalar_view<double> inertia,
                 raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Compute balanced k-means clustering and predicts cluster index for each sample
 * in the input.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int64_t n_features = 15, n_clusters = 8;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
 *   auto labels = raft::make_device_vector<int, int64_t>(handle, X.extent(0));
 *
 *   kmeans::fit_predict(handle,
 *                       params,
 *                       X,
 *                       centroids.view(),
 *                       labels.view());
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must be
 *                              in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[inout]  centroids     Optional
 *                              [in] When init is InitMethod::Array, use
 *                              centroids  as the initial cluster centers
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    labels        Index of the cluster each sample in X belongs
 *                              to.
 *                              [len = n_samples]
 */
void fit_predict(const raft::resources& handle,
                 cuvs::cluster::kmeans::balanced_params const& params,
                 raft::device_matrix_view<const float, int64_t> X,
                 raft::device_matrix_view<float, int64_t> centroids,
                 raft::device_vector_view<uint32_t, int64_t> labels);

/**
 * @brief Compute balanced k-means clustering and predicts cluster index for each sample
 * in the input.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int64_t n_features = 15, n_clusters = 8;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
 *   auto labels = raft::make_device_vector<int, int64_t>(handle, X.extent(0));
 *
 *   kmeans::fit_predict(handle,
 *                       params,
 *                       X,
 *                       centroids.view(),
 *                       labels.view());
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must be
 *                              in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[inout]  centroids     Optional
 *                              [in] When init is InitMethod::Array, use
 *                              centroids  as the initial cluster centers
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    labels        Index of the cluster each sample in X belongs
 *                              to.
 *                              [len = n_samples]
 */
void fit_predict(const raft::resources& handle,
                 cuvs::cluster::kmeans::balanced_params const& params,
                 raft::device_matrix_view<const int8_t, int64_t> X,
                 raft::device_matrix_view<float, int64_t> centroids,
                 raft::device_vector_view<uint32_t, int64_t> labels);

/**
 * @brief Transform X to a cluster-distance space.
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format
 *                              [dim = n_samples x n_features]
 * @param[in]     centroids     Cluster centroids. The data must be in row-major format.
 *                              [dim = n_clusters x n_features]
 * @param[out]    X_new         X transformed in the new space.
 *                              [dim = n_samples x n_features]
 */
void transform(raft::resources const& handle,
               const kmeans::params& params,
               raft::device_matrix_view<const float, int> X,
               raft::device_matrix_view<const float, int> centroids,
               raft::device_matrix_view<float, int> X_new);

/**
 * @brief Transform X to a cluster-distance space.
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format
 *                              [dim = n_samples x n_features]
 * @param[in]     centroids     Cluster centroids. The data must be in row-major format.
 *                              [dim = n_clusters x n_features]
 * @param[out]    X_new         X transformed in the new space.
 *                              [dim = n_samples x n_features]
 */
void transform(raft::resources const& handle,
               const kmeans::params& params,
               raft::device_matrix_view<const double, int> X,
               raft::device_matrix_view<const double, int> centroids,
               raft::device_matrix_view<double, int> X_new);

/**
 * @brief Compute (optionally weighted) cluster cost
 *
 * @param[in]  handle         The raft handle
 * @param[in]  X              Training instances to cluster. The data must
 *                            be in row-major format.
 *                            [dim = n_samples x n_features]
 * @param[in]  centroids      Cluster centroids. The data must be in
 *                            row-major format.
 *                            [dim = n_clusters x n_features]
 * @param[out] cost           Resulting cluster cost
 * @param[in]  sample_weight  Optional per-sample weights.
 *                            [len = n_samples]
 *
 */
void cluster_cost(
  const raft::resources& handle,
  raft::device_matrix_view<const float, int> X,
  raft::device_matrix_view<const float, int> centroids,
  raft::host_scalar_view<float> cost,
  std::optional<raft::device_vector_view<const float, int>> sample_weight = std::nullopt);

/**
 * @brief Compute cluster cost
 *
 * @param[in]  handle         The raft handle
 * @param[in]  X              Training instances to cluster. The data must
 *                            be in row-major format.
 *                            [dim = n_samples x n_features]
 * @param[in]  centroids      Cluster centroids. The data must be in
 *                            row-major format.
 *                            [dim = n_clusters x n_features]
 * @param[out] cost           Resulting cluster cost
 * @param[in]  sample_weight  Optional per-sample weights.
 *                            [len = n_samples]
 */
void cluster_cost(
  const raft::resources& handle,
  raft::device_matrix_view<const double, int> X,
  raft::device_matrix_view<const double, int> centroids,
  raft::host_scalar_view<double> cost,
  std::optional<raft::device_vector_view<const double, int>> sample_weight = std::nullopt);

/**
 * @brief Compute (optionally weighted) cluster cost
 *
 * @param[in]  handle         The raft handle
 * @param[in]  X              Training instances to cluster. The data must
 *                            be in row-major format.
 *                            [dim = n_samples x n_features]
 * @param[in]  centroids      Cluster centroids. The data must be in
 *                            row-major format.
 *                            [dim = n_clusters x n_features]
 * @param[out] cost           Resulting cluster cost
 * @param[in]  sample_weight  Optional per-sample weights.
 *                            [len = n_samples]
 */
void cluster_cost(
  const raft::resources& handle,
  raft::device_matrix_view<const float, int64_t> X,
  raft::device_matrix_view<const float, int64_t> centroids,
  raft::host_scalar_view<float> cost,
  std::optional<raft::device_vector_view<const float, int64_t>> sample_weight = std::nullopt);

/**
 * @brief Compute (optionally weighted) cluster cost
 *
 * @param[in]  handle         The raft handle
 * @param[in]  X              Training instances to cluster. The data must
 *                            be in row-major format.
 *                            [dim = n_samples x n_features]
 * @param[in]  centroids      Cluster centroids. The data must be in
 *                            row-major format.
 *                            [dim = n_clusters x n_features]
 * @param[out] cost           Resulting cluster cost
 * @param[in]  sample_weight  Optional per-sample weights.
 *                            [len = n_samples]
 */
void cluster_cost(
  const raft::resources& handle,
  raft::device_matrix_view<const double, int64_t> X,
  raft::device_matrix_view<const double, int64_t> centroids,
  raft::host_scalar_view<double> cost,
  std::optional<raft::device_vector_view<const double, int64_t>> sample_weight = std::nullopt);
/**
 * @}
 */

#ifdef CUVS_BUILD_MG_ALGOS
namespace mg {
/**
 * @defgroup kmeans_mg Multi-GPU / out-of-core k-means fit
 * @{
 *
 * @brief Explicit multi-GPU k-means entry points.
 *
 * All multi-GPU k-means APIs live in this namespace
 * (`cuvs::cluster::kmeans::mg`). To run k-means on multiple GPUs a `handle` that
 * carries either an SNMG clique (`raft::resource::is_multi_gpu(handle)`) or
 * initialized RAFT comms (`raft::resource::comms_initialized(handle)`) must be used.
 *
 * Migration from earlier releases (breaking change in cuVS 26.08): before
 * this release, the single-GPU `cuvs::cluster::kmeans::fit` overloads would
 * silently dispatch to the multi-GPU backend when the supplied `handle`
 * carried RAFT comms or an SNMG clique. That implicit dispatch has been
 * removed: the single-GPU `fit` is now strictly single-GPU. Existing
 * multi-GPU call sites must be updated to invoke
 * `cuvs::cluster::kmeans::mg::fit` directly. Two flavors of the multi-GPU
 * API are provided here:
 *   - A single mdspan per rank (drop-in replacement for the old single-GPU
 *     signature; cuVS wraps it into a one-element vector internally).
 *   - A `std::vector` of mdspan partitions per rank (multiple partitions per
 *     rank, typical for Dask/Ray and out-of-core host-data flows).
 *
 * @code{.cpp}
 *   // Before (cuVS <= 26.06): implicit multi-GPU dispatch via single-GPU API.
 *   //   raft::resources handle; // attached to NCCL comms or SNMG clique
 *   //   cuvs::cluster::kmeans::fit(handle, params, local_X, std::nullopt,
 *   //                              centroids, inertia, n_iter);
 *
 *   // After (cuVS >= 26.08): explicit multi-GPU API.
 *   raft::resources handle; // NCCL comms or SNMG clique attached
 *   cuvs::cluster::kmeans::mg::fit(handle, params, local_X, std::nullopt,
 *                                  centroids, inertia, n_iter);
 * @endcode
 */

/**
 * @brief Multi-GPU k-means fit with one or more local data
 *        partitions per rank.
 *
 * Each rank supplies its local training data as a vector of partitions. For
 * host-resident partitions the implementation streams each partition through
 * Lloyd iterations using `params.streaming_batch_size` (per rank). For
 * device-resident partitions `streaming_batch_size` is ignored and each local
 * partition is processed in full.
 *
 * The active backend is selected by the resources attached to
 * `handle`:
 *   - When `raft::resource::is_multi_gpu(handle)` is true (SNMG clique), the
 *     call must be issued from inside an OpenMP region with one thread per
 *     rank in the clique.
 *   - Otherwise, multi-process NCCL comms must be initialized on the handle
 *     (`raft::resource::comms_initialized(handle)`); each process supplies its
 *     own local partitions.
 *
 * @param[in]     handle              The raft handle. Must have NCCL comms or
 *                                    a SNMG clique initialized.
 * @param[in]     params              K-means parameters. For host-resident
 *                                    partitions the per-rank streaming batch
 *                                    size is read from
 *                                    `params.streaming_batch_size`; it is
 *                                    ignored for device-resident partitions.
 * @param[in]     X_parts             Per-partition local data on this rank.
 *                                    Each entry is [n_rows_i x n_features].
 * @param[in]     sample_weight_parts Optional per-partition row weights with
 *                                    one vector per data partition.
 * @param[inout]  centroids           Device matrix [n_clusters x n_features].
 *                                    On entry, used as the initial centers
 *                                    when `params.init == InitMethod::Array`.
 *                                    On return, holds the converged
 *                                    centroids.
 * @param[out]    inertia             Host scalar receiving the final
 *                                    clustering cost.
 * @param[out]    n_iter              Host scalar receiving the iteration
 *                                    count at which the run terminated.
 */
void fit(
  raft::resources const& handle,
  const cuvs::cluster::kmeans::params& params,
  const std::vector<raft::device_matrix_view<const float, int>>& X_parts,
  const std::optional<std::vector<raft::device_vector_view<const float, int>>>& sample_weight_parts,
  raft::device_matrix_view<float, int> centroids,
  raft::host_scalar_view<float> inertia,
  raft::host_scalar_view<int> n_iter);

/**
 * @brief Multi-GPU k-means fit.
 */
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::device_matrix_view<const float, int64_t>>& X_parts,
         const std::optional<std::vector<raft::device_vector_view<const float, int64_t>>>&
           sample_weight_parts,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Multi-GPU k-means fit.
 */
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::device_matrix_view<const double, int>>& X_parts,
         const std::optional<std::vector<raft::device_vector_view<const double, int>>>&
           sample_weight_parts,
         raft::device_matrix_view<double, int> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int> n_iter);

/**
 * @brief Multi-GPU k-means fit.
 */
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::device_matrix_view<const double, int64_t>>& X_parts,
         const std::optional<std::vector<raft::device_vector_view<const double, int64_t>>>&
           sample_weight_parts,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Multi-GPU / out-of-core k-means fit.
 */
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::host_matrix_view<const float, int64_t>>& X_parts,
         const std::optional<std::vector<raft::host_vector_view<const float, int64_t>>>&
           sample_weight_parts,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Multi-GPU / out-of-core k-means fit.
 */
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         const std::vector<raft::host_matrix_view<const double, int64_t>>& X_parts,
         const std::optional<std::vector<raft::host_vector_view<const double, int64_t>>>&
           sample_weight_parts,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Multi-GPU k-means fit, single mdspan per rank.
 *
 * Convenience overload for the common case where each rank has exactly one
 * local partition. The mdspan is wrapped in a one-element vector and routed
 * through the vector-of-partitions overload above. See that overload's
 * documentation for backend selection and handle requirements.
 */
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int> X,
         std::optional<raft::device_vector_view<const float, int>> sample_weight,
         raft::device_matrix_view<float, int> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int> n_iter);

/**
 * @brief Multi-GPU k-means fit, single mdspan per rank.
 */
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int64_t> X,
         std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Multi-GPU k-means fit, single mdspan per rank.
 */
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const double, int> X,
         std::optional<raft::device_vector_view<const double, int>> sample_weight,
         raft::device_matrix_view<double, int> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int> n_iter);

/**
 * @brief Multi-GPU k-means fit, single mdspan per rank.
 */
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const double, int64_t> X,
         std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Multi-GPU / out-of-core k-means fit, single mdspan per rank.
 *
 * Dispatches to the SNMG-clique (batched per-rank) backend when the handle
 * carries an SNMG clique, and to the NCCL multi-process backend otherwise.
 */
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const float, int64_t> X,
         std::optional<raft::host_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Multi-GPU / out-of-core k-means fit, single mdspan per rank.
 */
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const double, int64_t> X,
         std::optional<raft::host_vector_view<const double, int64_t>> sample_weight,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter);

/**
 * @}
 */
}  // namespace mg
#endif

namespace helpers {
/**
 * @defgroup kmeans_helpers k-means API helpers
 * @{
 */

/**
 * @brief Automatically find the optimal value of k using a binary search.
 * This method maximizes the Calinski-Harabasz Index while minimizing the per-cluster inertia.
 *
 *  @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *
 *   #include <raft/random/make_blobs.cuh>
 *
 *   using namespace  cuvs::cluster;
 *
 *   raft::handle_t handle;
 *   int n_samples = 100, n_features = 15, n_clusters = 10;
 *   auto X = raft::make_device_matrix<float, int>(handle, n_samples, n_features);
 *   auto labels = raft::make_device_vector<float, int>(handle, n_samples);
 *
 *   raft::random::make_blobs(handle, X, labels, n_clusters);
 *
 *   auto best_k = raft::make_host_scalar<int>(0);
 *   auto n_iter = raft::make_host_scalar<int>(0);
 *   auto inertia = raft::make_host_scalar<int>(0);
 *
 *   kmeans::find_k(handle, X, best_k.view(), inertia.view(), n_iter.view(), n_clusters+1);
 *
 * @endcode
 *
 * @param handle raft handle
 * @param X input observations (shape n_samples, n_dims)
 * @param best_k best k found from binary search
 * @param inertia inertia of best k found
 * @param n_iter number of iterations used to find best k
 * @param kmax maximum k to try in search
 * @param kmin minimum k to try in search (should be >= 1)
 * @param maxiter maximum number of iterations to run
 * @param tol tolerance for early stopping convergence
 */
void find_k(raft::resources const& handle,
            raft::device_matrix_view<const float, int> X,
            raft::host_scalar_view<int> best_k,
            raft::host_scalar_view<float> inertia,
            raft::host_scalar_view<int> n_iter,
            int kmax,
            int kmin    = 1,
            int maxiter = 100,
            float tol   = 1e-3);
/**
 * @}
 */
}  // namespace helpers

}  // namespace kmeans
}  // namespace cluster
}  // namespace CUVS_EXPORT cuvs
