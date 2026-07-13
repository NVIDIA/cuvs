/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../kmeans.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/functional>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>

namespace cuvs::cluster::kmeans::mg::detail {

#define CUVS_LOG_KMEANS(handle, fmt, ...)                    \
  do {                                                       \
    bool isRoot = true;                                      \
    if (raft::resource::comms_initialized(handle)) {         \
      const auto& comm  = raft::resource::get_comms(handle); \
      const int my_rank = comm.get_rank();                   \
      isRoot            = my_rank == 0;                      \
    }                                                        \
    if (isRoot) { RAFT_LOG_DEBUG(fmt, ##__VA_ARGS__); }      \
  } while (0)

#define KMEANS_COMM_ROOT 0

static cuvs::cluster::kmeans::params default_params;

// Selects 'n_clusters' samples randomly from X
template <typename DataT, typename IndexT>
void initRandom(const raft::resources& handle,
                const cuvs::cluster::kmeans::params& params,
                std::mt19937_64& gen_64,
                raft::device_matrix_view<const DataT, IndexT> X,
                raft::device_matrix_view<DataT, IndexT> centroids)
{
  const auto& comm     = raft::resource::get_comms(handle);
  cudaStream_t stream  = raft::resource::get_cuda_stream(handle);
  auto n_local_samples = X.extent(0);
  auto n_features      = X.extent(1);
  auto n_clusters      = params.n_clusters;

  const int my_rank = comm.get_rank();
  const int n_ranks = comm.get_size();

  std::vector<int> nCentroidsSampledByRank(n_ranks, 0);
  std::vector<size_t> nCentroidsElementsToReceiveFromRank(n_ranks, 0);

  const int nranks_reqd = std::min(n_ranks, n_clusters);
  ASSERT(KMEANS_COMM_ROOT < nranks_reqd, "KMEANS_COMM_ROOT must be in [0,  %d)\n", nranks_reqd);

  for (int rank = 0; rank < nranks_reqd; ++rank) {
    int nCentroidsSampledInRank = n_clusters / nranks_reqd;
    if (rank == KMEANS_COMM_ROOT) {
      nCentroidsSampledInRank += n_clusters - nCentroidsSampledInRank * nranks_reqd;
    }
    nCentroidsSampledByRank[rank]             = nCentroidsSampledInRank;
    nCentroidsElementsToReceiveFromRank[rank] = nCentroidsSampledInRank * n_features;
  }

  auto nCentroidsSampledInRank = nCentroidsSampledByRank[my_rank];
  ASSERT((IndexT)nCentroidsSampledInRank <= (IndexT)n_local_samples,
         "# random samples requested from rank-%d is larger than the available "
         "samples at the rank (requested is %lu, available is %lu)",
         my_rank,
         (size_t)nCentroidsSampledInRank,
         (size_t)n_local_samples);

  auto centroidsSampledInRank =
    raft::make_device_matrix<DataT, IndexT>(handle, nCentroidsSampledInRank, n_features);

  uint64_t gpu_seed = gen_64();
  cuvs::cluster::kmeans::shuffle_and_gather(
    handle, X, centroidsSampledInRank.view(), nCentroidsSampledInRank, gpu_seed);

  std::vector<size_t> displs(n_ranks);
  std::exclusive_scan(nCentroidsElementsToReceiveFromRank.begin(),
                      nCentroidsElementsToReceiveFromRank.end(),
                      displs.begin(),
                      size_t(0));

  // gather centroids from all ranks
  comm.allgatherv<DataT>(centroidsSampledInRank.data_handle(),        // sendbuff
                         centroids.data_handle(),                     // recvbuff
                         nCentroidsElementsToReceiveFromRank.data(),  // recvcount
                         displs.data(),
                         stream);
}

/*
 * @brief Selects 'n_clusters' samples from X using scalable kmeans++ algorithm
 * Scalable kmeans++ pseudocode
 * 1: C = sample a point uniformly at random from X
 * 2: psi = phi_X (C)
 * 3: for O( log(psi) ) times do
 * 4:   C' = sample each point x in X independently with probability
 *           p_x = l * ( d^2(x, C) / phi_X (C) )
 * 5:   C = C U C'
 * 6: end for
 * 7: For x in C, set w_x to be the number of points in X closer to x than any
 *    other point in C
 * 8: Recluster the weighted points in C into k clusters
 */
template <typename DataT, typename IndexT>
void initKMeansPlusPlus(const raft::resources& handle,
                        const cuvs::cluster::kmeans::params& params,
                        std::mt19937_64& gen_64,
                        raft::device_matrix_view<const DataT, IndexT> X,
                        raft::device_matrix_view<DataT, IndexT> centroidsRawData,
                        rmm::device_uvector<char>& workspace)
{
  const auto& comm    = raft::resource::get_comms(handle);
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  const int my_rank   = comm.get_rank();
  const int n_rank    = comm.get_size();

  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);
  auto n_clusters = params.n_clusters;
  auto metric     = params.metric;


  // <<<< Step-1 >>> : C <- sample a point uniformly at random from X
  //    1.1 - Select a rank r' at random from the available n_rank ranks with a
  //          probability of 1/n_rank [Note - with same seed all rank selects
  //          the same r' which avoids a call to comm]
  //    1.2 - Rank r' samples a point uniformly at random from the local dataset
  //          X which will be used as the initial centroid for kmeans++
  //    1.3 - Communicate the initial centroid chosen by rank-r' to all other
  //          ranks
  // Choose rp on rank 0 and broadcast to all ranks to guarantee agreement
  int rp = 0;
  if (my_rank == KMEANS_COMM_ROOT) {
    std::uniform_int_distribution<> dis(0, n_rank - 1);
    rp = dis(gen_64);
  }
  {
    rmm::device_scalar<int> rp_d(stream);
    raft::copy(
      handle, raft::make_device_scalar_view(rp_d.data()), raft::make_host_scalar_view(&rp));
    comm.bcast<int>(rp_d.data(), 1, /*root=*/KMEANS_COMM_ROOT, stream);
    raft::copy(
      handle, raft::make_host_scalar_view(&rp), raft::make_device_scalar_view(rp_d.data()));
    raft::resource::sync_stream(handle);
  }

  // buffer to flag the sample that is chosen as initial centroids
  std::vector<std::uint8_t> h_isSampleCentroid(n_samples);
  std::fill(h_isSampleCentroid.begin(), h_isSampleCentroid.end(), 0);

  auto initialCentroid = raft::make_device_matrix<DataT, IndexT>(handle, 1, n_features);
  CUVS_LOG_KMEANS(
    handle, "@Rank-%d : KMeans|| : initial centroid is sampled at rank-%d\n", my_rank, rp);

  //    1.2 - Rank r' samples a point uniformly at random from the local dataset
  //          X which will be used as the initial centroid for kmeans++
  if (my_rank == rp) {
    std::uniform_int_distribution<> dis(0, n_samples - 1);

    int cIdx           = dis(gen_64);
    auto centroidsView = raft::make_device_matrix_view<const DataT, IndexT>(
      X.data_handle() + cIdx * n_features, 1, n_features);

    raft::copy(handle,
               raft::make_device_vector_view(initialCentroid.data_handle(), centroidsView.size()),
               raft::make_device_vector_view(centroidsView.data_handle(), centroidsView.size()));

    h_isSampleCentroid[cIdx] = 1;
  }

  // 1.3 - Communicate the initial centroid chosen by rank-r' to all other ranks
  comm.bcast<DataT>(initialCentroid.data_handle(), initialCentroid.size(), rp, stream);

  // device buffer to flag the sample that is chosen as initial centroid
  auto isSampleCentroid = raft::make_device_vector<std::uint8_t, IndexT>(handle, n_samples);

  raft::copy(handle,
             raft::make_device_vector_view(isSampleCentroid.data_handle(), isSampleCentroid.size()),
             raft::make_host_vector_view(h_isSampleCentroid.data(), isSampleCentroid.size()));

  rmm::device_uvector<DataT> centroidsBuf(0, stream);

  // reset buffer to store the chosen centroid
  centroidsBuf.resize(initialCentroid.size(), stream);
  raft::copy(handle,
             raft::make_device_vector_view(centroidsBuf.begin(), initialCentroid.size()),
             raft::make_device_vector_view(initialCentroid.data_handle(), initialCentroid.size()));

  auto potentialCentroids = raft::make_device_matrix_view<DataT, IndexT>(
    centroidsBuf.data(), initialCentroid.extent(0), initialCentroid.extent(1));
  // <<< End of Step-1 >>>

  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // L2 norm of X: ||x||^2
  auto L2NormX = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_device_matrix_view<const DataT, IndexT, raft::row_major>(
        X.data_handle(), n_samples, n_features),
      L2NormX.view());
  }

  auto minClusterDistance = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  auto uniformRands       = raft::make_device_vector<DataT, IndexT>(handle, n_samples);

  // <<< Step-2 >>>: psi <- phi_X (C)
  auto clusterCost = raft::make_device_scalar<DataT>(handle, 0);

  cuvs::cluster::kmeans::min_cluster_distance(handle,
                                              X,
                                              potentialCentroids,
                                              minClusterDistance.view(),
                                              L2NormX.view(),
                                              L2NormBuf_OR_DistBuf,
                                              params.metric,
                                              params.batch_samples,
                                              params.batch_centroids,
                                              workspace);

  // compute partial cluster cost from the samples in rank
  cuvs::cluster::kmeans::cluster_cost(
    handle,
    minClusterDistance.view(),
    workspace,
    clusterCost.view(),
    cuda::proclaim_return_type<DataT>(
      [] __device__(const DataT& a, const DataT& b) { return a + b; }));

  // compute total cluster cost by accumulating the partial cost from all the
  // ranks
  comm.allreduce(
    clusterCost.data_handle(), clusterCost.data_handle(), 1, raft::comms::op_t::SUM, stream);

  DataT psi = 0;
  raft::copy(handle, raft::make_host_scalar_view(&psi), clusterCost.view());

  // <<< End of Step-2 >>>

  ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
         "An error occurred in the distributed operation. This can result from "
         "a failed rank");

  // Scalable kmeans++ paper claims 8 rounds is sufficient
  int niter = std::min(8, (int)ceil(log(psi)));
  CUVS_LOG_KMEANS(handle,
                  "@Rank-%d:KMeans|| :phi - %f, max # of iterations for kmeans++ loop - "
                  "%d\n",
                  my_rank,
                  psi,
                  niter);

  // <<<< Step-3 >>> : for O( log(psi) ) times do
  for (int iter = 0; iter < niter; ++iter) {
    CUVS_LOG_KMEANS(handle,
                    "@Rank-%d:KMeans|| - Iteration %d: # potential centroids sampled - "
                    "%d\n",
                    my_rank,
                    iter,
                    potentialCentroids.extent(0));

    cuvs::cluster::kmeans::min_cluster_distance(handle,
                                                X,
                                                potentialCentroids,
                                                minClusterDistance.view(),
                                                L2NormX.view(),
                                                L2NormBuf_OR_DistBuf,
                                                params.metric,
                                                params.batch_samples,
                                                params.batch_centroids,
                                                workspace);

    cuvs::cluster::kmeans::cluster_cost(
      handle,
      minClusterDistance.view(),
      workspace,
      clusterCost.view(),
      cuda::proclaim_return_type<DataT>(
        [] __device__(const DataT& a, const DataT& b) { return a + b; }));
    comm.allreduce(
      clusterCost.data_handle(), clusterCost.data_handle(), 1, raft::comms::op_t::SUM, stream);
    raft::copy(handle, raft::make_host_scalar_view(&psi), clusterCost.view());
    ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed operation. This can result "
           "from a failed rank");

    // <<<< Step-4 >>> : Sample each point x in X independently and identify new
    // potentialCentroids
    uint64_t gpu_seed;
    gpu_seed = gen_64();
    raft::random::RngState rng(gpu_seed, params.rng_state.type);
    raft::random::uniform(
      handle, rng, uniformRands.data_handle(), uniformRands.extent(0), (DataT)0, (DataT)1);
    cuvs::cluster::kmeans::SamplingOp<DataT, IndexT> select_op(psi,
                                                               params.oversampling_factor,
                                                               n_clusters,
                                                               uniformRands.data_handle(),
                                                               isSampleCentroid.data_handle());

    rmm::device_uvector<DataT> inRankCp(0, stream);
    cuvs::cluster::kmeans::sample_centroids(handle,
                                            X,
                                            minClusterDistance.view(),
                                            isSampleCentroid.view(),
                                            select_op,
                                            inRankCp,
                                            workspace);
    /// <<<< End of Step-4 >>>>

    int* nPtsSampledByRank;
    RAFT_CUDA_TRY(cudaMallocHost(&nPtsSampledByRank, n_rank * sizeof(int)));

    /// <<<< Step-5 >>> : C = C U C'
    // append the data in Cp from all ranks to the buffer holding the
    // potentialCentroids
    // RAFT_CUDA_TRY(cudaMemsetAsync(nPtsSampledByRank, 0, n_rank * sizeof(int), stream));
    std::fill(nPtsSampledByRank, nPtsSampledByRank + n_rank, 0);
    nPtsSampledByRank[my_rank] = inRankCp.size() / n_features;
    comm.allgather(&(nPtsSampledByRank[my_rank]), nPtsSampledByRank, 1, stream);
    ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed operation. This can result "
           "from a failed rank");

    auto nPtsSampled = std::reduce(nPtsSampledByRank, nPtsSampledByRank + n_rank, 0);

    // gather centroids from all ranks
    std::vector<size_t> sizes(n_rank);
    std::transform(
      nPtsSampledByRank, nPtsSampledByRank + n_rank, sizes.begin(), [n_features](int val) {
        return static_cast<size_t>(val) * n_features;
      });

    RAFT_CUDA_TRY_NO_THROW(cudaFreeHost(nPtsSampledByRank));

    std::vector<size_t> displs(n_rank);
    std::exclusive_scan(sizes.begin(), sizes.end(), displs.begin(), size_t(0));

    centroidsBuf.resize(centroidsBuf.size() + nPtsSampled * n_features, stream);
    comm.allgatherv<DataT>(inRankCp.data(),
                           centroidsBuf.end() - nPtsSampled * n_features,
                           sizes.data(),
                           displs.data(),
                           stream);

    auto tot_centroids = potentialCentroids.extent(0) + nPtsSampled;
    potentialCentroids =
      raft::make_device_matrix_view<DataT, IndexT>(centroidsBuf.data(), tot_centroids, n_features);
    /// <<<< End of Step-5 >>>
  }  /// <<<< Step-6 >>>

  CUVS_LOG_KMEANS(handle,
                  "@Rank-%d:KMeans||: # potential centroids sampled - %d\n",
                  my_rank,
                  potentialCentroids.extent(0));

  if ((IndexT)potentialCentroids.extent(0) > (IndexT)n_clusters) {
    // <<< Step-7 >>>: For x in C, set w_x to be the number of pts closest to X
    // temporary buffer to store the sample count per cluster, destructor
    // releases the resource

    auto weight = raft::make_device_vector<DataT, IndexT>(handle, potentialCentroids.extent(0));

    cuvs::cluster::kmeans::count_samples_in_cluster(
      handle, params, X, L2NormX.view(), potentialCentroids, workspace, weight.view());

    // merge the local histogram from all ranks
    comm.allreduce<DataT>(weight.data_handle(),  // sendbuff
                          weight.data_handle(),  // recvbuff
                          weight.size(),         // count
                          raft::comms::op_t::SUM,
                          stream);

    // <<< end of Step-7 >>>

    // Step-8: Recluster the weighted points in C into k clusters
    // Note - reclustering step is duplicated across all ranks and with the same
    // seed they should generate the same potentialCentroids
    auto const_centroids = raft::make_device_matrix_view<const DataT, IndexT>(
      potentialCentroids.data_handle(), potentialCentroids.extent(0), potentialCentroids.extent(1));
    auto params_copy = params;
    params_copy.rng_state.seed = gen_64();
    cuvs::cluster::kmeans::init_plus_plus(
      handle, params_copy, const_centroids, centroidsRawData, workspace);

    auto inertia = raft::make_host_scalar<DataT>(0);
    auto n_iter  = raft::make_host_scalar<IndexT>(0);
    auto weight_view =
      raft::make_device_vector_view<const DataT, IndexT>(weight.data_handle(), weight.extent(0));
    // Update the seed one more time
    params_copy.rng_state.seed                     = gen_64();
    cuvs::cluster::kmeans::fit_main<DataT, IndexT>(handle,
                                                   params_copy,
                                                   const_centroids,
                                                   weight_view,
                                                   centroidsRawData,
                                                   inertia.view(),
                                                   n_iter.view(),
                                                   workspace);

  } else if ((IndexT)potentialCentroids.extent(0) < (IndexT)n_clusters) {
    // supplement with random
    auto n_random_clusters = n_clusters - potentialCentroids.extent(0);
    CUVS_LOG_KMEANS(handle,
                    "[Warning!] KMeans||: found fewer than %d centroids during "
                    "initialization (found %d centroids, remaining %d centroids will be "
                    "chosen randomly from input samples)\n",
                    n_clusters,
                    potentialCentroids.extent(0),
                    n_random_clusters);

    // generate `n_random_clusters` centroids
    cuvs::cluster::kmeans::params rand_params = params;
    rand_params.rng_state.seed                     = gen_64();
    rand_params.init                          = cuvs::cluster::kmeans::params::InitMethod::Random;
    rand_params.n_clusters                    = n_random_clusters;
    initRandom(handle, rand_params, gen_64, X, centroidsRawData);

    // copy centroids generated during kmeans|| iteration to the buffer
    raft::copy(
      handle,
      raft::make_device_vector_view(centroidsRawData.data_handle() + n_random_clusters * n_features,
                                    potentialCentroids.size()),
      raft::make_device_vector_view(potentialCentroids.data_handle(), potentialCentroids.size()));

  } else {
    // found the required n_clusters
    raft::copy(
      handle,
      raft::make_device_vector_view(centroidsRawData.data_handle(), potentialCentroids.size()),
      raft::make_device_vector_view(potentialCentroids.data_handle(), potentialCentroids.size()));
  }
}

template <typename DataT, typename IndexT>
void checkWeights(const raft::resources& handle,
                  rmm::device_uvector<char>& workspace,
                  raft::device_vector_view<DataT, IndexT> weight)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto d_wt_sum       = raft::make_device_scalar<DataT>(handle, DataT{0});

  const auto& comm = raft::resource::get_comms(handle);

  auto n_samples = weight.extent(0);
  raft::linalg::mapThenSumReduce(
    d_wt_sum.data_handle(), n_samples, raft::identity_op{}, stream, weight.data_handle());

  comm.allreduce<DataT>(d_wt_sum.data_handle(),  // sendbuff
                        d_wt_sum.data_handle(),  // recvbuff
                        1,                       // count
                        raft::comms::op_t::SUM,
                        stream);

  // Normalize weights so they sum to n_samples (per rank). Reading the sum from
  // a device pointer avoids a host copy / stream sync. When the sum already
  // equals n_samples this is a numerical no-op (matches single-GPU behavior).
  const DataT* d_wt_sum_ptr = d_wt_sum.data_handle();
  raft::linalg::map(
    handle,
    weight,
    [n_samples, d_wt_sum_ptr] __device__(DataT w) {
      return w * static_cast<DataT>(n_samples) / *d_wt_sum_ptr;
    },
    raft::make_const_mdspan(weight));
}

template <typename DataT, typename IndexT>
void fit(const raft::resources& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const DataT, IndexT> X,
         std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
         raft::device_matrix_view<DataT, IndexT> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<IndexT> n_iter,
         rmm::device_uvector<char>& workspace)
{
  RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded ||
                 params.metric == cuvs::distance::DistanceType::L2SqrtExpanded,
               "kmeans only supports L2Expanded or L2SqrtExpanded distance metrics.");
  const auto& comm    = raft::resource::get_comms(handle);
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  const int my_rank = comm.get_rank();
  const int n_ranks = comm.get_size();

  std::mt19937_64 gen_64(params.rng_state.seed + (uint64_t(my_rank) << 32));

  auto weight = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  if (sample_weight) {
    raft::copy(handle, weight.view(), sample_weight.value());
  } else {
    raft::matrix::fill(handle, weight.view(), DataT(1));
  }

  // check if weights sum up to n_samples
  checkWeights(handle, workspace, weight.view());

  if (params.init == cuvs::cluster::kmeans::params::InitMethod::Random) {
    // initializing with random samples from input dataset
    CUVS_LOG_KMEANS(handle,
                    "KMeans.fit: initialize cluster centers by randomly choosing from the "
                    "input data.\n");
    initRandom<DataT, IndexT>(handle, params, gen_64, X, centroids);
  } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus) {
    // default method to initialize is kmeans++
    CUVS_LOG_KMEANS(handle, "KMeans.fit: initialize cluster centers using k-means++ algorithm.\n");
    initKMeansPlusPlus<DataT, IndexT>(handle, params, gen_64, X, centroids, workspace);
  } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::Array) {
    CUVS_LOG_KMEANS(handle,
                    "KMeans.fit: initialize cluster centers from the ndarray array input "
                    "passed to init argument.\n");

  } else {
    THROW("unknown initialization method to select initial centers");
  }

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);

  // temporary buffer to store L2 norm of centroids or distance matrix,
  // destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // temporary buffer to store intermediate centroids, destructor releases the
  // resource
  auto newCentroids = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);

  // temporary buffer to store the weights per cluster, destructor releases
  // the resource
  auto wtInCluster = raft::make_device_vector<DataT, IndexT>(handle, n_clusters);

  // L2 norm of X: ||x||^2
  auto L2NormX = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_device_matrix_view<const DataT, IndexT, raft::row_major>(
        X.data_handle(), n_samples, n_features),
      L2NormX.view());
  }

  DataT priorClusteringCost = 0;
  for (n_iter[0] = 1; n_iter[0] <= params.max_iter; ++n_iter[0]) {
    CUVS_LOG_KMEANS(handle,
                    "KMeans.fit: Iteration-%d: fitting the model using the initialize "
                    "cluster centers\n",
                    n_iter[0]);

    auto const_centroids = raft::make_device_matrix_view<const DataT, IndexT>(
      centroids.data_handle(), centroids.extent(0), centroids.extent(1));
    // computes minClusterAndDistance[0:n_samples) where
    // minClusterAndDistance[i] is a <key, value> pair where
    //   'key' is index to an sample in 'centroids' (index of the nearest
    //   centroid) and 'value' is the distance between the sample 'X[i]' and the
    //   'centroid[key]'
    cuvs::cluster::kmeans::min_cluster_and_distance(handle,
                                                    X,
                                                    const_centroids,
                                                    minClusterAndDistance.view(),
                                                    L2NormX.view(),
                                                    L2NormBuf_OR_DistBuf,
                                                    params.metric,
                                                    params.batch_samples,
                                                    params.batch_centroids,
                                                    workspace);

    workspace.resize(n_samples, stream);

    cuda::transform_iterator keys_itr(
      minClusterAndDistance.data_handle(),
      cuvs::cluster::kmeans::detail::KeyValueIndexOp<IndexT, DataT>{});
    raft::linalg::reduce_rows_by_key((DataT*)X.data_handle(),
                                     X.extent(1),
                                     keys_itr,
                                     weight.data_handle(),
                                     workspace.data(),
                                     X.extent(0),
                                     X.extent(1),
                                     static_cast<IndexT>(n_clusters),
                                     newCentroids.data_handle(),
                                     stream);

    // Reduce weights by key to compute weight in each cluster
    raft::linalg::reduce_cols_by_key(weight.data_handle(),
                                     keys_itr,
                                     wtInCluster.data_handle(),
                                     (IndexT)1,
                                     (IndexT)weight.extent(0),
                                     (IndexT)n_clusters,
                                     stream);

    // merge the local histogram from all ranks
    comm.allreduce<DataT>(wtInCluster.data_handle(),  // sendbuff
                          wtInCluster.data_handle(),  // recvbuff
                          wtInCluster.size(),         // count
                          raft::comms::op_t::SUM,
                          stream);

    // reduces newCentroids from all ranks
    comm.allreduce<DataT>(newCentroids.data_handle(),  // sendbuff
                          newCentroids.data_handle(),  // recvbuff
                          newCentroids.size(),         // count
                          raft::comms::op_t::SUM,
                          stream);

    // Computes newCentroids[i] = newCentroids[i]/wtInCluster[i] where
    //   newCentroids[n_clusters x n_features] - 2D array, newCentroids[i] has
    //   sum of all the samples assigned to cluster-i
    //   wtInCluster[n_clusters] - 1D array, wtInCluster[i] contains # of
    //   samples in cluster-i.
    // Note - when wtInCluster[i] is 0, newCentroid[i] is reset to 0

    raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
      handle,
      raft::make_const_mdspan(newCentroids.view()),
      raft::make_const_mdspan(wtInCluster.view()),
      newCentroids.view(),
      cuda::proclaim_return_type<DataT>([=] __device__(DataT mat, DataT vec) {
        if (vec == 0)
          return DataT(0);
        else
          return mat / vec;
      }));

    // copy the centroids[i] to newCentroids[i] when wtInCluster[i] is 0
    cub::ArgIndexInputIterator<DataT*> itr_wt(wtInCluster.data_handle());
    raft::matrix::gather_if(
      centroids.data_handle(),
      centroids.extent(1),
      centroids.extent(0),
      itr_wt,
      itr_wt,
      wtInCluster.extent(0),
      newCentroids.data_handle(),
      cuda::proclaim_return_type<bool>(
        [=] __device__(raft::KeyValuePair<ptrdiff_t, DataT> map) {  // predicate
          // copy when the # of samples in the cluster is 0
          if (map.value == 0)
            return true;
          else
            return false;
        }),
      cuda::proclaim_return_type<ptrdiff_t>(
        [=] __device__(raft::KeyValuePair<ptrdiff_t, DataT> map) {  // map
          return map.key;
        }),
      stream);

    // compute the squared norm between the newCentroids and the original
    // centroids, destructor releases the resource
    auto sqrdNorm = raft::make_device_scalar<DataT>(handle, 1);
    raft::linalg::mapThenSumReduce(
      sqrdNorm.data_handle(),
      newCentroids.size(),
      cuda::proclaim_return_type<DataT>([=] __device__(const DataT a, const DataT b) {
        DataT diff = a - b;
        return diff * diff;
      }),
      stream,
      centroids.data_handle(),
      newCentroids.data_handle());

    DataT sqrdNormError = 0;
    raft::copy(handle, raft::make_host_scalar_view(&sqrdNormError), sqrdNorm.view());

    raft::copy(handle,
               raft::make_device_vector_view(centroids.data_handle(), newCentroids.size()),
               raft::make_device_vector_view(newCentroids.data_handle(), newCentroids.size()));

    bool done = false;
    rmm::device_scalar<raft::KeyValuePair<IndexT, DataT>> clusterCostD(stream);

    // calculate cluster cost phi_x(C)
    cuvs::cluster::kmeans::cluster_cost(
      handle,
      minClusterAndDistance.view(),
      workspace,
      raft::make_device_scalar_view(clusterCostD.data()),
      cuda::proclaim_return_type<raft::KeyValuePair<IndexT, DataT>>(
        [] __device__(const raft::KeyValuePair<IndexT, DataT>& a,
                      const raft::KeyValuePair<IndexT, DataT>& b) {
          raft::KeyValuePair<IndexT, DataT> res;
          res.key   = 0;
          res.value = a.value + b.value;
          return res;
        }));

    // Cluster cost phi_x(C) from all ranks
    comm.allreduce(&(clusterCostD.data()->value),
                   &(clusterCostD.data()->value),
                   1,
                   raft::comms::op_t::SUM,
                   stream);

    DataT curClusteringCost = 0;
    raft::copy(handle,
               raft::make_host_scalar_view(&curClusteringCost),
               raft::make_device_scalar_view(&(clusterCostD.data()->value)));

    ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed operation. This can result "
           "from a failed rank");
    if (curClusteringCost == (DataT)0.0) {
      RAFT_LOG_WARN("Zero clustering cost detected: all points coincide with their centroids.");
    } else if (n_iter[0] > 1) {
      DataT delta = curClusteringCost / priorClusteringCost;
      if (delta > 1 - params.tol) done = true;
    }
    priorClusteringCost = curClusteringCost;

    raft::resource::sync_stream(handle, stream);
    if (sqrdNormError < params.tol) done = true;

    if (done) {
      CUVS_LOG_KMEANS(
        handle, "Threshold triggered after %d iterations. Terminating early.\n", n_iter[0]);
      break;
    }
  }
  n_iter[0] = std::min(n_iter[0], static_cast<IndexT>(params.max_iter));
}

};  // namespace cuvs::cluster::kmeans::mg::detail
