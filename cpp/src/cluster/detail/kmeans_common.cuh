/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../distance/distance.cuh"
#include <cstdint>
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/memory_type.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/random/permute.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cuda.h>
#include <cuda/iterator>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>

#include <raft/linalg/add.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <optional>
#include <random>

namespace cuvs::cluster::kmeans::detail {

/**
 * @brief Returns true if the fused distance NN implementation should be used.
 *
 * On Ampere (SM <= 8.x) always use fused.
 * On Hopper (SM 9.x) use fused when m or n >= 4096.
 * On Blackwell (SM >= 10.x) use fused.
 */
template <typename MathT, typename IdxT, typename LabelT>
bool use_fused(const raft::resources& handle, IdxT m, IdxT n, IdxT k)
{
  cudaDeviceProp prop;
  prop = raft::resource::get_device_properties(handle);
  if (prop.major <= 8) {
    // Use fused for Ampere or before
    return true;
  } else if (prop.major == 9 && (m >= 4096 || n >= 4096)) {
    // On Hopper if m, n are bigger than 4096, use fused
    return true;
  } else if (prop.major >= 10) {
    // On Blackwell onwards, use fused
    return true;
  }
  return false;
}

template <typename DataT, typename IndexT>
struct SamplingOp {
  DataT* rnd;
  uint8_t* flag;
  DataT cluster_cost;
  double oversampling_factor;
  IndexT n_clusters;

  CUB_RUNTIME_FUNCTION __forceinline__
  SamplingOp(DataT c, double l, IndexT k, DataT* rand, uint8_t* ptr)
    : cluster_cost(c), oversampling_factor(l), n_clusters(k), rnd(rand), flag(ptr)
  {
  }

  __host__ __device__ __forceinline__ bool operator()(
    const raft::KeyValuePair<ptrdiff_t, DataT>& a) const
  {
    DataT prob_threshold = (DataT)rnd[a.key];

    DataT prob_x = ((oversampling_factor * n_clusters * a.value) / cluster_cost);

    return !flag[a.key] && (prob_x > prob_threshold);
  }
};

template <typename IndexT, typename DataT>
struct KeyValueIndexOp {
  __host__ __device__ __forceinline__ IndexT
  operator()(const raft::KeyValuePair<IndexT, DataT>& a) const
  {
    return a.key;
  }
};

// Computes the intensity histogram from a sequence of labels
template <typename SampleIteratorT, typename CounterT, typename IndexT>
void countLabels(raft::resources const& handle,
                 SampleIteratorT labels,
                 CounterT* count,
                 IndexT n_samples,
                 IndexT n_clusters,
                 rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  // CUB::DeviceHistogram requires a signed index type
  typedef typename std::make_signed_t<IndexT> CubIndexT;

  CubIndexT num_levels  = n_clusters + 1;
  CubIndexT lower_level = 0;
  CubIndexT upper_level = n_clusters;

  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    temp_storage_bytes,
                                                    labels,
                                                    count,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    static_cast<CubIndexT>(n_samples),
                                                    stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(workspace.data(),
                                                    temp_storage_bytes,
                                                    labels,
                                                    count,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    static_cast<CubIndexT>(n_samples),
                                                    stream));
}

/**
 * @brief Compute the sum of sample weights into a device scalar.
 *
 * Device-accessible mdspans are reduced on device. Host mdspans are summed on the host.
 * When `check_positive` is true, the resulting sum is brought to host and asserted to be > 0.
 */
template <typename DataT, typename IndexT, typename Accessor>
void weightSum(
  raft::resources const& handle,
  raft::mdspan<const DataT, raft::vector_extent<IndexT>, raft::layout_right, Accessor> weight,
  raft::device_scalar_view<DataT> d_wt_sum,
  bool check_positive = true)
{
  auto n_samples = weight.extent(0);
  auto stream    = raft::resource::get_cuda_stream(handle);
  DataT wt_sum_h = DataT{0};

  if constexpr (raft::is_device_mdspan_v<decltype(weight)>) {
    raft::linalg::mapThenSumReduce(
      d_wt_sum.data_handle(), n_samples, raft::identity_op{}, stream, weight.data_handle());
    if (check_positive) {
      raft::copy(&wt_sum_h, d_wt_sum.data_handle(), 1, stream);
      raft::resource::sync_stream(handle);
    }
  } else {
    for (IndexT i = 0; i < n_samples; ++i) {
      wt_sum_h += weight(i);
    }
    raft::copy(d_wt_sum.data_handle(), &wt_sum_h, 1, stream);
  }
  if (check_positive) {
    RAFT_EXPECTS(wt_sum_h > DataT{0}, "invalid parameter (sum of sample weights must be positive)");
  }
}

template <typename IndexT>
IndexT getDataBatchSize(int batch_samples, IndexT n_samples)
{
  auto minVal = std::min(static_cast<IndexT>(batch_samples), n_samples);
  return (minVal == 0) ? n_samples : minVal;
}

template <typename IndexT>
IndexT getCentroidsBatchSize(int batch_centroids, IndexT n_local_clusters)
{
  auto minVal = std::min(static_cast<IndexT>(batch_centroids), n_local_clusters);
  return (minVal == 0) ? n_local_clusters : minVal;
}

template <typename InputT,
          typename OutputT,
          typename MainOpT,
          typename ReductionOpT,
          typename IndexT = int>
void computeClusterCost(raft::resources const& handle,
                        raft::device_vector_view<InputT, IndexT> minClusterDistance,
                        rmm::device_uvector<char>& workspace,
                        raft::device_scalar_view<OutputT> clusterCost,
                        MainOpT main_op,
                        ReductionOpT reduction_op)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  cuda::transform_iterator itr(minClusterDistance.data_handle(), main_op);

  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(nullptr,
                                          temp_storage_bytes,
                                          itr,
                                          clusterCost.data_handle(),
                                          minClusterDistance.size(),
                                          reduction_op,
                                          OutputT(),
                                          stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(workspace.data(),
                                          temp_storage_bytes,
                                          itr,
                                          clusterCost.data_handle(),
                                          minClusterDistance.size(),
                                          reduction_op,
                                          OutputT(),
                                          stream));
}

template <typename DataT, typename IndexT>
void sampleCentroids(raft::resources const& handle,
                     raft::device_matrix_view<const DataT, IndexT> X,
                     raft::device_vector_view<DataT, IndexT> minClusterDistance,
                     raft::device_vector_view<uint8_t, IndexT> isSampleCentroid,
                     SamplingOp<DataT, IndexT>& select_op,
                     rmm::device_uvector<DataT>& inRankCp,
                     rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream  = raft::resource::get_cuda_stream(handle);
  auto n_local_samples = X.extent(0);
  auto n_features      = X.extent(1);

  auto nSelected = raft::make_device_scalar<IndexT>(handle, 0);
  cub::ArgIndexInputIterator<DataT*> ip_itr(minClusterDistance.data_handle());
  auto sampledMinClusterDistance =
    raft::make_device_vector<raft::KeyValuePair<ptrdiff_t, DataT>, IndexT>(handle, n_local_samples);
  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceSelect::If(nullptr,
                                      temp_storage_bytes,
                                      ip_itr,
                                      sampledMinClusterDistance.data_handle(),
                                      nSelected.data_handle(),
                                      n_local_samples,
                                      select_op,
                                      stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceSelect::If(workspace.data(),
                                      temp_storage_bytes,
                                      ip_itr,
                                      sampledMinClusterDistance.data_handle(),
                                      nSelected.data_handle(),
                                      n_local_samples,
                                      select_op,
                                      stream));

  IndexT nPtsSampledInRank = 0;
  raft::copy(handle,
             raft::make_host_scalar_view(&nPtsSampledInRank),
             raft::make_device_scalar_view(nSelected.data_handle()));
  raft::resource::sync_stream(handle);

  uint8_t* rawPtr_isSampleCentroid = isSampleCentroid.data_handle();
  thrust::for_each_n(raft::resource::get_thrust_policy(handle),
                     sampledMinClusterDistance.data_handle(),
                     nPtsSampledInRank,
                     [=] __device__(raft::KeyValuePair<ptrdiff_t, DataT> val) {
                       rawPtr_isSampleCentroid[val.key] = 1;
                     });

  inRankCp.resize(nPtsSampledInRank * n_features, stream);

  raft::matrix::gather((DataT*)X.data_handle(),
                       X.extent(1),
                       X.extent(0),
                       sampledMinClusterDistance.data_handle(),
                       nPtsSampledInRank,
                       inRankCp.data(),
                       raft::key_op{},
                       stream);
}

// calculate pairwise distance between 'dataset[n x d]' and 'centroids[k x d]',
// result will be stored in 'pairwiseDistance[n x k]'
template <typename DataT, typename IndexT>
void pairwise_distance_kmeans(raft::resources const& handle,
                              raft::device_matrix_view<const DataT, IndexT> X,
                              raft::device_matrix_view<const DataT, IndexT> centroids,
                              raft::device_matrix_view<DataT, IndexT> pairwiseDistance,
                              cuvs::distance::DistanceType metric)
{
  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);
  auto n_clusters = centroids.extent(0);

  ASSERT(X.extent(1) == centroids.extent(1),
         "# features in dataset and centroids are different (must be same)");

  if (metric == cuvs::distance::DistanceType::L2Expanded) {
    cuvs::distance::distance<cuvs::distance::DistanceType::L2Expanded,
                             DataT,
                             DataT,
                             DataT,
                             raft::layout_c_contiguous,
                             IndexT>(handle, X, centroids, pairwiseDistance);
  } else if (metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    cuvs::distance::distance<cuvs::distance::DistanceType::L2SqrtExpanded,
                             DataT,
                             DataT,
                             DataT,
                             raft::layout_c_contiguous,
                             IndexT>(handle, X, centroids, pairwiseDistance);
  } else {
    RAFT_FAIL("kmeans requires L2Expanded or L2SqrtExpanded distance, have %i",
              static_cast<int>(metric));
  }
}

// shuffle and randomly select 'n_samples_to_gather' from input 'in' and stores
// in 'out' does not modify the input
template <typename DataT, typename IndexT>
void shuffleAndGather(raft::resources const& handle,
                      raft::device_matrix_view<const DataT, IndexT> in,
                      raft::device_matrix_view<DataT, IndexT> out,
                      uint32_t n_samples_to_gather,
                      uint64_t seed)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = in.extent(0);
  auto n_features     = in.extent(1);

  auto indices = raft::make_device_vector<IndexT, IndexT>(handle, n_samples);

  // shuffle indices on device
  raft::random::permute<DataT, IndexT, IndexT>(indices.data_handle(),
                                               nullptr,
                                               nullptr,
                                               (IndexT)in.extent(1),
                                               (IndexT)in.extent(0),
                                               true,
                                               stream);

  raft::matrix::gather((DataT*)in.data_handle(),
                       in.extent(1),
                       in.extent(0),
                       indices.data_handle(),
                       static_cast<IndexT>(n_samples_to_gather),
                       out.data_handle(),
                       stream);
}

// Calculates a <key, value> pair for every sample in input 'X' where key is an
// index to an sample in 'centroids' (index of the nearest centroid) and 'value'
// is the distance between the sample and the 'centroid[key]'
template <typename DataT, typename IndexT>
void minClusterAndDistanceCompute(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<const DataT, IndexT> centroids,
  raft::device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT> minClusterAndDistance,
  raft::device_vector_view<const DataT, IndexT> L2NormX,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  cuvs::distance::DistanceType metric,
  int batch_samples,
  int batch_centroids,
  rmm::device_uvector<char>& workspace);

#define EXTERN_TEMPLATE_MIN_CLUSTER_AND_DISTANCE(DataT, IndexT)                                \
  extern template void minClusterAndDistanceCompute<DataT, IndexT>(                            \
    raft::resources const& handle,                                                             \
    raft::device_matrix_view<const DataT, IndexT> X,                                           \
    raft::device_matrix_view<const DataT, IndexT> centroids,                                   \
    raft::device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT> minClusterAndDistance, \
    raft::device_vector_view<const DataT, IndexT> L2NormX,                                     \
    rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,                                          \
    cuvs::distance::DistanceType metric,                                                       \
    int batch_samples,                                                                         \
    int batch_centroids,                                                                       \
    rmm::device_uvector<char>& workspace);

EXTERN_TEMPLATE_MIN_CLUSTER_AND_DISTANCE(float, int64_t)
EXTERN_TEMPLATE_MIN_CLUSTER_AND_DISTANCE(float, int)
EXTERN_TEMPLATE_MIN_CLUSTER_AND_DISTANCE(double, int64_t)
EXTERN_TEMPLATE_MIN_CLUSTER_AND_DISTANCE(double, int)

#undef EXTERN_TEMPLATE_MIN_CLUSTER_AND_DISTANCE

template <typename DataT, typename IndexT>
void minClusterDistanceCompute(raft::resources const& handle,
                               raft::device_matrix_view<const DataT, IndexT> X,
                               raft::device_matrix_view<DataT, IndexT> centroids,
                               raft::device_vector_view<DataT, IndexT> minClusterDistance,
                               raft::device_vector_view<DataT, IndexT> L2NormX,
                               rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
                               cuvs::distance::DistanceType metric,
                               int batch_samples,
                               int batch_centroids,
                               rmm::device_uvector<char>& workspace);

#define EXTERN_TEMPLATE_MIN_CLUSTER_DISTANCE(DataT, IndexT)      \
  extern template void minClusterDistanceCompute<DataT, IndexT>( \
    raft::resources const& handle,                               \
    raft::device_matrix_view<const DataT, IndexT> X,             \
    raft::device_matrix_view<DataT, IndexT> centroids,           \
    raft::device_vector_view<DataT, IndexT> minClusterDistance,  \
    raft::device_vector_view<DataT, IndexT> L2NormX,             \
    rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,            \
    cuvs::distance::DistanceType metric,                         \
    int batch_samples,                                           \
    int batch_centroids,                                         \
    rmm::device_uvector<char>& workspace);

EXTERN_TEMPLATE_MIN_CLUSTER_DISTANCE(float, int64_t)
EXTERN_TEMPLATE_MIN_CLUSTER_DISTANCE(double, int64_t)
EXTERN_TEMPLATE_MIN_CLUSTER_DISTANCE(float, int)
EXTERN_TEMPLATE_MIN_CLUSTER_DISTANCE(double, int)

#undef EXTERN_TEMPLATE_MIN_CLUSTER_DISTANCE

template <typename DataT, typename IndexT>
void countSamplesInCluster(raft::resources const& handle,
                           const cuvs::cluster::kmeans::params& params,
                           raft::device_matrix_view<const DataT, IndexT> X,
                           raft::device_vector_view<const DataT, IndexT> L2NormX,
                           raft::device_matrix_view<DataT, IndexT> centroids,
                           rmm::device_uvector<char>& workspace,
                           raft::device_vector_view<DataT, IndexT> sampleCountInCluster)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = centroids.extent(0);

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);

  // temporary buffer to store distance matrix, destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // computes minClusterAndDistance[0:n_samples) where  minClusterAndDistance[i]
  // is a <key, value> pair where
  //   'key' is index to an sample in 'centroids' (index of the nearest
  //   centroid) and 'value' is the distance between the sample 'X[i]' and the
  //   'centroid[key]'
  cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute(
    handle,
    X,
    (raft::device_matrix_view<const DataT, IndexT>)centroids,
    minClusterAndDistance.view(),
    L2NormX,
    L2NormBuf_OR_DistBuf,
    params.metric,
    params.batch_samples,
    params.batch_centroids,
    workspace);

  cuda::transform_iterator itr(minClusterAndDistance.data_handle(),
                               cuvs::cluster::kmeans::detail::KeyValueIndexOp<IndexT, DataT>{});

  // count # of samples in each cluster
  countLabels(handle,
              itr,
              sampleCountInCluster.data_handle(),
              (IndexT)n_samples,
              (IndexT)n_clusters,
              workspace);
}

/**
 * @brief Compute centroid adjustments (weighted sums and counts per cluster)
 *
 * This helper function computes:
 * 1. Weighted sum of samples per cluster using reduce_rows_by_key
 * 2. Sum of weights per cluster using reduce_cols_by_key
 *
 * @tparam DataT Data type for samples and weights
 * @tparam IndexT Index type
 * @tparam LabelsIterator Iterator type for cluster labels
 *
 * @param[in]    handle             RAFT resources handle
 * @param[in]    X                  Input samples [n_samples x n_features]
 * @param[in]    sample_weights     Weights for each sample [n_samples]
 * @param[in]    cluster_labels     Cluster assignment for each sample (iterator)
 * @param[in]    n_clusters         Number of clusters
 * @param[inout] centroid_sums      Weighted sum per cluster [n_clusters x n_features]
 * @param[inout] weight_per_cluster Sum of weights per cluster [n_clusters]. Follows the same
 *                                  overwrite-vs-accumulate semantics as `centroid_sums`
 * @param[inout] workspace          Workspace buffer for intermediate operations
 * @param[in]    reset_sums         If true (default), outputs are reset to zero before reducing;
 *                                  if false, this call's contribution is accumulated into the
 *                                  existing `centroid_sums`
 */
template <typename DataT, typename IndexT, typename LabelsIterator>
void compute_centroid_adjustments(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT, raft::row_major> X,
  raft::device_vector_view<const DataT, IndexT> sample_weights,
  LabelsIterator cluster_labels,
  IndexT n_clusters,
  raft::device_matrix_view<DataT, IndexT, raft::row_major> centroid_sums,
  raft::device_vector_view<DataT, IndexT> weight_per_cluster,
  rmm::device_uvector<char>& workspace,
  bool reset_sums = true)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);

  workspace.resize(n_samples, stream);

  raft::linalg::reduce_rows_by_key(X.data_handle(),
                                   X.extent(1),
                                   cluster_labels,
                                   sample_weights.data_handle(),
                                   workspace.data(),
                                   X.extent(0),
                                   X.extent(1),
                                   n_clusters,
                                   centroid_sums.data_handle(),
                                   stream,
                                   reset_sums);

  raft::linalg::reduce_cols_by_key(sample_weights.data_handle(),
                                   cluster_labels,
                                   weight_per_cluster.data_handle(),
                                   static_cast<IndexT>(1),
                                   static_cast<IndexT>(n_samples),
                                   n_clusters,
                                   stream,
                                   reset_sums);
}
/**
 * @brief Finalize centroids by dividing accumulated sums by counts.
 *
 * For clusters with zero count, the old centroid is preserved.
 *
 * @tparam DataT  Data type
 * @tparam IndexT Index type
 *
 * @param[in]  handle          RAFT resources handle
 * @param[in]  centroid_sums   Accumulated weighted sums per cluster [n_clusters x n_features]
 * @param[in]  weight_per_cluster  Sum of weights per cluster [n_clusters]
 * @param[in]  old_centroids   Previous centroids (used for empty clusters) [n_clusters x
 * n_features]
 * @param[out] new_centroids   Output centroids [n_clusters x n_features]
 */
template <typename DataT, typename IndexT>
void finalize_centroids(raft::resources const& handle,
                        raft::device_matrix_view<const DataT, IndexT> centroid_sums,
                        raft::device_vector_view<const DataT, IndexT> weight_per_cluster,
                        raft::device_matrix_view<const DataT, IndexT> old_centroids,
                        raft::device_matrix_view<DataT, IndexT> new_centroids)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(handle,
                                                             raft::make_const_mdspan(centroid_sums),
                                                             weight_per_cluster,
                                                             new_centroids,
                                                             raft::div_checkzero_op{});

  // For empty clusters (count == 0), copy old centroid back
  cub::ArgIndexInputIterator<const DataT*> itr_wt(weight_per_cluster.data_handle());
  raft::matrix::gather_if(
    old_centroids.data_handle(),
    static_cast<int>(old_centroids.extent(1)),
    static_cast<int>(old_centroids.extent(0)),
    itr_wt,
    itr_wt,
    static_cast<int>(weight_per_cluster.size()),
    new_centroids.data_handle(),
    [=] __device__(raft::KeyValuePair<ptrdiff_t, DataT> map) { return map.value == DataT{0}; },
    raft::key_op{},
    stream);
}

/**
 * @brief Compute the squared norm difference between two centroid sets.
 *
 * Writes sum((old_centroids - new_centroids)^2) into @p sqrd_norm_out.
 * Used for convergence checking. Fully asynchronous — no stream sync.
 */
template <typename DataT, typename IndexT>
void compute_centroid_shift(raft::resources const& handle,
                            raft::device_matrix_view<const DataT, IndexT> old_centroids,
                            raft::device_matrix_view<const DataT, IndexT> new_centroids,
                            raft::device_scalar_view<DataT> sqrd_norm_out)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  raft::linalg::mapThenSumReduce(sqrd_norm_out.data_handle(),
                                 old_centroids.size(),
                                 raft::sqdiff_op{},
                                 stream,
                                 old_centroids.data_handle(),
                                 new_centroids.data_handle());
}

/**
 * @brief Evaluate convergence criteria entirely on device.
 *
 * Checks the cost-ratio and centroid-shift stopping conditions and writes
 * 0 or 1 into @p done_flag, and advances @p prior_clustering_cost.
 * @p FlagT is deduced from @p done_flag (default `int`).
 */
template <typename DataT, typename FlagT = int>
__device__ void check_convergence(raft::device_scalar_view<const DataT> clustering_cost,
                                  raft::device_scalar_view<DataT> prior_clustering_cost,
                                  raft::device_scalar_view<const DataT> sqrd_norm_error,
                                  DataT tol,
                                  int n_iter,
                                  raft::device_scalar_view<FlagT> done_flag)
{
  DataT cur_cost = *clustering_cost.data_handle();
  DataT norm_err = *sqrd_norm_error.data_handle();
  FlagT done     = FlagT{0};

  if (cur_cost != DataT{0} && n_iter > 1) {
    DataT delta = cur_cost / *prior_clustering_cost.data_handle();
    if (delta > DataT{1} - tol) done = FlagT{1};
  }
  if (norm_err < tol) done = FlagT{1};

  *prior_clustering_cost.data_handle() = cur_cost;
  *done_flag.data_handle()             = done;
}

/**
 * @brief Process a single batch of data in the Lloyd iteration.
 *
 * Given one batch of data + precomputed norms + weights + current centroids it
 *   1. finds the nearest centroid for every sample,
 *   2. accumulates weighted centroid sums and counts into the running accumulators,
 *   3. accumulates the weighted clustering cost (inertia).
 *
 * Data norms must be precomputed by the caller and passed in via L2NormBatch.
 *
 * @tparam DataT  Data / weight type (float, double)
 * @tparam IndexT Index type (int, int64_t)
 *
 * @param[in]     handle               RAFT resources handle
 * @param[in]     batch_data           Device batch data [batch_size x n_features]
 * @param[in]     batch_weights        Device batch weights [batch_size]
 * @param[in]     centroids            Current centroids [n_clusters x n_features]
 * @param[in]     metric               Distance metric
 * @param[in]     batch_samples_param  Batch-samples param forwarded to minClusterAndDistanceCompute
 * @param[in]     batch_centroids_param Batch-centroids param forwarded to
 *                                      minClusterAndDistanceCompute
 * @param[inout]  minClusterAndDistance Work buffer [batch_size]
 * @param[in]     L2NormBatch          Precomputed data norms [batch_size]
 * @param[inout]  L2NormBuf_OR_DistBuf Resizable scratch
 * @param[inout]  workspace            Resizable scratch
 * @param[inout]  centroid_sums        Running weighted sums [n_clusters x n_features] (added into)
 * @param[inout]  weight_per_cluster   Running weight counts [n_clusters] (added into)
 * @param[inout]  clustering_cost      Running cost scalar (device) (added into)
 */
template <typename DataT, typename IndexT>
void process_batch(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> batch_data,
  raft::device_vector_view<const DataT, IndexT> batch_weights,
  raft::device_matrix_view<const DataT, IndexT> centroids,
  cuvs::distance::DistanceType metric,
  int batch_samples_param,
  int batch_centroids_param,
  raft::device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT> minClusterAndDistance,
  raft::device_vector_view<const DataT, IndexT> L2NormBatch,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  rmm::device_uvector<char>& workspace,
  raft::device_matrix_view<DataT, IndexT> centroid_sums,
  raft::device_vector_view<DataT, IndexT> weight_per_cluster,
  raft::device_scalar_view<DataT> clustering_cost,
  rmm::device_uvector<char>& batch_workspace)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  minClusterAndDistanceCompute<DataT, IndexT>(handle,
                                              batch_data,
                                              centroids,
                                              minClusterAndDistance,
                                              L2NormBatch,
                                              L2NormBuf_OR_DistBuf,
                                              metric,
                                              batch_samples_param,
                                              batch_centroids_param,
                                              workspace);

  KeyValueIndexOp<IndexT, DataT> conversion_op;
  thrust::transform_iterator<KeyValueIndexOp<IndexT, DataT>,
                             const raft::KeyValuePair<IndexT, DataT>*>
    labels_itr(minClusterAndDistance.data_handle(), conversion_op);

  compute_centroid_adjustments(handle,
                               batch_data,
                               batch_weights,
                               labels_itr,
                               static_cast<IndexT>(centroid_sums.extent(0)),
                               centroid_sums,
                               weight_per_cluster,
                               batch_workspace,
                               /*reset_sums=*/false);

  raft::linalg::map(
    handle,
    minClusterAndDistance,
    [=] __device__(const raft::KeyValuePair<IndexT, DataT> kvp, DataT wt) {
      raft::KeyValuePair<IndexT, DataT> res;
      res.value = kvp.value * wt;
      res.key   = kvp.key;
      return res;
    },
    raft::make_const_mdspan(minClusterAndDistance),
    batch_weights);

  auto batch_cost = raft::make_device_scalar<DataT>(handle, DataT{0});
  computeClusterCost(
    handle, minClusterAndDistance, workspace, batch_cost.view(), raft::value_op{}, raft::add_op{});
  raft::linalg::add(clustering_cost.data_handle(),
                    clustering_cost.data_handle(),
                    batch_cost.data_handle(),
                    1,
                    stream);
}

}  // namespace cuvs::cluster::kmeans::detail
