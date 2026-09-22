/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <optional>

#include <dlpack/dlpack.h>

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/cluster/mg_kmeans.h>
#include <cuvs/core/c_api.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace {

template <typename ParamsT>
cuvs::cluster::kmeans::params convert_params(const ParamsT& params)
{
  auto kmeans_params       = cuvs::cluster::kmeans::params();
  kmeans_params.metric     = static_cast<cuvs::distance::DistanceType>(params.metric);
  kmeans_params.init       = static_cast<cuvs::cluster::kmeans::params::InitMethod>(params.init);
  kmeans_params.n_clusters = params.n_clusters;
  kmeans_params.max_iter   = params.max_iter;
  kmeans_params.tol        = params.tol;
  kmeans_params.n_init     = params.n_init;
  kmeans_params.oversampling_factor  = params.oversampling_factor;
  kmeans_params.batch_samples        = params.batch_samples;
  kmeans_params.batch_centroids      = params.batch_centroids;
  kmeans_params.init_size            = params.init_size;
  kmeans_params.device_buffer_samples = params.device_buffer_samples;
  return kmeans_params;
}

void validate_host_tensor(DLManagedTensor* tensor, const char* name)
{
  RAFT_EXPECTS(tensor != nullptr, "%s must not be NULL", name);
  auto dl_tensor = tensor->dl_tensor;
  RAFT_EXPECTS(dl_tensor.data != nullptr, "%s data must not be NULL", name);
  RAFT_EXPECTS(dl_tensor.shape != nullptr, "%s shape must not be NULL", name);
  RAFT_EXPECTS(
    cuvs::core::is_dlpack_host_compatible(dl_tensor), "%s must be host accessible", name);
  RAFT_EXPECTS(dl_tensor.device.device_type != kDLCUDA, "%s must reside in host memory", name);
  RAFT_EXPECTS(cuvs::core::is_c_contiguous(tensor), "%s must be C-contiguous", name);
}

bool dtype_equal(const DLTensor& lhs, const DLTensor& rhs)
{
  return lhs.dtype.code == rhs.dtype.code && lhs.dtype.bits == rhs.dtype.bits &&
         lhs.dtype.lanes == rhs.dtype.lanes;
}

void validate_float_dtype(const DLTensor& tensor, const char* name)
{
  RAFT_EXPECTS(
    tensor.dtype.code == kDLFloat && (tensor.dtype.bits == 32 || tensor.dtype.bits == 64),
    "%s must have dtype float32 or float64",
    name);
  RAFT_EXPECTS(tensor.dtype.lanes == 1, "%s must have one DLPack lane", name);
}

template <typename ParamsT>
void validate_inputs(const ParamsT& params,
                     DLManagedTensor* X_tensor,
                     DLManagedTensor* sample_weight_tensor,
                     DLManagedTensor* centroids_tensor)
{
  RAFT_EXPECTS(params.n_clusters > 0, "n_clusters must be positive");
  RAFT_EXPECTS(!params.hierarchical, "hierarchical kmeans is not supported by SNMG kmeans");

  validate_host_tensor(X_tensor, "X");
  validate_host_tensor(centroids_tensor, "centroids");

  auto X         = X_tensor->dl_tensor;
  auto centroids = centroids_tensor->dl_tensor;

  RAFT_EXPECTS(X.ndim == 2, "X must be a 2D tensor");
  RAFT_EXPECTS(centroids.ndim == 2, "centroids must be a 2D tensor");
  RAFT_EXPECTS(X.shape[0] > 0, "X must have at least one row");
  RAFT_EXPECTS(X.shape[1] > 0, "X must have at least one column");
  RAFT_EXPECTS(centroids.shape[0] == params.n_clusters,
               "centroids row count must equal n_clusters");
  RAFT_EXPECTS(centroids.shape[1] == X.shape[1],
               "centroids column count must equal X column count");

  validate_float_dtype(X, "X");
  RAFT_EXPECTS(dtype_equal(X, centroids), "centroids dtype must match X dtype");

  if (sample_weight_tensor != nullptr) {
    validate_host_tensor(sample_weight_tensor, "sample_weight");
    auto sample_weight = sample_weight_tensor->dl_tensor;
    RAFT_EXPECTS(sample_weight.ndim == 1, "sample_weight must be a 1D tensor");
    RAFT_EXPECTS(sample_weight.shape[0] == X.shape[0],
                 "sample_weight length must equal X row count");
    RAFT_EXPECTS(dtype_equal(X, sample_weight), "sample_weight dtype must match X dtype");
  }
}

template <typename T, typename ParamsT, typename IdxT = int64_t>
void fit_snmg(cuvsResources_t res,
              const ParamsT& params,
              DLManagedTensor* X_tensor,
              DLManagedTensor* sample_weight_tensor,
              DLManagedTensor* centroids_tensor,
              double* inertia,
              int* n_iter)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  RAFT_EXPECTS(res_ptr != nullptr, "res must not be NULL");
  RAFT_EXPECTS(raft::resource::is_multi_gpu(*res_ptr),
               "cuvsMultiGpuKMeansFit requires a MultiGpuResources handle");

  auto X         = X_tensor->dl_tensor;
  auto centroids = centroids_tensor->dl_tensor;

  auto n_samples  = static_cast<IdxT>(X.shape[0]);
  auto n_features = static_cast<IdxT>(X.shape[1]);
  auto n_clusters = static_cast<IdxT>(params.n_clusters);

  auto X_view = raft::make_host_matrix_view<T const, IdxT>(
    reinterpret_cast<T const*>(X.data), n_samples, n_features);

  std::optional<raft::host_vector_view<T const, IdxT>> sample_weight;
  if (sample_weight_tensor != nullptr) {
    auto sw = sample_weight_tensor->dl_tensor;
    sample_weight =
      raft::make_host_vector_view<T const, IdxT>(reinterpret_cast<T const*>(sw.data), n_samples);
  }

  auto const& rank0_res  = raft::resource::set_current_device_to_rank(*res_ptr, 0);
  auto stream            = raft::resource::get_cuda_stream(rank0_res);
  auto d_centroids       = raft::make_device_matrix<T, IdxT>(rank0_res, n_clusters, n_features);
  auto n_centroid_values = n_clusters * n_features;

  if (params.init == Array) {
    raft::update_device(d_centroids.data_handle(),
                        reinterpret_cast<T const*>(centroids.data),
                        n_centroid_values,
                        stream);
    raft::resource::sync_stream(rank0_res, stream);
  }

  T inertia_temp     = T{0};
  IdxT n_iter_temp   = IdxT{0};
  auto kmeans_params = convert_params(params);
  cuvs::cluster::kmeans::fit(*res_ptr,
                             kmeans_params,
                             X_view,
                             sample_weight,
                             d_centroids.view(),
                             raft::make_host_scalar_view<T>(&inertia_temp),
                             raft::make_host_scalar_view<IdxT>(&n_iter_temp));

  raft::update_host(
    reinterpret_cast<T*>(centroids.data), d_centroids.data_handle(), n_centroid_values, stream);
  raft::resource::sync_stream(rank0_res, stream);

  *inertia = static_cast<double>(inertia_temp);
  *n_iter  = static_cast<int>(n_iter_temp);
}

template <typename ParamsT>
void dispatch_fit(cuvsResources_t res,
                  ParamsT params,
                  DLManagedTensor* X,
                  DLManagedTensor* sample_weight,
                  DLManagedTensor* centroids,
                  double* inertia,
                  int* n_iter)
{
  RAFT_EXPECTS(res != 0, "res must not be NULL");
  RAFT_EXPECTS(params != nullptr, "params must not be NULL");
  RAFT_EXPECTS(inertia != nullptr, "inertia must not be NULL");
  RAFT_EXPECTS(n_iter != nullptr, "n_iter must not be NULL");

  validate_inputs(*params, X, sample_weight, centroids);

  auto dataset = X->dl_tensor;
  if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
    fit_snmg<float>(res, *params, X, sample_weight, centroids, inertia, n_iter);
  } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
    fit_snmg<double>(res, *params, X, sample_weight, centroids, inertia, n_iter);
  } else {
    RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
              dataset.dtype.code,
              dataset.dtype.bits);
  }
}

}  // namespace

extern "C" cuvsError_t cuvsMultiGpuKMeansFit(cuvsResources_t res,
                                             cuvsKMeansParams_t params,
                                             DLManagedTensor* X,
                                             DLManagedTensor* sample_weight,
                                             DLManagedTensor* centroids,
                                             double* inertia,
                                             int* n_iter)
{
  return cuvs::core::translate_exceptions(
    [=] { dispatch_fit(res, params, X, sample_weight, centroids, inertia, n_iter); });
}
