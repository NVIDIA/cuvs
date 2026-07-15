/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <dlpack/dlpack.h>

#include <cuvs/cluster/kmeans.h>
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/core/c_api.h>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace {

// The conversions are templated on the C struct type and reused by both API surfaces.
template <typename ParamsT>
cuvs::cluster::kmeans::params convert_params(const ParamsT& params)
{
  auto kmeans_params                = cuvs::cluster::kmeans::params();
  kmeans_params.metric              = static_cast<cuvs::distance::DistanceType>(params.metric);
  kmeans_params.init = static_cast<cuvs::cluster::kmeans::params::InitMethod>(params.init);
  kmeans_params.n_clusters          = params.n_clusters;
  kmeans_params.max_iter            = params.max_iter;
  kmeans_params.tol                 = params.tol;
  kmeans_params.n_init              = params.n_init;
  kmeans_params.oversampling_factor = params.oversampling_factor;
  kmeans_params.batch_samples       = params.batch_samples;
  kmeans_params.batch_centroids     = params.batch_centroids;
  kmeans_params.init_size             = params.init_size;
  kmeans_params.streaming_batch_size  = params.streaming_batch_size;
  return kmeans_params;
}

template <typename ParamsT>
cuvs::cluster::kmeans::balanced_params convert_balanced_params(const ParamsT& params)
{
  auto kmeans_params    = cuvs::cluster::kmeans::balanced_params();
  kmeans_params.metric  = static_cast<cuvs::distance::DistanceType>(params.metric);
  kmeans_params.n_iters = params.hierarchical_n_iters;
  return kmeans_params;
}

template <typename T, typename ParamsT, typename IdxT = int64_t>
void _fit(cuvsResources_t res,
          const ParamsT& params,
          DLManagedTensor* X_tensor,
          DLManagedTensor* sample_weight_tensor,
          DLManagedTensor* centroids_tensor,
          double* inertia,
          int* n_iter)
{
  auto X       = X_tensor->dl_tensor;
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  if (!cuvs::core::is_dlpack_device_compatible(X)) {
    auto n_samples  = static_cast<IdxT>(X.shape[0]);
    auto n_features = static_cast<IdxT>(X.shape[1]);

    if (params.hierarchical) {
      RAFT_FAIL("hierarchical kmeans is not supported with host data");
    }

    auto centroids_dl = centroids_tensor->dl_tensor;
    if (!cuvs::core::is_dlpack_device_compatible(centroids_dl)) {
      RAFT_FAIL("centroids must be on device memory");
    }

    auto X_view = raft::make_host_matrix_view<T const, IdxT>(
      reinterpret_cast<T const*>(X.data), n_samples, n_features);
    auto centroids_view =
      cuvs::core::from_dlpack<raft::device_matrix_view<T, IdxT, raft::row_major>>(
        centroids_tensor);

    std::optional<raft::host_vector_view<T const, IdxT>> sample_weight;
    if (sample_weight_tensor != NULL) {
      auto sw = sample_weight_tensor->dl_tensor;
      if (!cuvs::core::is_dlpack_host_compatible(sw)) {
        RAFT_FAIL("sample_weight must be host accessible when X is on host");
      }
      sample_weight = raft::make_host_vector_view<T const, IdxT>(
        reinterpret_cast<T const*>(sw.data), n_samples);
    }

    T inertia_temp;
    IdxT n_iter_temp;

    auto kmeans_params = convert_params(params);
    cuvs::cluster::kmeans::fit(*res_ptr,
                               kmeans_params,
                               X_view,
                               sample_weight,
                               centroids_view,
                               raft::make_host_scalar_view<T>(&inertia_temp),
                               raft::make_host_scalar_view<IdxT>(&n_iter_temp));

    *inertia = inertia_temp;
    *n_iter  = n_iter_temp;

  } else {
    using const_mdspan_type = raft::device_matrix_view<T const, IdxT, raft::row_major>;
    using mdspan_type       = raft::device_matrix_view<T, IdxT, raft::row_major>;

    if (params.hierarchical) {
      if (sample_weight_tensor != NULL) {
        RAFT_FAIL("sample_weight cannot be used with hierarchical kmeans");
      }

      if constexpr (std::is_same_v<T, double>) {
        RAFT_FAIL("float64 is an unsupported dtype for hierarchical kmeans");
      } else {
        auto kmeans_params = convert_balanced_params(params);
        T inertia_temp;
        auto inertia_view = raft::make_host_scalar_view<T>(&inertia_temp);
        cuvs::cluster::kmeans::fit(*res_ptr, kmeans_params, cuvs::core::from_dlpack<const_mdspan_type>(X_tensor), cuvs::core::from_dlpack<mdspan_type>(centroids_tensor), std::make_optional(inertia_view));
        *inertia = inertia_temp;
        *n_iter  = params.hierarchical_n_iters;
      }
    } else {
      T inertia_temp;
      IdxT n_iter_temp;

      std::optional<raft::device_vector_view<T const, IdxT>> sample_weight;
      if (sample_weight_tensor != NULL) {
        sample_weight =
          cuvs::core::from_dlpack<raft::device_vector_view<T const, IdxT>>(sample_weight_tensor);
      }

      auto kmeans_params = convert_params(params);
      cuvs::cluster::kmeans::fit(*res_ptr,
                                 kmeans_params,
                                 cuvs::core::from_dlpack<const_mdspan_type>(X_tensor),
                                 sample_weight,
                                 cuvs::core::from_dlpack<mdspan_type>(centroids_tensor),
                                 raft::make_host_scalar_view<T>(&inertia_temp),
                                 raft::make_host_scalar_view<IdxT>(&n_iter_temp));
      *inertia = inertia_temp;
      *n_iter  = n_iter_temp;
    }
  }
}

template <typename T, typename ParamsT, typename IdxT = int32_t, typename LabelsT = int32_t>
void _predict(cuvsResources_t res,
              const ParamsT& params,
              DLManagedTensor* X_tensor,
              DLManagedTensor* sample_weight_tensor,
              DLManagedTensor* centroids_tensor,
              DLManagedTensor* labels_tensor,
              bool normalize_weight,
              double* inertia)
{
  auto X       = X_tensor->dl_tensor;
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  if (cuvs::core::is_dlpack_device_compatible(X)) {
    using labels_mdspan_type = raft::device_vector_view<LabelsT, IdxT, raft::row_major>;
    using const_mdspan_type  = raft::device_matrix_view<T const, IdxT, raft::row_major>;

    if (params.hierarchical) {
      if (sample_weight_tensor != NULL) {
        RAFT_FAIL("sample_weight cannot be used with hierarchical kmeans");
      }

      if constexpr (std::is_same_v<T, double>) {
        RAFT_FAIL("float64 is an unsupported dtype for hierarchical kmeans");
      } else {
        auto kmeans_params = convert_balanced_params(params);
        cuvs::cluster::kmeans::predict(*res_ptr,
                                       kmeans_params,
                                       cuvs::core::from_dlpack<const_mdspan_type>(X_tensor),
                                       cuvs::core::from_dlpack<const_mdspan_type>(centroids_tensor),
                                       cuvs::core::from_dlpack<labels_mdspan_type>(labels_tensor));
        *inertia = 0;
      }
    } else {
      auto kmeans_params = convert_params(params);
      T inertia_temp;
      std::optional<raft::device_vector_view<T const, IdxT>> sample_weight;
      if (sample_weight_tensor != NULL) {
        sample_weight =
          cuvs::core::from_dlpack<raft::device_vector_view<T const, IdxT>>(sample_weight_tensor);
      }
      cuvs::cluster::kmeans::predict(*res_ptr,
                                     kmeans_params,
                                     cuvs::core::from_dlpack<const_mdspan_type>(X_tensor),
                                     sample_weight,
                                     cuvs::core::from_dlpack<const_mdspan_type>(centroids_tensor),
                                     cuvs::core::from_dlpack<labels_mdspan_type>(labels_tensor),
                                     normalize_weight,
                                     raft::make_host_scalar_view<T>(&inertia_temp));
      *inertia = inertia_temp;
    }
  } else {
    RAFT_FAIL("X dataset must be accessible on device memory");
  }
}

template <typename T, typename IdxT = int32_t>
void _cluster_cost(cuvsResources_t res,
                   DLManagedTensor* X_tensor,
                   DLManagedTensor* centroids_tensor,
                   double* cost)
{
  auto X       = X_tensor->dl_tensor;
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  T cost_temp;

  if (cuvs::core::is_dlpack_device_compatible(X)) {
    using mdspan_type = raft::device_matrix_view<const T, IdxT, raft::row_major>;

    cuvs::cluster::kmeans::cluster_cost(*res_ptr,
                                        cuvs::core::from_dlpack<mdspan_type>(X_tensor),
                                        cuvs::core::from_dlpack<mdspan_type>(centroids_tensor),
                                        raft::make_host_scalar_view<T>(&cost_temp));
  } else {
    RAFT_FAIL("X dataset must be accessible on device memory");
  }

  *cost = cost_temp;
}
}  // namespace

extern "C" cuvsError_t cuvsKMeansParamsCreate(cuvsKMeansParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    cuvs::cluster::kmeans::params cpp_params;
    cuvs::cluster::kmeans::balanced_params cpp_balanced_params;
    *params = new cuvsKMeansParams{
      .metric               = static_cast<cuvsDistanceType>(cpp_params.metric),
      .n_clusters           = cpp_params.n_clusters,
      .init                 = static_cast<cuvsKMeansInitMethod>(cpp_params.init),
      .max_iter             = cpp_params.max_iter,
      .tol                  = cpp_params.tol,
      .n_init               = cpp_params.n_init,
      .oversampling_factor  = cpp_params.oversampling_factor,
      .batch_samples        = cpp_params.batch_samples,
      .batch_centroids      = cpp_params.batch_centroids,
      .inertia_check        = false,
      .hierarchical         = false,
      .hierarchical_n_iters = static_cast<int>(cpp_balanced_params.n_iters),
      .streaming_batch_size = cpp_params.streaming_batch_size,
      .init_size            = cpp_params.init_size};
  });
}

extern "C" cuvsError_t cuvsKMeansParamsDestroy(cuvsKMeansParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsKMeansFit(cuvsResources_t res,
                                     cuvsKMeansParams_t params,
                                     DLManagedTensor* X,
                                     DLManagedTensor* sample_weight,
                                     DLManagedTensor* centroids,
                                     double* inertia,
                                     int* n_iter)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = X->dl_tensor;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      _fit<float>(res, *params, X, sample_weight, centroids, inertia, n_iter);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      _fit<double>(res, *params, X, sample_weight, centroids, inertia, n_iter);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsKMeansPredict(cuvsResources_t res,
                                         cuvsKMeansParams_t params,
                                         DLManagedTensor* X,
                                         DLManagedTensor* sample_weight,
                                         DLManagedTensor* centroids,
                                         DLManagedTensor* labels,
                                         bool normalize_weight,
                                         double* inertia)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = X->dl_tensor;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
        _predict<float>(res, *params, X, sample_weight, centroids, labels, normalize_weight, inertia);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
        _predict<double>(res, *params, X, sample_weight, centroids, labels, normalize_weight, inertia);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsKMeansParamsCreate_v2(cuvsKMeansParams_v2_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    cuvs::cluster::kmeans::params cpp_params;
    cuvs::cluster::kmeans::balanced_params cpp_balanced_params;
    *params = new cuvsKMeansParams_v2{
      .metric               = static_cast<cuvsDistanceType>(cpp_params.metric),
      .n_clusters           = cpp_params.n_clusters,
      .init                 = static_cast<cuvsKMeansInitMethod>(cpp_params.init),
      .max_iter             = cpp_params.max_iter,
      .tol                  = cpp_params.tol,
      .n_init               = cpp_params.n_init,
      .oversampling_factor  = cpp_params.oversampling_factor,
      .batch_samples        = cpp_params.batch_samples,
      .batch_centroids      = cpp_params.batch_centroids,
      .hierarchical         = false,
      .hierarchical_n_iters = static_cast<int>(cpp_balanced_params.n_iters),
      .streaming_batch_size = cpp_params.streaming_batch_size,
      .init_size            = cpp_params.init_size};
  });
}

extern "C" cuvsError_t cuvsKMeansParamsDestroy_v2(cuvsKMeansParams_v2_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsKMeansFit_v2(cuvsResources_t res,
                                        cuvsKMeansParams_v2_t params,
                                        DLManagedTensor* X,
                                        DLManagedTensor* sample_weight,
                                        DLManagedTensor* centroids,
                                        double* inertia,
                                        int* n_iter)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = X->dl_tensor;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      _fit<float>(res, *params, X, sample_weight, centroids, inertia, n_iter);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      _fit<double>(res, *params, X, sample_weight, centroids, inertia, n_iter);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsKMeansPredict_v2(cuvsResources_t res,
                                            cuvsKMeansParams_v2_t params,
                                            DLManagedTensor* X,
                                            DLManagedTensor* sample_weight,
                                            DLManagedTensor* centroids,
                                            DLManagedTensor* labels,
                                            bool normalize_weight,
                                            double* inertia)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = X->dl_tensor;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      _predict<float>(res, *params, X, sample_weight, centroids, labels, normalize_weight, inertia);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      _predict<double>(
        res, *params, X, sample_weight, centroids, labels, normalize_weight, inertia);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsKMeansClusterCost(cuvsResources_t res,
                                             DLManagedTensor* X,
                                             DLManagedTensor* centroids,
                                             double* cost)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = X->dl_tensor;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      _cluster_cost<float>(res, X, centroids, cost);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      _cluster_cost<double>(res, X, centroids, cost);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}
