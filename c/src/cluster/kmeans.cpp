/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <limits>

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

constexpr int64_t kKMeansInt32IndexMax = std::numeric_limits<int32_t>::max();

bool dlpack_shape_exceeds_int32_index(const DLTensor& tensor)
{
  for (int i = 0; i < tensor.ndim; ++i) {
    if (tensor.shape[i] > kKMeansInt32IndexMax) { return true; }
  }
  return false;
}

bool kmeans_tensor_shapes_use_int64_index(DLManagedTensor* X, DLManagedTensor* centroids)
{
  if (dlpack_shape_exceeds_int32_index(X->dl_tensor)) { return true; }
  if (dlpack_shape_exceeds_int32_index(centroids->dl_tensor)) { return true; }
  return false;
}

bool kmeans_fit_uses_int64_index(DLManagedTensor* X,
                                 DLManagedTensor* centroids,
                                 int n_clusters)
{
  if (static_cast<int64_t>(n_clusters) > kKMeansInt32IndexMax) { return true; }
  return kmeans_tensor_shapes_use_int64_index(X, centroids);
}

bool kmeans_labels_use_int64_index(const DLTensor& labels)
{
  return labels.dtype.code == kDLInt && labels.dtype.bits == 64;
}

void validate_kmeans_labels_dtype(const DLTensor& labels)
{
  if (labels.dtype.code == kDLInt && (labels.dtype.bits == 32 || labels.dtype.bits == 64)) {
    return;
  }
  RAFT_FAIL("Unsupported labels DLtensor dtype: %d and bits: %d",
            labels.dtype.code,
            labels.dtype.bits);
}

bool kmeans_predict_uses_int64_index(DLManagedTensor* X,
                                     DLManagedTensor* centroids,
                                     DLManagedTensor* labels,
                                     int n_clusters)
{
  validate_kmeans_labels_dtype(labels->dl_tensor);
  if (kmeans_labels_use_int64_index(labels->dl_tensor)) { return true; }
  return kmeans_fit_uses_int64_index(X, centroids, n_clusters);
}

template <typename T, typename ParamsT, typename IdxT>
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
    // Host fit overloads are only exposed with int64_t index types.
    using HostIdxT = int64_t;
    auto n_samples  = static_cast<HostIdxT>(X.shape[0]);
    auto n_features = static_cast<HostIdxT>(X.shape[1]);

    if (params.hierarchical) {
      RAFT_FAIL("hierarchical kmeans is not supported with host data");
    }

    auto centroids_dl = centroids_tensor->dl_tensor;
    if (!cuvs::core::is_dlpack_device_compatible(centroids_dl)) {
      RAFT_FAIL("centroids must be on device memory");
    }

    auto X_view = raft::make_host_matrix_view<T const, HostIdxT>(
      reinterpret_cast<T const*>(X.data), n_samples, n_features);
    auto centroids_view =
      cuvs::core::from_dlpack<raft::device_matrix_view<T, HostIdxT, raft::row_major>>(
        centroids_tensor);

    std::optional<raft::host_vector_view<T const, HostIdxT>> sample_weight;
    if (sample_weight_tensor != NULL) {
      auto sw = sample_weight_tensor->dl_tensor;
      if (!cuvs::core::is_dlpack_host_compatible(sw)) {
        RAFT_FAIL("sample_weight must be host accessible when X is on host");
      }
      sample_weight = raft::make_host_vector_view<T const, HostIdxT>(
        reinterpret_cast<T const*>(sw.data), n_samples);
    }

    T inertia_temp;
    HostIdxT n_iter_temp;

    auto kmeans_params = convert_params(params);
    cuvs::cluster::kmeans::fit(*res_ptr,
                               kmeans_params,
                               X_view,
                               sample_weight,
                               centroids_view,
                               raft::make_host_scalar_view<T>(&inertia_temp),
                               raft::make_host_scalar_view<HostIdxT>(&n_iter_temp));

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
        // Balanced fit overloads are only exposed with int64_t index types.
        using BalancedIdxT = int64_t;
        using balanced_const_mdspan_type =
          raft::device_matrix_view<T const, BalancedIdxT, raft::row_major>;
        using balanced_mdspan_type = raft::device_matrix_view<T, BalancedIdxT, raft::row_major>;
        auto kmeans_params         = convert_balanced_params(params);
        T inertia_temp;
        auto inertia_view = raft::make_host_scalar_view<T>(&inertia_temp);
        cuvs::cluster::kmeans::fit(
          *res_ptr,
          kmeans_params,
          cuvs::core::from_dlpack<balanced_const_mdspan_type>(X_tensor),
          cuvs::core::from_dlpack<balanced_mdspan_type>(centroids_tensor),
          std::make_optional(inertia_view));
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

template <typename T, typename ParamsT, typename IdxT, typename LabelsT>
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
      } else if constexpr (!std::is_same_v<LabelsT, int32_t>) {
        RAFT_FAIL("int64 labels are unsupported for hierarchical kmeans");
      } else {
        // Balanced predict overloads are only exposed with int64_t index types and int32 labels.
        using BalancedIdxT = int64_t;
        using balanced_const_mdspan_type =
          raft::device_matrix_view<T const, BalancedIdxT, raft::row_major>;
        using balanced_labels_mdspan_type =
          raft::device_vector_view<int32_t, BalancedIdxT, raft::row_major>;
        auto kmeans_params = convert_balanced_params(params);
        cuvs::cluster::kmeans::predict(
          *res_ptr,
          kmeans_params,
          cuvs::core::from_dlpack<balanced_const_mdspan_type>(X_tensor),
          cuvs::core::from_dlpack<balanced_const_mdspan_type>(centroids_tensor),
          cuvs::core::from_dlpack<balanced_labels_mdspan_type>(labels_tensor));
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

template <typename T, typename IdxT>
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
    const bool use_int64_index =
      kmeans_fit_uses_int64_index(X, centroids, params->n_clusters);
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      if (use_int64_index) {
        _fit<float, cuvsKMeansParams, int64_t>(
          res, *params, X, sample_weight, centroids, inertia, n_iter);
      } else {
        _fit<float, cuvsKMeansParams, int32_t>(
          res, *params, X, sample_weight, centroids, inertia, n_iter);
      }
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      if (use_int64_index) {
        _fit<double, cuvsKMeansParams, int64_t>(
          res, *params, X, sample_weight, centroids, inertia, n_iter);
      } else {
        _fit<double, cuvsKMeansParams, int32_t>(
          res, *params, X, sample_weight, centroids, inertia, n_iter);
      }
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
    const bool use_int64_index =
      kmeans_predict_uses_int64_index(X, centroids, labels, params->n_clusters);
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      if (use_int64_index) {
        _predict<float, cuvsKMeansParams, int64_t, int64_t>(
          res, *params, X, sample_weight, centroids, labels, normalize_weight, inertia);
      } else {
        _predict<float, cuvsKMeansParams, int32_t, int32_t>(
          res, *params, X, sample_weight, centroids, labels, normalize_weight, inertia);
      }
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      if (use_int64_index) {
        _predict<double, cuvsKMeansParams, int64_t, int64_t>(
          res, *params, X, sample_weight, centroids, labels, normalize_weight, inertia);
      } else {
        _predict<double, cuvsKMeansParams, int32_t, int32_t>(
          res, *params, X, sample_weight, centroids, labels, normalize_weight, inertia);
      }
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
    const bool use_int64_index =
      kmeans_fit_uses_int64_index(X, centroids, params->n_clusters);
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      if (use_int64_index) {
        _fit<float, cuvsKMeansParams_v2, int64_t>(
          res, *params, X, sample_weight, centroids, inertia, n_iter);
      } else {
        _fit<float, cuvsKMeansParams_v2, int32_t>(
          res, *params, X, sample_weight, centroids, inertia, n_iter);
      }
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      if (use_int64_index) {
        _fit<double, cuvsKMeansParams_v2, int64_t>(
          res, *params, X, sample_weight, centroids, inertia, n_iter);
      } else {
        _fit<double, cuvsKMeansParams_v2, int32_t>(
          res, *params, X, sample_weight, centroids, inertia, n_iter);
      }
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
    const bool use_int64_index =
      kmeans_predict_uses_int64_index(X, centroids, labels, params->n_clusters);
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      if (use_int64_index) {
        _predict<float, cuvsKMeansParams_v2, int64_t, int64_t>(
          res, *params, X, sample_weight, centroids, labels, normalize_weight, inertia);
      } else {
        _predict<float, cuvsKMeansParams_v2, int32_t, int32_t>(
          res, *params, X, sample_weight, centroids, labels, normalize_weight, inertia);
      }
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      if (use_int64_index) {
        _predict<double, cuvsKMeansParams_v2, int64_t, int64_t>(
          res, *params, X, sample_weight, centroids, labels, normalize_weight, inertia);
      } else {
        _predict<double, cuvsKMeansParams_v2, int32_t, int32_t>(
          res, *params, X, sample_weight, centroids, labels, normalize_weight, inertia);
      }
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
    const bool use_int64_index = kmeans_tensor_shapes_use_int64_index(X, centroids);
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      if (use_int64_index) {
        _cluster_cost<float, int64_t>(res, X, centroids, cost);
      } else {
        _cluster_cost<float, int32_t>(res, X, centroids, cost);
      }
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      if (use_int64_index) {
        _cluster_cost<double, int64_t>(res, X, centroids, cost);
      } else {
        _cluster_cost<double, int32_t>(res, X, centroids, cost);
      }
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}
