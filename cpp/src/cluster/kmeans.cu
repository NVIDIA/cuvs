/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../neighbors/detail/ann_utils.cuh"
#include "detail/kmeans_auto_find_k.cuh"
#include "kmeans_balanced.cuh"
#include "kmeans_impl.cuh"

#include <raft/core/resources.hpp>

#include <optional>
#include <vector>

namespace cuvs::cluster::kmeans {

#define CUVS_INST_KMEANS_FIT(DataT, MatrixViewT, VectorViewT, IndexT)     \
  void fit(raft::resources const& handle,                                 \
           const cuvs::cluster::kmeans::params& params,                   \
           MatrixViewT<const DataT, IndexT> X,                            \
           std::optional<VectorViewT<const DataT, IndexT>> sample_weight, \
           raft::device_matrix_view<DataT, IndexT> centroids,             \
           raft::host_scalar_view<DataT> inertia,                         \
           raft::host_scalar_view<IndexT> n_iter)                         \
  {                                                                       \
    cuvs::cluster::kmeans::fit<DataT, IndexT>(                            \
      handle, params, X, sample_weight, centroids, inertia, n_iter);      \
  }

CUVS_INST_KMEANS_FIT(float, raft::device_matrix_view, raft::device_vector_view, int)
CUVS_INST_KMEANS_FIT(float, raft::device_matrix_view, raft::device_vector_view, int64_t)
CUVS_INST_KMEANS_FIT(float, raft::host_matrix_view, raft::host_vector_view, int64_t)
CUVS_INST_KMEANS_FIT(double, raft::device_matrix_view, raft::device_vector_view, int)
CUVS_INST_KMEANS_FIT(double, raft::device_matrix_view, raft::device_vector_view, int64_t)
CUVS_INST_KMEANS_FIT(double, raft::host_matrix_view, raft::host_vector_view, int64_t)

#undef CUVS_INST_KMEANS_FIT

#ifdef CUVS_BUILD_MG_ALGOS
// Multi-GPU overloads that accept several local partitions per rank.
#define CUVS_INST_KMEANS_FIT_MG(DataT, MatrixViewT, VectorViewT, IndexT)                     \
  void fit(                                                                                  \
    raft::resources const& handle,                                                           \
    const cuvs::cluster::kmeans::params& params,                                             \
    const std::vector<MatrixViewT<const DataT, IndexT>>& X_parts,                            \
    const std::optional<std::vector<VectorViewT<const DataT, IndexT>>>& sample_weight_parts, \
    raft::device_matrix_view<DataT, IndexT> centroids,                                       \
    raft::host_scalar_view<DataT> inertia,                                                   \
    raft::host_scalar_view<IndexT> n_iter)                                                   \
  {                                                                                          \
    cuvs::cluster::kmeans::mg::detail::mnmg_fit<DataT, IndexT>(                              \
      handle, params, X_parts, sample_weight_parts, centroids, inertia, n_iter);             \
  }

CUVS_INST_KMEANS_FIT_MG(float, raft::device_matrix_view, raft::device_vector_view, int)
CUVS_INST_KMEANS_FIT_MG(float, raft::device_matrix_view, raft::device_vector_view, int64_t)
CUVS_INST_KMEANS_FIT_MG(float, raft::host_matrix_view, raft::host_vector_view, int64_t)
CUVS_INST_KMEANS_FIT_MG(double, raft::device_matrix_view, raft::device_vector_view, int)
CUVS_INST_KMEANS_FIT_MG(double, raft::device_matrix_view, raft::device_vector_view, int64_t)
CUVS_INST_KMEANS_FIT_MG(double, raft::host_matrix_view, raft::host_vector_view, int64_t)

#undef CUVS_INST_KMEANS_FIT_MG
#endif

#define CUVS_INST_KMEANS_PREDICT(DataT, IndexT)                                            \
  void predict(raft::resources const& handle,                                              \
               const kmeans::params& params,                                               \
               raft::device_matrix_view<const DataT, IndexT> X,                            \
               std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight, \
               raft::device_matrix_view<const DataT, IndexT> centroids,                    \
               raft::device_vector_view<IndexT, IndexT> labels,                            \
               bool normalize_weight,                                                      \
               raft::host_scalar_view<DataT> inertia)                                      \
  {                                                                                        \
    cuvs::cluster::kmeans::predict<DataT, IndexT>(                                         \
      handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);     \
  }

CUVS_INST_KMEANS_PREDICT(float, int)
CUVS_INST_KMEANS_PREDICT(float, int64_t)
CUVS_INST_KMEANS_PREDICT(double, int)
CUVS_INST_KMEANS_PREDICT(double, int64_t)

#undef CUVS_INST_KMEANS_PREDICT

#define CUVS_INST_KMEANS_CLUSTER_COST(DataT, IndexT)                                               \
  void cluster_cost(const raft::resources& handle,                                                 \
                    raft::device_matrix_view<const DataT, IndexT> X,                               \
                    raft::device_matrix_view<const DataT, IndexT> centroids,                       \
                    raft::host_scalar_view<DataT> cost,                                            \
                    std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight)    \
  {                                                                                                \
    cuvs::cluster::kmeans::cluster_cost<DataT, IndexT>(handle, X, centroids, cost, sample_weight); \
  }

CUVS_INST_KMEANS_CLUSTER_COST(float, int)
CUVS_INST_KMEANS_CLUSTER_COST(float, int64_t)
CUVS_INST_KMEANS_CLUSTER_COST(double, int)
CUVS_INST_KMEANS_CLUSTER_COST(double, int64_t)

#undef CUVS_INST_KMEANS_CLUSTER_COST

#define CUVS_INST_KMEANS_FIT_PREDICT(DataT, IndexT)                                            \
  void fit_predict(raft::resources const& handle,                                              \
                   const kmeans::params& params,                                               \
                   raft::device_matrix_view<const DataT, IndexT> X,                            \
                   std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight, \
                   std::optional<raft::device_matrix_view<DataT, IndexT>> centroids,           \
                   raft::device_vector_view<IndexT, IndexT> labels,                            \
                   raft::host_scalar_view<DataT> inertia,                                      \
                   raft::host_scalar_view<IndexT> n_iter)                                      \
  {                                                                                            \
    cuvs::cluster::kmeans::fit_predict<DataT, IndexT>(                                         \
      handle, params, X, sample_weight, centroids, labels, inertia, n_iter);                   \
  }

CUVS_INST_KMEANS_FIT_PREDICT(float, int)
CUVS_INST_KMEANS_FIT_PREDICT(float, int64_t)
CUVS_INST_KMEANS_FIT_PREDICT(double, int)
CUVS_INST_KMEANS_FIT_PREDICT(double, int64_t)

#undef CUVS_INST_KMEANS_FIT_PREDICT

#define CUVS_INST_KMEANS_TRANSFORM(DataT)                                              \
  void transform(raft::resources const& handle,                                        \
                 const kmeans::params& params,                                         \
                 raft::device_matrix_view<const DataT, int> X,                         \
                 raft::device_matrix_view<const DataT, int> centroids,                 \
                 raft::device_matrix_view<DataT, int> X_new)                           \
  {                                                                                    \
    cuvs::cluster::kmeans::transform<DataT, int>(handle, params, X, centroids, X_new); \
  }

CUVS_INST_KMEANS_TRANSFORM(float)
CUVS_INST_KMEANS_TRANSFORM(double)

#undef CUVS_INST_KMEANS_TRANSFORM

#define CUVS_INST_KMEANS_BALANCED_FIT(DataT)                                                       \
  void fit(const raft::resources& handle,                                                          \
           cuvs::cluster::kmeans::balanced_params const& params,                                   \
           raft::device_matrix_view<const DataT, int64_t> X,                                       \
           raft::device_matrix_view<float, int64_t> centroids,                                     \
           std::optional<raft::host_scalar_view<float>> inertia)                                   \
  {                                                                                                \
    cuvs::cluster::kmeans_balanced::fit(                                                           \
      handle, params, X, centroids, cuvs::spatial::knn::detail::utils::mapping<float>{}, inertia); \
  }

CUVS_INST_KMEANS_BALANCED_FIT(float)
CUVS_INST_KMEANS_BALANCED_FIT(half)
CUVS_INST_KMEANS_BALANCED_FIT(int8_t)
CUVS_INST_KMEANS_BALANCED_FIT(uint8_t)

#undef CUVS_INST_KMEANS_BALANCED_FIT

#define CUVS_INST_KMEANS_BALANCED_PREDICT(DataT, LabelT)                                          \
  void predict(const raft::resources& handle,                                                     \
               cuvs::cluster::kmeans::balanced_params const& params,                              \
               raft::device_matrix_view<const DataT, int64_t> X,                                  \
               raft::device_matrix_view<const float, int64_t> centroids,                          \
               raft::device_vector_view<LabelT, int64_t> labels)                                  \
  {                                                                                               \
    cuvs::cluster::kmeans_balanced::predict(                                                      \
      handle, params, X, centroids, labels, cuvs::spatial::knn::detail::utils::mapping<float>{}); \
  }

CUVS_INST_KMEANS_BALANCED_PREDICT(float, uint32_t)
CUVS_INST_KMEANS_BALANCED_PREDICT(float, int)
CUVS_INST_KMEANS_BALANCED_PREDICT(half, uint32_t)
CUVS_INST_KMEANS_BALANCED_PREDICT(int8_t, uint32_t)
CUVS_INST_KMEANS_BALANCED_PREDICT(int8_t, int)
CUVS_INST_KMEANS_BALANCED_PREDICT(uint8_t, uint32_t)

#undef CUVS_INST_KMEANS_BALANCED_PREDICT

#define CUVS_INST_KMEANS_BALANCED_FIT_PREDICT(DataT, LabelT)                           \
  void fit_predict(const raft::resources& handle,                                      \
                   cuvs::cluster::kmeans::balanced_params const& params,               \
                   raft::device_matrix_view<const DataT, int64_t> X,                   \
                   raft::device_matrix_view<float, int64_t> centroids,                 \
                   raft::device_vector_view<LabelT, int64_t> labels)                   \
  {                                                                                    \
    cuvs::cluster::kmeans_balanced::fit_predict(handle, params, X, centroids, labels); \
  }

CUVS_INST_KMEANS_BALANCED_FIT_PREDICT(float, uint32_t)
CUVS_INST_KMEANS_BALANCED_FIT_PREDICT(int8_t, uint32_t)
CUVS_INST_KMEANS_BALANCED_FIT_PREDICT(int8_t, int)

#undef CUVS_INST_KMEANS_BALANCED_FIT_PREDICT

}  // namespace cuvs::cluster::kmeans

namespace cuvs::cluster::kmeans_balanced::helpers {

void build_clusters(const raft::resources& handle,
                    const cuvs::cluster::kmeans::balanced_params& params,
                    raft::device_matrix_view<const float, int64_t> X,
                    raft::device_matrix_view<float, int64_t> centroids,
                    raft::device_vector_view<uint32_t, int64_t> labels,
                    raft::device_vector_view<uint32_t, int64_t> cluster_sizes,
                    cuvs::spatial::knn::detail::utils::mapping<float> mapping_op,
                    std::optional<raft::device_vector_view<const float>> X_norm)
{
  build_clusters<float,
                 float,
                 int64_t,
                 uint32_t,
                 uint32_t,
                 cuvs::spatial::knn::detail::utils::mapping<float>>(
    handle, params, X, centroids, labels, cluster_sizes, mapping_op, X_norm);
}

}  // namespace cuvs::cluster::kmeans_balanced::helpers

namespace cuvs::cluster::kmeans::helpers {

void find_k(raft::resources const& handle,
            raft::device_matrix_view<const float, int> X,
            raft::host_scalar_view<int> best_k,
            raft::host_scalar_view<float> inertia,
            raft::host_scalar_view<int> n_iter,
            int kmax,
            int kmin,
            int maxiter,
            float tol)
{
  cuvs::cluster::kmeans::detail::find_k<int, float>(
    handle, X, best_k, inertia, n_iter, kmax, kmin, maxiter, tol);
}

}  // namespace cuvs::cluster::kmeans::helpers
