/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include "kmeans_impl.cuh"
#include <raft/core/logger.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resources.hpp>

#ifdef CUVS_BUILD_MG_ALGOS
#include "detail/kmeans_mg_batched.cuh"
#endif

namespace cuvs::cluster::kmeans {

#define INSTANTIATE_FIT_MAIN(DataT, IndexT)                       \
  template void fit_main<DataT, IndexT>(                          \
    raft::resources const& handle,                                \
    const kmeans::params& params,                                 \
    raft::device_matrix_view<const DataT, IndexT> X,              \
    raft::device_vector_view<const DataT, IndexT> sample_weights, \
    raft::device_matrix_view<DataT, IndexT> centroids,            \
    raft::host_scalar_view<DataT> inertia,                        \
    raft::host_scalar_view<IndexT> n_iter,                        \
    rmm::device_uvector<char>& workspace);

#define INSTANTIATE_FIT(DataT, IndexT)                                          \
  template void fit<DataT, IndexT>(                                             \
    raft::resources const& handle,                                              \
    const kmeans::params& params,                                               \
    raft::device_matrix_view<const DataT, IndexT> X,                            \
    std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight, \
    raft::device_matrix_view<DataT, IndexT> centroids,                          \
    raft::host_scalar_view<DataT> inertia,                                      \
    raft::host_scalar_view<IndexT> n_iter);

INSTANTIATE_FIT_MAIN(double, int)
INSTANTIATE_FIT_MAIN(double, int64_t)

INSTANTIATE_FIT(double, int)
INSTANTIATE_FIT(double, int64_t)

#undef INSTANTIATE_FIT_MAIN
#undef INSTANTIATE_FIT

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const double, int> X,
         std::optional<raft::device_vector_view<const double, int>> sample_weight,
         raft::device_matrix_view<double, int> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int> n_iter)
{
  cuvs::cluster::kmeans::fit<double, int>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const double, int64_t> X,
         std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  cuvs::cluster::kmeans::fit<double, int64_t>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const double, int64_t> X,
         std::optional<raft::host_vector_view<const double, int64_t>> sample_weight,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
#ifdef CUVS_BUILD_MG_ALGOS
  if (raft::resource::is_multi_gpu(handle)) {
    mg::detail::batched_fit_omp<double, int64_t>(
      handle, params, X, sample_weight, centroids, inertia, n_iter);
  } else if (raft::resource::comms_initialized(handle)) {
    mg::detail::mnmg_fit<double, int64_t>(
      handle, params, X, sample_weight, centroids, inertia, n_iter);
  } else
#else
  if (raft::resource::is_multi_gpu(handle) || raft::resource::comms_initialized(handle)) {
    RAFT_LOG_WARN(
      "MG handle detected but cuVS built without MG support; falling back to single-GPU.");
  }
#endif
  {
    cuvs::cluster::kmeans::detail::fit<double, int64_t>(
      handle, params, X, sample_weight, centroids, inertia, n_iter);
  }
}

}  // namespace cuvs::cluster::kmeans
