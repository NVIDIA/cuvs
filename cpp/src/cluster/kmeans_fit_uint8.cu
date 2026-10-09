/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include "kmeans_impl.cuh"

#include <raft/core/error.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <optional>

namespace cuvs::cluster::kmeans {

template void fit<uint8_t, float, int64_t>(
  raft::resources const& handle,
  const kmeans::params& params,
  raft::device_matrix_view<const uint8_t, int64_t> X,
  std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
  raft::device_matrix_view<float, int64_t> centroids,
  raft::host_scalar_view<float> inertia,
  raft::host_scalar_view<int64_t> n_iter);

namespace {

void expect_single_gpu_uint8_fit(raft::resources const& handle)
{
  RAFT_EXPECTS(!raft::resource::is_multi_gpu(handle) && !raft::resource::comms_initialized(handle),
               "Regular Lloyd k-means fit with uint8_t input supports single-GPU handles only");
}

}  // namespace

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const uint8_t, int64_t> X,
         std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  expect_single_gpu_uint8_fit(handle);
  cuvs::cluster::kmeans::fit<uint8_t, float, int64_t>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const uint8_t, int64_t> X,
         std::optional<raft::host_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  expect_single_gpu_uint8_fit(handle);
  cuvs::cluster::kmeans::detail::fit<uint8_t, float, int64_t>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

}  // namespace cuvs::cluster::kmeans
