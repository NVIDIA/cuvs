/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace epsilon_neighborhood {

/**
 * @defgroup epsilon_neighborhood_cpp_l2 Epsilon Neighborhood L2 Operations
 * @{
 */

/**
 * @brief Computes epsilon neighborhood for the given distance metric and ball size.
 * The epsilon neighbors is represented by a dense boolean adjacency matrix of size m * n and
 * an array of degrees for each vertex, which can be used as a compressed sparse row (CSR)
 * indptr array.
 *
 * Currently, only L2Unexpanded (L2-squared) distance metric is supported. Other metrics will
 * throw an exception.
 *
 * @code{.cpp}
 *  #include <cuvs/neighbors/epsilon_neighborhood.hpp>
 *  #include <raft/core/resources.hpp>
 *  #include <raft/core/device_mdarray.hpp>
 *  using namespace cuvs::neighbors;
 *  raft::resources handle;
 *  ...
 *  auto x = raft::make_device_matrix<float, int64_t>(handle, m, k);
 *  auto y = raft::make_device_matrix<float, int64_t>(handle, n, k);
 *  auto adj = raft::make_device_matrix<bool, int64_t>(handle, m, n);
 *  auto vd = raft::make_device_vector<int, int64_t>(handle, m + 1);
 *  epsilon_neighborhood::compute(handle, x.view(), y.view(), adj.view(), vd.view(),
 *                                eps, cuvs::distance::DistanceType::L2Unexpanded);
 * @endcode
 *
 * @tparam value_t   IO and math type
 * @tparam idx_t    Index type
 * @tparam matrix_idx_t matrix indexing type
 *
 * @param[in]  handle raft handle to manage library resources
 * @param[in]  x      first matrix [row-major] [on device] [dim = m x k]
 * @param[in]  y      second matrix [row-major] [on device] [dim = n x k]
 * @param[out] adj    adjacency matrix [row-major] [on device] [dim = m x n]
 * @param[out] vd     vertex degree array [on device] [len = m + 1]
 *                    `vd + m` stores the total number of edges in the adjacency
 *                    matrix. Pass a nullptr if you don't need this info.
 * @param[in]  eps    defines epsilon neighborhood radius (should be passed as
 *                    squared when using L2Unexpanded metric)
 * @param[in]  metric distance metric to use. Currently only L2Unexpanded is supported.
 */
template <typename value_t, typename idx_t, typename matrix_idx_t>
void compute(raft::resources const& handle,
             raft::device_matrix_view<const value_t, matrix_idx_t, raft::row_major> x,
             raft::device_matrix_view<const value_t, matrix_idx_t, raft::row_major> y,
             raft::device_matrix_view<bool, matrix_idx_t, raft::row_major> adj,
             raft::device_vector_view<idx_t, matrix_idx_t> vd,
             value_t eps,
             cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

/** @} */  // end group epsilon_neighborhood_cpp_l2

}  // namespace epsilon_neighborhood
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
