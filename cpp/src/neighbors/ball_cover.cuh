/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "ball_cover/ball_cover.cuh"
#include "ball_cover/registers_types.cuh"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ball_cover.hpp>

#include <thrust/transform.h>

#include <cstdint>

namespace cuvs::neighbors::ball_cover::detail {

/**
 * @defgroup random_ball_cover Random Ball Cover algorithm
 * @{
 */

/**
 * Builds and populates a previously unbuilt cuvs::neighbors::ball_cover::index
 *
 * Usage example:
 * @code{.cpp}
 *
 *  #include <raft/core/resources.hpp>
 *  #include <cuvs/neighbors/ball_cover.cuh>
 *  #include <cuvs/distance/distance.hpp>
 *  using namespace cuvs::neighbors;
 *
 *  raft::resources handle;
 *  ...
 *  auto metric = cuvs::distance::DistanceType::L2Expanded;
 *  cuvs::neighbors::ball_cover::index index(handle, X, metric);
 *
 *  ball_cover::build_index(handle, index);
 * @endcode
 *
 * @tparam idx_t knn index type
 * @tparam value_t knn value type
 * @tparam int_t integral type for knn params
 * @tparam matrix_idx_t matrix indexing type
 * @param[in] handle library resource management handle
 * @param[inout] index an empty (and not previous built) instance of
 * cuvs::neighbors::ball_cover::index
 */
template <typename idx_t, typename value_t>
void build_index(raft::resources const& handle,
                 cuvs::neighbors::ball_cover::index<idx_t, value_t>& index)
{
  RAFT_EXPECTS(index.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                 index.metric == cuvs::distance::DistanceType::L2SqrtUnexpanded ||
                 index.metric == cuvs::distance::DistanceType::Haversine,
               "Metric not supported");
  cuvs::neighbors::ball_cover::detail::rbc_build_index(handle, index);

  index.set_index_trained();
}

/** @} */  // end group random_ball_cover

/**
 * Performs a faster exact knn in metric spaces using the triangle
 * inequality with a number of landmark points to reduce the
 * number of distance computations from O(n^2) to O(sqrt(n)). This
 * performs an all neighbors knn, which can reuse memory when
 * the index and query are the same array. This function will
 * build the index and assumes rbc_build_index() has not already
 * been called.
 * @tparam idx_t knn index type
 * @tparam value_t knn distance type
 * @tparam int_t type for integers, such as number of rows/cols
 * @param[in] handle raft handle for resource management
 * @param[inout] index ball cover index which has not yet been built
 * @param[in] k number of nearest neighbors to find
 * @param[in] perform_post_filtering if this is false, only the closest k landmarks
 *                               are considered (which will return approximate
 *                               results).
 * @param[out] inds output knn indices
 * @param[out] dists output knn distances
 * @param[in] weight a weight for overlap between the closest landmark and
 *               the radius of other landmarks when pruning distances.
 *               Setting this value below 1 can effectively turn off
 *               computing distances against many other balls, enabling
 *               approximate nearest neighbors. Recall can be adjusted
 *               based on how many relevant balls are ignored. Note that
 *               many datasets can still have great recall even by only
 *               looking in the closest landmark.
 */
template <typename idx_t, typename value_t>
void all_knn_query(raft::resources const& handle,
                   cuvs::neighbors::ball_cover::index<idx_t, value_t>& index,
                   int64_t k,
                   idx_t* inds,
                   value_t* dists,
                   bool perform_post_filtering = true,
                   float weight                = 1.0)
{
  ASSERT(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  RAFT_EXPECTS(index.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                 index.metric == cuvs::distance::DistanceType::L2SqrtUnexpanded ||
                 index.metric == cuvs::distance::DistanceType::Haversine,
               "Metric not supported");
  cuvs::neighbors::ball_cover::detail::rbc_all_knn_query(
    handle, index, k, inds, dists, perform_post_filtering, weight);

  index.set_index_trained();
}

/**
 * @ingroup random_ball_cover
 * @{
 */

/**
 * Performs a faster exact knn in metric spaces using the triangle
 * inequality with a number of landmark points to reduce the
 * number of distance computations from O(n^2) to O(sqrt(n)). This
 * performs an all neighbors knn, which can reuse memory when
 * the index and query are the same array. This function will
 * build the index and assumes rbc_build_index() has not already
 * been called.
 *
 * Usage example:
 * @code{.cpp}
 *
 *  #include <raft/core/resources.hpp>
 *  #include <cuvs/neighbors/ball_cover.hpp>
 *  #include <cuvs/distance/distance.hpp>
 *  using namespace cuvs::neighbors;
 *
 *  raft::resources handle;
 *  ...
 *  auto metric = cuvs::distance::DistanceType::L2Expanded;
 *
 *  // Construct a ball cover index
 *  cuvs::neighbors::ball_cover::index index(handle, X, metric);
 *
 *  // Perform all neighbors knn query
 *  ball_cover::all_knn_query(handle, index, inds, dists, k);
 * @endcode
 *
 * @tparam idx_t knn index type
 * @tparam value_t knn distance type
 * @tparam int_t type for integers, such as number of rows/cols
 * @tparam matrix_idx_t matrix indexing type
 *
 * @param[in] handle raft handle for resource management
 * @param[in] index ball cover index which has not yet been built
 * @param[out] inds output knn indices
 * @param[out] dists output knn distances
 * @param[in] perform_post_filtering if this is false, only the closest k landmarks
 *                               are considered (which will return approximate
 *                               results).
 * @param[in] weight a weight for overlap between the closest landmark and
 *               the radius of other landmarks when pruning distances.
 *               Setting this value below 1 can effectively turn off
 *               computing distances against many other balls, enabling
 *               approximate nearest neighbors. Recall can be adjusted
 *               based on how many relevant balls are ignored. Note that
 *               many datasets can still have great recall even by only
 *               looking in the closest landmark.
 */
template <typename idx_t, typename value_t>
void all_knn_query(raft::resources const& handle,
                   cuvs::neighbors::ball_cover::index<idx_t, value_t>& index,
                   raft::device_matrix_view<idx_t, int64_t, raft::row_major> inds,
                   raft::device_matrix_view<value_t, int64_t, raft::row_major> dists,
                   bool perform_post_filtering = true,
                   float weight                = 1.0)
{
  RAFT_EXPECTS(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  RAFT_EXPECTS(inds.extent(1) <= index.m,
               "k must be less than or equal to the number of data points in the index");
  RAFT_EXPECTS(inds.extent(1) == dists.extent(1),
               "Number of columns in output indices and distances matrices must be equal");

  RAFT_EXPECTS(inds.extent(0) == dists.extent(0) && dists.extent(0) == index.get_X().extent(0),
               "Number of rows in output indices and distances matrices must equal number of rows "
               "in index matrix.");

  all_knn_query(handle,
                index,
                inds.extent(1),
                inds.data_handle(),
                dists.data_handle(),
                perform_post_filtering,
                weight);
}

/** @} */

/**
 * Performs a faster exact knn in metric spaces using the triangle
 * inequality with a number of landmark points to reduce the
 * number of distance computations from O(n^2) to O(sqrt(n)). This
 * function does not build the index and assumes rbc_build_index() has
 * already been called. Use this function when the index and
 * query arrays are different, otherwise use rbc_all_knn_query().
 * @tparam idx_t index type
 * @tparam value_t distances type
 * @tparam int_t integer type for size info
 * @param[in] handle raft handle for resource management
 * @param[inout] index ball cover index which has not yet been built
 * @param[in] k number of nearest neighbors to find
 * @param[in] query the
 * @param[in] perform_post_filtering if this is false, only the closest k landmarks
 *                               are considered (which will return approximate
 *                               results).
 * @param[out] inds output knn indices
 * @param[out] dists output knn distances
 * @param[in] weight a weight for overlap between the closest landmark and
 *               the radius of other landmarks when pruning distances.
 *               Setting this value below 1 can effectively turn off
 *               computing distances against many other balls, enabling
 *               approximate nearest neighbors. Recall can be adjusted
 *               based on how many relevant balls are ignored. Note that
 *               many datasets can still have great recall even by only
 *               looking in the closest landmark.
 * @param[in] n_query_pts number of query points
 */
template <typename idx_t, typename value_t>
void knn_query(raft::resources const& handle,
               const cuvs::neighbors::ball_cover::index<idx_t, value_t>& index,
               int64_t k,
               const value_t* query,
               int64_t n_query_pts,
               idx_t* inds,
               value_t* dists,
               bool perform_post_filtering = true,
               float weight                = 1.0)
{
  ASSERT(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  RAFT_EXPECTS(index.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                 index.metric == cuvs::distance::DistanceType::L2SqrtUnexpanded ||
                 index.metric == cuvs::distance::DistanceType::Haversine,
               "Metric not supported");
  cuvs::neighbors::ball_cover::detail::rbc_knn_query(
    handle, index, k, query, n_query_pts, inds, dists, perform_post_filtering, weight);
}

/**
 * @brief Computes epsilon neighborhood for the L2 distance metric using rbc
 *
 * @tparam value_t   IO and math type
 * @tparam idx_t    Index type
 *
 * @param[in] handle raft handle for resource management
 * @param[in] index ball cover index which has been built
 * @param[out] adj    adjacency matrix [row-major] [on device] [dim = m x n]
 * @param[out] vd     vertex degree array [on device] [len = m + 1]
 *                    `vd + m` stores the total number of edges in the adjacency
 *                    matrix. Pass a nullptr if you don't need this info.
 * @param[in]  query  first matrix [row-major] [on device] [dim = m x k]
 * @param[in]  eps    defines epsilon neighborhood radius
 */
template <typename idx_t, typename value_t>
void eps_nn(raft::resources const& handle,
            const cuvs::neighbors::ball_cover::index<idx_t, value_t>& index,
            raft::device_matrix_view<bool, int64_t, raft::row_major> adj,
            raft::device_vector_view<idx_t, int64_t> vd,
            raft::device_matrix_view<const value_t, int64_t, raft::row_major> query,
            value_t eps)
{
  ASSERT(index.n == query.extent(1), "vector dimension needs to be the same for index and queries");
  ASSERT(index.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
           index.metric == cuvs::distance::DistanceType::L2SqrtUnexpanded,
         "Metric not supported");
  ASSERT(index.is_index_trained(), "index must be previously trained");

  // run query
  cuvs::neighbors::ball_cover::detail::rbc_eps_nn_query(
    handle,
    index,
    eps,
    query.data_handle(),
    query.extent(0),
    adj.data_handle(),
    vd.data_handle(),
    cuvs::neighbors::ball_cover::detail::EuclideanSqFunc<value_t, int64_t>());
}

/**
 * @brief Computes epsilon neighborhood for the L2 distance metric using rbc
 *
 * @tparam value_t   IO and math type
 * @tparam idx_t    Index type
 *
 * @param[in] handle raft handle for resource management
 * @param[in] index ball cover index which has been built
 * @param[out] adj_ia    adjacency matrix CSR row offsets
 * @param[out] adj_ja    adjacency matrix CSR column indices, needs to be nullptr
 *                       in first pass with max_k nullopt
 * @param[out] vd     vertex degree array [on device] [len = m + 1]
 *                    `vd + m` stores the total number of edges in the adjacency
 *                    matrix. Pass a nullptr if you don't need this info.
 * @param[in]  query  first matrix [row-major] [on device] [dim = m x k]
 * @param[in]  eps    defines epsilon neighborhood radius
 * @param[inout] max_k if nullopt (default), the user needs to make 2 subsequent calls:
 *                     The first call computes row offsets in adj_ia, where adj_ia[m]
 *                     contains the minimum required size for adj_ja.
 *                     The second call fills in adj_ja based on adj_ia.
 *                     If max_k != nullopt the algorithm only fills up neighbors up to a
 *                     maximum number of max_k for each row in a single pass. Note
 *                     that it is not guarantueed to return the nearest neighbors.
 *                     Upon return max_k is overwritten with the actual max_k found during
 *                     computation.
 * @param[out] dists   An optional vector (with same length as adj_ja). If provided, will be
 *                     used to store the corresponding distances for the computed neighbors.
 */
template <typename idx_t, typename value_t>
void eps_nn(raft::resources const& handle,
            const cuvs::neighbors::ball_cover::index<idx_t, value_t>& index,
            raft::device_vector_view<idx_t, int64_t> adj_ia,
            raft::device_vector_view<idx_t, int64_t> adj_ja,
            raft::device_vector_view<idx_t, int64_t> vd,
            raft::device_matrix_view<const value_t, int64_t, raft::row_major> query,
            value_t eps,
            std::optional<raft::host_scalar_view<int64_t, int64_t>> max_k   = std::nullopt,
            std::optional<raft::device_vector_view<value_t, int64_t>> dists = std::nullopt)
{
  ASSERT(index.n == query.extent(1), "vector dimension needs to be the same for index and queries");
  ASSERT(index.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
           index.metric == cuvs::distance::DistanceType::L2SqrtUnexpanded,
         "Metric not supported");
  ASSERT(index.is_index_trained(), "index must be previously trained");

  int64_t* max_k_ptr = nullptr;
  if (max_k.has_value()) { max_k_ptr = max_k.value().data_handle(); }

  value_t* dists_ptr = nullptr;
  if (dists.has_value()) { dists_ptr = dists.value().data_handle(); }

  // run query
  cuvs::neighbors::ball_cover::detail::rbc_eps_nn_query(
    handle,
    index,
    eps,
    max_k_ptr,
    query.data_handle(),
    query.extent(0),
    adj_ia.data_handle(),
    adj_ja.data_handle(),
    vd.data_handle(),
    dists_ptr,
    cuvs::neighbors::ball_cover::detail::EuclideanSqFunc<value_t, int64_t>());
}

/**
 * @ingroup random_ball_cover
 * @{
 */

/**
 * Performs a faster exact knn in metric spaces using the triangle
 * inequality with a number of landmark points to reduce the
 * number of distance computations from O(n^2) to O(sqrt(n)). This
 * function does not build the index and assumes rbc_build_index() has
 * already been called. Use this function when the index and
 * query arrays are different, otherwise use rbc_all_knn_query().
 *
 * Usage example:
 * @code{.cpp}
 *
 *  #include <raft/core/resources.hpp>
 *  #include <cuvs/neighbors/ball_cover.cuh>
 *  #include <cuvs/distance/distance.hpp>
 *  using namespace cuvs::neighbors;
 *
 *  raft::resources handle;
 *  ...
 *  auto metric = cuvs::distance::DistanceType::L2Expanded;
 *
 *  // Build a ball cover index
 *  cuvs::neighbors::ball_cover::index index(handle, X, metric);
 *  ball_cover::build_index(handle, index);
 *
 *  // Perform all neighbors knn query
 *  ball_cover::knn_query(handle, index, inds, dists, k);
 * @endcode

 *
 * @tparam idx_t index type
 * @tparam value_t distances type
 * @tparam int_t integer type for size info
 * @tparam matrix_idx_t
 * @param[in] handle raft handle for resource management
 * @param[in] index ball cover index which has not yet been built
 * @param[in] query device matrix containing query data points
 * @param[out] inds output knn indices
 * @param[out] dists output knn distances
 * @param[in] perform_post_filtering if this is false, only the closest k landmarks
 *                               are considered (which will return approximate
 *                               results).
 * @param[in] weight a weight for overlap between the closest landmark and
 *               the radius of other landmarks when pruning distances.
 *               Setting this value below 1 can effectively turn off
 *               computing distances against many other balls, enabling
 *               approximate nearest neighbors. Recall can be adjusted
 *               based on how many relevant balls are ignored. Note that
 *               many datasets can still have great recall even by only
 *               looking in the closest landmark.
 */
template <typename idx_t, typename value_t>
void knn_query(raft::resources const& handle,
               const cuvs::neighbors::ball_cover::index<idx_t, value_t>& index,
               raft::device_matrix_view<const value_t, int64_t, raft::row_major> query,
               raft::device_matrix_view<idx_t, int64_t, raft::row_major> inds,
               raft::device_matrix_view<value_t, int64_t, raft::row_major> dists,
               bool perform_post_filtering = true,
               float weight                = 1.0)
{
  RAFT_EXPECTS(inds.extent(1) <= index.m,
               "k must be less than or equal to the number of data points in the index");
  RAFT_EXPECTS(inds.extent(1) == dists.extent(1),
               "Number of columns in output indices and distances matrices must be equal");

  RAFT_EXPECTS(inds.extent(0) == dists.extent(0) && dists.extent(0) == query.extent(0),
               "Number of rows in output indices and distances matrices must equal number of rows "
               "in search matrix.");

  RAFT_EXPECTS(query.extent(1) == index.get_X().extent(1),
               "Number of columns in query and index matrices must match.");

  knn_query(handle,
            index,
            inds.extent(1),
            query.data_handle(),
            query.extent(0),
            inds.data_handle(),
            dists.data_handle(),
            perform_post_filtering,
            weight);
}

/** @} */

// TODO: implement functions for:
//  4. rbc_eps_neigh() - given a populated index, perform query against different query array
//  5. rbc_all_eps_neigh() - populate a cuvs::neighbors::ball_cover::index and query against
//  training data

}  // namespace cuvs::neighbors::ball_cover::detail
