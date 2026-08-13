/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>

#include <cuvs/core/export.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
/**
 * @defgroup ann_refine Approximate Nearest Neighbors Refinement
 * @{
 */

/**
 * @brief Refine nearest neighbor search.
 *
 * Refinement is an operation that follows an approximate NN search. The approximate search has
 * already selected n_candidates neighbor candidates for each query. We narrow it down to k
 * neighbors. For each query, we calculate the exact distance between the query and its
 * n_candidates neighbor candidate, and select the k nearest ones.
 *
 * The k nearest neighbors and distances are returned.
 *
 * Example usage
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // search m = 4 * k nearest neighbours for each of the N queries
 *   ivf_pq::search(handle, search_params, index, queries, neighbor_candidates,
 *                  out_dists_tmp);
 *   // refine it to the k nearest one
 *   refine(handle, dataset, queries, neighbor_candidates, out_indices, out_dists,
 *           index.metric());
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] dataset device matrix that stores the dataset [n_rows, dims]
 * @param[in] queries device matrix of the queries [n_queris, dims]
 * @param[in] neighbor_candidates indices of candidate vectors [n_queries, n_candidates], where
 *   n_candidates >= k
 * @param[out] indices device matrix that stores the refined indices [n_queries, k]
 * @param[out] distances device matrix that stores the refined distances [n_queries, k]
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 */
void refine(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> indices,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

/**
 * @brief Refine nearest neighbor search.
 *
 * Refinement is an operation that follows an approximate NN search. The approximate search has
 * already selected n_candidates neighbor candidates for each query. We narrow it down to k
 * neighbors. For each query, we calculate the exact distance between the query and its
 * n_candidates neighbor candidate, and select the k nearest ones.
 *
 * The k nearest neighbors and distances are returned.
 *
 * Example usage
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // search m = 4 * k nearest neighbours for each of the N queries
 *   ivf_pq::search(handle, search_params, index, queries, neighbor_candidates,
 *                  out_dists_tmp);
 *   // refine it to the k nearest one
 *   refine(handle, dataset, queries, neighbor_candidates, out_indices, out_dists,
 *           index.metric());
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] dataset device matrix that stores the dataset [n_rows, dims]
 * @param[in] queries device matrix of the queries [n_queris, dims]
 * @param[in] neighbor_candidates indices of candidate vectors [n_queries, n_candidates], where
 *   n_candidates >= k
 * @param[out] indices device matrix that stores the refined indices [n_queries, k]
 * @param[out] distances device matrix that stores the refined distances [n_queries, k]
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 */
void refine(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<const uint32_t, int64_t, raft::row_major> neighbor_candidates,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> indices,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

/**
 * @brief Refine nearest neighbor search.
 *
 * Refinement is an operation that follows an approximate NN search. The approximate search has
 * already selected n_candidates neighbor candidates for each query. We narrow it down to k
 * neighbors. For each query, we calculate the exact distance between the query and its
 * n_candidates neighbor candidate, and select the k nearest ones.
 *
 * The k nearest neighbors and distances are returned.
 *
 * Example usage
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // search m = 4 * k nearest neighbours for each of the N queries
 *   ivf_pq::search(handle, search_params, index, queries, neighbor_candidates,
 *                  out_dists_tmp);
 *   // refine it to the k nearest one
 *   refine(handle, dataset, queries, neighbor_candidates, out_indices, out_dists,
 *           index.metric());
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] dataset device matrix that stores the dataset [n_rows, dims]
 * @param[in] queries device matrix of the queries [n_queris, dims]
 * @param[in] neighbor_candidates indices of candidate vectors [n_queries, n_candidates], where
 *   n_candidates >= k
 * @param[out] indices device matrix that stores the refined indices [n_queries, k]
 * @param[out] distances device matrix that stores the refined distances [n_queries, k]
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 */
void refine(raft::resources const& handle,
            raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> indices,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

/**
 * @brief Refine nearest neighbor search.
 *
 * Refinement is an operation that follows an approximate NN search. The approximate search has
 * already selected n_candidates neighbor candidates for each query. We narrow it down to k
 * neighbors. For each query, we calculate the exact distance between the query and its
 * n_candidates neighbor candidate, and select the k nearest ones.
 *
 * The k nearest neighbors and distances are returned.
 *
 * Example usage
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // search m = 4 * k nearest neighbours for each of the N queries
 *   ivf_pq::search(handle, search_params, index, queries, neighbor_candidates,
 *                  out_dists_tmp);
 *   // refine it to the k nearest one
 *   refine(handle, dataset, queries, neighbor_candidates, out_indices, out_dists,
 *           index.metric());
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] dataset device matrix that stores the dataset [n_rows, dims]
 * @param[in] queries device matrix of the queries [n_queris, dims]
 * @param[in] neighbor_candidates indices of candidate vectors [n_queries, n_candidates], where
 *   n_candidates >= k
 * @param[out] indices device matrix that stores the refined indices [n_queries, k]
 * @param[out] distances device matrix that stores the refined distances [n_queries, k]
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 */
void refine(raft::resources const& handle,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> indices,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

/**
 * @brief Refine nearest neighbor search.
 *
 * Refinement is an operation that follows an approximate NN search. The approximate search has
 * already selected n_candidates neighbor candidates for each query. We narrow it down to k
 * neighbors. For each query, we calculate the exact distance between the query and its
 * n_candidates neighbor candidate, and select the k nearest ones.
 *
 * The k nearest neighbors and distances are returned.
 *
 * Example usage
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // search m = 4 * k nearest neighbours for each of the N queries
 *   ivf_pq::search(handle, search_params, index, queries, neighbor_candidates,
 *                  out_dists_tmp);
 *   // refine it to the k nearest one
 *   refine(handle, dataset, queries, neighbor_candidates, out_indices, out_dists,
 *           index.metric());
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] dataset device matrix that stores the dataset [n_rows, dims]
 * @param[in] queries device matrix of the queries [n_queris, dims]
 * @param[in] neighbor_candidates indices of candidate vectors [n_queries, n_candidates], where
 *   n_candidates >= k
 * @param[out] indices device matrix that stores the refined indices [n_queries, k]
 * @param[out] distances device matrix that stores the refined distances [n_queries, k]
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 */
void refine(raft::resources const& handle,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> indices,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

/**
 * @brief Refine nearest neighbor search.
 *
 * Refinement is an operation that follows an approximate NN search. The approximate search has
 * already selected n_candidates neighbor candidates for each query. We narrow it down to k
 * neighbors. For each query, we calculate the exact distance between the query and its
 * n_candidates neighbor candidate, and select the k nearest ones.
 *
 * The k nearest neighbors and distances are returned.
 *
 * Example usage
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // search m = 4 * k nearest neighbours for each of the N queries
 *   ivf_pq::search(handle, search_params, index, queries, neighbor_candidates,
 *                  out_dists_tmp);
 *   // refine it to the k nearest one
 *   refine(handle, dataset, queries, neighbor_candidates, out_indices, out_dists,
 *           index.metric());
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] dataset host matrix that stores the dataset [n_rows, dims]
 * @param[in] queries host matrix of the queries [n_queris, dims]
 * @param[in] neighbor_candidates host matrix with indices of candidate vectors [n_queries,
 *   n_candidates], where n_candidates >= k
 * @param[out] indices host matrix that stores the refined indices [n_queries, k]
 * @param[out] distances host matrix that stores the refined distances [n_queries, k]
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 */
void refine(raft::resources const& handle,
            raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
            raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::host_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
            raft::host_matrix_view<int64_t, int64_t, raft::row_major> indices,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances,
            cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

/**
 * @brief Refine nearest neighbor search.
 *
 * Refinement is an operation that follows an approximate NN search. The approximate search has
 * already selected n_candidates neighbor candidates for each query. We narrow it down to k
 * neighbors. For each query, we calculate the exact distance between the query and its
 * n_candidates neighbor candidate, and select the k nearest ones.
 *
 * The k nearest neighbors and distances are returned.
 *
 * Example usage
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // search m = 4 * k nearest neighbours for each of the N queries
 *   ivf_pq::search(handle, search_params, index, queries, neighbor_candidates,
 *                  out_dists_tmp);
 *   // refine it to the k nearest one
 *   refine(handle, dataset, queries, neighbor_candidates, out_indices, out_dists,
 *           index.metric());
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] dataset host matrix that stores the dataset [n_rows, dims]
 * @param[in] queries host matrix of the queries [n_queris, dims]
 * @param[in] neighbor_candidates host matrix with indices of candidate vectors [n_queries,
 *   n_candidates], where n_candidates >= k
 * @param[out] indices host matrix that stores the refined indices [n_queries, k]
 * @param[out] distances host matrix that stores the refined distances [n_queries, k]
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 */
void refine(raft::resources const& handle,
            raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
            raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::host_matrix_view<const uint32_t, int64_t, raft::row_major> neighbor_candidates,
            raft::host_matrix_view<uint32_t, int64_t, raft::row_major> indices,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances,
            cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

/**
 * @brief Refine nearest neighbor search.
 *
 * Refinement is an operation that follows an approximate NN search. The approximate search has
 * already selected n_candidates neighbor candidates for each query. We narrow it down to k
 * neighbors. For each query, we calculate the exact distance between the query and its
 * n_candidates neighbor candidate, and select the k nearest ones.
 *
 * The k nearest neighbors and distances are returned.
 *
 * Example usage
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // search m = 4 * k nearest neighbours for each of the N queries
 *   ivf_pq::search(handle, search_params, index, queries, neighbor_candidates,
 *                  out_dists_tmp);
 *   // refine it to the k nearest one
 *   refine(handle, dataset, queries, neighbor_candidates, out_indices, out_dists,
 *           index.metric());
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] dataset host matrix that stores the dataset [n_rows, dims]
 * @param[in] queries host matrix of the queries [n_queris, dims]
 * @param[in] neighbor_candidates host matrix with indices of candidate vectors [n_queries,
 *   n_candidates], where n_candidates >= k
 * @param[out] indices host matrix that stores the refined indices [n_queries, k]
 * @param[out] distances host matrix that stores the refined distances [n_queries, k]
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 */
void refine(raft::resources const& handle,
            raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
            raft::host_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::host_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
            raft::host_matrix_view<int64_t, int64_t, raft::row_major> indices,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances,
            cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

/**
 * @brief Refine nearest neighbor search.
 *
 * Refinement is an operation that follows an approximate NN search. The approximate search has
 * already selected n_candidates neighbor candidates for each query. We narrow it down to k
 * neighbors. For each query, we calculate the exact distance between the query and its
 * n_candidates neighbor candidate, and select the k nearest ones.
 *
 * The k nearest neighbors and distances are returned.
 *
 * Example usage
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // search m = 4 * k nearest neighbours for each of the N queries
 *   ivf_pq::search(handle, search_params, index, queries, neighbor_candidates,
 *                  out_dists_tmp);
 *   // refine it to the k nearest one
 *   refine(handle, dataset, queries, neighbor_candidates, out_indices, out_dists,
 *           index.metric());
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] dataset host matrix that stores the dataset [n_rows, dims]
 * @param[in] queries host matrix of the queries [n_queris, dims]
 * @param[in] neighbor_candidates host matrix with indices of candidate vectors [n_queries,
 *   n_candidates], where n_candidates >= k
 * @param[out] indices host matrix that stores the refined indices [n_queries, k]
 * @param[out] distances host matrix that stores the refined distances [n_queries, k]
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 */
void refine(raft::resources const& handle,
            raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
            raft::host_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::host_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
            raft::host_matrix_view<int64_t, int64_t, raft::row_major> indices,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances,
            cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

/**
 * @brief Refine nearest neighbor search.
 *
 * Refinement is an operation that follows an approximate NN search. The approximate search has
 * already selected n_candidates neighbor candidates for each query. We narrow it down to k
 * neighbors. For each query, we calculate the exact distance between the query and its
 * n_candidates neighbor candidate, and select the k nearest ones.
 *
 * The k nearest neighbors and distances are returned.
 *
 * Example usage
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // search m = 4 * k nearest neighbours for each of the N queries
 *   ivf_pq::search(handle, search_params, index, queries, neighbor_candidates,
 *                  out_dists_tmp);
 *   // refine it to the k nearest one
 *   refine(handle, dataset, queries, neighbor_candidates, out_indices, out_dists,
 *           index.metric());
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] dataset host matrix that stores the dataset [n_rows, dims]
 * @param[in] queries host matrix of the queries [n_queris, dims]
 * @param[in] neighbor_candidates host matrix with indices of candidate vectors [n_queries,
 *   n_candidates], where n_candidates >= k
 * @param[out] indices host matrix that stores the refined indices [n_queries, k]
 * @param[out] distances host matrix that stores the refined distances [n_queries, k]
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 */
void refine(raft::resources const& handle,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::host_matrix_view<const int64_t, int64_t, raft::row_major> neighbor_candidates,
            raft::host_matrix_view<int64_t, int64_t, raft::row_major> indices,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances,
            cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);

}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
