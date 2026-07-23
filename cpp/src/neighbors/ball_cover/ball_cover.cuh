/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"
#include "registers.hpp"
#include <cuvs/neighbors/ball_cover.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/init.cuh>
#include <raft/random/rng.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/tuple>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <limits.h>

#include <cstdint>

namespace cuvs::neighbors::ball_cover::detail {

/**
 * Given a set of points in row-major order which are to be
 * used as a set of index points, uniformly samples a subset
 * of points to be used as landmarks.
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param index
 */
template <typename value_idx, typename value_t>
void sample_landmarks(raft::resources const& handle,
                      cuvs::neighbors::ball_cover::index<value_idx, value_t>& index)
{
  rmm::device_uvector<value_idx> R_1nn_cols2(index.n_landmarks,
                                             raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<value_t> R_1nn_ones(index.m, raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<value_idx> R_indices(index.n_landmarks,
                                           raft::resource::get_cuda_stream(handle));

  raft::linalg::map_offset(handle, index.get_R_1nn_cols(), raft::identity_op{});

  raft::matrix::fill(
    handle, raft::make_device_vector_view(R_1nn_ones.data(), R_1nn_ones.size()), value_t(1.0));

  raft::matrix::fill(
    handle, raft::make_device_vector_view(R_indices.data(), R_indices.size()), value_idx(0));

  /**
   * 1. Randomly sample sqrt(n) points from X
   */
  raft::random::RngState rng_state(12345);
  raft::random::sampleWithoutReplacement(handle,
                                         rng_state,
                                         R_indices.data(),
                                         R_1nn_cols2.data(),
                                         index.get_R_1nn_cols().data_handle(),
                                         R_1nn_ones.data(),
                                         (value_idx)index.n_landmarks,
                                         (value_idx)index.m);

  auto x = index.get_X();
  auto r = index.get_R();

  raft::matrix::copy_rows<value_t, value_idx>(
    handle,
    raft::make_device_matrix_view<const value_t, value_idx>(
      x.data_handle(), x.extent(0), x.extent(1)),
    raft::make_device_matrix_view<value_t, value_idx>(r.data_handle(), r.extent(0), r.extent(1)),
    raft::make_device_vector_view(R_1nn_cols2.data(), index.n_landmarks));
}

/**
 * Constructs a 1-nn index mapping each landmark to their closest points.
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param R_knn_inds_ptr
 * @param R_knn_dists_ptr
 * @param k
 * @param index
 */
template <typename value_idx, typename value_t>
void construct_landmark_1nn(raft::resources const& handle,
                            const value_idx* R_knn_inds_ptr,
                            const value_t* R_knn_dists_ptr,
                            int64_t k,
                            cuvs::neighbors::ball_cover::index<value_idx, value_t>& index)
{
  auto R_1nn_inds = raft::make_device_vector<value_idx, value_idx>(handle, index.m);

  raft::matrix::fill(handle,
                     raft::make_device_vector_view<value_idx>(R_1nn_inds.data_handle(), index.m),
                     std::numeric_limits<value_idx>::max());

  raft::linalg::map_offset(handle, R_1nn_inds.view(), [R_knn_inds_ptr, k] __device__(value_idx i) {
    return R_knn_inds_ptr[i * k];
  });
  raft::linalg::map_offset(
    handle, index.get_R_1nn_dists(), [R_knn_dists_ptr, k] __device__(value_idx i) {
      return R_knn_dists_ptr[i * k];
    });

  auto keys = thrust::make_zip_iterator(
    cuda::std::make_tuple(R_1nn_inds.data_handle(), index.get_R_1nn_dists().data_handle()));

  // group neighborhoods for each reference landmark and sort each group by distance
  thrust::sort_by_key(raft::resource::get_thrust_policy(handle),
                      keys,
                      keys + index.m,
                      index.get_R_1nn_cols().data_handle(),
                      NNComp());

  // convert to CSR for fast lookup
  raft::sparse::convert::sorted_coo_to_csr(R_1nn_inds.data_handle(),
                                           index.m,
                                           index.get_R_indptr().data_handle(),
                                           index.n_landmarks + 1,
                                           raft::resource::get_cuda_stream(handle));

  // reorder X to allow aligned access
  raft::matrix::copy_rows<value_t, value_idx>(
    handle, index.get_X(), index.get_X_reordered(), index.get_R_1nn_cols());
}

/**
 * Computes the k closest landmarks to a set of query points.
 * @tparam value_idx
 * @tparam value_t
 * @tparam value_int
 * @param handle
 * @param index
 * @param query_pts
 * @param n_query_pts
 * @param k
 * @param R_knn_inds
 * @param R_knn_dists
 */
template <typename value_idx, typename value_t>
void k_closest_landmarks(raft::resources const& handle,
                         const cuvs::neighbors::ball_cover::index<value_idx, value_t>& index,
                         const value_t* query_pts,
                         int64_t n_query_pts,
                         int64_t k,
                         value_idx* R_knn_inds,
                         value_t* R_knn_dists)
{
  raft::device_matrix_view<const value_t, int64_t> inputs = index.get_R();
  auto bf_index_params   = cuvs::neighbors::brute_force::index_params();
  bf_index_params.metric = index.get_metric();
  auto bfknn             = cuvs::neighbors::brute_force::build(handle, bf_index_params, inputs);

  auto bf_search_params = cuvs::neighbors::brute_force::search_params();
  cuvs::neighbors::brute_force::search(
    handle,
    bf_search_params,
    bfknn,
    raft::make_device_matrix_view<const value_t, int64_t>(query_pts, n_query_pts, inputs.extent(1)),
    raft::make_device_matrix_view<value_idx, int64_t>(R_knn_inds, n_query_pts, k),
    raft::make_device_matrix_view<value_t, int64_t>(R_knn_dists, n_query_pts, k));
}

/**
 * Uses the sorted data points in the 1-nn landmark index to compute
 * an array of radii for each landmark.
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param index
 */
template <typename value_idx, typename value_t>
void compute_landmark_radii(raft::resources const& handle,
                            cuvs::neighbors::ball_cover::index<value_idx, value_t>& index)
{
  const value_idx* R_indptr_ptr  = index.get_R_indptr().data_handle();
  const value_t* R_1nn_dists_ptr = index.get_R_1nn_dists().data_handle();
  raft::linalg::map_offset(
    handle, index.get_R_radius(), [R_indptr_ptr, R_1nn_dists_ptr] __device__(value_idx input) {
      value_idx last_row_idx = R_indptr_ptr[input + 1] - 1;
      return R_1nn_dists_ptr[last_row_idx];
    });
}

/**
 * 4. Perform k-select over original KNN, using L_r to filter distances
 *
 * a. Map 1 row to each warp/block
 * b. Add closest k R points to heap
 * c. Iterate through batches of R, having each thread in the warp load a set
 * of distances y from R (only if d(q, r) < 3 * distance to closest r) and
 * marking the distance to be computed between x, y only
 * if knn[k].distance >= d(x_i, R_k) + d(R_k, y)
 */
template <typename value_idx, typename value_t>
void perform_rbc_query(raft::resources const& handle,
                       const cuvs::neighbors::ball_cover::index<value_idx, value_t>& index,
                       const value_t* query,
                       int64_t n_query_pts,
                       int64_t k,
                       const value_idx* R_knn_inds,
                       const value_t* R_knn_dists,
                       value_idx* inds,
                       value_t* dists,
                       float weight                = 1.0,
                       bool perform_post_filtering = true)
{
  // initialize output inds and dists
  raft::matrix::fill(handle,
                     raft::make_device_vector_view<value_idx>(inds, k * n_query_pts),
                     std::numeric_limits<value_idx>::max());
  raft::matrix::fill(handle,
                     raft::make_device_vector_view<value_t>(dists, k * n_query_pts),
                     std::numeric_limits<value_t>::max());

  // Compute nearest k for each neighborhood in each closest R
  rbc_low_dim_pass_one<value_idx, value_t>(
    handle, index, query, n_query_pts, k, R_knn_inds, R_knn_dists, inds, dists, weight, index.n);

  if (perform_post_filtering) {
    rbc_low_dim_pass_two<value_idx, value_t>(
      handle, index, query, n_query_pts, k, R_knn_inds, R_knn_dists, inds, dists, weight, index.n);
  }
}

/**
 * Perform eps-select
 *
 */
template <typename value_idx, typename value_t, typename dist_func>
void perform_rbc_eps_nn_query(raft::resources const& handle,
                              const cuvs::neighbors::ball_cover::index<value_idx, value_t>& index,
                              const value_t* query,
                              int64_t n_query_pts,
                              value_t eps,
                              const value_t* landmarks,
                              dist_func dfunc,
                              bool* adj,
                              value_idx* vd)
{
  // initialize output
  raft::matrix::fill(
    handle, raft::make_device_vector_view(adj, index.m * n_query_pts), bool(false));

  raft::resource::sync_stream(handle);

  rbc_eps_pass<value_idx, value_t>(
    handle, index, query, n_query_pts, eps, landmarks, dfunc, adj, vd);

  raft::resource::sync_stream(handle);
}

template <typename value_idx, typename value_t, typename dist_func>
void perform_rbc_eps_nn_query(raft::resources const& handle,
                              const cuvs::neighbors::ball_cover::index<value_idx, value_t>& index,
                              const value_t* query,
                              int64_t n_query_pts,
                              value_t eps,
                              int64_t* max_k,
                              const value_t* landmarks,
                              dist_func dfunc,
                              value_idx* adj_ia,
                              value_idx* adj_ja,
                              value_idx* vd,
                              value_t* dists)
{
  rbc_eps_pass<value_idx, value_t>(
    handle, index, query, n_query_pts, eps, max_k, landmarks, dfunc, adj_ia, adj_ja, vd, dists);

  raft::resource::sync_stream(handle);
}

/**
 * Similar to a ball tree, the random ball cover algorithm
 * uses the triangle inequality to prune distance computations
 * in any metric space with a guarantee of sqrt(n) * c^{3/2}
 * where `c` is an expansion constant based on the distance
 * metric.
 *
 * This function variant performs an all nearest neighbors
 * query which is useful for algorithms that need to perform
 * A * A.T.
 */
template <typename value_idx = std::int64_t, typename value_t>
void rbc_build_index(raft::resources const& handle,
                     cuvs::neighbors::ball_cover::index<value_idx, value_t>& index)
{
  ASSERT(!index.is_index_trained(), "index cannot be previously trained");

  rmm::device_uvector<value_idx> R_knn_inds(index.m, raft::resource::get_cuda_stream(handle));

  // Initialize the uvectors
  raft::matrix::fill(handle,
                     raft::make_device_vector_view<value_idx>(R_knn_inds.data(), R_knn_inds.size()),
                     std::numeric_limits<value_idx>::max());
  raft::matrix::fill(handle,
                     raft::make_device_vector_view<value_t>(
                       index.get_R_closest_landmark_dists().data_handle(), index.m),
                     std::numeric_limits<value_t>::max());

  /**
   * 1. Randomly sample sqrt(n) points from X
   */
  sample_landmarks<value_idx, value_t>(handle, index);

  /**
   * 2. Perform knn = bfknn(X, R, k)
   */
  int64_t k = 1;
  k_closest_landmarks(handle,
                      index,
                      index.get_X().data_handle(),
                      index.m,
                      k,
                      R_knn_inds.data(),
                      index.get_R_closest_landmark_dists().data_handle());

  /**
   * 3. Create L_r = knn[:,0].T (CSR)
   *
   * Slice closest neighboring R
   * Secondary sort by (R_knn_inds, R_knn_dists)
   */
  construct_landmark_1nn(
    handle, R_knn_inds.data(), index.get_R_closest_landmark_dists().data_handle(), k, index);

  /**
   * Compute radius of each R for filtering: p(q, r) <= p(q, q_r) + radius(r)
   * (need to take the
   */
  compute_landmark_radii(handle, index);
}

/**
 * Performs an all neighbors knn query (e.g. index == query)
 */
template <typename value_idx = std::int64_t, typename value_t>
void rbc_all_knn_query(raft::resources const& handle,
                       cuvs::neighbors::ball_cover::index<value_idx, value_t>& index,
                       int64_t k,
                       value_idx* inds,
                       value_t* dists,
                       // approximate nn options
                       bool perform_post_filtering = true,
                       float weight                = 1.0)
{
  ASSERT(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  ASSERT(index.n_landmarks >= k, "number of landmark samples must be >= k");
  ASSERT(!index.is_index_trained(), "index cannot be previously trained");

  rmm::device_uvector<value_idx> R_knn_inds(k * index.m, raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<value_t> R_knn_dists(k * index.m, raft::resource::get_cuda_stream(handle));

  // Initialize the uvectors
  raft::matrix::fill(handle,
                     raft::make_device_vector_view<value_idx>(R_knn_inds.data(), R_knn_inds.size()),
                     std::numeric_limits<value_idx>::max());
  raft::matrix::fill(handle,
                     raft::make_device_vector_view<value_t>(R_knn_dists.data(), R_knn_dists.size()),
                     std::numeric_limits<value_t>::max());

  raft::matrix::fill(handle,
                     raft::make_device_vector_view<value_idx>(inds, k * index.m),
                     std::numeric_limits<value_idx>::max());
  raft::matrix::fill(handle,
                     raft::make_device_vector_view<value_t>(dists, k * index.m),
                     std::numeric_limits<value_t>::max());

  sample_landmarks<value_idx, value_t>(handle, index);

  k_closest_landmarks(
    handle, index, index.get_X().data_handle(), index.m, k, R_knn_inds.data(), R_knn_dists.data());

  construct_landmark_1nn(handle, R_knn_inds.data(), R_knn_dists.data(), k, index);

  compute_landmark_radii(handle, index);

  perform_rbc_query(handle,
                    index,
                    index.get_X().data_handle(),
                    index.m,
                    k,
                    R_knn_inds.data(),
                    R_knn_dists.data(),
                    inds,
                    dists,
                    weight,
                    perform_post_filtering);
}

/**
 * Performs a knn query against an index. This assumes the index has
 * already been built.
 */
template <typename value_idx = std::int64_t, typename value_t>
void rbc_knn_query(raft::resources const& handle,
                   const cuvs::neighbors::ball_cover::index<value_idx, value_t>& index,
                   int64_t k,
                   const value_t* query,
                   int64_t n_query_pts,
                   value_idx* inds,
                   value_t* dists,
                   // approximate nn options
                   bool perform_post_filtering = true,
                   float weight                = 1.0)
{
  ASSERT(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  ASSERT(index.n_landmarks >= k, "number of landmark samples must be >= k");
  ASSERT(index.is_index_trained(), "index must be previously trained");

  rmm::device_uvector<value_idx> R_knn_inds(k * n_query_pts,
                                            raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<value_t> R_knn_dists(k * n_query_pts,
                                           raft::resource::get_cuda_stream(handle));

  // Initialize the uvectors
  raft::matrix::fill(handle,
                     raft::make_device_vector_view<value_idx>(R_knn_inds.data(), R_knn_inds.size()),
                     std::numeric_limits<value_idx>::max());
  raft::matrix::fill(handle,
                     raft::make_device_vector_view<value_t>(R_knn_dists.data(), R_knn_dists.size()),
                     std::numeric_limits<value_t>::max());

  raft::matrix::fill(handle,
                     raft::make_device_vector_view<value_idx>(inds, k * n_query_pts),
                     std::numeric_limits<value_idx>::max());
  raft::matrix::fill(handle,
                     raft::make_device_vector_view<value_t>(dists, k * n_query_pts),
                     std::numeric_limits<value_t>::max());

  k_closest_landmarks(handle, index, query, n_query_pts, k, R_knn_inds.data(), R_knn_dists.data());

  perform_rbc_query(handle,
                    index,
                    query,
                    n_query_pts,
                    k,
                    R_knn_inds.data(),
                    R_knn_dists.data(),
                    inds,
                    dists,
                    weight,
                    perform_post_filtering);
}

template <typename value_idx, typename value_t>
void compute_landmark_dists(raft::resources const& handle,
                            const cuvs::neighbors::ball_cover::index<value_idx, value_t>& index,
                            const value_t* query_pts,
                            int64_t n_query_pts,
                            value_t* R_dists)
{
  // compute distances for all queries against all landmarks
  // index.get_R() -- landmark points in row order (index.n_landmarks x index.k)
  // query_pts     -- query points in row order (n_query_pts x index.k)
  RAFT_EXPECTS(std::max<size_t>(index.n_landmarks, n_query_pts) * index.n <
                 static_cast<size_t>(std::numeric_limits<int>::max()),
               "Too large input for pairwise_distance with `int` index.");
  RAFT_EXPECTS(n_query_pts * static_cast<size_t>(index.n_landmarks) <
                 static_cast<size_t>(std::numeric_limits<int>::max()),
               "Too large input for pairwise_distance with `int` index.");
  cuvs::distance::pairwise_distance(handle,
                                    query_pts,
                                    index.get_R().data_handle(),
                                    R_dists,
                                    n_query_pts,
                                    index.n_landmarks,
                                    index.n,
                                    index.get_metric());
}

/**
 * Performs a knn query against an index. This assumes the index has
 * already been built.
 * Modified version that takes an eps as threshold and outputs to a dense adj matrix (row-major)
 * we are assuming that there are sufficiently many landmarks
 */
template <typename value_idx = std::int64_t, typename value_t, typename distance_func>
void rbc_eps_nn_query(raft::resources const& handle,
                      const cuvs::neighbors::ball_cover::index<value_idx, value_t>& index,
                      const value_t eps,
                      const value_t* query,
                      int64_t n_query_pts,
                      bool* adj,
                      value_idx* vd,
                      distance_func dfunc)
{
  ASSERT(index.is_index_trained(), "index must be previously trained");

  // query all points and write to adj
  perform_rbc_eps_nn_query(
    handle, index, query, n_query_pts, eps, index.get_R().data_handle(), dfunc, adj, vd);
}

template <typename value_idx = std::int64_t, typename value_t, typename distance_func>
void rbc_eps_nn_query(raft::resources const& handle,
                      const cuvs::neighbors::ball_cover::index<value_idx, value_t>& index,
                      const value_t eps,
                      int64_t* max_k,
                      const value_t* query,
                      int64_t n_query_pts,
                      value_idx* adj_ia,
                      value_idx* adj_ja,
                      value_idx* vd,
                      value_t* dists,
                      distance_func dfunc)
{
  ASSERT(index.is_index_trained(), "index must be previously trained");

  // query all points and write to adj
  perform_rbc_eps_nn_query(handle,
                           index,
                           query,
                           n_query_pts,
                           eps,
                           max_k,
                           index.get_R().data_handle(),
                           dfunc,
                           adj_ia,
                           adj_ja,
                           vd,
                           dists);
}

};  // namespace cuvs::neighbors::ball_cover::detail
