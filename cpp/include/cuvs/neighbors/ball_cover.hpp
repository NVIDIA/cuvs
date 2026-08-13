/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <cuvs/core/export.hpp>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace ball_cover {

/**
 * @ingroup random_ball_cover
 * @{
 */

/**
 * Stores raw index data points, sampled landmarks, the 1-nns of index points
 * to their closest landmarks, and the ball radii of each landmark. This
 * class is intended to be constructed once and reused across subsequent
 * queries.
 * @tparam int64_t
 * @tparam float
 * @tparam int
 */
template <typename idx_t, typename value_t>
struct index : cuvs::neighbors::index {
 public:
  explicit index(raft::resources const& handle_,
                 raft::device_matrix_view<const float, int64_t, raft::row_major> X_,
                 cuvs::distance::DistanceType metric_)
    : handle(handle_),
      X(X_),
      m(X_.extent(0)),
      n(X_.extent(1)),
      metric(metric_),
      /**
       * the sqrt() here makes the sqrt(m)^2 a linear-time lower bound
       *
       * Total memory footprint of index: (2 * sqrt(m)) + (n * sqrt(m)) + (2 * m)
       */
      n_landmarks(raft::sqrt(X_.extent(0))),
      R_indptr(raft::make_device_vector<idx_t, int64_t>(handle, raft::sqrt(X_.extent(0)) + 1)),
      R_1nn_cols(raft::make_device_vector<idx_t, int64_t>(handle, X_.extent(0))),
      R_1nn_dists(raft::make_device_vector<float, int64_t>(handle, X_.extent(0))),
      R_closest_landmark_dists(raft::make_device_vector<float, int64_t>(handle, X_.extent(0))),
      R(raft::make_device_matrix<float, int64_t>(handle, raft::sqrt(X_.extent(0)), X_.extent(1))),
      X_reordered(raft::make_device_matrix<float, int64_t>(handle, X_.extent(0), X_.extent(1))),
      R_radius(raft::make_device_vector<float, int64_t>(handle, raft::sqrt(X_.extent(0)))),
      index_trained(false)
  {
  }

  auto get_R_indptr() const -> raft::device_vector_view<const idx_t, int64_t>
  {
    return R_indptr.view();
  }
  auto get_R_1nn_cols() const -> raft::device_vector_view<const idx_t, int64_t>
  {
    return R_1nn_cols.view();
  }
  auto get_R_1nn_dists() const -> raft::device_vector_view<const float, int64_t>
  {
    return R_1nn_dists.view();
  }
  auto get_R_radius() const -> raft::device_vector_view<const float, int64_t>
  {
    return R_radius.view();
  }
  auto get_R() const -> raft::device_matrix_view<const float, int64_t, raft::row_major>
  {
    return R.view();
  }
  auto get_R_closest_landmark_dists() const -> raft::device_vector_view<const float, int64_t>
  {
    return R_closest_landmark_dists.view();
  }
  auto get_X_reordered() const -> raft::device_matrix_view<const float, int64_t, raft::row_major>
  {
    return X_reordered.view();
  }

  raft::device_vector_view<idx_t, int64_t> get_R_indptr() { return R_indptr.view(); }
  raft::device_vector_view<idx_t, int64_t> get_R_1nn_cols() { return R_1nn_cols.view(); }
  raft::device_vector_view<float, int64_t> get_R_1nn_dists() { return R_1nn_dists.view(); }
  raft::device_vector_view<float, int64_t> get_R_radius() { return R_radius.view(); }
  raft::device_matrix_view<float, int64_t, raft::row_major> get_R() { return R.view(); }
  raft::device_vector_view<float, int64_t> get_R_closest_landmark_dists()
  {
    return R_closest_landmark_dists.view();
  }
  raft::device_matrix_view<float, int64_t, raft::row_major> get_X_reordered()
  {
    return X_reordered.view();
  }
  raft::device_matrix_view<const float, int64_t, raft::row_major> get_X() const { return X; }

  cuvs::distance::DistanceType get_metric() const { return metric; }

  int get_n_landmarks() const { return n_landmarks; }
  bool is_index_trained() const { return index_trained; };

  // This should only be set by internal functions
  void set_index_trained() { index_trained = true; }

  raft::resources const& handle;

  int64_t m;
  int64_t n;
  int64_t n_landmarks;

  raft::device_matrix_view<const float, idx_t, raft::row_major> X;

  cuvs::distance::DistanceType metric;

 private:
  // CSR storing the neighborhoods for each data point
  raft::device_vector<idx_t, int64_t> R_indptr;
  raft::device_vector<idx_t, int64_t> R_1nn_cols;
  raft::device_vector<float, int64_t> R_1nn_dists;
  raft::device_vector<float, int64_t> R_closest_landmark_dists;

  raft::device_vector<float, int64_t> R_radius;

  raft::device_matrix<float, int64_t, raft::row_major> R;
  raft::device_matrix<float, int64_t, raft::row_major> X_reordered;

 protected:
  bool index_trained;
};

/** @} */

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
 *  #include <cuvs/neighbors/ball_cover.hpp>
 *  #include <cuvs/distance/distance.hpp>
 *  using namespace cuvs::neighbors;
 *
 *  raft::resources handle;
 *  ...
 *  auto metric = cuvs::distance::DistanceType::L2Expanded;
 *  ball_cover::index index(handle, X, metric);
 *  ball_cover::build_index(handle, index);
 * @endcode
 *
 * @param[in] handle library resource management handle
 * @param[inout] index an empty (and not previous built) instance of
 * cuvs::neighbors::ball_cover::index
 */
void build(raft::resources const& handle, index<int64_t, float>& index);

/** @} */  // end group random_ball_cover

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
 *  ball_cover::index index(handle, X, metric);
 *
 *  // Perform all neighbors knn query
 *  ball_cover::all_knn_query(handle, index, inds, dists, k);
 * @endcode
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
void all_knn_query(raft::resources const& handle,
                   index<int64_t, float>& index,
                   raft::device_matrix_view<int64_t, int64_t, raft::row_major> inds,
                   raft::device_matrix_view<float, int64_t, raft::row_major> dists,
                   bool perform_post_filtering = true,
                   float weight                = 1.0);

/** @} */

/**
 * @brief Computes epsilon neighborhood for the L2 distance metric using rbc
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
void eps_nn(raft::resources const& handle,
            const index<int64_t, float>& index,
            raft::device_matrix_view<bool, int64_t, raft::row_major> adj,
            raft::device_vector_view<int64_t, int64_t> vd,
            raft::device_matrix_view<const float, int64_t, raft::row_major> query,
            float eps);
/**
 * @brief Computes epsilon neighborhood for the L2 distance metric using rbc
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
 */
void eps_nn(raft::resources const& handle,
            const index<int64_t, float>& index,
            raft::device_vector_view<int64_t, int64_t> adj_ia,
            raft::device_vector_view<int64_t, int64_t> adj_ja,
            raft::device_vector_view<int64_t, int64_t> vd,
            raft::device_matrix_view<const float, int64_t, raft::row_major> query,
            float eps,
            std::optional<raft::host_scalar_view<int64_t, int64_t>> max_k = std::nullopt);

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
 *  #include <cuvs/neighbors/ball_cover.hpp>
 *  #include <cuvs/distance/distance.hpp>
 *  using namespace cuvs::neighbors;
 *
 *  raft::resources handle;
 *  ...
 *  auto metric = cuvs::distance::DistanceType::L2Expanded;
 *
 *  // Build a ball cover index
 *  ball_cover::index index(handle, X, metric);
 *  ball_cover::build_index(handle, index);
 *
 *  // Perform all neighbors knn query
 *  ball_cover::knn_query(handle, index, inds, dists, k);
 * @endcode
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
void knn_query(raft::resources const& handle,
               const index<int64_t, float>& index,
               raft::device_matrix_view<const float, int64_t, raft::row_major> query,
               raft::device_matrix_view<int64_t, int64_t, raft::row_major> inds,
               raft::device_matrix_view<float, int64_t, raft::row_major> dists,
               bool perform_post_filtering = true,
               float weight                = 1.0);

/** @} */

}  // namespace ball_cover
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
