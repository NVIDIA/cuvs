/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "../../core/spectral_launchers.hpp"
#include "./knn_brute_force.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

namespace cuvs::neighbors::detail::reachability {

/**
 * Extract core distances from KNN graph. This is essentially
 * performing a knn_dists[:,min_pts]
 * @tparam value_idx data type for integrals
 * @tparam value_t data type for distance
 * @tparam tpb block size for kernel
 * @param[in] knn_dists knn distance array (size n * k)
 * @param[in] min_samples this neighbor will be selected for core distances
 * @param[in] n_neighbors the number of neighbors of each point in the knn graph
 * @param[in] n number of samples
 * @param[out] out output array (size n)
 * @param[in] stream stream for which to order cuda operations
 */
template <typename value_idx, typename value_t, int tpb = 256>
void core_distances(raft::resources const& handle,
                    value_t* knn_dists,
                    int min_samples,
                    int n_neighbors,
                    size_t n,
                    value_t* out)
{
  ASSERT(n_neighbors >= min_samples,
         "the size of the neighborhood should be greater than or equal to min_samples");

  auto out_view = raft::make_device_vector_view<value_t, value_idx>(out, n);

  raft::linalg::map_offset(handle, out_view, [=] __device__(value_idx row) {
    return knn_dists[row * n_neighbors + (min_samples - 1)];
  });
}

/**
 * Wraps the brute force knn API, to be used for both training and prediction
 * @tparam value_idx data type for integrals
 * @tparam value_t data type for distance
 * @param[in] handle raft handle for resource reuse
 * @param[in] X input data points (size m * n)
 * @param[out] inds nearest neighbor indices (size n_search_items * k)
 * @param[out] dists nearest neighbor distances (size n_search_items * k)
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] search_items array of items to search of dimensionality D (size n_search_items * n)
 * @param[in] n_search_items number of rows in search_items
 * @param[in] k number of nearest neighbors
 * @param[in] metric distance metric to use
 */
template <typename value_idx, typename value_t>
void compute_knn(const raft::resources& handle,
                 const value_t* X,
                 value_idx* inds,
                 value_t* dists,
                 size_t m,
                 size_t n,
                 const value_t* search_items,
                 size_t n_search_items,
                 int k,
                 cuvs::distance::DistanceType metric)
{
  // perform knn
  tiled_brute_force_knn(handle, X, search_items, m, n_search_items, n, k, dists, inds, metric);
}

/*
  @brief Internal function for CPU->GPU interop
         to compute core_dists
*/
template <typename value_idx, typename value_t>
void _compute_core_dists(const raft::resources& handle,
                         const value_t* X,
                         value_t* core_dists,
                         size_t m,
                         size_t n,
                         cuvs::distance::DistanceType metric,
                         int min_samples)
{
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2SqrtExpanded,
               "Currently only L2 expanded distance is supported");

  auto stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<value_idx> inds(min_samples * m, stream);
  rmm::device_uvector<value_t> dists(min_samples * m, stream);

  // perform knn
  compute_knn(handle, X, inds.data(), dists.data(), m, n, X, m, min_samples, metric);

  // Slice core distances (distances to kth nearest neighbor)
  core_distances<value_idx>(handle, dists.data(), min_samples, min_samples, m, core_dists);
}

//  Functor to post-process distances into reachability space
template <typename value_idx, typename value_t>
struct ReachabilityPostProcess {
  DI value_t operator()(value_t value, value_idx row, value_idx col) const
  {
    return max(core_dists[col], max(core_dists[row], alpha * value));
  }

  const value_t* core_dists;
  value_t alpha;
  size_t n;  // size of core_dists array
};

/**
 * Given core distances, Fuses computations of L2 distances between all
 * points, projection into mutual reachability space, and k-selection.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[out] out_inds  output indices array (size m * k)
 * @param[out] out_dists output distances array (size m * k)
 * @param[in] X input data points (size m * n)
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] k neighborhood size (includes self-loop)
 * @param[in] core_dists array of core distances (size m)
 */
template <typename value_idx, typename value_t>
void mutual_reachability_knn_l2(const raft::resources& handle,
                                value_idx* out_inds,
                                value_t* out_dists,
                                const value_t* X,
                                size_t m,
                                size_t n,
                                int k,
                                value_t* core_dists,
                                value_t alpha)
{
  // Create a functor to postprocess distances into mutual reachability space
  // Note that we can't use a lambda for this here, since we get errors like:
  // `A type local to a function cannot be used in the template argument of the
  // enclosing parent function (and any parent classes) of an extended __device__
  // or __host__ __device__ lambda`
  auto epilogue = ReachabilityPostProcess<value_idx, value_t>{core_dists, alpha, m};

  cuvs::neighbors::detail::
    tiled_brute_force_knn<value_t, value_idx, value_t, ReachabilityPostProcess<value_idx, value_t>>(
      handle,
      X,
      X,
      m,
      m,
      n,
      k,
      out_dists,
      out_inds,
      cuvs::distance::DistanceType::L2SqrtExpanded,
      2.0,
      0,
      0,
      nullptr,
      nullptr,
      nullptr,
      epilogue);
}

template <typename value_idx, typename value_t, typename nnz_t>
void mutual_reachability_graph(const raft::resources& handle,
                               const value_t* X,
                               value_idx m,
                               value_idx n,
                               cuvs::distance::DistanceType metric,
                               int min_samples,
                               value_t alpha,
                               value_idx* indptr,
                               value_t* core_dists,
                               raft::sparse::COO<value_t, value_idx, nnz_t>& out)
{
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2SqrtExpanded,
               "Currently only L2 expanded distance is supported");

  auto stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<value_idx> coo_rows(min_samples * m, stream);
  rmm::device_uvector<value_idx> inds(min_samples * m, stream);
  rmm::device_uvector<value_t> dists(min_samples * m, stream);

  // perform knn
  compute_knn(handle, X, inds.data(), dists.data(), m, n, X, m, min_samples, metric);

  // Slice core distances (distances to kth nearest neighbor)
  core_distances<value_idx>(handle, dists.data(), min_samples, min_samples, m, core_dists);

  /**
   * Compute L2 norm
   */
  mutual_reachability_knn_l2(
    handle, inds.data(), dists.data(), X, m, n, min_samples, core_dists, (value_t)1.0 / alpha);

  // self-loops get max distance
  auto coo_rows_view =
    raft::make_device_vector_view<value_idx, value_idx>(coo_rows.data(), m * min_samples);
  raft::linalg::map_offset(
    handle, coo_rows_view, [min_samples] __device__(value_idx c) -> value_idx {
      return c / min_samples;
    });

  cuvs::detail::symmetrize(handle,
                           coo_rows.data(),
                           inds.data(),
                           dists.data(),
                           m,
                           m,
                           static_cast<nnz_t>(min_samples * m),
                           out);

  raft::sparse::convert::sorted_coo_to_csr(out.rows(), out.nnz, indptr, m + 1, stream);

  // self-loops get max distance
  auto rows_view     = raft::make_device_vector_view<const value_idx, nnz_t>(out.rows(), out.nnz);
  auto cols_view     = raft::make_device_vector_view<const value_idx, nnz_t>(out.cols(), out.nnz);
  auto vals_in_view  = raft::make_device_vector_view<const value_t, nnz_t>(out.vals(), out.nnz);
  auto vals_out_view = raft::make_device_vector_view<value_t, nnz_t>(out.vals(), out.nnz);

  raft::linalg::map(
    handle,
    vals_out_view,
    [=] __device__(const value_idx row, const value_idx col, const value_t val) {
      return row == col ? std::numeric_limits<value_t>::max() : val;
    },
    rows_view,
    cols_view,
    vals_in_view);
}

}  // namespace cuvs::neighbors::detail::reachability
