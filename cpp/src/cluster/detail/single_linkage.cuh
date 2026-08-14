/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../neighbors/detail/reachability.cuh"
#include "agglomerative.cuh"
#include "connectivities.cuh"
#include "mst.cuh"
#include "raft/core/device_mdspan.hpp"
#include <cuvs/cluster/agglomerative.hpp>
#include <cuvs/neighbors/all_neighbors.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

namespace cuvs::cluster::agglomerative::detail {

/**
 * Constructs a linkage by computing the minimum spanning tree and dendrogram in the Mutual
 * Reachability space. Returns mst edges sorted by weight and the dendrogram.
 * @tparam value_t
 * @tparam value_idx
 * @tparam nnz_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] X data points (size m * n)
 * @param[in] metric distance metric to use
 * @param[in] min_samples this neighborhood will be selected for core distances
 * @param[in] alpha weight applied when internal distance is chosen for mutual reachability (value
 * of 1.0 disables the weighting)
 * @param[out] core_dists core distances (size m)
 * @param[out] out_mst output MST sorted by edge weights (size m - 1)
 * @param[out] out_dendrogram output dendrogram
 * @param[out] out_distances distances of output
 * @param[out] out_sizes cluster sizes of output
 */
template <typename value_t   = float,
          typename value_idx = int,
          typename nnz_t     = size_t,
          typename Accessor  = raft::device_accessor<cuda::std::default_accessor<value_t>>>
void build_mr_linkage(
  raft::resources const& handle,
  raft::mdspan<const value_t, raft::matrix_extent<value_idx>, raft::row_major, Accessor> X,
  value_idx min_samples,
  float alpha,
  cuvs::distance::DistanceType metric,
  raft::device_vector_view<value_t, value_idx> core_dists,
  raft::device_coo_matrix_view<value_t, value_idx, value_idx, nnz_t> out_mst,
  raft::device_matrix_view<value_idx, value_idx> out_dendrogram,
  raft::device_vector_view<value_t, value_idx> out_distances,
  raft::device_vector_view<value_idx, value_idx> out_sizes,
  cuvs::neighbors::all_neighbors::all_neighbors_params all_neighbors_p)
{
  size_t m    = X.extent(0);
  size_t n    = X.extent(1);
  auto stream = raft::resource::get_cuda_stream(handle);

  {  // scope to drop mr_coo and mr_indptr early
    std::optional<raft::sparse::COO<value_t, value_idx, nnz_t>> mr_coo;

    {  // scope to drop inds and dists matrices early
      auto inds  = raft::make_device_matrix<value_idx, value_idx>(handle, m, min_samples);
      auto dists = raft::make_device_matrix<value_t, value_idx>(handle, m, min_samples);

      if (all_neighbors_p.metric != metric) {
        RAFT_LOG_WARN("Setting all neighbors metric to given metric for build_mr_linkage");
        all_neighbors_p.metric = metric;
      }
      cuvs::neighbors::all_neighbors::build(
        handle, all_neighbors_p, X, inds.view(), dists.view(), core_dists, alpha);

      // allocate memory after all neighbors build
      mr_coo.emplace(stream, min_samples * m * 2);
      // self-loops get max distance
      auto coo_rows = raft::make_device_vector<value_idx, value_idx>(handle, min_samples * m);
      raft::linalg::map_offset(handle, coo_rows.view(), raft::div_const_op<value_idx>(min_samples));

      raft::sparse::linalg::symmetrize(handle,
                                       coo_rows.data_handle(),
                                       inds.data_handle(),
                                       dists.data_handle(),
                                       static_cast<value_idx>(m),
                                       static_cast<value_idx>(m),
                                       static_cast<nnz_t>(min_samples * m),
                                       mr_coo.value());
    }  // scope to drop inds and dists matrices early
    auto mr_indptr = raft::make_device_vector<value_idx, value_idx>(handle, m + 1);
    raft::sparse::convert::sorted_coo_to_csr(
      mr_coo.value().rows(), mr_coo.value().nnz, mr_indptr.data_handle(), m + 1, stream);

    auto rows_view    = raft::make_device_vector_view<const value_idx, nnz_t>(mr_coo.value().rows(),
                                                                           mr_coo.value().nnz);
    auto cols_view    = raft::make_device_vector_view<const value_idx, nnz_t>(mr_coo.value().cols(),
                                                                           mr_coo.value().nnz);
    auto vals_in_view = raft::make_device_vector_view<const value_t, nnz_t>(mr_coo.value().vals(),
                                                                            mr_coo.value().nnz);
    auto vals_out_view =
      raft::make_device_vector_view<value_t, nnz_t>(mr_coo.value().vals(), mr_coo.value().nnz);

    raft::linalg::map(
      handle,
      vals_out_view,
      [=] __device__(const value_idx row, const value_idx col, const value_t val) {
        return row == col ? std::numeric_limits<value_t>::max() : val;
      },
      rows_view,
      cols_view,
      vals_in_view);

    rmm::device_uvector<value_idx> color(m, raft::resource::get_cuda_stream(handle));
    cuvs::sparse::neighbors::MutualReachabilityFixConnectivitiesRedOp<value_idx, value_t>
      reduction_op(core_dists.data_handle(), m);

    size_t nnz = m * min_samples;

    detail::build_sorted_mst<value_idx, value_t>(handle,
                                                 X.data_handle(),
                                                 mr_indptr.data_handle(),
                                                 mr_coo.value().cols(),
                                                 mr_coo.value().vals(),
                                                 m,
                                                 n,
                                                 out_mst.structure_view().get_rows().data(),
                                                 out_mst.structure_view().get_cols().data(),
                                                 out_mst.get_elements().data(),
                                                 color.data(),
                                                 mr_coo.value().nnz,
                                                 reduction_op,
                                                 metric,
                                                 10);
  }  // scope to drop mr_coo and mr_indptr early
  /**
   * Perform hierarchical labeling
   */
  size_t n_edges = m - 1;

  detail::build_dendrogram_host<value_idx, value_t>(handle,
                                                    out_mst.structure_view().get_rows().data(),
                                                    out_mst.structure_view().get_cols().data(),
                                                    out_mst.get_elements().data(),
                                                    n_edges,
                                                    out_dendrogram.data_handle(),
                                                    out_distances.data_handle(),
                                                    out_sizes.data_handle());
}

static const size_t EMPTY = 0;

/**
 * Constructs a linkage by computing the minimum spanning tree and dendrogram in the Mutual
 * Reachability space. Returns mst edges sorted by weight and the dendrogram.
 * @tparam value_t
 * @tparam value_idx
 * @tparam nnz_t
 * @tparam dist_type method to use for constructing connectivities graph
 * @param[in] handle raft handle for resource reuse
 * @param[in] X data points (size m * n)
 * @param[in] c a constant used when constructing linkage from knn graph. Allows the indirect
 * control of k. The algorithm will set `k = log(n) + c`
 * @param[in] metric distance metric to use
 * @param[out] out_mst output MST sorted by edge weights (size m - 1)
 * @param[out] out_dendrogram output dendrogram
 * @param[out] out_distances distances of output
 * @param[out] out_sizes cluster sizes of output
 */
template <typename value_t, typename value_idx, typename nnz_t, Linkage dist_type>
void build_dist_linkage(raft::resources const& handle,
                        raft::device_matrix_view<const value_t, value_idx, raft::row_major> X,
                        int c,
                        cuvs::distance::DistanceType metric,
                        raft::device_coo_matrix_view<value_t, value_idx, value_idx, nnz_t> out_mst,
                        raft::device_matrix_view<value_idx, value_idx> out_dendrogram,
                        raft::device_vector_view<value_t, value_idx> out_distances,
                        raft::device_vector_view<value_idx, value_idx> out_sizes)
{
  size_t m    = X.extent(0);
  size_t n    = X.extent(1);
  auto stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<value_idx> indptr(EMPTY, stream);
  rmm::device_uvector<value_idx> indices(EMPTY, stream);
  rmm::device_uvector<value_t> pw_dists(EMPTY, stream);

  /**
   * 1. Construct distance graph
   */
  detail::get_distance_graph<value_idx, value_t, dist_type>(handle,
                                                            X.data_handle(),
                                                            static_cast<value_idx>(m),
                                                            static_cast<value_idx>(n),
                                                            metric,
                                                            indptr,
                                                            indices,
                                                            pw_dists,
                                                            c);

  /**
   * 2. Construct MST, sorted by weights
   */
  rmm::device_uvector<value_idx> color(m, stream);
  cuvs::sparse::neighbors::FixConnectivitiesRedOp<value_idx, value_t> op(m);

  size_t n_edges = m - 1;

  detail::build_sorted_mst<value_idx, value_t>(handle,
                                               X.data_handle(),
                                               indptr.data(),
                                               indices.data(),
                                               pw_dists.data(),
                                               m,
                                               n,
                                               out_mst.structure_view().get_rows().data(),
                                               out_mst.structure_view().get_cols().data(),
                                               out_mst.get_elements().data(),
                                               color.data(),
                                               indices.size(),
                                               op,
                                               metric);
  pw_dists.release();

  /**
   * Perform hierarchical labeling
   */
  detail::build_dendrogram_host<value_idx, value_t>(handle,
                                                    out_mst.structure_view().get_rows().data(),
                                                    out_mst.structure_view().get_cols().data(),
                                                    out_mst.get_elements().data(),
                                                    n_edges,
                                                    out_dendrogram.data_handle(),
                                                    out_distances.data_handle(),
                                                    out_sizes.data_handle());
}

/**
 * Single-linkage clustering, capable of constructing a KNN graph to
 * scale the algorithm beyond the n^2 memory consumption of implementations
 * that use the fully-connected graph of pairwise distances by connecting
 * a knn graph when k is not large enough to connect it.

 * @tparam value_idx
 * @tparam value_t
 * @tparam dist_type method to use for constructing connectivities graph
 * @param[in] handle raft handle
 * @param[in] X dense input matrix in row-major layout
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] metric distance metric to use when constructing connectivities graph
 * @param[out] out struct containing output dendrogram and cluster assignments
 * @param[in] c a constant used when constructing connectivities from knn graph. Allows the indirect
 control
 *            of k. The algorithm will set `k = log(n) + c`
 * @param[in] n_clusters number of clusters to assign data samples
 */
template <typename value_idx, typename value_t, Linkage dist_type>
void single_linkage(raft::resources const& handle,
                    const value_t* X,
                    size_t m,
                    size_t n,
                    cuvs::distance::DistanceType metric,
                    single_linkage_output<value_idx>* out,
                    int c,
                    size_t n_clusters)
{
  ASSERT(n_clusters <= m, "n_clusters must be less than or equal to the number of data points");

  value_idx n_edges = m - 1;
  auto mst_rows     = raft::make_device_vector<value_idx, value_idx>(handle, n_edges);
  auto mst_cols     = raft::make_device_vector<value_idx, value_idx>(handle, n_edges);
  auto mst_weights  = raft::make_device_vector<value_t, value_idx>(handle, n_edges);
  auto structure_view =
    raft::make_device_coordinate_structure_view<value_idx, value_idx, value_idx>(
      mst_rows.data_handle(), mst_cols.data_handle(), m, m, n_edges);
  auto mst_view = raft::make_device_coo_matrix_view<value_t, value_idx, value_idx, value_idx>(
    mst_weights.data_handle(), structure_view);

  auto out_delta = raft::make_device_vector<value_t, value_idx>(handle, n_edges);
  auto out_sizes = raft::make_device_vector<value_idx, value_idx>(handle, n_edges);

  build_dist_linkage<value_t, value_idx, value_idx, dist_type>(
    handle,
    raft::make_device_matrix_view<const value_t, value_idx, raft::row_major>(
      X, static_cast<value_idx>(m), static_cast<value_idx>(n)),
    c,
    metric,
    mst_view,
    raft::make_device_matrix_view<value_idx, value_idx, raft::row_major>(out->children, n_edges, 2),
    out_delta.view(),
    out_sizes.view());

  detail::extract_flattened_clusters(handle, out->labels, out->children, n_clusters, m);

  out->m                      = m;
  out->n_clusters             = n_clusters;
  out->n_leaves               = m;
  out->n_connected_components = 1;
}
};  // namespace  cuvs::cluster::agglomerative::detail
