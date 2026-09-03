/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../sparse/neighbors/cross_component_nn.cuh"
#include "raft/core/device_mdspan.hpp"
#include "raft/core/operators.hpp"
#include <cuvs/distance/distance.hpp>
#include <raft/core/memory_type.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/map.cuh>
#include <raft/matrix/detail/gather.cuh>
#include <raft/matrix/diagonal.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/sparse/solver/mst.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/tuple>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

namespace cuvs::cluster::agglomerative::detail {

template <typename value_idx, typename value_t>
void merge_msts(raft::sparse::solver::Graph_COO<value_idx, value_idx, value_t>& coo1,
                raft::sparse::solver::Graph_COO<value_idx, value_idx, value_t>& coo2,
                cudaStream_t stream)
{
  /** Add edges to existing mst **/
  int final_nnz = coo2.n_edges + coo1.n_edges;

  coo1.src.resize(final_nnz, stream);
  coo1.dst.resize(final_nnz, stream);
  coo1.weights.resize(final_nnz, stream);

  /**
   * Construct final edge list
   */
  raft::copy_async(coo1.src.data() + coo1.n_edges, coo2.src.data(), coo2.n_edges, stream);
  raft::copy_async(coo1.dst.data() + coo1.n_edges, coo2.dst.data(), coo2.n_edges, stream);
  raft::copy_async(coo1.weights.data() + coo1.n_edges, coo2.weights.data(), coo2.n_edges, stream);

  coo1.n_edges = final_nnz;
}

/**
 * Connect an unconnected knn graph (one in which mst returns an msf). The
 * device buffers underlying the Graph_COO object are modified in-place.
 * @tparam value_idx index type
 * @tparam value_t floating-point value type
 * @param[in] handle raft handle
 * @param[in] X original dense data on device memory from which knn graph was constructed
 * @param[inout] msf edge list containing the mst result
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[inout] color the color labels array returned from the mst invocation
 * @return updated MST edge list
 */
template <typename value_idx, typename value_t, typename red_op>
void connect_knn_graph(
  raft::resources const& handle,
  raft::device_matrix_view<const value_t, value_idx> X,
  raft::sparse::solver::Graph_COO<value_idx, value_idx, value_t>& msf,
  size_t m,
  size_t n,
  value_idx* color,
  red_op reduction_op,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtExpanded)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  raft::sparse::COO<value_t, value_idx> connected_edges(stream.get());

  // default row and column batch sizes are chosen for computing cross component nearest neighbors.
  // Reference: PR #1445
  static constexpr size_t default_row_batch_size = 4096;
  static constexpr size_t default_col_batch_size = 16;

  cuvs::sparse::neighbors::cross_component_nn<value_idx, value_t>(handle,
                                                                  connected_edges,
                                                                  X.data_handle(),
                                                                  color,
                                                                  m,
                                                                  n,
                                                                  reduction_op,
                                                                  min(m, default_row_batch_size),
                                                                  min(n, default_col_batch_size));

  rmm::device_uvector<value_idx> indptr2(m + 1, stream);
  raft::sparse::convert::sorted_coo_to_csr(
    connected_edges.rows(), connected_edges.nnz, indptr2.data(), m + 1, stream.get());

  // On the second call, we hand the MST the original colors
  // and the new set of edges and let it restart the optimization process
  auto new_mst =
    raft::sparse::solver::mst<value_idx, value_idx, value_t, double>(handle,
                                                                     indptr2.data(),
                                                                     connected_edges.cols(),
                                                                     connected_edges.vals(),
                                                                     m,
                                                                     connected_edges.nnz,
                                                                     color,
                                                                     stream.get(),
                                                                     false,
                                                                     false);

  merge_msts<value_idx, value_t>(msf, new_mst, stream.get());
}

/**
 * Connect an unconnected knn graph (one in which mst returns an msf). The
 * device buffers underlying the Graph_COO object are modified in-place.
 * @tparam value_idx index type
 * @tparam value_t floating-point value type
 * @param[in] handle raft handle
 * @param[in] X original dense data on host memory from which knn graph was constructed
 * @param[inout] msf edge list containing the mst result
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[inout] color the color labels array returned from the mst invocation
 * @return updated MST edge list
 */
template <typename value_idx, typename value_t, typename red_op = raft::identity_op>
void connect_knn_graph(
  raft::resources const& handle,
  raft::host_matrix_view<const value_t, value_idx> X,
  raft::sparse::solver::Graph_COO<value_idx, value_idx, value_t>& msf,
  size_t m,
  size_t n,
  value_idx* color,
  red_op reduction_op                 = red_op{},
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtExpanded)
{
  using namespace cuvs::sparse::neighbors;
  static_assert(
    std::is_same_v<red_op, raft::identity_op> ||
      std::is_same_v<red_op, MutualReachabilityFixConnectivitiesRedOp<value_idx, value_t>> ||
      std::is_same_v<red_op, FixConnectivitiesRedOp<value_idx, value_t>>,
    "reduction_op must be identity_op, MutualReachabilityFixConnectivitiesRedOp, or "
    "FixConnectivitiesRedOp");

  auto stream      = raft::resource::get_cuda_stream(handle);
  int n_components = get_n_components(color, m, stream.get());

  rmm::device_uvector<value_idx> d_color_remapped(m, stream);
  raft::label::make_monotonic(d_color_remapped.data(), color, m, stream.get(), true);

  std::vector<value_idx> h_color(m);
  raft::copy(handle,
             raft::make_host_vector_view(h_color.data(), m),
             raft::make_device_vector_view<const value_idx>(d_color_remapped.data(), m));
  raft::resource::sync_stream(handle, stream);

  // make key (color) : value (vector of ids that have that color)
  std::unordered_map<value_idx, std::vector<value_idx>> component_map;
  for (value_idx i = 0; i < static_cast<value_idx>(m); ++i) {
    component_map[h_color[i]].push_back(i);
  }

  std::vector<std::tuple<value_idx, value_idx, value_t>> selected_edges;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis;

  std::vector<value_idx> host_u_indices;
  std::vector<value_idx> host_v_indices;

  // connect i-1 component and i component
  for (int i = 1; i < n_components; ++i) {
    value_idx color_a = i - 1;
    value_idx color_b = i;

    const auto& nodes_a = component_map[color_a];
    const auto& nodes_b = component_map[color_b];

    // Randomly pick a data index from each component
    dis.param(std::uniform_int_distribution<>::param_type(0, nodes_a.size() - 1));
    value_idx u = nodes_a[dis(gen)];

    dis.param(std::uniform_int_distribution<>::param_type(0, nodes_b.size() - 1));
    value_idx v = nodes_b[dis(gen)];

    host_u_indices.push_back(u);
    host_v_indices.push_back(v);
  }

  size_t new_nnz = n_components - 1;

  auto device_u_indices = raft::make_device_vector<value_idx, value_idx>(handle, new_nnz);
  auto device_v_indices = raft::make_device_vector<value_idx, value_idx>(handle, new_nnz);

  raft::copy(
    handle,
    device_u_indices.view(),
    raft::make_host_vector_view<const value_idx>(host_u_indices.data(), value_idx(new_nnz)));
  raft::copy(
    handle,
    device_v_indices.view(),
    raft::make_host_vector_view<const value_idx>(host_v_indices.data(), value_idx(new_nnz)));

  auto data_u = raft::make_device_matrix<value_t, value_idx>(handle, new_nnz, n);
  auto data_v = raft::make_device_matrix<value_t, value_idx>(handle, new_nnz, n);

  raft::matrix::detail::gather(
    handle, X, raft::make_const_mdspan(device_u_indices.view()), data_u.view());
  raft::matrix::detail::gather(
    handle, X, raft::make_const_mdspan(device_v_indices.view()), data_v.view());

  auto pairwise_dist = raft::make_device_matrix<value_t, value_idx>(handle, new_nnz, new_nnz);
  cuvs::distance::pairwise_distance(handle,
                                    raft::make_const_mdspan(data_u.view()),
                                    raft::make_const_mdspan(data_v.view()),
                                    pairwise_dist.view(),
                                    metric);

  auto pairwise_dist_vec = raft::make_device_vector<value_t, value_idx>(handle, n_components - 1);
  raft::matrix::get_diagonal(
    handle, raft::make_const_mdspan(pairwise_dist.view()), pairwise_dist_vec.view());

  if constexpr (std::is_same<red_op,
                             MutualReachabilityFixConnectivitiesRedOp<value_idx, value_t>>::value) {
    raft::linalg::map_offset(
      handle,
      pairwise_dist_vec.view(),
      [pairwise_dist_ptr    = pairwise_dist_vec.data_handle(),
       core_dist_ptr        = reduction_op.core_dists,
       device_u_indices_ptr = device_u_indices.data_handle(),
       device_v_indices_ptr = device_v_indices.data_handle()] __device__(auto i) {
        float dist        = pairwise_dist_ptr[i];
        float u_core_dist = core_dist_ptr[device_u_indices_ptr[i]];
        float v_core_dist = core_dist_ptr[device_v_indices_ptr[i]];
        return fmaxf(dist, fmaxf(u_core_dist, v_core_dist));
      });
  }

  // sort in order of rows to run sorted_coo_to_csr
  auto rows_begin = thrust::device_pointer_cast(device_u_indices.data_handle());
  auto cols_begin = thrust::device_pointer_cast(device_v_indices.data_handle());
  auto dist_begin = thrust::device_pointer_cast(pairwise_dist_vec.data_handle());

  auto zipped_begin = thrust::make_zip_iterator(cuda::std::make_tuple(cols_begin, dist_begin));
  thrust::sort_by_key(rows_begin, rows_begin + new_nnz, zipped_begin);

  rmm::device_uvector<value_idx> indptr2(m + 1, stream);
  raft::sparse::convert::sorted_coo_to_csr(
    device_u_indices.data_handle(), new_nnz, indptr2.data(), m + 1, stream.get());

  // On the second call, we hand the MST the original colors
  // and the new set of edges and let it restart the optimization process
  auto new_mst = raft::sparse::solver::mst<value_idx, value_idx, value_t, double>(
    handle,
    indptr2.data(),
    device_v_indices.data_handle(),
    pairwise_dist_vec.data_handle(),
    m,
    new_nnz,
    color,
    stream.get(),
    false,
    false);

  merge_msts<value_idx, value_t>(msf, new_mst, stream.get());
}

/**
 * Constructs an MST and sorts the resulting edges in ascending
 * order by their weight.
 *
 * Hierarchical clustering heavily relies upon the ordering
 * and vertices returned in the MST. If the result of the
 * MST was actually a minimum-spanning forest, the CSR
 * being passed into the MST is not connected. In such a
 * case, this graph will be connected by performing a
 * KNN across the components.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle
 * @param[in] X dataset residing on host or device memory
 * @param[in] indptr CSR indptr of connectivities graph
 * @param[in] indices CSR indices array of connectivities graph
 * @param[in] pw_dists CSR weights array of connectivities graph
 * @param[in] m number of rows in X / src vertices in connectivities graph
 * @param[in] n number of columns in X
 * @param[out] mst_src output src edges
 * @param[out] mst_dst output dst edges
 * @param[out] mst_weight output weights (distances)
 * @param[in] max_iter maximum iterations to run knn graph connection. This
 *  argument is really just a safeguard against the potential for infinite loops.
 */
template <typename value_idx, typename value_t, typename red_op>
void build_sorted_mst(
  raft::resources const& handle,
  const value_t* X,
  const value_idx* indptr,
  const value_idx* indices,
  const value_t* pw_dists,
  size_t m,
  size_t n,
  value_idx* mst_src,
  value_idx* mst_dst,
  value_t* mst_weight,
  value_idx* color,
  size_t nnz,
  red_op reduction_op,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtExpanded,
  int max_iter                        = 10)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  // We want to have MST initialize colors on first call.
  auto mst_coo = raft::sparse::solver::mst<value_idx, value_idx, value_t, double>(
    handle, indptr, indices, pw_dists, (value_idx)m, nnz, color, stream.get(), false, true);

  int iters        = 1;
  int n_components = cuvs::sparse::neighbors::get_n_components(color, m, stream.get());

  bool data_on_device = raft::memory_type_from_pointer(X) != raft::memory_type::host;

  while (n_components > 1 && iters < max_iter) {
    if (data_on_device) {
      connect_knn_graph<value_idx, value_t>(
        handle,
        raft::make_device_matrix_view<const value_t, value_idx>(X, m, n),
        mst_coo,
        m,
        n,
        color,
        reduction_op);
    } else {
      connect_knn_graph<value_idx, value_t>(
        handle,
        raft::make_host_matrix_view<const value_t, value_idx>(X, m, n),
        mst_coo,
        m,
        n,
        color,
        reduction_op,
        metric);
    }

    iters++;

    n_components = cuvs::sparse::neighbors::get_n_components(color, m, stream.get());
  }

  /**
   * The `max_iter` argument was introduced only to prevent the potential for an infinite loop.
   * Ideally the log2(n) guarantees of the MST should be enough to connect KNN graphs with a
   * massive number of data samples in very few iterations. If it does not, there are 3 likely
   * reasons why (in order of their likelihood):
   * 1. There is a bug in this code somewhere
   * 2. Either the given KNN graph wasn't generated from X or the same metric is not being used
   *    to generate the 1-nn (currently only L2SqrtExpanded is supported).
   * 3. max_iter was not large enough to connect the graph (less likely).
   *
   * Note that a KNN graph generated from 50 random isotropic balls (with significant overlap)
   * was able to be connected in a single iteration.
   */
  RAFT_EXPECTS(n_components == 1,
               "KNN graph could not be connected in %d iterations. "
               "Please verify that the input knn graph is generated from X "
               "(and the same distance metric used),"
               " or increase 'max_iter'",
               max_iter);

  raft::sparse::op::coo_sort_by_weight(
    mst_coo.src.data(), mst_coo.dst.data(), mst_coo.weights.data(), mst_coo.n_edges, stream.get());

  raft::copy_async(mst_src, mst_coo.src.data(), mst_coo.n_edges, stream.get());
  raft::copy_async(mst_dst, mst_coo.dst.data(), mst_coo.n_edges, stream.get());
  raft::copy_async(mst_weight, mst_coo.weights.data(), mst_coo.n_edges, stream.get());
}

};  // namespace  cuvs::cluster::agglomerative::detail
