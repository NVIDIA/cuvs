/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "../../core/map_launchers.hpp"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/all_neighbors.hpp>

#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/map.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/util/cuda_dev_essentials.cuh>

#include <rmm/device_uvector.hpp>

#include <algorithm>

namespace cuvs::neighbors::detail {

template <typename value_idx>
value_idx build_k(value_idx n_samples, int c)
{
  // from "kNN-MST-Agglomerative: A fast & scalable graph-based data clustering
  // approach on GPU"
  return std::min(n_samples, std::max((value_idx)2, (value_idx)floor(raft::log2(n_samples)) + c));
}
/**
 * Constructs a (symmetrized) knn graph edge list from
 * dense input vectors.
 *
 * Note: The resulting KNN graph is not guaranteed to be connected.
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam nnz_t
 * @param[in] res raft res
 * @param[in] X dense matrix of input data samples and observations (size m * n)
 * @param[in] metric distance metric to use when constructing neighborhoods
 * @param[out] out output edge list
 * @param[in] c a constant used when constructing linkage from knn graph. Allows the indirect
 control of k. The algorithm will set `k = log(n) + c`
 */
template <typename value_idx = int, typename value_t = float, typename nnz_t = size_t>
void knn_graph(raft::resources const& res,
               raft::device_matrix_view<const value_t, value_idx> X,
               cuvs::distance::DistanceType metric,
               raft::sparse::COO<value_t, value_idx, nnz_t>& out,
               int c = 15)
{
  size_t m = X.extent(0);
  size_t n = X.extent(1);
  size_t k = build_k(m, c);

  auto stream = raft::resource::get_cuda_stream(res);

  nnz_t nnz = m * k;

  rmm::device_uvector<value_idx> rows(nnz, stream);
  rmm::device_uvector<value_idx> indices(nnz, stream);
  rmm::device_uvector<value_t> data(nnz, stream);

  auto rows_view = raft::make_device_vector_view<value_idx, nnz_t>(rows.data(), nnz);

  raft::linalg::map_offset(res, rows_view, [k] __device__(nnz_t i) { return value_idx(i / k); });

  cuvs::neighbors::all_neighbors::all_neighbors_params params;
  params.metric = metric;

  cuvs::neighbors::graph_build_params::brute_force_params bf_params;
  bf_params.build_params.metric = metric;
  params.graph_build_params     = bf_params;

  params.n_clusters     = 1;
  params.overlap_factor = 1;

  rmm::device_uvector<int64_t> indices_64(nnz, stream);
  auto indices_64_view = raft::make_device_matrix_view<int64_t, int64_t>(indices_64.data(), m, k);
  auto distances_view  = raft::make_device_matrix_view<value_t, int64_t>(data.data(), m, k);

  cuvs::neighbors::all_neighbors::build(
    res,
    params,
    raft::make_device_matrix_view<const value_t, int64_t>(X.data_handle(), m, n),
    indices_64_view,
    distances_view);

  cuvs::detail::map(res,
                    raft::make_device_vector_view<value_idx, nnz_t>(indices.data(), nnz),
                    raft::cast_op<value_idx>{},
                    raft::make_const_mdspan(indices_64_view));

  raft::sparse::linalg::symmetrize(res,
                                   rows.data(),
                                   indices.data(),
                                   data.data(),
                                   static_cast<value_idx>(m),
                                   static_cast<value_idx>(k),
                                   nnz,
                                   out);
}
};  // namespace cuvs::neighbors::detail
