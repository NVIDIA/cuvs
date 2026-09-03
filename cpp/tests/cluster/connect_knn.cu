/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/distance/distance.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/map.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>

#include "../../src/cluster/detail/mst.cuh"
#include "../../src/neighbors/detail/knn_brute_force.cuh"
#include "../../src/neighbors/detail/reachability.cuh"
#include "../../src/sparse/neighbors/cross_component_nn.cuh"
#include "../neighbors/naive_knn.cuh"
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>
#include <stdio.h>
#include <test_utils.h>
#include <vector>

namespace cuvs {

struct ConnectKNNInputs {
  int n_rows;
  int dim;
  int n_clusters;
  int k;
  cuvs::distance::DistanceType metric;
  bool mutual_reach;
};

template <typename T>
class ConnectKNNTest : public ::testing::TestWithParam<ConnectKNNInputs> {
 public:
  ConnectKNNTest()
    : stream(handle.get_stream().get()),
      ps(::testing::TestWithParam<ConnectKNNInputs>::GetParam()),
      database(0, stream)
  {
  }

  void basicTest()
  {
    int queries_size = ps.n_rows * ps.k;
    rmm::device_uvector<T> dists(queries_size, stream);
    rmm::device_uvector<int64_t> inds(queries_size, stream);
    cuvs::neighbors::naive_knn<T, T, int64_t>(handle,
                                              dists.data(),
                                              inds.data(),
                                              database.data(),
                                              database.data(),
                                              ps.n_rows,
                                              ps.n_rows,
                                              ps.dim,
                                              ps.k,
                                              ps.metric);

    rmm::device_uvector<T> core_dists(ps.n_rows, stream);
    if (ps.mutual_reach) {
      cuvs::neighbors::detail::reachability::core_distances<int64_t, T>(
        handle, dists.data(), ps.k, ps.k, (size_t)ps.n_rows, core_dists.data());

      auto epilogue = cuvs::neighbors::detail::reachability::ReachabilityPostProcess<int64_t, T>{
        core_dists.data(), 1.0};
      cuvs::neighbors::detail::tiled_brute_force_knn<
        T,
        int64_t,
        T,
        cuvs::neighbors::detail::reachability::ReachabilityPostProcess<int64_t, T>>(handle,
                                                                                    database.data(),
                                                                                    database.data(),
                                                                                    ps.n_rows,
                                                                                    ps.n_rows,
                                                                                    ps.dim,
                                                                                    ps.k,
                                                                                    dists.data(),
                                                                                    inds.data(),
                                                                                    ps.metric,
                                                                                    2.0,
                                                                                    0,
                                                                                    0,
                                                                                    nullptr,
                                                                                    nullptr,
                                                                                    nullptr,
                                                                                    epilogue);
    }

    rmm::device_uvector<int64_t> coo_rows(queries_size, stream);

    raft::sparse::COO<T, int64_t> knn_coo(stream, queries_size * 2);
    rmm::device_uvector<int64_t> indptr(ps.n_rows + 1, stream);

    // changing inds and dists to sparse format
    int64_t k = ps.k;
    auto coo_rows_view =
      raft::make_device_vector_view<int64_t, int64_t>(coo_rows.data(), ps.n_rows * ps.k);
    raft::linalg::map_offset(
      handle, coo_rows_view, [k] __device__(int64_t c) -> int64_t { return c / k; });

    raft::sparse::linalg::symmetrize(handle,
                                     coo_rows.data(),
                                     inds.data(),
                                     dists.data(),
                                     (int64_t)ps.n_rows,
                                     (int64_t)ps.n_rows,
                                     static_cast<size_t>(queries_size),
                                     knn_coo);

    raft::sparse::convert::sorted_coo_to_csr(
      knn_coo.rows(), knn_coo.nnz, indptr.data(), ps.n_rows + 1, stream);

    // run mst solver
    rmm::device_uvector<int64_t> color(ps.n_rows, stream);
    auto mst_coo = raft::sparse::solver::mst<int64_t, int64_t, T, double>(handle,
                                                                          indptr.data(),
                                                                          knn_coo.cols(),
                                                                          knn_coo.vals(),
                                                                          (int64_t)ps.n_rows,
                                                                          knn_coo.nnz,
                                                                          color.data(),
                                                                          stream,
                                                                          false,
                                                                          true);

    // connect knn graph on host and checking final n_components
    // because the dataset is a blobs dataset, original n_components results in n_clusters of the
    // dataset
    int n_components = cuvs::sparse::neighbors::get_n_components(color.data(), ps.n_rows, stream);

    auto database_h = raft::make_host_matrix<T, int64_t>(ps.n_rows, ps.dim);
    raft::copy(database_h.data_handle(), database.data(), ps.n_rows * ps.dim, stream);

    if (ps.mutual_reach) {
      cuvs::sparse::neighbors::MutualReachabilityFixConnectivitiesRedOp<int64_t, T> reduction_op(
        core_dists.data(), (int64_t)ps.n_rows);
      cuvs::cluster::agglomerative::detail::connect_knn_graph<int64_t, T>(
        handle,
        raft::make_const_mdspan(database_h.view()),
        mst_coo,
        ps.n_rows,
        ps.dim,
        color.data(),
        reduction_op,
        ps.metric);
    } else {
      cuvs::cluster::agglomerative::detail::connect_knn_graph<int64_t, T>(
        handle,
        raft::make_const_mdspan(database_h.view()),
        mst_coo,
        ps.n_rows,
        ps.dim,
        color.data(),
        raft::identity_op{},
        ps.metric);
    }
    n_components = cuvs::sparse::neighbors::get_n_components(color.data(), ps.n_rows, stream);

    ASSERT_TRUE(n_components == 1);
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream);
    auto database_view =
      raft::make_device_matrix_view<T, int64_t>(database.data(), ps.n_rows, ps.dim);
    auto labels = raft::make_device_vector<int64_t, int64_t>(handle, ps.n_rows);
    raft::random::make_blobs(handle, database_view, labels.view(), (int64_t)ps.n_clusters);
    raft::resource::sync_stream(handle);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;
  ConnectKNNInputs ps;
  rmm::device_uvector<T> database;
};

const std::vector<ConnectKNNInputs> inputs = raft::util::itertools::product<ConnectKNNInputs>(
  {5000, 7151},                                    // n_rows
  {64, 137},                                       // dim
  {5, 10},                                         // n_clusters of make_blobs data
  {16},                                            // k
  {cuvs::distance::DistanceType::L2SqrtExpanded},  // metric
  {true, false});                                  // mutual_reach

typedef ConnectKNNTest<float> ConnectKNNTestF;
TEST_P(ConnectKNNTestF, ConnectKNN) { this->basicTest(); }

INSTANTIATE_TEST_CASE_P(ConnectKNNTests, ConnectKNNTestF, ::testing::ValuesIn(inputs));

}  // end namespace cuvs
