/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../src/neighbors/detail/knn_brute_force.cuh"
#include "../../src/neighbors/detail/reachability.cuh"
#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include "naive_knn.cuh"
#include <cstddef>
#include <cstring>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources_snmg.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>
#include <rapids_logger/logger.hpp>
#include <rmm/device_uvector.hpp>
#include <tuple>
#include <vector>

namespace cuvs::neighbors::all_neighbors {

enum knn_build_algo { BRUTE_FORCE, NN_DESCENT, IVF_PQ };

struct AllNeighborsInputs {
  std::tuple<knn_build_algo, cuvs::distance::DistanceType, double> build_algo_metric_recall;
  std::tuple<size_t, size_t> cluster_nearestcluster;
  int n_rows;
  int dim;
  int k;
  bool data_on_host;
  bool mutual_reach;
  bool output_on_host;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AllNeighborsInputs& p)
{
  os << "dataset shape=" << p.n_rows << "x" << p.dim << ", k=" << p.k
     << ", metric=" << static_cast<int>(std::get<1>(p.build_algo_metric_recall))
     << ", clusters=" << std::get<0>(p.cluster_nearestcluster)
     << ", overlap_factor=" << std::get<1>(p.cluster_nearestcluster)
     << ", output_on_host=" << p.output_on_host << std::endl;
  return os;
}

template <typename DistanceT, typename DataT, typename IdxT = int64_t>
void get_graphs(raft::resources& handle,
                rmm::device_uvector<DataT>& database,
                std::vector<IdxT>& indices_bf,
                std::vector<DistanceT>& distances_bf,
                std::vector<IdxT>& indices_allNN,
                std::vector<DistanceT>& distances_allNN,
                AllNeighborsInputs& ps,
                size_t queries_size)
{
  all_neighbors_params params;
  params.n_clusters     = std::get<0>(ps.cluster_nearestcluster);
  params.overlap_factor = std::get<1>(ps.cluster_nearestcluster);
  params.metric         = std::get<1>(ps.build_algo_metric_recall);

  auto build_algo = std::get<0>(ps.build_algo_metric_recall);

  if (build_algo == BRUTE_FORCE) {
    auto brute_force_params                = graph_build_params::brute_force_params{};
    brute_force_params.build_params.metric = params.metric;
    params.graph_build_params              = brute_force_params;
  } else if (build_algo == NN_DESCENT) {
    auto nn_descent_params                      = graph_build_params::nn_descent_params{};
    nn_descent_params.max_iterations            = 100;
    nn_descent_params.graph_degree              = ps.k;
    nn_descent_params.intermediate_graph_degree = ps.k * 2;
    nn_descent_params.metric                    = params.metric;
    params.graph_build_params                   = nn_descent_params;
  } else if (build_algo == IVF_PQ) {
    auto ivfpq_build_params                = graph_build_params::ivf_pq_params{};
    ivfpq_build_params.build_params.metric = params.metric;
    ivfpq_build_params.refinement_rate     = 2.0;
    // heuristically good ivfpq n_lists
    ivfpq_build_params.build_params.n_lists = std::max(
      5u, static_cast<uint32_t>(ps.n_rows * params.overlap_factor / (5000 * params.n_clusters)));
    params.graph_build_params = ivfpq_build_params;
  }

  auto metric      = std::get<1>(ps.build_algo_metric_recall);
  auto cuda_stream = raft::resource::get_cuda_stream(handle);
  {
    rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, cuda_stream);
    rmm::device_uvector<IdxT> indices_naive_dev(queries_size, cuda_stream);
    naive_knn<DistanceT, DataT, IdxT>(handle,
                                      distances_naive_dev.data(),
                                      indices_naive_dev.data(),
                                      database.data(),
                                      database.data(),
                                      ps.n_rows,
                                      ps.n_rows,
                                      ps.dim,
                                      ps.k,
                                      std::get<1>(ps.build_algo_metric_recall));

    if (ps.mutual_reach) {
      rmm::device_uvector<DistanceT> core_dists_dev(ps.n_rows, cuda_stream);

      cuvs::neighbors::detail::reachability::core_distances<IdxT, DistanceT>(
        handle, distances_naive_dev.data(), ps.k, ps.k, ps.n_rows, core_dists_dev.data());

      auto epilogue =
        cuvs::neighbors::detail::reachability::ReachabilityPostProcess<IdxT, DistanceT>{
          core_dists_dev.data(), 1.0, static_cast<size_t>(ps.n_rows)};

      cuvs::neighbors::detail::tiled_brute_force_knn<
        DataT,
        IdxT,
        DistanceT,
        cuvs::neighbors::detail::reachability::ReachabilityPostProcess<IdxT, DistanceT>>(
        handle,
        database.data(),
        database.data(),
        ps.n_rows,
        ps.n_rows,
        ps.dim,
        ps.k,
        distances_naive_dev.data(),
        indices_naive_dev.data(),
        metric,
        2.0,
        0,
        0,
        nullptr,
        nullptr,
        nullptr,
        epilogue);
    }

    raft::update_host(indices_bf.data(), indices_naive_dev.data(), queries_size, cuda_stream);
    raft::update_host(distances_bf.data(), distances_naive_dev.data(), queries_size, cuda_stream);

    raft::resource::sync_stream(handle);
  }

  {
    if (ps.output_on_host) {
      auto indices_allNN_h   = raft::make_host_matrix<IdxT, IdxT>(ps.n_rows, ps.k);
      auto distances_allNN_h = raft::make_host_matrix<DistanceT, IdxT>(ps.n_rows, ps.k);

      if (ps.data_on_host) {
        auto database_h = raft::make_host_matrix<DataT, IdxT>(ps.n_rows, ps.dim);
        raft::copy(database_h.data_handle(), database.data(), ps.n_rows * ps.dim, cuda_stream);

        all_neighbors::build(
          handle,
          params,
          raft::make_const_mdspan(database_h.view()),
          indices_allNN_h.view(),
          std::make_optional(distances_allNN_h.view()),
          ps.mutual_reach
            ? std::make_optional(raft::make_host_vector<DistanceT, IdxT>(ps.n_rows).view())
            : std::nullopt);
      } else {
        all_neighbors::build(
          handle,
          params,
          raft::make_device_matrix_view<const DataT, IdxT>(database.data(), ps.n_rows, ps.dim),
          indices_allNN_h.view(),
          std::make_optional(distances_allNN_h.view()),
          ps.mutual_reach
            ? std::make_optional(raft::make_host_vector<DistanceT, IdxT>(ps.n_rows).view())
            : std::nullopt);
      }

      memcpy(indices_allNN.data(), indices_allNN_h.data_handle(), queries_size * sizeof(IdxT));
      memcpy(
        distances_allNN.data(), distances_allNN_h.data_handle(), queries_size * sizeof(DistanceT));
    } else {
      rmm::device_uvector<DistanceT> distances_allNN_dev(queries_size, cuda_stream);
      rmm::device_uvector<IdxT> indices_allNN_dev(queries_size, cuda_stream);

      if (ps.data_on_host) {
        auto database_h = raft::make_host_matrix<DataT, IdxT>(ps.n_rows, ps.dim);
        raft::copy(database_h.data_handle(), database.data(), ps.n_rows * ps.dim, cuda_stream);

        all_neighbors::build(
          handle,
          params,
          raft::make_const_mdspan(database_h.view()),
          raft::make_device_matrix_view<IdxT>(indices_allNN_dev.data(), ps.n_rows, ps.k),
          raft::make_device_matrix_view<DistanceT>(distances_allNN_dev.data(), ps.n_rows, ps.k),
          ps.mutual_reach
            ? std::make_optional(raft::make_device_vector<DistanceT>(handle, ps.n_rows).view())
            : std::nullopt);
      } else {
        all_neighbors::build(
          handle,
          params,
          raft::make_device_matrix_view<const DataT, IdxT>(database.data(), ps.n_rows, ps.dim),
          raft::make_device_matrix_view<IdxT>(indices_allNN_dev.data(), ps.n_rows, ps.k),
          raft::make_device_matrix_view<DistanceT>(distances_allNN_dev.data(), ps.n_rows, ps.k),
          ps.mutual_reach
            ? std::make_optional(raft::make_device_vector<DistanceT>(handle, ps.n_rows).view())
            : std::nullopt);
      }

      raft::copy(indices_allNN.data(), indices_allNN_dev.data(), queries_size, cuda_stream);
      raft::copy(distances_allNN.data(), distances_allNN_dev.data(), queries_size, cuda_stream);
      raft::resource::sync_stream(handle);
    }
  }
}

template <typename DistanceT, typename DataT, typename IdxT = int64_t>
class AllNeighborsTest : public ::testing::TestWithParam<AllNeighborsInputs> {
 public:
  AllNeighborsTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AllNeighborsInputs>::GetParam()),
      database(0, stream_)
  {
  }

 protected:
  void run()
  {
    size_t queries_size = static_cast<size_t>(ps.n_rows) * static_cast<size_t>(ps.k);

    std::vector<IdxT> indices_allNN(queries_size);
    std::vector<DistanceT> distances_allNN(queries_size);
    std::vector<DistanceT> distances_bf(queries_size);
    std::vector<IdxT> indices_bf(queries_size);

    get_graphs(handle_,
               database,
               indices_bf,
               distances_bf,
               indices_allNN,
               distances_allNN,
               ps,
               queries_size);

    double min_recall = std::get<2>(ps.build_algo_metric_recall);

    EXPECT_TRUE(eval_recall(indices_bf, indices_allNN, ps.n_rows, ps.k, 0.01, min_recall, true));
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    auto database_view =
      raft::make_device_matrix_view<float, IdxT>(database.data(), ps.n_rows, ps.dim);
    auto labels = raft::make_device_vector<IdxT, IdxT>(handle_, ps.n_rows);
    raft::random::make_blobs(handle_, database_view, labels.view());
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
  }

 private:
  raft::device_resources_snmg handle_;
  rmm::cuda_stream_view stream_;
  AllNeighborsInputs ps;
  rmm::device_uvector<DataT> database;
};

const std::vector<AllNeighborsInputs> inputsSingle =
  raft::util::itertools::product<AllNeighborsInputs>(
    {std::make_tuple(BRUTE_FORCE, cuvs::distance::DistanceType::L2Expanded, 0.9),
     std::make_tuple(BRUTE_FORCE, cuvs::distance::DistanceType::L2SqrtExpanded, 0.9),
     std::make_tuple(BRUTE_FORCE, cuvs::distance::DistanceType::InnerProduct, 0.9),
     std::make_tuple(BRUTE_FORCE, cuvs::distance::DistanceType::CosineExpanded, 0.9),
     std::make_tuple(IVF_PQ, cuvs::distance::DistanceType::L2Expanded, 0.9),
     std::make_tuple(NN_DESCENT, cuvs::distance::DistanceType::L2Expanded, 0.9),
     std::make_tuple(NN_DESCENT, cuvs::distance::DistanceType::L2SqrtExpanded, 0.9),
     std::make_tuple(NN_DESCENT, cuvs::distance::DistanceType::CosineExpanded, 0.9),
     std::make_tuple(NN_DESCENT, cuvs::distance::DistanceType::InnerProduct, 0.8)},
    {std::make_tuple(1lu, 2lu)},  // min_recall, n_clusters, overlap_factor
    {5000, 7151},                 // n_rows
    {64, 137},                    // dim
    {16, 23},                     // graph_degree
    {false, true},                // data on host
    {false},                      // mutual_reach
    {false, true}                 // output on host
  );

const std::vector<AllNeighborsInputs> inputsBatch =
  raft::util::itertools::product<AllNeighborsInputs>(
    {std::make_tuple(BRUTE_FORCE, cuvs::distance::DistanceType::L2Expanded, 0.9),
     std::make_tuple(BRUTE_FORCE, cuvs::distance::DistanceType::L2SqrtExpanded, 0.9),
     std::make_tuple(BRUTE_FORCE, cuvs::distance::DistanceType::CosineExpanded, 0.9),
     std::make_tuple(BRUTE_FORCE, cuvs::distance::DistanceType::InnerProduct, 0.9),
     std::make_tuple(IVF_PQ, cuvs::distance::DistanceType::L2Expanded, 0.9),
     std::make_tuple(NN_DESCENT, cuvs::distance::DistanceType::L2Expanded, 0.9),
     std::make_tuple(NN_DESCENT, cuvs::distance::DistanceType::L2SqrtExpanded, 0.9),
     std::make_tuple(NN_DESCENT, cuvs::distance::DistanceType::CosineExpanded, 0.9),
     std::make_tuple(NN_DESCENT, cuvs::distance::DistanceType::InnerProduct, 0.9)},
    {
      std::make_tuple(4lu, 2lu),
      std::make_tuple(7lu, 2lu),
      std::make_tuple(10lu, 2lu),
    },             // min_recall, n_clusters, overlap_factor
    {5000, 7151},  // n_rows
    {64, 137},     // dim
    {16, 23},      // graph_degree
    {true},        // data on host
    {false},       // mutual_reach
    {false, true}  // output on host
  );

const std::vector<AllNeighborsInputs> mutualReachSingle =
  raft::util::itertools::product<AllNeighborsInputs>(
    {std::make_tuple(BRUTE_FORCE, cuvs::distance::DistanceType::L2Expanded, 0.9),
     std::make_tuple(BRUTE_FORCE, cuvs::distance::DistanceType::L2SqrtExpanded, 0.9),
     std::make_tuple(BRUTE_FORCE, cuvs::distance::DistanceType::CosineExpanded, 0.9),
     std::make_tuple(NN_DESCENT, cuvs::distance::DistanceType::L2Expanded, 0.9),
     std::make_tuple(NN_DESCENT, cuvs::distance::DistanceType::L2SqrtExpanded, 0.9),
     std::make_tuple(NN_DESCENT, cuvs::distance::DistanceType::CosineExpanded, 0.9)},
    {std::make_tuple(1lu, 2lu)},  // n_clusters, overlap_factor
    {5000, 7151},                 // n_rows
    {64, 137},                    // dim
    {16, 23},                     // graph_degree
    {false, true},                // data on host
    {true},                       // mutual_reach
    {false, true}                 // output on host
  );

const std::vector<AllNeighborsInputs> mutualReachBatch =
  raft::util::itertools::product<AllNeighborsInputs>(
    {std::make_tuple(BRUTE_FORCE, cuvs::distance::DistanceType::L2Expanded, 0.9),
     std::make_tuple(BRUTE_FORCE, cuvs::distance::DistanceType::L2SqrtExpanded, 0.9),
     std::make_tuple(BRUTE_FORCE, cuvs::distance::DistanceType::CosineExpanded, 0.9),
     std::make_tuple(NN_DESCENT, cuvs::distance::DistanceType::L2Expanded, 0.9),
     std::make_tuple(NN_DESCENT, cuvs::distance::DistanceType::L2SqrtExpanded, 0.9),
     std::make_tuple(NN_DESCENT, cuvs::distance::DistanceType::CosineExpanded, 0.9)},
    {
      std::make_tuple(4lu, 2lu),
      std::make_tuple(7lu, 2lu),
      std::make_tuple(10lu, 2lu),
    },             // n_clusters, overlap_factor
    {5000, 7151},  // n_rows
    {64, 137},     // dim
    {16, 23},      // graph_degree
    {true},        // data on host
    {true},        // mutual_reach
    {false, true}  // output on host
  );

}  // namespace cuvs::neighbors::all_neighbors
