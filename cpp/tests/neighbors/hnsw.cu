/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "ann_utils.cuh"

#include <cstdint>
#include <cuvs/neighbors/hnsw.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/itertools.hpp>

#include "naive_knn.cuh"

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace cuvs::neighbors::hnsw {

struct AnnHNSWInputs {
  int n_rows;
  int dim;
  int graph_degree;
  cuvs::distance::DistanceType metric;
  int k;
  int n_queries;
  int ef;
  double min_recall;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnHNSWInputs& p)
{
  os << "dataset shape=" << p.n_rows << "x" << p.dim << ", graph_degree=" << p.graph_degree
     << ", metric="
     << cuvs::neighbors::print_metric{static_cast<cuvs::distance::DistanceType>((int)p.metric)}
     << ", ef=" << (p.ef);
  return os;
}

template <typename DistanceT, typename DataT, typename IdxT>
class AnnHNSWTest : public ::testing::TestWithParam<AnnHNSWInputs> {
 public:
  AnnHNSWTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnHNSWInputs>::GetParam()),
      database(0, stream_),
      queries(0, stream_)
  {
  }

 protected:
  void testHNSW()
  {
    std::vector<IdxT> indices_HNSW(ps.n_queries * ps.k);
    std::vector<DistanceT> distances_HNSW(ps.n_queries * ps.k);
    std::vector<IdxT> indices_naive(ps.n_queries * ps.k);
    std::vector<DistanceT> distances_naive(ps.n_queries * ps.k);

    std::vector<DataT> queries_h(ps.n_queries * ps.dim);
    raft::update_host(queries_h.data(), queries.data(), ps.n_queries * ps.dim, stream_);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(ps.n_queries * ps.k, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(ps.n_queries * ps.k, stream_);
      naive_knn<DistanceT, DataT, IdxT>(handle_,
                                        distances_naive_dev.data(),
                                        indices_naive_dev.data(),
                                        queries.data(),
                                        database.data(),
                                        ps.n_queries,
                                        ps.n_rows,
                                        ps.dim,
                                        ps.k,
                                        ps.metric);
      raft::update_host(
        indices_naive.data(), indices_naive_dev.data(), ps.n_queries * ps.k, stream_);
      raft::update_host(
        distances_naive.data(), distances_naive_dev.data(), ps.n_queries * ps.k, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      cuvs::neighbors::cagra::index_params index_params;
      index_params.metric                    = ps.metric;
      index_params.graph_degree              = ps.graph_degree;
      index_params.intermediate_graph_degree = 2 * ps.graph_degree;

      auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
        (const DataT*)database.data(), ps.n_rows, ps.dim);

      auto index = cuvs::neighbors::cagra::build(handle_, index_params, database_view);
      raft::resource::sync_stream(handle_);

      cuvs::neighbors::hnsw::search_params search_params;
      search_params.ef = ps.ef;
      cuvs::neighbors::hnsw::index_params hnsw_params;
      auto hnsw_index = cuvs::neighbors::hnsw::from_cagra(handle_, hnsw_params, index);
      auto queries_HNSW_view =
        raft::make_host_matrix_view<DataT, int64_t>(queries_h.data(), ps.n_queries, ps.dim);
      auto indices_HNSW_view =
        raft::make_host_matrix_view<uint64_t, int64_t>(indices_HNSW.data(), ps.n_queries, ps.k);
      auto distances_HNSW_view =
        raft::make_host_matrix_view<float, int64_t>(distances_HNSW.data(), ps.n_queries, ps.k);
      cuvs::neighbors::hnsw::search(handle_,
                                    search_params,
                                    *hnsw_index.get(),
                                    queries_HNSW_view,
                                    indices_HNSW_view,
                                    distances_HNSW_view);
    }

    double min_recall = ps.min_recall;
    EXPECT_TRUE(eval_neighbours(indices_naive,
                                indices_HNSW,
                                distances_naive,
                                distances_HNSW,
                                ps.n_queries,
                                ps.k,
                                0.006,
                                min_recall));
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    queries.resize(((size_t)ps.n_queries) * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same_v<DataT, float> || std::is_same_v<DataT, half>) {
      raft::random::uniform(
        handle_, r, database.data(), ps.n_rows * ps.dim, DataT(0.1), DataT(2.0));
      raft::random::uniform(
        handle_, r, queries.data(), ps.n_queries * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.n_rows * ps.dim, DataT(1), DataT(100));
      raft::random::uniformInt(
        handle_, r, queries.data(), ps.n_queries * ps.dim, DataT(1), DataT(100));
    }
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnHNSWInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> queries;
};

const std::vector<AnnHNSWInputs> inputs = raft::util::itertools::product<AnnHNSWInputs>(
  {1000, 2000},                          // n_rows
  {5, 10, 25, 50, 100, 250, 500, 1000},  // dim
  {32, 64},                              // graph_degree
  {cuvs::distance::DistanceType::L2Expanded, cuvs::distance::DistanceType::InnerProduct},  // metric
  {50},                                                                                    // k
  {500},  // n_queries
  {250},  // ef
  {0.98}  // min_recall
);

typedef AnnHNSWTest<float, float, uint64_t> AnnHNSW_F;
TEST_P(AnnHNSW_F, AnnHNSW) { this->testHNSW(); }

INSTANTIATE_TEST_CASE_P(AnnHNSWTest, AnnHNSW_F, ::testing::ValuesIn(inputs));

typedef AnnHNSWTest<float, half, uint64_t> AnnHNSW_H;
TEST_P(AnnHNSW_H, AnnHNSW) { this->testHNSW(); }

INSTANTIATE_TEST_CASE_P(AnnHNSWTest, AnnHNSW_H, ::testing::ValuesIn(inputs));

typedef AnnHNSWTest<float, int8_t, uint64_t> AnnHNSW_I8;
TEST_P(AnnHNSW_I8, AnnHNSW) { this->testHNSW(); }

INSTANTIATE_TEST_CASE_P(AnnHNSWTest, AnnHNSW_I8, ::testing::ValuesIn(inputs));

typedef AnnHNSWTest<float, uint8_t, uint64_t> AnnHNSW_U8;
TEST_P(AnnHNSW_U8, AnnHNSW) { this->testHNSW(); }

INSTANTIATE_TEST_CASE_P(AnnHNSWTest, AnnHNSW_U8, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::hnsw
