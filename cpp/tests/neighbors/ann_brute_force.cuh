/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../test_utils.cuh"

#include "knn_utils.cuh"
#include "naive_knn.cuh"

#include <cuvs/neighbors/brute_force.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>

#include <gtest/gtest.h>

namespace cuvs::neighbors::brute_force {

template <typename IdxT>
struct AnnBruteForceInputs {
  IdxT num_queries;
  IdxT num_db_vecs;
  IdxT dim;
  IdxT k;
  cuvs::distance::DistanceType metric;
  float metric_arg = 0.0f;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const AnnBruteForceInputs<IdxT>& p)
{
  os << "{ " << p.num_queries << ", " << p.num_db_vecs << ", " << p.dim << ", " << p.k << ", "
     << static_cast<int>(p.metric) << static_cast<T>(p.metric_arg) << '}';
  return os;
}

template <typename T, typename DataT, typename IdxT>
class AnnBruteForceTest : public ::testing::TestWithParam<AnnBruteForceInputs<IdxT>> {
 public:
  AnnBruteForceTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnBruteForceInputs<IdxT>>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

  void testBruteForce()
  {
    size_t queries_size = ps.num_queries * ps.k;

    rmm::device_uvector<T> distances_naive_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);

    cuvs::neighbors::naive_knn<T, DataT, IdxT>(handle_,
                                               distances_naive_dev.data(),
                                               indices_naive_dev.data(),
                                               search_queries.data(),
                                               database.data(),
                                               ps.num_queries,
                                               ps.num_db_vecs,
                                               ps.dim,
                                               ps.k,
                                               ps.metric);
    raft::resource::sync_stream(handle_);

    {
      // Require exact result for brute force
      rmm::device_uvector<T> distances_bruteforce_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_bruteforce_dev(queries_size, stream_);

      auto idx = [this]() {
        auto database_view = raft::make_device_matrix_view<const DataT, IdxT>(
          (const DataT*)database.data(), ps.num_db_vecs, ps.dim);

        return brute_force::build(handle_, database_view, ps.metric, ps.metric_arg);
      }();

      auto search_queries_view = raft::make_device_matrix_view<const DataT, IdxT>(
        search_queries.data(), ps.num_queries, ps.dim);
      auto indices_out_view = raft::make_device_matrix_view<IdxT, IdxT>(
        indices_bruteforce_dev.data(), ps.num_queries, ps.k);
      auto dists_out_view = raft::make_device_matrix_view<T, IdxT>(
        distances_bruteforce_dev.data(), ps.num_queries, ps.k);

      brute_force::search(handle_,
                          idx,
                          search_queries_view,
                          indices_out_view,
                          dists_out_view,
                          cuvs::neighbors::filtering::none_sample_filter{});

      raft::resource::sync_stream(handle_);

      ASSERT_TRUE(cuvs::neighbors::devArrMatchKnnPair(indices_naive_dev.data(),
                                                      indices_bruteforce_dev.data(),
                                                      distances_naive_dev.data(),
                                                      distances_bruteforce_dev.data(),
                                                      ps.num_queries,
                                                      ps.k,
                                                      0.001f,
                                                      stream_,
                                                      true));

      tmp_index_file index_file;
      brute_force::serialize(handle_, index_file.filename, idx, true);
      auto index_loaded = brute_force::index<DataT, T>(handle_);
      brute_force::deserialize(handle_, index_file.filename, &index_loaded);

      brute_force::search(handle_,
                          index_loaded,
                          search_queries_view,
                          indices_out_view,
                          dists_out_view,
                          cuvs::neighbors::filtering::none_sample_filter{});
      raft::resource::sync_stream(handle_);

      ASSERT_TRUE(cuvs::neighbors::devArrMatchKnnPair(indices_naive_dev.data(),
                                                      indices_bruteforce_dev.data(),
                                                      distances_naive_dev.data(),
                                                      distances_bruteforce_dev.data(),
                                                      ps.num_queries,
                                                      ps.k,
                                                      0.001f,
                                                      stream_,
                                                      true));
    }
  }

  void SetUp() override
  {
    database.resize(ps.num_db_vecs * ps.dim, stream_);
    search_queries.resize(ps.num_queries * ps.dim, stream_);

    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{} || std::is_same<DataT, half>{}) {
      raft::random::uniform(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0));
      raft::random::uniform(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(1), DataT(20));
      raft::random::uniformInt(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(1), DataT(20));
    }
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnBruteForceInputs<IdxT> ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

const std::vector<AnnBruteForceInputs<int64_t>> inputs = {
  // test various dims (aligned and not aligned to vector sizes)
  {1000, 10000, 1, 16, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 2, 16, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 3, 16, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 4, 16, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 5, 16, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 8, 16, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 5, 16, cuvs::distance::DistanceType::L2SqrtExpanded},
  {1000, 10000, 8, 16, cuvs::distance::DistanceType::L2SqrtExpanded},

  // test dims that do not fit into kernel shared memory limits
  {1000, 10000, 2048, 16, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 2049, 16, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 2050, 16, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 2051, 16, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 2052, 16, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 2053, 16, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 2056, 16, cuvs::distance::DistanceType::L2Expanded},

  // test fused_l2_knn
  {100, 1000, 16, 10, cuvs::distance::DistanceType::L2Expanded},
  {256, 256, 30, 10, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 10, cuvs::distance::DistanceType::L2Expanded},
  {100, 1000, 16, 50, cuvs::distance::DistanceType::L2Expanded},
  {20, 10000, 16, 10, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 50, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 32, 50, cuvs::distance::DistanceType::L2Expanded},
  {10000, 40000, 32, 30, cuvs::distance::DistanceType::L2Expanded},
  {100, 1000, 16, 10, cuvs::distance::DistanceType::L2Unexpanded},
  {1000, 10000, 16, 10, cuvs::distance::DistanceType::L2Unexpanded},
  {100, 1000, 16, 50, cuvs::distance::DistanceType::L2Unexpanded},
  {20, 10000, 16, 50, cuvs::distance::DistanceType::L2Unexpanded},
  {1000, 10000, 16, 50, cuvs::distance::DistanceType::L2Unexpanded},
  {1000, 10000, 32, 50, cuvs::distance::DistanceType::L2Unexpanded},
  {10000, 40000, 32, 30, cuvs::distance::DistanceType::L2Unexpanded},

  // test tile
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2Expanded},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2Unexpanded},
  {256, 512, 16, 8, cuvs::distance::DistanceType::InnerProduct},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2SqrtExpanded},
  {10000, 40000, 32, 30, cuvs::distance::DistanceType::L2Expanded},
  {789, 20516, 64, 256, cuvs::distance::DistanceType::L2SqrtExpanded},
  {4, 12, 32, 6, cuvs::distance::DistanceType::L2Expanded},
  {1, 40, 32, 30, cuvs::distance::DistanceType::L2Expanded},
  {1000, 500000, 128, 128, cuvs::distance::DistanceType::L2Expanded}};
}  // namespace cuvs::neighbors::brute_force
