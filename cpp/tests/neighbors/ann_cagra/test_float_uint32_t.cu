/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

typedef AnnCagraTest<float, float, std::uint32_t> AnnCagraTestF_U32;
TEST_P(AnnCagraTestF_U32, AnnCagra_U32) { this->testCagra<uint32_t>(); }
TEST_P(AnnCagraTestF_U32, AnnCagra_I64) { this->testCagra<int64_t>(); }

typedef AnnCagraAddNodesTest<float, float, std::uint32_t> AnnCagraAddNodesTestF_U32;
TEST_P(AnnCagraAddNodesTestF_U32, AnnCagraAddNodes) { this->testCagra(); }

typedef AnnCagraFilterTest<float, float, std::uint32_t> AnnCagraFilterTestF_U32;
TEST_P(AnnCagraFilterTestF_U32, AnnCagra) { this->testCagra(); }

typedef AnnCagraIndexMergeTest<float, float, std::uint32_t> AnnCagraIndexMergeTestF_U32;
TEST_P(AnnCagraIndexMergeTestF_U32, AnnCagraIndexMerge_U32) { this->testCagra<uint32_t>(); }
TEST_P(AnnCagraIndexMergeTestF_U32, AnnCagraIndexMerge_I64) { this->testCagra<int64_t>(); }

typedef AnnCagraIndexFilteredMergeTest<float, float, std::uint32_t>
  AnnCagraIndexFilteredMergeTestF_U32;
TEST_P(AnnCagraIndexFilteredMergeTestF_U32, AnnCagraIndexFilteredMerge_U32)
{
  this->testCagra<uint32_t>();
}

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestF_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,
                        AnnCagraAddNodesTestF_U32,
                        ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(AnnCagraFilterTest,
                        AnnCagraFilterTestF_U32,
                        ::testing::ValuesIn(inputs_filtering));
INSTANTIATE_TEST_CASE_P(AnnCagraIndexMergeTest,
                        AnnCagraIndexMergeTestF_U32,
                        ::testing::ValuesIn(inputs));

INSTANTIATE_TEST_CASE_P(AnnCagraIndexFilteredMergeTest,
                        AnnCagraIndexFilteredMergeTestF_U32,
                        ::testing::ValuesIn(inputs));

typedef AnnCagraMultiPartitionTest<float, float, std::uint32_t> AnnCagraMultiPartitionTestF_U32;
TEST_P(AnnCagraMultiPartitionTestF_U32, Search) { this->testSearch(); }
TEST_P(AnnCagraMultiPartitionTestF_U32, FilteredSearch) { this->testFilteredSearch(); }

INSTANTIATE_TEST_CASE_P(AnnCagraMultiPartitionTest,
                        AnnCagraMultiPartitionTestF_U32,
                        ::testing::ValuesIn(inputs_mp));

// MULTI_KERNEL is intentionally unsupported in the multi-partition path; the call must fail rather
// than silently fall back. The rejection is invariant to dtype / metric / partition layout, so it
// is checked once with a tiny index instead of being swept across the parameterized fixture.
TEST(AnnCagraMultiPartition, MultiKernelRejected)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  constexpr int n_rows = 256, dim = 8, n_queries = 10, k = 4, num_partitions = 2;
  constexpr int part_size = n_rows / num_partitions;
  const auto metric       = cuvs::distance::DistanceType::L2Expanded;

  rmm::device_uvector<float> database(static_cast<size_t>(n_rows) * dim, stream);
  rmm::device_uvector<float> queries(static_cast<size_t>(n_queries) * dim, stream);
  raft::random::RngState r(1234ULL);
  InitDataset(handle, database.data(), n_rows, dim, metric, r);
  InitDataset(handle, queries.data(), n_queries, dim, metric, r);
  raft::resource::sync_stream(handle);

  cagra::index_params index_params;
  index_params.metric                    = metric;
  index_params.graph_degree              = 16;
  index_params.intermediate_graph_degree = 32;
  index_params.graph_build_params =
    graph_build_params::nn_descent_params(index_params.intermediate_graph_degree, metric);

  std::vector<cagra::index<float, std::uint32_t>> part_indices;
  for (int i = 0; i < num_partitions; i++) {
    auto view = raft::make_device_matrix_view<const float, int64_t>(
      database.data() + static_cast<size_t>(i) * part_size * dim, part_size, dim);
    part_indices.push_back(cagra::build(handle, index_params, view));
  }
  std::vector<const cagra::index<float, std::uint32_t>*> index_ptrs;
  for (auto& idx : part_indices) {
    index_ptrs.push_back(&idx);
  }

  const size_t out_size = static_cast<size_t>(n_queries) * k;
  rmm::device_uvector<uint32_t> partition_ids(out_size, stream);
  rmm::device_uvector<uint32_t> neighbors(out_size, stream);
  rmm::device_uvector<float> distances(out_size, stream);

  cagra::search_params search_params;
  search_params.algo = search_algo::MULTI_KERNEL;

  auto queries_view =
    raft::make_device_matrix_view<const float, int64_t>(queries.data(), n_queries, dim);
  auto part_ids_view =
    raft::make_device_matrix_view<uint32_t, int64_t>(partition_ids.data(), n_queries, k);
  auto neighbors_view =
    raft::make_device_matrix_view<uint32_t, int64_t>(neighbors.data(), n_queries, k);
  auto dists_view = raft::make_device_matrix_view<float, int64_t>(distances.data(), n_queries, k);

  EXPECT_THROW(
    cagra::search(
      handle, search_params, index_ptrs, queries_view, part_ids_view, neighbors_view, dists_view),
    std::exception);
}

}  // namespace cuvs::neighbors::cagra
