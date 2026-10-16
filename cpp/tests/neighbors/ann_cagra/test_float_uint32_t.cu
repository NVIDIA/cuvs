/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../../../src/neighbors/detail/cagra/cagra_ace.cuh"
#include "../ann_cagra.cuh"

#include <utility>
#include <vector>

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

// Builds one CAGRA index per {metric, graph_degree} spec over a shared random dataset, then asserts
// a multi-partition search over them throws. Shared by the rejection tests below, which each
// violate one "all partitions must be uniform / supported" precondition. The rejections are
// invariant to dtype / layout, so they are checked once here instead of swept across the fixture.
namespace {
void expect_multi_partition_search_throws(
  const std::vector<std::pair<cuvs::distance::DistanceType, int>>& partition_specs,
  const cagra::search_params& search_params)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  constexpr int n_rows = 256, dim = 8, n_queries = 10, k = 4;
  const int num_partitions = static_cast<int>(partition_specs.size());
  const int part_size      = n_rows / num_partitions;

  rmm::device_uvector<float> database(static_cast<size_t>(n_rows) * dim, stream);
  rmm::device_uvector<float> queries(static_cast<size_t>(n_queries) * dim, stream);
  raft::random::RngState r(1234ULL);
  InitDataset(handle, database.data(), n_rows, dim, cuvs::distance::DistanceType::L2Expanded, r);
  InitDataset(handle, queries.data(), n_queries, dim, cuvs::distance::DistanceType::L2Expanded, r);
  raft::resource::sync_stream(handle);

  std::vector<cagra::index<float, std::uint32_t>> part_indices;
  // An index only holds a view, so any padded copy has to outlive it.
  std::vector<cuvs::neighbors::test::padded_device_matrix_for_cagra<float>> part_padded;
  part_padded.reserve(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    const auto [metric, graph_degree] = partition_specs[i];
    cagra::index_params index_params;
    index_params.metric                    = metric;
    index_params.graph_degree              = graph_degree;
    index_params.intermediate_graph_degree = graph_degree * 2;
    index_params.graph_build_params =
      graph_build_params::nn_descent_params(index_params.intermediate_graph_degree, metric);
    auto view = raft::make_device_matrix_view<const float, int64_t>(
      database.data() + static_cast<size_t>(i) * part_size * dim, part_size, dim);
    part_padded.emplace_back(handle, view);
    auto const& padded = part_padded.back().view;
    part_indices.push_back(cagra::build(handle, index_params, padded));
    auto& part_index = part_indices.back();
    part_index       = cagra::update_dataset(handle, std::move(part_index), padded);
  }
  std::vector<const cagra::index<float, std::uint32_t>*> index_ptrs;
  for (auto& idx : part_indices) {
    index_ptrs.push_back(&idx);
  }

  const size_t out_size = static_cast<size_t>(n_queries) * k;
  rmm::device_uvector<uint32_t> partition_ids(out_size, stream);
  rmm::device_uvector<uint32_t> neighbors(out_size, stream);
  rmm::device_uvector<float> distances(out_size, stream);

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
}  // namespace

// MULTI_KERNEL is intentionally unsupported in the multi-partition path; the call must fail rather
// than silently fall back.
TEST(AnnCagraMultiPartition, MultiKernelRejected)
{
  cagra::search_params search_params;
  search_params.algo = search_algo::MULTI_KERNEL;
  expect_multi_partition_search_throws({{cuvs::distance::DistanceType::L2Expanded, 16},
                                        {cuvs::distance::DistanceType::L2Expanded, 16}},
                                       search_params);
}

// The shared plan descriptor and the cross-partition select_k direction are derived from
// indices[0], so all partitions must share one metric; a mismatch must be rejected.
TEST(AnnCagraMultiPartition, MixedMetricRejected)
{
  expect_multi_partition_search_throws({{cuvs::distance::DistanceType::L2Expanded, 16},
                                        {cuvs::distance::DistanceType::InnerProduct, 16}},
                                       cagra::search_params{});
}

// The shared plan descriptor is sized from indices[0]'s graph degree, so all partitions must share
// one graph degree; a mismatch must be rejected.
TEST(AnnCagraMultiPartition, MixedGraphDegreeRejected)
{
  expect_multi_partition_search_throws({{cuvs::distance::DistanceType::L2Expanded, 16},
                                        {cuvs::distance::DistanceType::L2Expanded, 32}},
                                       cagra::search_params{});
}

}  // namespace cuvs::neighbors::cagra

namespace cuvs::neighbors::cagra::detail {

TEST(AceBudget, RejectsSpillOffsetOverflow)
{
  constexpr uint64_t limit = std::numeric_limits<uint32_t>::max();
  EXPECT_TRUE(ace_spill_count_fits<uint32_t>(0));
  EXPECT_TRUE(ace_spill_count_fits<uint32_t>(limit));
  EXPECT_FALSE(ace_spill_count_fits<uint32_t>(limit + 1));
  // Original IDs fit, but three spill copies per row do not.
  EXPECT_FALSE(ace_spill_count_fits<uint32_t>(1'500'000'000ULL * 3));
}

TEST(AceBudget, SubsamplesFiniteAndInfiniteQuantileTies)
{
  for (const float score : {0.0f, std::numeric_limits<float>::infinity()}) {
    auto gate = ace_target_alpha_threshold(std::vector<float>(300, score), 100, 3, 0.25);
    EXPECT_EQ(gate.threshold, score);
    EXPECT_DOUBLE_EQ(gate.tie_probability, 1.0 / 12.0);
    size_t accepted = 0;
    for (size_t row = 0; row < 100'000; ++row) {
      for (size_t partition = 1; partition <= 3; ++partition) {
        accepted += gate.accepts(score, row, partition);
      }
    }
    EXPECT_NEAR(static_cast<double>(accepted) / 100'000, 0.25, 0.01);
  }
}

TEST(AceBudget, AccountsForScoresBelowThreshold)
{
  auto gate = ace_target_alpha_threshold({0, 1, 1, 1, 2, 2}, 2, 3, 1.0);
  EXPECT_EQ(gate.threshold, 1);
  EXPECT_DOUBLE_EQ(gate.tie_probability, 1.0 / 3.0);
  EXPECT_TRUE(gate.accepts(0, 5, 1));
  EXPECT_FALSE(gate.accepts(2, 5, 1));
}

TEST(AceBudget, ZeroAndMaximumBudgets)
{
  const std::vector<float> scores{0, 1, std::numeric_limits<float>::infinity()};
  auto zero = ace_target_alpha_threshold(scores, 1, 3, 0.0);
  auto full = ace_target_alpha_threshold(scores, 1, 3, 3.0);
  for (const float score : scores) {
    EXPECT_FALSE(zero.accepts(score, 0, 1));
    EXPECT_TRUE(full.accepts(score, 0, 1));
  }
}

TEST(AceBudget, LabelingBatchRespectsHostAndGpuBudgetsAbove8192Partitions)
{
  constexpr size_t rows = 2'000'000, partitions = 16384, dim = 128;
  constexpr size_t samples    = 1'638'400;
  constexpr size_t gpu_fixed  = sizeof(float) * (samples + partitions) * dim;
  constexpr size_t pairs      = sizeof(float) * partitions * partitions;
  constexpr size_t host_fixed = sizeof(float) * (samples * dim + 3 * samples) + pairs;
  constexpr size_t per_row    = sizeof(float) * (dim + partitions);
  constexpr size_t ample      = 8ULL << 30;
  EXPECT_EQ(ace_labeling_sample_count(1000, 20), 1000);
  EXPECT_EQ(ace_labeling_batch_size(rows, dim, partitions, 1.0, ample, ample), 32768);
  // This GPU budget fits the pair matrix but only 20000 assignment rows.
  EXPECT_EQ(ace_labeling_batch_size(rows, dim, partitions, 1.0, gpu_fixed + 20000 * per_row, ample),
            20000);
  EXPECT_EQ(ace_labeling_batch_size(rows, dim, partitions, 1.0, ample, host_fixed + 7 * per_row),
            7);
  EXPECT_EQ(ace_labeling_batch_size(rows, dim, partitions, 1.0, ample, host_fixed + per_row - 1),
            0);
  EXPECT_EQ(ace_labeling_batch_size(rows, dim, partitions, 1.0, gpu_fixed + pairs - 1, ample), 0);
  // Core-only builds need no pair matrix or score vectors.
  EXPECT_GT(ace_labeling_batch_size(rows, dim, partitions, 0.0, gpu_fixed + pairs - 1, ample), 0);
  EXPECT_EQ(ace_labeling_batch_size(10, dim, 2, 1.0, ample, ample), 10);
  EXPECT_EQ(ace_labeling_batch_size(rows, dim, partitions, 1.0, 0, ample), 0);
}

TEST(AceBudget, LabelingMemoryRejectsImpossibleDimensions)
{
  EXPECT_EQ(ace_labeling_batch_size(1'000'000,
                                    std::numeric_limits<size_t>::max(),
                                    16384,
                                    1.0,
                                    std::numeric_limits<size_t>::max(),
                                    std::numeric_limits<size_t>::max()),
            0);
}

}  // namespace cuvs::neighbors::cagra::detail
