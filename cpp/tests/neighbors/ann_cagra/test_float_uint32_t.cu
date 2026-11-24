/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../../../src/neighbors/detail/cagra/cagra_ace.cuh"
#include "../ann_cagra.cuh"

#include <algorithm>
#include <limits>
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

namespace {
// Runs a filtered multi-partition MULTI_CTA search where every partition keeps only ~4% of its rows
// (still more than k each) and reports the number of queries that returned fewer than k neighbors
// and the recall against an exact search over the kept rows.
void search_with_selective_filter(float filtering_rate, int& short_queries, double& recall)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  constexpr int num_partitions = 4, part_size = 500, dim = 128, n_queries = 500, k = 10;
  // Keep one row in `keep_every` (~4% of each partition, i.e. ~21 rows per partition).
  constexpr int keep_every = 24;
  constexpr int n_rows     = num_partitions * part_size;

  rmm::device_uvector<float> database(static_cast<size_t>(n_rows) * dim, stream);
  rmm::device_uvector<float> queries(static_cast<size_t>(n_queries) * dim, stream);
  raft::random::RngState r(1234ULL);
  InitDataset(handle, database.data(), n_rows, dim, cuvs::distance::DistanceType::L2Expanded, r);
  InitDataset(handle, queries.data(), n_queries, dim, cuvs::distance::DistanceType::L2Expanded, r);
  raft::resource::sync_stream(handle);

  cagra::index_params index_params;
  index_params.graph_degree              = 32;
  index_params.intermediate_graph_degree = 64;
  index_params.graph_build_params        = graph_build_params::nn_descent_params(
    index_params.intermediate_graph_degree, index_params.metric);

  std::vector<cagra::index<float, std::uint32_t>> part_indices;
  // An index only holds a view, so any padded copy has to outlive it.
  std::vector<cuvs::neighbors::test::padded_device_matrix_for_cagra<float>> part_padded;
  part_padded.reserve(num_partitions);
  std::vector<cuvs::core::bitset<uint32_t, int64_t>> part_bitsets;
  part_bitsets.reserve(num_partitions);
  std::vector<cuvs::core::bitset_view<uint32_t, int64_t>> partition_bitsets;
  std::vector<int64_t> removed_local;
  for (int64_t i = 0; i < part_size; i++) {
    if (i % keep_every != 0) { removed_local.push_back(i); }
  }
  auto removed =
    raft::make_device_vector<int64_t, int64_t>(handle, static_cast<int64_t>(removed_local.size()));
  raft::update_device(removed.data_handle(), removed_local.data(), removed_local.size(), stream);
  for (int p = 0; p < num_partitions; p++) {
    auto view = raft::make_device_matrix_view<const float, int64_t>(
      database.data() + static_cast<size_t>(p) * part_size * dim, part_size, dim);
    part_padded.emplace_back(handle, view);
    auto const& padded = part_padded.back().view;
    part_indices.push_back(cagra::build(handle, index_params, padded));
    auto& part_index = part_indices.back();
    part_index       = cagra::update_dataset(handle, std::move(part_index), padded);
    part_bitsets.emplace_back(handle, removed.view(), part_size);
    partition_bitsets.push_back(part_bitsets.back().view());
  }
  std::vector<const cagra::index<float, std::uint32_t>*> index_ptrs;
  for (auto& idx : part_indices) {
    index_ptrs.push_back(&idx);
  }

  const size_t out_size = static_cast<size_t>(n_queries) * k;
  rmm::device_uvector<uint32_t> partition_ids_dev(out_size, stream);
  rmm::device_uvector<uint32_t> neighbors_dev(out_size, stream);
  rmm::device_uvector<float> distances_dev(out_size, stream);

  // A small itopk_size, as used for small k, on the MULTI_CTA algorithm that AUTO picks for a
  // handful of queries over a handful of partitions.
  cagra::search_params search_params;
  search_params.algo           = search_algo::MULTI_CTA;
  search_params.itopk_size     = k;
  search_params.filtering_rate = filtering_rate;

  cagra::search(
    handle,
    search_params,
    index_ptrs,
    raft::make_device_matrix_view<const float, int64_t>(queries.data(), n_queries, dim),
    raft::make_device_matrix_view<uint32_t, int64_t>(partition_ids_dev.data(), n_queries, k),
    raft::make_device_matrix_view<uint32_t, int64_t>(neighbors_dev.data(), n_queries, k),
    raft::make_device_matrix_view<float, int64_t>(distances_dev.data(), n_queries, k),
    partition_bitsets);

  std::vector<uint32_t> partition_ids(out_size);
  std::vector<uint32_t> neighbors(out_size);
  std::vector<float> distances(out_size);
  std::vector<float> host_database(static_cast<size_t>(n_rows) * dim);
  std::vector<float> host_queries(static_cast<size_t>(n_queries) * dim);
  raft::update_host(partition_ids.data(), partition_ids_dev.data(), out_size, stream);
  raft::update_host(neighbors.data(), neighbors_dev.data(), out_size, stream);
  raft::update_host(distances.data(), distances_dev.data(), out_size, stream);
  raft::update_host(host_database.data(), database.data(), host_database.size(), stream);
  raft::update_host(host_queries.data(), queries.data(), host_queries.size(), stream);
  raft::resource::sync_stream(handle);

  // Exact top-k over the rows that pass the filter, for recall.
  std::vector<int64_t> kept_rows;
  for (int64_t g = 0; g < n_rows; g++) {
    if ((g % part_size) % keep_every == 0) { kept_rows.push_back(g); }
  }
  ASSERT_GT(kept_rows.size(), static_cast<size_t>(k));

  short_queries  = 0;
  size_t matches = 0;
  for (int q = 0; q < n_queries; q++) {
    std::vector<std::pair<float, int64_t>> exact;
    for (auto g : kept_rows) {
      float d = 0;
      for (int j = 0; j < dim; j++) {
        const float diff = host_queries[q * dim + j] - host_database[g * dim + j];
        d += diff * diff;
      }
      exact.emplace_back(d, g);
    }
    std::partial_sort(exact.begin(), exact.begin() + k, exact.end());
    std::vector<int64_t> expected;
    for (int j = 0; j < k; j++) {
      expected.push_back(exact[j].second);
    }

    int valid = 0;
    for (int j = 0; j < k; j++) {
      const size_t i = static_cast<size_t>(q) * k + j;
      if (distances[i] == std::numeric_limits<float>::max()) { continue; }
      ASSERT_LT(partition_ids[i], static_cast<uint32_t>(num_partitions));
      ASSERT_LT(neighbors[i], static_cast<uint32_t>(part_size));
      ASSERT_EQ(neighbors[i] % keep_every, 0u) << "filtered-out row returned";
      valid++;
      const int64_t g = static_cast<int64_t>(partition_ids[i]) * part_size + neighbors[i];
      if (std::find(expected.begin(), expected.end(), g) != expected.end()) { matches++; }
    }
    if (valid < k) { short_queries++; }
  }
  recall = static_cast<double>(matches) / out_size;
}
}  // namespace

// A selective per-partition filter must not leave the search short of neighbors. Single-partition
// search derives the filtering rate from the bitset and enlarges itopk_size accordingly; the
// multi-partition search must do the same, or MULTI_CTA can return fewer than k valid neighbors
// even though every partition holds more than k rows that pass the filter.
TEST(AnnCagraMultiPartition, SelectiveFilterReturnsKNeighbors)
{
  int short_queries = 0;
  double recall     = 0;
  search_with_selective_filter(-1.0f, short_queries, recall);
  EXPECT_EQ(short_queries, 0) << "queries returning fewer than k neighbors";
  EXPECT_GE(recall, 0.95);
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
