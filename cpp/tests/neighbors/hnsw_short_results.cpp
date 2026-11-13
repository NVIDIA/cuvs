/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/hnsw.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <gtest/gtest.h>
#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <unistd.h>

namespace cuvs::neighbors::hnsw {

namespace {

// Search an index written by stock hnswlib, which needs no GPU, and compare with exact search over
// the elements that were not deleted. Slots after the available elements have to hold the sentinel.
void test_short_results(HnswHierarchy hierarchy, int k, int n_deleted)
{
  constexpr int n_rows    = 8;
  constexpr int dim       = 4;
  constexpr int n_queries = 3;

  std::mt19937 rng(1234);
  auto const random_vectors = [&rng](int n) {
    std::vector<float> values(n * dim);
    std::generate(values.begin(), values.end(), [&rng] { return static_cast<float>(rng() % 100); });
    return values;
  };
  auto const dataset = random_vectors(n_rows);
  auto const queries = random_vectors(n_queries);

  auto const path = std::filesystem::temp_directory_path() /
                    ("cuvs_hnsw_short_results_" + std::to_string(::getpid()) + ".bin");
  {
    hnswlib::L2Space<float, float> space(dim);
    hnswlib::HierarchicalNSW<float> stock_index(&space, n_rows);
    for (int i = 0; i < n_rows; i++) {
      stock_index.addPoint(dataset.data() + i * dim, i);
    }
    for (int i = 0; i < n_deleted; i++) {
      stock_index.markDelete(i);
    }
    stock_index.saveIndex(path.string());
  }

  raft::resources res;
  index_params params;
  params.hierarchy = hierarchy;
  index<float>* loaded{nullptr};
  deserialize(res, params, path.string(), dim, cuvs::distance::DistanceType::L2Expanded, &loaded);
  std::unique_ptr<index<float>> idx(loaded);
  std::filesystem::remove(path);

  std::vector<uint64_t> neighbors(n_queries * k);
  std::vector<float> distances(n_queries * k);
  search_params query_params;
  query_params.ef = k;
  search(res,
         query_params,
         *idx,
         raft::make_host_matrix_view<const float, int64_t>(queries.data(), n_queries, dim),
         raft::make_host_matrix_view<uint64_t, int64_t>(neighbors.data(), n_queries, k),
         raft::make_host_matrix_view<float, int64_t>(distances.data(), n_queries, k));

  for (int q = 0; q < n_queries; q++) {
    std::vector<std::tuple<float, uint64_t>> expected;
    for (int i = n_deleted; i < n_rows; i++) {
      float distance = 0;
      for (int d = 0; d < dim; d++) {
        auto const diff = queries[q * dim + d] - dataset[i * dim + d];
        distance += diff * diff;
      }
      expected.emplace_back(distance, i);
    }
    std::sort(expected.begin(), expected.end());

    for (int j = 0; j < k; j++) {
      SCOPED_TRACE(::testing::Message() << "query " << q << ", slot " << j);
      auto const neighbor = neighbors[q * k + j];
      auto const distance = distances[q * k + j];
      if (j < static_cast<int>(expected.size())) {
        EXPECT_FLOAT_EQ(distance, std::get<0>(expected[j]));
        EXPECT_EQ(neighbor, std::get<1>(expected[j]));
      } else {
        EXPECT_EQ(neighbor, std::numeric_limits<uint64_t>::max());
        EXPECT_EQ(distance, std::numeric_limits<float>::max());
      }
    }
  }
}

}  // namespace

TEST(HNSWShortResults, KLargerThanIndex)
{
  test_short_results(HnswHierarchy::CPU, /*k=*/12, /*n_deleted=*/0);
}

TEST(HNSWShortResults, KLargerThanIndexBaseLayerOnly)
{
  test_short_results(HnswHierarchy::NONE, /*k=*/12, /*n_deleted=*/0);
}

// k does not exceed the index size here, so comparing k to the size cannot detect this case.
TEST(HNSWShortResults, FewerReachableThanK)
{
  test_short_results(HnswHierarchy::CPU, /*k=*/6, /*n_deleted=*/4);
}

TEST(HNSWShortResults, FullResultsAreUnchanged)
{
  test_short_results(HnswHierarchy::CPU, /*k=*/8, /*n_deleted=*/0);
}

}  // namespace cuvs::neighbors::hnsw
