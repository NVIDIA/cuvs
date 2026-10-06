/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../src/common/dataset.hpp"
#include "../src/common/training_sample.hpp"
#include "../src/faiss/faiss_cpu_binary_ivf_flat_wrapper.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cuvs::bench {
namespace {

template <typename T, std::size_t Size>
void write_blob(const std::filesystem::path& path, const std::array<T, Size>& values)
{
  const std::array<std::uint32_t, 2> shape{1, Size};
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(shape.data()), sizeof(shape));
  output.write(reinterpret_cast<const char*>(values.data()), sizeof(values));
}

TEST(FaissCpuBinaryIvfFlat, ExhaustiveSearchReturnsExactHammingDistances)
{
  std::vector<std::uint8_t> dataset(256);
  for (std::size_t i = 0; i < dataset.size(); ++i) {
    dataset[i] = static_cast<std::uint8_t>(i);
  }

  faiss_cpu_binary_ivf_flat::build_param build_param;
  build_param.nlist                         = 4;
  build_param.niter                         = 5;
  build_param.build_threads                 = 2;
  build_param.max_train_points_per_centroid = 32;

  faiss_cpu_binary_ivf_flat index(Metric::kBitwiseHamming, 1, build_param);
  index.build(dataset.data(), dataset.size());

  const std::vector<std::uint8_t> queries{0x00, 0xff};
  constexpr int k = 8;
  std::vector<algo_base::index_type> neighbors(queries.size() * k);
  std::vector<float> distances(queries.size() * k);

  faiss_cpu_binary_ivf_flat::search_param search_param;
  search_param.nprobe    = build_param.nlist;
  search_param.k         = k;
  search_param.n_queries = queries.size();
  index.set_search_param(search_param, nullptr);
  index.search(queries.data(), queries.size(), k, neighbors.data(), distances.data());

  for (std::size_t query = 0; query < queries.size(); ++query) {
    EXPECT_EQ(distances[query * k], 0.0f);
    EXPECT_EQ(dataset[neighbors[query * k]], queries[query]);
    for (int rank = 0; rank < k; ++rank) {
      const auto offset      = query * k + rank;
      const auto neighbor_id = neighbors[offset];
      ASSERT_GE(neighbor_id, 0);
      ASSERT_LT(static_cast<std::size_t>(neighbor_id), dataset.size());
      EXPECT_EQ(distances[offset],
                static_cast<float>(
                  std::popcount(static_cast<unsigned int>(queries[query] ^ dataset[neighbor_id]))));
      if (rank > 0) { EXPECT_LE(distances[offset - 1], distances[offset]); }
    }
  }
}

TEST(FaissCpuBinaryIvfFlat, ValidatesMetricAndProbeCount)
{
  faiss_cpu_binary_ivf_flat::build_param build_param;
  build_param.nlist = 2;
  EXPECT_THROW(faiss_cpu_binary_ivf_flat(Metric::kEuclidean, 1, build_param),
               std::invalid_argument);

  faiss_cpu_binary_ivf_flat index(Metric::kBitwiseHamming, 1, build_param);
  faiss_cpu_binary_ivf_flat::search_param search_param;
  search_param.nprobe    = 3;
  search_param.k         = 1;
  search_param.n_queries = 1;
  EXPECT_THROW(index.set_search_param(search_param, nullptr), std::invalid_argument);
}

TEST(TrainingSample, IsDeterministicAndSupportsAllRows)
{
  const auto first  = uniform_sample_ids(100, 16, 42);
  const auto second = uniform_sample_ids(100, 16, 42);
  EXPECT_EQ(first, second);
  EXPECT_EQ(first.size(), 16);
  EXPECT_TRUE(std::is_sorted(first.begin(), first.end()));
  EXPECT_EQ(training_sample_size(100, 4, 1'000), 100);
}

TEST(TieAwareRecall, AcceptsDistinctNeighborsAtTheBoundaryDistance)
{
  const auto fixture = std::filesystem::path(::testing::TempDir()) / "binary-ivf-tie-aware";
  std::filesystem::create_directories(fixture);
  const auto neighbors_path = fixture / "neighbors.ibin";
  const auto distances_path = fixture / "distances.fbin";
  write_blob(neighbors_path, std::array<std::int32_t, 4>{0, 1, 2, 3});
  write_blob(distances_path, std::array<float, 4>{0.0f, 1.0f, 1.0f, 2.0f});

  std::optional<blob<std::uint32_t>> filter_bitset;
  const ground_truth_map<std::int32_t> ground_truth{
    neighbors_path.string(), 1, distances_path.string(), filter_bitset};

  const std::array<algo_base::index_type, 3> candidates{0, 8, 9};
  const std::array<float, 3> candidate_distances{0.0f, 1.0f, 1.0f};
  EXPECT_EQ(ground_truth.count_matches(0, candidates.data(), candidates.size()),
            std::make_pair(std::size_t{1}, std::size_t{3}));
  EXPECT_EQ(ground_truth.count_tie_aware_matches(
              0, candidates.data(), candidate_distances.data(), candidates.size()),
            std::make_pair(std::size_t{3}, std::size_t{3}));

  const std::array<algo_base::index_type, 3> duplicate_candidates{0, 8, 8};
  EXPECT_EQ(
    ground_truth.count_tie_aware_matches(
      0, duplicate_candidates.data(), candidate_distances.data(), duplicate_candidates.size()),
    std::make_pair(std::size_t{2}, std::size_t{3}));
}

}  // namespace
}  // namespace cuvs::bench
