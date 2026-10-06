/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../src/faiss/faiss_cpu_binary_ivf_flat_wrapper.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cuvs::bench {
namespace {

TEST(FaissCpuBinaryIvfFlat, ExhaustiveSearchReturnsExactHammingDistances)
{
  std::vector<std::uint8_t> dataset(256);
  for (std::size_t i = 0; i < dataset.size(); ++i) {
    dataset[i] = static_cast<std::uint8_t>(i);
  }

  faiss_cpu_binary_ivf_flat::build_param build_param;
  build_param.nlist                   = 4;
  build_param.niter                   = 5;
  build_param.build_threads           = 2;
  build_param.max_points_per_centroid = 32;

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

}  // namespace
}  // namespace cuvs::bench
