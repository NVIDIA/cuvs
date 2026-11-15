/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_rabitq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <random>
#include <vector>

namespace cuvs::neighbors::ivf_rabitq {

struct ivf_rabitq_exact_match_inputs {
  uint32_t dim;
  uint32_t bits_per_dim;
  search_mode mode;
};

inline auto operator<<(std::ostream& os, const ivf_rabitq_exact_match_inputs& p) -> std::ostream&
{
  const char* mode = p.mode == search_mode::LUT16    ? "LUT16"
                     : p.mode == search_mode::LUT32  ? "LUT32"
                     : p.mode == search_mode::QUANT4 ? "QUANT4"
                                                     : "QUANT8";
  return os << "ivf_rabitq_exact_match_inputs {.dim = " << p.dim
            << ", .bits_per_dim = " << p.bits_per_dim << ", .mode = " << mode << "}";
}

/**
 * Queries that coincide with a list centroid.
 *
 * The dataset has as many distinct rows as there are lists, and each row is repeated many times, so
 * each list holds the copies of one row and its centroid is that row. Each query is one of the
 * distinct rows, so the probe distance ||q - c||^2 to its own list is zero up to rounding. Search
 * must return copies of the query at distance ~0, for every search mode.
 */
class ivf_rabitq_exact_match_test : public ::testing::TestWithParam<ivf_rabitq_exact_match_inputs> {
 protected:
  void run()
  {
    constexpr int64_t kDistinct = 32;
    constexpr int64_t kCopies   = 128;
    constexpr int64_t kRows     = kDistinct * kCopies;
    constexpr int64_t kK        = 10;
    constexpr uint32_t kLists   = kDistinct;

    auto ps     = GetParam();
    auto dim    = static_cast<int64_t>(ps.dim);
    auto stream = raft::resource::get_cuda_stream(handle_);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> uniform(1.0f, 2.0f);
    auto h_queries = raft::make_host_matrix<float, int64_t>(kDistinct, dim);
    for (int64_t i = 0; i < kDistinct; i++) {
      for (int64_t j = 0; j < dim; j++) {
        h_queries(i, j) = uniform(gen);
      }
    }
    auto h_dataset = raft::make_host_matrix<float, int64_t>(kRows, dim);
    for (int64_t i = 0; i < kRows; i++) {
      for (int64_t j = 0; j < dim; j++) {
        h_dataset(i, j) = h_queries(i % kDistinct, j);
      }
    }

    // The squared distance from each distinct row to the nearest other distinct row.
    std::vector<float> gap(kDistinct, std::numeric_limits<float>::max());
    for (int64_t a = 0; a < kDistinct; a++) {
      for (int64_t b = 0; b < kDistinct; b++) {
        if (a == b) { continue; }
        float d = 0;
        for (int64_t j = 0; j < dim; j++) {
          float diff = h_queries(a, j) - h_queries(b, j);
          d += diff * diff;
        }
        gap[a] = std::min(gap[a], d);
      }
    }

    auto d_dataset = raft::make_device_matrix<float, int64_t>(handle_, kRows, dim);
    auto d_queries = raft::make_device_matrix<float, int64_t>(handle_, kDistinct, dim);
    raft::copy(d_dataset.data_handle(), h_dataset.data_handle(), h_dataset.size(), stream);
    raft::copy(d_queries.data_handle(), h_queries.data_handle(), h_queries.size(), stream);

    index_params ipams;
    ipams.n_lists      = kLists;
    ipams.bits_per_dim = ps.bits_per_dim;
    auto idx           = build(handle_, ipams, raft::make_const_mdspan(d_dataset.view()));

    search_params sparams;
    sparams.n_probes = kLists;
    sparams.mode     = ps.mode;
    auto neighbors   = raft::make_device_matrix<int64_t, int64_t>(handle_, kDistinct, kK);
    auto distances   = raft::make_device_matrix<float, int64_t>(handle_, kDistinct, kK);
    search(handle_,
           sparams,
           idx,
           raft::make_const_mdspan(d_queries.view()),
           neighbors.view(),
           distances.view());

    std::vector<int64_t> h_neighbors(kDistinct * kK);
    std::vector<float> h_distances(kDistinct * kK);
    raft::copy(h_neighbors.data(), neighbors.data_handle(), h_neighbors.size(), stream);
    raft::copy(h_distances.data(), distances.data_handle(), h_distances.size(), stream);
    raft::resource::sync_stream(handle_);

    // Every row has kCopies > kK copies, so all kK neighbors must be exact copies of the query.
    int64_t n_bad = 0;
    for (int64_t q = 0; q < kDistinct; q++) {
      bool bad = false;
      for (int64_t j = 0; j < kK; j++) {
        auto neighbor = h_neighbors[q * kK + j];
        auto dist     = h_distances[q * kK + j];
        if (neighbor < 0 || neighbor >= kRows || neighbor % kDistinct != q ||
            !(std::abs(dist) <= 0.25f * gap[q])) {
          bad = true;
        }
      }
      if (bad) {
        n_bad++;
        auto msg = ::testing::Message() << ps << ": query " << q << " (gap " << gap[q] << "):";
        for (int64_t j = 0; j < kK; j++) {
          msg << " " << h_neighbors[q * kK + j] << "(row " << h_neighbors[q * kK + j] % kDistinct
              << ", " << h_distances[q * kK + j] << ")";
        }
        ADD_FAILURE() << msg;
      }
    }
    EXPECT_EQ(n_bad, 0) << ps << ": " << n_bad << " of " << kDistinct
                        << " queries did not return copies of themselves";
  }

  raft::resources handle_;
};

inline auto exact_match_inputs() -> std::vector<ivf_rabitq_exact_match_inputs>
{
  std::vector<ivf_rabitq_exact_match_inputs> xs;
  for (uint32_t dim : {64, 100, 128}) {
    for (uint32_t bits_per_dim : {1, 3}) {
      for (auto mode :
           {search_mode::LUT16, search_mode::LUT32, search_mode::QUANT4, search_mode::QUANT8}) {
        xs.push_back({dim, bits_per_dim, mode});
      }
    }
  }
  return xs;
}

TEST_P(ivf_rabitq_exact_match_test, build_search) { this->run(); }  // NOLINT
INSTANTIATE_TEST_SUITE_P(IvfRabitq,
                         ivf_rabitq_exact_match_test,
                         ::testing::ValuesIn(exact_match_inputs()));  // NOLINT

}  // namespace cuvs::neighbors::ivf_rabitq
