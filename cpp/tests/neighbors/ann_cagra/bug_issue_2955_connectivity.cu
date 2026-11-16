/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../../../src/neighbors/detail/cagra/graph_shared.cuh"
#include "../cagra_padded_build_helpers.cuh"
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra {
namespace {

enum class build_kind { ivf_pq, nn_descent, iterative };

struct connectivity_case {
  build_kind build;
  cuvs::distance::DistanceType metric;
  bool host_dataset;
  bool guarantee_connectivity;
};

auto count_components(std::vector<uint32_t> const& graph, int64_t n, int64_t degree) -> int64_t
{
  std::vector<uint32_t> parent(n);
  std::iota(parent.begin(), parent.end(), 0u);
  auto find = [&parent](uint32_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x         = parent[x];
    }
    return x;
  };
  int64_t components = n;
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < degree; j++) {
      const uint32_t v = graph[i * degree + j];
      if (v >= n) { continue; }
      const uint32_t a = find(static_cast<uint32_t>(i));
      const uint32_t b = find(v);
      if (a == b) { continue; }
      parent[std::max(a, b)] = std::min(a, b);
      components--;
    }
  }
  return components;
}

// TODO(cuvs#2955): remove with the temporary CUVS_CAGRA_BRIDGE switch.
auto bridging_disabled() -> bool
{
  const char* env = std::getenv("CUVS_CAGRA_BRIDGE");
  return env != nullptr && (std::string(env) == "0" || std::string(env) == "off");
}

}  // namespace

/**
 * Issue #2955: on well-separated clusters every kNN list stays inside its cluster, so optimize
 * leaves one weak component per cluster. The build must connect them, and the search must still
 * find the true neighbors.
 */
class cagra_connectivity_test : public ::testing::TestWithParam<connectivity_case> {
 protected:
  static constexpr int64_t kClusters    = 64;
  static constexpr int64_t kClusterSize = 160;
  static constexpr int64_t kRows        = kClusters * kClusterSize;
  static constexpr int64_t kDim         = 24;
  static constexpr int64_t kDegree      = 32;
  static constexpr int64_t kQueries     = 256;
  static constexpr int64_t kK           = 10;

  void run()
  {
    auto const& p = GetParam();
    if (!p.guarantee_connectivity && bridging_disabled()) {
      GTEST_SKIP() << "graph bridging is disabled by CUVS_CAGRA_BRIDGE";
    }
    const bool ip = p.metric == cuvs::distance::DistanceType::InnerProduct;
    auto data     = clusters(kRows, ip);
    auto queries  = clusters(kQueries, ip);
    auto stream   = raft::resource::get_cuda_stream(res);

    index_params params;
    params.metric                    = p.metric;
    params.graph_degree              = kDegree;
    params.intermediate_graph_degree = 2 * kDegree;
    params.guarantee_connectivity    = p.guarantee_connectivity;
    switch (p.build) {
      case build_kind::ivf_pq:
        params.graph_build_params =
          graph_build_params::ivf_pq_params(raft::matrix_extent<int64_t>(kRows, kDim), p.metric);
        break;
      case build_kind::nn_descent:
        params.graph_build_params =
          graph_build_params::nn_descent_params(params.intermediate_graph_degree, p.metric);
        break;
      case build_kind::iterative:
        params.graph_build_params = graph_build_params::iterative_search_params();
        break;
    }

    std::vector<uint32_t> graph(kRows * kDegree);
    if (p.host_dataset) {
      auto view = raft::make_host_matrix_view<const float, int64_t>(data.data(), kRows, kDim);
      auto index =
        cagra::build(res, params, cuvs::neighbors::make_host_standard_dataset_view(view));
      raft::update_host(graph.data(), index.graph().data_handle(), graph.size(), stream);
      raft::resource::sync_stream(res);
      EXPECT_EQ(count_components(graph, kRows, kDegree), 1);
      return;
    }

    auto d_data = raft::make_device_matrix<float, int64_t>(res, kRows, kDim);
    raft::update_device(d_data.data_handle(), data.data(), data.size(), stream);
    cuvs::neighbors::test::padded_device_matrix_for_cagra<float> padded(
      res, raft::make_const_mdspan(d_data.view()));
    auto index = cagra::build(res, params, padded.view);
    raft::update_host(graph.data(), index.graph().data_handle(), graph.size(), stream);
    raft::resource::sync_stream(res);
    EXPECT_EQ(count_components(graph, kRows, kDegree), 1);

    auto d_queries   = raft::make_device_matrix<float, int64_t>(res, kQueries, kDim);
    auto d_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, kQueries, kK);
    auto d_distances = raft::make_device_matrix<float, int64_t>(res, kQueries, kK);
    raft::update_device(d_queries.data_handle(), queries.data(), queries.size(), stream);
    search_params search;
    search.itopk_size = 64;
    cagra::search(res,
                  search,
                  index,
                  raft::make_const_mdspan(d_queries.view()),
                  d_neighbors.view(),
                  d_distances.view());
    std::vector<uint32_t> neighbors(kQueries * kK);
    raft::update_host(neighbors.data(), d_neighbors.data_handle(), neighbors.size(), stream);
    raft::resource::sync_stream(res);
    EXPECT_GE(recall(data, queries, neighbors, ip), 0.9);
  }

  /**
   * Rows i of cluster i % kClusters: a center in [-50, 50]^kDim plus Gaussian noise. Rows of a
   * cluster are ~35 apart and ~130 from other clusters. Normalized, their inner products spread
   * over ~0.02, which the fp16 distances of the IVF-PQ graph build still resolve.
   */
  auto clusters(int64_t rows, bool normalize) -> std::vector<float>
  {
    std::mt19937 centers_gen(2955);
    std::uniform_real_distribution<float> coordinate(-50.f, 50.f);
    std::vector<float> centers(kClusters * kDim);
    for (auto& c : centers) {
      c = coordinate(centers_gen);
    }
    std::normal_distribution<float> noise(0.f, 5.f);
    std::vector<float> out(rows * kDim);
    for (int64_t i = 0; i < rows; i++) {
      float* x       = out.data() + i * kDim;
      const float* c = centers.data() + (i % kClusters) * kDim;
      float norm     = 0.f;
      for (int64_t j = 0; j < kDim; j++) {
        x[j] = c[j] + noise(gen);
        norm += x[j] * x[j];
      }
      if (!normalize) { continue; }
      for (int64_t j = 0; j < kDim; j++) {
        x[j] /= std::sqrt(norm);
      }
    }
    return out;
  }

  static auto recall(std::vector<float> const& data,
                     std::vector<float> const& queries,
                     std::vector<uint32_t> const& neighbors,
                     bool ip) -> double
  {
    int64_t hits = 0;
    std::vector<std::pair<float, uint32_t>> scored(kRows);
    for (int64_t q = 0; q < kQueries; q++) {
      for (int64_t i = 0; i < kRows; i++) {
        float d = 0.f;
        for (int64_t j = 0; j < kDim; j++) {
          const float a = queries[q * kDim + j];
          const float b = data[i * kDim + j];
          d += ip ? -a * b : (a - b) * (a - b);
        }
        scored[i] = {d, static_cast<uint32_t>(i)};
      }
      std::partial_sort(scored.begin(), scored.begin() + kK, scored.end());
      for (int64_t k = 0; k < kK; k++) {
        const uint32_t id = neighbors[q * kK + k];
        hits += std::any_of(
          scored.begin(), scored.begin() + kK, [id](auto const& s) { return s.second == id; });
      }
    }
    return static_cast<double>(hits) / (kQueries * kK);
  }

  raft::device_resources res;
  std::mt19937 gen{1};
};

TEST_P(cagra_connectivity_test, connects_clusters) { run(); }

using cuvs::distance::DistanceType;
INSTANTIATE_TEST_SUITE_P(
  bug_issue_2955,
  cagra_connectivity_test,
  ::testing::Values(
    connectivity_case{build_kind::ivf_pq, DistanceType::L2Expanded, false, false},
    connectivity_case{build_kind::ivf_pq, DistanceType::L2Expanded, false, true},
    connectivity_case{build_kind::ivf_pq, DistanceType::L2Expanded, true, false},
    connectivity_case{build_kind::ivf_pq, DistanceType::InnerProduct, false, false},
    connectivity_case{build_kind::nn_descent, DistanceType::L2Expanded, false, false},
    connectivity_case{build_kind::nn_descent, DistanceType::L2Expanded, false, true},
    connectivity_case{build_kind::nn_descent, DistanceType::L2Expanded, true, false},
    connectivity_case{build_kind::nn_descent, DistanceType::InnerProduct, false, false},
    connectivity_case{build_kind::iterative, DistanceType::L2Expanded, false, false},
    connectivity_case{build_kind::iterative, DistanceType::L2Expanded, false, true}));

/**
 * Graphs of node pairs with more components than a bridging pass can sample, so joining connects
 * them: by exact neighbors between the pair roots, or by a chain beyond 2^17 roots. In every
 * seventh pair only 2i + 1 -> 2i is linked, which leaves 2i + 1 without in-edges.
 */
class cagra_bridge_fallback_test : public ::testing::Test {
 protected:
  static constexpr int64_t kDim    = 8;
  static constexpr int64_t kDegree = 4;
  static constexpr uint32_t kEmpty = std::numeric_limits<uint32_t>::max();

  void run(int64_t num_pairs, bool device_graph)
  {
    if (bridging_disabled()) { GTEST_SKIP() << "graph bridging is disabled by CUVS_CAGRA_BRIDGE"; }
    const int64_t n = 2 * num_pairs;
    std::vector<uint32_t> graph(n * kDegree, kEmpty);
    for (int64_t p = 0; p < num_pairs; p++) {
      graph[(2 * p + 1) * kDegree] = 2 * p;
      if (p % 7 != 0) { graph[2 * p * kDegree] = 2 * p + 1; }
    }
    auto stream = raft::resource::get_cuda_stream(res);
    auto data   = raft::make_device_matrix<float, int64_t>(res, n, kDim);
    raft::random::RngState rng(2955);
    raft::random::uniform(res, rng, data.data_handle(), n * kDim, -1.f, 1.f);
    const auto metric = cuvs::distance::DistanceType::L2Expanded;
    if (device_graph) {
      auto d_graph = raft::make_device_matrix<uint32_t, int64_t>(res, n, kDegree);
      raft::update_device(d_graph.data_handle(), graph.data(), graph.size(), stream);
      detail::graph::bridge_components(res, metric, data.data_handle(), true, kDim, d_graph.view());
      raft::update_host(graph.data(), d_graph.data_handle(), graph.size(), stream);
      raft::resource::sync_stream(res);
    } else {
      detail::graph::bridge_components(
        res,
        metric,
        data.data_handle(),
        true,
        kDim,
        raft::make_host_matrix_view<uint32_t, int64_t>(graph.data(), n, kDegree));
    }

    EXPECT_EQ(count_components(graph, n, kDegree), 1);
    std::vector<uint32_t> in_degree(n, 0);
    for (int64_t i = 0; i < n; i++) {
      for (int64_t j = 0; j < kDegree; j++) {
        const uint32_t v = graph[i * kDegree + j];
        if (v == kEmpty) { continue; }
        ASSERT_LT(int64_t{v}, n);
        ASSERT_NE(int64_t{v}, i);
        in_degree[v]++;
      }
    }
    EXPECT_EQ(std::count(in_degree.begin(), in_degree.end(), 0u), 0);
  }

  raft::device_resources res;
};

TEST_F(cagra_bridge_fallback_test, joins_by_exact_neighbors) { run(100000, false); }

TEST_F(cagra_bridge_fallback_test, joins_device_graph) { run(100000, true); }

TEST_F(cagra_bridge_fallback_test, chains_beyond_the_sample) { run(150000, false); }

}  // namespace cuvs::neighbors::cagra
