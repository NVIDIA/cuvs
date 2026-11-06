/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace cuvs::neighbors::cagra {
namespace {

// Independent set-based reference, including the empty-set convention.
template <typename T>
float jaccard_reference(const T* a, const T* b, int64_t dim)
{
  int intersection = 0;
  int set_union    = 0;
  for (int64_t i = 0; i < dim; ++i) {
    bool in_a = a[i] != T(0);
    bool in_b = b[i] != T(0);
    intersection += in_a && in_b;
    set_union += in_a || in_b;
  }
  return set_union ? 1.0f - float(intersection) / set_union : 0.0f;
}

template <typename T>
auto binary_data(int64_t rows, int64_t dim, uint32_t seed = 1158)
{
  auto data = raft::make_host_matrix<T, int64_t>(rows, dim);
  std::mt19937 rng(seed);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < dim; ++j) {
      // Two empty rows, two full rows, then varying cardinalities.
      data(i, j) = T(i < 2 ? 0 : (i < 4 ? 1 : rng() % 8 < static_cast<uint64_t>(1 + i % 6)));
    }
  }
  return data;
}

template <typename T>
class CagraJaccardTest : public ::testing::Test {};
using JaccardTypes = ::testing::Types<float, half, int8_t, uint8_t>;
TYPED_TEST_SUITE(CagraJaccardTest, JaccardTypes);

TYPED_TEST(CagraJaccardTest, SearchCompleteGraph)
{
  using T = TypeParam;
  raft::device_resources res;
  constexpr int64_t rows          = 64;
  constexpr int64_t queries_count = 5;
  constexpr int64_t k             = 32;
  for (int64_t dim : {1, 17, 65, 129, 257, 513, 1031}) {
    SCOPED_TRACE(dim);
    auto data = binary_data<T>(rows, dim);
    auto padded =
      cuvs::neighbors::make_device_padded_dataset(res, raft::make_const_mdspan(data.view()));
    auto graph = raft::make_host_matrix<uint32_t, int64_t>(rows, rows);
    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t j = 0; j < rows; ++j) {
        graph(i, j) = j;
      }
    }
    device_padded_index<T> index(res,
                                 cuvs::distance::DistanceType::JaccardExpanded,
                                 padded->view(),
                                 raft::make_const_mdspan(graph.view()));
    auto queries = raft::make_device_matrix<T, int64_t>(res, queries_count, dim);
    raft::copy(queries.data_handle(), data.data_handle(), queries.size(), res.get_stream());
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, queries_count, k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, queries_count, k);
    std::vector<uint32_t> ids(queries_count * k);
    std::vector<float> actual(queries_count * k);

    for (auto algo : {search_algo::SINGLE_CTA, search_algo::MULTI_CTA, search_algo::MULTI_KERNEL}) {
      SCOPED_TRACE(static_cast<int>(algo));
      search_params params;
      params.algo           = algo;
      params.itopk_size     = 64;
      params.min_iterations = 2;
      params.max_iterations = 8;
      search(res,
             params,
             index,
             raft::make_const_mdspan(queries.view()),
             neighbors.view(),
             distances.view());
      raft::copy(ids.data(), neighbors.data_handle(), ids.size(), res.get_stream());
      raft::copy(actual.data(), distances.data_handle(), actual.size(), res.get_stream());
      res.sync_stream();
      for (int64_t q = 0; q < queries_count; ++q) {
        std::vector<float> expected(rows);
        for (int64_t i = 0; i < rows; ++i) {
          expected[i] =
            jaccard_reference(data.data_handle() + q * dim, data.data_handle() + i * dim, dim);
        }
        std::sort(expected.begin(), expected.end());
        for (int64_t j = 0; j < k; ++j) {
          auto pos = q * k + j;
          ASSERT_LT(ids[pos], rows);
          EXPECT_NEAR(actual[pos], expected[j], 1e-6);
          EXPECT_NEAR(actual[pos],
                      jaccard_reference(
                        data.data_handle() + q * dim, data.data_handle() + ids[pos] * dim, dim),
                      1e-6);
        }
      }
    }
  }
}

TYPED_TEST(CagraJaccardTest, DatasetReplacementRefreshesNorms)
{
  using T = TypeParam;
  raft::device_resources res;
  constexpr int64_t rows = 64;
  constexpr int64_t dim  = 129;
  auto original          = raft::make_host_matrix<T, int64_t>(rows, dim);
  auto replacement       = raft::make_host_matrix<T, int64_t>(rows, dim);
  auto graph             = raft::make_host_matrix<uint32_t, int64_t>(rows, rows);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < dim; ++j) {
      original(i, j)    = T(1);
      replacement(i, j) = T(j == 0);
    }
    for (int64_t j = 0; j < rows; ++j) {
      graph(i, j) = j;
    }
  }
  auto original_device =
    cuvs::neighbors::make_device_padded_dataset(res, raft::make_const_mdspan(original.view()));
  auto replacement_device =
    cuvs::neighbors::make_device_padded_dataset(res, raft::make_const_mdspan(replacement.view()));
  device_padded_index<T> index(res,
                               cuvs::distance::DistanceType::JaccardExpanded,
                               original_device->view(),
                               raft::make_const_mdspan(graph.view()));
  auto queries   = raft::make_device_matrix<T, int64_t>(res, 1, dim);
  auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, 1, 1);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 1, 1);
  raft::copy(queries.data_handle(), replacement.data_handle(), dim, res.get_stream());
  search_params params;
  params.algo       = search_algo::SINGLE_CTA;
  params.itopk_size = 64;
  auto check        = [&](auto const& current, float expected) {
    search(res,
           params,
           current,
           raft::make_const_mdspan(queries.view()),
           neighbors.view(),
           distances.view());
    float actual;
    raft::copy(&actual, distances.data_handle(), 1, res.get_stream());
    res.sync_stream();
    EXPECT_NEAR(actual, expected, 1e-6);
  };
  check(index, 1.0f - 1.0f / dim);
  device_padded_index<T> replaced(res, std::move(index), replacement_device->view());
  check(replaced, 0.0f);
}

TYPED_TEST(CagraJaccardTest, NNDescentDistancesAndBuild)
{
  using T = TypeParam;
  raft::device_resources res;
  constexpr int64_t rows = 512;
  constexpr int64_t dim  = 65;
  auto data              = binary_data<T>(rows, dim);
  auto device_data       = raft::make_device_matrix<T, int64_t>(res, rows, dim);
  raft::copy(device_data.data_handle(), data.data_handle(), data.size(), res.get_stream());
  nn_descent::index_params params;
  params.metric                    = cuvs::distance::DistanceType::JaccardExpanded;
  params.graph_degree              = 32;
  params.intermediate_graph_degree = 64;
  params.return_distances          = true;
  // Exercise both the SIMT and tensor-core dot-product paths, with host and device input.
  for (auto dtype : {nn_descent::DIST_COMP_DTYPE::FP32, nn_descent::DIST_COMP_DTYPE::FP16}) {
    params.dist_comp_dtype = dtype;
    for (bool host : {false, true}) {
      auto index = host
                     ? nn_descent::build(res, params, raft::make_const_mdspan(data.view()))
                     : nn_descent::build(res, params, raft::make_const_mdspan(device_data.view()));
      ASSERT_TRUE(index.distances().has_value());
      std::vector<float> distances(rows * params.graph_degree);
      raft::copy(
        distances.data(), index.distances()->data_handle(), distances.size(), res.get_stream());
      res.sync_stream();
      for (int64_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < params.graph_degree; ++j) {
          auto neighbor = index.graph()(i, j);
          ASSERT_LT(neighbor, rows);
          EXPECT_NEAR(distances[i * params.graph_degree + j],
                      jaccard_reference(
                        data.data_handle() + i * dim, data.data_handle() + neighbor * dim, dim),
                      1e-6);
        }
      }
    }
  }

  // Build CAGRA with an explicit NN-descent configuration.
  auto padded =
    cuvs::neighbors::make_device_padded_dataset(res, raft::make_const_mdspan(data.view()));
  index_params cagra_params;
  cagra_params.metric                    = params.metric;
  cagra_params.intermediate_graph_degree = 64;
  cagra_params.graph_degree              = 32;
  cagra_params.graph_build_params        = graph_build_params::nn_descent_params(
    cagra_params.intermediate_graph_degree, cagra_params.metric);
  auto index = build(res, cagra_params, padded->view());
  EXPECT_EQ(index.metric(), params.metric);
  auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, 4, 1);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 4, 1);
  search(res,
         search_params{},
         index,
         raft::make_device_matrix_view<const T, int64_t>(device_data.data_handle(), 4, dim),
         neighbors.view(),
         distances.view());
  std::vector<uint32_t> ids(4);
  std::vector<float> actual(4);
  raft::copy(ids.data(), neighbors.data_handle(), ids.size(), res.get_stream());
  raft::copy(actual.data(), distances.data_handle(), actual.size(), res.get_stream());
  res.sync_stream();
  for (int64_t q = 0; q < 4; ++q) {
    ASSERT_LT(ids[q], rows);
    EXPECT_NEAR(
      actual[q],
      jaccard_reference(data.data_handle() + q * dim, data.data_handle() + ids[q] * dim, dim),
      1e-6);
  }
}

TYPED_TEST(CagraJaccardTest, IterativeBuildAndSearch)
{
  using T = TypeParam;
  raft::device_resources res;
  constexpr int64_t rows = 512;
  constexpr int64_t nq   = 32;
  constexpr int64_t k    = 10;
  for (int64_t dim : {17, 65, 128, 129}) {
    SCOPED_TRACE(dim);
    auto data    = binary_data<T>(rows, dim);
    auto queries = binary_data<T>(nq, dim, 1159);
    auto padded =
      cuvs::neighbors::make_device_padded_dataset(res, raft::make_const_mdspan(data.view()));
    index_params params;
    params.metric                    = cuvs::distance::DistanceType::JaccardExpanded;
    params.graph_degree              = 32;
    params.intermediate_graph_degree = 64;
    params.graph_build_params        = graph_build_params::iterative_search_params{};
    auto index                       = build(res, params, padded->view());
    EXPECT_EQ(index.metric(), params.metric);
    EXPECT_EQ(index.size(), rows);
    EXPECT_EQ(index.graph_degree(), params.graph_degree);

    auto device_queries = raft::make_device_matrix<T, int64_t>(res, nq, dim);
    raft::copy(
      device_queries.data_handle(), queries.data_handle(), queries.size(), res.get_stream());
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, nq, k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, nq, k);
    search_params search_params;
    search_params.itopk_size = 128;
    search(res,
           search_params,
           index,
           raft::make_const_mdspan(device_queries.view()),
           neighbors.view(),
           distances.view());
    std::vector<uint32_t> ids(nq * k);
    std::vector<float> actual(nq * k);
    raft::copy(ids.data(), neighbors.data_handle(), ids.size(), res.get_stream());
    raft::copy(actual.data(), distances.data_handle(), actual.size(), res.get_stream());
    res.sync_stream();

    int64_t hits = 0;
    for (int64_t q = 0; q < nq; ++q) {
      std::vector<float> expected(rows);
      for (int64_t i = 0; i < rows; ++i) {
        expected[i] =
          jaccard_reference(queries.data_handle() + q * dim, data.data_handle() + i * dim, dim);
      }
      auto sorted = expected;
      std::sort(sorted.begin(), sorted.end());
      const auto cutoff = sorted[k - 1];
      const auto strict_count =
        std::count_if(expected.begin(), expected.end(), [cutoff](float d) { return d < cutoff; });
      int64_t strict_hits   = 0;
      int64_t boundary_hits = 0;
      std::vector<bool> seen(rows, false);
      for (int64_t j = 0; j < k; ++j) {
        const auto pos = q * k + j;
        ASSERT_LT(ids[pos], rows);
        EXPECT_FALSE(seen[ids[pos]]);
        seen[ids[pos]] = true;
        EXPECT_NEAR(actual[pos], expected[ids[pos]], 1e-6);
        if (j > 0) { EXPECT_LE(actual[pos - 1], actual[pos]); }
        strict_hits += expected[ids[pos]] < cutoff;
        boundary_hits += expected[ids[pos]] == cutoff;
      }
      hits += strict_hits + std::min<int64_t>(boundary_hits, k - strict_count);
    }
    EXPECT_GE(double(hits) / (nq * k), 0.9);
  }
}

}  // namespace

TEST(CagraJaccard, IVFPQBuilderIsRejected)
{
  raft::device_resources res;
  auto data = binary_data<float>(512, 65);
  auto padded =
    cuvs::neighbors::make_device_padded_dataset(res, raft::make_const_mdspan(data.view()));
  index_params params;
  params.metric = cuvs::distance::DistanceType::JaccardExpanded;

  params.graph_build_params = graph_build_params::ivf_pq_params{};
  EXPECT_THROW(build(res, params, padded->view()), raft::exception);
}

}  // namespace cuvs::neighbors::cagra
