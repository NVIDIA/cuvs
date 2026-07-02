/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cuvs::neighbors::cagra {
namespace {

constexpr int64_t n_rows                   = 768;
constexpr int64_t n_dim                    = 16;
constexpr int64_t n_queries                = 6;
constexpr int64_t k                        = 8;
constexpr int64_t threshold                = 192;
constexpr int64_t high_filtering_threshold = 704;
constexpr float high_filtering_rate =
  static_cast<float>(high_filtering_threshold) / static_cast<float>(n_rows);

struct tenant_filter_context {
  const uint32_t* row_tenants;
  const uint32_t* query_tenants;
};

struct cagra_search_result {
  std::vector<uint32_t> neighbors;
  std::vector<float> distances;
};

std::string accept_all_udf_source()
{
  return R"cpp(
    __device__ bool cuvs_filter_udf(uint32_t, source_index_t, void*) { return true; }
  )cpp";
}

std::string threshold_udf_source()
{
  return R"cpp(
    __device__ bool cuvs_filter_udf(uint32_t, source_index_t source_id, void*)
    {
      return source_id >= 192;
    }
  )cpp";
}

std::string reject_all_udf_source()
{
  return R"cpp(
    __device__ bool cuvs_filter_udf(uint32_t, source_index_t, void*) { return false; }
  )cpp";
}

std::string high_filtering_rate_udf_source()
{
  return R"cpp(
    __device__ bool cuvs_filter_udf(uint32_t, source_index_t source_id, void*)
    {
      return source_id >= 704;
    }
  )cpp";
}

std::string tenant_udf_source()
{
  return R"cpp(
    struct tenant_filter_context {
      const uint32_t* row_tenants;
      const uint32_t* query_tenants;
    };

    __device__ bool cuvs_filter_udf(uint32_t query_id, source_index_t source_id, void* filter_data)
    {
      auto* ctx = static_cast<const tenant_filter_context*>(filter_data);
      return ctx->row_tenants[source_id] == ctx->query_tenants[query_id];
    }
  )cpp";
}

void expect_same_results(cagra_search_result const& expected, cagra_search_result const& actual)
{
  ASSERT_EQ(expected.neighbors, actual.neighbors);
  ASSERT_EQ(expected.distances.size(), actual.distances.size());
  for (size_t i = 0; i < expected.distances.size(); ++i) {
    EXPECT_FLOAT_EQ(expected.distances[i], actual.distances[i]);
  }
}

class CagraUdfFilterTest : public ::testing::TestWithParam<cagra::search_algo> {
 protected:
  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<float, int64_t>(res, n_rows, n_dim));
    queries.emplace(raft::make_device_matrix<float, int64_t>(res, n_queries, n_dim));

    raft::random::RngState rng(1234ULL);
    raft::random::uniform(res, rng, dataset->data_handle(), dataset->size(), -1.0f, 1.0f);
    raft::random::uniform(res, rng, queries->data_handle(), queries->size(), -1.0f, 1.0f);

    cagra::index_params index_params;
    index_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
    index_params.graph_degree              = 32;
    index_params.intermediate_graph_degree = 64;
    index_params.graph_build_params =
      cagra::graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);

    index.emplace(cagra::build(res, index_params, raft::make_const_mdspan(dataset->view())));
    raft::resource::sync_stream(res);
  }

  void TearDown() override
  {
    index.reset();
    queries.reset();
    dataset.reset();
    raft::resource::sync_stream(res);
  }

  cagra_search_result search(cuvs::neighbors::filtering::base_filter const& filter,
                             float filtering_rate = -1.0f)
  {
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);

    cagra::search_params search_params;
    search_params.algo              = GetParam();
    search_params.itopk_size        = 64;
    search_params.max_queries       = 2;
    search_params.thread_block_size = 256;
    search_params.filtering_rate    = filtering_rate;

    cagra::search(res,
                  search_params,
                  *index,
                  raft::make_const_mdspan(queries->view()),
                  neighbors.view(),
                  distances.view(),
                  filter);

    auto stream = raft::resource::get_cuda_stream(res);
    cagra_search_result result{std::vector<uint32_t>(n_queries * k),
                               std::vector<float>(n_queries * k)};
    raft::copy(result.neighbors.data(), neighbors.data_handle(), result.neighbors.size(), stream);
    raft::copy(result.distances.data(), distances.data_handle(), result.distances.size(), stream);
    raft::resource::sync_stream(res);
    return result;
  }

  raft::resources res;
  std::optional<raft::device_matrix<float, int64_t>> dataset = std::nullopt;
  std::optional<raft::device_matrix<float, int64_t>> queries = std::nullopt;
  std::optional<cagra::index<float, uint32_t>> index         = std::nullopt;
};

class CagraUdfFilterHalfTest : public ::testing::TestWithParam<cagra::search_algo> {
 protected:
  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<half, int64_t>(res, n_rows, n_dim));
    queries.emplace(raft::make_device_matrix<half, int64_t>(res, n_queries, n_dim));

    raft::random::RngState rng(1234ULL);
    InitDataset(res,
                dataset->data_handle(),
                static_cast<std::uint32_t>(n_rows),
                static_cast<std::uint32_t>(n_dim),
                cuvs::distance::DistanceType::L2Expanded,
                rng);
    InitDataset(res,
                queries->data_handle(),
                static_cast<std::uint32_t>(n_queries),
                static_cast<std::uint32_t>(n_dim),
                cuvs::distance::DistanceType::L2Expanded,
                rng);

    cagra::index_params index_params;
    index_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
    index_params.graph_degree              = 32;
    index_params.intermediate_graph_degree = 64;
    index_params.graph_build_params =
      cagra::graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);

    index.emplace(cagra::build(res, index_params, raft::make_const_mdspan(dataset->view())));
    raft::resource::sync_stream(res);
  }

  void TearDown() override
  {
    index.reset();
    queries.reset();
    dataset.reset();
    raft::resource::sync_stream(res);
  }

  cagra_search_result search(cuvs::neighbors::filtering::base_filter const& filter,
                             float filtering_rate = -1.0f)
  {
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);

    cagra::search_params search_params;
    search_params.algo              = GetParam();
    search_params.itopk_size        = 64;
    search_params.max_queries       = 2;
    search_params.thread_block_size = 256;
    search_params.filtering_rate    = filtering_rate;

    cagra::search(res,
                  search_params,
                  *index,
                  raft::make_const_mdspan(queries->view()),
                  neighbors.view(),
                  distances.view(),
                  filter);

    auto stream = raft::resource::get_cuda_stream(res);
    cagra_search_result result{std::vector<uint32_t>(n_queries * k),
                               std::vector<float>(n_queries * k)};
    raft::copy(result.neighbors.data(), neighbors.data_handle(), result.neighbors.size(), stream);
    raft::copy(result.distances.data(), distances.data_handle(), result.distances.size(), stream);
    raft::resource::sync_stream(res);
    return result;
  }

  raft::resources res;
  std::optional<raft::device_matrix<half, int64_t>> dataset = std::nullopt;
  std::optional<raft::device_matrix<half, int64_t>> queries = std::nullopt;
  std::optional<cagra::index<half, uint32_t>> index         = std::nullopt;
};

TEST_P(CagraUdfFilterTest, AcceptAllMatchesNoFilter)
{
  cuvs::neighbors::filtering::none_sample_filter no_filter;
  auto expected = search(no_filter, 0.0f);

  cuvs::neighbors::filtering::udf_filter udf_filter(accept_all_udf_source(), nullptr, 0.0f);
  auto actual = search(udf_filter);

  expect_same_results(expected, actual);
}

TEST_P(CagraUdfFilterHalfTest, ThresholdReturnsOnlyValidNeighbors)
{
  float const filtering_rate = static_cast<float>(threshold) / static_cast<float>(n_rows);
  cuvs::neighbors::filtering::udf_filter udf_filter(
    threshold_udf_source(), nullptr, filtering_rate);
  auto result = search(udf_filter, filtering_rate);

  for (auto source_id : result.neighbors) {
    if (source_id < static_cast<uint32_t>(n_rows)) {
      EXPECT_GE(source_id, static_cast<uint32_t>(threshold));
    }
  }
}

TEST_P(CagraUdfFilterTest, RejectAllReturnsNoValidNeighbors)
{
  cuvs::neighbors::filtering::udf_filter udf_filter(reject_all_udf_source(), nullptr, 0.999f);
  auto result = search(udf_filter);

  // CAGRA algorithms do not all normalize empty-result sentinels the same way. Single-CTA
  // clears the internal high-bit marker before writing output, so 0xffffffff can become
  // 0x7fffffff; other paths may leave 0xffffffff. Both are invalid row ids for this index.
  for (auto source_id : result.neighbors) {
    EXPECT_GE(source_id, static_cast<uint32_t>(n_rows));
  }
}

TEST_P(CagraUdfFilterTest, HighFilteringRateReturnsOnlyValidNeighbors)
{
  cuvs::neighbors::filtering::udf_filter udf_filter(
    high_filtering_rate_udf_source(), nullptr, high_filtering_rate);
  auto result = search(udf_filter);

  for (auto source_id : result.neighbors) {
    if (source_id < static_cast<uint32_t>(n_rows)) {
      EXPECT_GE(source_id, static_cast<uint32_t>(high_filtering_threshold));
    }
  }
}

TEST_P(CagraUdfFilterTest, RepeatedUdfSearchWithSameSourceMatches)
{
  cuvs::neighbors::filtering::udf_filter udf_filter(accept_all_udf_source(), nullptr, 0.0f);

  auto first  = search(udf_filter);
  auto second = search(udf_filter);

  expect_same_results(first, second);
}

TEST_P(CagraUdfFilterTest, InvalidSourceThrows)
{
  cuvs::neighbors::filtering::udf_filter udf_filter("this is not valid cuda source", nullptr, 0.0f);

  EXPECT_THROW(search(udf_filter), std::exception);
}

TEST_P(CagraUdfFilterTest, ThresholdMatchesEquivalentBitset)
{
  auto removed_indices = raft::make_device_vector<int64_t, int64_t>(res, threshold);
  thrust::sequence(raft::resource::get_thrust_policy(res),
                   thrust::device_pointer_cast(removed_indices.data_handle()),
                   thrust::device_pointer_cast(removed_indices.data_handle() + threshold));
  raft::resource::sync_stream(res);

  cuvs::core::bitset<std::uint32_t, int64_t> removed_indices_bitset(
    res, removed_indices.view(), n_rows);
  cuvs::neighbors::filtering::bitset_filter bitset_filter(removed_indices_bitset.view());

  float const filtering_rate = static_cast<float>(threshold) / static_cast<float>(n_rows);
  auto expected              = search(bitset_filter, filtering_rate);

  cuvs::neighbors::filtering::udf_filter udf_filter(
    threshold_udf_source(), nullptr, filtering_rate);
  auto actual = search(udf_filter, filtering_rate);

  expect_same_results(expected, actual);
}

TEST_P(CagraUdfFilterTest, TenantContextHonorsQuerySpecificMetadata)
{
  std::vector<uint32_t> host_row_tenants(n_rows);
  std::vector<uint32_t> host_query_tenants(n_queries);
  for (int64_t i = 0; i < n_rows; ++i) {
    host_row_tenants[static_cast<size_t>(i)] = static_cast<uint32_t>((i / 5) % 3);
  }
  for (int64_t q = 0; q < n_queries; ++q) {
    host_query_tenants[static_cast<size_t>(q)] = static_cast<uint32_t>(q % 3);
  }

  auto row_tenants   = raft::make_device_vector<uint32_t, int64_t>(res, n_rows);
  auto query_tenants = raft::make_device_vector<uint32_t, int64_t>(res, n_queries);
  auto context       = raft::make_device_vector<tenant_filter_context, int64_t>(res, 1);

  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(row_tenants.data_handle(), host_row_tenants.data(), host_row_tenants.size(), stream);
  raft::copy(
    query_tenants.data_handle(), host_query_tenants.data(), host_query_tenants.size(), stream);

  tenant_filter_context host_context{row_tenants.data_handle(), query_tenants.data_handle()};
  raft::copy(context.data_handle(), &host_context, 1, stream);
  raft::resource::sync_stream(res);

  cuvs::neighbors::filtering::udf_filter udf_filter(
    tenant_udf_source(), context.data_handle(), 2.0f / 3.0f);
  auto result = search(udf_filter);

  for (int64_t q = 0; q < n_queries; ++q) {
    auto query_tenant = host_query_tenants[static_cast<size_t>(q)];
    for (int64_t i = 0; i < k; ++i) {
      auto source_id = result.neighbors[static_cast<size_t>(q * k + i)];
      ASSERT_LT(source_id, static_cast<uint32_t>(n_rows));
      EXPECT_EQ(host_row_tenants[source_id], query_tenant);
    }
  }
}

INSTANTIATE_TEST_CASE_P(CagraUdfFilters,
                        CagraUdfFilterTest,
                        ::testing::Values(cagra::search_algo::SINGLE_CTA,
                                          cagra::search_algo::MULTI_CTA,
                                          cagra::search_algo::MULTI_KERNEL));

INSTANTIATE_TEST_CASE_P(CagraUdfFilterHalf,
                        CagraUdfFilterHalfTest,
                        ::testing::Values(cagra::search_algo::SINGLE_CTA,
                                          cagra::search_algo::MULTI_CTA,
                                          cagra::search_algo::MULTI_KERNEL));

}  // namespace
}  // namespace cuvs::neighbors::cagra
