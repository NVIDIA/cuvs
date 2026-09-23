/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/flowann.hpp>

#include <neighbors/detail/flowann/queue.cuh>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_setter.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace flowann = cuvs::neighbors::cagra::experimental::flowann;

namespace {

void pack_two_nodes(raft::host_matrix<std::uint8_t, int64_t>& inner,
                    int64_t row,
                    std::uint8_t first_neighbor,
                    std::uint8_t second_neighbor,
                    std::uint32_t packed_bits)
{
  inner(row, 0) = 1;
  inner(row, 1) = 1;
  for (int64_t col = 2; col < inner.extent(1); ++col) {
    inner(row, col) = 0;
  }
  for (std::uint32_t node = 0; node < 2; ++node) {
    auto const value = node == 0 ? first_neighbor : second_neighbor;
    for (std::uint32_t bit = 0; bit < packed_bits; ++bit) {
      auto const position = node * packed_bits + bit;
      inner(row, 2 + position / 8) |= ((std::uint32_t{value} >> bit) & 1u) << (position % 8);
    }
  }
}

struct search_result {
  raft::host_matrix<std::uint32_t, int64_t> neighbors;
  raft::host_matrix<float, int64_t> distances;
  flowann::queue_statistics queue_statistics;
  bool pollers_paused;
};

auto run_dense_search(cuvs::neighbors::cagra::search_algo algo,
                      std::vector<float> const& query_values,
                      std::vector<std::uint32_t> const& seeds = {},
                      std::uint32_t max_iterations            = 16,
                      bool keep_pollers_running               = false,
                      std::uint32_t max_batch_queries         = 0,
                      std::uint32_t itopk_size                = 32,
                      std::uint32_t search_width              = 1,
                      std::uint32_t num_queues                = 2,
                      bool alternate_modes                    = false,
                      std::uint32_t packed_bits               = 2,
                      std::uint32_t row_padding               = 0) -> search_result
{
  raft::resources res;
  constexpr int64_t rows = 8;
  constexpr int64_t dim  = 4;

  auto host_dataset = raft::make_host_matrix<float, int64_t>(rows, dim);
  for (int64_t row = 0; row < rows; ++row) {
    auto const group     = row < 4 ? 0.0f : 10.0f;
    host_dataset(row, 0) = group + static_cast<float>(row % 4);
    for (int64_t col = 1; col < dim; ++col) {
      host_dataset(row, col) = 0.0f;
    }
  }
  auto device_dataset = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  raft::copy(device_dataset.data_handle(),
             host_dataset.data_handle(),
             host_dataset.size(),
             raft::resource::get_cuda_stream(res));
  auto dataset = cuvs::neighbors::make_device_padded_dataset_view(res, device_dataset.view());

  flowann::device_padded_index<float> index(
    res, cuvs::distance::DistanceType::L2Expanded, dataset, 2, packed_bits);
  auto inner =
    raft::make_host_matrix<std::uint8_t, int64_t>(4, 2 + (2 * packed_bits + 7) / 8 + row_padding);
  pack_two_nodes(inner, 0, 1, 2, packed_bits);
  pack_two_nodes(inner, 1, 3, 0, packed_bits);
  pack_two_nodes(inner, 2, 1, 2, packed_bits);
  pack_two_nodes(inner, 3, 3, 0, packed_bits);

  auto cross = raft::make_host_matrix<std::uint32_t, int64_t>(rows, 2);
  for (std::uint32_t row = 0; row < rows; ++row) {
    cross(row, 0) = 1;
    cross(row, 1) = row < 4 ? row + 4 : row - 4;
  }
  index.update_graph(
    res, raft::make_const_mdspan(inner.view()), raft::make_const_mdspan(cross.view()), 2, 1);

  auto offsets      = raft::make_host_vector<std::uint32_t, int64_t>(2);
  offsets(0)        = 0;
  offsets(1)        = 4;
  auto subgraph_ids = raft::make_host_vector<std::uint32_t, int64_t>(rows);
  auto source_ids   = raft::make_host_vector<std::uint32_t, int64_t>(rows);
  for (std::uint32_t row = 0; row < rows; ++row) {
    subgraph_ids(row) = row / 4;
    source_ids(row)   = rows - 1 - row;
  }
  index.update_subgraph_layout(
    res,
    raft::make_host_vector_view<const std::uint32_t, int64_t>(offsets.data_handle(), 2),
    raft::make_host_vector_view<const std::uint32_t, int64_t>(subgraph_ids.data_handle(), rows));
  index.update_source_indices(
    res, raft::make_host_vector_view<const std::uint32_t, int64_t>(source_ids.data_handle(), rows));

  if (!seeds.empty()) {
    index.update_seeds(res,
                       raft::make_host_vector_view<const std::uint32_t, int64_t>(
                         seeds.data(), static_cast<int64_t>(seeds.size())));
  }

  auto const query_count = static_cast<int64_t>(query_values.size());
  auto host_queries      = raft::make_host_matrix<float, int64_t>(query_count, dim);
  for (int64_t row = 0; row < query_count; ++row) {
    host_queries(row, 0) = query_values[row];
    for (int64_t col = 1; col < dim; ++col) {
      host_queries(row, col) = 0.0f;
    }
  }
  auto queries = raft::make_device_matrix<float, int64_t>(res, query_count, dim);
  raft::copy(queries.data_handle(),
             host_queries.data_handle(),
             host_queries.size(),
             raft::resource::get_cuda_stream(res));
  auto neighbors = raft::make_device_matrix<std::uint32_t, int64_t>(res, query_count, 2);
  auto distances = raft::make_device_matrix<float, int64_t>(res, query_count, 2);

  flowann::queue_params queue_params{};
  queue_params.num_queues         = num_queues;
  queue_params.collect_statistics = true;
  flowann::search_context context(index.cross_graph(), queue_params);
  flowann::search_params params{};
  params.algo           = algo;
  params.itopk_size     = itopk_size;
  params.search_width   = search_width;
  params.max_iterations = max_iterations;
  params.max_queries    = max_batch_queries == 0 ? query_count : max_batch_queries;
  params.num_seeds      = static_cast<std::uint32_t>(seeds.size());
  std::optional<flowann::search_session> session;
  if (keep_pollers_running) { session.emplace(context); }
  if (alternate_modes) {
    for (auto alternate_algo : {cuvs::neighbors::cagra::search_algo::SINGLE_CTA,
                                cuvs::neighbors::cagra::search_algo::MULTI_CTA}) {
      params.algo = alternate_algo;
      for (auto alternate_width : {1u, 2u}) {
        params.search_width = alternate_width;
        for (int64_t count : {int64_t{1}, query_count}) {
          flowann::search(
            res,
            params,
            index,
            context,
            raft::make_device_matrix_view<const float, int64_t>(queries.data_handle(), count, dim),
            raft::make_device_matrix_view<std::uint32_t, int64_t>(
              neighbors.data_handle(), count, 2),
            raft::make_device_matrix_view<float, int64_t>(distances.data_handle(), count, 2));
          auto observed_neighbors = raft::make_host_matrix<std::uint32_t, int64_t>(count, 2);
          auto observed_distances = raft::make_host_matrix<float, int64_t>(count, 2);
          raft::copy(observed_neighbors.data_handle(),
                     neighbors.data_handle(),
                     observed_neighbors.size(),
                     raft::resource::get_cuda_stream(res));
          raft::copy(observed_distances.data_handle(),
                     distances.data_handle(),
                     observed_distances.size(),
                     raft::resource::get_cuda_stream(res));
          raft::resource::sync_stream(res);
          for (int64_t query = 0; query < count; ++query) {
            int64_t nearest = 0;
            for (int64_t row = 1; row < rows; ++row) {
              if (std::abs(query_values[query] - host_dataset(row, 0)) <
                  std::abs(query_values[query] - host_dataset(nearest, 0))) {
                nearest = row;
              }
            }
            auto const delta = query_values[query] - host_dataset(nearest, 0);
            EXPECT_EQ(observed_neighbors(query, 0), source_ids(nearest));
            EXPECT_NEAR(observed_distances(query, 0), delta * delta, 1e-4f);
          }
        }
      }
    }
    params.algo         = algo;
    params.search_width = search_width;
  }
  flowann::search(res,
                  params,
                  index,
                  context,
                  raft::make_const_mdspan(queries.view()),
                  neighbors.view(),
                  distances.view());
  raft::resource::sync_stream(res);
  auto const pollers_paused = flowann::detail::search_context_access::is_paused(context);
  session.reset();

  auto host_neighbors = raft::make_host_matrix<std::uint32_t, int64_t>(query_count, 2);
  auto host_distances = raft::make_host_matrix<float, int64_t>(query_count, 2);
  raft::copy(host_neighbors.data_handle(),
             neighbors.data_handle(),
             neighbors.size(),
             raft::resource::get_cuda_stream(res));
  raft::copy(host_distances.data_handle(),
             distances.data_handle(),
             distances.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  return {
    std::move(host_neighbors), std::move(host_distances), context.statistics(), pollers_paused};
}

class FlowannSearchModesTest
  : public ::testing::TestWithParam<cuvs::neighbors::cagra::search_algo> {};

TEST_P(FlowannSearchModesTest, TraversesInnerAndCrossEdgesAndMapsSourceIds)
{
  auto result = run_dense_search(GetParam(), {0.1f, 12.2f});

  EXPECT_EQ(result.neighbors(0, 0), 7);
  EXPECT_EQ(result.neighbors(1, 0), 1);
  EXPECT_NEAR(result.distances(0, 0), 0.01f, 1e-4f);
  EXPECT_NEAR(result.distances(1, 0), 0.04f, 1e-4f);
  EXPECT_GT(result.queue_statistics.processed_commands, 0);
}

TEST_P(FlowannSearchModesTest, DecodesPackedRowsAcrossByteBoundaries)
{
  for (std::uint32_t bits : {2u, 3u, 7u, 8u, 15u, 24u, 31u, 32u}) {
    SCOPED_TRACE(bits);
    auto result =
      run_dense_search(GetParam(), {0.1f, 12.2f}, {}, 16, true, 2, 256, 8, 1, false, bits);
    EXPECT_EQ(result.neighbors(0, 0), 7u);
    EXPECT_EQ(result.neighbors(1, 0), 1u);
    EXPECT_NEAR(result.distances(0, 0), 0.01f, 1e-4f);
    EXPECT_NEAR(result.distances(1, 0), 0.04f, 1e-4f);
    EXPECT_GT(result.queue_statistics.processed_commands, 0u);
  }
}

TEST_P(FlowannSearchModesTest, Decodes24BitRowsWithEveryRowAlignment)
{
  for (std::uint32_t padding : {0u, 1u, 2u, 3u}) {
    for (std::uint32_t width : {1u, 8u}) {
      SCOPED_TRACE(padding);
      SCOPED_TRACE(width);
      auto result = run_dense_search(
        GetParam(), {0.1f, 12.2f}, {}, 16, true, 2, 256, width, 1, false, 24, padding);
      EXPECT_EQ(result.neighbors(0, 0), 7u);
      EXPECT_EQ(result.neighbors(1, 0), 1u);
      EXPECT_NEAR(result.distances(0, 0), 0.01f, 1e-4f);
      EXPECT_NEAR(result.distances(1, 0), 0.04f, 1e-4f);
      EXPECT_GT(result.queue_statistics.processed_commands, 0u);
    }
  }
}

TEST_P(FlowannSearchModesTest, UsesFixedSeedCrossEdgeCacheOnFirstExpansion)
{
  auto result = run_dense_search(GetParam(), {10.1f}, {0}, 2);

  EXPECT_EQ(result.neighbors(0, 0), 3);
  EXPECT_NEAR(result.distances(0, 0), 0.01f, 1e-4f);
  EXPECT_EQ(result.queue_statistics.processed_commands, 0);
}

TEST_P(FlowannSearchModesTest, PreservesSearchSessionPollers)
{
  auto result = run_dense_search(GetParam(), {0.1f, 12.2f}, {}, 16, true);

  EXPECT_FALSE(result.pollers_paused);
}

TEST(FlowannSearch, RejectsContextFromDifferentCudaDevice)
{
  if (raft::device_setter::get_device_count() < 2) {
    GTEST_SKIP() << "Requires at least two CUDA devices";
  }

  raft::device_setter device_zero(0);
  raft::resources res;
  flowann::device_padded_index<float> index(res);
  auto queries   = raft::make_device_matrix<float, int64_t>(res, 1, 1);
  auto neighbors = raft::make_device_matrix<std::uint32_t, int64_t>(res, 1, 1);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 1, 1);
  auto cross     = raft::make_host_matrix<std::uint32_t, int64_t>(1, 1);
  cross(0, 0)    = 0;

  raft::device_setter device_one(1);
  flowann::search_context context(raft::make_const_mdspan(cross.view()));
  raft::device_setter restore_device_zero(0);
  try {
    flowann::search(res,
                    flowann::search_params{},
                    index,
                    context,
                    raft::make_const_mdspan(queries.view()),
                    neighbors.view(),
                    distances.view());
    FAIL() << "Expected a mismatched-device FlowANN context to be rejected";
  } catch (raft::exception const& error) {
    EXPECT_NE(std::string{error.what()}.find(
                "FlowANN search context must be used on the CUDA device where it was created"),
              std::string::npos);
  }
}

TEST(FlowannMultiCtaSearch, EvaluatesSeedsBeyondGraphDegree)
{
  // This graph has degree two. Stop before expansion so that node four can only be found
  // by evaluating the third seed. Source IDs are reversed by run_dense_search.
  auto result =
    run_dense_search(cuvs::neighbors::cagra::search_algo::MULTI_CTA, {10.1f}, {0, 1, 4}, 1);

  EXPECT_EQ(result.neighbors(0, 0), 3);
  EXPECT_NEAR(result.distances(0, 0), 0.01f, 1e-4f);
  EXPECT_EQ(result.queue_statistics.processed_commands, 0);
}

TEST_P(FlowannSearchModesTest, UniversalProviderPreservesSharedQueueBatchesAndWidths)
{
  std::vector<float> const all_queries = {0.1f, 12.2f, 1.1f, 11.2f, 2.1f, 10.2f, 3.1f, 13.2f};
  std::vector<std::uint32_t> const expected_sources = {7, 1, 6, 2, 5, 3, 4, 0};
  std::vector<float> const expected_distances       = {
    0.01f, 0.04f, 0.01f, 0.04f, 0.01f, 0.04f, 0.01f, 0.04f};
  for (std::uint32_t batch : {1u, 2u, 8u}) {
    auto queries = std::vector<float>(all_queries.begin(), all_queries.begin() + batch);
    for (std::uint32_t width : {1u, 2u, 4u}) {
      auto result = run_dense_search(GetParam(), queries, {}, 16, true, batch, 256, width, 1);
      for (std::uint32_t query = 0; query < batch; ++query) {
        EXPECT_EQ(result.neighbors(query, 0), expected_sources[query]);
        EXPECT_NEAR(result.distances(query, 0), expected_distances[query], 1e-4f);
      }
      EXPECT_GT(result.queue_statistics.processed_commands, 0);
      EXPECT_FALSE(result.pollers_paused);
    }
  }
}

TEST(FlowannMultiCtaSearch, UniversalProviderHandlesInternallyChunkedRequests)
{
  auto result = run_dense_search(
    cuvs::neighbors::cagra::search_algo::MULTI_CTA, {0.1f, 12.2f}, {}, 16, true, 1, 256, 2, 1);
  EXPECT_EQ(result.neighbors(0, 0), 7u);
  EXPECT_EQ(result.neighbors(1, 0), 1u);
}

TEST(FlowannSearchModes, AlternatesAlgorithmsWidthsAndShapesInOneSession)
{
  auto result = run_dense_search(cuvs::neighbors::cagra::search_algo::MULTI_CTA,
                                 {0.1f, 12.2f, 1.1f, 11.2f, 2.1f, 10.2f, 3.1f, 13.2f},
                                 {},
                                 16,
                                 true,
                                 8,
                                 256,
                                 1,
                                 1,
                                 true);
  EXPECT_EQ(result.neighbors(0, 0), 7u);
  EXPECT_EQ(result.neighbors(1, 0), 1u);
  EXPECT_FALSE(result.pollers_paused);
}

INSTANTIATE_TEST_SUITE_P(SingleAndMultiCta,
                         FlowannSearchModesTest,
                         ::testing::Values(cuvs::neighbors::cagra::search_algo::SINGLE_CTA,
                                           cuvs::neighbors::cagra::search_algo::MULTI_CTA));

}  // namespace
