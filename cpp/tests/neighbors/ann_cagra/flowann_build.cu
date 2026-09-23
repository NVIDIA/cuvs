/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/flowann_build.hpp>

#include <neighbors/detail/flowann/build.hpp>
#include <neighbors/detail/flowann/grouping.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace flowann = cuvs::neighbors::cagra::experimental::flowann;

TEST(FlowannBuild, ComputesMinimalGlobalIdWidth)
{
  EXPECT_EQ(flowann::detail::required_id_bits(1), 1);
  EXPECT_EQ(flowann::detail::required_id_bits(2), 1);
  EXPECT_EQ(flowann::detail::required_id_bits(3), 2);
  EXPECT_EQ(flowann::detail::required_id_bits(87'555'327), 27);
  EXPECT_EQ(flowann::detail::required_id_bits(1'000'000'000), 30);
}

TEST(FlowannBuild, SelectsLargestUniformDegreeWithinBudget)
{
  constexpr std::uint64_t rows          = 100;
  constexpr std::uint16_t nodes_per_row = 2;
  constexpr std::uint32_t degree        = 32;
  auto const bits                       = flowann::detail::required_id_bits(rows);

  auto const row_bytes_10 = flowann::detail::packed_row_bytes(nodes_per_row, bits, 10);
  auto const budget       = (rows / nodes_per_row) * row_bytes_10;
  auto const plan = flowann::detail::plan_uniform_layout(rows, degree, nodes_per_row, budget);

  EXPECT_EQ(plan.n_bits, bits);
  EXPECT_EQ(plan.resident_degree, 10);
  EXPECT_EQ(plan.cross_degree, degree - 10);
  EXPECT_EQ(plan.inner_row_bytes, row_bytes_10);
  EXPECT_EQ(plan.inner_graph_bytes, budget);
}

TEST(FlowannBuild, SupportsZeroAndFullResidentDegree)
{
  auto const zero = flowann::detail::plan_uniform_layout(101, 8, 2, 0);
  EXPECT_EQ(zero.resident_degree, 0);
  EXPECT_EQ(zero.cross_degree, 8);
  EXPECT_EQ(zero.inner_row_bytes, 0);
  EXPECT_EQ(zero.inner_graph_bytes, 0);

  auto const full_row = flowann::detail::packed_row_bytes(2, 7, 8);
  auto const full     = flowann::detail::plan_uniform_layout(101, 8, 2, 51 * full_row);
  EXPECT_EQ(full.resident_degree, 8);
  EXPECT_EQ(full.cross_degree, 0);
  EXPECT_EQ(full.inner_graph_bytes, 51 * full_row);
}

TEST(FlowannBuild, RejectsGraphSizesOutsideUint32IdSpace)
{
  EXPECT_ANY_THROW(flowann::detail::plan_uniform_layout(
    std::numeric_limits<std::uint64_t>::max(), 32, 2, std::numeric_limits<std::uint64_t>::max()));
}

TEST(FlowannBuild, GroupedCapacityUsesBudgetRatherThanAverageLocalDegree)
{
  auto row_bytes = flowann::detail::budgeted_grouped_row_bytes;
  EXPECT_EQ(row_bytes(87'555'327, 32, 2, 24, 7'529'758'208), 172);
  EXPECT_EQ(row_bytes(101, 32, 2, 24, 51 * 139), 139);
  EXPECT_EQ(row_bytes(101, 32, 2, 24, 51 * 172 + 50), 172);
  EXPECT_EQ(row_bytes(101, 32, 2, 24, 51 * 194), 194);
  EXPECT_EQ(row_bytes(101, 32, 2, 24, 51 * 1000), 194);
  EXPECT_EQ(row_bytes(101, 32, 2, 24, 0), 0);
  EXPECT_EQ(row_bytes(101, 32, 2, 24, 51 * 4), 0);
  EXPECT_EQ(row_bytes(101, 32, 2, 24, 51 * 5), 5);
  EXPECT_EQ(row_bytes(101, 8, 2, 6, 51 * 14), 14);
}

namespace {

auto read_packed(std::uint8_t const* row, std::uint64_t bit_offset, std::uint16_t bits)
  -> std::uint32_t
{
  auto result = std::uint32_t{0};
  for (std::uint16_t bit = 0; bit < bits; ++bit) {
    result |=
      static_cast<std::uint32_t>((row[(bit_offset + bit) >> 3] >> ((bit_offset + bit) & 7)) & 1u)
      << bit;
  }
  return result;
}

auto l2(float const* lhs, float const* rhs, std::uint32_t dim) -> float
{
  auto result = 0.0f;
  for (std::uint32_t column = 0; column < dim; ++column) {
    auto const diff = lhs[column] - rhs[column];
    result += diff * diff;
  }
  return result;
}

auto metric_score(float const* lhs,
                  float const* rhs,
                  std::uint32_t dim,
                  cuvs::distance::DistanceType metric) -> float
{
  auto dot      = 0.0f;
  auto lhs_norm = 0.0f;
  auto rhs_norm = 0.0f;
  for (std::uint32_t column = 0; column < dim; ++column) {
    dot += lhs[column] * rhs[column];
    lhs_norm += lhs[column] * lhs[column];
    rhs_norm += rhs[column] * rhs[column];
  }
  if (metric == cuvs::distance::DistanceType::InnerProduct) { return -dot; }
  return 1.0f - dot / std::sqrt(lhs_norm * rhs_norm);
}

}  // namespace

TEST(FlowannBuild, IntegratedBuildKeepsShortestNeighborsResident)
{
  constexpr int64_t rows                  = 128;
  constexpr int64_t dim                   = 8;
  constexpr std::uint32_t full_degree     = 8;
  constexpr std::uint32_t resident_degree = 4;
  constexpr std::uint16_t nodes_per_row   = 2;

  raft::resources res;
  auto host = raft::make_host_matrix<float, int64_t>(rows, dim);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < dim; ++column) {
      host(row, column) =
        static_cast<float>((row * 17 + column * 13) % 101) + static_cast<float>(row) * 0.001f;
    }
  }
  auto dataset = cuvs::neighbors::make_device_padded_dataset(res, host.view());

  flowann::index_params params;
  params.grouping.enabled                       = false;
  params.cagra_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.cagra_params.graph_degree              = full_degree;
  params.cagra_params.intermediate_graph_degree = 16;
  params.cagra_params.graph_build_params =
    cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
      16, cuvs::distance::DistanceType::L2Expanded);
  params.node_per_cacheline = nodes_per_row;
  auto const bits           = flowann::detail::required_id_bits(rows);
  auto const row_bytes = flowann::detail::packed_row_bytes(nodes_per_row, bits, resident_degree);
  params.device_graph_budget_bytes = (rows / nodes_per_row) * row_bytes;

  auto built = flowann::build(res, params, dataset->as_dataset_view());
  ASSERT_EQ(built.index.graph_degree(), full_degree);
  ASSERT_EQ(built.index.resident_degree(), resident_degree);
  ASSERT_EQ(built.index.cross_degree(), full_degree - resident_degree);

  auto inner = raft::make_host_matrix<std::uint8_t, int64_t>(built.index.inner_graph().extent(0),
                                                             built.index.inner_graph().extent(1));
  raft::copy(inner.data_handle(),
             built.index.inner_graph().data_handle(),
             inner.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  auto const cross = built.index.cross_graph();

  for (std::uint32_t node = 0; node < rows; ++node) {
    auto const packed_row = node / nodes_per_row;
    auto const packed_col = node % nodes_per_row;
    auto const* row       = inner.data_handle() + packed_row * inner.extent(1);
    ASSERT_EQ(row[packed_col], resident_degree);
    auto bit_offset = static_cast<std::uint64_t>(nodes_per_row) * 8;
    for (std::uint16_t column = 0; column < packed_col; ++column) {
      bit_offset += static_cast<std::uint64_t>(row[column]) * bits;
    }

    std::vector<std::pair<float, std::uint32_t>> resident;
    std::vector<std::pair<float, std::uint32_t>> spilled;
    std::set<std::uint32_t> all_neighbors;
    for (std::uint32_t rank = 0; rank < resident_degree; ++rank) {
      auto const neighbor = read_packed(row, bit_offset + rank * bits, bits);
      resident.emplace_back(l2(&host(node, 0), &host(neighbor, 0), dim), neighbor);
      all_neighbors.insert(neighbor);
    }
    ASSERT_EQ(cross(node, 0), full_degree - resident_degree);
    for (std::uint32_t edge = 0; edge < full_degree - resident_degree; ++edge) {
      auto const neighbor = cross(node, edge + 1);
      spilled.emplace_back(l2(&host(node, 0), &host(neighbor, 0), dim), neighbor);
      all_neighbors.insert(neighbor);
    }
    for (std::size_t rank = 1; rank < resident.size(); ++rank) {
      EXPECT_LE(resident[rank - 1].first, resident[rank].first + 1e-3f) << "node=" << node;
    }
    auto const closest_spilled = *std::min_element(spilled.begin(), spilled.end());
    EXPECT_LE(resident.back().first, closest_spilled.first + 1e-3f) << "node=" << node;
    EXPECT_EQ(all_neighbors.size(), full_degree);
  }

  auto query_host = raft::make_host_matrix<float, int64_t>(2, dim);
  std::copy_n(host.data_handle(), query_host.size(), query_host.data_handle());
  auto queries = raft::make_device_matrix<float, int64_t>(res, 2, dim);
  raft::copy(queries.data_handle(),
             query_host.data_handle(),
             query_host.size(),
             raft::resource::get_cuda_stream(res));
  auto neighbors = raft::make_device_matrix<std::uint32_t, int64_t>(res, 2, 2);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 2, 2);
  for (auto algorithm : {cuvs::neighbors::cagra::search_algo::SINGLE_CTA,
                         cuvs::neighbors::cagra::search_algo::MULTI_CTA}) {
    flowann::search_context context(built.index.cross_graph());
    flowann::search_params search_params;
    search_params.algo           = algorithm;
    search_params.itopk_size     = 32;
    search_params.search_width   = 1;
    search_params.max_iterations = 16;
    search_params.max_queries    = 2;
    flowann::search(res,
                    search_params,
                    built.index,
                    context,
                    raft::make_const_mdspan(queries.view()),
                    neighbors.view(),
                    distances.view());
    auto neighbor_host = raft::make_host_matrix<std::uint32_t, int64_t>(2, 2);
    raft::copy(neighbor_host.data_handle(),
               neighbors.data_handle(),
               neighbors.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    for (std::size_t i = 0; i < neighbor_host.size(); ++i) {
      EXPECT_LT(neighbor_host.data_handle()[i], rows);
    }
  }
}

TEST(FlowannBuild, IntegratedBuildGroupsRearrangesAndPreservesSourceIds)
{
  constexpr int64_t rows              = 265;
  constexpr int64_t dim               = 8;
  constexpr std::uint32_t group_count = 5;
  raft::resources res;
  auto host = raft::make_host_matrix<float, int64_t>(rows, dim);
  for (int64_t row = 0; row < rows; ++row) {
    auto const group = row / (rows / group_count);
    for (int64_t column = 0; column < dim; ++column) {
      host(row, column) = static_cast<float>(group * 100 + column) +
                          static_cast<float>(row % (rows / group_count)) * 0.001f;
    }
  }
  auto dataset = cuvs::neighbors::make_device_padded_dataset(res, host.view());

  flowann::index_params params;
  params.cagra_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.cagra_params.graph_degree              = 8;
  params.cagra_params.intermediate_graph_degree = 16;
  params.cagra_params.graph_build_params =
    cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
      16, cuvs::distance::DistanceType::L2Expanded);
  params.node_per_cacheline             = 2;
  params.device_graph_budget_bytes      = ((rows + 1) / 2) * 64;
  params.grouping.n_groups              = 0;
  params.grouping.n_bits                = 6;
  params.grouping.balance_tolerance     = 0.10f;
  params.grouping.training_rows         = 128;
  params.grouping.assignment_batch_rows = 64;
  params.grouping.kmeans_n_iters        = 4;
  params.grouping.validate              = true;

  auto built = flowann::build(res, params, dataset->as_dataset_view());
  EXPECT_EQ(built.index.graph_degree(), 8);
  EXPECT_EQ(built.index.resident_degree(), 0);
  EXPECT_GT(built.index.inner_graph().size(), 0);
  EXPECT_EQ(built.index.inner_graph().extent(1), 14);  // Two headers plus 16 six-bit IDs.

  auto source_view = built.index.source_indices();
  ASSERT_TRUE(source_view.has_value());
  auto source = raft::make_host_vector<std::uint32_t, int64_t>(rows);
  raft::copy(
    source.data_handle(), source_view->data_handle(), rows, raft::resource::get_cuda_stream(res));
  auto ids = std::set<std::uint32_t>();
  for (int64_t row = 0; row < rows; ++row) {
    ids.insert(source(row));
  }
  EXPECT_EQ(ids.size(), rows);
  EXPECT_EQ(*ids.begin(), 0);
  EXPECT_EQ(*ids.rbegin(), rows - 1);

  auto offsets_view = built.index.subgraph_offsets();
  auto groups_view  = built.index.subgraph_ids();
  ASSERT_TRUE(offsets_view.has_value());
  ASSERT_TRUE(groups_view.has_value());
  ASSERT_EQ(offsets_view->extent(0), group_count);
  auto groups = raft::make_host_vector<std::uint32_t, int64_t>(rows);
  raft::copy(
    groups.data_handle(), groups_view->data_handle(), rows, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  auto sizes = std::array<std::uint32_t, group_count>{};
  for (int64_t row = 0; row < rows; ++row) {
    ASSERT_LT(groups(row), group_count);
    ++sizes[groups(row)];
  }
  for (auto size : sizes) {
    EXPECT_GE(size, 47);
    EXPECT_LE(size, 59);
    EXPECT_LE(size, 64);
  }
}

TEST(FlowannBuild, IntegratedBuildGeneratesUniqueMedoidSeeds)
{
  constexpr int64_t rows = 128;
  constexpr int64_t dim  = 8;
  raft::resources res;
  auto host = raft::make_host_matrix<float, int64_t>(rows, dim);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < dim; ++column) {
      host(row, column) = static_cast<float>((row * 19 + column * 7) % 113);
    }
  }
  auto dataset = cuvs::neighbors::make_device_padded_dataset(res, host.view());

  flowann::index_params params;
  params.grouping.enabled                       = false;
  params.cagra_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.cagra_params.graph_degree              = 8;
  params.cagra_params.intermediate_graph_degree = 16;
  params.cagra_params.graph_build_params =
    cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
      16, cuvs::distance::DistanceType::L2Expanded);
  params.device_graph_budget_bytes = 0;
  params.num_seeds                 = 4;
  params.seed_training_rows        = 64;

  auto built = flowann::build(res, params, dataset->as_dataset_view());
  auto seeds = built.index.seeds();
  ASSERT_TRUE(seeds.has_value());
  ASSERT_EQ(seeds->extent(0), params.num_seeds);
  auto host_seeds = raft::make_host_vector<std::uint32_t, int64_t>(params.num_seeds);
  raft::copy(host_seeds.data_handle(),
             seeds->data_handle(),
             params.num_seeds,
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  auto unique = std::set<std::uint32_t>();
  for (std::uint32_t i = 0; i < params.num_seeds; ++i) {
    EXPECT_LT(host_seeds(i), rows);
    unique.insert(host_seeds(i));
  }
  EXPECT_EQ(unique.size(), params.num_seeds);
}

TEST(FlowannBuild, IntegratedBuildUsesCagraMetricForRanking)
{
  constexpr int64_t rows                  = 128;
  constexpr int64_t dim                   = 8;
  constexpr std::uint32_t resident_degree = 4;
  raft::resources res;
  auto host = raft::make_host_matrix<float, int64_t>(rows, dim);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < dim; ++column) {
      host(row, column) = 0.1f + static_cast<float>((row * 29 + column * 17) % 137) * 0.01f;
    }
  }
  auto dataset = cuvs::neighbors::make_device_padded_dataset(res, host.view());

  for (auto metric :
       {cuvs::distance::DistanceType::InnerProduct, cuvs::distance::DistanceType::CosineExpanded}) {
    flowann::index_params params;
    params.grouping.enabled                       = false;
    params.cagra_params.metric                    = metric;
    params.cagra_params.graph_degree              = 8;
    params.cagra_params.intermediate_graph_degree = 16;
    params.cagra_params.graph_build_params =
      cuvs::neighbors::cagra::graph_build_params::nn_descent_params(16, metric);
    auto const bits                  = flowann::detail::required_id_bits(rows);
    auto const row_bytes             = flowann::detail::packed_row_bytes(2, bits, resident_degree);
    params.device_graph_budget_bytes = (rows / 2) * row_bytes;

    auto built = flowann::build(res, params, dataset->as_dataset_view());
    auto inner = raft::make_host_matrix<std::uint8_t, int64_t>(built.index.inner_graph().extent(0),
                                                               built.index.inner_graph().extent(1));
    raft::copy(inner.data_handle(),
               built.index.inner_graph().data_handle(),
               inner.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    auto const cross = built.index.cross_graph();

    for (std::uint32_t node = 0; node < rows; ++node) {
      auto const packed_row = node / 2;
      auto const packed_col = node % 2;
      auto const* row       = inner.data_handle() + packed_row * inner.extent(1);
      auto bit_offset       = std::uint64_t{16};
      if (packed_col != 0) { bit_offset += static_cast<std::uint64_t>(row[0]) * bits; }
      auto worst_resident = -std::numeric_limits<float>::infinity();
      for (std::uint32_t rank = 0; rank < resident_degree; ++rank) {
        auto const neighbor = read_packed(row, bit_offset + rank * bits, bits);
        worst_resident =
          std::max(worst_resident, metric_score(&host(node, 0), &host(neighbor, 0), dim, metric));
      }
      auto best_spilled = std::numeric_limits<float>::infinity();
      for (std::uint32_t edge = 0; edge < cross(node, 0); ++edge) {
        auto const neighbor = cross(node, edge + 1);
        best_spilled =
          std::min(best_spilled, metric_score(&host(node, 0), &host(neighbor, 0), dim, metric));
      }
      EXPECT_LE(worst_resident, best_spilled + 1e-4f) << "node=" << node;
    }
  }
}

TEST(FlowannBuild, IntegratedHostBuildCreatesSearchableVpqIndex)
{
  constexpr int64_t rows = 512;
  constexpr int64_t dim  = 16;
  raft::resources res;
  auto host = raft::make_host_matrix<float, int64_t>(rows, dim);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < dim; ++column) {
      host(row, column) = static_cast<float>((row * 23 + column * 11) % 127) * 0.01f;
    }
  }
  auto dataset = cuvs::neighbors::make_host_standard_dataset_view(host.view());

  flowann::index_params params;
  params.grouping.enabled                       = false;
  params.cagra_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.cagra_params.graph_degree              = 8;
  params.cagra_params.intermediate_graph_degree = 16;
  params.cagra_params.graph_build_params =
    cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
      16, cuvs::distance::DistanceType::L2Expanded);
  auto const bits                  = flowann::detail::required_id_bits(rows);
  auto const row_bytes             = flowann::detail::packed_row_bytes(2, bits, 4);
  params.device_graph_budget_bytes = (rows / 2) * row_bytes;

  cuvs::neighbors::vpq_params vpq_params;
  vpq_params.pq_dim                          = 4;
  vpq_params.vq_n_centers                    = 8;
  vpq_params.kmeans_n_iters                  = 2;
  vpq_params.max_train_points_per_pq_code    = 16;
  vpq_params.max_train_points_per_vq_cluster = 16;
  auto built                                 = flowann::build(res, params, vpq_params, dataset);
  ASSERT_EQ(built.dataset->n_rows(), rows);
  ASSERT_EQ(built.dataset->dim(), dim);
  ASSERT_EQ(built.index.graph_degree(), 8);
  ASSERT_EQ(built.index.resident_degree(), 4);
  ASSERT_EQ(built.index.cross_degree(), 4);

  auto query_host = raft::make_host_matrix<float, int64_t>(2, dim);
  std::copy_n(host.data_handle(), query_host.size(), query_host.data_handle());
  auto queries = raft::make_device_matrix<float, int64_t>(res, 2, dim);
  raft::copy(queries.data_handle(),
             query_host.data_handle(),
             query_host.size(),
             raft::resource::get_cuda_stream(res));
  auto neighbors = raft::make_device_matrix<std::uint32_t, int64_t>(res, 2, 2);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 2, 2);
  flowann::search_context context(built.index.cross_graph());
  flowann::search_params search_params;
  search_params.algo           = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
  search_params.itopk_size     = 32;
  search_params.search_width   = 1;
  search_params.max_iterations = 16;
  search_params.max_queries    = 2;
  search_params.smem_dtype     = cuvs::neighbors::cagra::internal_dtype::F16;
  flowann::search(res,
                  search_params,
                  built.index,
                  context,
                  raft::make_const_mdspan(queries.view()),
                  neighbors.view(),
                  distances.view());
  auto neighbor_host = raft::make_host_matrix<std::uint32_t, int64_t>(2, 2);
  raft::copy(neighbor_host.data_handle(),
             neighbors.data_handle(),
             neighbors.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  for (std::size_t i = 0; i < neighbor_host.size(); ++i) {
    EXPECT_LT(neighbor_host.data_handle()[i], rows);
  }
}

TEST(FlowannBuild, IntegratedHostBuildGroupsAndReordersVpqIndex)
{
  constexpr int64_t rows = 260;
  constexpr int64_t dim  = 16;
  raft::resources res;
  auto host = raft::make_host_matrix<float, int64_t>(rows, dim);
  for (int64_t row = 0; row < rows; ++row) {
    auto const group = row / 65;
    for (int64_t column = 0; column < dim; ++column) {
      host(row, column) =
        static_cast<float>(group * 100 + column) + static_cast<float>(row % 65) * 0.001f;
    }
  }
  auto dataset = cuvs::neighbors::make_host_standard_dataset_view(host.view());

  flowann::index_params params;
  params.cagra_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.cagra_params.graph_degree              = 8;
  params.cagra_params.intermediate_graph_degree = 16;
  params.cagra_params.graph_build_params =
    cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
      16, cuvs::distance::DistanceType::L2Expanded);
  params.node_per_cacheline        = 2;
  params.device_graph_budget_bytes = (rows / 2) * 64;
  // Small groups force cross edges, so the average-local-degree heuristic would underfill rows.
  params.grouping.n_groups              = 32;
  params.grouping.n_bits                = 24;
  params.grouping.training_rows         = 128;
  params.grouping.assignment_batch_rows = 64;
  params.grouping.kmeans_n_iters        = 4;
  params.grouping.validate              = true;

  cuvs::neighbors::vpq_params vpq_params;
  vpq_params.pq_dim                          = 4;
  vpq_params.vq_n_centers                    = 8;
  vpq_params.kmeans_n_iters                  = 2;
  vpq_params.max_train_points_per_pq_code    = 16;
  vpq_params.max_train_points_per_vq_cluster = 16;
  auto built                                 = flowann::build(res, params, vpq_params, dataset);

  EXPECT_EQ(built.dataset->n_rows(), rows);
  EXPECT_EQ(built.dataset->dim(), dim);
  EXPECT_EQ(built.index.graph_degree(), 8);
  EXPECT_EQ(built.index.resident_degree(), 0);
  EXPECT_GT(built.index.inner_graph().size(), 0);
  EXPECT_EQ(built.index.inner_graph().extent(1), 50);  // Two headers plus 16 24-bit IDs.

  auto source_view = built.index.source_indices();
  ASSERT_TRUE(source_view.has_value());
  auto source = raft::make_host_vector<std::uint32_t, int64_t>(rows);
  raft::copy(
    source.data_handle(), source_view->data_handle(), rows, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  auto ids = std::set<std::uint32_t>();
  for (int64_t row = 0; row < rows; ++row) {
    ids.insert(source(row));
  }
  EXPECT_EQ(ids.size(), rows);
  EXPECT_EQ(*ids.begin(), 0);
  EXPECT_EQ(*ids.rbegin(), rows - 1);
}

TEST(FlowannBuild, SupportsIvfPqAndIterativeCagraGraphBuilders)
{
  constexpr int64_t rows = 512;
  constexpr int64_t dim  = 16;
  raft::resources res;
  auto host = raft::make_host_matrix<float, int64_t>(rows, dim);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < dim; ++column) {
      host(row, column) = static_cast<float>((row * 37 + column * 13) % 211) * 0.01f;
    }
  }
  auto dataset = cuvs::neighbors::make_device_padded_dataset(res, host.view());

  for (bool iterative : {false, true}) {
    flowann::index_params params;
    params.grouping.enabled                       = false;
    params.cagra_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
    params.cagra_params.graph_degree              = 8;
    params.cagra_params.intermediate_graph_degree = 16;
    params.cagra_params.graph_build_params =
      iterative
        ? cuvs::neighbors::cagra::graph_build_params_t{cuvs::neighbors::cagra::graph_build_params::
                                                         iterative_search_params{}}
        : cuvs::neighbors::cagra::graph_build_params_t{
            cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
              raft::matrix_extent<int64_t>(rows, dim), params.cagra_params.metric)};

    auto built = flowann::build(res, params, dataset->as_dataset_view());
    EXPECT_EQ(built.index.size(), rows);
    EXPECT_EQ(built.index.graph_degree(), 8);
    EXPECT_EQ(built.index.resident_degree(), 0);
    EXPECT_EQ(built.index.cross_degree(), 8);

    if (iterative) {
      // Upstream iterative CAGRA now returns a device graph and requires padded device input.
      // FlowANN's host builder must stage that input and bring the graph back for tiering.
      cuvs::neighbors::vpq_params vpq_params;
      vpq_params.pq_dim                          = 4;
      vpq_params.vq_n_centers                    = 8;
      vpq_params.kmeans_n_iters                  = 2;
      vpq_params.max_train_points_per_pq_code    = 16;
      vpq_params.max_train_points_per_vq_cluster = 16;
      auto host_built                            = flowann::build(
        res, params, vpq_params, cuvs::neighbors::make_host_standard_dataset_view(host.view()));
      EXPECT_EQ(host_built.index.size(), rows);
      EXPECT_EQ(host_built.index.graph_degree(), 8);
      EXPECT_EQ(host_built.index.cross_degree(), 8);
    }
  }
}

TEST(FlowannBuild, SearchesZeroAndFullResidentLayouts)
{
  constexpr int64_t rows         = 128;
  constexpr int64_t dim          = 8;
  constexpr std::uint32_t degree = 8;
  raft::resources res;
  auto host = raft::make_host_matrix<float, int64_t>(rows, dim);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < dim; ++column) {
      host(row, column) = static_cast<float>((row * 31 + column * 5) % 149) * 0.01f;
    }
  }
  auto dataset = cuvs::neighbors::make_device_padded_dataset(res, host.view());

  for (bool all_resident : {false, true}) {
    flowann::index_params params;
    params.grouping.enabled                       = false;
    params.cagra_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
    params.cagra_params.graph_degree              = degree;
    params.cagra_params.intermediate_graph_degree = 16;
    params.cagra_params.graph_build_params =
      cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
        16, cuvs::distance::DistanceType::L2Expanded);
    if (all_resident) {
      auto const bits                  = flowann::detail::required_id_bits(rows);
      auto const row_bytes             = flowann::detail::packed_row_bytes(2, bits, degree);
      params.device_graph_budget_bytes = (rows / 2) * row_bytes;
    }

    auto built = flowann::build(res, params, dataset->as_dataset_view());
    EXPECT_EQ(built.index.resident_degree(), all_resident ? degree : 0);
    EXPECT_EQ(built.index.cross_degree(), all_resident ? 0 : degree);

    auto query_host = raft::make_host_matrix<float, int64_t>(2, dim);
    std::copy_n(host.data_handle(), query_host.size(), query_host.data_handle());
    auto queries = raft::make_device_matrix<float, int64_t>(res, 2, dim);
    raft::copy(queries.data_handle(),
               query_host.data_handle(),
               query_host.size(),
               raft::resource::get_cuda_stream(res));
    auto neighbors = raft::make_device_matrix<std::uint32_t, int64_t>(res, 2, 2);
    auto distances = raft::make_device_matrix<float, int64_t>(res, 2, 2);
    flowann::search_context context(built.index.cross_graph());
    flowann::search_params search_params;
    search_params.algo           = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
    search_params.itopk_size     = 32;
    search_params.search_width   = 1;
    search_params.max_iterations = 16;
    search_params.max_queries    = 2;
    flowann::search(res,
                    search_params,
                    built.index,
                    context,
                    raft::make_const_mdspan(queries.view()),
                    neighbors.view(),
                    distances.view());
    auto neighbor_host = raft::make_host_matrix<std::uint32_t, int64_t>(2, 2);
    raft::copy(neighbor_host.data_handle(),
               neighbors.data_handle(),
               neighbors.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    for (std::size_t i = 0; i < neighbor_host.size(); ++i) {
      EXPECT_LT(neighbor_host.data_handle()[i], rows);
    }
  }
}

TEST(FlowannBuild, AutomaticGroupingCapacityPlan)
{
  using flowann::detail::plan_grouping;
  for (auto const [rows, bits] : std::vector<std::pair<std::uint64_t, std::uint16_t>>{
         {1, 4}, {16, 4}, {17, 8}, {256, 8}, {257, 12}, {1u << 20, 20}, {1u << 24, 24}}) {
    EXPECT_EQ(plan_grouping(rows).n_bits, bits);
    EXPECT_EQ(plan_grouping(rows).n_groups, 1);
  }
  EXPECT_EQ(plan_grouping((1u << 24) + 1).n_groups, 2);
  EXPECT_EQ(plan_grouping(87'555'327).n_groups, 6);
  EXPECT_EQ(plan_grouping(10ull * (1u << 24)).n_groups, 11);
  EXPECT_EQ(plan_grouping(1'000'000'000).n_groups, 66);
  // Arithmetic scalability only: public graph construction still uses uint32 IDs.
  EXPECT_EQ(plan_grouping(10'000'000'000ull).n_groups, 656);
  EXPECT_EQ(plan_grouping(16, 4).upper, 16);
  EXPECT_THROW(plan_grouping(17, 4, 1), raft::exception);
  EXPECT_THROW(plan_grouping(0), raft::exception);
  EXPECT_THROW(plan_grouping(256, 33), raft::exception);
  EXPECT_THROW(plan_grouping(256, 0, 0, 0), raft::exception);
  for (std::uint32_t leaves : {2, 16, 17, 32, 66, 656, 65536}) {
    auto check = [&](auto&& self, std::uint32_t n) -> std::uint32_t {
      if (n == 1) { return 1; }
      auto children = flowann::detail::grouping_children(n);
      EXPECT_LE(children.size(), 16);
      std::uint32_t total = 0;
      for (auto child : children) {
        total += self(self, child);
      }
      EXPECT_EQ(total, n);
      return total;
    };
    EXPECT_EQ(check(check, leaves), leaves);
  }
}

TEST(FlowannBuild, AutomaticWidthIsIndependentOfGraphBudget)
{
  raft::resources res;
  constexpr int64_t n = 256, dim = 4, degree = 32;
  auto rows  = raft::make_host_matrix<float, int64_t>(n, dim);
  auto graph = raft::make_host_matrix<std::uint32_t, int64_t>(n, degree);
  auto dataset =
    cuvs::neighbors::make_host_standard_dataset_view(raft::make_const_mdspan(rows.view()));
  flowann::index_params params;
  // One group needs neither training nor graph inspection.
  for (auto budget : {0, 172, 194}) {
    params.device_graph_budget_bytes = (n / 2) * budget;
    auto result = flowann::group_graph(res, params, raft::make_const_mdspan(graph.view()), dataset);
    EXPECT_EQ(result.n_bits, 8);
    EXPECT_EQ(result.n_groups, 1);
    EXPECT_TRUE(std::all_of(result.labels.data_handle(),
                            result.labels.data_handle() + n,
                            [](auto label) { return label == 0; }));
  }
  params.grouping.n_bits = 24;
  EXPECT_EQ(
    flowann::group_graph(res, params, raft::make_const_mdspan(graph.view()), dataset).n_bits, 24);
}

TEST(FlowannBuild, HierarchicalGroupingEnforcesEverySubtreeCapacity)
{
  raft::resources res;
  constexpr int64_t n = 15'200, dim = 2, degree = 4;
  auto rows  = raft::make_host_matrix<float, int64_t>(n, dim);
  auto graph = raft::make_host_matrix<std::uint32_t, int64_t>(n, degree);
  for (int64_t i = 0; i < n; ++i) {
    rows(i, 0) = float(i / 230);
    rows(i, 1) = float(i % 230) * .001f;
    for (int64_t j = 0; j < degree; ++j) {
      graph(i, j) = (i + j + 1) % n;
    }
  }
  auto dataset =
    cuvs::neighbors::make_host_standard_dataset_view(raft::make_const_mdspan(rows.view()));
  flowann::index_params params;
  params.grouping.n_bits                = 8;
  params.grouping.kmeans_n_iters        = 4;
  params.grouping.assignment_batch_rows = 97;
  auto grouped = flowann::group_graph(res, params, raft::make_const_mdspan(graph.view()), dataset);
  ASSERT_EQ(grouped.n_bits, 8);
  ASSERT_EQ(grouped.n_groups, 66);
  auto plan = flowann::detail::plan_grouping(n, 8);
  std::vector<std::uint64_t> sizes(66);
  std::uint64_t local = 0;
  for (int64_t i = 0; i < n; ++i) {
    ASSERT_LT(grouped.labels(i), 66);
    ++sizes[grouped.labels(i)];
    for (int64_t j = 0; j < degree; ++j) {
      local += grouped.labels(i) == grouped.labels(graph(i, j));
    }
  }
  auto check = [&](auto&& self, std::uint32_t first, std::uint32_t leaves) -> void {
    auto count =
      std::accumulate(sizes.begin() + first, sizes.begin() + first + leaves, std::uint64_t{0});
    EXPECT_GE(count, leaves * plan.lower);
    EXPECT_LE(count, leaves * plan.upper);
    EXPECT_LE(count, leaves * plan.capacity);
    if (leaves == 1) { return; }
    for (auto child : flowann::detail::grouping_children(leaves)) {
      self(self, first, child);
      first += child;
    }
  };
  check(check, 0, 66);
  EXPECT_GT(double(local) / (n * degree), .8);
}

TEST(FlowannBuild, CapacityRepairMovesCheapestPointsAndHandlesTies)
{
  // The first row is far from group 1; row-order repair would move the wrong point.
  float costs[] = {0, 100, 0, 1, 0, 20, 20, 0};
  std::vector<std::uint8_t> labels{0, 0, 0, 1};
  auto sizes = flowann::detail::rebalance_group_distances(costs, labels, {2, 2}, {2, 2});
  EXPECT_EQ(labels, (std::vector<std::uint8_t>{0, 1, 0, 1}));
  EXPECT_EQ(sizes, (std::vector<std::uint64_t>{2, 2}));
  // All points initially choose the same centroid, with unequal subtree capacities.
  std::vector<float> tied_costs(100 * 3, 0);
  labels.assign(100, 0);
  sizes = flowann::detail::rebalance_group_distances(
    tied_costs.data(), labels, {40, 20, 20}, {50, 25, 25});
  EXPECT_EQ(sizes, (std::vector<std::uint64_t>{50, 25, 25}));
  labels = {0, 0, 0, 1};
  // No overflow, but the lower-bound repair must still choose the nearest eligible point.
  sizes = flowann::detail::rebalance_group_distances(costs, labels, {1, 2}, {4, 4});
  EXPECT_EQ(labels, (std::vector<std::uint8_t>{0, 1, 0, 1}));
}

TEST(FlowannBuild, LargeCapacityRepairPreservesDistanceAndIdPriority)
{
  // Exceed the parallel radix-selection threshold; include negative costs and a tie at zero.
  constexpr std::size_t rows = 1'050'000;
  std::vector<float> costs(rows * 2);
  std::vector<std::uint8_t> labels(rows, 0);
  for (std::size_t row = 0; row < rows; ++row) {
    costs[row * 2 + 1] = static_cast<int>(row % 1000) - 500.0f;
  }
  auto const requested = rows / 2 + 1;
  auto sizes           = flowann::detail::rebalance_group_distances(
    costs.data(), labels, {rows - requested, requested}, {rows - requested, requested});
  EXPECT_EQ(sizes[1], requested);
  for (std::size_t row = 0; row < rows; ++row) {
    ASSERT_EQ(labels[row], (row % 1000 < 500 || row == 500) ? 1 : 0) << row;
  }
}

TEST(FlowannBuild, SearchesAutomaticAndExplicitWidths)
{
  constexpr int64_t rows         = 128;
  constexpr int64_t dim          = 8;
  constexpr std::uint32_t degree = 8;
  raft::resources res;
  auto host = raft::make_host_matrix<float, int64_t>(rows, dim);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < dim; ++column) {
      host(row, column) = static_cast<float>((row * 31 + column * 5) % 149) * 0.01f;
    }
  }
  auto dataset = cuvs::neighbors::make_device_padded_dataset(res, host.view());

  for (std::uint16_t requested_bits : {0, 20}) {
    flowann::index_params params;
    params.grouping.enabled                       = true;
    params.grouping.n_bits                        = requested_bits;
    params.grouping.validate                      = true;
    params.cagra_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
    params.cagra_params.graph_degree              = degree;
    params.cagra_params.intermediate_graph_degree = 16;
    params.cagra_params.graph_build_params =
      cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
        16, cuvs::distance::DistanceType::L2Expanded);
    {
      auto const bits                  = requested_bits == 0 ? 8 : requested_bits;
      auto const row_bytes             = flowann::detail::packed_row_bytes(2, bits, degree);
      params.device_graph_budget_bytes = (rows / 2) * row_bytes;
    }

    auto built = flowann::build(res, params, dataset->as_dataset_view());
    ASSERT_EQ(built.index.n_bits(), requested_bits == 0 ? 8 : requested_bits);
    EXPECT_EQ(built.index.resident_degree(), 0);
    EXPECT_EQ(built.index.cross_degree(), degree);

    auto query_host = raft::make_host_matrix<float, int64_t>(2, dim);
    std::copy_n(host.data_handle(), query_host.size(), query_host.data_handle());
    auto queries = raft::make_device_matrix<float, int64_t>(res, 2, dim);
    raft::copy(queries.data_handle(),
               query_host.data_handle(),
               query_host.size(),
               raft::resource::get_cuda_stream(res));
    auto neighbors = raft::make_device_matrix<std::uint32_t, int64_t>(res, 2, 2);
    auto distances = raft::make_device_matrix<float, int64_t>(res, 2, 2);
    flowann::search_context context(built.index.cross_graph());
    flowann::search_params search_params;
    search_params.algo           = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
    search_params.itopk_size     = 32;
    search_params.search_width   = 1;
    search_params.max_iterations = 16;
    search_params.max_queries    = 2;
    flowann::search(res,
                    search_params,
                    built.index,
                    context,
                    raft::make_const_mdspan(queries.view()),
                    neighbors.view(),
                    distances.view());
    auto neighbor_host = raft::make_host_matrix<std::uint32_t, int64_t>(2, 2);
    raft::copy(neighbor_host.data_handle(),
               neighbors.data_handle(),
               neighbors.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    EXPECT_EQ(neighbor_host(0, 0), 0);
    EXPECT_EQ(neighbor_host(1, 0), 1);
    for (std::size_t i = 0; i < neighbor_host.size(); ++i) {
      EXPECT_LT(neighbor_host.data_handle()[i], rows);
    }
  }
}
