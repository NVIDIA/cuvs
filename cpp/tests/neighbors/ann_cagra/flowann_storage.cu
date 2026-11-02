/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * FlowANN is described in "Disentangling Graph Dependencies for Efficient
 * Billion-Scale GPU Vector Search" (OSDI 2026).
 */

#include <neighbors/detail/flowann/flowann.hpp>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <gtest/gtest.h>

namespace flowann = cuvs::neighbors::cagra::experimental::flowann;

TEST(FlowannStorage, OwnsCrossGraphAndPreservesItAcrossMove)
{
  raft::resources res;
  flowann::device_padded_index<float> index(res);
  auto inner  = raft::make_host_matrix<std::uint8_t, int64_t>(0, 0);
  auto cross  = raft::make_host_matrix<std::uint32_t, int64_t>(2, 2);
  cross(0, 0) = 1;
  cross(0, 1) = 1;
  cross(1, 0) = 1;
  cross(1, 1) = 0;
  index.update_graph(
    res, raft::make_const_mdspan(inner.view()), raft::make_const_mdspan(cross.view()), 1, 0);
  cross(0, 1) = 0;
  auto moved  = std::move(index);
  EXPECT_EQ(moved.size(), 2);
  EXPECT_EQ(moved.graph_degree(), 1);
  EXPECT_EQ(moved.resident_degree(), 0);
  EXPECT_EQ(moved.cross_graph()(0, 1), 1);
  raft::resource::sync_stream(res);
}

TEST(FlowannStorage, RejectsInvalidCrossEdgesBeforeReplacingGraph)
{
  raft::resources res;
  flowann::device_padded_index<float> index(res);
  auto inner  = raft::make_host_matrix<std::uint8_t, int64_t>(0, 0);
  auto cross  = raft::make_host_matrix<std::uint32_t, int64_t>(2, 2);
  cross(0, 0) = 1;
  cross(0, 1) = 1;
  cross(1, 0) = 1;
  cross(1, 1) = 0;
  index.update_graph(
    res, raft::make_const_mdspan(inner.view()), raft::make_const_mdspan(cross.view()), 1, 0);
  cross(0, 1) = 2;
  EXPECT_ANY_THROW(index.update_graph(
    res, raft::make_const_mdspan(inner.view()), raft::make_const_mdspan(cross.view()), 1, 0));
  EXPECT_EQ(index.cross_graph()(0, 1), 1);
  EXPECT_EQ(index.size(), 2);
  raft::resource::sync_stream(res);
}

TEST(FlowannStorage, ComputesCosineNormsWithoutSearchImplementation)
{
  raft::resources res;
  auto host    = raft::make_host_matrix<float, int64_t>(2, 2);
  host(0, 0)   = 3.0f;
  host(0, 1)   = 4.0f;
  host(1, 0)   = 0.0f;
  host(1, 1)   = 2.0f;
  auto dataset = cuvs::neighbors::make_device_padded_dataset(res, host.view());
  flowann::device_padded_index<float> index(
    res, cuvs::distance::DistanceType::CosineExpanded, dataset->as_dataset_view());
  ASSERT_TRUE(index.dataset_norms().has_value());
  float norms[2]{};
  raft::copy(norms, index.dataset_norms()->data_handle(), 2, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  EXPECT_FLOAT_EQ(norms[0], 5.0f);
  EXPECT_FLOAT_EQ(norms[1], 2.0f);
}


TEST(FlowannStorage, RejectsStoredSeedsBeforeReplacingGraph)
{
  for (int overload = 0; overload < 4; ++overload) {
    raft::resources res;
    flowann::device_padded_index<float> index(res);
    auto inner = raft::make_host_matrix<std::uint8_t, int64_t>(0, 0);
    auto original = raft::make_host_matrix<std::uint32_t, int64_t>(4, 2);
    for (std::uint32_t row = 0; row < 4; ++row) {
      original(row, 0) = 1;
      original(row, 1) = (row + 1) % 4;
    }
    index.update_graph(res, raft::make_const_mdspan(inner.view()),
                       raft::make_const_mdspan(original.view()), 1, 0);
    std::uint32_t seed = 3;
    index.update_seeds(res, raft::make_host_vector_view<const std::uint32_t, int64_t>(&seed, 1));
    auto replacement = raft::make_host_matrix<std::uint32_t, int64_t>(2, 2);
    replacement(0, 0) = replacement(1, 0) = 1;
    replacement(0, 1) = 1;
    replacement(1, 1) = 0;
    if (overload == 0) {
      EXPECT_ANY_THROW(index.update_graph(res, raft::make_const_mdspan(inner.view()),
                                          raft::make_const_mdspan(replacement.view()), 1, 0));
    } else if (overload == 1) {
      EXPECT_ANY_THROW(index.update_graph(res, std::move(inner), std::move(replacement), 1, 0));
    } else {
      auto device_inner = raft::make_device_matrix<std::uint8_t, int64_t>(res, 0, 0);
      if (overload == 2) {
        EXPECT_ANY_THROW(index.update_graph(res, std::move(device_inner), std::move(replacement), true));
      } else {
        EXPECT_ANY_THROW(index.update_graph(res, std::move(device_inner), std::move(replacement), 1, 0));
      }
    }
    ASSERT_EQ(index.size(), 4) << "overload " << overload;
    EXPECT_EQ(index.cross_graph()(3, 1), 0);
    ASSERT_TRUE(index.seeds().has_value());
    raft::resource::sync_stream(res);
  }
}

TEST(FlowannStorage, RowCountChangeClearsDependentMappings)
{
  raft::resources res;
  flowann::device_padded_index<float> index(res);
  auto inner = raft::make_host_matrix<std::uint8_t, int64_t>(0, 0);
  auto graph = raft::make_host_matrix<std::uint32_t, int64_t>(4, 2);
  for (std::uint32_t row = 0; row < 4; ++row) {
    graph(row, 0) = 1;
    graph(row, 1) = (row + 1) % 4;
  }
  index.update_graph(res, raft::make_const_mdspan(inner.view()),
                     raft::make_const_mdspan(graph.view()), 1, 0);
  std::uint32_t ids[4] = {3, 2, 1, 0};
  std::uint32_t groups[4] = {0, 0, 0, 0};
  std::uint32_t offset = 0;
  std::uint32_t seed = 1;
  index.update_source_indices(res, raft::make_host_vector_view<const std::uint32_t, int64_t>(ids, 4));
  index.update_subgraph_layout(res,
    raft::make_host_vector_view<const std::uint32_t, int64_t>(&offset, 1),
    raft::make_host_vector_view<const std::uint32_t, int64_t>(groups, 4));
  index.update_seeds(res, raft::make_host_vector_view<const std::uint32_t, int64_t>(&seed, 1));
  auto larger = raft::make_host_matrix<std::uint32_t, int64_t>(6, 2);
  for (std::uint32_t row = 0; row < 6; ++row) {
    larger(row, 0) = 1;
    larger(row, 1) = (row + 1) % 6;
  }
  index.update_graph(res, raft::make_const_mdspan(inner.view()),
                     raft::make_const_mdspan(larger.view()), 1, 0);
  EXPECT_EQ(index.size(), 6);
  EXPECT_FALSE(index.source_indices().has_value());
  EXPECT_FALSE(index.subgraph_offsets().has_value());
  EXPECT_FALSE(index.subgraph_ids().has_value());
  EXPECT_FALSE(index.seeds().has_value());
  EXPECT_FALSE(index.seed_cross_graph().has_value());
  EXPECT_FALSE(index.seed_lookup_keys().has_value());
  EXPECT_FALSE(index.seed_lookup_values().has_value());
  raft::resource::sync_stream(res);
}
