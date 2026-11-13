/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../src/neighbors/nn_descent.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>

#include <gtest/gtest.h>

namespace cuvs::neighbors::nn_descent::detail {

TEST(NNDescentConvergenceTest, ZeroTerminationThresholdRunsMaxIterations)
{
  constexpr int n_rows         = 33;
  constexpr int dim            = 4;
  constexpr int graph_degree   = 32;
  constexpr int max_iterations = 10;

  raft::resources res;
  auto dataset = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  raft::random::RngState rng(1234ULL);
  raft::random::normal(res, rng, dataset.data_handle(), dataset.size(), 0.1f, 2.0f);

  auto graph = raft::make_host_matrix<int, int64_t, raft::row_major>(n_rows, graph_degree);
  BuildConfig config{.max_dataset_size      = n_rows,
                     .dataset_dim           = dim,
                     .node_degree           = graph_degree,
                     .internal_node_degree  = graph_degree,
                     .max_iterations        = max_iterations,
                     .termination_threshold = 0.0f,
                     .output_graph_degree   = graph_degree};

  GNND<const float, int> nn_descent(res, config);
  nn_descent.build(dataset.data_handle(), n_rows, graph.data_handle(), false, nullptr);

  EXPECT_EQ(nn_descent.num_iterations(), max_iterations);
}

TEST(NNDescentConvergenceTest, OversizedDegreeUsesEffectiveOutputShape)
{
  constexpr int n_rows = 33;
  constexpr int dim    = 4;

  raft::resources res;
  auto dataset = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  raft::random::RngState rng(1234ULL);
  raft::random::normal(res, rng, dataset.data_handle(), dataset.size(), 0.1f, 2.0f);

  index_params params;
  params.graph_degree              = 64;
  params.intermediate_graph_degree = 64;
  params.max_iterations            = 2;

  auto index =
    cuvs::neighbors::nn_descent::build(res, params, raft::make_const_mdspan(dataset.view()));

  EXPECT_EQ(index.graph_degree(), n_rows - 1);
  EXPECT_EQ(index.graph().extent(0), n_rows);
  ASSERT_TRUE(index.distances().has_value());
  EXPECT_EQ(index.distances()->extent(0), n_rows);
  EXPECT_EQ(index.distances()->extent(1), n_rows - 1);
}

}  // namespace cuvs::neighbors::nn_descent::detail
