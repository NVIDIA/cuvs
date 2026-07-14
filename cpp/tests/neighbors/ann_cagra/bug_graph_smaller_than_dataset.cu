/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>

namespace cuvs::neighbors::cagra {

/**
 * @brief Test verifying graph.extent(0) is used for random seed selection
 *
 * This test ensures that CAGRA search kernels correctly use graph.extent(0)
 * (graph size) rather than dataset.size for random seed node selection.
 *
 * The bug: random seed selection previously used dataset_desc.size, which
 * could cause OOB access if the graph size differed from dataset size
 * (e.g., in CAGRA-Q iterative builds with compression).
 *
 * The fix: kernels now receive graph.extent(0) as graph_size parameter,
 * ensuring seeds are always within valid graph node range [0, graph_size).
 */
class cagra_graph_smaller_than_dataset_test : public ::testing::Test {
 public:
  using data_type  = float;
  using index_type = uint32_t;

 protected:
  void run()
  {
    // Create a dataset with 10000 points
    constexpr int64_t n_dataset = 10000;
    constexpr int64_t n_dim     = 128;
    constexpr int64_t n_queries = 100;
    constexpr int64_t k         = 10;

    // Build index normally
    auto dataset = raft::make_device_matrix<data_type, int64_t>(res, n_dataset, n_dim);
    raft::random::RngState r(1234ULL);
    raft::random::uniform(
      res, r, dataset.data_handle(), n_dataset * n_dim, data_type(-1), data_type(1));

    cagra::index_params index_params;
    index_params.graph_degree              = 32;
    index_params.intermediate_graph_degree = 64;

    auto index = cagra::build(res, index_params, raft::make_const_mdspan(dataset.view()));
    raft::resource::sync_stream(res);

    // Get the graph from the index
    auto original_graph = index.graph();
    ASSERT_EQ(original_graph.extent(0), n_dataset);

    // Recreate the bug scenario: LARGE dataset, SMALL graph
    // (like iterative_build_graph does in intermediate iterations)
    constexpr int64_t n_graph = n_dataset / 2;  // Only 5000 nodes in graph

    // Step 1: Build index on SMALL subset (5000 points)
    auto small_dataset_view = raft::make_device_matrix_view<const data_type, int64_t>(
      dataset.data_handle(), n_graph, n_dim);

    cagra::index_params small_index_params;
    small_index_params.graph_degree = 32;
    auto small_index                = cagra::build(res, small_index_params, small_dataset_view);
    raft::resource::sync_stream(res);

    // Step 2: Update to FULL dataset (10000 points) but keep small graph (5000 nodes)
    // This creates the exact bug scenario: dataset.size=10000, graph.extent(0)=5000
    small_index.update_dataset(res, raft::make_const_mdspan(dataset.view()));

    // Verify the mismatch - THIS IS THE BUG SCENARIO!
    ASSERT_EQ(small_index.graph().extent(0), n_graph);             // Graph has 5000 nodes
    ASSERT_EQ(small_index.size(), n_dataset);                      // Dataset has 10000 points
    ASSERT_NE(small_index.graph().extent(0), small_index.size());  // Mismatch!

    // Create queries
    auto queries = raft::make_device_matrix<data_type, int64_t>(res, n_queries, n_dim);
    raft::random::uniform(
      res, r, queries.data_handle(), n_queries * n_dim, data_type(-1), data_type(1));

    // Allocate output
    auto neighbors = raft::make_device_matrix<index_type, int64_t>(res, n_queries, k);
    auto distances = raft::make_device_matrix<data_type, int64_t>(res, n_queries, k);

    // Setup search params
    cagra::search_params search_params;
    search_params.itopk_size     = 64;
    search_params.search_width   = 1;
    search_params.max_iterations = 10;
    search_params.algo           = cagra::search_algo::SINGLE_CTA;

    // THIS SHOULD NOT CRASH OR CAUSE OOB ACCESS
    // Before fix: random seeds use dataset.size (10000) -> tries to access graph[7000] -> CRASH!
    // After fix: random seeds use graph.extent(0) (5000) -> only accesses graph[0-4999] -> SAFE!
    cagra::search(res,
                  search_params,
                  small_index,
                  raft::make_const_mdspan(queries.view()),
                  neighbors.view(),
                  distances.view());

    raft::resource::sync_stream(res);

    // Verify results are valid (neighbors should be < graph size)
    auto neighbors_host = raft::make_host_matrix<index_type, int64_t>(n_queries, k);
    raft::copy(neighbors_host.data_handle(),
               neighbors.data_handle(),
               n_queries * k,
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    // All neighbor indices should be valid (< n_graph)
    for (int64_t i = 0; i < n_queries * k; i++) {
      ASSERT_LT(neighbors_host.data_handle()[i], n_graph)
        << "Neighbor index " << neighbors_host.data_handle()[i] << " is >= graph size " << n_graph;
    }

    // Test with MULTI_CTA algorithm as well (also had the same bug)
    search_params.algo = cagra::search_algo::MULTI_CTA;

    cagra::search(res,
                  search_params,
                  small_index,
                  raft::make_const_mdspan(queries.view()),
                  neighbors.view(),
                  distances.view());

    raft::resource::sync_stream(res);

    // Verify again
    raft::copy(neighbors_host.data_handle(),
               neighbors.data_handle(),
               n_queries * k,
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    for (int64_t i = 0; i < n_queries * k; i++) {
      ASSERT_LT(neighbors_host.data_handle()[i], n_graph)
        << "Neighbor index " << neighbors_host.data_handle()[i] << " is >= graph size " << n_graph
        << " (MULTI_CTA)";
    }
  }

 private:
  raft::resources res;
};

TEST_F(cagra_graph_smaller_than_dataset_test, search_with_smaller_graph) { this->run(); }

}  // namespace cuvs::neighbors::cagra
