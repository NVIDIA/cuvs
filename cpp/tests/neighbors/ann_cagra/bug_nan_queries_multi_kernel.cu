/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../cagra_padded_build_helpers.cuh"
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>
#include <limits>
#include <vector>

namespace cuvs::neighbors::cagra {

/**
 * Searching with NaN queries must not access memory out of bounds.
 *
 * The random seed selection of the multi-kernel search used to store an uninitialized index when
 * all candidate distances were NaN, which was later used to read the graph out of bounds.
 */
class cagra_nan_queries_test : public ::testing::TestWithParam<search_algo> {
 protected:
  void run()
  {
    auto stream  = raft::resource::get_cuda_stream(res);
    auto dataset = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
    auto queries = raft::make_device_matrix<float, int64_t>(res, n_queries, dim);
    raft::random::RngState rng(1234ULL);
    raft::random::uniform(res, rng, dataset.data_handle(), dataset.size(), -1.0f, 1.0f);
    raft::random::uniform(res, rng, queries.data_handle(), queries.size(), -1.0f, 1.0f);
    // The second half of the queries is NaN
    raft::matrix::fill(
      res,
      raft::make_device_matrix_view<float, int64_t>(
        queries.data_handle() + n_valid_queries * dim, n_queries - n_valid_queries, dim),
      std::numeric_limits<float>::quiet_NaN());

    index_params ix_ps;
    ix_ps.graph_degree              = 32;
    ix_ps.intermediate_graph_degree = 64;
    cuvs::neighbors::test::padded_device_matrix_for_cagra<float> padded(
      res, raft::make_const_mdspan(dataset.view()));
    auto ix = build(res, ix_ps, padded.view);

    search_params sp;
    sp.algo        = GetParam();
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
    search(
      res, sp, ix, raft::make_const_mdspan(queries.view()), neighbors.view(), distances.view());

    std::vector<uint32_t> neighbors_h(n_queries * k);
    raft::update_host(neighbors_h.data(), neighbors.data_handle(), neighbors_h.size(), stream);
    raft::resource::sync_stream(res);

    // The results of the valid queries must be valid dataset indices
    for (int64_t i = 0; i < n_valid_queries * k; i++) {
      ASSERT_LT(neighbors_h[i], n_rows) << "query " << i / k;
    }
  }

  raft::resources res;

  constexpr static int64_t n_rows          = 10000;
  constexpr static int64_t dim             = 32;
  constexpr static int64_t n_queries       = 64;
  constexpr static int64_t n_valid_queries = 32;
  constexpr static int64_t k               = 16;
};

TEST_P(cagra_nan_queries_test, search) { this->run(); }

INSTANTIATE_TEST_CASE_P(cagra_nan_queries_test,
                        cagra_nan_queries_test,
                        ::testing::Values(search_algo::SINGLE_CTA,
                                          search_algo::MULTI_CTA,
                                          search_algo::MULTI_KERNEL));

}  // namespace cuvs::neighbors::cagra
