/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_resources_snmg.hpp>
#include <raft/core/host_mdspan.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace cuvs::neighbors::mg {

/**
 * A multi-GPU CAGRA index built from a padded host dataset takes the row stride as its dimension:
 * the queries have to be as wide as the (padded) rows. The padding is zero, so the distances equal
 * those of the original vectors.
 */
class mg_cagra_padded_host_test : public ::testing::Test {
 protected:
  static constexpr int64_t n_rows    = 4000;
  static constexpr int64_t n_queries = 64;
  static constexpr int64_t k         = 1;

  void run(int64_t dim)
  {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> rows(n_rows * dim);
    for (auto& v : rows) {
      v = dist(rng);
    }
    auto dataset =
      raft::make_host_matrix_view<const float, int64_t, raft::row_major>(rows.data(), n_rows, dim);

    cuvs::neighbors::mg_index_params<cagra::index_params> params;
    params.graph_degree              = 32;
    params.intermediate_graph_degree = 64;

    // The aligned rows are used in place; the other rows are padded in a copy.
    bool const aligned = cuvs::neighbors::matrix_row_width_matches_cagra_required(dataset);
    auto owned = aligned ? nullptr : cuvs::neighbors::make_host_padded_dataset(clique_, dataset);
    auto index =
      aligned
        ? cagra::build(clique_, params, cuvs::neighbors::make_host_padded_dataset_view(dataset))
        : cagra::build(clique_, params, std::as_const(*owned).view());
    int64_t const index_dim = index.ann_interfaces_[0].index_->dim();
    ASSERT_GE(index_dim, dim);
    ASSERT_EQ(aligned, index_dim == dim);

    // The first rows serve as queries: zero-pad them to the index width.
    std::vector<float> queries(n_queries * index_dim, 0.0f);
    for (int64_t i = 0; i < n_queries; i++) {
      std::copy_n(rows.data() + i * dim, dim, queries.data() + i * index_dim);
    }
    std::vector<int64_t> neighbors(n_queries * k);
    std::vector<float> distances(n_queries * k);

    mg_search_params<cagra::search_params> search_params;
    cagra::search(
      clique_,
      index,
      search_params,
      raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
        queries.data(), n_queries, index_dim),
      raft::make_host_matrix_view<int64_t, int64_t, raft::row_major>(
        neighbors.data(), n_queries, k),
      raft::make_host_matrix_view<float, int64_t, raft::row_major>(distances.data(), n_queries, k));
    raft::resource::sync_stream(clique_);

    int64_t found = 0;
    for (int64_t i = 0; i < n_queries; i++) {
      found += neighbors[i] == i ? 1 : 0;
    }
    ASSERT_GE(found, n_queries * 9 / 10) << "dim " << dim << ", index dim " << index_dim;
  }

  raft::device_resources_snmg clique_;
};

TEST_F(mg_cagra_padded_host_test, aligned_rows_are_viewed) { this->run(8); }

TEST_F(mg_cagra_padded_host_test, unaligned_rows_take_the_padded_width) { this->run(7); }

}  // namespace cuvs::neighbors::mg
