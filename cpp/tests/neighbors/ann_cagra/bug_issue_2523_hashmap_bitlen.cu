/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../cagra_padded_build_helpers.cuh"
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/random/rng.cuh>

#include <cstdint>
#include <utility>

namespace cuvs::neighbors::cagra {

/**
 * @brief Regression test for issue #2523: hash table sizing must not loop forever.
 *
 * The hash tables in search_plan_impl::calc_hashmap_params() are sized by loops that grow
 * a bit length until the requested table fits, with the supported maximum checked only
 * after the loop. hashmap::get_size() is `1U << bitlen`, which is undefined once bitlen
 * reaches 32 and in practice wraps back to a small value, so a request that needs a table
 * larger than 2^31 entries made the loop condition permanently true. The bit length grew
 * without bound, the post-loop RAFT_EXPECTS was never reached, and the search hung on the
 * host instead of failing.
 *
 * check_params() only caps itopk_size at 1024 for SINGLE_CTA, so MULTI_CTA and MULTI_KERNEL
 * are the two algorithms that can reach the sizing loops with an oversized request.
 *
 * These searches must now raise the existing "hash_bitlen cannot be larger than ..." error
 * rather than hanging. A regression reappears as a test timeout, not a failed assertion.
 */
class cagra_hashmap_bitlen_no_hang_test : public ::testing::Test {
 public:
  using data_type  = float;
  using index_type = uint32_t;

 protected:
  // Large enough that the traversed-node hash table would need more than 2^31 entries,
  // which is what previously drove the sizing loop past the wrap point.
  constexpr static size_t oversized_itopk = 1'100'000'000;
  constexpr static size_t valid_itopk     = 64;

  constexpr static int64_t n_dataset = 1000;
  constexpr static int64_t n_dim     = 32;
  constexpr static int64_t n_queries = 4;
  constexpr static int64_t k         = 10;

  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<data_type, int64_t>(res, n_dataset, n_dim));
    queries.emplace(raft::make_device_matrix<data_type, int64_t>(res, n_queries, n_dim));
    neighbors.emplace(raft::make_device_matrix<index_type, int64_t>(res, n_queries, k));
    distances.emplace(raft::make_device_matrix<data_type, int64_t>(res, n_queries, k));

    raft::random::RngState r(1234ULL);
    raft::random::uniform(
      res, r, dataset->data_handle(), n_dataset * n_dim, data_type(-1), data_type(1));
    raft::random::uniform(
      res, r, queries->data_handle(), n_queries * n_dim, data_type(-1), data_type(1));

    cagra::index_params index_params;
    index_params.graph_degree              = 32;
    index_params.intermediate_graph_degree = 64;

    padded_.emplace(res, raft::make_const_mdspan(dataset->view()));
    index_.emplace(cagra::build(res, index_params, padded_->view));
    raft::resource::sync_stream(res);
  }

  void TearDown() override
  {
    index_.reset();
    padded_.reset();
    dataset.reset();
    queries.reset();
    neighbors.reset();
    distances.reset();
    raft::resource::sync_stream(res);
  }

  void search_with(cagra::search_algo algo, size_t itopk_size, uint32_t max_iterations)
  {
    cagra::search_params search_params;
    search_params.algo           = algo;
    search_params.itopk_size     = itopk_size;
    search_params.search_width   = 8;
    search_params.max_iterations = max_iterations;

    cagra::search(res,
                  search_params,
                  *index_,
                  raft::make_const_mdspan(queries->view()),
                  neighbors->view(),
                  distances->view());
    raft::resource::sync_stream(res);
  }

  raft::resources res;
  std::optional<cuvs::neighbors::test::padded_device_matrix_for_cagra<data_type>> padded_{};
  std::optional<cagra::index<data_type, index_type>> index_         = std::nullopt;
  std::optional<raft::device_matrix<data_type, int64_t>> dataset    = std::nullopt;
  std::optional<raft::device_matrix<data_type, int64_t>> queries    = std::nullopt;
  std::optional<raft::device_matrix<index_type, int64_t>> neighbors = std::nullopt;
  std::optional<raft::device_matrix<data_type, int64_t>> distances  = std::nullopt;
};

// MULTI_CTA sizes the shared traversed-node table from itopk_size and search_width.
// This is the path confirmed to hang before the fix.
TEST_F(cagra_hashmap_bitlen_no_hang_test, MultiCtaOversizedItopkThrows)
{
  EXPECT_THROW(search_with(cagra::search_algo::MULTI_CTA, oversized_itopk, 0), raft::exception);
}

// MULTI_KERNEL reaches the small-hash loop first and then the normal-hash loop.
// max_iterations is pinned so the sizing inputs do not depend on the auto-derived value.
TEST_F(cagra_hashmap_bitlen_no_hang_test, MultiKernelOversizedItopkThrows)
{
  EXPECT_THROW(search_with(cagra::search_algo::MULTI_KERNEL, oversized_itopk, 32), raft::exception);
}

// Guards against the bound rejecting requests that were previously accepted.
TEST_F(cagra_hashmap_bitlen_no_hang_test, ValidItopkStillSucceeds)
{
  EXPECT_NO_THROW(search_with(cagra::search_algo::MULTI_CTA, valid_itopk, 0));
  EXPECT_NO_THROW(search_with(cagra::search_algo::MULTI_KERNEL, valid_itopk, 0));
  EXPECT_NO_THROW(search_with(cagra::search_algo::SINGLE_CTA, valid_itopk, 0));
}

}  // namespace cuvs::neighbors::cagra
