/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"
#include "../cagra_padded_build_helpers.cuh"

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>

#include <cstdint>

namespace cuvs::neighbors::cagra {

class AnnCagraBugMultiCTACrash : public ::testing::TestWithParam<cagra::search_algo> {
 public:
  using data_type = half;

 protected:
  void run()
  {
    cagra::index_params cagra_index_params;
    cagra_index_params.graph_degree              = 32;
    cagra_index_params.intermediate_graph_degree = 48;

    build_padded_.emplace(res, raft::make_const_mdspan(dataset->view()));
    auto cagra_index = cagra::build(res, cagra_index_params, build_padded_->view);
    cagra_index.update_dataset(res, build_padded_->view);
    raft::resource::sync_stream(res);

    cagra::search_params cagra_search_params;
    cagra_search_params.itopk_size        = 32;
    cagra_search_params.thread_block_size = 256;
    cagra_search_params.search_width      = 1;
    cagra_search_params.max_iterations    = 0;
    cagra_search_params.algo = ::testing::TestWithParam<cagra::search_algo>::GetParam();

    // NOTE: when using one resource/stream for everything, the bug is NOT reproducible
    raft::resources res_search;
    cagra::search(res_search,
                  cagra_search_params,
                  cagra_index,
                  raft::make_const_mdspan(queries->view()),
                  neighbors->view(),
                  distances->view());

    raft::resource::sync_stream(res_search);
  }

  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<data_type, int64_t>(res, n_samples, n_dim));
    queries.emplace(raft::make_device_matrix<data_type, int64_t>(res, n_queries, n_dim));
    neighbors.emplace(raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k));
    distances.emplace(raft::make_device_matrix<float, int64_t>(res, n_queries, k));
    raft::random::RngState r(1234ULL);
    InitDataset(res, dataset->data_handle(), n_samples, n_dim, metric, r);
    // NOTE: when initializing queries with "normal" data, the bug is NOT reproducible
    raft::linalg::map(
      res, queries->view(), raft::const_op<data_type>{raft::upper_bound<data_type>()});
    // InitDataset(res, queries->data_handle(), n_queries, n_dim, metric, r);
    raft::resource::sync_stream(res);
  }

  void TearDown() override
  {
    build_padded_.reset();
    dataset.reset();
    queries.reset();
    neighbors.reset();
    distances.reset();
    raft::resource::sync_stream(res);
  }

 private:
  raft::resources res;
  std::optional<cuvs::neighbors::test::padded_device_matrix_for_cagra<data_type>> build_padded_{};
  std::optional<raft::device_matrix<data_type, int64_t>> dataset  = std::nullopt;
  std::optional<raft::device_matrix<data_type, int64_t>> queries  = std::nullopt;
  std::optional<raft::device_matrix<uint32_t, int64_t>> neighbors = std::nullopt;
  std::optional<raft::device_matrix<float, int64_t>> distances    = std::nullopt;

  constexpr static int64_t n_samples                   = 1183514;
  constexpr static int64_t n_dim                       = 100;
  constexpr static int64_t n_queries                   = 30;
  constexpr static int64_t k                           = 10;
  constexpr static cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
};

TEST_P(AnnCagraBugMultiCTACrash, AnnCagraBugMultiCTACrash) { this->run(); }

INSTANTIATE_TEST_CASE_P(AnnCagraBugMultiCTACrashReproducer,
                        AnnCagraBugMultiCTACrash,
                        ::testing::Values(cagra::search_algo::MULTI_CTA));

}  // namespace cuvs::neighbors::cagra
