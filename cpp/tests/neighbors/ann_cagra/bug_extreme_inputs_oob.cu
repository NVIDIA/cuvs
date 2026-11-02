/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../cagra_padded_build_helpers.cuh"
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/rng.cuh>

#include <cstdint>

namespace cuvs::neighbors::cagra {

class cagra_extreme_inputs_oob_test : public ::testing::Test {
 public:
  using data_type = float;

 protected:
  void run()
  {
    cagra::index_params ix_ps;
    graph_build_params::ivf_pq_params gb_params{};
    gb_params.refinement_rate       = 2;
    ix_ps.graph_build_params        = gb_params;
    ix_ps.graph_degree              = 64;
    ix_ps.intermediate_graph_degree = 128;

    try {
      cuvs::neighbors::test::padded_device_matrix_for_cagra<data_type> padded(
        res, raft::make_const_mdspan(dataset->view()));
      [[maybe_unused]] auto ix = cagra::build(res, ix_ps, padded.view);
      raft::resource::sync_stream(res);
    } catch (const std::exception&) {
      SUCCEED();
      return;
    }
    FAIL();
  }

  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<data_type, int64_t>(res, n_samples, n_dim));
    raft::random::RngState r(1234ULL);
    raft::random::normal(
      res, r, dataset->data_handle(), n_samples * n_dim, data_type(0), data_type(1e20));
    raft::resource::sync_stream(res);
  }

  void TearDown() override
  {
    dataset.reset();
    raft::resource::sync_stream(res);
  }

 private:
  raft::resources res;
  std::optional<raft::device_matrix<data_type, int64_t>> dataset = std::nullopt;

  constexpr static int64_t n_samples                   = 100000;
  constexpr static int64_t n_dim                       = 200;
  constexpr static cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
};

TEST_F(cagra_extreme_inputs_oob_test, cagra_extreme_inputs_oob_test) { this->run(); }

}  // namespace cuvs::neighbors::cagra
