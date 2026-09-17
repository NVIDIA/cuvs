/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../cagra_padded_build_helpers.cuh"
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/rng.cuh>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace cuvs::neighbors::cagra {

template <typename DataT>
class CagraIterativeBuildBugTest : public ::testing::Test {
 public:
  using data_type = DataT;

 protected:
  // The bug manifests when graph_degree is equal to intermediate_graph_degree
  // see issue https://github.com/rapidsai/cuvs/issues/1818
  static auto bug_index_params() -> cagra::index_params
  {
    cagra::index_params index_params;
    // The bug manifests when graph_degree is equal to intermediate_graph_degree
    // see issue https://github.com/nvidia/cuvs/issues/1818
    index_params.graph_degree              = 16;
    index_params.intermediate_graph_degree = 16;

    // Use iterative CAGRA search for graph building
    index_params.graph_build_params = graph_build_params::iterative_search_params();
    return index_params;
  }

  void run()
  {
    auto index_params = bug_index_params();

    cuvs::neighbors::test::padded_device_matrix_for_cagra<data_type> padded(
      res, raft::make_const_mdspan(dataset->view()));
    auto cagra_index = cagra::build(res, index_params, padded.view);
    cagra_index      = cagra::update_dataset(res, std::move(cagra_index), padded.view);
    raft::resource::sync_stream(res);

    // Verify the index was built successfully
    ASSERT_GT(cagra_index.size(), 0);
    ASSERT_EQ(cagra_index.dim(), n_dim);
  }

  // Same bug, reached through iterative CAGRA-Q: the graph is built from a PQ-compressed dataset,
  // so the searches driving the build run on compressed rows instead of dense ones.
  void run_compressed()
  {
    cuvs::neighbors::vpq_params vpq_params;
    // pq_len = n_dim / pq_dim must be 2, 4 or 8 for CAGRA-Q. Codebook quality is irrelevant here,
    // since only graph construction is under test, so training stays short.
    vpq_params.pq_dim         = static_cast<uint32_t>(n_dim / 4);
    vpq_params.vq_n_centers   = 64;
    vpq_params.kmeans_n_iters = 5;

    auto compressed = cuvs::preprocessing::quantize::pq::make_vpq_dataset(
      res, vpq_params, raft::make_const_mdspan(dataset->view()));

    // No padding and no attach step: a compressed dataset is read through its own view, which the
    // index holds on to, so `compressed` has to outlive the index.
    auto cagra_index = cagra::build(res, bug_index_params(), compressed.as_dataset_view());
    raft::resource::sync_stream(res);

    ASSERT_GT(cagra_index.size(), 0);
    ASSERT_EQ(cagra_index.dim(), n_dim);
    ASSERT_EQ(cagra_index.graph_degree(), 16u);
  }

  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<data_type, int64_t>(res, n_samples, n_dim));
    raft::random::RngState r(1234ULL);

    // Generate random data based on type
    if constexpr (std::is_same_v<data_type, float>) {
      raft::random::normal(
        res, r, dataset->data_handle(), n_samples * n_dim, data_type(0), data_type(1));
    } else if constexpr (std::is_same_v<data_type, int8_t>) {
      raft::random::uniformInt(
        res, r, dataset->data_handle(), n_samples * n_dim, int8_t(-128), int8_t(127));
    } else if constexpr (std::is_same_v<data_type, uint8_t>) {
      raft::random::uniformInt(
        res, r, dataset->data_handle(), n_samples * n_dim, uint8_t(0), uint8_t(255));
    }
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

  constexpr static int64_t n_samples = 10000;
  constexpr static int64_t n_dim     = 1024;
};

// Instantiate test for different data types
using TestTypes = ::testing::Types<float, int8_t, uint8_t>;
TYPED_TEST_SUITE(CagraIterativeBuildBugTest, TestTypes);

TYPED_TEST(CagraIterativeBuildBugTest, IterativeBuildTest) { this->run(); }

TYPED_TEST(CagraIterativeBuildBugTest, IterativeBuildFromCompressedDatasetTest)
{
  this->run_compressed();
}

}  // namespace cuvs::neighbors::cagra
