/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>
#include <cuvs/neighbors/cagra.hpp>
#include <gtest/gtest.h>
#include <raft/core/copy.cuh>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <sstream>
#include <vector>

namespace cagra = cuvs::neighbors::cagra;

TEST(CagraTiered, OwnsDataAcrossMoveAndSerialization)
{
  constexpr int64_t rows = 128, dim = 8, nq = 4, k = 4;
  raft::resources res;
  auto host = raft::make_host_matrix<float, int64_t>(rows, dim);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < dim; ++j) {
      host(i, j) = static_cast<float>((i * 31 + j * 5) % 149) * .01f;
    }
  }
  auto queries = raft::make_device_matrix<float, int64_t>(res, nq, dim);
  raft::copy(
    queries.data_handle(), host.data_handle(), nq * dim, raft::resource::get_cuda_stream(res));
  auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, nq, k);
  auto distances = raft::make_device_matrix<float, int64_t>(res, nq, k);
  cagra::search_params sp;
  sp.algo           = cagra::search_algo::SINGLE_CTA;
  sp.itopk_size     = 32;
  sp.max_iterations = 32;
  sp.tiered.emplace();
  // Include CPU-only, mixed and entirely resident representations.
  for (size_t budget : {size_t{0}, size_t{512}, size_t{4096}}) {
    cagra::index_params bp;
    bp.graph_degree              = 8;
    bp.intermediate_graph_degree = 16;
    bp.graph_build_params =
      cagra::graph_build_params::nn_descent_params(16, cuvs::distance::DistanceType::L2Expanded);
    bp.graph_storage = cagra::graph_storage_kind::tiered;
    bp.tiered.emplace();
    bp.tiered->device_graph_budget_bytes = budget;
    auto input    = cuvs::neighbors::make_device_padded_dataset(res, host.view());
    auto original = cagra::build(res, bp, input->as_dataset_view());
    input.reset();
    auto idx = std::move(original);
    EXPECT_EQ(idx.graph_storage(), cagra::graph_storage_kind::tiered);
    EXPECT_EQ(idx.size(), rows);
    EXPECT_EQ(idx.dim(), dim);
    EXPECT_EQ(idx.graph_degree(), 8);
    EXPECT_THROW((void)idx.graph(), raft::exception);
    auto run = [&](auto const& index) {
      cagra::search(res,
                    sp,
                    index,
                    raft::make_const_mdspan(queries.view()),
                    neighbors.view(),
                    distances.view());
      std::vector<uint32_t> ids(nq * k);
      raft::copy(
        ids.data(), neighbors.data_handle(), ids.size(), raft::resource::get_cuda_stream(res));
      raft::resource::sync_stream(res);
      for (auto id : ids)
        EXPECT_LT(id, rows);
      for (int64_t i = 0; i < nq; ++i)
        EXPECT_EQ(ids[i * k], i);
      return ids;
    };
    auto expected                   = run(idx);
    sp.tiered->keep_pollers_running = true;
    EXPECT_EQ(run(idx), expected);
    sp.tiered->keep_pollers_running = false;
    EXPECT_EQ(run(idx), expected);
    std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
    cagra::serialize(res, saved, idx, true);
    // Destroy both the original index and its polling context before loading.
    idx = cagra::device_padded_index<float>(res);
    cagra::deserialize(res, saved, &idx);
    EXPECT_EQ(idx.graph_storage(), cagra::graph_storage_kind::tiered);
    EXPECT_EQ(run(idx), expected);
    auto replacement = cuvs::neighbors::make_device_padded_dataset(res, host.view());
    EXPECT_THROW(cagra::update_dataset(res, std::move(idx), replacement->as_dataset_view()),
                 raft::exception);
    EXPECT_EQ(run(idx), expected);
  }
}

TEST(CagraTiered, CompressedBuildRoundTrip)
{
  constexpr int64_t rows = 512, dim = 16, nq = 4, k = 8;
  raft::resources res;
  auto host = raft::make_host_matrix<float, int64_t>(rows, dim);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < dim; ++j) {
      host(i, j) = static_cast<float>((i * 23 + j * 11) % 509) * .01f;
    }
  }
  cagra::index_params bp;
  bp.graph_degree              = 8;
  bp.intermediate_graph_degree = 16;
  bp.graph_build_params =
    cagra::graph_build_params::nn_descent_params(16, cuvs::distance::DistanceType::L2Expanded);
  bp.graph_storage = cagra::graph_storage_kind::tiered;
  bp.tiered.emplace();
  bp.tiered->device_graph_budget_bytes = 4096;
  cuvs::neighbors::vpq_params pq;
  pq.pq_dim                          = 4;
  pq.vq_n_centers                    = 8;
  pq.kmeans_n_iters                  = 2;
  pq.max_train_points_per_pq_code    = 16;
  pq.max_train_points_per_vq_cluster = 16;
  auto idx =
    cagra::build(res, bp, pq, cuvs::neighbors::make_host_standard_dataset_view(host.view()));
  EXPECT_EQ(idx.graph_storage(), cagra::graph_storage_kind::tiered);
  auto queries = raft::make_device_matrix<float, int64_t>(res, nq, dim);
  raft::copy(
    queries.data_handle(), host.data_handle(), nq * dim, raft::resource::get_cuda_stream(res));
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
  auto distances = raft::make_device_matrix<float, int64_t>(res, nq, k);
  cagra::search_params sp;
  sp.algo           = cagra::search_algo::MULTI_CTA;
  sp.itopk_size     = 64;
  sp.max_iterations = 32;
  auto run          = [&] {
    cagra::search(
      res, sp, idx, raft::make_const_mdspan(queries.view()), neighbors.view(), distances.view());
    std::vector<int64_t> ids(nq * k);
    raft::copy(
      ids.data(), neighbors.data_handle(), ids.size(), raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    for (auto id : ids) {
      EXPECT_GE(id, 0);
      EXPECT_LT(id, rows);
    }
    return ids;
  };
  auto expected = run();
  auto repeated = run();
  std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
  cagra::serialize(res, saved, idx);
  idx = cagra::device_pq_index<float>(res);
  cagra::deserialize(res, saved, &idx);
  std::stringstream reserialized(std::ios::in | std::ios::out | std::ios::binary);
  cagra::serialize(res, reserialized, idx);
  EXPECT_EQ(saved.str(), reserialized.str());
  auto restored = run();
  for (auto const* result : {&expected, &repeated, &restored}) {
    for (int64_t q = 0; q < nq; ++q) {
      EXPECT_NE(std::find(result->begin() + q * k, result->begin() + (q + 1) * k, q),
                result->begin() + (q + 1) * k);
    }
  }
}
