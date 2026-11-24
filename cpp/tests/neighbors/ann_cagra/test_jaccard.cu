/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>

namespace cuvs::neighbors::cagra {
TEST(CagraJaccard, EmptyAndFullBitsets)
{
  using T = uint8_t;
  raft::device_resources res;
  constexpr int64_t rows = 128;
  constexpr int64_t dim  = 3;
  auto data              = raft::make_host_matrix<T, int64_t>(rows, dim);
  auto queries           = raft::make_host_matrix<T, int64_t>(2, dim);
  std::fill_n(queries.data_handle(), dim, T(0));
  std::fill_n(queries.data_handle() + dim, dim, T(uint8_t(255)));
  auto device_queries = raft::make_device_matrix<T, int64_t>(res, 2, dim);
  raft::copy(device_queries.data_handle(), queries.data_handle(), queries.size(), res.get_stream());

  index_params params;
  params.metric                    = cuvs::distance::DistanceType::BitwiseJaccard;
  params.graph_degree              = 32;
  params.intermediate_graph_degree = 64;
  params.graph_build_params        = graph_build_params::iterative_search_params{};

  for (bool full : {false, true}) {
    SCOPED_TRACE(full);
    std::fill_n(data.data_handle(), data.size(), full ? T(uint8_t(255)) : T(0));
    auto padded =
      cuvs::neighbors::make_device_padded_dataset(res, raft::make_const_mdspan(data.view()));
    auto index     = build(res, params, padded->view());
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, 2, 1);
    auto distances = raft::make_device_matrix<float, int64_t>(res, 2, 1);
    search(res,
           search_params{},
           index,
           raft::make_const_mdspan(device_queries.view()),
           neighbors.view(),
           distances.view());
    float actual[2];
    raft::copy(actual, distances.data_handle(), 2, res.get_stream());
    res.sync_stream();
    EXPECT_FLOAT_EQ(actual[0], full ? 1.0f : 0.0f);
    EXPECT_FLOAT_EQ(actual[1], full ? 0.0f : 1.0f);
  }
}

TEST(CagraJaccard, ApiChecks)
{
  raft::device_resources res;
  index_params params;
  params.metric = cuvs::distance::DistanceType::BitwiseJaccard;

  // IVF-PQ cannot build a packed Jaccard index.
  auto bytes = raft::make_host_matrix<uint8_t, int64_t>(512, 65);
  std::fill_n(bytes.data_handle(), bytes.size(), uint8_t(0));
  auto padded_bytes =
    cuvs::neighbors::make_device_padded_dataset(res, raft::make_const_mdspan(bytes.view()));
  params.graph_build_params = graph_build_params::ivf_pq_params{};
  EXPECT_THROW(build(res, params, padded_bytes->view()), raft::exception);

  // CAGRA and NN-Descent require unsigned byte inputs.
  auto floats = raft::make_host_matrix<float, int64_t>(512, 65);
  std::fill_n(floats.data_handle(), floats.size(), 0.0f);
  auto padded_floats =
    cuvs::neighbors::make_device_padded_dataset(res, raft::make_const_mdspan(floats.view()));
  params.graph_build_params = graph_build_params::nn_descent_params(128, params.metric);
  EXPECT_THROW(build(res, params, padded_floats->view()), raft::exception);
  nn_descent::index_params nn_params;
  nn_params.metric = params.metric;
  EXPECT_THROW(nn_descent::build(res, nn_params, raft::make_const_mdspan(floats.view())),
               raft::exception);

  auto signed_bytes = raft::make_host_matrix<int8_t, int64_t>(512, 65);
  std::fill_n(signed_bytes.data_handle(), signed_bytes.size(), int8_t(0));
  auto padded_signed =
    cuvs::neighbors::make_device_padded_dataset(res, raft::make_const_mdspan(signed_bytes.view()));
  EXPECT_THROW(build(res, params, padded_signed->view()), raft::exception);
  EXPECT_THROW(nn_descent::build(res, nn_params, raft::make_const_mdspan(signed_bytes.view())),
               raft::exception);
}

}  // namespace cuvs::neighbors::cagra
