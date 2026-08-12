/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "knn_utils.cuh"

#include <cuvs/neighbors/knn_merge_parts.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace cuvs::neighbors {
namespace {

void run_merge(bool select_min,
               const std::vector<float>& expected_distances,
               const std::vector<int64_t>& expected_neighbors)
{
  constexpr int64_t n_queries = 2;
  constexpr int64_t n_parts   = 2;
  constexpr int64_t k         = 3;

  ASSERT_EQ(expected_distances.size(), n_queries * k);
  ASSERT_EQ(expected_neighbors.size(), n_queries * k);

  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  // Input layout is [part][query][neighbor]. Indices are local to each part.
  const std::vector<float> input_distances{
    10.0f, 8.0f, 6.0f, -1.0f, -3.0f, -5.0f, 9.0f, 7.0f, 5.0f, 4.0f, 2.0f, 0.0f};
  const std::vector<int64_t> input_neighbors{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  const std::vector<int64_t> translations{0, 100};

  auto input_distances_device =
    raft::make_device_matrix<float, int64_t>(res, n_parts * n_queries, k);
  auto input_neighbors_device =
    raft::make_device_matrix<int64_t, int64_t>(res, n_parts * n_queries, k);
  auto output_distances_device   = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
  auto output_neighbors_device   = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, k);
  auto expected_distances_device = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
  auto expected_neighbors_device = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, k);
  auto translations_device       = raft::make_device_vector<int64_t, int64_t>(res, n_parts);

  raft::update_device(
    input_distances_device.data_handle(), input_distances.data(), input_distances.size(), stream);
  raft::update_device(
    input_neighbors_device.data_handle(), input_neighbors.data(), input_neighbors.size(), stream);
  raft::update_device(expected_distances_device.data_handle(),
                      expected_distances.data(),
                      expected_distances.size(),
                      stream);
  raft::update_device(expected_neighbors_device.data_handle(),
                      expected_neighbors.data(),
                      expected_neighbors.size(),
                      stream);
  raft::update_device(
    translations_device.data_handle(), translations.data(), translations.size(), stream);

  if (!select_min) {
    raft::linalg::map(res,
                      input_distances_device.view(),
                      raft::mul_const_op<float>(-1),
                      raft::make_const_mdspan(input_distances_device.view()));
  }

  knn_merge_parts(res,
                  input_distances_device.view(),
                  input_neighbors_device.view(),
                  output_distances_device.view(),
                  output_neighbors_device.view(),
                  translations_device.view());

  if (!select_min) {
    raft::linalg::map(res,
                      output_distances_device.view(),
                      raft::mul_const_op<float>(-1),
                      raft::make_const_mdspan(output_distances_device.view()));
  }

  ASSERT_TRUE(devArrMatchKnnPair(expected_neighbors_device.data_handle(),
                                 output_neighbors_device.data_handle(),
                                 expected_distances_device.data_handle(),
                                 output_distances_device.data_handle(),
                                 n_queries,
                                 k,
                                 0.0f,
                                 stream,
                                 true));
}

TEST(KnnMergeParts, SelectsSmallestByDefault)
{
  run_merge(true, {5.0f, 6.0f, 7.0f, -5.0f, -3.0f, -1.0f}, {102, 2, 101, 2, 1, 0});
}

TEST(KnnMergeParts, SelectsLargestAfterNegation)
{
  run_merge(false, {10.0f, 9.0f, 8.0f, 4.0f, 2.0f, 0.0f}, {0, 100, 1, 100, 101, 102});
}

}  // namespace
}  // namespace cuvs::neighbors
