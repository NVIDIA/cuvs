/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_ivf_flat.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/init.cuh>
#include <raft/matrix/init.cuh>
#include <raft/random/rng.cuh>

#include <algorithm>
#include <numeric>
#include <optional>
#include <sstream>
#include <vector>

namespace cuvs::neighbors::ivf_flat {

typedef AnnIVFFlatTest<float, float, int64_t> AnnIVFFlatTestF_float;
TEST_P(AnnIVFFlatTestF_float, AnnIVFFlat)
{
  this->testIVFFlat();
  this->testPacker();
  this->testFilter();
}

INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest, AnnIVFFlatTestF_float, ::testing::ValuesIn(inputs));

TEST(IVFFlatSerialization, StreamRoundTrip)
{
  raft::resources handle;
  auto dataset = raft::make_host_matrix<float, int64_t>(32, 4);
  std::iota(dataset.data_handle(), dataset.data_handle() + dataset.size(), 0.0f);

  index_params params;
  params.n_lists   = 4;
  const auto index = build(handle, params, raft::make_const_mdspan(dataset.view()));

  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  serialize(handle, stream, index);
  stream.seekg(0);

  ivf_flat::index<float, int64_t> loaded(handle);
  deserialize(handle, stream, &loaded);
  EXPECT_EQ(loaded.size(), index.size());
  EXPECT_EQ(loaded.dim(), index.dim());
  EXPECT_EQ(loaded.n_lists(), index.n_lists());
}

// k-means gives an empty cluster a zero center. With the cosine metric, the coarse search must
// still rank such lists instead of producing NaN distances: when fewer than n_probes lists have a
// finite distance, the NaN columns are never selected and the remaining probe slots are padded
// with list 0, which then is scanned several times and returns duplicated neighbors.
TEST(AnnIVFFlatTest, CosineZeroNormCenters)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  constexpr int64_t n_rows        = 10000;
  constexpr int64_t n_queries     = 500;
  constexpr int64_t dim           = 8;
  constexpr uint32_t n_lists      = 1024;
  constexpr uint32_t n_zero_lists = 1000;
  constexpr uint32_t n_probes     = 40;
  constexpr uint32_t k            = 16;

  auto data    = raft::make_device_matrix<float, int64_t>(handle, n_rows, dim);
  auto queries = raft::make_device_matrix<float, int64_t>(handle, n_queries, dim);
  raft::random::RngState rng(1234ULL);
  raft::random::uniform(handle, rng, data.data_handle(), data.size(), 0.1f, 2.0f);
  raft::random::uniform(handle, rng, queries.data_handle(), queries.size(), 0.1f, 2.0f);

  index_params params;
  params.n_lists           = n_lists;
  params.metric            = cuvs::distance::DistanceType::CosineExpanded;
  params.add_data_on_build = false;
  auto idx                 = build(handle, params, raft::make_const_mdspan(data.view()));

  // Zero the centers of lists 1..n_zero_lists, as if k-means had left these clusters empty. Fewer
  // lists than n_probes keep a non-zero center, and list 0 is one of them.
  std::vector<float> centers_h(idx.centers().size());
  raft::update_host(centers_h.data(), idx.centers().data_handle(), centers_h.size(), stream);
  raft::resource::sync_stream(handle);
  std::fill(centers_h.begin() + dim, centers_h.begin() + (n_zero_lists + 1) * dim, 0.0f);
  raft::update_device(idx.centers().data_handle(), centers_h.data(), centers_h.size(), stream);
  extend(handle, raft::make_const_mdspan(data.view()), std::nullopt, &idx);

  // All data rows are in the lists with a non-zero center, and all of those lists are probed.
  search_params sparams;
  sparams.n_probes = n_probes;
  auto neighbors   = raft::make_device_matrix<int64_t, int64_t>(handle, n_queries, k);
  auto distances   = raft::make_device_matrix<float, int64_t>(handle, n_queries, k);
  search(handle,
         sparams,
         idx,
         raft::make_const_mdspan(queries.view()),
         neighbors.view(),
         distances.view());

  auto naive_neighbors = raft::make_device_matrix<int64_t, int64_t>(handle, n_queries, k);
  auto naive_distances = raft::make_device_matrix<float, int64_t>(handle, n_queries, k);
  cuvs::neighbors::naive_knn<float, float, int64_t>(handle,
                                                    naive_distances.data_handle(),
                                                    naive_neighbors.data_handle(),
                                                    queries.data_handle(),
                                                    data.data_handle(),
                                                    n_queries,
                                                    n_rows,
                                                    dim,
                                                    k,
                                                    params.metric);

  std::vector<int64_t> neighbors_h(n_queries * k), naive_neighbors_h(n_queries * k);
  std::vector<float> distances_h(n_queries * k), naive_distances_h(n_queries * k);
  raft::update_host(neighbors_h.data(), neighbors.data_handle(), neighbors_h.size(), stream);
  raft::update_host(distances_h.data(), distances.data_handle(), distances_h.size(), stream);
  raft::update_host(
    naive_neighbors_h.data(), naive_neighbors.data_handle(), naive_neighbors_h.size(), stream);
  raft::update_host(
    naive_distances_h.data(), naive_distances.data_handle(), naive_distances_h.size(), stream);
  raft::resource::sync_stream(handle);

  EXPECT_TRUE(eval_neighbours(
    naive_neighbors_h, neighbors_h, naive_distances_h, distances_h, n_queries, k, 0.001, 0.99));
}

TEST(AnnIVFFlatTest, RepeatedExtendCopyPreservesSharedListWithinCapacity)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  constexpr int64_t base_rows = 100;
  constexpr int64_t grow_rows = 20;
  constexpr int64_t rows      = base_rows + grow_rows;
  constexpr int64_t dim       = 4;

  auto data = raft::make_device_matrix<float, int64_t>(handle, rows, dim);
  raft::matrix::fill(handle, data.view(), 0.0f);

  index_params params;
  params.n_lists                        = 1;
  params.metric                         = cuvs::distance::DistanceType::L2Expanded;
  params.add_data_on_build              = false;
  params.kmeans_trainset_fraction       = 1.0;
  params.adaptive_centers               = false;
  params.conservative_memory_allocation = false;

  auto all_data_view =
    raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), rows, dim);
  auto empty_index = build(handle, params, all_data_view);

  auto base_data_view =
    raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), base_rows, dim);
  auto base_index = extend(handle, base_data_view, std::nullopt, empty_index);
  raft::resource::sync_stream(handle);

  ASSERT_EQ(base_index.lists()[0]->get_size(), base_rows);
  ASSERT_EQ(base_index.lists()[0]->indices_capacity(), raft::Pow2<kIndexGroupSize>::roundUp(rows));

  auto indices = raft::make_device_vector<int64_t, int64_t>(handle, grow_rows);
  raft::linalg::range(indices.data_handle(), base_rows, rows, stream.get());

  auto grow_data_view = raft::make_device_matrix_view<const float, int64_t>(
    data.data_handle() + base_rows * dim, grow_rows, dim);
  auto grow_indices_view =
    raft::make_device_vector_view<const int64_t, int64_t>(indices.data_handle(), grow_rows);
  auto first_grown_index =
    extend(handle,
           grow_data_view,
           std::make_optional<raft::device_vector_view<const int64_t, int64_t>>(grow_indices_view),
           base_index);
  raft::resource::sync_stream(handle);

  ASSERT_EQ(first_grown_index.lists()[0].get(), base_index.lists()[0].get());
  ASSERT_EQ(first_grown_index.lists()[0]->get_size(), rows);

  auto second_grown_index =
    extend(handle,
           grow_data_view,
           std::make_optional<raft::device_vector_view<const int64_t, int64_t>>(grow_indices_view),
           base_index);
  raft::resource::sync_stream(handle);

  EXPECT_NE(first_grown_index.lists()[0].get(), second_grown_index.lists()[0].get());
  EXPECT_EQ(second_grown_index.lists()[0]->get_size(), rows);
}

}  // namespace cuvs::neighbors::ivf_flat
