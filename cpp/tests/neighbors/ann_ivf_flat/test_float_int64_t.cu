/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_ivf_flat.cuh"

#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/linalg/init.cuh>
#include <raft/matrix/init.cuh>

#include <memory>
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

TEST(AnnIVFFlatTest, AdaptiveCentersWithPrefetchedHostBatches)
{
  raft::resources handle;
  // Avoid implicit cudaFree synchronization hiding staging-buffer reuse races.
  raft::resource::set_workspace_to_pool_resource(handle, 64 * 1024 * 1024);
  raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
  auto stream = raft::resource::get_cuda_stream(handle);

  // Cross multiple staging-buffer boundaries, including a partial final batch. Distinct values
  // in each batch expose overwrites by the copy stream during center accumulation.
  constexpr int64_t batch_rows = 65536;
  constexpr int64_t rows       = 3 * batch_rows + 17;
  constexpr int64_t dim        = 32;
  // Pinned host memory permits the next H2D copy to overlap the center-update kernel.
  auto dataset = raft::make_pinned_matrix<float, int64_t>(handle, rows, dim);
  std::vector<double> sums(dim, 0.0);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < dim; ++col) {
      auto value        = float((row / batch_rows + 1) * (col + 1));
      dataset(row, col) = value;
      sums[col] += value;
    }
  }

  index_params params;
  params.n_lists                  = 1;
  params.adaptive_centers         = true;
  params.add_data_on_build        = false;
  params.kmeans_trainset_fraction = 1.0;
  auto training_view =
    raft::make_host_matrix_view<const float, int64_t>(dataset.data_handle(), 1, dim);
  auto idx = build(handle, params, training_view);
  auto data_view =
    raft::make_host_matrix_view<const float, int64_t>(dataset.data_handle(), rows, dim);
  extend(handle, data_view, std::nullopt, &idx);

  std::vector<float> center(dim);
  uint32_t list_size = 0;
  raft::update_host(center.data(), idx.centers().data_handle(), dim, stream);
  raft::update_host(&list_size, idx.list_sizes().data_handle(), 1, stream);
  raft::resource::sync_stream(handle);

  EXPECT_EQ(idx.size(), rows);
  EXPECT_EQ(list_size, rows);
  for (int64_t col = 0; col < dim; ++col) {
    EXPECT_NEAR(center[col], sums[col] / rows, 1e-4) << "column " << col;
  }
}

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
