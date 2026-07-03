/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_ivf_flat.cuh"

#include <numeric>
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

TEST(AnnIVFFlatTest, RepeatedExtendCopyPreservesSharedListWithinCapacity)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  constexpr int64_t base_rows = 100;
  constexpr int64_t grow_rows = 20;
  constexpr int64_t rows      = base_rows + grow_rows;
  constexpr int64_t dim       = 4;

  std::vector<float> host_data(rows * dim);
  for (int64_t row = 0; row < rows; row++) {
    for (int64_t col = 0; col < dim; col++) {
      host_data[row * dim + col] = static_cast<float>(row + col);
    }
  }

  auto data = raft::make_device_matrix<float, int64_t>(handle, rows, dim);
  raft::copy(data.data_handle(), host_data.data(), host_data.size(), stream);

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
  ASSERT_GE(base_index.lists()[0]->indices_capacity(), rows);

  std::vector<int64_t> host_indices(grow_rows);
  std::iota(host_indices.begin(), host_indices.end(), base_rows);
  auto indices = raft::make_device_vector<int64_t, int64_t>(handle, grow_rows);
  raft::copy(indices.data_handle(), host_indices.data(), host_indices.size(), stream);

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
