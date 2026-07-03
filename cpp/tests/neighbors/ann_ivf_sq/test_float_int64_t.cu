/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_ivf_sq.cuh"

#include <numeric>
#include <vector>

namespace cuvs::neighbors::ivf_sq {

typedef AnnIVFSQTest<float, float, int64_t> AnnIVFSQTestF_float;
TEST_P(AnnIVFSQTestF_float, AnnIVFSQ) { this->testAll(); }

INSTANTIATE_TEST_CASE_P(AnnIVFSQTest, AnnIVFSQTestF_float, ::testing::ValuesIn(inputs));

TEST(AnnIVFSQTest, ExtendInPlaceUpdatesListSizeWithinCapacity)
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
  params.max_train_points_per_cluster   = 256;
  params.conservative_memory_allocation = false;

  auto all_data_view =
    raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), rows, dim);
  auto index = build(handle, params, all_data_view);

  auto base_data_view =
    raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), base_rows, dim);
  extend(handle, base_data_view, std::nullopt, &index);
  raft::resource::sync_stream(handle);

  ASSERT_EQ(index.lists()[0]->get_size(), base_rows);
  ASSERT_GE(index.lists()[0]->indices_capacity(), rows);

  std::vector<int64_t> host_indices(grow_rows);
  std::iota(host_indices.begin(), host_indices.end(), base_rows);
  auto indices = raft::make_device_vector<int64_t, int64_t>(handle, grow_rows);
  raft::copy(indices.data_handle(), host_indices.data(), host_indices.size(), stream);

  auto grow_data_view = raft::make_device_matrix_view<const float, int64_t>(
    data.data_handle() + base_rows * dim, grow_rows, dim);
  auto grow_indices_view =
    raft::make_device_vector_view<const int64_t, int64_t>(indices.data_handle(), grow_rows);
  extend(handle,
         grow_data_view,
         std::make_optional<raft::device_vector_view<const int64_t, int64_t>>(grow_indices_view),
         &index);
  raft::resource::sync_stream(handle);

  EXPECT_EQ(index.lists()[0]->get_size(), rows);
}

}  // namespace cuvs::neighbors::ivf_sq
