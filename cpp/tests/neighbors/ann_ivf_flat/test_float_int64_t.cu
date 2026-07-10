/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_ivf_flat.cuh"

#include <numeric>
#include <sstream>

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

}  // namespace cuvs::neighbors::ivf_flat
