/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_vamana.cuh"

#include <numeric>

namespace cuvs::neighbors::vamana {

typedef AnnVamanaTest<float, float, std::uint32_t> AnnVamanaTestF_U32;
TEST_P(AnnVamanaTestF_U32, AnnVamana) { this->testVamana(); }

INSTANTIATE_TEST_CASE_P(AnnVamanaTest, AnnVamanaTestF_U32, ::testing::ValuesIn(inputs));

// Returns a srand() seed for which the build's first inserted point is also its medoid.
// The build calls rand() n times to shuffle the insert order, then once for the medoid.
static unsigned seed_where_first_insert_is_medoid(uint32_t n)
{
  for (unsigned seed = 1;; seed++) {
    srand(seed);
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    for (uint32_t i = 0; i < n; i++) {
      std::swap(order[i], order[rand() % n]);
    }
    if (order[0] == rand() % n) { return seed; }
  }
}

// https://github.com/NVIDIA/cuvs/issues/2767: the first batch then has no reverse edges.
TEST(AnnVamanaBuildTest, FirstInsertIsMedoid)
{
  raft::resources res;
  constexpr int64_t n = 410, dim = 16;
  auto dataset = raft::make_device_matrix<float, int64_t>(res, n, dim);
  raft::random::RngState r(1234ULL);
  raft::random::uniform(res, r, dataset.data_handle(), n * dim, -1.0f, 1.0f);

  srand(seed_where_first_insert_is_medoid(n));
  auto idx = build(res, index_params{}, raft::make_const_mdspan(dataset.view()));
  EXPECT_EQ(idx.graph().extent(0), n);
}

}  // namespace cuvs::neighbors::vamana
