/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_vamana.cuh"

namespace cuvs::neighbors::vamana {

typedef AnnVamanaTest<float, half, std::uint32_t> AnnVamanaTestF16_U32;
TEST_P(AnnVamanaTestF16_U32, AnnVamana) { this->testVamana(); }

INSTANTIATE_TEST_CASE_P(AnnVamanaTest, AnnVamanaTestF16_U32, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::vamana
