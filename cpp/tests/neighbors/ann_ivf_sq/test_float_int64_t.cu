/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_ivf_sq.cuh"

namespace cuvs::neighbors::ivf_sq {

typedef AnnIVFSQTest<float, float, int64_t> AnnIVFSQTestF_float;
TEST_P(AnnIVFSQTestF_float, AnnIVFSQ) { this->testAll(); }

INSTANTIATE_TEST_CASE_P(AnnIVFSQTest, AnnIVFSQTestF_float, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::ivf_sq
