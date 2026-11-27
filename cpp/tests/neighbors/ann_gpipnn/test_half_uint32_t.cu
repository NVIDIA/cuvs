/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_gpipnn.cuh"

#include <cstdint>

namespace cuvs::neighbors::experimental::gpipnn {

/**
 * The half entry point, over the whole pipeline rather than only its output.
 *
 * `AnnGpipnnTestF_U32.HalfInput` already builds a graph from half input and checks its structure,
 * which is enough to show the overload reaches the builder. What it cannot show is that the
 * stages behave on half data: carving placing every point, the reservoirs filling deeply enough
 * to prune from, the build staying reproducible, and the result still being searchable to the
 * recall the float fixture requires. Those are what this file adds.
 *
 * Half is the one type with no conversion in front of it -- the dataset is used directly as the
 * scoring operand -- so it is also the only type where a defect in the conversion prologue cannot
 * mask, or be mistaken for, a defect in the stages.
 */
using AnnGpipnnTestF16_U32 = AnnGpipnnTest<float, half, std::uint32_t>;

TEST_P(AnnGpipnnTestF16_U32, Recall) { this->testGpipnn(); }
TEST_P(AnnGpipnnTestF16_U32, Carving) { this->testCarving(); }
TEST_P(AnnGpipnnTestF16_U32, Accumulator) { this->testAccumulator(); }
TEST_P(AnnGpipnnTestF16_U32, CosineRecall)
{
  this->testGpipnn(cuvs::distance::DistanceType::CosineExpanded);
}
// Rescaling a half row by a power of two only moves its exponent, so the bit-identity this
// asserts is as exact here as it is on floats.
TEST_P(AnnGpipnnTestF16_U32, CosineScaleInvariance) { this->testCosineScaleInvariance(); }
TEST_P(AnnGpipnnTestF16_U32, Determinism) { this->testDeterminism(); }

INSTANTIATE_TEST_CASE_P(AnnGpipnnTest, AnnGpipnnTestF16_U32, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::experimental::gpipnn
