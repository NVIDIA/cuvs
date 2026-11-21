/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_gpipnn.cuh"

#include <cstdint>

namespace cuvs::neighbors::experimental::gpipnn {

/**
 * The int8 entry point, over the whole pipeline rather than only its output.
 *
 * `AnnGpipnnTestF_U32.IntegerInput` already asserts the stronger property for the conversion
 * itself -- that an int8 build is bit-identical to the float build of the same values -- but it
 * does so on components in [0, 255] mapped to [-128, 127] and stops at the graph's structure. It
 * never searches the graph, never looks at a stage, and never runs the build twice.
 *
 * This file fills that in, and does it on data drawn across the type's full range, where the
 * squared distances in 128 dimensions reach ~8e6. That is the same magnitude `AnnGpipnnWideRange`
 * reaches by scaling floats, so it is a regime the pipeline is known to need care in, arrived at
 * here by nothing more exotic than using int8 the way a real int8 dataset does.
 */
using AnnGpipnnTestI8_U32 = AnnGpipnnTest<float, std::int8_t, std::uint32_t>;

TEST_P(AnnGpipnnTestI8_U32, Recall) { this->testGpipnn(); }
TEST_P(AnnGpipnnTestI8_U32, Carving) { this->testCarving(); }
TEST_P(AnnGpipnnTestI8_U32, Accumulator) { this->testAccumulator(); }
TEST_P(AnnGpipnnTestI8_U32, CosineRecall)
{
  this->testGpipnn(cuvs::distance::DistanceType::CosineExpanded);
}
TEST_P(AnnGpipnnTestI8_U32, Determinism) { this->testDeterminism(); }

INSTANTIATE_TEST_CASE_P(AnnGpipnnTest, AnnGpipnnTestI8_U32, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::experimental::gpipnn
