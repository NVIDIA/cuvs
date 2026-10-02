/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CAGRA index merge (AnnCagraIndexMergeTest) over the L2Expanded slice of `inputs`. The test is
// defined in test_uint8_t_uint32_t_merge.cu.

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

INSTANTIATE_TEST_SUITE_P(AnnCagraIndexMergeTestL2,
                         AnnCagraIndexMergeTestU8_U32,
                         ::testing::ValuesIn(inputs_l2));

}  // namespace cuvs::neighbors::cagra
