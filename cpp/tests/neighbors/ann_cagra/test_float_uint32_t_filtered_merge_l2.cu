/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Filtered CAGRA index merge (AnnCagraIndexFilteredMergeTest) over the L2Expanded slice of
// `inputs`. The test is defined in test_float_uint32_t_filtered_merge.cu.

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

INSTANTIATE_TEST_SUITE_P(AnnCagraIndexFilteredMergeTestL2,
                         AnnCagraIndexFilteredMergeTestF_U32,
                         ::testing::ValuesIn(inputs_l2));

}  // namespace cuvs::neighbors::cagra
