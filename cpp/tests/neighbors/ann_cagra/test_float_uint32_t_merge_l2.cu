/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CAGRA index merge (AnnCagraIndexMergeTest) over the L2Expanded slice of `inputs`, for both
// the U32 and the I64 merge test executables. They run the tests defined in
// test_float_uint32_t_merge_u32.cu and test_float_uint32_t_merge_i64.cu respectively.

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

INSTANTIATE_TEST_SUITE_P(AnnCagraIndexMergeTestL2,
                         AnnCagraIndexMergeTestF_U32,
                         ::testing::ValuesIn(inputs_l2));

}  // namespace cuvs::neighbors::cagra
