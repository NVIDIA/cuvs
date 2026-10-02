/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// AnnCagraTest, AnnCagraIndexMergeTest and AnnCagraIndexFilteredMergeTest over the
// CosineExpanded / L1 / BitwiseHamming slice of `inputs`. Each of these suites is small on this
// slice, so they share one executable (and its per-metric JIT warm-up). The tests are defined in
// test_float_uint32_t_search.cu, test_float_uint32_t_merge_u32.cu,
// test_float_uint32_t_merge_i64.cu and test_float_uint32_t_filtered_merge.cu.

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

INSTANTIATE_TEST_SUITE_P(AnnCagraTestOtherMetrics,
                         AnnCagraTestF_U32,
                         ::testing::ValuesIn(inputs_other_metrics));
INSTANTIATE_TEST_SUITE_P(AnnCagraIndexMergeTestOtherMetrics,
                         AnnCagraIndexMergeTestF_U32,
                         ::testing::ValuesIn(inputs_other_metrics));
INSTANTIATE_TEST_SUITE_P(AnnCagraIndexFilteredMergeTestOtherMetrics,
                         AnnCagraIndexFilteredMergeTestF_U32,
                         ::testing::ValuesIn(inputs_other_metrics));

}  // namespace cuvs::neighbors::cagra
