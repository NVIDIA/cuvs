/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// AnnCagraTest and AnnCagraIndexMergeTest over the CosineExpanded / L1 / BitwiseHamming slice of
// `inputs`. Both suites are small on this slice, so they share one executable (and its
// per-metric JIT warm-up). The tests are defined in test_half_uint32_t_search.cu,
// test_half_uint32_t_merge_u32.cu and test_half_uint32_t_merge_i64.cu.

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

INSTANTIATE_TEST_SUITE_P(AnnCagraTestOtherMetrics,
                         AnnCagraTestF16_U32,
                         ::testing::ValuesIn(inputs_other_metrics));
INSTANTIATE_TEST_SUITE_P(AnnCagraIndexMergeTestOtherMetrics,
                         AnnCagraIndexMergeTestF16_U32,
                         ::testing::ValuesIn(inputs_other_metrics));

}  // namespace cuvs::neighbors::cagra
