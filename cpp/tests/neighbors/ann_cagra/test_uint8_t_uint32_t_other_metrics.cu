/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// AnnCagraTest and AnnCagraIndexMergeTest over the CosineExpanded / L1 / BitwiseHamming slice of
// `inputs`. Both suites are small on this slice, so they share one executable (and its
// per-metric JIT warm-up). The tests are defined in test_uint8_t_uint32_t_search.cu and
// test_uint8_t_uint32_t_merge.cu.

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

INSTANTIATE_TEST_SUITE_P(AnnCagraTestOtherMetrics,
                         AnnCagraTestU8_U32,
                         ::testing::ValuesIn(inputs_other_metrics));
INSTANTIATE_TEST_SUITE_P(AnnCagraIndexMergeTestOtherMetrics,
                         AnnCagraIndexMergeTestU8_U32,
                         ::testing::ValuesIn(inputs_other_metrics));

}  // namespace cuvs::neighbors::cagra
