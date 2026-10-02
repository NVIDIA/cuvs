/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Filtered CAGRA index merge (AnnCagraIndexFilteredMergeTest) over the InnerProduct slice of
// `inputs`. The test is defined in test_float_uint32_t_filtered_merge.cu.

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

INSTANTIATE_TEST_SUITE_P(AnnCagraIndexFilteredMergeTestIP,
                         AnnCagraIndexFilteredMergeTestF_U32,
                         ::testing::ValuesIn(inputs_ip));

}  // namespace cuvs::neighbors::cagra
