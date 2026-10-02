/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CAGRA index merge test with U32 search indices (AnnCagraIndexMergeTest), compiled once into an
// OBJECT library. Each test executable that links it instantiates it over its slice of `inputs`, in
// test_half_uint32_t_merge_{l2,ip}.cu and test_half_uint32_t_other_metrics.cu.

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

TEST_P(AnnCagraIndexMergeTestF16_U32, AnnCagraIndexMerge_U32) { this->testCagra<uint32_t>(); }

}  // namespace cuvs::neighbors::cagra
