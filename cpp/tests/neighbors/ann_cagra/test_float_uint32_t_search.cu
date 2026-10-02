/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CAGRA build/serialize/search tests (AnnCagraTest), compiled once into an OBJECT library. Each
// test executable that links it instantiates them over its slice of `inputs`, in
// test_float_uint32_t_search_{l2,ip}.cu and test_float_uint32_t_other_metrics.cu.

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

TEST_P(AnnCagraTestF_U32, AnnCagra_U32) { this->testCagra<uint32_t>(); }
TEST_P(AnnCagraTestF_U32, AnnCagra_I64) { this->testCagra<int64_t>(); }

}  // namespace cuvs::neighbors::cagra
