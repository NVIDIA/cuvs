/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CAGRA build/serialize/search test (AnnCagraTest), compiled once into an OBJECT library. Each test
// executable that links it instantiates it over its slice of `inputs`, in
// test_uint8_t_uint32_t_search_{l2,ip}.cu and test_uint8_t_uint32_t_other_metrics.cu.

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

TEST_P(AnnCagraTestU8_U32, AnnCagra) { this->testCagra(); }

}  // namespace cuvs::neighbors::cagra
