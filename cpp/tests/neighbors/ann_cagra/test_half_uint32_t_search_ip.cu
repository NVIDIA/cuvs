/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CAGRA build/serialize/search (AnnCagraTest) over the InnerProduct slice of `inputs`. The tests
// are defined in test_half_uint32_t_search.cu.

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

INSTANTIATE_TEST_SUITE_P(AnnCagraTestIP, AnnCagraTestF16_U32, ::testing::ValuesIn(inputs_ip));

}  // namespace cuvs::neighbors::cagra
