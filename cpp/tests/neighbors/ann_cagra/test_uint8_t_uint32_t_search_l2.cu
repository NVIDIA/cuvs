/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CAGRA build/serialize/search (AnnCagraTest) over the L2Expanded slice of `inputs`. The tests are
// defined in test_uint8_t_uint32_t_search.cu.

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

INSTANTIATE_TEST_SUITE_P(AnnCagraTestL2, AnnCagraTestU8_U32, ::testing::ValuesIn(inputs_l2));

}  // namespace cuvs::neighbors::cagra
