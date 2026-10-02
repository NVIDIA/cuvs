/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Filtered CAGRA search (AnnCagraFilterTest) and CAGRA extend (AnnCagraAddNodesTest).

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

typedef AnnCagraAddNodesTest<float, std::uint8_t, std::uint32_t> AnnCagraAddNodesTestU8_U32;
TEST_P(AnnCagraAddNodesTestU8_U32, AnnCagra) { this->testCagra(); }
typedef AnnCagraFilterTest<float, std::uint8_t, std::uint32_t> AnnCagraFilterTestU8_U32;
TEST_P(AnnCagraFilterTestU8_U32, AnnCagra) { this->testCagra(); }

INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,
                        AnnCagraAddNodesTestU8_U32,
                        ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(AnnCagraFilterTest,
                        AnnCagraFilterTestU8_U32,
                        ::testing::ValuesIn(inputs_filtering));

}  // namespace cuvs::neighbors::cagra
