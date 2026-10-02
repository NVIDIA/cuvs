/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CAGRA extend (AnnCagraAddNodesTest) and multi-partition CAGRA search
// (AnnCagraMultiPartitionTest).

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

typedef AnnCagraAddNodesTest<float, half, std::uint32_t> AnnCagraAddNodesTestF16_U32;
TEST_P(AnnCagraAddNodesTestF16_U32, AnnCagraAddNodes) { this->testCagra(); }

INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,
                        AnnCagraAddNodesTestF16_U32,
                        ::testing::ValuesIn(inputs_addnode));

typedef AnnCagraMultiPartitionTest<float, half, std::uint32_t> AnnCagraMultiPartitionTestF16_U32;
TEST_P(AnnCagraMultiPartitionTestF16_U32, Search) { this->testSearch(); }
TEST_P(AnnCagraMultiPartitionTestF16_U32, FilteredSearch) { this->testFilteredSearch(); }

INSTANTIATE_TEST_CASE_P(AnnCagraMultiPartitionTest,
                        AnnCagraMultiPartitionTestF16_U32,
                        ::testing::ValuesIn(inputs_mp));

}  // namespace cuvs::neighbors::cagra
