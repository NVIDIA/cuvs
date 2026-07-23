/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

typedef AnnCagraTest<float, std::uint8_t, std::uint32_t> AnnCagraTestU8_U32;
TEST_P(AnnCagraTestU8_U32, AnnCagra) { this->testCagra(); }
typedef AnnCagraAddNodesTest<float, std::uint8_t, std::uint32_t> AnnCagraAddNodesTestU8_U32;
TEST_P(AnnCagraAddNodesTestU8_U32, AnnCagra) { this->testCagra(); }
typedef AnnCagraFilterTest<float, std::uint8_t, std::uint32_t> AnnCagraFilterTestU8_U32;
TEST_P(AnnCagraFilterTestU8_U32, AnnCagra) { this->testCagra(); }
typedef AnnCagraIndexMergeTest<float, std::uint8_t, std::uint32_t> AnnCagraIndexMergeTestU8_U32;
TEST_P(AnnCagraIndexMergeTestU8_U32, AnnCagra) { this->testCagra(); }

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestU8_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,
                        AnnCagraAddNodesTestU8_U32,
                        ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(AnnCagraFilterTest,
                        AnnCagraFilterTestU8_U32,
                        ::testing::ValuesIn(inputs_filtering));
INSTANTIATE_TEST_CASE_P(AnnCagraIndexMergeTest,
                        AnnCagraIndexMergeTestU8_U32,
                        ::testing::ValuesIn(inputs));

typedef AnnCagraMultiPartitionTest<float, std::uint8_t, std::uint32_t>
  AnnCagraMultiPartitionTestU8_U32;
TEST_P(AnnCagraMultiPartitionTestU8_U32, Search) { this->testSearch(); }
TEST_P(AnnCagraMultiPartitionTestU8_U32, FilteredSearch) { this->testFilteredSearch(); }

INSTANTIATE_TEST_CASE_P(AnnCagraMultiPartitionTest,
                        AnnCagraMultiPartitionTestU8_U32,
                        ::testing::ValuesIn(inputs_mp));

}  // namespace cuvs::neighbors::cagra
