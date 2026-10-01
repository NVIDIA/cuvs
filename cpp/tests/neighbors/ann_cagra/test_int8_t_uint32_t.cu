/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

typedef AnnCagraTest<float, std::int8_t, std::uint32_t> AnnCagraTestI8_U32;
TEST_P(AnnCagraTestI8_U32, AnnCagra) { this->testCagra(); }
typedef AnnCagraAddNodesTest<float, std::int8_t, std::uint32_t> AnnCagraAddNodesTestI8_U32;
TEST_P(AnnCagraAddNodesTestI8_U32, AnnCagra) { this->testCagra(); }
typedef AnnCagraFilterTest<float, std::int8_t, std::uint32_t> AnnCagraFilterTestI8_U32;
TEST_P(AnnCagraFilterTestI8_U32, AnnCagra) { this->testCagra(); }
typedef AnnCagraIndexMergeTest<float, std::int8_t, std::uint32_t> AnnCagraIndexMergeTestI8_U32;
TEST_P(AnnCagraIndexMergeTestI8_U32, AnnCagra) { this->testCagra(); }

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestI8_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,
                        AnnCagraAddNodesTestI8_U32,
                        ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(AnnCagraFilterTest,
                        AnnCagraFilterTestI8_U32,
                        ::testing::ValuesIn(inputs_filtering));
INSTANTIATE_TEST_CASE_P(AnnCagraIndexMergeTest,
                        AnnCagraIndexMergeTestI8_U32,
                        ::testing::ValuesIn(inputs));

typedef AnnCagraMultiPartitionTest<float, std::int8_t, std::uint32_t>
  AnnCagraMultiPartitionTestI8_U32;
TEST_P(AnnCagraMultiPartitionTestI8_U32, Search) { this->testSearch(); }
TEST_P(AnnCagraMultiPartitionTestI8_U32, FilteredSearch) { this->testFilteredSearch(); }

INSTANTIATE_TEST_CASE_P(AnnCagraMultiPartitionTest,
                        AnnCagraMultiPartitionTestI8_U32,
                        ::testing::ValuesIn(inputs_mp));

}  // namespace cuvs::neighbors::cagra
