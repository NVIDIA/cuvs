/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

typedef AnnCagraTest<float, half, std::uint32_t> AnnCagraTestF16_U32;
TEST_P(AnnCagraTestF16_U32, AnnCagra_U32) { this->testCagra<uint32_t>(); }
TEST_P(AnnCagraTestF16_U32, AnnCagra_I64) { this->testCagra<int64_t>(); }

typedef AnnCagraAddNodesTest<float, half, std::uint32_t> AnnCagraAddNodesTestF16_U32;
TEST_P(AnnCagraAddNodesTestF16_U32, AnnCagraAddNodes) { this->testCagra(); }

typedef AnnCagraIndexMergeTest<float, half, std::uint32_t> AnnCagraIndexMergeTestF16_U32;
TEST_P(AnnCagraIndexMergeTestF16_U32, AnnCagraIndexMerge_U32) { this->testCagra<uint32_t>(); }
TEST_P(AnnCagraIndexMergeTestF16_U32, AnnCagraIndexMerge_I64) { this->testCagra<int64_t>(); }

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestF16_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,
                        AnnCagraAddNodesTestF16_U32,
                        ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(AnnCagraIndexMergeTest,
                        AnnCagraIndexMergeTestF16_U32,
                        ::testing::ValuesIn(inputs));

typedef AnnCagraMultiPartitionTest<float, half, std::uint32_t> AnnCagraMultiPartitionTestF16_U32;
TEST_P(AnnCagraMultiPartitionTestF16_U32, Search) { this->testSearch(); }
TEST_P(AnnCagraMultiPartitionTestF16_U32, FilteredSearch) { this->testFilteredSearch(); }

INSTANTIATE_TEST_CASE_P(AnnCagraMultiPartitionTest,
                        AnnCagraMultiPartitionTestF16_U32,
                        ::testing::ValuesIn(inputs_mp));

}  // namespace cuvs::neighbors::cagra
