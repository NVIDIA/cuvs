/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_hnsw_ace.cuh"

namespace cuvs::neighbors::hnsw {

TEST(CagraAceWorkspaceHalf, FailurePreservesCallerDirectory)
{
  test_ace_workspace_failure_preserves_caller_directory<half>();
}

TEST(CagraAceWorkspaceHalf, FailureDoesNotTruncateExistingArtifact)
{
  test_ace_workspace_failure_does_not_truncate_existing_artifact<half>();
}

typedef AnnHnswAceTest<float, half, uint32_t> AnnHnswAceTest_half;
TEST_P(AnnHnswAceTest_half, AnnHnswAceBuild) { this->testHnswAceBuild(); }

INSTANTIATE_TEST_CASE_P(AnnHnswAceTest, AnnHnswAceTest_half, ::testing::ValuesIn(hnsw_ace_inputs));

// Test for memory limit fallback to disk mode
typedef AnnHnswAceTest<float, half, uint32_t> AnnHnswAceMemoryFallbackTest_half;
TEST_P(AnnHnswAceMemoryFallbackTest_half, AnnHnswAceMemoryLimitFallback)
{
  this->testHnswAceMemoryLimitFallback();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceMemoryFallbackTest,
                        AnnHnswAceMemoryFallbackTest_half,
                        ::testing::ValuesIn(hnsw_ace_memory_fallback_inputs));

// Test for in-memory CAGRA -> HNSW disk-spill conversion
typedef AnnHnswAceTest<float, half, uint32_t> AnnHnswInmemSpillTest_half;
TEST_P(AnnHnswInmemSpillTest_half, AnnHnswFromCagraInmemSpill)
{
  this->testHnswFromCagraInmemSpill();
}

INSTANTIATE_TEST_CASE_P(AnnHnswInmemSpillTest,
                        AnnHnswInmemSpillTest_half,
                        ::testing::ValuesIn(hnsw_inmem_spill_inputs));

}  // namespace cuvs::neighbors::hnsw
