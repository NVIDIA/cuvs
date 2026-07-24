/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_hnsw_ace.cuh"

namespace cuvs::neighbors::hnsw {

typedef AnnHnswAceTest<float, int8_t, uint32_t> AnnHnswAceTest_int8_t;
TEST_P(AnnHnswAceTest_int8_t, AnnHnswAceBuild) { this->testHnswAceBuild(); }

INSTANTIATE_TEST_CASE_P(AnnHnswAceTest,
                        AnnHnswAceTest_int8_t,
                        ::testing::ValuesIn(hnsw_ace_inputs));

typedef AnnHnswAceTest<float, int8_t, uint32_t> AnnHnswAceInvalidPartitionTest_int8_t;
TEST_P(AnnHnswAceInvalidPartitionTest_int8_t, RejectsTooManyPartitions)
{
  this->testHnswAceRejectsTooManyPartitions();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceInvalidPartitionTest,
                        AnnHnswAceInvalidPartitionTest_int8_t,
                        ::testing::ValuesIn(hnsw_ace_invalid_partition_inputs));

// Test for memory limit fallback to disk mode
typedef AnnHnswAceTest<float, int8_t, uint32_t> AnnHnswAceMemoryFallbackTest_int8_t;
TEST_P(AnnHnswAceMemoryFallbackTest_int8_t, AnnHnswAceMemoryLimitFallback)
{
  this->testHnswAceMemoryLimitFallback();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceMemoryFallbackTest,
                        AnnHnswAceMemoryFallbackTest_int8_t,
                        ::testing::ValuesIn(hnsw_ace_memory_fallback_inputs));

// Test for in-memory CAGRA -> HNSW disk-spill conversion
typedef AnnHnswAceTest<float, int8_t, uint32_t> AnnHnswInmemSpillTest_int8_t;
TEST_P(AnnHnswInmemSpillTest_int8_t, AnnHnswFromCagraInmemSpill)
{
  this->testHnswFromCagraInmemSpill();
}

INSTANTIATE_TEST_CASE_P(AnnHnswInmemSpillTest,
                        AnnHnswInmemSpillTest_int8_t,
                        ::testing::ValuesIn(hnsw_inmem_spill_inputs));

}  // namespace cuvs::neighbors::hnsw
