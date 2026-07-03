/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_hnsw_ace.cuh"

namespace cuvs::neighbors::hnsw {

typedef AnnHnswAceTest<float, float, uint32_t> AnnHnswAceTest_float;
TEST_P(AnnHnswAceTest_float, AnnHnswAceBuild) { this->testHnswAceBuild(); }

INSTANTIATE_TEST_CASE_P(AnnHnswAceTest, AnnHnswAceTest_float, ::testing::ValuesIn(hnsw_ace_inputs));

// Test for memory limit fallback to disk mode
typedef AnnHnswAceTest<float, float, uint32_t> AnnHnswAceMemoryFallbackTest_float;
TEST_P(AnnHnswAceMemoryFallbackTest_float, AnnHnswAceMemoryLimitFallback)
{
  this->testHnswAceMemoryLimitFallback();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceMemoryFallbackTest,
                        AnnHnswAceMemoryFallbackTest_float,
                        ::testing::ValuesIn(hnsw_ace_memory_fallback_inputs));

typedef AnnHnswAceTest<float, float, uint32_t> AnnHnswAceLayeredTest_float;
TEST_P(AnnHnswAceLayeredTest_float, AnnHnswAceLayeredBuildDeserializeSearch)
{
  this->testHnswAceLayeredBuildDeserializeSearch();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceLayeredTest,
                        AnnHnswAceLayeredTest_float,
                        ::testing::ValuesIn(hnsw_ace_layered_inputs));

// Test for in-memory CAGRA -> HNSW disk-spill conversion
typedef AnnHnswAceTest<float, float, uint32_t> AnnHnswInmemSpillTest_float;
TEST_P(AnnHnswInmemSpillTest_float, AnnHnswFromCagraInmemSpill)
{
  this->testHnswFromCagraInmemSpill();
}

INSTANTIATE_TEST_CASE_P(AnnHnswInmemSpillTest,
                        AnnHnswInmemSpillTest_float,
                        ::testing::ValuesIn(hnsw_inmem_spill_inputs));

}  // namespace cuvs::neighbors::hnsw
