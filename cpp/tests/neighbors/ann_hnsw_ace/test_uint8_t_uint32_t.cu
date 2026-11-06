/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_hnsw_ace.cuh"

namespace cuvs::neighbors::hnsw {

TEST(CagraAceWorkspaceUint8, FailurePreservesCallerDirectory)
{
  test_ace_workspace_failure_preserves_caller_directory<uint8_t>();
}

TEST(CagraAceWorkspaceUint8, FailureDoesNotTruncateExistingArtifact)
{
  test_ace_workspace_failure_does_not_truncate_existing_artifact<uint8_t>();
}

typedef AnnHnswAceTest<float, uint8_t, uint32_t> AnnHnswAceTest_uint8_t;
TEST_P(AnnHnswAceTest_uint8_t, AnnHnswAceBuild) { this->testHnswAceBuild(); }

INSTANTIATE_TEST_CASE_P(AnnHnswAceTest,
                        AnnHnswAceTest_uint8_t,
                        ::testing::ValuesIn(hnsw_ace_inputs));

typedef AnnHnswAceTest<float, uint8_t, uint32_t> AnnHnswAceInvalidPartitionTest_uint8_t;
TEST_P(AnnHnswAceInvalidPartitionTest_uint8_t, RejectsTooManyPartitions)
{
  this->testHnswAceRejectsTooManyPartitions();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceInvalidPartitionTest,
                        AnnHnswAceInvalidPartitionTest_uint8_t,
                        ::testing::ValuesIn(hnsw_ace_invalid_partition_inputs));

// Test for memory limit fallback to disk mode
typedef AnnHnswAceTest<float, uint8_t, uint32_t> AnnHnswAceMemoryFallbackTest_uint8_t;
TEST_P(AnnHnswAceMemoryFallbackTest_uint8_t, AnnHnswAceMemoryLimitFallback)
{
  this->testHnswAceMemoryLimitFallback();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceMemoryFallbackTest,
                        AnnHnswAceMemoryFallbackTest_uint8_t,
                        ::testing::ValuesIn(hnsw_ace_memory_fallback_inputs));

typedef AnnHnswAceTest<float, uint8_t, uint32_t> AnnHnswAceLayeredTest_uint8_t;
TEST_P(AnnHnswAceLayeredTest_uint8_t, AnnHnswAceLayeredBuildDeserializeSearch)
{
  this->testHnswAceLayeredBuildDeserializeSearch();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceLayeredTest,
                        AnnHnswAceLayeredTest_uint8_t,
                        ::testing::ValuesIn(hnsw_ace_layered_inputs));

// Test for in-memory CAGRA -> HNSW disk-spill conversion
typedef AnnHnswAceTest<float, uint8_t, uint32_t> AnnHnswInmemSpillTest_uint8_t;
TEST_P(AnnHnswInmemSpillTest_uint8_t, AnnHnswFromCagraInmemSpill)
{
  this->testHnswFromCagraInmemSpill();
}

INSTANTIATE_TEST_CASE_P(AnnHnswInmemSpillTest,
                        AnnHnswInmemSpillTest_uint8_t,
                        ::testing::ValuesIn(hnsw_inmem_spill_inputs));

// One nonaligned shape per scalar; avoid crossing the full ACE parameter matrix.
typedef AnnHnswAceTest<float, uint8_t, uint32_t> AnnHnswLayeredSourcesTest_uint8_t;
TEST_P(AnnHnswLayeredSourcesTest_uint8_t, RegularBuild)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::regular);
}

TEST_P(AnnHnswLayeredSourcesTest_uint8_t, InMemoryAceBuild)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::inmem_ace);
}

TEST_P(AnnHnswLayeredSourcesTest_uint8_t, DeviceStandardAttached)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::device_standard_attached);
}

TEST_P(AnnHnswLayeredSourcesTest_uint8_t, DeviceStandardExplicit)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::device_standard_explicit);
}

TEST_P(AnnHnswLayeredSourcesTest_uint8_t, DevicePaddedAttached)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::device_padded_attached);
}

TEST_P(AnnHnswLayeredSourcesTest_uint8_t, DevicePaddedExplicit)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::device_padded_explicit);
}

TEST_P(AnnHnswLayeredSourcesTest_uint8_t, HostStandard)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::host_standard);
}

TEST_P(AnnHnswLayeredSourcesTest_uint8_t, HostPadded)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::host_padded);
}

INSTANTIATE_TEST_CASE_P(
  AnnHnswLayeredSourcesTest,
  AnnHnswLayeredSourcesTest_uint8_t,
  ::testing::Values(AnnHnswAceInputs{
    10, 2000, 17, 10, 2, 100, false, cuvs::distance::DistanceType::L2Expanded, 0.9}));

}  // namespace cuvs::neighbors::hnsw
