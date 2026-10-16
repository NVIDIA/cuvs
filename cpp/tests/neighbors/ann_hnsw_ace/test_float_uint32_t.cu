/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../src/neighbors/detail/hnsw_layered_gather.hpp"
#include "../ann_hnsw_ace.cuh"

namespace cuvs::neighbors::hnsw {

TEST(HnswLayeredGather, PreservesSelectedBytesWithAndWithoutPadding)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  const std::vector<size_t> row_ids{4, 1, 4, 0};
  constexpr size_t source_rows = 5;
  // Cover nonaligned vectors with one-, two-, and four-byte scalar representations.
  for (const size_t row_bytes : {size_t{17}, size_t{34}, size_t{68}}) {
    for (const size_t padding : {size_t{0}, size_t{12}}) {
      const size_t stride = row_bytes + padding;
      std::vector<uint8_t> source(source_rows * stride, 0xff);
      for (size_t row = 0; row < source_rows; ++row) {
        for (size_t col = 0; col < row_bytes; ++col) {
          source[row * stride + col] = static_cast<uint8_t>(row * row_bytes + col);
        }
      }
      rmm::device_uvector<uint8_t> device_source(source.size(), stream);
      raft::copy(device_source.data(), source.data(), source.size(), stream);
      std::vector<uint8_t> actual(row_ids.size() * row_bytes);
      detail::gather_layered_hnsw_vectors(handle,
                                          device_source.data(),
                                          stride,
                                          row_bytes,
                                          row_ids.data(),
                                          row_ids.size(),
                                          actual.data());
      std::vector<uint8_t> expected;
      for (const auto row : row_ids) {
        expected.insert(
          expected.end(), source.begin() + row * stride, source.begin() + row * stride + row_bytes);
      }
      EXPECT_EQ(actual, expected) << "row_bytes=" << row_bytes << " padding=" << padding;
    }
  }
}

TEST(HnswLayeredGather, EmptySelectionDoesNotAccessStorage)
{
  raft::resources handle;
  EXPECT_NO_THROW(detail::gather_layered_hnsw_vectors(handle, nullptr, 0, 0, nullptr, 0, nullptr));
}

TEST(CagraAceWorkspace, FailurePreservesCallerDirectory)
{
  test_ace_workspace_failure_preserves_caller_directory<float>();
}

TEST(CagraAceWorkspace, FailureDoesNotTruncateExistingArtifact)
{
  test_ace_workspace_failure_does_not_truncate_existing_artifact<float>();
}

TEST(FileIo, ExclusiveNumpyCreateFailureRemovesPartialFile)
{
  test_exclusive_numpy_create_failure_removes_partial_file();
}

TEST(HnswAceWorkspace, ExistingIndexIsNotTruncated)
{
  test_hnsw_ace_build_does_not_truncate_existing_index<float>();
}

typedef AnnHnswAceTest<float, float, uint32_t> AnnHnswAceTest_float;
TEST_P(AnnHnswAceTest_float, AnnHnswAceBuild) { this->testHnswAceBuild(); }

INSTANTIATE_TEST_CASE_P(AnnHnswAceTest, AnnHnswAceTest_float, ::testing::ValuesIn(hnsw_ace_inputs));

typedef AnnHnswAceTest<float, float, uint32_t> AnnHnswAceInvalidPartitionTest_float;
TEST_P(AnnHnswAceInvalidPartitionTest_float, RejectsTooManyPartitions)
{
  this->testHnswAceRejectsTooManyPartitions();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceInvalidPartitionTest,
                        AnnHnswAceInvalidPartitionTest_float,
                        ::testing::ValuesIn(hnsw_ace_invalid_partition_inputs));

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

// One nonaligned shape per scalar; avoid crossing the full ACE parameter matrix.
typedef AnnHnswAceTest<float, float, uint32_t> AnnHnswLayeredSourcesTest_float;
TEST_P(AnnHnswLayeredSourcesTest_float, RegularBuild)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::regular);
}

TEST_P(AnnHnswLayeredSourcesTest_float, InMemoryAceBuild)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::inmem_ace);
}

TEST_P(AnnHnswLayeredSourcesTest_float, DeviceStandardAttached)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::device_standard_attached);
}

TEST_P(AnnHnswLayeredSourcesTest_float, DeviceStandardExplicit)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::device_standard_explicit);
}

TEST_P(AnnHnswLayeredSourcesTest_float, DevicePaddedAttached)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::device_padded_attached);
}

TEST_P(AnnHnswLayeredSourcesTest_float, DevicePaddedExplicit)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::device_padded_explicit);
}

TEST_P(AnnHnswLayeredSourcesTest_float, HostStandard)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::host_standard);
}

TEST_P(AnnHnswLayeredSourcesTest_float, HostPadded)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::host_padded);
}

INSTANTIATE_TEST_CASE_P(
  AnnHnswLayeredSourcesTest,
  AnnHnswLayeredSourcesTest_float,
  ::testing::Values(AnnHnswAceInputs{
    10, 2000, 17, 10, 2, 100, false, cuvs::distance::DistanceType::L2Expanded, 0.9}));

// Keep a small input regression alongside the larger conversion cases.
typedef AnnHnswAceTest<float, float, uint32_t> AnnHnswLayeredTinyTest_float;
TEST_P(AnnHnswLayeredTinyTest_float, RegularBuild)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::regular);
}
TEST_P(AnnHnswLayeredTinyTest_float, InMemoryAceBuild)
{
  this->testHnswAceLayeredBuildDeserializeSearch(LayeredSource::inmem_ace);
}
INSTANTIATE_TEST_CASE_P(
  AnnHnswLayeredTinyTest,
  AnnHnswLayeredTinyTest_float,
  ::testing::Values(AnnHnswAceInputs{
    10, 256, 17, 10, 2, 100, false, cuvs::distance::DistanceType::L2Expanded, 0.9}));

}  // namespace cuvs::neighbors::hnsw
