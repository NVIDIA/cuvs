/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Multi-partition CAGRA search (AnnCagraMultiPartitionTest).

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

typedef AnnCagraMultiPartitionTest<float, std::int8_t, std::uint32_t>
  AnnCagraMultiPartitionTestI8_U32;
TEST_P(AnnCagraMultiPartitionTestI8_U32, Search) { this->testSearch(); }
TEST_P(AnnCagraMultiPartitionTestI8_U32, FilteredSearch) { this->testFilteredSearch(); }

INSTANTIATE_TEST_CASE_P(AnnCagraMultiPartitionTest,
                        AnnCagraMultiPartitionTestI8_U32,
                        ::testing::ValuesIn(inputs_mp));

}  // namespace cuvs::neighbors::cagra
