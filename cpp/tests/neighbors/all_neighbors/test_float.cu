/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../all_neighbors.cuh"

namespace cuvs::neighbors::all_neighbors {

typedef AllNeighborsTest<float, float> AllNeighborsTestF;
TEST_P(AllNeighborsTestF, AllNeighbors) { this->run(); }

INSTANTIATE_TEST_CASE_P(AllNeighborsSingleTest,
                        AllNeighborsTestF,
                        ::testing::ValuesIn(inputsSingle));
INSTANTIATE_TEST_CASE_P(AllNeighborsSingleTestDataTransfer,
                        AllNeighborsTestF,
                        ::testing::ValuesIn(inputsSingleDataTransfer));

INSTANTIATE_TEST_CASE_P(AllNeighborsBatchTestLow,
                        AllNeighborsTestF,
                        ::testing::ValuesIn(inputsBatchLow));
INSTANTIATE_TEST_CASE_P(AllNeighborsBatchTestMed,
                        AllNeighborsTestF,
                        ::testing::ValuesIn(inputsBatchMed));
INSTANTIATE_TEST_CASE_P(AllNeighborsBatchTestHigh,
                        AllNeighborsTestF,
                        ::testing::ValuesIn(inputsBatchHigh));

INSTANTIATE_TEST_CASE_P(AllNeighborsSingleMutualTest,
                        AllNeighborsTestF,
                        ::testing::ValuesIn(mutualReachSingle));
INSTANTIATE_TEST_CASE_P(AllNeighborsSingleMutualTestDataTransfer,
                        AllNeighborsTestF,
                        ::testing::ValuesIn(mutualReachSingleDataTransfer));

INSTANTIATE_TEST_CASE_P(AllNeighborsBatchMutualTestLow,
                        AllNeighborsTestF,
                        ::testing::ValuesIn(mutualReachBatchLow));
INSTANTIATE_TEST_CASE_P(AllNeighborsBatchMutualTestMed,
                        AllNeighborsTestF,
                        ::testing::ValuesIn(mutualReachBatchMed));
INSTANTIATE_TEST_CASE_P(AllNeighborsBatchMutualTestHigh,
                        AllNeighborsTestF,
                        ::testing::ValuesIn(mutualReachBatchHigh));
}  // namespace cuvs::neighbors::all_neighbors
