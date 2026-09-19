/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_cagra_bbq.cuh"

#include <gtest/gtest.h>

namespace cuvs::neighbors::cagra {

TEST_P(AnnCagraBbqTest, AnnCagraBbqSearchRecall) { this->testSearchRecall(); }
TEST_P(AnnCagraBbqTest, AnnCagraBbqGraphShape) { this->testGraphShape(); }
TEST_P(AnnCagraBbqTest, AnnCagraBbqGraphOnlyBuild) { this->testGraphOnlyBuild(); }
TEST_P(AnnCagraBbqTest, AnnCagraBbqUnsupportedParams) { this->testUnsupportedParams(); }

INSTANTIATE_TEST_CASE_P(AnnCagraBbqTest, AnnCagraBbqTest, ::testing::ValuesIn(bbq_inputs));

}  // namespace cuvs::neighbors::cagra
