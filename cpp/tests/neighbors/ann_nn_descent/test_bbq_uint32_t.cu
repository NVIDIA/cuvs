/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_nn_descent_bbq.cuh"

#include <gtest/gtest.h>

namespace cuvs::neighbors::nn_descent {
TEST_P(AnnNNDescentBbqTest, AnnNNDescentBbq) { this->testNNDescent(); }

INSTANTIATE_TEST_CASE_P(AnnNNDescentBbqTest, AnnNNDescentBbqTest, ::testing::ValuesIn(bbq_inputs));
}  // namespace cuvs::neighbors::nn_descent
