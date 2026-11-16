/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_vamana.cuh"

// The default-path tests reseed the process-wide rand() state with srand(), and every Vamana build
// draws its insertion order from it. These tests run in their own executable so that the existing
// Vamana tests keep the rand() sequence they had.

namespace cuvs::neighbors::vamana {

typedef AnnVamanaTest<float, float, std::uint32_t> AnnVamanaGraphInitTestF_U32;
TEST_P(AnnVamanaGraphInitTestF_U32, AnnVamana) { this->testVamana(); }
INSTANTIATE_TEST_CASE_P(AnnVamanaNNDescentInitTest,
                        AnnVamanaGraphInitTestF_U32,
                        ::testing::ValuesIn(nn_descent_init_inputs));

typedef AnnVamanaTest<float, half, std::uint32_t> AnnVamanaGraphInitTestF16_U32;
TEST_P(AnnVamanaGraphInitTestF16_U32, AnnVamana) { this->testVamana(); }
INSTANTIATE_TEST_CASE_P(AnnVamanaNNDescentInitTest,
                        AnnVamanaGraphInitTestF16_U32,
                        ::testing::ValuesIn(nn_descent_init_inputs));

typedef AnnVamanaTest<float, int8_t, std::uint32_t> AnnVamanaGraphInitTestI8_U32;
TEST_P(AnnVamanaGraphInitTestI8_U32, AnnVamana) { this->testVamana(); }
INSTANTIATE_TEST_CASE_P(AnnVamanaNNDescentInitTest,
                        AnnVamanaGraphInitTestI8_U32,
                        ::testing::ValuesIn(nn_descent_init_inputs));

typedef AnnVamanaTest<float, uint8_t, std::uint32_t> AnnVamanaGraphInitTestU8_U32;
TEST_P(AnnVamanaGraphInitTestU8_U32, AnnVamana) { this->testVamana(); }
INSTANTIATE_TEST_CASE_P(AnnVamanaNNDescentInitTest,
                        AnnVamanaGraphInitTestU8_U32,
                        ::testing::ValuesIn(nn_descent_init_inputs));

TEST(AnnVamanaGraphInit, DefaultIsUnchangedF_U32) { check_default_graph_init<float>(); }
TEST(AnnVamanaGraphInit, DefaultIsUnchangedF16_U32) { check_default_graph_init<half>(); }
TEST(AnnVamanaGraphInit, DefaultIsUnchangedI8_U32) { check_default_graph_init<int8_t>(); }
TEST(AnnVamanaGraphInit, DefaultIsUnchangedU8_U32) { check_default_graph_init<uint8_t>(); }

}  // namespace cuvs::neighbors::vamana
