/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_vamana.cuh"

namespace cuvs::neighbors::vamana {

// A build that starts from a CAGRA graph. Every Vamana build draws its insertion order from rand(),
// so this file comes last in NEIGHBORS_ANN_VAMANA_TEST to leave the draws of the other cases as
// they were.
typedef AnnVamanaTest<float, float, std::uint32_t> AnnVamanaInitialGraphF_U32;
TEST_P(AnnVamanaInitialGraphF_U32, AnnVamana) { this->testVamana(); }

inline AnnVamanaInputs initial_graph_input()
{
  AnnVamanaInputs ps = inputs.front();  // 1000 rows, L2Expanded, device dataset
  ps.dim             = 16;
  ps.graph_degree    = 32;
  ps.visited_size    = 128;
  ps.insert_iters    = 1.0;
  ps.min_recall      = 0.9;
  ps.initial_graph   = true;
  return ps;
}

INSTANTIATE_TEST_CASE_P(AnnVamanaTest,
                        AnnVamanaInitialGraphF_U32,
                        ::testing::Values(initial_graph_input()));

}  // namespace cuvs::neighbors::vamana
