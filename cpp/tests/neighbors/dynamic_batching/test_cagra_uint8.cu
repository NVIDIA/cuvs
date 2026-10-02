/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "test_cagra.cuh"

namespace cuvs::neighbors::dynamic_batching {

TEST_P(cagra_U8, defaults)
{
  set_default_cagra_params(*this);
  build_all();
  search_all();
  check_neighbors();
}

INSTANTIATE_TEST_CASE_P(dynamic_batching, cagra_U8, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::dynamic_batching
