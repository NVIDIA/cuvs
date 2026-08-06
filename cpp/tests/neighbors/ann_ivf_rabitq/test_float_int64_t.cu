/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_ivf_rabitq.cuh"

namespace cuvs::neighbors::ivf_rabitq {

using f32_f32_i64 = ivf_rabitq_test<float, float, int64_t>;

TEST_BUILD_SERIALIZE_SEARCH(f32_f32_i64)
TEST_BUILD_HOST_INPUT_SERIALIZE_SEARCH(f32_f32_i64)
TEST_BUILD_FORCED_STREAMING(f32_f32_i64)
INSTANTIATE(f32_f32_i64,
            defaults() + small_dims() + big_dims() + var_n_probes() + var_k() + var_bits_per_dim() +
              var_search_mode() + var_search_mode_1_bit());

// InnerProduct cases in their own instantiation so they can be run in isolation via
// --gtest_filter='IvfRabitqInnerProduct/*'.
INSTANTIATE_TEST_SUITE_P(IvfRabitqInnerProduct,  // NOLINT
                         f32_f32_i64,
                         ::testing::ValuesIn(var_metric()));

}  // namespace cuvs::neighbors::ivf_rabitq
