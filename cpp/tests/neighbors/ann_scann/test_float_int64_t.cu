/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_scann.cuh"

namespace cuvs::neighbors::scann {

using f32_i64 = scann_test<float, int64_t>;

TEST_BUILD(f32_i64)
TEST_BUILD_HOST_INPUT(f32_i64)
TEST_BUILD_HOST_INPUT_OVERLAP(f32_i64);

INSTANTIATE(f32_i64,
            defaults() + small_dims_all_pq_bits() + big_dims_all_pq_bits() + bf16() + bf16_avq() +
              avq() + soar() + two_level());

TEST_INVALID_BUILD_PARAMS()

}  // namespace cuvs::neighbors::scann
