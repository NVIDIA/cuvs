/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_scann.cuh"

namespace cuvs::neighbors::scann {

using i8_i64 = scann_test<int8_t, int64_t>;

TEST_BUILD(i8_i64)
TEST_BUILD_HOST_INPUT(i8_i64)
TEST_BUILD_HOST_INPUT_OVERLAP(i8_i64);

INSTANTIATE(i8_i64,
            defaults() + small_dims_all_pq_bits() + big_dims_all_pq_bits() + bf16() + bf16_avq() +
              avq() + soar() + two_level());

}  // namespace cuvs::neighbors::scann
