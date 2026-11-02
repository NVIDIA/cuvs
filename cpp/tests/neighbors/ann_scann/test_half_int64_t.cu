/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_scann.cuh"

namespace cuvs::neighbors::scann {

using f16_i64 = scann_test<half, int64_t>;

TEST_BUILD(f16_i64)
TEST_BUILD_HOST_INPUT(f16_i64)
TEST_BUILD_HOST_INPUT_OVERLAP(f16_i64);

INSTANTIATE(f16_i64,
            defaults() + small_dims_all_pq_bits() + big_dims_all_pq_bits() + bf16() + bf16_avq() +
              avq() + soar() + two_level());

}  // namespace cuvs::neighbors::scann
