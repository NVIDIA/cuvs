/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_scann.cuh"

namespace cuvs::neighbors::scann {

using u8_i64 = scann_test<uint8_t, int64_t>;

TEST_BUILD(u8_i64)
TEST_BUILD_HOST_INPUT(u8_i64)
TEST_BUILD_HOST_INPUT_OVERLAP(u8_i64);

INSTANTIATE(u8_i64,
            defaults() + small_dims_all_pq_bits() + big_dims_all_pq_bits() + bf16() + bf16_avq() +
              avq() + soar() + two_level());

}  // namespace cuvs::neighbors::scann
