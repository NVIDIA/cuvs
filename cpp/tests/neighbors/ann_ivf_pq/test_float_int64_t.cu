/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_ivf_pq.cuh"

namespace cuvs::neighbors::ivf_pq {

using f32_f32_i64             = ivf_pq_test<float, float, int64_t>;
using f32_f32_i64_flat_layout = ivf_pq_flat_layout_test<float, float, int64_t>;
using f32_f32_i64_filter      = ivf_pq_filter_test<float, float, int64_t>;

// big_dims_moderate_lut() cases are instantiated in test_big_dims_*.cu
TEST_BUILD_HOST_INPUT_SEARCH(f32_f32_i64)
TEST_BUILD_HOST_INPUT_OVERLAP_SEARCH(f32_f32_i64)
TEST_BUILD_EXTEND_SEARCH(f32_f32_i64)
TEST_BUILD_SERIALIZE_SEARCH(f32_f32_i64)
TEST_BUILD_PRECOMPUTED(f32_f32_i64)
INSTANTIATE(f32_f32_i64,
            defaults() + small_dims() + enum_variety_l2() + enum_variety_l2sqrt() +
              enum_variety_ip() + enum_variety_cosine());

TEST_FLAT_LAYOUT_CODES(f32_f32_i64_flat_layout)
INSTANTIATE(f32_f32_i64_flat_layout, flat_layout_tests());

TEST_BUILD_SEARCH(f32_f32_i64_filter)
INSTANTIATE(f32_f32_i64_filter,
            defaults() + small_dims() + enum_variety_l2() + enum_variety_l2sqrt() +
              enum_variety_ip() + enum_variety_cosine());

}  // namespace cuvs::neighbors::ivf_pq
