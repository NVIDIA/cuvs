/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_ivf_pq.cuh"

namespace cuvs::neighbors::ivf_pq {

using f32_i08_i64        = ivf_pq_test<float, int8_t, int64_t>;
using f32_f32_i64_filter = ivf_pq_filter_test<float, float, int64_t>;
using f32_i08_i64_filter = ivf_pq_filter_test<float, int8_t, int64_t>;

// Big-dimension cases of the device-input build + (filtered) search tests (see
// test_float_int64_t.cu and test_int8_t_int64_t.cu for the remaining cases).
TEST_BUILD_SEARCH(f32_i08_i64)
INSTANTIATE_BIG_DIMS(f32_i08_i64, big_dims());

TEST_BUILD_SEARCH(f32_f32_i64_filter)
INSTANTIATE_BIG_DIMS(f32_f32_i64_filter, big_dims_moderate_lut());

TEST_BUILD_SEARCH(f32_i08_i64_filter)
INSTANTIATE_BIG_DIMS(f32_i08_i64_filter, big_dims());

}  // namespace cuvs::neighbors::ivf_pq
