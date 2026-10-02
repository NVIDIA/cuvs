/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_ivf_pq.cuh"

namespace cuvs::neighbors::ivf_pq {

using f32_f32_i64 = ivf_pq_test<float, float, int64_t>;
using f32_i08_i64 = ivf_pq_test<float, int8_t, int64_t>;

// Big-dimension cases of the host-input build tests (see test_float_int64_t.cu and
// test_int8_t_int64_t_host_input.cu for the remaining cases).
TEST_BUILD_HOST_INPUT_SEARCH(f32_f32_i64)
TEST_BUILD_HOST_INPUT_OVERLAP_SEARCH(f32_f32_i64)
INSTANTIATE_BIG_DIMS(f32_f32_i64, big_dims_moderate_lut());

TEST_BUILD_HOST_INPUT_SEARCH(f32_i08_i64)
TEST_BUILD_HOST_INPUT_OVERLAP_SEARCH(f32_i08_i64)
INSTANTIATE_BIG_DIMS(f32_i08_i64, big_dims());

}  // namespace cuvs::neighbors::ivf_pq
