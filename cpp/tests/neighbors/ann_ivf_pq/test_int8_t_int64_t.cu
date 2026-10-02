/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_ivf_pq.cuh"

namespace cuvs::neighbors::ivf_pq {

using f32_i08_i64        = ivf_pq_test<float, int8_t, int64_t>;
using f32_i08_i64_filter = ivf_pq_filter_test<float, int8_t, int64_t>;

// The host-input build variants of these cases are in test_int8_t_int64_t_host_input.cu;
// big_dims() cases are instantiated in test_big_dims_*.cu
TEST_BUILD_SEARCH(f32_i08_i64)
TEST_BUILD_SERIALIZE_SEARCH(f32_i08_i64)
INSTANTIATE(f32_i08_i64,
            defaults() + var_k() + enum_variety_l2() + enum_variety_ip() + enum_variety_cosine());

TEST_BUILD_SEARCH(f32_i08_i64_filter)
INSTANTIATE(f32_i08_i64_filter,
            defaults() + var_k() + enum_variety_l2() + enum_variety_ip() + enum_variety_cosine());
}  // namespace cuvs::neighbors::ivf_pq
