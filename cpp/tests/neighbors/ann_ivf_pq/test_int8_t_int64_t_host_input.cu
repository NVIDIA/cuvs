/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_ivf_pq.cuh"

namespace cuvs::neighbors::ivf_pq {

using f32_i08_i64 = ivf_pq_test<float, int8_t, int64_t>;

// Same cases as in test_int8_t_int64_t.cu, built from host input. These are kept in a separate
// test executable because the large-k var_k() cases make each int8 test variant expensive.
TEST_BUILD_HOST_INPUT_SEARCH(f32_i08_i64)
TEST_BUILD_HOST_INPUT_OVERLAP_SEARCH(f32_i08_i64)
INSTANTIATE(f32_i08_i64,
            defaults() + var_k() + enum_variety_l2() + enum_variety_ip() + enum_variety_cosine());

}  // namespace cuvs::neighbors::ivf_pq
