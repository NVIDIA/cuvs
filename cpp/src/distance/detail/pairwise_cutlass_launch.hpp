/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace cuvs::distance::detail {
template <typename T, int Alignment, bool RowMajor>
void launch_pairwise_cutlass_l2(int m,
                                int n,
                                int k,
                                const T* a,
                                const T* b,
                                const T* xn,
                                T* yn,
                                T* output,
                                int64_t lda,
                                int64_t ldb,
                                int64_t ldd,
                                bool sqrt,
                                cudaStream_t stream);
}  // namespace cuvs::distance::detail
