/**
 * SPDX-FileCopyrightText: Copyright (c) Facebook, Inc. and its affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file thirdparty/LICENSES/LICENSE.faiss
 */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

namespace cuvs::neighbors::detail::faiss_select {

template <typename T>
struct Comparator {
  __device__ static inline bool lt(T a, T b) { return a < b; }

  __device__ static inline bool gt(T a, T b) { return a > b; }
};

template <>
struct Comparator<half> {
  // `__hlt` / `__hgt` are FP16 ALU instructions and require sm_53. Older targets (sm_50) can only
  // store halves, so the comparison is done after widening to fp32 -- which is exact, since every
  // finite fp16 value is representable in fp32.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
  __device__ static inline bool lt(half a, half b) { return __hlt(a, b); }

  __device__ static inline bool gt(half a, half b) { return __hgt(a, b); }
#else
  __device__ static inline bool lt(half a, half b) { return __half2float(a) < __half2float(b); }

  __device__ static inline bool gt(half a, half b) { return __half2float(a) > __half2float(b); }
#endif
};

}  // namespace cuvs::neighbors::detail::faiss_select
