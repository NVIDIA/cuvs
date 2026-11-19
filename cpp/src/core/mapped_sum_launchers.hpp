/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace cuvs::detail {
template <typename InType,
          typename MapOp,
          typename IdxType = std::uint32_t,
          int TPB          = 256,
          typename... Args,
          typename OutType = InType>
void mapThenSumReduce(
  OutType* out, IdxType len, MapOp map, cudaStream_t stream, const InType* in, Args... args);
}
