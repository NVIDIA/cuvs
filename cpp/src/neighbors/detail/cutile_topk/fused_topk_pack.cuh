/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::detail {

/** Copy stored coordinates unchanged and zero only the short alignment tail.
 *
 * This replaces a full-buffer memset plus a pitched device copy. Inputs and
 * outputs must not overlap. The caller has checked positive dimensions/rows
 * and destination byte-size overflow. No cached/prepared-corpus semantics.
 */
template <typename DataT, int ColumnsPerRow>
__global__ void fused_topk_pack_rows_kernel(
  DataT* destination, const DataT* source, size_t rows, int dimensions, int padded_dimensions)
{
  constexpr int rows_per_block = 256 / ColumnsPerRow;
  const size_t first_feature   = threadIdx.x % ColumnsPerRow;
  const size_t row_stride      = static_cast<size_t>(gridDim.x) * rows_per_block;
  for (size_t row = static_cast<size_t>(blockIdx.x) * rows_per_block + threadIdx.x / ColumnsPerRow;
       row < rows;
       row += row_stride) {
    const size_t input_row  = row * static_cast<size_t>(dimensions);
    const size_t output_row = row * static_cast<size_t>(padded_dimensions);
    for (size_t feature = first_feature; feature < static_cast<size_t>(padded_dimensions);
         feature += ColumnsPerRow) {
      destination[output_row + feature] =
        feature < static_cast<size_t>(dimensions) ? source[input_row + feature] : DataT(0.0f);
    }
  }
}

template <typename DataT>
void fused_topk_pack_rows(DataT* destination,
                          const DataT* source,
                          size_t rows,
                          int dimensions,
                          int padded_dimensions,
                          cudaStream_t stream)
{
  // Broad width classes only; no dimension-specific lookup table.
  if (padded_dimensions <= 64) {
    const auto blocks = static_cast<unsigned>(std::min<size_t>((rows + 7) / 8, 4096));
    fused_topk_pack_rows_kernel<DataT, 32>
      <<<blocks, 256, 0, stream>>>(destination, source, rows, dimensions, padded_dimensions);
  } else {
    const auto blocks = static_cast<unsigned>(std::min<size_t>((rows + 1) / 2, 4096));
    fused_topk_pack_rows_kernel<DataT, 128>
      <<<blocks, 256, 0, stream>>>(destination, source, rows, dimensions, padded_dimensions);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::neighbors::detail
