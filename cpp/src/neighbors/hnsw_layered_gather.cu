/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "detail/hnsw_layered_gather.hpp"

#include <algorithm>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

namespace cuvs::neighbors::hnsw::detail {
namespace {
__global__ void gather_rows(const uint8_t* source,
                            size_t stride,
                            size_t row_bytes,
                            const size_t* row_ids,
                            size_t bytes,
                            uint8_t* output)
{
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < bytes;
       i += size_t(blockDim.x) * gridDim.x) {
    output[i] = source[size_t(row_ids[i / row_bytes]) * stride + i % row_bytes];
  }
}
}  // namespace

void gather_layered_hnsw_vectors(raft::resources const& res,
                                 const void* source,
                                 size_t source_stride_bytes,
                                 size_t row_bytes,
                                 const size_t* host_row_ids,
                                 size_t rows,
                                 void* host_output)
{
  if (rows == 0) { return; }
  RAFT_EXPECTS(row_bytes > 0 && source_stride_bytes >= row_bytes, "Invalid vector row stride");
  const auto stream = raft::resource::get_cuda_stream(res);
  const auto batch_rows =
    std::min(rows, std::max<size_t>(1, (64 * 1024 * 1024) / (row_bytes + sizeof(size_t))));
  rmm::device_uvector<size_t> row_ids(batch_rows, stream);
  rmm::device_uvector<uint8_t> values(batch_rows * row_bytes, stream);
  for (size_t start = 0; start < rows; start += batch_rows) {
    const auto count = std::min(batch_rows, rows - start);
    RAFT_CUDA_TRY(cudaMemcpyAsync(row_ids.data(),
                                  host_row_ids + start,
                                  count * sizeof(size_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
    gather_rows<<<std::min<size_t>(65535, (count * row_bytes + 255) / 256), 256, 0, stream>>>(
      static_cast<const uint8_t*>(source),
      source_stride_bytes,
      row_bytes,
      row_ids.data(),
      count * row_bytes,
      values.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    RAFT_CUDA_TRY(cudaMemcpyAsync(static_cast<uint8_t*>(host_output) + start * row_bytes,
                                  values.data(),
                                  count * row_bytes,
                                  cudaMemcpyDeviceToHost,
                                  stream));
    raft::resource::sync_stream(res);
  }
}
}  // namespace cuvs::neighbors::hnsw::detail
