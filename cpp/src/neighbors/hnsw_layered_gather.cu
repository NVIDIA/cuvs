/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "detail/hnsw_layered_gather.hpp"

#include <algorithm>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/gather.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cuvs::neighbors::hnsw::detail {
namespace {
// Present padded rows as a compact byte matrix to RAFT's iterator-based gather.
struct strided_source_byte {
  const uint8_t* source;
  size_t stride;
  size_t row_bytes;

  __host__ __device__ uint8_t operator()(size_t offset) const
  {
    return source[(offset / row_bytes) * stride + offset % row_bytes];
  }
};
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
  cudaStream_t stream     = raft::resource::get_cuda_stream(res).get();
  const auto source_bytes = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_t>(0),
    strided_source_byte{static_cast<const uint8_t*>(source), source_stride_bytes, row_bytes});
  // The source has at least enough rows to include every requested row ID.
  const auto source_rows = *std::max_element(host_row_ids, host_row_ids + rows) + size_t{1};
  const auto batch_rows =
    std::min(rows, std::max<size_t>(1, (64 * 1024 * 1024) / (row_bytes + sizeof(size_t))));
  rmm::device_uvector<size_t> row_ids(batch_rows, stream);
  rmm::device_uvector<uint8_t> values(batch_rows * row_bytes, stream);
  for (size_t start = 0; start < rows; start += batch_rows) {
    const auto count = std::min(batch_rows, rows - start);
    raft::update_device(row_ids.data(), host_row_ids + start, count, stream);
    raft::matrix::gather(
      source_bytes, row_bytes, source_rows, row_ids.data(), count, values.data(), stream);
    RAFT_CUDA_TRY(cudaGetLastError());
    raft::update_host(static_cast<uint8_t*>(host_output) + start * row_bytes,
                      values.data(),
                      count * row_bytes,
                      stream);
    raft::resource::sync_stream(res);
  }
}
}  // namespace cuvs::neighbors::hnsw::detail
