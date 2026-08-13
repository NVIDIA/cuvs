/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/util/integer_utils.hpp>

#include <cstddef>
#include <cstdint>

namespace cuvs::neighbors {
template <class data_t, class math_t>
__global__ void decode_vpq_dataset_kernel(data_t* const decoded_dataset_ptr,
                                          const uint32_t ldd,
                                          const math_t* const vq_codebook_ptr,
                                          const uint32_t ldv,
                                          const math_t* const pq_codebook_ptr,
                                          const uint32_t pq_subspace_dim,
                                          const uint32_t pq_table_size,
                                          const uint32_t dataset_dim,
                                          const size_t dataset_size,
                                          const uint8_t* const data_ptr,
                                          const uint32_t ldi)
{
  constexpr uint32_t warp_size = 32;
  const size_t batch_id        = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;
  if (batch_id >= dataset_size) { return; }

  const auto local_data_ptr = data_ptr + ldi * batch_id;
  const auto vq_code        = *reinterpret_cast<const uint32_t*>(local_data_ptr);
  const auto pq_code_ptr    = local_data_ptr + sizeof(uint32_t);
  const auto vq_vec_ptr     = vq_codebook_ptr + vq_code * ldv;
  auto local_dst_ptr        = decoded_dataset_ptr + batch_id * ldd;

  const auto lane_id = threadIdx.x % warp_size;
  for (uint32_t i = lane_id; i < dataset_dim; i += warp_size) {
    const auto pq_code = pq_code_ptr[i / pq_subspace_dim];
    const auto pq_v    = pq_codebook_ptr[pq_code * pq_subspace_dim + (i % pq_subspace_dim)];

    local_dst_ptr[i] = static_cast<data_t>(vq_vec_ptr[i]) + static_cast<data_t>(pq_v);
  }
}

template <class data_t, class math_t>
void decode_vpq_dataset(raft::device_matrix_view<data_t, int64_t> decoded_dataset,
                        const cuvs::neighbors::device_vpq_dataset<math_t, int64_t>& vpq_dataset,
                        cudaStream_t cuda_stream)
{
  const auto dataset_size = decoded_dataset.extent(0);
  RAFT_EXPECTS(vpq_dataset.data.extent(0) == dataset_size, "Dataset sizes mismatch");
  RAFT_EXPECTS(vpq_dataset.pq_bits() == 8,
               "decode_vpq_dataset currently only supports pq_bits == 8 (got %u)",
               vpq_dataset.pq_bits());

  constexpr uint32_t block_size  = 256;
  constexpr uint32_t warp_size   = 32;
  constexpr int64_t vecs_per_cta = block_size / warp_size;
  const auto grid_size = raft::div_rounding_up_safe(decoded_dataset.extent(0), vecs_per_cta);

  decode_vpq_dataset_kernel<data_t, math_t>
    <<<grid_size, block_size, 0, cuda_stream>>>(decoded_dataset.data_handle(),
                                                decoded_dataset.stride(0),
                                                vpq_dataset.vq_code_book.data_handle(),
                                                vpq_dataset.vq_code_book.stride(0),
                                                vpq_dataset.pq_code_book.data_handle(),
                                                vpq_dataset.pq_len(),
                                                1u << vpq_dataset.pq_bits(),
                                                vpq_dataset.dim(),
                                                dataset_size,
                                                vpq_dataset.data.data_handle(),
                                                vpq_dataset.data.stride(0));
}
}  // namespace cuvs::neighbors
