/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/cuda_fp16.hpp>
#include <neighbors/ivf_pq/ivf_pq_encode.hpp>

#include <neighbors/ivf_pq/ivf_pq_codepacking.cuh>
#include <neighbors/ivf_pq/ivf_pq_process_and_fill_codes.cuh>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>

namespace cuvs::neighbors::ivf_pq::detail {

template <uint32_t BlockSize, uint32_t PqBits, typename IdxT>
__launch_bounds__(BlockSize) static __global__ void transform_codes_kernel(
  raft::device_matrix_view<float, IdxT> dataset_residual,
  raft::device_vector_view<uint32_t, IdxT> output_labels,
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> output_dataset,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
  codebook_gen codebook_kind)
{
  constexpr uint32_t kSubWarpSize = std::min<uint32_t>(raft::WarpSize, 1u << PqBits);
  using subwarp_align             = raft::Pow2<kSubWarpSize>;
  const uint32_t lane_id          = subwarp_align::mod(threadIdx.x);
  const IdxT row_ix = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= dataset_residual.extent(0)) { return; }

  const uint32_t cluster_ix = output_labels[row_ix];
  const uint32_t pq_dim     = dataset_residual.extent(1) / pq_centers.extent(1);

  auto encoder =
    encode_vectors<kSubWarpSize, IdxT>{pq_centers, dataset_residual, codebook_kind, cluster_ix};

  write_vector_flat<PqBits, kSubWarpSize, IdxT>(
    output_dataset.data_handle(), row_ix, row_ix, pq_dim, output_dataset.extent(1), encoder);
}

template <typename IdxT>
void encode_residuals(
  raft::device_matrix_view<float, IdxT> dataset_residual,
  raft::device_vector_view<uint32_t, IdxT> output_labels,
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> output_dataset,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
  codebook_gen codebook_kind,
  uint32_t pq_bits,
  uint32_t pq_book_size,
  cudaStream_t stream)
{
  constexpr uint32_t kBlockSize  = 256;
  const uint32_t threads_per_vec = std::min<uint32_t>(raft::WarpSize, pq_book_size);
  dim3 blocks(
    raft::div_rounding_up_safe<IdxT>(dataset_residual.extent(0), kBlockSize / threads_per_vec),
    1,
    1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4: return transform_codes_kernel<kBlockSize, 4, IdxT>;
      case 5: return transform_codes_kernel<kBlockSize, 5, IdxT>;
      case 6: return transform_codes_kernel<kBlockSize, 6, IdxT>;
      case 7: return transform_codes_kernel<kBlockSize, 7, IdxT>;
      case 8: return transform_codes_kernel<kBlockSize, 8, IdxT>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }(pq_bits);

  kernel<<<blocks, threads, 0, stream>>>(
    dataset_residual, output_labels, output_dataset, pq_centers, codebook_kind);
}

template void encode_residuals<long>(
  raft::device_matrix_view<float, long> dataset_residual,
  raft::device_vector_view<uint32_t, long> output_labels,
  raft::device_matrix_view<uint8_t, long, raft::row_major> output_dataset,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
  codebook_gen codebook_kind,
  uint32_t pq_bits,
  uint32_t pq_book_size,
  cudaStream_t stream);

}  // namespace cuvs::neighbors::ivf_pq::detail
