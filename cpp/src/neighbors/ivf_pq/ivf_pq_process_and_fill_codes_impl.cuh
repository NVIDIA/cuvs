/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "ivf_pq_codepacking.cuh"
#include "ivf_pq_process_and_fill_codes.cuh"
#include <cstdint>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/pow2_utils.cuh>
#include <variant>

namespace cuvs::neighbors::ivf_pq::detail {

template <uint32_t BlockSize, uint32_t PqBits, typename IdxT>
__launch_bounds__(BlockSize) static __global__ void process_and_fill_codes_kernel(
  raft::device_matrix_view<const float, IdxT, raft::row_major> new_vectors,
  std::variant<IdxT, const IdxT*> src_offset_or_indices,
  const uint32_t* new_labels,
  raft::device_vector_view<uint32_t, uint32_t, raft::row_major> list_sizes,
  raft::device_vector_view<IdxT*, uint32_t, raft::row_major> inds_ptrs,
  raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> data_ptrs,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
  codebook_gen codebook_kind,
  list_layout codes_layout,
  uint32_t bytes_per_vector)
{
  constexpr uint32_t kSubWarpSize = std::min<uint32_t>(raft::WarpSize, 1u << PqBits);
  using subwarp_align             = raft::Pow2<kSubWarpSize>;
  const uint32_t lane_id          = subwarp_align::mod(threadIdx.x);
  const IdxT row_ix = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= new_vectors.extent(0)) { return; }

  const uint32_t cluster_ix = new_labels[row_ix];
  uint32_t out_ix;
  if (lane_id == 0) { out_ix = atomicAdd(&list_sizes(cluster_ix), 1); }
  out_ix = raft::shfl(out_ix, 0, kSubWarpSize);

  // write the label (one record per subwarp)
  auto pq_indices = inds_ptrs(cluster_ix);
  if (lane_id == 0) {
    if (std::holds_alternative<IdxT>(src_offset_or_indices)) {
      pq_indices[out_ix] = std::get<IdxT>(src_offset_or_indices) + row_ix;
    } else {
      pq_indices[out_ix] = std::get<const IdxT*>(src_offset_or_indices)[row_ix];
    }
  }

  // write the codes (one record per subwarp)
  const uint32_t pq_dim = new_vectors.extent(1) / pq_centers.extent(1);
  auto encode_action =
    encode_vectors<kSubWarpSize, IdxT>{pq_centers, new_vectors, codebook_kind, cluster_ix};

  if (codes_layout == list_layout::FLAT) {
    write_vector_flat<PqBits, kSubWarpSize>(
      data_ptrs[cluster_ix], out_ix, row_ix, pq_dim, bytes_per_vector, encode_action);
  } else {
    auto pq_extents =
      list_spec_interleaved<uint32_t, IdxT>{PqBits, pq_dim, true}.make_list_extents(out_ix + 1);
    auto pq_dataset = raft::make_mdspan<uint8_t, uint32_t, raft::row_major, false, true>(
      data_ptrs[cluster_ix], pq_extents);
    write_vector_interleaved<PqBits, kSubWarpSize>(
      pq_dataset, out_ix, row_ix, pq_dim, encode_action);
  }
}

template <typename IdxT>
void launch_process_and_fill_codes_kernel(
  raft::resources const& handle,
  index<IdxT>& index,
  raft::device_matrix_view<float, IdxT> new_vectors_residual,
  std::variant<IdxT, const IdxT*> src_offset_or_indices,
  const uint32_t* new_labels,
  IdxT n_rows)
{
  constexpr uint32_t kBlockSize  = 256;
  const uint32_t threads_per_vec = std::min<uint32_t>(raft::WarpSize, index.pq_book_size());
  dim3 blocks(raft::div_rounding_up_safe<IdxT>(n_rows, kBlockSize / threads_per_vec), 1, 1);
  dim3 threads(kBlockSize, 1, 1);

  const uint32_t bytes_per_vector =
    raft::div_rounding_up_safe(index.pq_dim() * index.pq_bits(), 8u);

  auto kernel = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4: return process_and_fill_codes_kernel<kBlockSize, 4, IdxT>;
      case 5: return process_and_fill_codes_kernel<kBlockSize, 5, IdxT>;
      case 6: return process_and_fill_codes_kernel<kBlockSize, 6, IdxT>;
      case 7: return process_and_fill_codes_kernel<kBlockSize, 7, IdxT>;
      case 8: return process_and_fill_codes_kernel<kBlockSize, 8, IdxT>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }(index.pq_bits());

  kernel<<<blocks, threads, 0, raft::resource::get_cuda_stream(handle).get()>>>(
    new_vectors_residual,
    src_offset_or_indices,
    new_labels,
    index.list_sizes(),
    index.inds_ptrs(),
    index.data_ptrs(),
    index.pq_centers(),
    index.codebook_kind(),
    index.codes_layout(),
    bytes_per_vector);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}
}  // namespace cuvs::neighbors::ivf_pq::detail
