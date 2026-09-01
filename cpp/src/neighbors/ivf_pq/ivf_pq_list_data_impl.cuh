/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "ivf_pq_codepacking.cuh"
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/pow2_utils.cuh>
#include <variant>

namespace cuvs::neighbors::ivf_pq::detail {

/**
 * A consumer for the `run_on_list` and `run_on_vector` that just flattens PQ codes
 * one-per-byte. That is, independent of the code width (pq_bits), one code uses
 * the whole byte, hence one vectors uses pq_dim bytes.
 */
struct unpack_codes {
  raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes;

  /**
   * Create a callable to be passed to `run_on_list`.
   *
   * @param[out] out_codes the destination for the read codes.
   */
  __device__ inline unpack_codes(
    raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes)
    : out_codes{out_codes}
  {
  }

  /**  Write j-th component (code) of the i-th vector into the output array. */
  __device__ inline void operator()(uint8_t code, uint32_t i, uint32_t j)
  {
    out_codes(i, j) = code;
  }
};

template <uint32_t BlockSize, uint32_t PqBits>
__launch_bounds__(BlockSize) static __global__ void unpack_list_data_kernel(
  raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
  raft::device_mdspan<const uint8_t,
                      list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                      raft::row_major> in_list_data,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  const uint32_t pq_dim = out_codes.extent(1);
  auto unpack_action    = unpack_codes{out_codes};
  run_on_list<PqBits>(in_list_data, offset_or_indices, out_codes.extent(0), pq_dim, unpack_action);
}

/**
 * Unpack flat PQ codes from an existing list by the given offset.
 *
 * @param[out] codes flat PQ codes, one code per byte [n_rows, pq_dim]
 * @param[in] list_data the packed ivf::list data.
 * @param[in] offset_or_indices how many records in the list to skip or the exact indices.
 * @param[in] pq_bits codebook size (1 << pq_bits)
 * @param[in] stream
 */
inline void unpack_list_data_impl(
  raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes,
  raft::device_mdspan<const uint8_t,
                      list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                      raft::row_major> list_data,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  cuda::stream_ref stream)
{
  auto n_rows = codes.extent(0);
  if (n_rows == 0) { return; }

  constexpr uint32_t kBlockSize = 256;
  dim3 blocks(raft::div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [pq_bits]() {
    switch (pq_bits) {
      case 4: return unpack_list_data_kernel<kBlockSize, 4>;
      case 5: return unpack_list_data_kernel<kBlockSize, 5>;
      case 6: return unpack_list_data_kernel<kBlockSize, 6>;
      case 7: return unpack_list_data_kernel<kBlockSize, 7>;
      case 8: return unpack_list_data_kernel<kBlockSize, 8>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }();
  kernel<<<blocks, threads, 0, stream.get()>>>(codes, list_data, offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * A producer for the `write_list` and `write_vector` reads the codes byte-by-byte. That is,
 * independent of the code width (pq_bits), one code uses the whole byte, hence one vectors uses
 * pq_dim bytes.
 */
struct pass_codes {
  raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes;

  /**
   * Create a callable to be passed to `run_on_list`.
   *
   * @param[in] codes the source codes.
   */
  __device__ inline pass_codes(
    raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes)
    : codes{codes}
  {
  }

  /** Read j-th component (code) of the i-th vector from the source. */
  __device__ inline auto operator()(uint32_t i, uint32_t j) const -> uint8_t { return codes(i, j); }
};

template <uint32_t BlockSize, uint32_t PqBits>
__launch_bounds__(BlockSize) static __global__ void pack_list_data_kernel(
  raft::device_mdspan<uint8_t,
                      list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                      raft::row_major> list_data,
  raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  write_list<PqBits, 1>(
    list_data, offset_or_indices, codes.extent(0), codes.extent(1), pass_codes{codes});
}

/**
 * Write flat PQ codes into an existing list by the given offset.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_rows).
 *
 * @param[out] list_data the packed ivf::list data.
 * @param[in] codes flat PQ codes, one code per byte [n_rows, pq_dim]
 * @param[in] offset_or_indices how many records in the list to skip or the exact indices.
 * @param[in] pq_bits codebook size (1 << pq_bits)
 * @param[in] stream
 */
inline void pack_list_data_impl(
  raft::device_mdspan<uint8_t,
                      list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                      raft::row_major> list_data,
  raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  cuda::stream_ref stream)
{
  auto n_rows = codes.extent(0);
  if (n_rows == 0) { return; }

  constexpr uint32_t kBlockSize = 256;
  dim3 blocks(raft::div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [pq_bits]() {
    switch (pq_bits) {
      case 4: return pack_list_data_kernel<kBlockSize, 4>;
      case 5: return pack_list_data_kernel<kBlockSize, 5>;
      case 6: return pack_list_data_kernel<kBlockSize, 6>;
      case 7: return pack_list_data_kernel<kBlockSize, 7>;
      case 8: return pack_list_data_kernel<kBlockSize, 8>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }();
  kernel<<<blocks, threads, 0, stream.get()>>>(list_data, codes, offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}
};  // namespace cuvs::neighbors::ivf_pq::detail
