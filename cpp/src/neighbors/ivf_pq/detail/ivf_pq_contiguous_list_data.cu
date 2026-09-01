/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ivf_pq_contiguous_list_data_impl.cuh"
#include <cuvs/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::ivf_pq::detail {
void unpack_contiguous_list_data(
  uint8_t* codes,
  raft::device_mdspan<const uint8_t,
                      list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                      raft::row_major> list_data,
  uint32_t n_rows,
  uint32_t pq_dim,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  cuda::stream_ref stream)
{
  unpack_contiguous_list_data_impl(
    codes, list_data, n_rows, pq_dim, offset_or_indices, pq_bits, stream);
};

void pack_contiguous_list_data(
  raft::device_mdspan<uint8_t,
                      list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                      raft::row_major> list_data,
  const uint8_t* codes,
  uint32_t n_rows,
  uint32_t pq_dim,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  cuda::stream_ref stream)
{
  pack_contiguous_list_data_impl(
    list_data, codes, n_rows, pq_dim, offset_or_indices, pq_bits, stream);
};
};  // namespace cuvs::neighbors::ivf_pq::detail
