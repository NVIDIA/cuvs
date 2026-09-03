/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ivf_pq_list_data_impl.cuh"
#include <cuda/stream>
#include <cuvs/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::ivf_pq::detail {
void unpack_list_data(raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes,
                      raft::device_mdspan<const uint8_t,
                                          list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                                          raft::row_major> list_data,
                      std::variant<uint32_t, const uint32_t*> offset_or_indices,
                      uint32_t pq_bits,
                      cuda::stream_ref stream)
{
  unpack_list_data_impl(codes, list_data, offset_or_indices, pq_bits, stream);
};

void pack_list_data(raft::device_mdspan<uint8_t,
                                        list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                                        raft::row_major> list_data,
                    raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
                    std::variant<uint32_t, const uint32_t*> offset_or_indices,
                    uint32_t pq_bits,
                    cuda::stream_ref stream)
{
  pack_list_data_impl(list_data, codes, offset_or_indices, pq_bits, stream);
};
};  // namespace cuvs::neighbors::ivf_pq::detail
