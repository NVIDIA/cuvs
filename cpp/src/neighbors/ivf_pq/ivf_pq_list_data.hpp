/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdspan.hpp>
#include <variant>

namespace cuvs::neighbors::ivf_pq::detail {

// Interleaved layout versions
void unpack_list_data(raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes,
                      raft::device_mdspan<const uint8_t,
                                          list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                                          raft::row_major> list_data,
                      std::variant<uint32_t, const uint32_t*> offset_or_indices,
                      uint32_t pq_bits,
                      cuda::stream_ref stream);

void pack_list_data(raft::device_mdspan<uint8_t,
                                        list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                                        raft::row_major> list_data,
                    raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
                    std::variant<uint32_t, const uint32_t*> offset_or_indices,
                    uint32_t pq_bits,
                    cuda::stream_ref stream);

/** Unpack the list data; see the public interface for the api and usage. */
template <typename IdxT>
void unpack_list_data(raft::resources const& res,
                      const index<IdxT>& index,
                      raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
                      uint32_t label,
                      std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  auto stream = raft::resource::get_cuda_stream(res);
  // Currently only supports interleaved layout
  RAFT_EXPECTS(index.codes_layout() == list_layout::INTERLEAVED,
               "unpack_list_data currently only supports INTERLEAVED layout");
  auto typed_list =
    std::static_pointer_cast<const list_data_interleaved<IdxT>>(index.lists()[label]);
  unpack_list_data(out_codes, typed_list->data.view(), offset_or_indices, index.pq_bits(), stream);
}

template <typename IdxT>
void pack_list_data(raft::resources const& res,
                    index<IdxT>* index,
                    raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_codes,
                    uint32_t label,
                    std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  auto stream = raft::resource::get_cuda_stream(res);
  // Currently only supports interleaved layout
  RAFT_EXPECTS(index->codes_layout() == list_layout::INTERLEAVED,
               "pack_list_data currently only supports INTERLEAVED layout");
  auto typed_list = std::static_pointer_cast<list_data_interleaved<IdxT>>(index->lists()[label]);
  pack_list_data(typed_list->data.view(), new_codes, offset_or_indices, index->pq_bits(), stream);
}

}  // namespace cuvs::neighbors::ivf_pq::detail
