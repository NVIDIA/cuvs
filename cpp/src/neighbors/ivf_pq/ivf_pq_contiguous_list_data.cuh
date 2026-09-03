/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <cuda/stream>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdspan.hpp>
#include <variant>

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
  cuda::stream_ref stream);

template <typename IdxT>
void unpack_contiguous_list_data(raft::resources const& res,
                                 const index<IdxT>& index,
                                 uint8_t* out_codes,
                                 uint32_t n_rows,
                                 uint32_t label,
                                 std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  // Currently only supports interleaved layout
  RAFT_EXPECTS(index.codes_layout() == list_layout::INTERLEAVED,
               "unpack_contiguous_list_data currently only supports INTERLEAVED layout");
  auto typed_list =
    std::static_pointer_cast<const list_data_interleaved<IdxT>>(index.lists()[label]);
  unpack_contiguous_list_data(out_codes,
                              typed_list->data.view(),
                              n_rows,
                              index.pq_dim(),
                              offset_or_indices,
                              index.pq_bits(),
                              raft::resource::get_cuda_stream(res));
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
  cuda::stream_ref stream);

template <typename IdxT>
void pack_contiguous_list_data(raft::resources const& res,
                               index<IdxT>* index,
                               const uint8_t* new_codes,
                               uint32_t n_rows,
                               uint32_t label,
                               std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  // Currently only supports interleaved layout
  RAFT_EXPECTS(index->codes_layout() == list_layout::INTERLEAVED,
               "pack_contiguous_list_data currently only supports INTERLEAVED layout");
  auto typed_list = std::static_pointer_cast<list_data_interleaved<IdxT>>(index->lists()[label]);
  pack_contiguous_list_data(typed_list->data.view(),
                            new_codes,
                            n_rows,
                            index->pq_dim(),
                            offset_or_indices,
                            index->pq_bits(),
                            raft::resource::get_cuda_stream(res));
};

}  // namespace cuvs::neighbors::ivf_pq::detail
