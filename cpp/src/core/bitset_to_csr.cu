/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_transform_launchers.hpp"

#include "cub_scan_launchers.hpp"

#include "bitmap_csr_conversion_launchers.hpp"
#include "cub_ptx_query.hpp"

#include "cub_radix_launchers.hpp"
#include <cuvs/core/cuda_fp16.hpp>
#include <raft/sparse/convert/csr.cuh>

extern template void raft::sparse::convert::detail::calc_nnz_by_rows<unsigned int, long, long>(
  raft::resources const&, unsigned int const*, long, long, long*, size_t&, long&);

extern template void raft::sparse::convert::detail::gpu_repeat_csr<long, long>(
  raft::resources const&, long const*, long const*, long, long, long*, long*);

namespace cuvs::detail {

template <typename bitset_t, typename index_t, typename csr_matrix_t, typename>
void bitset_to_csr(raft::resources const& handle,
                   raft::core::bitset_view<bitset_t, index_t> bitset,
                   csr_matrix_t& csr)

{
  return raft::sparse::convert::bitset_to_csr(handle, bitset, csr);
}

}  // namespace cuvs::detail

template void cuvs::detail::bitset_to_csr(
  raft::resources const&,
  raft::core::bitset_view<unsigned int, long>,
  raft::csr_matrix<
    unsigned int,
    long,
    long,
    long,
    true,
    raft::device_container_policy,
    (raft::SparsityType)0,
    raft::compressed_structure<long, long, long, true, raft::device_container_policy>>&);
