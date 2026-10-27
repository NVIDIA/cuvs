/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bitmap_csr_conversion_launchers.hpp"
#include "cub_ptx_query.hpp"

#include "cub_radix_launchers.hpp"
#include <cuvs/core/cuda_fp16.hpp>
#include <raft/sparse/convert/csr.cuh>

namespace cuvs::detail {

template <typename bitmap_t, typename index_t, typename csr_matrix_t, typename>
void bitmap_to_csr(raft::resources const& handle,
                   raft::core::bitmap_view<bitmap_t, index_t> bitmap,
                   csr_matrix_t& csr)

{
  return raft::sparse::convert::bitmap_to_csr(handle, bitmap, csr);
}

}  // namespace cuvs::detail
