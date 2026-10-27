/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cuvs/core/export.hpp>
#include <optional>
#include <raft/core/bitmap.hpp>
#include <raft/core/bitset.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/coo.hpp>
#include <rmm/device_uvector.hpp>
namespace cuvs::detail {
using raft::col_major;
using raft::row_major;
template <typename bitmap_t,
          typename index_t,
          typename csr_matrix_t,
          typename = std::enable_if_t<raft::is_device_csr_matrix_v<csr_matrix_t>>>
void bitmap_to_csr(raft::resources const& handle,
                   raft::core::bitmap_view<bitmap_t, index_t> bitmap,
                   csr_matrix_t& csr);

template <typename bitset_t,
          typename index_t,
          typename csr_matrix_t,
          typename = std::enable_if_t<raft::is_device_csr_matrix_v<csr_matrix_t>>>
void bitset_to_csr(raft::resources const& handle,
                   raft::core::bitset_view<bitset_t, index_t> bitset,
                   csr_matrix_t& csr);
}  // namespace cuvs::detail

namespace cuvs::detail {
template <typename T, typename nnz_type, typename outT>
CUVS_EXPORT void sorted_coo_to_csr(
  const T* rows, nnz_type nnz, outT* row_ind, int m, cudaStream_t stream);

template <typename T, typename outT>
CUVS_EXPORT void sorted_coo_to_csr(raft::sparse::COO<T>* coo, outT* row_ind, cudaStream_t stream);
}  // namespace cuvs::detail
