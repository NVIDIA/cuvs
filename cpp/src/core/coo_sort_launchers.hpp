/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>
#include <cuvs/core/export.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/coo.hpp>

namespace cuvs::detail {
using raft::sparse::COO;
template <typename T, typename IdxT = int, typename nnz_t>
void coo_sort(IdxT m, IdxT n, nnz_t nnz, IdxT* rows, IdxT* cols, T* vals, cudaStream_t stream);

template <typename T, typename IdxT = int, typename nnz_t>
void coo_sort(
  raft::resources const& handle, IdxT m, IdxT n, nnz_t nnz, IdxT* rows, IdxT* cols, T* vals);

template <typename T, typename IdxT = int, typename nnz_t>
void coo_sort(COO<T, IdxT, nnz_t>* const in, cudaStream_t stream);

template <typename value_idx>
CUVS_EXPORT void compute_duplicates_mask(
  value_idx* mask, const value_idx* rows, const value_idx* cols, size_t nnz, cudaStream_t stream);
}  // namespace cuvs::detail
