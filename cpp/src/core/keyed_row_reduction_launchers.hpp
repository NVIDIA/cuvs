/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::detail {
using raft::row_major;
template <typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
void reduce_rows_by_key(const DataIteratorT d_A,
                        IdxT lda,
                        const KeysIteratorT d_keys,
                        const WeightT* d_weights,
                        char* d_keys_char,
                        IdxT nrows,
                        IdxT ncols,
                        IdxT nkeys,
                        SumsT* d_sums,
                        cudaStream_t stream,
                        bool reset_sums = true);

template <typename DataIteratorT, typename KeysIteratorT, typename SumsT, typename IdxT>
void reduce_rows_by_key(const DataIteratorT d_A,
                        IdxT lda,
                        const KeysIteratorT d_keys,
                        char* d_keys_char,
                        IdxT nrows,
                        IdxT ncols,
                        IdxT nkeys,
                        SumsT* d_sums,
                        cudaStream_t stream,
                        bool reset_sums = true);

template <typename ElementType, typename KeyType, typename WeightType, typename IndexType>
void reduce_rows_by_key(
  raft::resources const& handle,
  raft::device_matrix_view<const ElementType, IndexType, raft::row_major> d_A,
  raft::device_vector_view<const KeyType, IndexType> d_keys,
  raft::device_matrix_view<ElementType, IndexType, raft::row_major> d_sums,
  IndexType n_unique_keys,
  raft::device_vector_view<char, IndexType> d_keys_char,
  std::optional<raft::device_vector_view<const WeightType, IndexType>> d_weights = std::nullopt,
  bool reset_sums                                                                = true);
}  // namespace cuvs::detail
