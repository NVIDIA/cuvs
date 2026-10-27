/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/stream>
#include <optional>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/select_k_types.hpp>

namespace cuvs::detail {
template <typename T, typename Index>
void select_k(raft::resources const& handle,
              T const* values,
              Index const* indices,
              size_t rows,
              size_t columns,
              int k,
              T* selected_values,
              Index* selected_indices,
              bool select_min,
              bool sorted,
              raft::matrix::SelectAlgo algorithm,
              Index const* row_lengths);

template <typename T, typename Index>
void select_k(
  raft::resources const& handle,
  raft::device_matrix_view<const T, int64_t, raft::row_major> values,
  std::optional<raft::device_matrix_view<const Index, int64_t, raft::row_major>> indices,
  raft::device_matrix_view<T, int64_t, raft::row_major> selected_values,
  raft::device_matrix_view<Index, int64_t, raft::row_major> selected_indices,
  bool select_min,
  bool sorted                        = false,
  raft::matrix::SelectAlgo algorithm = raft::matrix::SelectAlgo::kAuto);

template <typename T, typename Index>
void select_k(raft::resources const& handle,
              raft::device_csr_matrix_view<const T, Index, Index, Index> values,
              std::optional<raft::device_vector_view<const Index, Index>> indices,
              raft::device_matrix_view<T, Index, raft::row_major> selected_values,
              raft::device_matrix_view<Index, Index, raft::row_major> selected_indices,
              bool select_min,
              bool sorted                        = false,
              raft::matrix::SelectAlgo algorithm = raft::matrix::SelectAlgo::kAuto);

struct warpsort_launch_params {
  int block_size;
  int min_grid_size;
};

template <template <int, bool, typename, typename> class WarpSortClass,
          typename T,
          typename IdxT,
          typename RowLayout>
struct warpsort_launch_setup {
  static warpsort_launch_params calc_optimal_params(int k, int block_size_limit);

  static void kernel(int k,
                     bool select_min,
                     size_t batch_size,
                     size_t len,
                     int num_blocks,
                     int block_dim,
                     size_t smem_size,
                     const T* in_key,
                     const IdxT* in_idx,
                     const IdxT* in_indptr,
                     T* out_key,
                     IdxT* out_idx,
                     cuda::stream_ref stream);
};

}  // namespace cuvs::detail
