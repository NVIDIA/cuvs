/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <optional>
#include <raft/core/bitmap.hpp>
#include <raft/core/bitset.hpp>
#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/coo.hpp>
#include <type_traits>

namespace cuvs::detail {
template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
raft::device_csr_matrix<std::remove_const_t<ElementType>,
                        std::remove_const_t<IndptrType>,
                        std::remove_const_t<IndicesType>,
                        std::remove_const_t<NZType>>
compute_graph_laplacian(
  raft::resources const& res,
  raft::device_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType> input);

template <typename ElementType, typename RowType, typename ColType, typename NZType>
raft::device_coo_matrix<ElementType, RowType, ColType, NZType> compute_graph_laplacian(
  raft::resources const& res,
  raft::device_coo_matrix_view<ElementType, RowType, ColType, NZType> input);

template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
raft::device_csr_matrix<std::remove_const_t<ElementType>,
                        std::remove_const_t<IndptrType>,
                        std::remove_const_t<IndicesType>,
                        std::remove_const_t<NZType>>
laplacian_normalized(
  raft::resources const& res,
  raft::device_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType> input,
  raft::device_vector_view<ElementType, IndptrType> diagonal_out);

template <typename ElementType, typename RowType, typename ColType, typename NZType>
raft::device_coo_matrix<std::remove_const_t<ElementType>,
                        std::remove_const_t<RowType>,
                        std::remove_const_t<ColType>,
                        std::remove_const_t<NZType>>
laplacian_normalized(raft::resources const& res,
                     raft::device_coo_matrix_view<ElementType, RowType, ColType, NZType> input,
                     raft::device_vector_view<ElementType, RowType> diagonal_out);

template <typename value_idx, typename value_t, typename nnz_t>
CUVS_EXPORT void symmetrize(raft::resources const& handle,
                            const value_idx* rows,
                            const value_idx* cols,
                            const value_t* vals,
                            value_idx m,
                            value_idx n,
                            nnz_t nnz,
                            raft::sparse::COO<value_t, value_idx, nnz_t>& out);

template <typename value_t, typename output_t, typename index_t, typename nnz_t, typename bitmap_t>
void masked_matmul(raft::resources const& handle,
                   raft::device_matrix_view<const value_t, index_t, raft::row_major> A,
                   raft::device_matrix_view<const value_t, index_t, raft::row_major> B,
                   raft::core::bitmap_view<bitmap_t, index_t> mask,
                   raft::device_csr_matrix_view<output_t, index_t, index_t, nnz_t> C,
                   std::optional<raft::host_scalar_view<output_t>> alpha = std::nullopt,
                   std::optional<raft::host_scalar_view<output_t>> beta  = std::nullopt);

template <typename value_t, typename output_t, typename index_t, typename nnz_t, typename bitset_t>
void masked_matmul(raft::resources const& handle,
                   raft::device_matrix_view<const value_t, index_t, raft::row_major> A,
                   raft::device_matrix_view<const value_t, index_t, raft::row_major> B,
                   raft::core::bitset_view<bitset_t, index_t> mask,
                   raft::device_csr_matrix_view<output_t, index_t, index_t, nnz_t> C,
                   std::optional<raft::host_scalar_view<output_t>> alpha = std::nullopt,
                   std::optional<raft::host_scalar_view<output_t>> beta  = std::nullopt);
}  // namespace cuvs::detail

namespace cuvs::detail {
template <typename T, typename Lambda>
void coo_symmetrize(raft::sparse::COO<T>* in,
                    raft::sparse::COO<T>* out,
                    Lambda reduction_op,  // two-argument reducer
                    cudaStream_t stream);

template <int TPB_X = 128, typename T, typename IdxT, typename nnz_t, typename Lambda>
void coo_symmetrize(raft::resources const& handle,
                    raft::device_coo_matrix_view<const T, IdxT, IdxT, nnz_t> in,
                    raft::device_coo_matrix<T, IdxT, IdxT, nnz_t>& out,
                    Lambda reduction_op);

template <typename T, typename idx_t, typename nnz_t>
void coo_remove_scalar(const idx_t* rows,
                       const idx_t* cols,
                       const T* vals,
                       nnz_t nnz,
                       idx_t* crows,
                       idx_t* ccols,
                       T* cvals,
                       nnz_t* cnnz,
                       nnz_t* cur_cnnz,
                       T scalar,
                       idx_t n,
                       cudaStream_t stream);

template <typename T, typename idx_t, typename nnz_t>
void coo_remove_scalar(raft::sparse::COO<T, idx_t, nnz_t>* in,
                       raft::sparse::COO<T, idx_t, nnz_t>* out,
                       T scalar,
                       cudaStream_t stream);

template <int TPB_X, typename T, typename idx_t, typename nnz_t>
void coo_remove_scalar(raft::resources const& handle,
                       raft::device_coo_matrix_view<const T, idx_t, idx_t, nnz_t> in,
                       raft::host_scalar_view<const T> scalar,
                       raft::device_coo_matrix<T, idx_t, idx_t, nnz_t>& out);
}  // namespace cuvs::detail

namespace cuvs::detail {
template <typename NNZType>
void symmetrize_connectivity_graph(
  raft::resources const& handle,
  raft::device_coo_matrix_view<const float, int, int, NNZType> input,
  raft::device_coo_matrix<float, int, int, NNZType>& output);
}
