/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::detail {
using raft::row_major;
template <typename InputIteratorT, typename MapIteratorT, typename OutputIteratorT, typename IndexT>
CUVS_EXPORT void gather(const InputIteratorT in,
                        IndexT D,
                        IndexT N,
                        const MapIteratorT map,
                        IndexT map_length,
                        OutputIteratorT out,
                        cudaStream_t stream);

template <typename InputIteratorT,
          typename MapIteratorT,
          typename MapTransformOp,
          typename OutputIteratorT,
          typename IndexT>
CUVS_EXPORT void gather(const InputIteratorT in,
                        IndexT D,
                        IndexT N,
                        const MapIteratorT map,
                        IndexT map_length,
                        OutputIteratorT out,
                        MapTransformOp transform_op,
                        cudaStream_t stream);

template <typename matrix_t,
          typename map_t,
          typename idx_t,
          typename map_xform_t = raft::identity_op>
CUVS_EXPORT void gather(const raft::resources& handle,
                        raft::device_matrix_view<const matrix_t, idx_t, row_major> in,
                        raft::device_vector_view<const map_t, idx_t> map,
                        raft::device_matrix_view<matrix_t, idx_t, row_major> out,
                        map_xform_t transform_op = raft::identity_op());

template <typename matrix_t,
          typename map_t,
          typename idx_t,
          typename accessor,
          typename map_xform_t = raft::identity_op>
CUVS_EXPORT void gather(
  const raft::resources& handle,
  raft::mdspan<const matrix_t, raft::matrix_extent<idx_t>, raft::layout_stride, accessor> in,
  raft::device_vector_view<const map_t, idx_t> map,
  raft::device_matrix_view<matrix_t, idx_t, row_major> out,
  map_xform_t transform_op = raft::identity_op());

template <typename matrix_t,
          typename map_t,
          typename idx_t,
          typename map_xform_t = raft::identity_op>
CUVS_EXPORT void gather(raft::resources const& handle,
                        raft::device_matrix_view<matrix_t, idx_t, raft::layout_c_contiguous> inout,
                        raft::device_vector_view<const map_t, idx_t, raft::layout_c_contiguous> map,
                        idx_t col_batch_size     = 0,
                        map_xform_t transform_op = raft::identity_op());

template <typename matrix_t,
          typename map_t,
          typename idx_t,
          typename map_xform_t = raft::identity_op>
CUVS_EXPORT void gather(raft::resources const& handle,
                        raft::device_matrix_view<matrix_t, idx_t, raft::layout_stride> inout,
                        raft::device_vector_view<const map_t, idx_t, raft::layout_c_contiguous> map,
                        idx_t col_batch_size     = 0,
                        map_xform_t transform_op = raft::identity_op());

template <typename m_t, typename idx_t, typename layout>
void copy_rows(raft::resources const& handle,
               raft::device_matrix_view<const m_t, idx_t, layout> in,
               raft::device_matrix_view<m_t, idx_t, layout> out,
               raft::device_vector_view<const idx_t, idx_t> indices);
}  // namespace cuvs::detail
