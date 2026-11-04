/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/norm_types.hpp>

namespace cuvs::detail {

using raft::row_major;
using raft::linalg::NormType;

template <typename ElementType,
          typename IndexType,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
void row_normalize(raft::resources const& handle,
                   raft::device_matrix_view<const ElementType, IndexType, row_major> in,
                   raft::device_matrix_view<ElementType, IndexType, row_major> out,
                   ElementType init,
                   MainLambda main_op,
                   ReduceLambda reduce_op,
                   FinalLambda fin_op,
                   ElementType eps = ElementType(1e-8));

template <NormType norm_type, typename ElementType, typename IndexType>
void row_normalize(raft::resources const& handle,
                   raft::device_matrix_view<const ElementType, IndexType, row_major> in,
                   raft::device_matrix_view<ElementType, IndexType, row_major> out,
                   ElementType eps = ElementType(1e-8));

}  // namespace cuvs::detail
