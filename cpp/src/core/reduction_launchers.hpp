/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuvs/core/export.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/types.hpp>
#include <raft/linalg/norm_types.hpp>

namespace cuvs::spatial::knn::detail::utils {
template <typename T>
struct mapping;
}

namespace cuvs::detail {
using raft::Apply;
using raft::linalg::NormType;
template <NormType norm_type,
          bool rowMajor,
          typename Type,
          typename IdxType = int,
          typename Lambda  = raft::identity_op,
          typename OutType = Type>
void rowNorm(OutType* dots,
             const Type* data,
             IdxType D,
             IdxType N,
             cudaStream_t stream,
             Lambda fin_op = raft::identity_op());

template <NormType norm_type,
          raft::Apply apply,
          typename ElementType,
          typename OutputType,
          typename LayoutPolicy,
          typename IndexType,
          typename Lambda = raft::identity_op>
CUVS_EXPORT void norm(raft::resources const& handle,
                      raft::device_matrix_view<const ElementType, IndexType, LayoutPolicy> in,
                      raft::device_vector_view<OutputType, IndexType> out,
                      Lambda fin_op = raft::identity_op());

template <Apply apply,
          typename InElementType,
          typename LayoutPolicy,
          typename OutElementType = InElementType,
          typename IdxType        = std::uint32_t,
          typename MainLambda     = raft::identity_op,
          typename ReduceLambda   = raft::add_op,
          typename FinalLambda    = raft::identity_op>
void reduce(raft::resources const& handle,
            raft::device_matrix_view<const InElementType, IdxType, LayoutPolicy> data,
            raft::device_vector_view<OutElementType, IdxType> dots,
            OutElementType init,
            bool inplace           = false,
            MainLambda main_op     = raft::identity_op(),
            ReduceLambda reduce_op = raft::add_op(),
            FinalLambda final_op   = raft::identity_op());
}  // namespace cuvs::detail

namespace cuvs::detail {
template <typename Type, typename IdxType = int, typename Lambda = raft::identity_op>
void rowNormCsr(raft::resources const& handle,
                const IdxType* ia,
                const Type* data,
                const IdxType nnz,
                const IdxType N,
                Type* norm,
                raft::linalg::NormType type,
                Lambda fin_op = raft::identity_op());
}

namespace cuvs::detail {
template <typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void coalescedReduction(OutType* dots,
                        const InType* data,
                        IdxType D,
                        IdxType N,
                        OutType init,
                        cudaStream_t stream,
                        bool inplace           = false,
                        MainLambda main_op     = raft::identity_op(),
                        ReduceLambda reduce_op = raft::add_op(),
                        FinalLambda final_op   = raft::identity_op());

template <bool rowMajor, typename Type, typename IdxType = int>
void mean(Type* mu, const Type* data, IdxType D, IdxType N, cudaStream_t stream);

template <bool rowMajor, typename Type, typename IdxType = int>
void mean(Type* mu, const Type* data, IdxType D, IdxType N, bool sample, cudaStream_t stream);

template <typename in_value_t, typename out_value_t, typename idx_t, typename layout_t>
void mean(raft::resources const& handle,
          raft::device_matrix_view<const in_value_t, idx_t, layout_t> data,
          raft::device_vector_view<out_value_t, idx_t> mu);

template <typename value_t, typename idx_t, typename layout_t>
void mean(raft::resources const& handle,
          raft::device_matrix_view<const value_t, idx_t, layout_t> data,
          raft::device_vector_view<value_t, idx_t> mu);

template <typename value_t, typename idx_t, typename layout_t>
void mean(raft::resources const& handle,
          raft::device_matrix_view<const value_t, idx_t, layout_t> data,
          raft::device_vector_view<value_t, idx_t> mu,
          bool sample);
}  // namespace cuvs::detail

namespace cuvs::detail {
template <typename IndexT>
void compute_tf32_norms(raft::resources const& handle,
                        const float* matrix,
                        float* norms,
                        IndexT rows,
                        IndexT cols,
                        bool take_sqrt);
}
