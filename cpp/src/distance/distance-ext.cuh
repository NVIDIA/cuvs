/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "detail/kernels/rbf_fin_op.cuh"  // rbf_fin_op
#include <cuvs/distance/distance.hpp>     // cuvs::distance::DistanceType
#include <raft/core/device_mdspan.hpp>    // raft::device_matrix_view
#include <raft/core/operators.hpp>        // raft::identity_op
#include <raft/core/resources.hpp>        // raft::resources
#include <raft/util/raft_explicit.hpp>    // RAFT_EXPLICIT

#include <rmm/device_uvector.hpp>  // rmm::device_uvector

#include <cuda_fp16.h>

namespace cuvs {
namespace distance {

template <cuvs::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinalLambda,
          typename IdxT = int>
void distance(raft::resources const& handle,
              const DataT* x,
              const DataT* y,
              OutT* dist,
              IdxT m,
              IdxT n,
              IdxT k,
              void* workspace,
              size_t worksize,
              FinalLambda fin_op,
              bool isRowMajor = true,
              OutT metric_arg = 2.0f) RAFT_EXPLICIT;

template <cuvs::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT = int>
void distance(raft::resources const& handle,
              const DataT* x,
              const DataT* y,
              OutT* dist,
              IdxT m,
              IdxT n,
              IdxT k,
              void* workspace,
              size_t worksize,
              bool isRowMajor = true,
              OutT metric_arg = 2.0f) RAFT_EXPLICIT;

template <cuvs::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT = int>
size_t getWorkspaceSize(const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k) RAFT_EXPLICIT;

template <cuvs::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT = int,
          typename layout>
size_t getWorkspaceSize(raft::device_matrix_view<DataT, IdxT, layout> const& x,
                        raft::device_matrix_view<DataT, IdxT, layout> const& y) RAFT_EXPLICIT;

template <cuvs::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT = int>
void distance(raft::resources const& handle,
              const DataT* x,
              const DataT* y,
              OutT* dist,
              IdxT m,
              IdxT n,
              IdxT k,
              bool isRowMajor = true,
              OutT metric_arg = 2.0f) RAFT_EXPLICIT;

template <typename Type, typename IdxT = int, typename DistT = Type>
void pairwise_distance(raft::resources const& handle,
                       const Type* x,
                       const Type* y,
                       DistT* dist,
                       IdxT m,
                       IdxT n,
                       IdxT k,
                       rmm::device_uvector<char>& workspace,
                       cuvs::distance::DistanceType metric,
                       bool isRowMajor  = true,
                       DistT metric_arg = DistT(2.0f)) RAFT_EXPLICIT;

template <typename Type, typename IdxT = int, typename DistT = Type>
void pairwise_distance(raft::resources const& handle,
                       const Type* x,
                       const Type* y,
                       DistT* dist,
                       IdxT m,
                       IdxT n,
                       IdxT k,
                       cuvs::distance::DistanceType metric,
                       bool isRowMajor  = true,
                       DistT metric_arg = DistT(2.0f)) RAFT_EXPLICIT;

template <cuvs::distance::DistanceType DistT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename layout = raft::layout_c_contiguous,
          typename IdxT   = int>
void distance(raft::resources const& handle,
              raft::device_matrix_view<const DataT, IdxT, layout> const x,
              raft::device_matrix_view<const DataT, IdxT, layout> const y,
              raft::device_matrix_view<OutT, IdxT, layout> dist,
              OutT metric_arg = 2.0f) RAFT_EXPLICIT;

template <typename Type,
          typename layout = raft::layout_c_contiguous,
          typename IdxT   = int,
          typename DistT  = Type>
void pairwise_distance(raft::resources const& handle,
                       raft::device_matrix_view<const Type, IdxT, layout> const x,
                       raft::device_matrix_view<const Type, IdxT, layout> const y,
                       raft::device_matrix_view<DistT, IdxT, layout> dist,
                       cuvs::distance::DistanceType metric,
                       DistT metric_arg = DistT(2.0f)) RAFT_EXPLICIT;

};  // namespace distance
};  // namespace cuvs

/*
 * Hierarchy of instantiations:
 *
 * This file defines the extern template instantiations for the public API of
 * cuvs::distance. To improve compile times, the extern template instantiation
 * of the distance kernels is handled in
 * distance/detail/pairwise_matrix/dispatch-ext.cuh.
 *
 * After adding an instance here, make sure to also add the instance to
 * dispatch-ext.cuh and the corresponding .cu files.
 */

#define instantiate_cuvs_distance_distance(DistT, DataT, AccT, OutT, IdxT)             \
  extern template void                                                                 \
  cuvs::distance::distance<DistT, DataT, AccT, OutT, raft::identity_op, IdxT>(         \
    raft::resources const& handle,                                                     \
    const DataT* x,                                                                    \
    const DataT* y,                                                                    \
    OutT* dist,                                                                        \
    IdxT m,                                                                            \
    IdxT n,                                                                            \
    IdxT k,                                                                            \
    void* workspace,                                                                   \
    size_t worksize,                                                                   \
    raft::identity_op fin_op,                                                          \
    bool isRowMajor,                                                                   \
    OutT metric_arg);                                                                  \
                                                                                       \
  extern template void cuvs::distance::distance<DistT, DataT, AccT, OutT, IdxT>(       \
    raft::resources const& handle,                                                     \
    const DataT* x,                                                                    \
    const DataT* y,                                                                    \
    OutT* dist,                                                                        \
    IdxT m,                                                                            \
    IdxT n,                                                                            \
    IdxT k,                                                                            \
    void* workspace,                                                                   \
    size_t worksize,                                                                   \
    bool isRowMajor,                                                                   \
    OutT metric_arg);                                                                  \
                                                                                       \
  extern template void cuvs::distance::distance<DistT, DataT, AccT, OutT, IdxT>(       \
    raft::resources const& handle,                                                     \
    const DataT* x,                                                                    \
    const DataT* y,                                                                    \
    OutT* dist,                                                                        \
    IdxT m,                                                                            \
    IdxT n,                                                                            \
    IdxT k,                                                                            \
    bool isRowMajor,                                                                   \
    OutT metric_arg);                                                                  \
                                                                                       \
  extern template void                                                                 \
  cuvs::distance::distance<DistT, DataT, AccT, OutT, raft::layout_f_contiguous, IdxT>( \
    raft::resources const& handle,                                                     \
    raft::device_matrix_view<const DataT, IdxT, raft::layout_f_contiguous> const x,    \
    raft::device_matrix_view<const DataT, IdxT, raft::layout_f_contiguous> const y,    \
    raft::device_matrix_view<OutT, IdxT, raft::layout_f_contiguous> dist,              \
    OutT metric_arg);                                                                  \
                                                                                       \
  extern template void                                                                 \
  cuvs::distance::distance<DistT, DataT, AccT, OutT, raft::layout_c_contiguous, IdxT>( \
    raft::resources const& handle,                                                     \
    raft::device_matrix_view<const DataT, IdxT, raft::layout_c_contiguous> const x,    \
    raft::device_matrix_view<const DataT, IdxT, raft::layout_c_contiguous> const y,    \
    raft::device_matrix_view<OutT, IdxT, raft::layout_c_contiguous> dist,              \
    OutT metric_arg)

#define instantiate_cuvs_distance_distance_by_algo(DistT)                 \
  instantiate_cuvs_distance_distance(DistT, float, float, float, int);    \
  instantiate_cuvs_distance_distance(DistT, double, double, double, int); \
  instantiate_cuvs_distance_distance(DistT, half, float, float, int)

instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::Canberra);
instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::CorrelationExpanded);
instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::CosineExpanded);
instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::HammingUnexpanded);

instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::HellingerExpanded);
instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::InnerProduct);
instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::JensenShannon);
instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::KLDivergence);

instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::L1);
instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::L2Expanded);
instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::L2SqrtExpanded);
instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::L2SqrtUnexpanded);

instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::L2Unexpanded);
instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::Linf);
instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::LpUnexpanded);
instantiate_cuvs_distance_distance_by_algo(cuvs::distance::DistanceType::RusselRaoExpanded);

instantiate_cuvs_distance_distance(
  cuvs::distance::DistanceType::L2Expanded, float, float, float, int64_t);
instantiate_cuvs_distance_distance(
  cuvs::distance::DistanceType::L2Expanded, double, double, double, int64_t);
instantiate_cuvs_distance_distance(
  cuvs::distance::DistanceType::L2SqrtExpanded, float, float, float, int64_t);
instantiate_cuvs_distance_distance(
  cuvs::distance::DistanceType::L2SqrtExpanded, double, double, double, int64_t);

#undef instantiate_cuvs_distance_distance_by_algo
#undef instantiate_cuvs_distance_distance

// The following two instances are used in test/distance/gram.cu. Note the use
// of int64_t for the index type.
#define instantiate_cuvs_distance_distance_extra(DistT, DataT, AccT, OutT, FinalLambda, IdxT) \
  extern template void cuvs::distance::distance<DistT, DataT, AccT, OutT, FinalLambda, IdxT>( \
    raft::resources const& handle,                                                            \
    const DataT* x,                                                                           \
    const DataT* y,                                                                           \
    OutT* dist,                                                                               \
    IdxT m,                                                                                   \
    IdxT n,                                                                                   \
    IdxT k,                                                                                   \
    void* workspace,                                                                          \
    size_t worksize,                                                                          \
    FinalLambda fin_op,                                                                       \
    bool isRowMajor,                                                                          \
    DataT metric_arg)

instantiate_cuvs_distance_distance_extra(cuvs::distance::DistanceType::L2Unexpanded,
                                         float,
                                         float,
                                         float,
                                         cuvs::distance::kernels::rbf_fin_op<float>,
                                         int64_t);
instantiate_cuvs_distance_distance_extra(cuvs::distance::DistanceType::L2Unexpanded,
                                         double,
                                         double,
                                         double,
                                         cuvs::distance::kernels::rbf_fin_op<double>,
                                         int64_t);

#undef instantiate_cuvs_distance_distance_extra

#define instantiate_cuvs_distance_getWorkspaceSize(DistT, DataT, AccT, OutT, IdxT)             \
  extern template size_t cuvs::distance::getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT>(     \
    const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k);                                   \
                                                                                               \
  extern template size_t                                                                       \
  cuvs::distance::getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT, raft::layout_f_contiguous>( \
    raft::device_matrix_view<DataT, IdxT, raft::layout_f_contiguous> const& x,                 \
    raft::device_matrix_view<DataT, IdxT, raft::layout_f_contiguous> const& y);                \
                                                                                               \
  extern template size_t                                                                       \
  cuvs::distance::getWorkspaceSize<DistT, DataT, AccT, OutT, IdxT, raft::layout_c_contiguous>( \
    raft::device_matrix_view<DataT, IdxT, raft::layout_c_contiguous> const& x,                 \
    raft::device_matrix_view<DataT, IdxT, raft::layout_c_contiguous> const& y)

#define instantiate_cuvs_distance_getWorkspaceSize_by_algo(DistT)                     \
  instantiate_cuvs_distance_getWorkspaceSize(DistT, float, float, float, int);        \
  instantiate_cuvs_distance_getWorkspaceSize(DistT, double, double, double, int);     \
  instantiate_cuvs_distance_getWorkspaceSize(DistT, half, float, float, int);         \
  instantiate_cuvs_distance_getWorkspaceSize(DistT, float, float, float, int64_t);    \
  instantiate_cuvs_distance_getWorkspaceSize(DistT, double, double, double, int64_t); \
  instantiate_cuvs_distance_getWorkspaceSize(DistT, half, float, float, int64_t)

instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::Canberra);
instantiate_cuvs_distance_getWorkspaceSize_by_algo(
  cuvs::distance::DistanceType::CorrelationExpanded);
instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::CosineExpanded);
instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::HammingUnexpanded);

instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::HellingerExpanded);
instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::InnerProduct);
instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::JensenShannon);
instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::KLDivergence);

instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::L1);
instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::L2Expanded);
instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::L2SqrtExpanded);
instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::L2SqrtUnexpanded);

instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::L2Unexpanded);
instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::Linf);
instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::LpUnexpanded);
instantiate_cuvs_distance_getWorkspaceSize_by_algo(cuvs::distance::DistanceType::RusselRaoExpanded);

#undef instantiate_cuvs_distance_getWorkspaceSize_by_algo
#undef instantiate_cuvs_distance_getWorkspaceSize

#define instantiate_cuvs_distance_pairwise_distance(DataT, IdxT, DistT)                        \
  extern template void cuvs::distance::pairwise_distance(raft::resources const& handle,        \
                                                         const DataT* x,                       \
                                                         const DataT* y,                       \
                                                         DistT* dist,                          \
                                                         IdxT m,                               \
                                                         IdxT n,                               \
                                                         IdxT k,                               \
                                                         rmm::device_uvector<char>& workspace, \
                                                         cuvs::distance::DistanceType metric,  \
                                                         bool isRowMajor,                      \
                                                         DistT metric_arg)

instantiate_cuvs_distance_pairwise_distance(float, int, float);
instantiate_cuvs_distance_pairwise_distance(double, int, double);
instantiate_cuvs_distance_pairwise_distance(half, int, float);

#undef instantiate_cuvs_distance_pairwise_distance

// Same, but without workspace
#define instantiate_cuvs_distance_pairwise_distance(DataT, IdxT, DistT)                       \
  extern template void cuvs::distance::pairwise_distance(raft::resources const& handle,       \
                                                         const DataT* x,                      \
                                                         const DataT* y,                      \
                                                         DistT* dist,                         \
                                                         IdxT m,                              \
                                                         IdxT n,                              \
                                                         IdxT k,                              \
                                                         cuvs::distance::DistanceType metric, \
                                                         bool isRowMajor,                     \
                                                         DistT metric_arg)

instantiate_cuvs_distance_pairwise_distance(float, int, float);
instantiate_cuvs_distance_pairwise_distance(double, int, double);
instantiate_cuvs_distance_pairwise_distance(half, int, float);

#undef instantiate_cuvs_distance_pairwise_distance

#define instantiate_cuvs_distance_pairwise_distance(DataT, layout, IdxT, DistT) \
  extern template void cuvs::distance::pairwise_distance(                       \
    raft::resources const& handle,                                              \
    raft::device_matrix_view<const DataT, IdxT, layout> const x,                \
    raft::device_matrix_view<const DataT, IdxT, layout> const y,                \
    raft::device_matrix_view<DistT, IdxT, layout> dist,                         \
    cuvs::distance::DistanceType metric,                                        \
    DistT metric_arg)

instantiate_cuvs_distance_pairwise_distance(float, raft::layout_c_contiguous, int, float);
instantiate_cuvs_distance_pairwise_distance(float, raft::layout_f_contiguous, int, float);
instantiate_cuvs_distance_pairwise_distance(double, raft::layout_c_contiguous, int, double);
instantiate_cuvs_distance_pairwise_distance(double, raft::layout_f_contiguous, int, double);
instantiate_cuvs_distance_pairwise_distance(half, raft::layout_c_contiguous, int, float);
instantiate_cuvs_distance_pairwise_distance(half, raft::layout_f_contiguous, int, float);

#undef instantiate_cuvs_distance_pairwise_distance
