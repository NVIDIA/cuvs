/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../distance_ops/all_ops.cuh"    // ops::*
#include "../distance_ops/cutlass.cuh"    // ops::has_cutlass_op
#include "../kernels/rbf_fin_op.cuh"      // rbf_fin_op
#include "../pairwise_matrix/params.cuh"  // pairwise_matrix_params
#include <raft/core/operators.hpp>        // raft::identity_op
#include <raft/util/raft_explicit.hpp>    // RAFT_EXPLICIT

namespace cuvs::distance::detail {

template <typename OpT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void pairwise_matrix_dispatch(OpT distance_op,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              const DataT* x,
                              const DataT* y,
                              const OutT* x_norm,
                              const OutT* y_norm,
                              OutT* out,
                              FinOpT fin_op,
                              cudaStream_t stream,
                              bool is_row_major) RAFT_EXPLICIT;

};  // namespace cuvs::distance::detail

#define instantiate_cuvs_distance_detail_pairwise_matrix_dispatch(                     \
  OpT, DataT, AccT, OutT, FinOpT, IdxT)                                                \
  extern template void cuvs::distance::detail::                                        \
    pairwise_matrix_dispatch<OpT<DataT, AccT, IdxT>, DataT, AccT, OutT, FinOpT, IdxT>( \
      OpT<DataT, AccT, IdxT> distance_op,                                              \
      IdxT m,                                                                          \
      IdxT n,                                                                          \
      IdxT k,                                                                          \
      const DataT* x,                                                                  \
      const DataT* y,                                                                  \
      const OutT* x_norm,                                                              \
      const OutT* y_norm,                                                              \
      OutT* out,                                                                       \
      FinOpT fin_op,                                                                   \
      cudaStream_t stream,                                                             \
      bool is_row_major)

#define instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(OpT, IdxT) \
  instantiate_cuvs_distance_detail_pairwise_matrix_dispatch(                                 \
    OpT, float, float, float, raft::identity_op, IdxT);                                      \
  instantiate_cuvs_distance_detail_pairwise_matrix_dispatch(                                 \
    OpT, double, double, double, raft::identity_op, IdxT);                                   \
  instantiate_cuvs_distance_detail_pairwise_matrix_dispatch(                                 \
    OpT, half, float, float, raft::identity_op, IdxT);

#define instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo(OpT, IdxT, FinOpT) \
  instantiate_cuvs_distance_detail_pairwise_matrix_dispatch(                                 \
    OpT, float, float, float, FinOpT<float>, IdxT);                                          \
  instantiate_cuvs_distance_detail_pairwise_matrix_dispatch(                                 \
    OpT, double, double, double, FinOpT<double>, IdxT);                                      \
  instantiate_cuvs_distance_detail_pairwise_matrix_dispatch(                                 \
    OpT, half, float, float, FinOpT<float>, IdxT);

/*
 * Hierarchy of instantiations:
 *
 * This file defines extern template instantiations of the distance kernels. The
 * instantiation of the public API is handled in cuvs/distance/distance-ext.cuh.
 *
 * After adding an instance here, make sure to also add the instance there.
 */

// The following two instances are used in the RBF kernel object. Note the use of int64_t for the
// index type.
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::canberra_distance_op, int);
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::correlation_distance_op, int);
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::cosine_distance_op, int);
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::hamming_distance_op, int);
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::hellinger_distance_op, int);
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::jensen_shannon_distance_op, int);
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::kl_divergence_op, int);
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::l1_distance_op, int);
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::l2_exp_distance_op, int);
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::l2_exp_distance_op, int64_t);
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::l2_unexp_distance_op, int);
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::l_inf_distance_op, int);
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::lp_unexp_distance_op, int);
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::russel_rao_distance_op, int);
instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo(
  cuvs::distance::detail::ops::l2_unexp_distance_op, int64_t, cuvs::distance::kernels::rbf_fin_op);

instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo_default(
  cuvs::distance::detail::ops::l2_exp_distance_op, int64_t);

#undef instantiate_cuvs_distance_detail_pairwise_matrix_dispatch_by_algo
#undef instantiate_cuvs_distance_detail_pairwise_matrix_dispatch
