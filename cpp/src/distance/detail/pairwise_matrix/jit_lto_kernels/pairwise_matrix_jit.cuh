/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "pairwise_matrix_planner.hpp"

#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <distance/detail/distance_ops/all_ops.cuh>
#include <distance/detail/kernels/rbf_fin_op.cuh>
#include <distance/detail/pairwise_distance_base.cuh>
#include <distance/detail/pairwise_matrix/dispatch_layout.cuh>
#include <distance/detail/pairwise_matrix/params.cuh>
#include <raft/core/operators.hpp>
#include <raft/linalg/contractions.cuh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace cuvs::distance::detail {

template <typename>
inline constexpr bool pairwise_matrix_jit_always_false_v = false;

template <typename T>
constexpr auto get_pairwise_scalar_type_tag()
{
  if constexpr (std::is_same_v<T, float>) {
    return cuvs::neighbors::detail::tag_f{};
  } else if constexpr (std::is_same_v<T, double>) {
    return cuvs::neighbors::detail::tag_d{};
  } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __half>) {
    return cuvs::neighbors::detail::tag_h{};
  } else {
    static_assert(pairwise_matrix_jit_always_false_v<T>,
                  "Pairwise matrix JIT LTO does not have a scalar tag for this type");
  }
}

template <typename IdxT>
constexpr auto get_pairwise_index_type_tag()
{
  if constexpr (std::is_same_v<IdxT, int>) {
    return cuvs::neighbors::detail::tag_index_i32{};
  } else if constexpr (std::is_same_v<IdxT, int64_t>) {
    return cuvs::neighbors::detail::tag_index_i64{};
  } else {
    static_assert(pairwise_matrix_jit_always_false_v<IdxT>,
                  "Pairwise matrix JIT LTO does not have an index tag for this type");
  }
}

template <bool RowMajor>
using pairwise_layout_tag_t = std::conditional_t<RowMajor, tag_layout_row, tag_layout_col>;

template <typename FinOpT>
struct pairwise_fin_op_tag {
  static_assert(pairwise_matrix_jit_always_false_v<FinOpT>,
                "Pairwise matrix JIT LTO does not have a final-op tag for this type");
};

template <>
struct pairwise_fin_op_tag<raft::identity_op> {
  using type = tag_fin_op_identity;
};

template <typename OutT>
struct pairwise_fin_op_tag<cuvs::distance::kernels::rbf_fin_op<OutT>> {
  using type = tag_fin_op_rbf;
};

template <typename FinOpT>
using pairwise_fin_op_tag_t = typename pairwise_fin_op_tag<FinOpT>::type;

template <typename OpT>
struct pairwise_distance_op_tag {
  static_assert(pairwise_matrix_jit_always_false_v<OpT>,
                "Pairwise matrix JIT LTO does not have a distance-op tag for this type");
};

template <typename DataT, typename AccT, typename IdxT>
struct pairwise_distance_op_tag<ops::canberra_distance_op<DataT, AccT, IdxT>> {
  using type = tag_distance_canberra;
};

template <typename DataT, typename AccT, typename IdxT>
struct pairwise_distance_op_tag<ops::correlation_distance_op<DataT, AccT, IdxT>> {
  using type = tag_distance_correlation;
};

template <typename DataT, typename AccT, typename IdxT>
struct pairwise_distance_op_tag<ops::cosine_distance_op<DataT, AccT, IdxT>> {
  using type = tag_distance_cosine;
};

template <typename DataT, typename AccT, typename IdxT>
struct pairwise_distance_op_tag<ops::hamming_distance_op<DataT, AccT, IdxT>> {
  using type = tag_distance_hamming_unexpanded;
};

template <typename DataT, typename AccT, typename IdxT>
struct pairwise_distance_op_tag<ops::hellinger_distance_op<DataT, AccT, IdxT>> {
  using type = tag_distance_hellinger_expanded;
};

template <typename DataT, typename AccT, typename IdxT>
struct pairwise_distance_op_tag<ops::jensen_shannon_distance_op<DataT, AccT, IdxT>> {
  using type = tag_distance_jensen_shannon;
};

template <typename DataT, typename AccT, typename IdxT>
struct pairwise_distance_op_tag<ops::kl_divergence_op<DataT, AccT, IdxT>> {
  using type = tag_distance_kl_divergence;
};

template <typename DataT, typename AccT, typename IdxT>
struct pairwise_distance_op_tag<ops::l1_distance_op<DataT, AccT, IdxT>> {
  using type = tag_distance_l1;
};

template <typename DataT, typename AccT, typename IdxT>
struct pairwise_distance_op_tag<ops::l2_exp_distance_op<DataT, AccT, IdxT>> {
  using type = tag_distance_l2_expanded;
};

template <typename DataT, typename AccT, typename IdxT>
struct pairwise_distance_op_tag<ops::l2_unexp_distance_op<DataT, AccT, IdxT>> {
  using type = tag_distance_l2_unexpanded;
};

template <typename DataT, typename AccT, typename IdxT>
struct pairwise_distance_op_tag<ops::l_inf_distance_op<DataT, AccT, IdxT>> {
  using type = tag_distance_l_inf;
};

template <typename DataT, typename AccT, typename IdxT>
struct pairwise_distance_op_tag<ops::lp_unexp_distance_op<DataT, AccT, IdxT>> {
  using type = tag_distance_lp_unexpanded;
};

template <typename DataT, typename AccT, typename IdxT>
struct pairwise_distance_op_tag<ops::russel_rao_distance_op<DataT, AccT, IdxT>> {
  using type = tag_distance_russel_rao;
};

template <typename OpT>
using pairwise_distance_op_tag_t = typename pairwise_distance_op_tag<OpT>::type;

template <typename OpT, typename IdxT, typename DataT, typename OutT, typename FinOpT>
using pairwise_matrix_jit_kernel_t = void(OpT, pairwise_matrix_params<IdxT, DataT, OutT, FinOpT>);

template <typename OpT, typename IdxT, typename DataT, typename OutT, typename FinOpT>
void pairwise_matrix_jit_dispatch(OpT distance_op,
                                  pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params,
                                  cudaStream_t stream)
{
  using AccT = typename OpT::AccT;

  int vec_len = determine_vec_len(params);

  auto launch = [&](auto row_major, auto vec_len_aligned) {
    constexpr int vec_len_op = OpT::expensive_inner_loop ? 1 : vec_len_aligned();
    constexpr int veclen     = std::min(vec_len_op, static_cast<int>(16 / sizeof(DataT)));

    using RowPolicy = typename raft::linalg::Policy4x4<DataT, veclen>::Policy;
    using ColPolicy = typename raft::linalg::Policy4x4<DataT, veclen>::ColPolicy;
    using Policy    = std::conditional_t<row_major(), RowPolicy, ColPolicy>;

    using DistanceTag = pairwise_distance_op_tag_t<OpT>;
    using DataTag     = decltype(get_pairwise_scalar_type_tag<DataT>());
    using AccTag      = decltype(get_pairwise_scalar_type_tag<AccT>());
    using OutTag      = decltype(get_pairwise_scalar_type_tag<OutT>());
    using IndexTag    = decltype(get_pairwise_index_type_tag<IdxT>());
    using FinOpTag    = pairwise_fin_op_tag_t<FinOpT>;
    using LayoutTag   = pairwise_layout_tag_t<row_major()>;

    PairwiseMatrixPlanner<DistanceTag,
                          DataTag,
                          AccTag,
                          OutTag,
                          IndexTag,
                          FinOpTag,
                          LayoutTag,
                          veclen>
      planner;
    planner.add_entrypoint();
    planner.add_compute_distance_function();
    planner.add_compute_distance_epilog_function();

    auto launcher = planner.get_launcher();

    dim3 block(Policy::Nthreads);
    int smem_size = OpT::template shared_mem_size<Policy>();
    dim3 grid =
      launchConfigGenerator<Policy>(params.m, params.n, smem_size, launcher->get_kernel());

    launcher->template dispatch<pairwise_matrix_jit_kernel_t<OpT, IdxT, DataT, OutT, FinOpT>>(
      stream, grid, block, smem_size, distance_op, params);
  };

  dispatch_layout(params.is_row_major, vec_len, launch);
}

}  // namespace cuvs::distance::detail
