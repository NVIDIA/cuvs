/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../distance_ops/l2_exp.cuh"     // ops::l2_exp_distance_op
#include "../pairwise_distance_base.cuh"  // PairwiseDistances
#include "cutlass_base.cuh"
#include "helper_structs.cuh"
#include "simt_kernel.cuh"
#include <raft/core/kvp.hpp>             // raft::KeyValuePair
#include <raft/core/operators.hpp>       // raft::identity_op
#include <raft/linalg/contractions.cuh>  // Policy
#include <raft/util/arch.cuh>            // raft::util::arch::SM_*
#include <raft/util/cuda_utils.cuh>      // raft::ceildiv, raft::shfl

#include <cstddef>  // size_t
#include <limits>   // std::numeric_limits

namespace cuvs {
namespace distance {

namespace detail {

template <typename DataT, typename IdxT, typename Policy, typename ReduceOpT, typename KVPReduceOpT>
void fusedL2NNImpl(IdxT* nearest_idx,
                   DataT* nearest_dist,
                   const DataT* x,
                   const DataT* y,
                   const DataT* xn,
                   const DataT* yn,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   int* workspace,
                   ReduceOpT redOp,
                   KVPReduceOpT pairRedOp,
                   bool sqrt,
                   bool initOutBuffer,
                   raft::KeyValuePair<IdxT, DataT>* cutlass_out,
                   cudaStream_t stream)
{
  typedef Policy P;

  dim3 blk(P::Nthreads);
  constexpr auto maxVal = std::numeric_limits<DataT>::max();
  typedef raft::KeyValuePair<IdxT, DataT> KVPair;

  if (initOutBuffer && cutlass_out == nullptr) {
    initFused1nnOutput(nearest_idx, nearest_dist, m, maxVal, stream);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  namespace arch = raft::util::arch;
  using AccT     = DataT;
  ops::l2_exp_distance_op<DataT, AccT, IdxT> distance_op{sqrt};

  raft::identity_op fin_op{};

  auto kernel = fusedDistanceNNkernel<DataT,
                                      KVPair,
                                      IdxT,
                                      P,
                                      ReduceOpT,
                                      KVPReduceOpT,
                                      decltype(distance_op),
                                      decltype(fin_op)>;

  void* kernel_ptr   = reinterpret_cast<void*>(kernel);
  auto runtime_arch  = arch::kernel_virtual_arch(kernel_ptr);
  auto cutlass_range = arch::SM_range(arch::SM_80(), arch::SM_future());

  if (cutlass_range.contains(runtime_arch)) {
    using L2Op                  = cuvs::distance::detail::ops::l2_exp_cutlass_op<DataT, DataT>;
    using kvp_cg_min_reduce_op_ = kvp_cg_min_reduce_op<DataT, IdxT>;
    kvp_cg_min_reduce_op_ cg_reduce_op;
    L2Op L2_dist_op(sqrt);

    IdxT lda, ldb, ldd;
    lda = k, ldb = k, ldd = n;

    cutlassFusedDistanceNN<DataT,
                           DataT,
                           KVPair,
                           IdxT,
                           P::Veclen,
                           kvp_cg_min_reduce_op_,
                           L2Op,
                           ReduceOpT,
                           KVPReduceOpT>(x,
                                         y,
                                         xn,
                                         yn,
                                         m,
                                         n,
                                         k,
                                         lda,
                                         ldb,
                                         ldd,
                                         cutlass_out,
                                         workspace,
                                         cg_reduce_op,
                                         L2_dist_op,
                                         redOp,
                                         pairRedOp,
                                         stream);
  } else {
    constexpr size_t shmemSize = P::SmemSize + ((P::Mblk + P::Nblk) * sizeof(DataT));
    dim3 grid                  = launchConfigGenerator<P>(m, n, shmemSize, kernel);

    kernel<<<grid, blk, shmemSize, stream>>>(
      cutlass_out, x, y, xn, yn, m, n, k, maxVal, workspace, redOp, pairRedOp, distance_op, fin_op);
    RAFT_CUDA_TRY(cudaGetLastError());
  }
}

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
