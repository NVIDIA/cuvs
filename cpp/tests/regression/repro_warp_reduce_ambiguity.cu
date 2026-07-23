/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal reproducer: warpReduce / raft::add_op ADL ambiguity (CUDA 13.2+ / CCCL 3.4+)
 *
 * COMPILE TO SEE THE ERROR (without raft fix):
 *
 *   # 1. Strip the explicit add_op overload from raft/util/reduction.cuh
 *   #    (the lines that add:
 *   #       template <typename T>
 *   #       DI T warpReduce(T val, raft::add_op reduce_op) { ... })
 *   # 2. Then compile this file:
 *   nvcc -std=c++17 -arch=sm_121 repro_warp_reduce_ambiguity.cu \
 *        -I<raft>/cpp/include -I<cccl>/thrust \
 *        -I<cccl>/libcudacxx/include -I<cccl>/cub \
 *        -I<rmm>/cpp/include -o /dev/null
 *
 * EXPECTED ERROR (without fix):
 *
 *   error: more than one instance of function template "warpReduce" matches:
 *     function template "T raft::warpReduce(T, ReduceLambda)"
 *     function template "Tp cub::detail::scan::warpReduce(Tp, ScanOpT &)"
 *     argument types are: (int, raft::add_op)
 *
 * ROOT CAUSE:
 *   CCCL 3.4+ (CUDA 13.2+) introduced cub::detail::scan::warpReduce in
 *   kernel_scan_warpspeed.cuh.  When thrust::inclusive_scan is called with
 *   raft::add_op{} as the binary op, this kernel is instantiated with
 *   ScanOpT = raft::add_op.  Inside that instantiation the unqualified call
 *   warpReduce(input, scan_op) has two equally good candidates:
 *     - cub::detail::scan::warpReduce  (same namespace, normal lookup)
 *     - raft::warpReduce<T,ReduceLambda>  (found via ADL on raft::add_op)
 *   Neither is more specialized than the other -> ambiguous.
 *
 * FIX (raft, zbrad/raft@cu132 commit d1345188):
 *   Added explicit non-template overload:
 *     template <typename T>
 *     DI T warpReduce(T val, raft::add_op reduce_op)
 *   More specialized than the ReduceLambda template, wins over the CUB
 *   candidate, ambiguity gone.
 *
 * FIX (cuvs, commit 99b38018):
 *   ivf_pq_build.cuh changed from raft::add_op{} to thrust::plus<>{}.
 *   thrust::plus is in thrust::, so ADL no longer pulls in raft::warpReduce
 *   and cub::detail::scan::warpReduce is the only candidate.
 */

#include <raft/core/operators.hpp>
#include <raft/util/reduction.cuh>

namespace cub { namespace detail { namespace scan {

// Mirrors cub::detail::scan::warpReduce from kernel_scan_warpspeed.cuh (CCCL 3.4+)
template <typename Tp, typename ScanOpT>
__device__ Tp warpReduce(const Tp input, ScanOpT& scan_op)
{
  return input;
}

__device__ void trigger_ambiguity(int val)
{
  raft::add_op op;
  // Without the explicit raft::add_op overload this is ambiguous:
  //   raft::warpReduce<int, raft::add_op>          (ADL on raft::add_op)
  //   cub::detail::scan::warpReduce<int, raft::add_op>  (same namespace)
  (void)warpReduce(val, op);
}

}}} // namespace cub::detail::scan
