/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/operators.hpp>            // raft::sqrt
#include <raft/util/cuda_dev_essentials.cuh>  // DI

#include <cuda_fp16.h>

namespace cuvs::distance::detail::ops {

/** @brief The correlation distance
 *
 * It computes the following equation:
 *
 * d(x, y) = ((x - mean(x)) ⋅ (y - mean(y)))
 *           /
 *           (|| x - mean(x) ||_2 || y - mean(y) ||_2)
 */
template <typename DataType, typename AccType, typename IdxType>
struct correlation_distance_op {
  using DataT = DataType;
  using AccT  = AccType;
  using IdxT  = IdxType;

  const AccT* x2n;
  const AccT* y2n;
  IdxT m;
  IdxT n;
  IdxT k;

  correlation_distance_op(
    bool is_row_major, const AccT* x2n_, const AccT* y2n_, IdxT m_, IdxT n_, IdxT k_) noexcept
    : x2n(x2n_), y2n(y2n_), m(m_), n(n_), k(k_)
  {
    // The distance op is typically created before the row-major/col-major
    // swapping has been done. So we do it here.
    if (!is_row_major) {
      std::swap<const AccT*>(x2n, y2n);
      std::swap(m, n);
    }
  }

  // Load norms of input data
  static constexpr bool use_norms = true;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool expensive_inner_loop = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy>
  static constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize + (2 * (Policy::Mblk + Policy::Nblk) * sizeof(AccT));
  }

  DI void core(AccT& acc, DataT& x, DataT& y) const
  {
    acc += raft::to_float(x) * raft::to_float(y);
  };

  template <typename Policy>
  DI void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 AccT* regxn,
                 AccT* regyn,
                 IdxT gridStrideX,
                 IdxT gridStrideY) const
  {
    // Note how we can sneakily get a pointer to shared memory here, to store
    // more data. If the implementation of PairwiseDistanceMatKernel ever
    // changes, this will be where we find the bugs.
    extern __shared__ char smem[];

    AccT regx2n[Policy::AccRowsPerTh], regy2n[Policy::AccColsPerTh];

    AccT* sx2Norm = (AccT*)(&smem[Policy::SmemSize + (Policy::Mblk + Policy::Nblk) * sizeof(AccT)]);
    AccT* sy2Norm = (&sx2Norm[Policy::Mblk]);

    // Load x & y norms required by this threadblock in shmem buffer
    if (gridStrideX == blockIdx.x * Policy::Nblk) {
      for (int i = threadIdx.x; i < Policy::Mblk; i += Policy::Nthreads) {
        auto idx   = gridStrideY + i;
        sx2Norm[i] = idx < m ? raft::to_float(x2n[idx]) : 0;
      }
    }

    for (int i = threadIdx.x; i < Policy::Nblk; i += Policy::Nthreads) {
      auto idx   = gridStrideX + i;
      sy2Norm[i] = idx < n ? raft::to_float(y2n[idx]) : 0;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
      regx2n[i] = sx2Norm[i * Policy::AccThRows + (threadIdx.x / Policy::AccThCols)];
    }
#pragma unroll
    for (int i = 0; i < Policy::AccColsPerTh; ++i) {
      regy2n[i] = sy2Norm[i * Policy::AccThCols + (threadIdx.x % Policy::AccThCols)];
    }

#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < Policy::AccColsPerTh; ++j) {
        auto numer   = k * acc[i][j] - (regxn[i] * regyn[j]);
        auto Q_denom = k * regx2n[i] - (regxn[i] * regxn[i]);
        auto R_denom = k * regy2n[j] - (regyn[j] * regyn[j]);

        acc[i][j] = 1 - (numer / raft::sqrt(Q_denom * R_denom));
      }
    }
  }
};

}  // namespace cuvs::distance::detail::ops
