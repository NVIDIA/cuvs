/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../distance_ops/l2_exp.cuh"     // ops::l2_exp_distance_op
#include "../pairwise_distance_base.cuh"  // PairwiseDistances
#include "cutlass_base.cuh"
#include "simt_kernel.cuh"
#include <raft/core/kvp.hpp>             // raft::KeyValuePair
#include <raft/core/operators.hpp>       // raft::identity_op
#include <raft/linalg/contractions.cuh>  // Policy
#include <raft/util/arch.cuh>            // raft::util::arch::SM_*
#include <raft/util/cuda_utils.cuh>      // raft::ceildiv, raft::shfl
#include <raft/util/device_atomics.cuh>

#include <cstddef>  // size_t
#include <limits>   // std::numeric_limits

namespace cuvs {
namespace distance {

namespace detail {

template <typename LabelT, typename DataT>
struct KVPMinReduceImpl {
  typedef raft::KeyValuePair<LabelT, DataT> KVP;
  DI KVP operator()(LabelT rit, const KVP& a, const KVP& b) { return b.value < a.value ? b : a; }
  DI KVP operator()(const KVP& a, const KVP& b) { return b.value < a.value ? b : a; }

};  // KVPMinReduce

/** Writes fused 1-NN results to separate idx/dist arrays (dist may be null). */
template <typename LabelT, typename DataT>
struct MinAndDistanceReduceOpImpl {
  typedef typename raft::KeyValuePair<LabelT, DataT> KVP;

  LabelT* out_idx{nullptr};
  DataT* out_dist{nullptr};
  /** When set, CUTLASS/SIMT global merge writes here instead of SoA (caller unpacks). */
  KVP* out_kvp{nullptr};

  DI void merge(LabelT rid, const KVP& other) const
  {
    if (out_kvp != nullptr) {
      if (other.value < out_kvp[rid].value) { out_kvp[rid] = other; }
    } else if (out_dist != nullptr) {
      if (other.value < out_dist[rid]) {
        out_dist[rid] = other.value;
        if (out_idx != nullptr) { out_idx[rid] = other.key; }
      }
    } else if (out_idx != nullptr) {
      // Idx-only output: dist must still be tracked for multi-tile merge; caller must provide
      // out_dist or use a single-pass backend (cuTile). KMeans always passes both buffers.
      out_idx[rid] = other.key;
    }
  }

  DI void operator()(LabelT rid, KVP* out, const KVP& other) const
  {
    if (out != nullptr && other.value < out->value) {
      out->key   = other.key;
      out->value = other.value;
    }
  }

  DI void operator()(LabelT rid, volatile KVP* out, const KVP& other) const
  {
    if (out != nullptr && other.value < out->value) {
      out->key   = other.key;
      out->value = other.value;
    }
  }

  DI void operator()(LabelT rid, DataT* out, const KVP& other) const
  {
    if (out != nullptr && other.value < *out) { *out = other.value; }
  }

  DI void operator()(LabelT rid, volatile DataT* out, const KVP& other) const
  {
    if (out != nullptr && other.value < *out) { *out = other.value; }
  }

  DI void operator()(LabelT rid, DataT* out, const DataT& other) const
  {
    if (out != nullptr && other < *out) { *out = other; }
  }

  DI void operator()(LabelT rid, volatile DataT* out, const DataT& other) const
  {
    if (out != nullptr && other < *out) { *out = other; }
  }

  DI void init(DataT* out, DataT maxVal) const
  {
    if (out != nullptr) { *out = maxVal; }
  }

  DI void init(KVP* out, DataT maxVal) const
  {
    out->value = maxVal;
    out->key   = LabelT(0);
  }

  DI void init_key(DataT& /*out*/, LabelT /*idx*/) const {}

  DI void init_key(KVP& out, LabelT idx) const { out.key = idx; }

  DI DataT get_value(KVP& out) const { return out.value; }

  DI DataT get_value(DataT& out) const { return out; }
};

template <typename LabelT, typename DataT>
struct MinReduceOpImpl {
  typedef typename raft::KeyValuePair<LabelT, DataT> KVP;
  DI void operator()(LabelT rid, DataT* out, const KVP& other)
  {
    if (other.value < *out) { *out = other.value; }
  }

  DI void init(DataT* out, DataT maxVal) { *out = maxVal; }
};

template <typename IdxT, typename DataT>
RAFT_KERNEL initFused1nnOutputKernel(IdxT* nearest_idx, DataT* nearest_dist, IdxT m, DataT maxVal)
{
  IdxT tid = IdxT(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < m) {
    if (nearest_idx != nullptr) { nearest_idx[tid] = IdxT(0); }
    if (nearest_dist != nullptr) { nearest_dist[tid] = maxVal; }
  }
}

template <typename IdxT, typename DataT>
void initFused1nnOutput(
  IdxT* nearest_idx, DataT* nearest_dist, IdxT m, DataT maxVal, cudaStream_t stream)
{
  if (nearest_idx == nullptr && nearest_dist == nullptr) { return; }
  auto blks = raft::ceildiv<IdxT>(m, 256);
  initFused1nnOutputKernel<IdxT, DataT>
    <<<blks, 256, 0, stream>>>(nearest_idx, nearest_dist, m, maxVal);
}

template <typename IdxT, typename DataT>
RAFT_KERNEL unpackFused1nnKvpToSoaKernel(IdxT* nearest_idx,
                                         DataT* nearest_dist,
                                         const raft::KeyValuePair<IdxT, DataT>* kvp,
                                         IdxT n)
{
  IdxT i = IdxT(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    if (nearest_idx != nullptr) { nearest_idx[i] = kvp[i].key; }
    if (nearest_dist != nullptr) { nearest_dist[i] = kvp[i].value; }
  }
}

template <typename IdxT, typename DataT>
void unpackFused1nnKvpToSoa(IdxT* nearest_idx,
                            DataT* nearest_dist,
                            const raft::KeyValuePair<IdxT, DataT>* kvp,
                            IdxT m,
                            cudaStream_t stream)
{
  if (nearest_idx == nullptr && nearest_dist == nullptr) { return; }
  auto blks = raft::ceildiv<IdxT>(m, 256);
  unpackFused1nnKvpToSoaKernel<IdxT, DataT>
    <<<blks, 256, 0, stream>>>(nearest_idx, nearest_dist, kvp, m);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT>
RAFT_KERNEL initKernel(OutT* min, IdxT m, DataT maxVal, ReduceOpT redOp)
{
  auto tid = IdxT(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < m) { redOp.init(min + tid, maxVal); }
}

template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT>
void initialize(OutT* min, IdxT m, DataT maxVal, ReduceOpT redOp, cudaStream_t stream)
{
  auto blks = raft::ceildiv<IdxT>(m, 256);
  initKernel<DataT, OutT, IdxT, ReduceOpT><<<blks, 256, 0, stream>>>(min, m, maxVal, redOp);
}

// cg::reduce functor for FusedDistanceNN used in its cutlass version
// to output the min distance value & key(loc id).
template <typename AccType, typename Index>
struct kvp_cg_min_reduce_op {
  typedef typename raft::KeyValuePair<Index, AccType> KVP;

  __host__ __device__ kvp_cg_min_reduce_op() noexcept {};

  using AccTypeT = AccType;
  using IndexT   = Index;
  __host__ __device__ KVP operator()(KVP a, KVP b) const { return a.value < b.value ? a : b; }

  __host__ __device__ AccType operator()(AccType a, AccType b) const { return min(a, b); }

  __host__ __device__ bool isAmin(AccType a, AccType b) const { return a < b ? true : false; }
};

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
