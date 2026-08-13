/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/limits>
#include <cuda_fp16.h>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/kvp.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <cuvs/distance/distance.hpp>
using cuvs::distance::DistanceType;

namespace cuvs::neighbors {

template <typename T>
_RAFT_HOST_DEVICE T max_val()
{
  if constexpr (std::is_same<T, half>::value) {
    return CUDART_MAX_NORMAL_FP16;
  } else {
    return cuda::std::numeric_limits<T>::max();
  }
}

template <typename T>
_RAFT_HOST_DEVICE T min_val()
{
  if constexpr (std::is_same<T, half>::value) {
    return CUDART_MIN_DENORM_FP16;
  } else {
    return cuda::std::numeric_limits<T>::min();
  }
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
__device__ AccT l2_distance(const DataT* v1, const DataT* v2, IdxT K)
{
  AccT th_dist = AccT(0.0);
  for (IdxT k = 0; k < K; k++) {
    AccT diff = AccT(v1[k]) - AccT(v2[k]);
    th_dist += (diff * diff);
  }
  return th_dist;
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
__device__ AccT cosine_distance(const DataT* v1, const DataT* v2, IdxT K)
{
  AccT v1_norm = AccT(0.0);
  AccT v2_norm = AccT(0.0);
  AccT v1v2    = AccT(0.0);

  for (IdxT k = 0; k < K; k++) {
    v1_norm += (AccT(v1[k]) * AccT(v1[k]));
    v2_norm += (AccT(v2[k]) * AccT(v2[k]));
    v1v2 += (AccT(v1[k]) * AccT(v2[k]));
  }

  v1_norm = raft::sqrt(v1_norm);
  v2_norm = raft::sqrt(v2_norm);
  return AccT(1.0) - (v1v2 / (v1_norm * v2_norm));
}

// This is a naive implementation of 1-NN computation
template <typename DataT, typename AccT, typename OutT, typename IdxT>
RAFT_KERNEL ref_nn_kernel(
  OutT* out, const DataT* A, const DataT* B, IdxT M, IdxT N, IdxT K, bool sqrt, DistanceType metric)
{
  IdxT tid = threadIdx.x + blockIdx.x * IdxT(blockDim.x);

  for (IdxT m = tid; m < M; m += (blockDim.x * gridDim.x)) {
    IdxT min_index = N + 1;
    AccT min_dist  = max_val<AccT>();

    for (IdxT n = 0; n < N; n++) {
      AccT dist;
      if (metric == DistanceType::L2SqrtExpanded || metric == DistanceType::L2Expanded) {
        dist = l2_distance<DataT, AccT, OutT, IdxT>(&A[m * K], &B[n * K], K);
      } else if (metric == DistanceType::CosineExpanded) {
        dist = cosine_distance<DataT, AccT, OutT, IdxT>(&A[m * K], &B[n * K], K);
      }
      if (dist < min_dist) {
        min_dist  = dist;
        min_index = n;
      }
    }

    if constexpr (std::is_fundamental<OutT>::value) {
      static_assert(std::is_same<OutT, AccT>::value, "OutT and AccT are not same type");
      out[m] = AccT(min_dist);
    } else {
      // output is a raft::KeyValuePair
      static_assert(std::is_same<OutT, raft::KeyValuePair<IdxT, AccT>>::value,
                    "OutT is not raft::KeyValuePair<> type");
      out[m].key = IdxT(min_index);
      if (sqrt) {
        out[m].value = raft::sqrt(AccT(min_dist));
      } else {
        out[m].value = AccT(min_dist);
      }
    }
  }
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
void ref_nn(OutT* out,
            const DataT* A,
            const DataT* B,
            IdxT m,
            IdxT n,
            IdxT k,
            bool sqrt,
            DistanceType metric,
            cudaStream_t stream)
{
  ref_nn_kernel<DataT, AccT, OutT, IdxT>
    <<<(m + 127) / 128, 128, 0, stream>>>(out, A, B, m, n, k, sqrt, metric);

  RAFT_CUDA_TRY(cudaGetLastError());
  return;
}

// Structure to track comparison failures
class ComparisonSummary {
 public:
  double max_diff;          // Maximum difference found
  uint64_t max_diff_index;  // where does the maximum difference occur
  double max_diff_a;        // What was the `a` value at max difference
  double max_diff_b;        // What was the `b` value at max difference
  double acc_diff;          // sum of all the diffs
  uint64_t n;               // How many items are compared
  uint64_t n_misses;        // How many were wrong
  int mutex;                // Simple mutex lock for thread synchronization

  void init()
  {
    max_diff       = 0.0;
    max_diff_index = 0;
    max_diff_a     = 0.0;
    max_diff_b     = 0.0;
    acc_diff       = 0.0;
    n              = 0;
    n_misses       = 0;
  }

  void update(double diff, uint64_t index, double a_val, double b_val, bool missed)
  {
    if (max_diff < diff) {
      max_diff       = diff;
      max_diff_index = index;
      max_diff_a     = a_val;
      max_diff_b     = b_val;
    }
    acc_diff += diff;
    n++;
    n_misses = missed ? n_misses + 1 : n_misses;
  }

  friend std::ostream& operator<<(std::ostream& os, const ComparisonSummary& summary)
  {
    if (summary.max_diff > 0.0) {
      os << "Total compared " << summary.n << std::endl;
      os << "Total missed " << summary.n_misses << std::endl;
      os << "Average diff: " << summary.acc_diff / summary.n << std::endl;
      os << "max_diff: " << summary.max_diff << " (" << summary.max_diff_a << " - "
         << summary.max_diff_b << ")" << std::endl;
      os << "max_diff_index: " << summary.max_diff_index << std::endl;
    }
    return os;
  }
};

template <typename OutT, typename IdxT>
void vector_compare(
  raft::resources const& handle, const OutT* a, const OutT* b, IdxT n, ComparisonSummary& summary)
{
  auto a_h = raft::make_host_vector<OutT, IdxT>(n);
  auto b_h = raft::make_host_vector<OutT, IdxT>(n);

  raft::copy(a_h.data_handle(), a, n, raft::resource::get_cuda_stream(handle));
  raft::copy(b_h.data_handle(), b, n, raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle, raft::resource::get_cuda_stream(handle));

  summary.init();

  for (IdxT i = 0; i < n; i++) {
    double diff, a_val, b_val;
    bool missed = false;
    if constexpr (std::is_fundamental_v<OutT> || std::is_same_v<OutT, half>) {
      // OutT is float, half or an integer type
      a_val = double(a_h(i));
      b_val = double(b_h(i));
    } else {
      // OutT is a raft::KeyValuePair
      a_val = double(a_h(i).value);
      b_val = double(b_h(i).value);

      missed = a_h(i).key != b_h(i).key;
    }

    diff = std::abs(a_val - b_val);
    summary.update(diff, i, a_val, b_val, missed);
  }
}

}  // namespace cuvs::neighbors
