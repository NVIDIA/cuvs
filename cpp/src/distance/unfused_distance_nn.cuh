/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __UNFUSED_DISTANCE_NN_H
#define __UNFUSED_DISTANCE_NN_H

#pragma once

#include <cub/block/block_reduce.cuh>

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resources.hpp>

namespace cuvs {
namespace distance {

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
T min_val()
{
  if constexpr (std::is_same<T, half>::value) {
    return CUDART_MIN_DENORM_FP16;
  } else {
    return cuda::std::numeric_limits<T>::min();
  }
}

template <typename AccT, typename IdxT>
struct Reducer {
  typedef raft::KeyValuePair<IdxT, AccT> KVType;

  __device__ KVType operator()(const KVType& a, const KVType& b)
  {
    if ((a.value < b.value) || (a.value == b.value && a.key < b.key)) {
      return a;
    } else {
      return b;
    }
  }

  __device__ AccT operator()(const AccT& a, const AccT& b) { return a < b ? a : b; }
};

template <typename DataT, typename AccT, typename OutT, typename IdxT, int TPB, DistanceType metric>
__global__ void reduce_min_kernel(OutT* out,
                                  const AccT* z,
                                  const AccT* x_norm,
                                  const AccT* y_norm,
                                  IdxT m,
                                  IdxT n,
                                  bool is_sqrt,
                                  bool initOutBuffer)
{
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  IdxT row = blockIdx.x;

  AccT x_norm_row = x_norm[row];

  typedef raft::KeyValuePair<IdxT, AccT> KVType;
  KVType thread_min;

  thread_min.value = max_val<AccT>();
  thread_min.key   = max_val<IdxT>();

  for (IdxT col = threadIdx.x; col < n; col += TPB) {
    AccT dist = 0.0;

    if constexpr (metric == DistanceType::L2SqrtExpanded || metric == DistanceType::L2Expanded) {
      dist = x_norm_row + y_norm[col] - AccT(2) * z[row * n + col];
      // GEMM round-off can produce slightly negative expanded distances; clamp to zero.
      dist = (dist > AccT(0)) ? dist : AccT(0);
    } else if constexpr (metric == DistanceType::CosineExpanded) {
      // Guard against zero-norm vectors to avoid inf/NaN from division by zero.
      AccT denom = x_norm_row * y_norm[col];
      denom      = (denom > AccT(0)) ? denom : AccT(1);
      dist       = AccT(1.0) - (z[row * n + col] / denom);
    }
    if (dist < thread_min.value) {
      thread_min.value = dist;
      thread_min.key   = col;
    }
  }

  typedef cub::BlockReduce<KVType, TPB> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  auto block_result = BlockReduceT(temp_storage).Reduce(thread_min, Reducer<AccT, IdxT>{});

  if (threadIdx.x == 0) {
    if (is_sqrt) {
      block_result.value = raft::sqrt(block_result.value > AccT(0) ? block_result.value : AccT(0));
    }
    if constexpr (std::is_same_v<OutT, KVType>) {
      if (initOutBuffer == true) {
        out[row] = block_result;
      } else {
        out[row] = block_result.value < out[row].value ? block_result : out[row];
      }
    } else {
      if (initOutBuffer == true) {
        out[row] = block_result.value;
      } else {
        out[row] = block_result.value < out[row] ? block_result.value : out[row];
      }
    }
  }
}

/**
 * @brief Reduces pairwise distances to find the minimum distance per row.
 *
 * This function launches a kernel that computes the final distance values from
 * the GEMM output and norms, then reduces across columns to find the minimum
 * distance (and optionally the index) for each row. Distance computation and
 * reduction is fused together in a single kernel that is why we can not use
 * RAFT reduction here.
 *
 * @tparam DataT   Input data type
 * @tparam AccT    Accumulator/distance type
 * @tparam OutT    Output type (either AccT for distances only, or KeyValuePair for index+distance)
 * @tparam IdxT    Index type
 * @tparam metric  Distance metric type (L2Expanded, L2SqrtExpanded, or CosineExpanded)
 *
 * @param[out] out           Output array containing minimum distances (and optionally indices)
 *                           per row. Length = `m`. (on device)
 * @param[in]  z             GEMM output matrix (x * y^T). Dim = `m x n`. (on device)
 * @param[in]  x_norm        Norms of rows in x. Length = `m`. (on device)
 * @param[in]  y_norm        Norms of rows in y. Length = `n`. (on device)
 * @param[in]  m             Number of rows in x (and output)
 * @param[in]  n             Number of rows in y (columns to reduce over)
 * @param[in]  stream        CUDA stream for kernel launch
 * @param[in]  is_sqrt       Whether to apply square root to the final distance
 * @param[in]  initOutBuffer Whether to initialize the output buffer or merge with existing values
 */
template <typename DataT, typename AccT, typename OutT, typename IdxT, DistanceType metric>
void reduce_min(OutT* out,
                const AccT* z,
                const AccT* x_norm,
                const AccT* y_norm,
                IdxT m,
                IdxT n,
                cudaStream_t stream,
                bool is_sqrt,
                bool initOutBuffer)
{
  const int TPB = 128;

  int blocks = m;
  reduce_min_kernel<DataT, AccT, OutT, IdxT, TPB, metric>
    <<<blocks, TPB, 0, stream>>>(out, z, x_norm, y_norm, m, n, is_sqrt, initOutBuffer);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
void pairwise_distance_gemm(raft::resources const& handle,
                            AccT* z,
                            const DataT* x,
                            const DataT* y,
                            IdxT M,
                            IdxT N,
                            IdxT K,
                            const AccT* x_norm,
                            const AccT* y_norm,
                            cudaStream_t stream)
{
  cudaDataType_t xyType, zType;
  cublasComputeType_t computeType;

  if constexpr (std::is_same_v<DataT, int8_t>) {
    xyType = CUDA_R_8I;
    if constexpr (std::is_same_v<AccT, int32_t>) {
      zType       = CUDA_R_32I;
      computeType = CUBLAS_COMPUTE_32I;
    } else if constexpr (std::is_same_v<AccT, float>) {
      zType       = CUDA_R_32F;
      computeType = CUBLAS_COMPUTE_32F;
    }
  } else if constexpr (std::is_same_v<DataT, half>) {
    xyType = CUDA_R_16F;
    if constexpr (std::is_same_v<AccT, half>) {
      zType       = CUDA_R_16F;
      computeType = CUBLAS_COMPUTE_16F;
      // Alternative: CUBLAS_COMPUTE_32F can be used for higher precision,
      // but requires changing alpha and beta to float type instead of half.
      // See: https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex
    } else if constexpr (std::is_same_v<AccT, float>) {
      zType       = CUDA_R_32F;
      computeType = CUBLAS_COMPUTE_32F;
    }
  } else if constexpr (std::is_same_v<DataT, float>) {
    xyType      = CUDA_R_32F;
    zType       = CUDA_R_32F;
    computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    // Note: Alternative compute types that can be used include:
    // CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_FAST_16F,
    // CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_COMPUTE_32F_EMULATED_16BFX9
  } else if constexpr (std::is_same_v<DataT, double>) {
    xyType      = CUDA_R_64F;
    zType       = CUDA_R_64F;
    computeType = CUBLAS_COMPUTE_64F;
  }

  const AccT alpha = static_cast<AccT>(1);
  const AccT beta  = static_cast<AccT>(0);

  auto cublas_h = raft::resource::get_cublas_handle(handle);
  RAFT_CUBLAS_TRY(cublasSetStream(cublas_h, stream));

  RAFT_CUBLAS_TRY(cublasGemmEx(cublas_h,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               N,  // Dimensions (swapped due to row/col-major difference)
                               M,
                               K,
                               reinterpret_cast<const void*>(&alpha),
                               y,
                               xyType,
                               K,
                               x,
                               xyType,
                               K,
                               reinterpret_cast<const void*>(&beta),
                               z,
                               zType,
                               N,
                               computeType,         // Computation type
                               CUBLAS_GEMM_DEFAULT  // Algorithm selection
                               ));
}

/**
 * \ingroup unfused_distance_nn
 * @{
 */
/**
 * @brief Unfused distance and 1-nearest-neighbor computation in a single call.
 *
 * Unlike the fused implementation, GEMM and reduction kernel are called separately.
 * This code path exists because the fused path is sometimes not the optimal choice.
 *
 * @tparam DataT      data type
 * @tparam OutT       output type to either store 1-NN indices and their minimum
 *                    distances or store only the min distances.
 * @tparam IdxT       indexing arithmetic type
 *
 * @param[out] min           will contain the reduced output (Length = `m`)
 *                           (on device)
 * @param[in]  x             first matrix. Row major. Dim = `m x k`.
 *                           (on device).
 * @param[in]  y             second matrix. Row major. Dim = `n x k`.
 *                           (on device).
 * @param[in]  xn            L2 squared norm of `x`. Length = `m`. (on device).
 * @param[in]  yn            L2 squared norm of `y`. Length = `n`. (on device)
 * @param[in]  m             gemm m
 * @param[in]  n             gemm n
 * @param[in]  k             gemm k
 * @param[in]  workspace     temp workspace. Size = sizeof(DataT) * m. (on device)
 * @param[in]  is_sqrt       Whether the output `min` should contain L2-sqrt
 * @param[in]  initOutBuffer whether to initialize the output buffer before the
 *                           main kernel launch
 * @param[in]  isRowMajor    whether the input/output is row or column major.
 * @param[in]  metric        Distance metric to be used (supports L2, cosine)
 * @param[in]  metric_arg    power argument for distances like Minkowski (not supported for now)
 * @param[in]  stream        cuda stream
 */
template <typename DataT, typename AccT, typename OutT, typename IdxT>
void unfusedDistanceNNMinReduce(raft::resources const& handle,
                                OutT* min,
                                const DataT* x,
                                const DataT* y,
                                const AccT* xn,
                                const AccT* yn,
                                IdxT m,
                                IdxT n,
                                IdxT k,
                                void* workspace,
                                bool is_sqrt,
                                bool initOutBuffer,
                                bool isRowMajor,
                                DistanceType metric,
                                float metric_arg,
                                cudaStream_t stream)
{
  ASSERT(isRowMajor, "unfusedDistanceNN only supports row major inputs");

  ASSERT(m > 0 && n > 0 && k > 0, "unfusedDistanceNN requires non-zero m, n, and k");
  pairwise_distance_gemm<DataT, AccT, OutT, IdxT>(
    handle, (AccT*)workspace, x, y, m, n, k, xn, yn, stream);

  ASSERT((metric == DistanceType::CosineExpanded) || (metric == DistanceType::L2Expanded) ||
           (metric == DistanceType::L2SqrtExpanded),
         "Only cosine and L2 distance types are supported");

  if (metric == DistanceType::L2Expanded) {
    reduce_min<DataT, AccT, OutT, IdxT, DistanceType::L2Expanded>(
      min, (AccT*)workspace, xn, yn, m, n, stream, is_sqrt, initOutBuffer);
  } else if (metric == DistanceType::L2SqrtExpanded) {
    reduce_min<DataT, AccT, OutT, IdxT, DistanceType::L2SqrtExpanded>(
      min, (AccT*)workspace, xn, yn, m, n, stream, is_sqrt, initOutBuffer);
  } else if (metric == DistanceType::CosineExpanded) {
    reduce_min<DataT, AccT, OutT, IdxT, DistanceType::CosineExpanded>(
      min, (AccT*)workspace, xn, yn, m, n, stream, is_sqrt, initOutBuffer);
  }
}

/** @} */

}  // namespace distance
}  // namespace cuvs

#endif
