/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gmm_kernels.cuh"

#include <cuvs/cluster/gmm.hpp>
#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/random/sample_without_replacement.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuvs::cluster::gmm::detail {

constexpr int E_STEP_BLOCK          = 64;
constexpr int E_STEP_LARGE64_TILE   = 64;
constexpr int E_STEP_THREAD64_BLOCK = 512;
constexpr int NORMALIZE_BLOCK       = 32;
constexpr int REDUCE_BLOCK          = 256;
constexpr size_t DEFAULT_SMEM_LIMIT = 48 * 1024;

inline size_t upper_tri_size(size_t d) { return (d * (d + 1)) / 2; }

inline void cublas_check(cublasStatus_t status, const char* what)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

inline void cusolver_check(cusolverStatus_t status, const char* what)
{
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed with cuSOLVER status " +
                             std::to_string(static_cast<int>(status)));
  }
}

inline const char* precision_error_message()
{
  return "Fitting the mixture model failed because some components have ill-defined empirical "
         "covariance (for instance caused by singleton or collapsed samples). Try to decrease the "
         "number of components, increase reg_covar, or scale the input data.";
}

// ----- cuBLAS gemm / gemv wrappers -----
template <typename T>
cublasStatus_t cublas_gemm(cublasHandle_t h,
                           cublasOperation_t ta,
                           cublasOperation_t tb,
                           int m,
                           int n,
                           int k,
                           const T* alpha,
                           const T* A,
                           int lda,
                           const T* B,
                           int ldb,
                           const T* beta,
                           T* C,
                           int ldc);
template <>
inline cublasStatus_t cublas_gemm<float>(cublasHandle_t h,
                                         cublasOperation_t ta,
                                         cublasOperation_t tb,
                                         int m,
                                         int n,
                                         int k,
                                         const float* alpha,
                                         const float* A,
                                         int lda,
                                         const float* B,
                                         int ldb,
                                         const float* beta,
                                         float* C,
                                         int ldc)
{
  return cublasSgemm(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
template <>
inline cublasStatus_t cublas_gemm<double>(cublasHandle_t h,
                                          cublasOperation_t ta,
                                          cublasOperation_t tb,
                                          int m,
                                          int n,
                                          int k,
                                          const double* alpha,
                                          const double* A,
                                          int lda,
                                          const double* B,
                                          int ldb,
                                          const double* beta,
                                          double* C,
                                          int ldc)
{
  return cublasDgemm(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename T>
cublasStatus_t cublas_gemv(cublasHandle_t h,
                           cublasOperation_t trans,
                           int m,
                           int n,
                           const T* alpha,
                           const T* A,
                           int lda,
                           const T* x,
                           int incx,
                           const T* beta,
                           T* y,
                           int incy);
template <>
inline cublasStatus_t cublas_gemv<float>(cublasHandle_t h,
                                         cublasOperation_t trans,
                                         int m,
                                         int n,
                                         const float* alpha,
                                         const float* A,
                                         int lda,
                                         const float* x,
                                         int incx,
                                         const float* beta,
                                         float* y,
                                         int incy)
{
  return cublasSgemv(h, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
template <>
inline cublasStatus_t cublas_gemv<double>(cublasHandle_t h,
                                          cublasOperation_t trans,
                                          int m,
                                          int n,
                                          const double* alpha,
                                          const double* A,
                                          int lda,
                                          const double* x,
                                          int incx,
                                          const double* beta,
                                          double* y,
                                          int incy)
{
  return cublasDgemv(h, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

// ----- cuSOLVER potrf / cuBLAS trsm wrappers -----
template <typename T>
cusolverStatus_t potrf_bufsize(
  cusolverDnHandle_t h, cublasFillMode_t uplo, int nn, T* A, int lda, int* lwork);
template <>
inline cusolverStatus_t potrf_bufsize<float>(
  cusolverDnHandle_t h, cublasFillMode_t uplo, int nn, float* A, int lda, int* lwork)
{
  return cusolverDnSpotrf_bufferSize(h, uplo, nn, A, lda, lwork);
}
template <>
inline cusolverStatus_t potrf_bufsize<double>(
  cusolverDnHandle_t h, cublasFillMode_t uplo, int nn, double* A, int lda, int* lwork)
{
  return cusolverDnDpotrf_bufferSize(h, uplo, nn, A, lda, lwork);
}

template <typename T>
cusolverStatus_t potrf(cusolverDnHandle_t h,
                       cublasFillMode_t uplo,
                       int nn,
                       T* A,
                       int lda,
                       T* work,
                       int lwork,
                       int* devInfo);
template <>
inline cusolverStatus_t potrf<float>(cusolverDnHandle_t h,
                                     cublasFillMode_t uplo,
                                     int nn,
                                     float* A,
                                     int lda,
                                     float* work,
                                     int lwork,
                                     int* devInfo)
{
  return cusolverDnSpotrf(h, uplo, nn, A, lda, work, lwork, devInfo);
}
template <>
inline cusolverStatus_t potrf<double>(cusolverDnHandle_t h,
                                      cublasFillMode_t uplo,
                                      int nn,
                                      double* A,
                                      int lda,
                                      double* work,
                                      int lwork,
                                      int* devInfo)
{
  return cusolverDnDpotrf(h, uplo, nn, A, lda, work, lwork, devInfo);
}

template <typename T>
cublasStatus_t cublas_trsm(cublasHandle_t h,
                           cublasSideMode_t side,
                           cublasFillMode_t uplo,
                           cublasOperation_t trans,
                           cublasDiagType_t diag,
                           int m,
                           int n,
                           const T* alpha,
                           const T* A,
                           int lda,
                           T* B,
                           int ldb);
template <>
inline cublasStatus_t cublas_trsm<float>(cublasHandle_t h,
                                         cublasSideMode_t side,
                                         cublasFillMode_t uplo,
                                         cublasOperation_t trans,
                                         cublasDiagType_t diag,
                                         int m,
                                         int n,
                                         const float* alpha,
                                         const float* A,
                                         int lda,
                                         float* B,
                                         int ldb)
{
  return cublasStrsm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}
template <>
inline cublasStatus_t cublas_trsm<double>(cublasHandle_t h,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          int m,
                                          int n,
                                          const double* alpha,
                                          const double* A,
                                          int lda,
                                          double* B,
                                          int ldb)
{
  return cublasDtrsm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <typename T>
cusolverStatus_t potrf_batched(cusolverDnHandle_t h,
                               cublasFillMode_t uplo,
                               int nn,
                               T* Aarray[],
                               int lda,
                               int* infoArray,
                               int batch);
template <>
inline cusolverStatus_t potrf_batched<float>(cusolverDnHandle_t h,
                                             cublasFillMode_t uplo,
                                             int nn,
                                             float* Aarray[],
                                             int lda,
                                             int* infoArray,
                                             int batch)
{
  return cusolverDnSpotrfBatched(h, uplo, nn, Aarray, lda, infoArray, batch);
}
template <>
inline cusolverStatus_t potrf_batched<double>(cusolverDnHandle_t h,
                                              cublasFillMode_t uplo,
                                              int nn,
                                              double* Aarray[],
                                              int lda,
                                              int* infoArray,
                                              int batch)
{
  return cusolverDnDpotrfBatched(h, uplo, nn, Aarray, lda, infoArray, batch);
}

template <typename T>
cublasStatus_t trsm_batched(cublasHandle_t h,
                            cublasSideMode_t side,
                            cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            cublasDiagType_t diag,
                            int m,
                            int n,
                            const T* alpha,
                            const T* const Aarray[],
                            int lda,
                            T* const Barray[],
                            int ldb,
                            int batch);
template <>
inline cublasStatus_t trsm_batched<float>(cublasHandle_t h,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          int m,
                                          int n,
                                          const float* alpha,
                                          const float* const Aarray[],
                                          int lda,
                                          float* const Barray[],
                                          int ldb,
                                          int batch)
{
  return cublasStrsmBatched(
    h, side, uplo, trans, diag, m, n, alpha, Aarray, lda, Barray, ldb, batch);
}
template <>
inline cublasStatus_t trsm_batched<double>(cublasHandle_t h,
                                           cublasSideMode_t side,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           cublasDiagType_t diag,
                                           int m,
                                           int n,
                                           const double* alpha,
                                           const double* const Aarray[],
                                           int lda,
                                           double* const Barray[],
                                           int ldb,
                                           int batch)
{
  return cublasDtrsmBatched(
    h, side, uplo, trans, diag, m, n, alpha, Aarray, lda, Barray, ldb, batch);
}

// Number of elements in the covariance / precision buffer for a covariance type.
inline size_t cov_elems(covariance_type ct, int d, int K)
{
  switch (ct) {
    case covariance_type::FULL: return (size_t)K * d * d;
    case covariance_type::TIED: return (size_t)d * d;
    case covariance_type::DIAG: return (size_t)K * d;
    case covariance_type::SPHERICAL: return (size_t)K;
  }
  return 0;
}

// ===========================================================================
// E-step
// ===========================================================================
template <typename T, int D>
void launch_small_fixed(const T* X,
                        const T* weights,
                        const T* means,
                        const T* prec_chol,
                        const T* log_det,
                        int n,
                        int K,
                        int prec_pc,
                        T* log_prob,
                        dim3 grid,
                        dim3 block,
                        cudaStream_t stream)
{
  size_t shmem = (D + upper_tri_size(D)) * sizeof(T);
  if (shmem > DEFAULT_SMEM_LIMIT) {
    RAFT_CUDA_TRY(cudaFuncSetAttribute(detail::e_step_log_prob_small_kernel<T, D>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int)shmem));
  }
  detail::e_step_log_prob_small_kernel<T, D><<<grid, block, shmem, stream>>>(
    X, weights, means, prec_chol, log_det, n, D, K, prec_pc, log_prob);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T>
void e_step(raft::resources const& handle,
            const params& params,
            const T* X,
            int n,
            int d,
            int K,
            const T* weights,
            const T* means,
            const T* prec_chol,
            const T* log_det,
            T* log_prob,
            T* resp,
            T* log_prob_norm)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  covariance_type ct  = params.cov_type;

  if (ct == covariance_type::FULL || (ct == covariance_type::TIED && d > 128)) {
    int prec_pc = (ct == covariance_type::FULL) ? 1 : 0;
    // Size-based solver selection: the fused
    // shared-memory kernels are fastest for the moderate-d regime; wide
    // feature counts (and float64 above 64) route through a cuBLAS E-step that
    // forms (X - means_k) @ prec_chol_k with a GEMM per component.
    bool use_cublas = (sizeof(T) == 4) ? (d >= 257) : (d > 64);
    if (use_cublas) {
      cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
      cublas_check(cublasSetStream(cublas, stream), "cublasSetStream");
      rmm::device_uvector<T> centered((size_t)n * d, stream);
      rmm::device_uvector<T> y((size_t)n * d, stream);
      T one = T(1), zero = T(0);
      int threads       = 256;
      int center_blocks = (int)(((size_t)n * d + threads - 1) / threads);
      int row_blocks    = (n - 1) / threads + 1;  // (n + threads - 1) overflows int
      for (int k = 0; k < K; ++k) {
        detail::e_step_center_kernel<T>
          <<<center_blocks, threads, 0, stream>>>(X, means, n, d, k, centered.data());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
        const T* pc_k = prec_chol + (size_t)(prec_pc ? k : 0) * d * d;
        // y = (X - means_k) @ prec_chol_k. cuBLAS is column-major: the row-major
        // (n, d) centered buffer is a column-major (d, n) matrix, so this GEMM
        // computes the column-major (d, n) result prec_chol_kᵀ_cm @ centered_cm,
        // which read back row-major is exactly the (n, d) matrix y[row, j].
        cublas_check(cublas_gemm<T>(cublas,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    d,
                                    n,
                                    d,
                                    &one,
                                    pc_k,
                                    d,
                                    centered.data(),
                                    d,
                                    &zero,
                                    y.data(),
                                    d),
                     "gemm(e_step_cublas)");
        detail::e_step_log_prob_from_y_kernel<T>
          <<<row_blocks, threads, 0, stream>>>(y.data(), weights, log_det, n, d, K, k, log_prob);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }
    } else if (d <= 64) {
      dim3 block(E_STEP_BLOCK);
      dim3 grid((n - 1) / E_STEP_BLOCK + 1, K);
      if (d == 16) {
        launch_small_fixed<T, 16>(
          X, weights, means, prec_chol, log_det, n, K, prec_pc, log_prob, grid, block, stream);
      } else if (d == 32) {
        launch_small_fixed<T, 32>(
          X, weights, means, prec_chol, log_det, n, K, prec_pc, log_prob, grid, block, stream);
      } else if (d == 50) {
        launch_small_fixed<T, 50>(
          X, weights, means, prec_chol, log_det, n, K, prec_pc, log_prob, grid, block, stream);
      } else if (d == 64) {
        launch_small_fixed<T, 64>(
          X, weights, means, prec_chol, log_det, n, K, prec_pc, log_prob, grid, block, stream);
      } else {
        size_t shmem = ((size_t)d + upper_tri_size(d)) * sizeof(T);
        if (shmem > DEFAULT_SMEM_LIMIT) {
          RAFT_CUDA_TRY(cudaFuncSetAttribute(detail::e_step_log_prob_small_kernel<T, 0>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             (int)shmem));
        }
        detail::e_step_log_prob_small_kernel<T><<<grid, block, shmem, stream>>>(
          X, weights, means, prec_chol, log_det, n, d, K, prec_pc, log_prob);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }
    } else {
      dim3 block(E_STEP_THREAD64_BLOCK);
      dim3 grid((n - 1) / E_STEP_THREAD64_BLOCK + 1, K);
      size_t shmem =
        ((size_t)E_STEP_LARGE64_TILE + (size_t)E_STEP_LARGE64_TILE * E_STEP_LARGE64_TILE) *
        sizeof(T);
      detail::e_step_log_prob_large_d_thread64_kernel<T, E_STEP_LARGE64_TILE>
        <<<grid, block, shmem, stream>>>(
          X, weights, means, prec_chol, log_det, n, d, K, prec_pc, log_prob);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  } else if (ct == covariance_type::DIAG ||
             ct == covariance_type::SPHERICAL) {  // fast register-tiled log-prob
    // (component-tiled in shared memory, so it scales to large K where the
    // per-mode kernels can't fit the means). Writes log_prob[n,k]; normalize below.
    rmm::device_uvector<T> const_k(K, stream);
    detail::fused_const_kernel<T>
      <<<dim3((K + 255) / 256), dim3(256), 0, stream>>>(weights, log_det, d, K, const_k.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    constexpr int FEAT = 32;
    constexpr int CELL = (sizeof(T) == 4) ? 64 : 32;
    int tpb            = 256;
    int gb             = (n - 1) / tpb + 1;
    if (ct == covariance_type::DIAG)
      detail::estep_tiled_kernel<T, CELL, FEAT, /*DIAG*/ true, /*WRITE_FULL*/ true>
        <<<gb, tpb, 0, stream>>>(
          X, means, prec_chol, const_k.data(), n, d, K, nullptr, nullptr, log_prob);
    else
      detail::estep_tiled_kernel<T, CELL, FEAT, /*DIAG*/ false, /*WRITE_FULL*/ true>
        <<<gb, tpb, 0, stream>>>(
          X, means, prec_chol, const_k.data(), n, d, K, nullptr, nullptr, log_prob);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else {  // TIED, d <= 128: shared Cholesky U -> transform X̃ = X·U, μ̃ = means·U,
            // then ‖Uᵀ(x-μ_k)‖² is Euclidean ‖x̃-μ̃_k‖² -> same register-tiled kernel.
    cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
    cublas_check(cublasSetStream(cublas, stream), "cublasSetStream");
    rmm::device_uvector<T> xt((size_t)n * d, stream);
    rmm::device_uvector<T> mut((size_t)K * d, stream);
    rmm::device_uvector<T> ones_pc(K, stream);
    rmm::device_uvector<T> const_k(K, stream);
    thrust::fill(thrust::cuda::par.on(stream), ones_pc.data(), ones_pc.data() + K, T(1));
    T one = T(1), zero = T(0);
    cublas_check(
      cublas_gemm<T>(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, d, n, d, &one, prec_chol, d, X, d, &zero, xt.data(), d),
      "gemm(tied_X)");
    cublas_check(cublas_gemm<T>(cublas,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                d,
                                K,
                                d,
                                &one,
                                prec_chol,
                                d,
                                means,
                                d,
                                &zero,
                                mut.data(),
                                d),
                 "gemm(tied_mu)");
    detail::fused_const_kernel<T>
      <<<dim3((K + 255) / 256), dim3(256), 0, stream>>>(weights, log_det, d, K, const_k.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    constexpr int FEAT = 32;
    constexpr int CELL = (sizeof(T) == 4) ? 64 : 32;
    int tpb            = 256;
    int gb             = (n - 1) / tpb + 1;
    detail::estep_tiled_kernel<T, CELL, FEAT, /*DIAG*/ false, /*WRITE_FULL*/ true>
      <<<gb, tpb, 0, stream>>>(
        xt.data(), mut.data(), ones_pc.data(), const_k.data(), n, d, K, nullptr, nullptr, log_prob);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  dim3 nb(NORMALIZE_BLOCK);
  dim3 ng(n);
  detail::e_step_normalize_kernel<T><<<ng, nb, 0, stream>>>(log_prob, n, K, resp, log_prob_norm);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// ===========================================================================
// M-step (weights, means, covariances) + precision Cholesky + log-det
// ===========================================================================
template <typename T>
struct MStepWorkspace {
  rmm::device_uvector<T> ones;
  rmm::device_uvector<T> N_k;
  rmm::device_uvector<T> num;
  rmm::device_uvector<T> num2;
  rmm::device_uvector<T> centered;
  rmm::device_uvector<T> Xsq;
  rmm::device_uvector<T> diag_var;
  rmm::device_uvector<T> scaled_means;
  rmm::device_uvector<T> XtX;
  rmm::device_uvector<T> Lbuf;
  rmm::device_uvector<T> potrf_work;
  rmm::device_uvector<int> devInfo;
  rmm::device_uvector<T> cov_work;  // batched copy of covariances for potrfBatched
  rmm::device_uvector<T*> dA_ptrs;  // device array of K pointers into cov_work
  rmm::device_uvector<T*> dB_ptrs;  // device array of K pointers into precisions_chol
  int lwork;

  MStepWorkspace(raft::resources const& handle, const params& params, int n, int d, int K)
    : ones(0, raft::resource::get_cuda_stream(handle)),
      N_k(K, raft::resource::get_cuda_stream(handle)),
      num((size_t)K * d, raft::resource::get_cuda_stream(handle)),
      num2(0, raft::resource::get_cuda_stream(handle)),
      centered(0, raft::resource::get_cuda_stream(handle)),
      Xsq(0, raft::resource::get_cuda_stream(handle)),
      diag_var(0, raft::resource::get_cuda_stream(handle)),
      scaled_means(0, raft::resource::get_cuda_stream(handle)),
      XtX(0, raft::resource::get_cuda_stream(handle)),
      Lbuf(0, raft::resource::get_cuda_stream(handle)),
      potrf_work(0, raft::resource::get_cuda_stream(handle)),
      devInfo(K, raft::resource::get_cuda_stream(handle)),
      cov_work(0, raft::resource::get_cuda_stream(handle)),
      dA_ptrs(0, raft::resource::get_cuda_stream(handle)),
      dB_ptrs(0, raft::resource::get_cuda_stream(handle)),
      lwork(0)
  {
    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    covariance_type ct  = params.cov_type;
    ones.resize(n, stream);
    thrust::fill(thrust::cuda::par.on(stream), ones.data(), ones.data() + n, T(1));
    if (ct == covariance_type::FULL) {
      centered.resize((size_t)n * d, stream);
      // batched precision-Cholesky scratch
      cov_work.resize((size_t)K * d * d, stream);
      dA_ptrs.resize(K, stream);
      dB_ptrs.resize(K, stream);
    } else if (ct == covariance_type::TIED) {
      scaled_means.resize((size_t)K * d, stream);
      XtX.resize((size_t)d * d, stream);
    } else {  // diag / spherical
      Xsq.resize((size_t)n * d, stream);
      num2.resize((size_t)K * d, stream);
      if (ct == covariance_type::SPHERICAL) diag_var.resize((size_t)K * d, stream);
    }
    if (ct == covariance_type::TIED) {
      Lbuf.resize((size_t)d * d, stream);
      int lw = 0;
      cusolver_check(potrf_bufsize<T>(raft::resource::get_cusolver_dn_handle(handle),
                                      CUBLAS_FILL_MODE_LOWER,
                                      d,
                                      Lbuf.data(),
                                      d,
                                      &lw),
                     "potrf_bufferSize");
      lwork = lw;
      potrf_work.resize((size_t)lw, stream);
    }
  }
};

// Accumulate responsibility-weighted sufficient statistics for one batch into the
// workspace: N_k += Σr, num += Σr·x, plus the second-moment term (num2 += Σr·x²
// for diag/sph; covariances += Σr·x·xᵀ for full; XtX += Σx·xᵀ for tied). beta=0
// starts fresh, beta=1 adds on — so fit can stream N in tiles, never holding (N,K).
template <typename T>
void m_accumulate(raft::resources const& handle,
                  const params& params,
                  const T* X,
                  int n,
                  int d,
                  int K,
                  const T* resp,
                  MStepWorkspace<T>& ws,
                  T beta)
{
  cudaStream_t stream   = raft::resource::get_cuda_stream(handle);
  cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
  cublas_check(cublasSetStream(cublas, stream), "cublasSetStream");
  covariance_type ct = params.cov_type;
  T one              = T(1);

  cublas_check(
    cublas_gemv<T>(
      cublas, CUBLAS_OP_N, K, n, &one, resp, K, ws.ones.data(), 1, &beta, ws.N_k.data(), 1),
    "gemv(N_k)");
  cublas_check(
    cublas_gemm<T>(
      cublas, CUBLAS_OP_N, CUBLAS_OP_T, d, K, n, &one, X, d, resp, K, &beta, ws.num.data(), d),
    "gemm(num)");

  // FULL accumulates its centered covariance in a separate pass (needs means);
  // here only N_k / num are gathered for it.
  if (ct == covariance_type::TIED) {
    cublas_check(
      cublas_gemm<T>(
        cublas, CUBLAS_OP_N, CUBLAS_OP_T, d, d, n, &one, X, d, X, d, &beta, ws.XtX.data(), d),
      "gemm(XtX)");
  } else if (ct == covariance_type::DIAG || ct == covariance_type::SPHERICAL) {
    int threads = 256;
    int blocks  = (int)(((size_t)n * d + threads - 1) / threads);
    detail::elementwise_square_kernel<T>
      <<<blocks, threads, 0, stream>>>(X, (size_t)n * d, ws.Xsq.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    cublas_check(cublas_gemm<T>(cublas,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                d,
                                K,
                                n,
                                &one,
                                ws.Xsq.data(),
                                d,
                                resp,
                                K,
                                &beta,
                                ws.num2.data(),
                                d),
                 "gemm(num2)");
  }
}

// Turn accumulated stats into weights / means / covariances (uncentered recovery).
template <typename T>
void m_finalize(raft::resources const& handle,
                const params& params,
                int n,
                int d,
                int K,
                MStepWorkspace<T>& ws,
                T* weights,
                T* means,
                T* covariances)
{
  cudaStream_t stream   = raft::resource::get_cuda_stream(handle);
  cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
  cublas_check(cublasSetStream(cublas, stream), "cublasSetStream");
  covariance_type ct = params.cov_type;
  T one = T(1), zero = T(0);
  T eps = std::numeric_limits<T>::epsilon();

  detail::m_step_finalize_means_kernel<T>
    <<<dim3(K), dim3(256), 0, stream>>>(ws.N_k.data(), ws.num.data(), weights, means, eps, n, d, K);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // FULL: covariance is finalized after its separate centered pass; nothing here.
  if (ct == covariance_type::TIED) {
    rmm::device_uvector<T> weighted_outer((size_t)d * d, stream);
    detail::scale_rows_by_kernel<T>
      <<<dim3(K), dim3(256), 0, stream>>>(means, ws.N_k.data(), eps, d, K, ws.scaled_means.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    cublas_check(cublas_gemm<T>(cublas,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                d,
                                d,
                                K,
                                &one,
                                ws.scaled_means.data(),
                                d,
                                means,
                                d,
                                &zero,
                                weighted_outer.data(),
                                d),
                 "gemm(weighted_outer)");
    std::vector<T> h_Nk(K);
    raft::copy(h_Nk.data(), ws.N_k.data(), K, stream);
    raft::resource::sync_stream(handle);
    double sum_nk = 0.0;
    for (int k = 0; k < K; ++k)
      sum_nk += (double)h_Nk[k] + 10.0 * (double)eps;
    size_t total = (size_t)d * d;
    int threads = 256, blocks = (int)((total + threads - 1) / threads);
    detail::m_step_finalize_tied_kernel<T><<<blocks, threads, 0, stream>>>(
      ws.XtX.data(), weighted_outer.data(), T(sum_nk), T(params.reg_covar), d, covariances);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else if (ct == covariance_type::DIAG || ct == covariance_type::SPHERICAL) {
    if (ct == covariance_type::DIAG) {
      detail::m_step_finalize_diag_kernel<T><<<dim3(K), dim3(256), 0, stream>>>(
        ws.N_k.data(), ws.num2.data(), means, T(params.reg_covar), eps, d, K, covariances);
    } else {
      detail::m_step_finalize_diag_kernel<T><<<dim3(K), dim3(256), 0, stream>>>(
        ws.N_k.data(), ws.num2.data(), means, T(params.reg_covar), eps, d, K, ws.diag_var.data());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      detail::m_step_spherical_from_diag_kernel<T>
        <<<dim3(K), dim3(256), 0, stream>>>(ws.diag_var.data(), d, K, covariances);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

// FULL covariance, pass 2: accumulate the centered moment Σ r·(x-mu)(x-mu)ᵀ for
// one batch into covariances (beta=0 fresh, beta=1 add). Centered (not Σr·x·xᵀ)
// to avoid the catastrophic cancellation of the uncentered form on data far from
// the origin. Requires the means from pass 1.
template <typename T>
void m_cov_full_pass(raft::resources const& handle,
                     const T* X,
                     int n,
                     int d,
                     int K,
                     const T* resp,
                     const T* means,
                     T* covariances,
                     MStepWorkspace<T>& ws,
                     T beta)
{
  cudaStream_t stream   = raft::resource::get_cuda_stream(handle);
  cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
  cublas_check(cublasSetStream(cublas, stream), "cublasSetStream");
  T one       = T(1);
  int threads = 256;
  int blocks  = (int)(((size_t)n * d + threads - 1) / threads);
  for (int k = 0; k < K; ++k) {
    detail::weighted_center_kernel<T>
      <<<blocks, threads, 0, stream>>>(X, resp, means, n, d, K, k, ws.centered.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    cublas_check(cublas_gemm<T>(cublas,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                d,
                                d,
                                n,
                                &one,
                                ws.centered.data(),
                                d,
                                ws.centered.data(),
                                d,
                                &beta,
                                covariances + (size_t)k * d * d,
                                d),
                 "gemm(cov_full)");
  }
}

// Finalize FULL covariances (divide by N_k, symmetrize, add reg_covar).
template <typename T>
void m_finalize_cov_full(raft::resources const& handle,
                         const params& params,
                         int d,
                         int K,
                         MStepWorkspace<T>& ws,
                         T* covariances)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  T eps               = std::numeric_limits<T>::epsilon();
  detail::m_step_finalize_cov_full_kernel<T>
    <<<dim3(K), dim3(256), 0, stream>>>(ws.N_k.data(), covariances, T(params.reg_covar), eps, d, K);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// One d×d precision-Cholesky: potrf(LOWER) then trsm solving L X = I; the
// resulting column-major L^{-1} is, read row-major, the upper precision factor.
template <typename T>
void precision_cholesky_one(
  raft::resources const& handle, const T* cov, T* prec_chol, int d, MStepWorkspace<T>& ws)
{
  cudaStream_t stream       = raft::resource::get_cuda_stream(handle);
  cusolverDnHandle_t solver = raft::resource::get_cusolver_dn_handle(handle);
  cublasHandle_t cublas     = raft::resource::get_cublas_handle(handle);
  cublas_check(cublasSetStream(cublas, stream), "cublasSetStream");
  cusolver_check(cusolverDnSetStream(solver, stream), "cusolverDnSetStream");

  raft::copy(ws.Lbuf.data(), cov, (size_t)d * d, stream);
  cusolver_check(potrf<T>(solver,
                          CUBLAS_FILL_MODE_LOWER,
                          d,
                          ws.Lbuf.data(),
                          d,
                          ws.potrf_work.data(),
                          ws.lwork,
                          ws.devInfo.data()),
                 "potrf");
  int info = 0;
  raft::copy(&info, ws.devInfo.data(), 1, stream);
  raft::resource::sync_stream(handle);
  if (info != 0) throw std::runtime_error(precision_error_message());

  detail::set_identity_kernel<T>
    <<<dim3((int)(((size_t)d * d + 255) / 256)), dim3(256), 0, stream>>>(prec_chol, d);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  T one = T(1);
  cublas_check(cublas_trsm<T>(cublas,
                              CUBLAS_SIDE_LEFT,
                              CUBLAS_FILL_MODE_LOWER,
                              CUBLAS_OP_N,
                              CUBLAS_DIAG_NON_UNIT,
                              d,
                              d,
                              &one,
                              ws.Lbuf.data(),
                              d,
                              prec_chol,
                              d),
               "trsm(precision_cholesky)");
}

// Batched precision-Cholesky for full covariance: one potrfBatched + one
// trsmBatched over all K components (single info sync), matching the batched
// approach a cupy/array-API solver uses, instead of K per-component calls each
// with a host sync.
template <typename T>
void precision_cholesky_full_batched(raft::resources const& handle,
                                     const T* covariances,
                                     T* prec_chol,
                                     int d,
                                     int K,
                                     MStepWorkspace<T>& ws)
{
  cudaStream_t stream       = raft::resource::get_cuda_stream(handle);
  cusolverDnHandle_t solver = raft::resource::get_cusolver_dn_handle(handle);
  cublasHandle_t cublas     = raft::resource::get_cublas_handle(handle);
  cublas_check(cublasSetStream(cublas, stream), "cublasSetStream");
  cusolver_check(cusolverDnSetStream(solver, stream), "cusolverDnSetStream");

  raft::copy(ws.cov_work.data(), covariances, (size_t)K * d * d, stream);
  std::vector<T*> hA(K), hB(K);
  for (int k = 0; k < K; ++k) {
    hA[k] = ws.cov_work.data() + (size_t)k * d * d;
    hB[k] = prec_chol + (size_t)k * d * d;
  }
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(ws.dA_ptrs.data(), hA.data(), sizeof(T*) * K, cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(ws.dB_ptrs.data(), hB.data(), sizeof(T*) * K, cudaMemcpyHostToDevice, stream));

  cusolver_check(
    potrf_batched<T>(solver, CUBLAS_FILL_MODE_LOWER, d, ws.dA_ptrs.data(), d, ws.devInfo.data(), K),
    "potrfBatched");
  detail::set_identity_batched_kernel<T>
    <<<dim3((int)(((size_t)K * d * d + 255) / 256)), dim3(256), 0, stream>>>(prec_chol, d, K);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  T one = T(1);
  cublas_check(trsm_batched<T>(cublas,
                               CUBLAS_SIDE_LEFT,
                               CUBLAS_FILL_MODE_LOWER,
                               CUBLAS_OP_N,
                               CUBLAS_DIAG_NON_UNIT,
                               d,
                               d,
                               &one,
                               ws.dA_ptrs.data(),
                               d,
                               ws.dB_ptrs.data(),
                               d,
                               K),
               "trsmBatched");

  std::vector<int> hinfo(K);
  raft::copy(hinfo.data(), ws.devInfo.data(), K, stream);
  raft::resource::sync_stream(handle);
  for (int k = 0; k < K; ++k)
    if (hinfo[k] != 0) throw std::runtime_error(precision_error_message());
}

// covariances -> precisions_chol and log_det, dispatched on covariance type.
template <typename T>
void update_precisions(raft::resources const& handle,
                       const params& params,
                       const T* covariances,
                       int d,
                       int K,
                       T* prec_chol,
                       T* log_det,
                       MStepWorkspace<T>& ws)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  covariance_type ct  = params.cov_type;
  if (ct == covariance_type::FULL) {
    precision_cholesky_full_batched<T>(handle, covariances, prec_chol, d, K, ws);
    detail::log_det_full_kernel<T>
      <<<dim3(K), dim3(REDUCE_BLOCK), 0, stream>>>(prec_chol, d, K, log_det);
  } else if (ct == covariance_type::TIED) {
    precision_cholesky_one<T>(handle, covariances, prec_chol, d, ws);
    detail::log_det_tied_kernel<T>
      <<<dim3(1), dim3(REDUCE_BLOCK), 0, stream>>>(prec_chol, d, K, log_det);
  } else {
    size_t total = cov_elems(ct, d, K);
    T min_var    = thrust::reduce(thrust::cuda::par.on(stream),
                               covariances,
                               covariances + total,
                               std::numeric_limits<T>::max(),
                               thrust::minimum<T>());
    if (min_var <= T(0)) {
      // a non-positive variance means the covariance is ill-defined
      throw std::runtime_error(precision_error_message());
    }
    detail::recip_sqrt_kernel<T>
      <<<dim3((total + 255) / 256), dim3(256), 0, stream>>>(covariances, total, prec_chol);
    if (ct == covariance_type::DIAG) {
      detail::log_det_diag_kernel<T>
        <<<dim3(K), dim3(REDUCE_BLOCK), 0, stream>>>(prec_chol, d, K, log_det);
    } else {
      detail::log_det_spherical_kernel<T>
        <<<dim3((K + 255) / 256), dim3(256), 0, stream>>>(prec_chol, d, K, log_det);
    }
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// precisions_ = prec_chol @ prec_cholᵀ (full/tied) or prec_chol^2 (diag/spherical).
template <typename T>
void compute_precisions(raft::resources const& handle,
                        const params& params,
                        const T* prec_chol,
                        int d,
                        int K,
                        T* precisions)
{
  cudaStream_t stream   = raft::resource::get_cuda_stream(handle);
  cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
  cublas_check(cublasSetStream(cublas, stream), "cublasSetStream");
  covariance_type ct = params.cov_type;
  T one = T(1), zero = T(0);
  if (ct == covariance_type::FULL) {
    for (int k = 0; k < K; ++k) {
      const T* U = prec_chol + (size_t)k * d * d;  // row-major upper U
      T* P       = precisions + (size_t)k * d * d;
      // P = U @ Uᵀ (row-major). In column-major: U_cm = Uᵀ (upper->lower). Use
      // gemm(OP_T, OP_N, d, d, d, U, U) which yields U_cmᵀ @ U_cm = U @ Uᵀ.
      cublas_check(
        cublas_gemm<T>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, d, d, d, &one, U, d, U, d, &zero, P, d),
        "gemm(precisions_full)");
    }
  } else if (ct == covariance_type::TIED) {
    cublas_check(cublas_gemm<T>(cublas,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                d,
                                d,
                                d,
                                &one,
                                prec_chol,
                                d,
                                prec_chol,
                                d,
                                &zero,
                                precisions,
                                d),
                 "gemm(precisions_tied)");
  } else {
    size_t total = cov_elems(ct, d, K);
    detail::elementwise_square_kernel<T>
      <<<dim3((total + 255) / 256), dim3(256), 0, stream>>>(prec_chol, total, precisions);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

// k-means (or k-means++ seeding) hard labels for GMM initialization. Centroids
// are computed internally; only the per-sample labels (length n) are returned.
template <typename T>
void kmeans_assign(raft::resources const& handle,
                   const params& params,
                   const T* X,
                   int n,
                   int d,
                   int K,
                   uint64_t init_seed,
                   int* labels_out)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  cuvs::cluster::kmeans::params kp;
  kp.n_clusters = K;
  // KMeansPlusPlus: seeding labels only (max_iter=0, no Lloyd); KMeans: full Lloyd.
  kp.max_iter  = (params.init == init_method::KMeansPlusPlus) ? 0 : 300;
  kp.tol       = 1e-4;
  kp.n_init    = 1;
  kp.init      = cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus;
  kp.rng_state = raft::random::RngState{init_seed, raft::random::GeneratorType::GenPhilox};
  rmm::device_uvector<T> centroids((size_t)K * d, stream);
  T inertia   = T(0);
  int km_iter = 0;
  auto X_view = raft::make_device_matrix_view<const T, int>(X, n, d);
  cuvs::cluster::kmeans::fit(handle,
                             kp,
                             X_view,
                             std::nullopt,
                             raft::make_device_matrix_view<T, int>(centroids.data(), K, d),
                             raft::make_host_scalar_view<T>(&inertia),
                             raft::make_host_scalar_view<int>(&km_iter));
  cuvs::cluster::kmeans::predict(
    handle,
    kp,
    X_view,
    std::nullopt,
    raft::make_device_matrix_view<const T, int>(centroids.data(), K, d),
    raft::make_device_vector_view<int, int>(labels_out, n),
    true,
    raft::make_host_scalar_view<T>(&inertia));
}

// ===========================================================================
// EM driver
// ===========================================================================
template <typename T>
T mean_device(raft::resources const& handle, const T* v, int n)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  T sum = thrust::reduce(thrust::cuda::par.on(stream), v, v + n, T(0), thrust::plus<T>());
  return sum / T(n);
}

template <typename T>
void fit_impl(raft::resources const& handle,
              const params& params,
              const T* X,
              int n,
              int d,
              T* weights,
              T* means,
              T* covariances,
              T* precisions_chol,
              T* precisions,
              int* labels,
              T& lower_bound,
              int& n_iter,
              bool& converged,
              bool warm_start)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  int K               = params.n_components;
  covariance_type ct  = params.cov_type;
  size_t cn           = cov_elems(ct, d, K);

  // Stream the E+M over N-tiles: only a bounded (tile, K) responsibility block
  // ever exists, never the full (N, K).
  int tile = std::min(n, 65536);

  MStepWorkspace<T> ws(handle, params, tile, d, K);
  // normalize overwrites log_prob with resp in place (each element is read once and
  // written once in the same expression), so one (tile, K) buffer serves both.
  rmm::device_uvector<T> resp((size_t)tile * K, stream);
  rmm::device_uvector<T> lpn(tile, stream);
  rmm::device_uvector<T> log_det(K, stream);

  rmm::device_uvector<T> best_w(K, stream);
  rmm::device_uvector<T> best_m((size_t)K * d, stream);
  rmm::device_uvector<T> best_cov(cn, stream);
  rmm::device_uvector<T> best_pc(cn, stream);

  raft::random::RngState rng(params.seed, raft::random::GeneratorType::GenPhilox);

  // FULL's M-step runs the E-step twice (first to accumulate weights/means, then
  // a second centered pass for the covariances). m_finalize overwrites
  // weights/means between the two passes, so the covariance pass must re-run the
  // E-step with the *pre-update* params to reproduce the same responsibilities
  // the means/weights were built from; snapshot them here. Sized 0 (unused) for
  // the other covariance types, which need no second pass.
  rmm::device_uvector<T> estep_w((ct == covariance_type::FULL) ? K : 0, stream);
  rmm::device_uvector<T> estep_m((ct == covariance_type::FULL) ? (size_t)K * d : 0, stream);

  // One E-step (+ optional M-step) over all data, streamed in tiles; returns the
  // mean lower bound. The E-step weights/means are passed in so FULL's covariance
  // pass can feed the snapshot above and reproduce the first pass's resp exactly.
  auto estep_tile = [&](int t0, int nt, const T* e_weights, const T* e_means) {
    e_step<T>(handle,
              params,
              X + (size_t)t0 * d,
              nt,
              d,
              K,
              e_weights,
              e_means,
              precisions_chol,
              log_det.data(),
              resp.data(),
              resp.data(),
              lpn.data());
  };
  auto em_step = [&](bool do_mstep) -> double {
    double lb_sum = 0.0;
    for (int t0 = 0; t0 < n; t0 += tile) {
      int nt = std::min(tile, n - t0);
      estep_tile(t0, nt, weights, means);
      if (do_mstep)
        m_accumulate<T>(
          handle, params, X + (size_t)t0 * d, nt, d, K, resp.data(), ws, (t0 == 0) ? T(0) : T(1));
      lb_sum += (double)mean_device<T>(handle, lpn.data(), nt) * nt;
    }
    if (do_mstep) {
      if (ct == covariance_type::FULL) {
        // Snapshot the params the E-step above used; m_finalize is about to
        // overwrite weights/means and the covariance pass must reproduce this resp.
        raft::copy(estep_w.data(), weights, K, stream);
        raft::copy(estep_m.data(), means, (size_t)K * d, stream);
      }
      m_finalize<T>(handle, params, n, d, K, ws, weights, means, covariances);
      if (ct == covariance_type::FULL) {
        for (int t0 = 0; t0 < n; t0 += tile) {
          int nt = std::min(tile, n - t0);
          estep_tile(t0, nt, estep_w.data(), estep_m.data());
          m_cov_full_pass<T>(handle,
                             X + (size_t)t0 * d,
                             nt,
                             d,
                             K,
                             resp.data(),
                             means,
                             covariances,
                             ws,
                             (t0 == 0) ? T(0) : T(1));
        }
        m_finalize_cov_full<T>(handle, params, d, K, ws, covariances);
      }
    }
    return lb_sum / n;
  };

  bool do_init   = !warm_start;
  int n_init     = do_init ? params.n_init : 1;
  double best_lb = -std::numeric_limits<double>::infinity();
  int best_iter  = 0;
  bool best_conv = false;
  bool have_best = false;

  for (int init = 0; init < n_init; ++init) {
    if (do_init) {
      uint64_t init_seed = params.seed + (uint64_t)init * 0x9E3779B97F4A7C15ULL;
      init_method im     = params.init;
      // Per-sample assignment is O(n)/O(K), not (N, K): k-means labels, or the K
      // random-from-data indices. The one-hot resp is then built one tile at a time.
      rmm::device_uvector<int> km_labels(0, stream);
      rmm::device_uvector<int> rfd_idx(0, stream);
      if (im == init_method::KMeans || im == init_method::KMeansPlusPlus) {
        km_labels.resize(n, stream);
        kmeans_assign<T>(handle, params, X, n, d, K, init_seed, km_labels.data());
      } else if (im == init_method::RandomFromData) {
        rfd_idx.resize(K, stream);
        auto idx = raft::random::excess_subsample<int, int>(handle, rng, n, K);
        raft::copy(rfd_idx.data(), idx.data_handle(), K, stream);
      }
      auto fill_init_resp = [&](int t0, int nt) {
        if (im == init_method::Random) {
          // Deterministic per-(init, tile): FULL fills the responsibilities twice
          // (means pass, then centered-covariance pass) and both must match. A
          // shared rng would advance between them and desync cov from the means.
          raft::random::RngState tile_rng(
            init_seed ^ (static_cast<uint64_t>(t0) * 0x9E3779B97F4A7C15ULL),
            raft::random::GeneratorType::GenPhilox);
          raft::random::uniform(handle, tile_rng, resp.data(), (size_t)nt * K, T(0), T(1));
          detail::normalize_rows_kernel<T>
            <<<dim3((nt + 255) / 256), dim3(256), 0, stream>>>(resp.data(), nt, K);
        } else if (im == init_method::RandomFromData) {
          RAFT_CUDA_TRY(cudaMemsetAsync(resp.data(), 0, sizeof(T) * (size_t)nt * K, stream));
          detail::scatter_onehot_tile_kernel<T><<<dim3((K + 255) / 256), dim3(256), 0, stream>>>(
            rfd_idx.data(), K, t0, nt, resp.data());
        } else {
          detail::labels_to_onehot_kernel<T><<<dim3((nt + 255) / 256), dim3(256), 0, stream>>>(
            km_labels.data() + t0, nt, K, resp.data());
        }
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      };
      for (int t0 = 0; t0 < n; t0 += tile) {
        int nt = std::min(tile, n - t0);
        fill_init_resp(t0, nt);
        m_accumulate<T>(
          handle, params, X + (size_t)t0 * d, nt, d, K, resp.data(), ws, (t0 == 0) ? T(0) : T(1));
      }
      m_finalize<T>(handle, params, n, d, K, ws, weights, means, covariances);
      if (ct == covariance_type::FULL) {
        for (int t0 = 0; t0 < n; t0 += tile) {
          int nt = std::min(tile, n - t0);
          fill_init_resp(t0, nt);
          m_cov_full_pass<T>(handle,
                             X + (size_t)t0 * d,
                             nt,
                             d,
                             K,
                             resp.data(),
                             means,
                             covariances,
                             ws,
                             (t0 == 0) ? T(0) : T(1));
        }
        m_finalize_cov_full<T>(handle, params, d, K, ws, covariances);
      }
      update_precisions<T>(handle, params, covariances, d, K, precisions_chol, log_det.data(), ws);
    } else {
      update_precisions<T>(handle, params, covariances, d, K, precisions_chol, log_det.data(), ws);
    }

    double lb = -std::numeric_limits<double>::infinity();
    bool conv = false;
    int iters = 0;
    for (int it = 1; it <= params.max_iter; ++it) {
      double prev = lb;
      lb          = em_step(/* do_mstep */ true);
      update_precisions<T>(handle, params, covariances, d, K, precisions_chol, log_det.data(), ws);
      iters = it;
      if (std::abs(lb - prev) < params.tol) {
        conv = true;
        break;
      }
    }
    if (params.max_iter == 0) lb = em_step(/* do_mstep */ false);

    if (!have_best || lb > best_lb) {
      best_lb   = lb;
      best_iter = iters;
      best_conv = conv;
      have_best = true;
      raft::copy(best_w.data(), weights, K, stream);
      raft::copy(best_m.data(), means, (size_t)K * d, stream);
      raft::copy(best_cov.data(), covariances, cn, stream);
      raft::copy(best_pc.data(), precisions_chol, cn, stream);
    }
  }

  // restore best parameters
  raft::copy(weights, best_w.data(), K, stream);
  raft::copy(means, best_m.data(), (size_t)K * d, stream);
  raft::copy(covariances, best_cov.data(), cn, stream);
  raft::copy(precisions_chol, best_pc.data(), cn, stream);

  // recompute log_det for the best precisions_chol, derive precisions_, labels
  if (ct == covariance_type::FULL)
    detail::log_det_full_kernel<T>
      <<<dim3(K), dim3(REDUCE_BLOCK), 0, stream>>>(precisions_chol, d, K, log_det.data());
  else if (ct == covariance_type::TIED)
    detail::log_det_tied_kernel<T>
      <<<dim3(1), dim3(REDUCE_BLOCK), 0, stream>>>(precisions_chol, d, K, log_det.data());
  else if (ct == covariance_type::DIAG)
    detail::log_det_diag_kernel<T>
      <<<dim3(K), dim3(REDUCE_BLOCK), 0, stream>>>(precisions_chol, d, K, log_det.data());
  else
    detail::log_det_spherical_kernel<T>
      <<<dim3((K + 255) / 256), dim3(256), 0, stream>>>(precisions_chol, d, K, log_det.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  compute_precisions<T>(handle, params, precisions_chol, d, K, precisions);

  // Hard labels = argmax responsibility, computed tile-by-tile (no full (N, K)).
  for (int t0 = 0; t0 < n; t0 += tile) {
    int nt = std::min(tile, n - t0);
    e_step<T>(handle,
              params,
              X + (size_t)t0 * d,
              nt,
              d,
              K,
              weights,
              means,
              precisions_chol,
              log_det.data(),
              resp.data(),
              resp.data(),
              lpn.data());
    detail::argmax_kernel<T>
      <<<dim3((nt + 255) / 256), dim3(256), 0, stream>>>(resp.data(), nt, K, labels + t0);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  lower_bound = (T)best_lb;
  n_iter      = best_iter;
  converged   = best_conv;
  raft::resource::sync_stream(handle);
}

// ===========================================================================
// Predict-family helpers (compute log_det from precisions_chol, run E-step)
// ===========================================================================
template <typename T>
void infer(raft::resources const& handle,
           const params& params,
           const T* X,
           int n,
           int d,
           const T* weights,
           const T* means,
           const T* precisions_chol,
           T* resp,
           T* log_prob_norm)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  int K               = params.n_components;
  covariance_type ct  = params.cov_type;

  rmm::device_uvector<T> log_prob((size_t)n * K, stream);
  rmm::device_uvector<T> log_det(K, stream);

  if (ct == covariance_type::FULL)
    detail::log_det_full_kernel<T>
      <<<dim3(K), dim3(REDUCE_BLOCK), 0, stream>>>(precisions_chol, d, K, log_det.data());
  else if (ct == covariance_type::TIED)
    detail::log_det_tied_kernel<T>
      <<<dim3(1), dim3(REDUCE_BLOCK), 0, stream>>>(precisions_chol, d, K, log_det.data());
  else if (ct == covariance_type::DIAG)
    detail::log_det_diag_kernel<T>
      <<<dim3(K), dim3(REDUCE_BLOCK), 0, stream>>>(precisions_chol, d, K, log_det.data());
  else
    detail::log_det_spherical_kernel<T>
      <<<dim3((K + 255) / 256), dim3(256), 0, stream>>>(precisions_chol, d, K, log_det.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  e_step<T>(handle,
            params,
            X,
            n,
            d,
            K,
            weights,
            means,
            precisions_chol,
            log_det.data(),
            log_prob.data(),
            resp,
            log_prob_norm);
}

// Launch the register-tiled fused E-step for spherical (DIAG=false) or diag
// (DIAG=true). Components are tiled into CELL register accumulators; features
// stream through a small shared-memory tile -> high occupancy, any d, and the
// (N, K) distance matrix is never materialized. CELL=64 (float) / 32 (double)
// with FEAT=32 is the occupancy sweet spot on Blackwell.
template <typename T, bool DIAG, bool WRITE_FULL = false>
void launch_estep_tiled(raft::resources const& handle,
                        const T* X,
                        const T* means,
                        const T* prec_chol,
                        const T* const_k,
                        int n,
                        int d,
                        int K,
                        T* log_prob_norm,
                        int* labels,
                        T* log_prob = nullptr)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  constexpr int FEAT  = 32;
  constexpr int CELL  = (sizeof(T) == 4) ? 64 : 32;
  constexpr int TPB   = 256;
  int gb              = (n - 1) / TPB + 1;
  detail::estep_tiled_kernel<T, CELL, FEAT, DIAG, WRITE_FULL><<<gb, TPB, 0, stream>>>(
    X, means, prec_chol, const_k, n, d, K, log_prob_norm, labels, log_prob);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// Fused E-step that writes log_prob_norm (and optionally argmax labels) WITHOUT
// materializing the (N, K) log_prob / responsibility matrix. Mirrors infer()'s
// math for every covariance type, via a per-sample online log-sum-exp:
//   - diag / spherical: fast tiled kernel (means in shared, x in registers),
//   - tied:             transform X̃ = X·U once (shared Cholesky), then it is a
//                       spherical problem ‖x̃ - μ̃_k‖² -> reuse the tiled kernel,
//   - full:             per-component center + GEMM (centered @ prec_chol_k)
//                       folded online (reuses the standard cuBLAS E-step math).
// This is what score_samples / predict use; predict_proba still needs the full
// (N, K) responsibilities so it keeps the standard infer() path.
template <typename T>
void fused_score(raft::resources const& handle,
                 const params& params,
                 const T* X,
                 int n,
                 int d,
                 const T* weights,
                 const T* means,
                 const T* precisions_chol,
                 T* log_prob_norm,
                 int* labels)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  int K               = params.n_components;
  covariance_type ct  = params.cov_type;

  rmm::device_uvector<T> log_det(K, stream);
  if (ct == covariance_type::FULL)
    detail::log_det_full_kernel<T>
      <<<dim3(K), dim3(REDUCE_BLOCK), 0, stream>>>(precisions_chol, d, K, log_det.data());
  else if (ct == covariance_type::TIED)
    detail::log_det_tied_kernel<T>
      <<<dim3(1), dim3(REDUCE_BLOCK), 0, stream>>>(precisions_chol, d, K, log_det.data());
  else if (ct == covariance_type::DIAG)
    detail::log_det_diag_kernel<T>
      <<<dim3(K), dim3(REDUCE_BLOCK), 0, stream>>>(precisions_chol, d, K, log_det.data());
  else
    detail::log_det_spherical_kernel<T>
      <<<dim3((K + 255) / 256), dim3(256), 0, stream>>>(precisions_chol, d, K, log_det.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Precompute the per-component constant once (avoids log() in the hot loop).
  rmm::device_uvector<T> const_k(K, stream);
  detail::fused_const_kernel<T><<<dim3((K + 255) / 256), dim3(256), 0, stream>>>(
    weights, log_det.data(), d, K, const_k.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  int threads = 256;
  int rb      = (n - 1) / threads + 1;

  if (ct == covariance_type::SPHERICAL) {
    launch_estep_tiled<T, /* DIAG */ false>(
      handle, X, means, precisions_chol, const_k.data(), n, d, K, log_prob_norm, labels);
  } else if (ct == covariance_type::DIAG) {
    launch_estep_tiled<T, /* DIAG */ true>(
      handle, X, means, precisions_chol, const_k.data(), n, d, K, log_prob_norm, labels);
  } else if (ct == covariance_type::TIED && d <= 128) {
    // Shared precision Cholesky U: transform X̃ = X·U and μ̃ = means·U once,
    // then the Mahalanobis distance ‖Uᵀ(x-μ_k)‖² becomes the Euclidean
    // ‖x̃ - μ̃_k‖² -> reuse the fast register-tiled kernel (prec = 1).
    cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
    cublas_check(cublasSetStream(cublas, stream), "cublasSetStream");
    rmm::device_uvector<T> xt((size_t)n * d, stream);
    rmm::device_uvector<T> mut((size_t)K * d, stream);
    rmm::device_uvector<T> ones_pc(K, stream);
    thrust::fill(thrust::cuda::par.on(stream), ones_pc.data(), ones_pc.data() + K, T(1));
    T one = T(1), zero = T(0);
    // X̃ = X @ U  and  μ̃ = means @ U  (same GEMM convention as the E-step).
    cublas_check(cublas_gemm<T>(cublas,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                d,
                                n,
                                d,
                                &one,
                                precisions_chol,
                                d,
                                X,
                                d,
                                &zero,
                                xt.data(),
                                d),
                 "gemm(tied_transform_X)");
    cublas_check(cublas_gemm<T>(cublas,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                d,
                                K,
                                d,
                                &one,
                                precisions_chol,
                                d,
                                means,
                                d,
                                &zero,
                                mut.data(),
                                d),
                 "gemm(tied_transform_mu)");
    launch_estep_tiled<T, /* DIAG */ false>(handle,
                                            xt.data(),
                                            mut.data(),
                                            ones_pc.data(),
                                            const_k.data(),
                                            n,
                                            d,
                                            K,
                                            log_prob_norm,
                                            labels);
  } else {
    // full (or tied with d > 128): un-centered E-step. y = M_k @ X is formed by
    // one GEMM per component, then folded online as ||y - c_k||^2 with the
    // projected center c_k = M_k @ mu_k. On this GDDR7 card the path is
    // memory-bound; dropping the explicit centering (a full N×d write + read
    // per component) cuts the dominant HBM traffic ~1.67x.
    int prec_pc           = (ct == covariance_type::FULL) ? 1 : 0;
    cublasHandle_t cublas = raft::resource::get_cublas_handle(handle);
    cublas_check(cublasSetStream(cublas, stream), "cublasSetStream");
    rmm::device_uvector<T> y((size_t)n * d, stream);
    rmm::device_uvector<T> c((size_t)K * d, stream);
    rmm::device_uvector<T> rmax(n, stream);
    rmm::device_uvector<T> rsum(n, stream);
    rmm::device_uvector<T> best_lp(labels ? n : 0, stream);
    T one = T(1), zero = T(0);
    detail::fused_center_proj_kernel<T>
      <<<(int)(((size_t)K * d + threads - 1) / threads), threads, 0, stream>>>(
        precisions_chol, means, d, K, prec_pc, c.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    detail::fused_lse_init_kernel<T><<<rb, threads, 0, stream>>>(
      rmax.data(), rsum.data(), labels ? best_lp.data() : nullptr, labels, n);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    for (int k = 0; k < K; ++k) {
      const T* pc_k = precisions_chol + (size_t)(prec_pc ? k : 0) * d * d;
      cublas_check(
        cublas_gemm<T>(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, d, n, d, &one, pc_k, d, X, d, &zero, y.data(), d),
        "gemm(fused_score)");
      detail::fused_fold_from_y_kernel<T>
        <<<rb, threads, 0, stream>>>(y.data(),
                                     const_k.data(),
                                     c.data(),
                                     n,
                                     d,
                                     k,
                                     rmax.data(),
                                     rsum.data(),
                                     labels ? best_lp.data() : nullptr,
                                     labels);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    detail::fused_lse_finalize_kernel<T>
      <<<rb, threads, 0, stream>>>(rmax.data(), rsum.data(), n, log_prob_norm);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

template <typename T>
void predict_impl(raft::resources const& handle,
                  const params& params,
                  const T* X,
                  int n,
                  int d,
                  const T* weights,
                  const T* means,
                  const T* precisions_chol,
                  int* labels)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<T> lpn(n, stream);
  fused_score<T>(handle, params, X, n, d, weights, means, precisions_chol, lpn.data(), labels);
  raft::resource::sync_stream(handle);
}

template <typename T>
void predict_proba_impl(raft::resources const& handle,
                        const params& params,
                        const T* X,
                        int n,
                        int d,
                        const T* weights,
                        const T* means,
                        const T* precisions_chol,
                        T* resp)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<T> lpn(n, stream);
  infer<T>(handle, params, X, n, d, weights, means, precisions_chol, resp, lpn.data());
  raft::resource::sync_stream(handle);
}

template <typename T>
void score_samples_impl(raft::resources const& handle,
                        const params& params,
                        const T* X,
                        int n,
                        int d,
                        const T* weights,
                        const T* means,
                        const T* precisions_chol,
                        T* log_prob_norm)
{
  fused_score<T>(handle, params, X, n, d, weights, means, precisions_chol, log_prob_norm, nullptr);
  raft::resource::sync_stream(handle);
}

}  // namespace cuvs::cluster::gmm::detail
