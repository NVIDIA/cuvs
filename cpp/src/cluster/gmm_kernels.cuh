/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * CUDA kernels for the Gaussian mixture E- and M-steps, covering all four
 * covariance parameterizations (full, tied, diag, spherical).
 */

#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

namespace cuvs::cluster::gmm::detail {

constexpr float LOG_2PI_F  = 1.8378770664093453f;
constexpr double LOG_2PI_D = 1.8378770664093453;

template <typename T>
__device__ __forceinline__ T log_2pi_const();
template <>
__device__ __forceinline__ float log_2pi_const<float>()
{
  return LOG_2PI_F;
}
template <>
__device__ __forceinline__ double log_2pi_const<double>()
{
  return LOG_2PI_D;
}

__device__ __forceinline__ int upper_tri_col_offset(int col) { return (col * (col + 1)) / 2; }

// ===========================================================================
// E-step: log Gaussian probability
// ===========================================================================

// ---------------------------------------------------------------------------
// full / tied, d <= 64. ``prec_per_component`` is 1 for full (prec_chol has a
// d×d matrix per component) and 0 for tied (a single shared d×d matrix).
// ---------------------------------------------------------------------------
template <typename T, int D = 0>
__global__ void e_step_log_prob_small_kernel(const T* __restrict__ X,
                                             const T* __restrict__ weights,
                                             const T* __restrict__ means,
                                             const T* __restrict__ prec_chol,
                                             const T* __restrict__ log_det,
                                             int n,
                                             int d,
                                             int K,
                                             int prec_per_component,
                                             T* __restrict__ log_prob)
{
  static_assert(D >= 0 && D <= 64, "GMM small E-step supports runtime d or fixed D <= 64");
  constexpr bool fixed_d = D != 0;
  int dim                = fixed_d ? D : d;
  int k                  = blockIdx.y;
  int n_idx              = blockIdx.x * blockDim.x + threadIdx.x;
  int tid                = threadIdx.x;
  int kpc                = prec_per_component ? k : 0;

  extern __shared__ unsigned char smem_raw[];
  T* sh_mean = reinterpret_cast<T*>(smem_raw);
  T* sh_pc   = sh_mean + dim;

  for (int i = tid; i < dim; i += blockDim.x)
    sh_mean[i] = means[(size_t)k * dim + i];
  int pc_size_dense = dim * dim;
  for (int i = tid; i < pc_size_dense; i += blockDim.x) {
    int row = i / dim;
    int col = i - row * dim;
    if (row <= col) {
      sh_pc[upper_tri_col_offset(col) + row] = prec_chol[(size_t)kpc * pc_size_dense + i];
    }
  }

  __shared__ T sh_const;
  if (tid == 0) { sh_const = T(-0.5) * T(dim) * log_2pi_const<T>() + log_det[k] + log(weights[k]); }

  __syncthreads();

  if (n_idx >= n) return;

  T centered_vals[fixed_d ? D : 64];
  if constexpr (fixed_d) {
#pragma unroll
    for (int dd = 0; dd < D; ++dd)
      centered_vals[dd] = X[(size_t)n_idx * D + dd] - sh_mean[dd];
  } else {
    for (int dd = 0; dd < dim; ++dd)
      centered_vals[dd] = X[(size_t)n_idx * dim + dd] - sh_mean[dd];
  }

  T mahal = T(0);
  if constexpr (fixed_d) {
#pragma unroll
    for (int j = 0; j < D; ++j) {
      T y        = T(0);
      int pc_col = upper_tri_col_offset(j);
#pragma unroll
      for (int dd = 0; dd <= j; ++dd) {
        y += centered_vals[dd] * sh_pc[pc_col + dd];
      }
      mahal += y * y;
    }
  } else {
    for (int j = 0; j < dim; ++j) {
      T y        = T(0);
      int pc_col = upper_tri_col_offset(j);
      for (int dd = 0; dd <= j; ++dd) {
        y += centered_vals[dd] * sh_pc[pc_col + dd];
      }
      mahal += y * y;
    }
  }
  log_prob[(size_t)n_idx * K + k] = sh_const - T(0.5) * mahal;
}

// ---------------------------------------------------------------------------
// full / tied, d > 64, tiled over 64-column blocks.
// ---------------------------------------------------------------------------
template <typename T, int TILE_D>
__global__ void e_step_log_prob_large_d_thread64_kernel(const T* __restrict__ X,
                                                        const T* __restrict__ weights,
                                                        const T* __restrict__ means,
                                                        const T* __restrict__ prec_chol,
                                                        const T* __restrict__ log_det,
                                                        int n,
                                                        int d,
                                                        int K,
                                                        int prec_per_component,
                                                        T* __restrict__ log_prob)
{
  static_assert(TILE_D == 64, "GMM thread64 E-step expects a 64-column precision tile");

  int k   = blockIdx.y;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  int kpc = prec_per_component ? k : 0;

  extern __shared__ unsigned char smem_raw[];
  T* sh_mean = reinterpret_cast<T*>(smem_raw);  // (64,)
  T* sh_pc   = sh_mean + TILE_D;                // (64, 64)

  __shared__ T sh_const;
  if (tid == 0) { sh_const = T(-0.5) * T(d) * log_2pi_const<T>() + log_det[k] + log(weights[k]); }

  T local_mahal = T(0);
  const T* pc   = prec_chol + (size_t)kpc * d * d;

  for (int j_base = 0; j_base < d; j_base += TILE_D) {
    int cols_in_tile = min(TILE_D, d - j_base);
    int dd_limit     = min(d, j_base + TILE_D);
    T y[TILE_D];
#pragma unroll
    for (int col = 0; col < TILE_D; ++col)
      y[col] = T(0);

    for (int dd_base = 0; dd_base < dd_limit; dd_base += TILE_D) {
      int feats_in_tile = min(TILE_D, dd_limit - dd_base);

      for (int idx = tid; idx < TILE_D; idx += blockDim.x) {
        sh_mean[idx] = (idx < feats_in_tile) ? means[(size_t)k * d + dd_base + idx] : T(0);
      }

      constexpr int pc_tile_elems = TILE_D * TILE_D;
      for (int idx = tid; idx < pc_tile_elems; idx += blockDim.x) {
        int feat      = idx / TILE_D;
        int col_local = idx - feat * TILE_D;
        int dd        = dd_base + feat;
        int col       = j_base + col_local;
        T val         = T(0);
        if (feat < feats_in_tile && col_local < cols_in_tile && dd <= col) {
          val = pc[(size_t)dd * d + col];
        }
        sh_pc[feat * TILE_D + col_local] = val;
      }

      __syncthreads();

      if (row < n) {
#pragma unroll
        for (int feat = 0; feat < TILE_D; ++feat) {
          if (feat >= feats_in_tile) break;
          T diff = X[(size_t)row * d + dd_base + feat] - sh_mean[feat];
#pragma unroll
          for (int col = 0; col < TILE_D; ++col) {
            if (col >= cols_in_tile) break;
            y[col] += diff * sh_pc[feat * TILE_D + col];
          }
        }
      }

      __syncthreads();
    }

    if (row < n) {
#pragma unroll
      for (int col = 0; col < TILE_D; ++col) {
        if (col >= cols_in_tile) break;
        local_mahal += y[col] * y[col];
      }
    }
  }

  if (row < n) log_prob[(size_t)row * K + k] = sh_const - T(0.5) * local_mahal;
}

// ---------------------------------------------------------------------------
// cuBLAS E-step (full/tied, wide d): center one component's data, X - means[k].
// ---------------------------------------------------------------------------
template <typename T>
__global__ void e_step_center_kernel(const T* __restrict__ X,
                                     const T* __restrict__ means,
                                     int n,
                                     int d,
                                     int k,
                                     T* __restrict__ centered)
{
  size_t idx   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)n * d;
  if (idx >= total) return;
  int col       = idx % d;
  centered[idx] = X[idx] - means[(size_t)k * d + col];
}

// Projected centers c[k] = M_k @ mu_k for the un-centered E-step, where M_k is
// the (column-major, lda=d) prec-chol matrix used by the cuBLAS GEMM. Lets the
// fused fold compute ||M_k x - c_k||^2 without first materializing (X - mu_k).
template <typename T>
__global__ void fused_center_proj_kernel(const T* __restrict__ prec_chol,
                                         const T* __restrict__ means,
                                         int d,
                                         int K,
                                         int prec_pc,
                                         T* __restrict__ c)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)K * d) return;
  int k       = idx / d;
  int i       = idx % d;
  const T* M  = prec_chol + (size_t)(prec_pc ? k : 0) * d * d;
  const T* mu = means + (size_t)k * d;
  T acc       = T(0);
  for (int j = 0; j < d; ++j)
    acc += M[(size_t)i + (size_t)j * d] * mu[j];
  c[(size_t)k * d + i] = acc;
}

// cuBLAS E-step: from y = (X - means[k]) @ prec_chol[k] (n, d), compute
// log_prob[:, k] = const_k - 0.5 * sum_j y[:, j]^2 (Kahan-compensated).
template <typename T>
__global__ void e_step_log_prob_from_y_kernel(const T* __restrict__ y,
                                              const T* __restrict__ weights,
                                              const T* __restrict__ log_det,
                                              int n,
                                              int d,
                                              int K,
                                              int k,
                                              T* __restrict__ log_prob)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;

  T mahal        = T(0);
  T compensation = T(0);
  for (int col = 0; col < d; ++col) {
    T v          = y[(size_t)row * d + col];
    T term       = v * v - compensation;
    T next       = mahal + term;
    compensation = (next - mahal) - term;
    mahal        = next;
  }
  T constant = T(-0.5) * T(d) * log_2pi_const<T>() + log_det[k] + log(weights[k]);
  log_prob[(size_t)row * K + k] = constant - T(0.5) * mahal;
}

// ---------------------------------------------------------------------------
// Per-row log-sum-exp normalize: resp = exp(log_prob - logsumexp); also writes
// the per-sample log-likelihood (= logsumexp) into log_prob_norm.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void e_step_normalize_kernel(
  const T* __restrict__ log_prob, int n, int K, T* __restrict__ resp, T* __restrict__ log_prob_norm)
{
  int n_idx = blockIdx.x;
  if (n_idx >= n) return;
  int tid = threadIdx.x;
  // 64-bit row offset: n_idx * K overflows int32 once n * K exceeds 2^31.
  size_t row = (size_t)n_idx * K;

  __shared__ T sh_max;
  __shared__ T sh_sum;

  T local_max = -CUDART_INF_F;
  for (int k = tid; k < K; k += blockDim.x) {
    T v = log_prob[row + k];
    if (v > local_max) local_max = v;
  }
  for (int off = 16; off > 0; off >>= 1) {
    T other = __shfl_down_sync(0xffffffff, local_max, off);
    if (other > local_max) local_max = other;
  }
  if (tid == 0) sh_max = local_max;
  __syncthreads();
  T mx = sh_max;

  T local_sum = T(0);
  for (int k = tid; k < K; k += blockDim.x) {
    local_sum += exp(log_prob[row + k] - mx);
  }
  for (int off = 16; off > 0; off >>= 1)
    local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
  if (tid == 0) {
    sh_sum               = local_sum;
    log_prob_norm[n_idx] = log(local_sum) + mx;
  }
  __syncthreads();
  T log_total = log(sh_sum) + mx;

  for (int k = tid; k < K; k += blockDim.x) {
    resp[row + k] = exp(log_prob[row + k] - log_total);
  }
}

// ===========================================================================
// M-step
// ===========================================================================

// means[k] = num[k] / N_k[k]; weights[k] = N_k[k] / n.
template <typename T>
__global__ void m_step_finalize_means_kernel(const T* __restrict__ N_k,
                                             const T* __restrict__ num,
                                             T* __restrict__ weights,
                                             T* __restrict__ means,
                                             T eps,
                                             int n,
                                             int d,
                                             int K)
{
  int k   = blockIdx.x;
  int tid = threadIdx.x;
  if (k >= K) return;

  T Nk     = N_k[k] + T(10) * eps;
  T inv_Nk = T(1) / Nk;
  if (tid == 0) weights[k] = Nk / T(n);

  for (int i = tid; i < d; i += blockDim.x)
    means[(size_t)k * d + i] = num[(size_t)k * d + i] * inv_Nk;
}

// sqrt(resp) * (X - means[k]) for one component k. Output feeds a GEMM forming
// the (unnormalized) full covariance.
template <typename T>
__global__ void weighted_center_kernel(const T* __restrict__ X,
                                       const T* __restrict__ resp,
                                       const T* __restrict__ means,
                                       int n,
                                       int d,
                                       int K,
                                       int k,
                                       T* __restrict__ centered)
{
  size_t idx   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)n * d;
  if (idx >= total) return;

  size_t row = idx / d;
  int col    = idx - row * d;
  T r        = resp[row * K + k];
  // means == nullptr -> sqrt(r)*x (uncentered moment); else sqrt(r)*(x - mu_k).
  T mu          = means ? means[(size_t)k * d + col] : T(0);
  centered[idx] = sqrt(r) * (X[idx] - mu);
}

// Finalize one full covariance from the centered moment Σ r·(x-mu)(x-mu)ᵀ:
// divide by N_k, symmetrize the row-major result the column-major GEMM produced,
// and add reg_covar to the diagonal.
template <typename T>
__global__ void m_step_finalize_cov_full_kernel(
  const T* __restrict__ N_k, T* __restrict__ covariances, T reg_covar, T eps, int d, int K)
{
  int k   = blockIdx.x;
  int tid = threadIdx.x;
  if (k >= K) return;

  T Nk         = N_k[k] + T(10) * eps;
  T inv_Nk     = T(1) / Nk;
  size_t total = (size_t)d * d;
  T* cov       = covariances + (size_t)k * d * d;

  for (size_t idx = tid; idx < total; idx += blockDim.x) {
    size_t i = idx / d;
    size_t j = idx % d;
    if (i > j) continue;

    T v = cov[j * d + i] * inv_Nk;
    if (i == j) v += reg_covar;
    cov[i * d + j] = v;
    if (i != j) cov[j * d + i] = v;
  }
}

// Square every element of X into Xsq (used by the diagonal M-step GEMM).
template <typename T>
__global__ void elementwise_square_kernel(const T* __restrict__ X,
                                          size_t total,
                                          T* __restrict__ out)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  T v      = X[idx];
  out[idx] = v * v;
}

// diag covariance: var[k,dd] = num2[k,dd]/N_k[k] - means[k,dd]^2 + reg_covar,
// where num2 = respᵀ @ (X*X).
template <typename T>
__global__ void m_step_finalize_diag_kernel(const T* __restrict__ N_k,
                                            const T* __restrict__ num2,
                                            const T* __restrict__ means,
                                            T reg_covar,
                                            T eps,
                                            int d,
                                            int K,
                                            T* __restrict__ variances)
{
  int k   = blockIdx.x;
  int tid = threadIdx.x;
  if (k >= K) return;
  T Nk     = N_k[k] + T(10) * eps;
  T inv_Nk = T(1) / Nk;
  for (int dd = tid; dd < d; dd += blockDim.x) {
    T mu                          = means[(size_t)k * d + dd];
    variances[(size_t)k * d + dd] = num2[(size_t)k * d + dd] * inv_Nk - mu * mu + reg_covar;
  }
}

// spherical covariance: var[k] = mean over features of the diagonal variances.
template <typename T>
__global__ void m_step_spherical_from_diag_kernel(const T* __restrict__ diag_var,
                                                  int d,
                                                  int K,
                                                  T* __restrict__ variances)
{
  int k   = blockIdx.x;
  int tid = threadIdx.x;
  if (k >= K) return;

  __shared__ T sh[256];
  T local = T(0);
  for (int dd = tid; dd < d; dd += blockDim.x)
    local += diag_var[(size_t)k * d + dd];
  sh[tid] = local;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) variances[k] = sh[0] / T(d);
}

// Scale each component mean by N_k (for the tied covariance weighted outer
// product (N_k * means)ᵀ @ means).
template <typename T>
__global__ void scale_rows_by_kernel(
  const T* __restrict__ in, const T* __restrict__ scale, T eps, int d, int K, T* __restrict__ out)
{
  int k   = blockIdx.x;
  int tid = threadIdx.x;
  if (k >= K) return;
  T s = scale[k] + T(10) * eps;
  for (int dd = tid; dd < d; dd += blockDim.x)
    out[(size_t)k * d + dd] = in[(size_t)k * d + dd] * s;
}

// tied covariance: cov = (XtX - weighted_means_outer) / sum(N_k) + reg_covar*I.
template <typename T>
__global__ void m_step_finalize_tied_kernel(const T* __restrict__ XtX,
                                            const T* __restrict__ weighted_outer,
                                            T sum_Nk,
                                            T reg_covar,
                                            int d,
                                            T* __restrict__ covariance)
{
  size_t idx   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)d * d;
  if (idx >= total) return;
  size_t i = idx / d;
  size_t j = idx % d;
  if (i > j) return;  // upper triangle drives a symmetric write
  T inv = T(1) / sum_Nk;
  T v   = (XtX[idx] - weighted_outer[idx]) * inv;
  if (i == j) v += reg_covar;
  covariance[(size_t)i * d + j] = v;
  if (i != j) covariance[(size_t)j * d + i] = v;
}

// ===========================================================================
// Precision Cholesky helpers (diag / spherical) and log-determinants
// ===========================================================================

// diag/spherical precision Cholesky: 1 / sqrt(variance), element-wise.
template <typename T>
__global__ void recip_sqrt_kernel(const T* __restrict__ var, size_t total, T* __restrict__ out)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  out[idx] = T(1) / sqrt(var[idx]);
}

// log-det of the precision Cholesky for full covariance:
// log_det[k] = sum_i log(prec_chol[k,i,i]).
template <typename T>
__global__ void log_det_full_kernel(const T* __restrict__ prec_chol,
                                    int d,
                                    int K,
                                    T* __restrict__ log_det)
{
  int k   = blockIdx.x;
  int tid = threadIdx.x;
  if (k >= K) return;
  __shared__ T sh[256];
  T local       = T(0);
  const T* pc_k = prec_chol + (size_t)k * d * d;
  for (int i = tid; i < d; i += blockDim.x)
    local += log(pc_k[(size_t)i * d + i]);
  sh[tid] = local;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) log_det[k] = sh[0];
}

// log-det for tied covariance: a single value broadcast to all K components.
template <typename T>
__global__ void log_det_tied_kernel(const T* __restrict__ prec_chol,
                                    int d,
                                    int K,
                                    T* __restrict__ log_det)
{
  int tid = threadIdx.x;
  __shared__ T sh[256];
  T local = T(0);
  for (int i = tid; i < d; i += blockDim.x)
    local += log(prec_chol[(size_t)i * d + i]);
  sh[tid] = local;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) {
    for (int k = 0; k < K; ++k)
      log_det[k] = sh[0];
  }
}

// log-det for diagonal covariance: log_det[k] = sum_dd log(prec_chol[k,dd]).
template <typename T>
__global__ void log_det_diag_kernel(const T* __restrict__ prec_chol,
                                    int d,
                                    int K,
                                    T* __restrict__ log_det)
{
  int k   = blockIdx.x;
  int tid = threadIdx.x;
  if (k >= K) return;
  __shared__ T sh[256];
  T local = T(0);
  for (int dd = tid; dd < d; dd += blockDim.x)
    local += log(prec_chol[(size_t)k * d + dd]);
  sh[tid] = local;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) log_det[k] = sh[0];
}

// log-det for spherical covariance: log_det[k] = d * log(prec_chol[k]).
template <typename T>
__global__ void log_det_spherical_kernel(const T* __restrict__ prec_chol,
                                         int d,
                                         int K,
                                         T* __restrict__ log_det)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  log_det[k] = T(d) * log(prec_chol[k]);
}

// ===========================================================================
// Miscellaneous
// ===========================================================================

// Identity matrix (d×d), row-major.
template <typename T>
__global__ void set_identity_kernel(T* __restrict__ A, int d)
{
  size_t idx   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)d * d;
  if (idx >= total) return;
  size_t i = idx / d;
  size_t j = idx % d;
  A[idx]   = (i == j) ? T(1) : T(0);
}

// Batched identity: K stacked d×d identity matrices, row-major (K, d, d).
template <typename T>
__global__ void set_identity_batched_kernel(T* __restrict__ A, int d, int K)
{
  size_t idx   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)K * d * d;
  if (idx >= total) return;
  size_t within = idx % ((size_t)d * d);
  int i         = (int)(within / d);
  int j         = (int)(within % d);
  A[idx]        = (i == j) ? T(1) : T(0);
}

// Argmax over components: labels[n] = argmax_k resp[n,k].
template <typename T>
__global__ void argmax_kernel(const T* __restrict__ resp, int n, int K, int* __restrict__ labels)
{
  int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_idx >= n) return;
  const T* r = resp + (size_t)n_idx * K;
  int best   = 0;
  T best_v   = r[0];
  for (int k = 1; k < K; ++k) {
    if (r[k] > best_v) {
      best_v = r[k];
      best   = k;
    }
  }
  labels[n_idx] = best;
}

// Scatter hard labels into one-hot responsibilities (n×K).
template <typename T>
__global__ void labels_to_onehot_kernel(const int* __restrict__ labels,
                                        int n,
                                        int K,
                                        T* __restrict__ resp)
{
  int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_idx >= n) return;
  int lab = labels[n_idx];
  for (int k = 0; k < K; ++k)
    resp[(size_t)n_idx * K + k] = (k == lab) ? T(1) : T(0);
}

// Tile variant: scatter the one-hot for any chosen index falling in [t0, t0+nt)
// into a tile-local (nt, K) buffer (memset to 0 by the caller).
template <typename T>
__global__ void scatter_onehot_tile_kernel(
  const int* __restrict__ indices, int K, int t0, int nt, T* __restrict__ resp_tile)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  int s = indices[k] - t0;
  if (s >= 0 && s < nt) resp_tile[(size_t)s * K + k] = T(1);
}

// Normalize each row of resp to sum to one (used by 'random' init).
template <typename T>
__global__ void normalize_rows_kernel(T* __restrict__ resp, int n, int K)
{
  int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_idx >= n) return;
  T* r  = resp + (size_t)n_idx * K;
  T sum = T(0);
  for (int k = 0; k < K; ++k)
    sum += r[k];
  T inv = T(1) / sum;
  for (int k = 0; k < K; ++k)
    r[k] *= inv;
}

// Fast tiled fused E-step (diag / spherical): thread-per-sample with the
// sample's feature vector held in registers and component means (and, for
// diag, reciprocal std-devs) streamed through shared memory in tiles of CELL
// components. Online log-sum-exp -> log_prob_norm (+ optional argmax label),
// no N×K materialization. ``MAXD`` is the register footprint for the feature
// vector (caller picks the smallest bucket >= d); ``DIAG`` selects the
// covariance form at compile time so the shared buffers are sized exactly.
// const_k[k] = -0.5 d log(2pi) + log_det[k] + log(weights[k]), precomputed once
// per component so the hot E-step loop never calls log() per (sample, k).
template <typename T>
__global__ void fused_const_kernel(const T* __restrict__ weights,
                                   const T* __restrict__ log_det,
                                   int d,
                                   int K,
                                   T* __restrict__ const_k)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  const_k[k] = T(-0.5) * T(d) * log_2pi_const<T>() + log_det[k] + log(weights[k]);
}

// Initialize the running (max, sum) accumulators for the cuBLAS fold path.
// The max starts at a finite sentinel (like estep_tiled_kernel), not -inf: a
// -inf first component (zero weight -> log(0)) would make the fold compute
// exp(-inf - -inf) = NaN and poison the whole row. Trade-off (shared with the
// tiled kernel's sentinel): a row whose log-probs are all finite but below
// -1e30 (~1e15 sigma from every mean, density 0 in any float format) scores
// -inf with label 0 instead of its true argmax.
template <typename T>
__global__ void fused_lse_init_kernel(T* __restrict__ rmax,
                                      T* __restrict__ rsum,
                                      T* __restrict__ best_lp,
                                      int* __restrict__ labels,
                                      int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  rmax[i] = T(-1e30);
  rsum[i] = T(0);
  if (best_lp) best_lp[i] = T(-1e30);
  if (labels) labels[i] = 0;
}

// Fold component k (full/tied) into the running log-sum-exp from
// y = (x - mu_k) @ prec_chol_k, with Kahan-compensated ‖y‖² (matches the
// standard cuBLAS E-step). Optionally tracks the argmax component.
template <typename T>
__global__ void fused_fold_from_y_kernel(const T* __restrict__ y,
                                         const T* __restrict__ const_k,
                                         const T* __restrict__ c,
                                         int n,
                                         int d,
                                         int k,
                                         T* __restrict__ rmax,
                                         T* __restrict__ rsum,
                                         T* __restrict__ best_lp,
                                         int* __restrict__ labels)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  // y = M_k @ x (un-centered). mahal = ||M_k(x - mu_k)||^2 = ||y - c_k||^2
  // with the projected center c_k = M_k @ mu_k precomputed once per component.
  const T* ck = c + (size_t)k * d;
  T mahal = T(0), comp = T(0);
  for (int col = 0; col < d; ++col) {
    T v    = y[(size_t)row * d + col] - ck[col];
    T term = v * v - comp;
    T next = mahal + term;
    comp   = (next - mahal) - term;
    mahal  = next;
  }
  T lp = const_k[k] - T(0.5) * mahal;
  if (labels && lp > best_lp[row]) {
    best_lp[row] = lp;
    labels[row]  = k;
  }
  T pm      = rmax[row];
  T nm      = fmax(pm, lp);
  rsum[row] = rsum[row] * exp(pm - nm) + exp(lp - nm);
  rmax[row] = nm;
}

// log_prob_norm[n] = rmax[n] + log(rsum[n]).
template <typename T>
__global__ void fused_lse_finalize_kernel(const T* __restrict__ rmax,
                                          const T* __restrict__ rsum,
                                          int n,
                                          T* __restrict__ log_prob_norm)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  log_prob_norm[i] = rmax[i] + log(rsum[i]);
}

// KDE-style register-tiled E-step for spherical / diag covariances: one thread
// per sample, components tiled in CELL register accumulators, features streamed
// through shared memory in FEAT-wide tiles. The (N, K) distance matrix is never
// materialized — distances fold straight into an online log-sum-exp. Holding
// only CELL accumulators (not all of x) keeps register pressure low -> high
// occupancy (mirrors cuvs::distance::kde's tiling, specialized per component).
// DIAG=false: spherical (scalar prec_chol[k]); DIAG=true: diag (prec_chol[k,:]).
// WRITE_FULL=false: online log-sum-exp -> log_prob_norm (+ argmax labels), no
// N×K. WRITE_FULL=true: also write the full per-component log_prob[n,k] (for the
// materialized fit / predict_proba path, replacing the slow per-mode kernels);
// log_prob_norm/labels are then left to the normalize kernel.
template <typename T, int CELL, int FEAT, bool DIAG, bool WRITE_FULL>
__global__ void estep_tiled_kernel(const T* __restrict__ X,
                                   const T* __restrict__ means,
                                   const T* __restrict__ prec_chol,
                                   const T* __restrict__ const_k,
                                   int n,
                                   int d,
                                   int K,
                                   T* __restrict__ log_prob_norm,
                                   int* __restrict__ labels,
                                   T* __restrict__ log_prob)
{
  int i      = blockIdx.x * blockDim.x + threadIdx.x;
  bool valid = i < n;
  __shared__ T smu[FEAT * CELL];
  __shared__ T spc[DIAG ? FEAT * CELL : 1];
  T rmax = T(-1e30), rsum = T(0), best_lp = T(-1e30);
  int best = 0;
  for (int k0 = 0; k0 < K; k0 += CELL) {
    int cells = min(CELL, K - k0);
    T acc[CELL];
#pragma unroll
    for (int c = 0; c < CELL; ++c)
      acc[c] = T(0);
    for (int f0 = 0; f0 < d; f0 += FEAT) {
      int feats = min(FEAT, d - f0);
      for (int idx = threadIdx.x; idx < FEAT * CELL; idx += blockDim.x) {
        int cell = idx / FEAT, feat = idx % FEAT;
        bool in                 = (cell < cells && feat < feats);
        size_t g                = (size_t)(k0 + cell) * d + f0 + feat;
        smu[feat * CELL + cell] = in ? means[g] : T(0);
        if constexpr (DIAG) spc[feat * CELL + cell] = in ? prec_chol[g] : T(0);
      }
      __syncthreads();
      if (valid) {
        for (int f = 0; f < feats; ++f) {
          T xq = X[(size_t)i * d + f0 + f];
#pragma unroll
          for (int c = 0; c < CELL; ++c) {
            T df = xq - smu[f * CELL + c];
            if constexpr (DIAG) {
              T y = df * spc[f * CELL + c];
              acc[c] += y * y;
            } else {
              acc[c] += df * df;
            }
          }
        }
      }
      __syncthreads();
    }
    if (valid) {
#pragma unroll
      for (int c = 0; c < CELL; ++c) {
        if (c >= cells) break;
        int k = k0 + c;
        T mahal;
        if constexpr (DIAG) {
          mahal = acc[c];
        } else {
          T pc  = prec_chol[k];
          mahal = (pc * pc) * acc[c];
        }
        T lp = const_k[k] - T(0.5) * mahal;
        if constexpr (WRITE_FULL) {
          log_prob[(size_t)i * K + k] = lp;
        } else {
          if (labels && lp > best_lp) {
            best_lp = lp;
            best    = k;
          }
          T nm = fmax(rmax, lp);
          rsum = rsum * exp(rmax - nm) + exp(lp - nm);
          rmax = nm;
        }
      }
    }
  }
  if (valid && !WRITE_FULL) {
    log_prob_norm[i] = rmax + log(rsum);
    if (labels) labels[i] = best;
  }
}

}  // namespace cuvs::cluster::gmm::detail
