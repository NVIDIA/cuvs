/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Imported from the standalone RaBitQ GPU quantizer. Behaviour unchanged;
 * only include paths, error macros and the enclosing namespace differ.
 */

//
// RaBitQ pipeline — orchestration only. Composes the rotator launchers and the
// fused quantizer entry points; the only device code here is trivial glue
// (row padding, centroid subtraction).
//

#include "pipeline.cuh"
#include "quantize.cuh"

#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

namespace cuvs::preprocessing::quantize::rabitq::detail {

namespace {

__global__ void pad_rows_kernel(const float* __restrict__ d_src,
                                float* __restrict__ d_dst,
                                int N, int dim, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;
    int row = idx / D;
    int col = idx % D;
    d_dst[idx] = (col < dim) ? d_src[row * dim + col] : 0.0f;
}

__global__ void subtract_row_kernel(float* __restrict__ d_rows,
                                    const float* __restrict__ d_row,
                                    int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;
    d_rows[idx] -= d_row[idx % D];
}

/// Pad (if dim < rotator.padded_dim) and rotate d_src into d_dst. When padding
/// is needed, pads directly into d_dst and rotates in place if the rotator
/// supports it (fht_kac); otherwise stages through a temporary (matmul).
void pad_and_rotate(raft::resources const& res,
                    const float* d_src, size_t N, size_t dim,
                    rotator_ref const& rotator, float* d_dst) {
    const size_t D = static_cast<size_t>(rotator.padded_dim);
    if (dim == D) {
        rotate(res, rotator, d_src, d_dst, static_cast<int64_t>(N));
        return;
    }
    auto stream = raft::resource::get_cuda_stream(res);
    constexpr int kBlock = 256;
    const size_t total = N * D;
    const int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    if (supports_inplace_rotate(rotator)) {
        pad_rows_kernel<<<grid, kBlock, 0, stream>>>(d_src, d_dst,
                                                     static_cast<int>(N), static_cast<int>(dim),
                                                     static_cast<int>(D));
        RAFT_CUDA_TRY(cudaGetLastError());
        rotate(res, rotator, d_dst, d_dst, static_cast<int64_t>(N));
    } else {
        rmm::device_uvector<float> d_tmp(
            total, stream, raft::resource::get_workspace_resource_ref(res));
        pad_rows_kernel<<<grid, kBlock, 0, stream>>>(d_src, d_tmp.data(),
                                                     static_cast<int>(N), static_cast<int>(dim),
                                                     static_cast<int>(D));
        RAFT_CUDA_TRY(cudaGetLastError());
        rotate(res, rotator, d_tmp.data(), d_dst, static_cast<int64_t>(N));
    }
}

/// d_residuals = rotate(pad(data)) [- rotate(pad(centroid))].
/// require_rotated_c: full-factor mode always needs the rotated-centroid
/// buffer (zero-filled when there is no centroid).
void prepare_residuals(raft::resources const& res,
                       const float* d_data, size_t N, size_t dim,
                       rotator_ref const& rotator,
                       const float* d_centroid, float* d_rotated_centroid,
                       float* d_residuals, bool require_rotated_c) {
    const size_t D = static_cast<size_t>(rotator.padded_dim);
    auto stream    = raft::resource::get_cuda_stream(res);
    pad_and_rotate(res, d_data, N, dim, rotator, d_residuals);
    if (d_centroid != nullptr) {
        pad_and_rotate(res, d_centroid, 1, dim, rotator, d_rotated_centroid);
        constexpr int kBlock = 256;
        const size_t total = N * D;
        subtract_row_kernel<<<static_cast<int>((total + kBlock - 1) / kBlock), kBlock, 0, stream>>>(
            d_residuals, d_rotated_centroid, static_cast<int>(N), static_cast<int>(D));
        RAFT_CUDA_TRY(cudaGetLastError());
    } else if (require_rotated_c) {
        RAFT_CUDA_TRY(cudaMemsetAsync(d_rotated_centroid, 0, D * sizeof(float), stream));
    }
}

/// The residual stage above runs on the resource stream, while the imported
/// quantizer entry points still launch on the (per-thread) default stream.
/// Bridge the two with a host-side sync until quantize.cuh takes a stream.
/// TODO(rabitq): drop once quantize_*_on_residuals is stream-aware.
void sync_before_quantize(raft::resources const& res) { raft::resource::sync_stream(res); }

/// Mirror of sync_before_quantize: make the codes written by the quantizer
/// visible to work the caller subsequently issues on the resource stream.
/// TODO(rabitq): drop together with sync_before_quantize.
void sync_after_quantize() { RAFT_CUDA_TRY(cudaStreamSynchronize(0)); }

template <typename CodeT>
void quantize_data_impl(raft::resources const& res,
                        const float* d_data, size_t N, size_t dim,
                        rotator_ref const& rotator,
                        const float* d_centroid, float* d_rotated_centroid,
                        size_t ex_bits, float const_scaling_factor, bool use_fast,
                        CodeT* d_total_code, float* d_delta, float* d_vl,
                        float* d_residuals,
                        int delta_mode, int coarse_samples, int fine_samples) {
    prepare_residuals(res, d_data, N, dim, rotator, d_centroid, d_rotated_centroid,
                      d_residuals, /*require_rotated_c=*/false);
    sync_before_quantize(res);
    quantize_fused_on_residuals(
        d_residuals, N, static_cast<size_t>(rotator.padded_dim), ex_bits,
        const_scaling_factor, use_fast,
        d_total_code, d_delta, d_vl, delta_mode, coarse_samples, fine_samples);
    sync_after_quantize();
}

template <typename CodeT>
void quantize_data_full_impl(raft::resources const& res,
                             const float* d_data, size_t N, size_t dim,
                             rotator_ref const& rotator,
                             const float* d_centroid, float* d_rotated_centroid,
                             size_t ex_bits, float const_scaling_factor, bool use_fast,
                             CodeT* d_total_code, float* d_factors,
                             float* d_residuals) {
    prepare_residuals(res, d_data, N, dim, rotator, d_centroid, d_rotated_centroid,
                      d_residuals, /*require_rotated_c=*/true);
    sync_before_quantize(res);
    quantize_full_on_residuals(
        d_residuals, d_rotated_centroid, N, static_cast<size_t>(rotator.padded_dim), ex_bits,
        const_scaling_factor, use_fast, d_total_code, d_factors);
    sync_after_quantize();
}

}  // namespace

void quantize_data(
    raft::resources const& res,
    const float* d_data, size_t N, size_t dim,
    rotator_ref const& rotator,
    const float* d_centroid, float* d_rotated_centroid,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint16_t* d_total_code, float* d_delta, float* d_vl,
    float* d_residuals,
    int delta_mode, int coarse_samples, int fine_samples) {
    quantize_data_impl(res, d_data, N, dim, rotator, d_centroid, d_rotated_centroid,
                       ex_bits, const_scaling_factor, use_fast,
                       d_total_code, d_delta, d_vl, d_residuals,
                       delta_mode, coarse_samples, fine_samples);
}

void quantize_data(
    raft::resources const& res,
    const float* d_data, size_t N, size_t dim,
    rotator_ref const& rotator,
    const float* d_centroid, float* d_rotated_centroid,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint8_t* d_total_code, float* d_delta, float* d_vl,
    float* d_residuals,
    int delta_mode, int coarse_samples, int fine_samples) {
    quantize_data_impl(res, d_data, N, dim, rotator, d_centroid, d_rotated_centroid,
                       ex_bits, const_scaling_factor, use_fast,
                       d_total_code, d_delta, d_vl, d_residuals,
                       delta_mode, coarse_samples, fine_samples);
}

void quantize_data_full(
    raft::resources const& res,
    const float* d_data, size_t N, size_t dim,
    rotator_ref const& rotator,
    const float* d_centroid, float* d_rotated_centroid,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint16_t* d_total_code, float* d_factors,
    float* d_residuals) {
    quantize_data_full_impl(res, d_data, N, dim, rotator, d_centroid, d_rotated_centroid,
                            ex_bits, const_scaling_factor, use_fast,
                            d_total_code, d_factors, d_residuals);
}

void quantize_data_full(
    raft::resources const& res,
    const float* d_data, size_t N, size_t dim,
    rotator_ref const& rotator,
    const float* d_centroid, float* d_rotated_centroid,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint8_t* d_total_code, float* d_factors,
    float* d_residuals) {
    quantize_data_full_impl(res, d_data, N, dim, rotator, d_centroid, d_rotated_centroid,
                            ex_bits, const_scaling_factor, use_fast,
                            d_total_code, d_factors, d_residuals);
}

}  // namespace cuvs::preprocessing::quantize::rabitq::detail
