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
#include <raft/core/operators.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/linewise_op.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

namespace cuvs::preprocessing::quantize::rabitq::detail {

namespace {

void pad_rows(raft::resources const& res, const float* d_src, float* d_dst,
              size_t N, size_t dim, size_t D) {
    raft::linalg::map_offset(res, raft::make_device_vector_view<float, size_t>(d_dst, N * D),
        [d_src, dim, D] __device__(size_t idx) {
            const size_t row = idx / D;
            const size_t col = idx % D;
            return col < dim ? d_src[row * dim + col] : 0.0f;
        });
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
    const size_t total = N * D;
    if (supports_inplace_rotate(rotator)) {
        pad_rows(res, d_src, d_dst, N, dim, D);
        rotate(res, rotator, d_dst, d_dst, static_cast<int64_t>(N));
    } else {
        rmm::device_uvector<float> d_tmp(
            total, stream, raft::resource::get_workspace_resource_ref(res));
        pad_rows(res, d_src, d_tmp.data(), N, dim, D);
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
        auto rows = raft::make_device_matrix_view<float, int64_t>(d_residuals, N, D);
        raft::matrix::linewise_op<raft::Apply::ALONG_ROWS>(
            res, raft::make_const_mdspan(rows), rows, raft::sub_op{},
            raft::make_device_vector_view<const float, int64_t>(d_rotated_centroid, D));
    } else if (require_rotated_c) {
        RAFT_CUDA_TRY(cudaMemsetAsync(d_rotated_centroid, 0, D * sizeof(float), stream));
    }
}

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
    quantize_fused_on_residuals(
        raft::resource::get_cuda_stream(res),
        d_residuals, N, static_cast<size_t>(rotator.padded_dim), ex_bits,
        const_scaling_factor, use_fast,
        d_total_code, d_delta, d_vl, delta_mode, coarse_samples, fine_samples);
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
    quantize_full_on_residuals(
        raft::resource::get_cuda_stream(res),
        d_residuals, d_rotated_centroid, N, static_cast<size_t>(rotator.padded_dim), ex_bits,
        const_scaling_factor, use_fast, d_total_code, d_factors);
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
