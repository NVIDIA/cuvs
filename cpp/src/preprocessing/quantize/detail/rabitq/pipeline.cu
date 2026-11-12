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

#include <raft/core/error.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/linewise_op.cuh>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

namespace cuvs::preprocessing::quantize::rabitq::detail {

namespace {

void pad_rows(
  raft::resources const& res, const float* d_src, float* d_dst, size_t N, size_t dim, size_t D)
{
  raft::linalg::map_offset(res,
                           raft::make_device_vector_view<float, size_t>(d_dst, N * D),
                           [d_src, dim, D] __device__(size_t idx) {
                             const size_t row = idx / D;
                             const size_t col = idx % D;
                             return col < dim ? d_src[row * dim + col] : 0.0f;
                           });
}

/// Pad (if dim < rotator.padded_dim) and rotate src into dst. When padding
/// is needed, pads directly into dst and rotates in place if the rotator
/// supports it (fht_kac); otherwise stages through a temporary (matmul).
void pad_and_rotate(raft::resources const& res,
                    raft::device_matrix_view<const float, int64_t, raft::row_major> src,
                    rotator_view<const float> rotator,
                    raft::device_matrix_view<float, int64_t, raft::row_major> dst)
{
  if (src.extent(1) == rotator.padded_dim) {
    rotate(res, rotator, src, dst);
    return;
  }
  auto stream      = raft::resource::get_cuda_stream(res);
  const size_t N   = static_cast<size_t>(src.extent(0));
  const size_t dim = static_cast<size_t>(src.extent(1));
  const size_t D   = static_cast<size_t>(dst.extent(1));
  if (supports_inplace_rotate(rotator)) {
    pad_rows(res, src.data_handle(), dst.data_handle(), N, dim, D);
    rotate(res, rotator, raft::make_const_mdspan(dst), dst);
  } else {
    rmm::device_uvector<float> d_tmp(
      static_cast<size_t>(dst.size()), stream, raft::resource::get_workspace_resource_ref(res));
    auto tmp =
      raft::make_device_matrix_view<float, int64_t>(d_tmp.data(), dst.extent(0), dst.extent(1));
    pad_rows(res, src.data_handle(), tmp.data_handle(), N, dim, D);
    rotate(res, rotator, raft::make_const_mdspan(tmp), dst);
  }
}

/// residuals = rotate(pad(data)) [- rotate(pad(centroid))].
/// require_rotated_c: full-factor mode always needs the rotated-centroid
/// buffer (zero-filled when there is no centroid).
void prepare_residuals(raft::resources const& res,
                       raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                       rotator_view<const float> rotator,
                       std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                       raft::device_vector_view<float, int64_t> rotated_centroid,
                       raft::device_matrix_view<float, int64_t, raft::row_major> residuals,
                       bool require_rotated_c)
{
  const int64_t D = rotator.padded_dim;
  auto stream     = raft::resource::get_cuda_stream(res);
  pad_and_rotate(res, data, rotator, residuals);
  if (centroid.has_value()) {
    pad_and_rotate(
      res,
      raft::make_device_matrix_view<const float, int64_t>(
        centroid->data_handle(), 1, data.extent(1)),
      rotator,
      raft::make_device_matrix_view<float, int64_t>(rotated_centroid.data_handle(), 1, D));
    raft::matrix::linewise_op<raft::Apply::ALONG_ROWS>(
      res,
      raft::make_const_mdspan(residuals),
      residuals,
      raft::sub_op{},
      raft::make_device_vector_view<const float, int64_t>(rotated_centroid.data_handle(), D));
  } else if (require_rotated_c) {
    RAFT_CUDA_TRY(cudaMemsetAsync(
      rotated_centroid.data_handle(), 0, static_cast<size_t>(D) * sizeof(float), stream.get()));
  }
}

template <typename CodeT>
void quantize_data_impl(raft::resources const& res,
                        raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                        rotator_view<const float> rotator,
                        std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                        raft::device_vector_view<float, int64_t> rotated_centroid,
                        size_t ex_bits,
                        float const_scaling_factor,
                        bool use_fast,
                        raft::device_matrix_view<CodeT, int64_t, raft::row_major> codes,
                        raft::device_vector_view<float, int64_t> delta,
                        raft::device_vector_view<float, int64_t> vl,
                        raft::device_matrix_view<float, int64_t, raft::row_major> residuals,
                        int delta_mode,
                        int coarse_samples,
                        int fine_samples)
{
  prepare_residuals(res,
                    data,
                    rotator,
                    centroid,
                    rotated_centroid,
                    residuals,
                    /*require_rotated_c=*/false);
  quantize_fused_on_residuals(raft::resource::get_cuda_stream(res).get(),
                              raft::make_const_mdspan(residuals),
                              ex_bits,
                              const_scaling_factor,
                              use_fast,
                              codes,
                              delta,
                              vl,
                              delta_mode,
                              coarse_samples,
                              fine_samples);
}

template <typename CodeT>
void quantize_data_full_impl(raft::resources const& res,
                             raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                             rotator_view<const float> rotator,
                             std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                             raft::device_vector_view<float, int64_t> rotated_centroid,
                             size_t ex_bits,
                             float const_scaling_factor,
                             bool use_fast,
                             raft::device_matrix_view<CodeT, int64_t, raft::row_major> codes,
                             raft::device_matrix_view<float, int64_t, raft::row_major> factors,
                             raft::device_matrix_view<float, int64_t, raft::row_major> residuals,
                             int coarse_samples,
                             int fine_samples)
{
  prepare_residuals(res,
                    data,
                    rotator,
                    centroid,
                    rotated_centroid,
                    residuals,
                    /*require_rotated_c=*/true);
  quantize_full_on_residuals(raft::resource::get_cuda_stream(res).get(),
                             raft::make_const_mdspan(residuals),
                             raft::make_const_mdspan(rotated_centroid),
                             ex_bits,
                             const_scaling_factor,
                             use_fast,
                             codes,
                             factors,
                             coarse_samples,
                             fine_samples);
}

}  // namespace

void quantize_data(raft::resources const& res,
                   raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                   rotator_view<const float> rotator,
                   std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                   raft::device_vector_view<float, int64_t> rotated_centroid,
                   size_t ex_bits,
                   float const_scaling_factor,
                   bool use_fast,
                   raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
                   raft::device_vector_view<float, int64_t> delta,
                   raft::device_vector_view<float, int64_t> vl,
                   raft::device_matrix_view<float, int64_t, raft::row_major> residuals,
                   int delta_mode,
                   int coarse_samples,
                   int fine_samples)
{
  quantize_data_impl(res,
                     data,
                     rotator,
                     centroid,
                     rotated_centroid,
                     ex_bits,
                     const_scaling_factor,
                     use_fast,
                     codes,
                     delta,
                     vl,
                     residuals,
                     delta_mode,
                     coarse_samples,
                     fine_samples);
}

void quantize_data(raft::resources const& res,
                   raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                   rotator_view<const float> rotator,
                   std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                   raft::device_vector_view<float, int64_t> rotated_centroid,
                   size_t ex_bits,
                   float const_scaling_factor,
                   bool use_fast,
                   raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
                   raft::device_vector_view<float, int64_t> delta,
                   raft::device_vector_view<float, int64_t> vl,
                   raft::device_matrix_view<float, int64_t, raft::row_major> residuals,
                   int delta_mode,
                   int coarse_samples,
                   int fine_samples)
{
  quantize_data_impl(res,
                     data,
                     rotator,
                     centroid,
                     rotated_centroid,
                     ex_bits,
                     const_scaling_factor,
                     use_fast,
                     codes,
                     delta,
                     vl,
                     residuals,
                     delta_mode,
                     coarse_samples,
                     fine_samples);
}

void quantize_data_full(raft::resources const& res,
                        raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                        rotator_view<const float> rotator,
                        std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                        raft::device_vector_view<float, int64_t> rotated_centroid,
                        size_t ex_bits,
                        float const_scaling_factor,
                        bool use_fast,
                        raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
                        raft::device_matrix_view<float, int64_t, raft::row_major> factors,
                        raft::device_matrix_view<float, int64_t, raft::row_major> residuals,
                        int coarse_samples,
                        int fine_samples)
{
  quantize_data_full_impl(res,
                          data,
                          rotator,
                          centroid,
                          rotated_centroid,
                          ex_bits,
                          const_scaling_factor,
                          use_fast,
                          codes,
                          factors,
                          residuals,
                          coarse_samples,
                          fine_samples);
}

void quantize_data_full(raft::resources const& res,
                        raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                        rotator_view<const float> rotator,
                        std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                        raft::device_vector_view<float, int64_t> rotated_centroid,
                        size_t ex_bits,
                        float const_scaling_factor,
                        bool use_fast,
                        raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
                        raft::device_matrix_view<float, int64_t, raft::row_major> factors,
                        raft::device_matrix_view<float, int64_t, raft::row_major> residuals,
                        int coarse_samples,
                        int fine_samples)
{
  quantize_data_full_impl(res,
                          data,
                          rotator,
                          centroid,
                          rotated_centroid,
                          ex_bits,
                          const_scaling_factor,
                          use_fast,
                          codes,
                          factors,
                          residuals,
                          coarse_samples,
                          fine_samples);
}

}  // namespace cuvs::preprocessing::quantize::rabitq::detail
