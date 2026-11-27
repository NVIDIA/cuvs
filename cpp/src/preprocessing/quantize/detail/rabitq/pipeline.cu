/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// RaBitQ pipeline — orchestration only. Composes the rotator launchers and the
// fused quantizer entry points, including centroid subtraction.
//

#include "pipeline.cuh"
#include "quantize.cuh"

#include <raft/core/error.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/linewise_op.cuh>
#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_runtime.h>

namespace cuvs::preprocessing::quantize::rabitq::detail {

namespace {

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
  rotate(res, rotator, data, residuals);
  if (centroid.has_value()) {
    rotate(res,
           rotator,
           raft::make_device_matrix_view<const float, int64_t>(
             centroid->data_handle(), 1, data.extent(1)),
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

}  // namespace

void quantize_data(raft::resources const& res,
                   raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                   rotator_view<const float> rotator,
                   std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                   raft::device_vector_view<float, int64_t> rotated_centroid,
                   size_t ex_bits,
                   float const_scaling_factor,
                   bool use_fast,
                   packed_codes_output codes,
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

void quantize_data_full(raft::resources const& res,
                        raft::device_matrix_view<const float, int64_t, raft::row_major> data,
                        rotator_view<const float> rotator,
                        std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                        raft::device_vector_view<float, int64_t> rotated_centroid,
                        size_t ex_bits,
                        float const_scaling_factor,
                        bool use_fast,
                        packed_codes_output codes,
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

}  // namespace cuvs::preprocessing::quantize::rabitq::detail
