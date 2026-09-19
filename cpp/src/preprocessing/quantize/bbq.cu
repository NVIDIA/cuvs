/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/preprocessing/quantize/bbq.hpp>
#include <optional>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>

namespace CUVS_EXPORT cuvs {

namespace preprocessing::quantize::bbq {

namespace helpers {
void resolve_dequant_factors(
  raft::resources const& res,
  raft::device_vector_view<float, int64_t> dequant_delta,
  raft::device_vector_view<float, int64_t> dequant_sum_delta,
  raft::device_vector_view<const float, int64_t> lower_intervals,
  raft::device_vector_view<const float, int64_t> upper_intervals,
  raft::device_vector_view<const int32_t, int64_t> quantized_component_sums,
  bbq_code_layout layout)
{
  const auto n_rows = dequant_delta.extent(0);
  RAFT_EXPECTS(dequant_sum_delta.extent(0) == n_rows && lower_intervals.extent(0) == n_rows &&
                 upper_intervals.extent(0) == n_rows &&
                 quantized_component_sums.extent(0) == n_rows,
               "resolve_dequant_factors: all vectors must have the same length");
  const float scale = 1.0f / static_cast<float>((uint32_t{1} << get_bit_width(layout)) - 1);
  auto* sum_delta   = dequant_sum_delta.data_handle();
  raft::linalg::map_offset(res,
                           dequant_delta,
                           [scale,
                            sum_delta,
                            lower = lower_intervals.data_handle(),
                            upper = upper_intervals.data_handle(),
                            sums  = quantized_component_sums.data_handle()] __device__(int64_t i) {
                             const float delta = (upper[i] - lower[i]) * scale;
                             sum_delta[i]      = delta * static_cast<float>(sums[i]);
                             return delta;
                           });
}
}  // namespace helpers
}  // namespace preprocessing::quantize::bbq
}  // namespace CUVS_EXPORT cuvs
