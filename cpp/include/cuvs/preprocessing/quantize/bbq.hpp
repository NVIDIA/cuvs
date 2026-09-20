/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>

#include <cstdint>

namespace CUVS_EXPORT cuvs {

namespace preprocessing::quantize::bbq {

/**
 * @defgroup bbq Better Binary Quantization utilities
 * @{
 */

/**
 * Storage layout of BBQ/OSQ quantized component codes in each dataset row.
 *
 */
enum class bbq_code_layout {
  packed_1b,     /** Each dimension is quantized to a single bit and packed into bytes. Reflects
                  * Lucene's OptimizedScalarQuantizer.packAsBinary. */
  transposed_2b, /** Each dimension is quantized to 2 bits, stored as 2 bitplanes.
                  * Reflects Lucene's OptimizedScalarQuantizer.transposeDibit. SIMT popc path only
                  * (paired with a transposed_4b or packed_1b operand); */
  transposed_4b, /** Each dimension is quantized to 4 bits, optimized for bitwise operations.
                  * Reflects Lucene's OptimizedScalarQuantizer.transposeHalfByte. the first bit of
                  * every dimension is in the first set dimensions bits, or (dimensions/8)
                  * bytes. The second, third, and fourth bits are in the second, third, and
                  * fourth set of dimensions bits, respectively. Format used for queries. */
  packed_4b,     /** Each dimension is quantized to 4 bits, two values are packed into each output
                  * byte. */
  packed_7b,     /** Each dimension is quantized to 7 bits and treated as a signed value. */
  packed_8b,     /** Each dimension is quantized to 8 bits and treated as an unsigned value. */

};

/**
 * Bit width of a layout.
 */
constexpr auto get_bit_width(bbq_code_layout layout) noexcept -> uint32_t
{
  switch (layout) {
    case bbq_code_layout::packed_1b: return 1;
    case bbq_code_layout::transposed_2b: return 2;
    case bbq_code_layout::transposed_4b:
    case bbq_code_layout::packed_4b: return 4;
    case bbq_code_layout::packed_7b: return 7;
    case bbq_code_layout::packed_8b: return 8;
  }
  return 0;
}

/** Bytes one row of @p dim components occupies once encoded in @p layout. */
constexpr auto get_encoded_row_length(uint32_t dim, bbq_code_layout layout) noexcept -> uint32_t
{
  switch (layout) {
    case bbq_code_layout::packed_1b: return raft::div_rounding_up_safe(dim, 8u);
    case bbq_code_layout::transposed_2b: return 2 * raft::div_rounding_up_safe(dim, 8u);
    case bbq_code_layout::transposed_4b: return 4 * raft::div_rounding_up_safe(dim, 8u);
    case bbq_code_layout::packed_4b: return raft::div_rounding_up_safe(dim, 2u);
    case bbq_code_layout::packed_7b: return dim;
    case bbq_code_layout::packed_8b: return dim;
  }
  return 0;
}

template <typename DataT, typename IdxT>
struct quantizer_view;

template <typename DataT, typename IdxT>
struct quantizer {
  raft::device_matrix<uint8_t, IdxT> codes;
  raft::device_vector<float, IdxT> lower_intervals;
  raft::device_vector<float, IdxT> upper_intervals;
  raft::device_vector<float, IdxT> additional_corrections;
  raft::device_vector<int32_t, IdxT> quantized_component_sums;
  raft::device_vector<DataT, IdxT> centroid;
  /** Precomputed per-row dequantization factors, derived once (offline) from lower/upper_intervals
   * and quantized_component_sums: dequant_delta = (upper-lower)/(2^bits-1) */
  raft::device_vector<float, IdxT> dequant_delta;
  /** Precomputed per-row dequantization factors, derived once (offline) from dequant_delta and
   * quantized_component_sums: dequant_sum_delta = dequant_delta * quantized_component_sums. */
  raft::device_vector<float, IdxT> dequant_sum_delta;
  /** Squared norm of the row in original (un-centered) vector space, ||x||^2 */
  raft::device_vector<float, IdxT> row_norm;

  bbq_code_layout layout{bbq_code_layout::packed_1b};
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};
  float centroid_norm_sq{};

  quantizer(raft::resources const& res,
            IdxT n_rows,
            uint32_t dim,
            bbq_code_layout layout,
            cuvs::distance::DistanceType metric)
    : codes{raft::make_device_matrix<uint8_t, IdxT>(
        res, n_rows, static_cast<IdxT>(get_encoded_row_length(dim, layout)))},
      lower_intervals{raft::make_device_vector<float, IdxT>(res, n_rows)},
      upper_intervals{raft::make_device_vector<float, IdxT>(res, n_rows)},
      additional_corrections{raft::make_device_vector<float, IdxT>(res, n_rows)},
      quantized_component_sums{raft::make_device_vector<int32_t, IdxT>(res, n_rows)},
      centroid{raft::make_device_vector<DataT, IdxT>(res, static_cast<IdxT>(dim))},
      dequant_delta{raft::make_device_vector<float, IdxT>(res, n_rows)},
      dequant_sum_delta{raft::make_device_vector<float, IdxT>(res, n_rows)},
      row_norm{raft::make_device_vector<float, IdxT>(res, n_rows)},
      layout{layout},
      metric{metric}
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return codes.extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(centroid.extent(0));
  }
  [[nodiscard]] constexpr auto encoded_row_length() const noexcept -> uint32_t
  {
    return get_encoded_row_length(dim(), layout);
  }
  [[nodiscard]] auto view() const noexcept -> quantizer_view<DataT, IdxT>
  {
    return quantizer_view<DataT, IdxT>{codes.view(),
                                       lower_intervals.view(),
                                       upper_intervals.view(),
                                       additional_corrections.view(),
                                       quantized_component_sums.view(),
                                       centroid.view(),
                                       dequant_delta.view(),
                                       dequant_sum_delta.view(),
                                       row_norm.view(),
                                       layout,
                                       metric,
                                       centroid_norm_sq};
  }
};

template <typename DataT, typename IdxT>
struct quantizer_view {
  raft::device_matrix_view<const uint8_t, IdxT> codes;
  raft::device_vector_view<const float, IdxT> lower_intervals;
  raft::device_vector_view<const float, IdxT> upper_intervals;
  raft::device_vector_view<const float, IdxT> additional_corrections;
  raft::device_vector_view<const int32_t, IdxT> quantized_component_sums;
  raft::device_vector_view<const DataT, IdxT> centroid;
  raft::device_vector_view<const float, IdxT> dequant_delta;
  raft::device_vector_view<const float, IdxT> dequant_sum_delta;
  raft::device_vector_view<const float, IdxT> row_norm;

  bbq_code_layout layout{bbq_code_layout::packed_1b};
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};
  float centroid_norm_sq{};

  quantizer_view(raft::device_matrix_view<const uint8_t, IdxT> codes_,
                 raft::device_vector_view<const float, IdxT> lower_intervals_,
                 raft::device_vector_view<const float, IdxT> upper_intervals_,
                 raft::device_vector_view<const float, IdxT> additional_corrections_,
                 raft::device_vector_view<const int32_t, IdxT> quantized_component_sums_,
                 raft::device_vector_view<const DataT, IdxT> centroid_,
                 raft::device_vector_view<const float, IdxT> dequant_delta_,
                 raft::device_vector_view<const float, IdxT> dequant_sum_delta_,
                 raft::device_vector_view<const float, IdxT> row_norm_,
                 bbq_code_layout layout_,
                 cuvs::distance::DistanceType metric_,
                 float centroid_norm_sq_) noexcept
    : codes{codes_},
      lower_intervals{lower_intervals_},
      upper_intervals{upper_intervals_},
      additional_corrections{additional_corrections_},
      quantized_component_sums{quantized_component_sums_},
      centroid{centroid_},
      dequant_delta{dequant_delta_},
      dequant_sum_delta{dequant_sum_delta_},
      row_norm{row_norm_},
      layout{layout_},
      metric{metric_},
      centroid_norm_sq{centroid_norm_sq_}
  {
  }

  [[nodiscard]] constexpr auto n_rows() const noexcept -> IdxT { return codes.extent(0); }
  [[nodiscard]] constexpr auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(centroid.extent(0));
  }
};

namespace helpers {
/**
 * Derives dequant_delta from lower/upper_intervals and the layout's code width, and
 * dequant_sum_delta from that delta and quantized_component_sums.
 */
void resolve_dequant_factors(
  raft::resources const& res,
  raft::device_vector_view<float, int64_t> dequant_delta,
  raft::device_vector_view<float, int64_t> dequant_sum_delta,
  raft::device_vector_view<const float, int64_t> lower_intervals,
  raft::device_vector_view<const float, int64_t> upper_intervals,
  raft::device_vector_view<const int32_t, int64_t> quantized_component_sums,
  bbq_code_layout layout);
}  // namespace helpers
/** @} */  // end of bbq group

}  // namespace preprocessing::quantize::bbq

}  // namespace CUVS_EXPORT cuvs
