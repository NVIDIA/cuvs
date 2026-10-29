/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace CUVS_EXPORT cuvs {

namespace preprocessing::quantize::bbq {

/**
 * @defgroup bbq Better Binary Quantization utilities
 * @{
 */

/**
 * Storage layout of BBQ/OSQ quantized component codes in each dataset row.
 * packed_1b: Each dimension is quantized to a single bit and packed into bytes. Reflects
 * Lucene's OptimizedScalarQuantizer.packAsBinary.
 * transposed_2b: Each dimension is quantized to 2 bits, stored as 2 bitplanes.
 * Reflects Lucene's OptimizedScalarQuantizer.transposeDibit. SIMT popc path only
 * (paired with a transposed_4b or packed_1b operand);
 * transposed_4b: Each dimension is quantized to 4 bits, optimized for bitwise operations.
 * Reflects Lucene's OptimizedScalarQuantizer.transposeHalfByte. the first bit of
 * every dimension is in the first set dimensions bits, or (dimensions/8)
 * bytes. The second, third, and fourth bits are in the second, third, and
 * fourth set of dimensions bits, respectively. Format used for queries.
 * packed_4b: Each dimension is quantized to 4 bits, two values are packed into each output
 * byte.
 * packed_7b: Each dimension is quantized to 7 bits and treated as a signed value.
 * packed_8b: Each dimension is quantized to 8 bits and treated as an unsigned value.
 */
enum class bbq_code_layout {
  packed_1b,
  transposed_2b,
  transposed_4b,
  packed_4b,
  packed_7b,
  packed_8b,
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

/**
 * @brief Non-owning view of a BBQ quantizer, with the constness semantics of an mdspan:
 * `quantizer_view<T>` can modify the centroid, `quantizer_view<const T>` cannot.
 *
 * @tparam T the element type of the vectors the codes were made of, possibly const
 */
template <typename T>
struct quantizer_view {
  /** The centroid the vectors were centered around [dim]. */
  raft::device_vector_view<T, int64_t> centroid;
  /** The layout of the codes. */
  bbq_code_layout layout{bbq_code_layout::packed_1b};
  /** The metric the corrections were computed for. */
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};
  /** The squared norm of the centroid. */
  float centroid_norm_sq{};

  quantizer_view() = default;

  quantizer_view(raft::device_vector_view<T, int64_t> centroid_,
                 bbq_code_layout layout_,
                 cuvs::distance::DistanceType metric_,
                 float centroid_norm_sq_) noexcept
    : centroid{centroid_}, layout{layout_}, metric{metric_}, centroid_norm_sq{centroid_norm_sq_}
  {
  }

  /** The number of dimensions of the quantized vectors. */
  [[nodiscard]] constexpr auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(centroid.extent(0));
  }

  /** The bytes one encoded row occupies. */
  [[nodiscard]] constexpr auto encoded_row_length() const noexcept -> uint32_t
  {
    return get_encoded_row_length(dim(), layout);
  }

  /** A mutable view decays to the read-only one, the way an mdspan does. */
  template <typename U>
  operator quantizer_view<U>() const noexcept  // NOLINT(google-explicit-constructor)
    requires(!std::is_const_v<T> && std::is_same_v<U, const T>)
  {
    return quantizer_view<U>{raft::make_const_mdspan(centroid), layout, metric, centroid_norm_sq};
  }
};

/**
 * @brief Better Binary Quantization
 * ([BBQ](https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch))
 * is a vector-quantization approach used in Elasticsearch and Apache Lucene. It builds on ideas
 * introduced in RaBitQ([Gao and Long](https://arxiv.org/pdf/2405.12497, [Gao et
 * al.](https://arxiv.org/pdf/2409.09913)): residual binary codes around a centroid, corrective
 * factors, and efficient bitwise comparison of codes at different bit widths. Lucene implements
 * this as optimized scalar quantization (OSQ) with packed and bit-plane layouts; Elasticsearch
 * exposes it as BBQ.
 *
 * BBQ in cuVS designed to be compatible with the Lucene/Elasticsearch dataset: a single shared
 * centroid, no random rotation, and OSQ codes.
 *
 * RaBitQ and BBQ in cuVS both compress centroid-relative vectors to low-bit codes and retain
 * additional per-vector information so search is better than naive sign-bit comparison. They differ
 * in transformation and scale representation. RaBitQ commonly separates residual magnitude from
 * direction, then applies a random orthogonal rotation before binary coding; BBQ uses per-vector
 * scalar intervals to interpret the compressed residual codes.
 *
 * The quantizer is the manifest of the quantized dataset `cuvs::device_bbq_dataset` (see
 * cuvs/core/dataset/bbq.hpp): the centroid the vectors were centered around, the layout of the
 * codes, the metric the corrections were computed for, and the cached squared norm of the
 * centroid. The encoded rows and their per-row terms are the data of the dataset.
 *
 * @tparam T the element type of the vectors the codes were made of
 */
template <typename T>
struct quantizer {
  static_assert(!std::is_const_v<T>, "An owning quantizer cannot hold const elements.");

  using view_type       = quantizer_view<T>;
  using const_view_type = quantizer_view<const T>;

  /** The centroid the vectors were centered around [dim]. */
  raft::device_vector<T, int64_t> centroid;
  /** The layout of the codes. */
  bbq_code_layout layout;
  /** The metric the corrections were computed for. */
  cuvs::distance::DistanceType metric;
  /** The squared norm of the centroid. */
  float centroid_norm_sq;

  /** The quantizer owning the given centroid; its dimension is the extent of the vector. */
  quantizer(raft::device_vector<T, int64_t>&& centroid_,
            bbq_code_layout layout_,
            cuvs::distance::DistanceType metric_,
            float centroid_norm_sq_)
    : centroid{std::move(centroid_)},
      layout{layout_},
      metric{metric_},
      centroid_norm_sq{centroid_norm_sq_}
  {
  }

  /** The number of dimensions of the quantized vectors. */
  [[nodiscard]] constexpr auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(centroid.extent(0));
  }

  /** The bytes one encoded row occupies. */
  [[nodiscard]] constexpr auto encoded_row_length() const noexcept -> uint32_t
  {
    return get_encoded_row_length(dim(), layout);
  }

  /** A non-owning view of the quantizer. */
  [[nodiscard]] auto view() noexcept -> view_type
  {
    return view_type{centroid.view(), layout, metric, centroid_norm_sq};
  }

  /** A read-only non-owning view of the quantizer. */
  [[nodiscard]] auto view() const noexcept -> const_view_type
  {
    return const_view_type{centroid.view(), layout, metric, centroid_norm_sq};
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
