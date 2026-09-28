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
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

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

template <typename DataT, typename IdxT>
struct quantizer_view;

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
 * additional per-vector information so search is better than naïve sign-bit comparison. They differ
 * in transformation and scale representation. RaBitQ commonly separates residual magnitude from
 * direction, then applies a random orthogonal rotation before binary coding; BBQ uses per-vector
 * scalar intervals to interpret the compressed residual codes.
 */
template <typename DataT, typename IdxT>
struct quantizer {
  raft::device_matrix<uint8_t, IdxT> codes;
  raft::device_vector<float, IdxT> lower_intervals;
  raft::device_vector<float, IdxT> upper_intervals;
  raft::device_vector<float, IdxT> additional_corrections;
  raft::device_vector<int32_t, IdxT> quantized_component_sums;
  raft::device_vector<DataT, IdxT> centroid;
  /** Precomputed per-row dequantization factors, derived once (offline) from
   * lower/upper_intervals and quantized_component_sums: dequant_delta = (upper-lower)/(2^bits-1) */
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

namespace neighbors {
struct bbq_dataset_container {
  template <typename DataT, typename IdxT>
  using owning_storage = cuvs::preprocessing::quantize::bbq::quantizer<DataT, IdxT>;
  template <typename DataT, typename IdxT>
  using view_storage = cuvs::preprocessing::quantize::bbq::quantizer_view<DataT, IdxT>;
};

template <typename DataT, typename IdxT, typename Accessor>
struct dataset<bbq_dataset_container, DataT, IdxT, Accessor> {
  using owning_storage_type = bbq_dataset_container::owning_storage<DataT, IdxT>;
  std::vector<owning_storage_type> quantizers;

  dataset(owning_storage_type&& quantizer) noexcept { add_quantizer(std::move(quantizer)); }
  [[nodiscard]] auto as_dataset_view() const noexcept
    -> dataset_view<bbq_dataset_container,
                    DataT,
                    IdxT,
                    detail::dataset_view_accessor_for_owning<DataT, Accessor>>
  {
    return dataset_view<bbq_dataset_container,
                        DataT,
                        IdxT,
                        detail::dataset_view_accessor_for_owning<DataT, Accessor>>{quantizers};
  }
  [[nodiscard]] constexpr auto n_rows() const noexcept -> IdxT
  {
    return quantizers.size() > 0 ? quantizers[0].n_rows() : 0;
  }
  [[nodiscard]] constexpr auto dim() const noexcept -> uint32_t
  {
    return quantizers.size() > 0 ? quantizers[0].dim() : 0;
  }

  void add_quantizer(owning_storage_type&& quantizer)
  {
    RAFT_EXPECTS(!has_layout(quantizer.layout), "Quantizer already exists with layout.");
    quantizers.push_back(std::move(quantizer));
  }
  bool has_layout(cuvs::preprocessing::quantize::bbq::bbq_code_layout layout) const noexcept
  {
    for (uint32_t i = 0; i < quantizers.size(); i++) {
      if (quantizers[i].layout == layout) { return true; }
    }
    return false;
  }
};

template <typename DataT, typename IdxT, typename Accessor>
struct dataset_view<bbq_dataset_container, DataT, IdxT, Accessor> {
  using owning_storage_type = bbq_dataset_container::owning_storage<DataT, IdxT>;
  using view_storage_type   = bbq_dataset_container::view_storage<DataT, IdxT>;
  std::vector<view_storage_type> quantizers;

  dataset_view() noexcept = default;

  dataset_view(const std::vector<owning_storage_type>& quantizers) noexcept
  {
    for (const auto& quantizer : quantizers) {
      add_quantizer(quantizer);
    }
  }
  [[nodiscard]] constexpr auto n_rows() const noexcept -> IdxT
  {
    return quantizers.size() > 0 ? quantizers[0].n_rows() : 0;
  }
  [[nodiscard]] constexpr auto dim() const noexcept -> uint32_t
  {
    return quantizers.size() > 0 ? quantizers[0].dim() : 0;
  }

  void add_quantizer(view_storage_type quantizer)
  {
    RAFT_EXPECTS(!has_layout(quantizer.layout), "Quantizer already exists with layout.");
    quantizers.push_back(quantizer);
  }
  void add_quantizer(const owning_storage_type& quantizer)
  {
    RAFT_EXPECTS(!has_layout(quantizer.layout), "Quantizer already exists with layout.");
    quantizers.push_back(quantizer.view());
  }
  bool has_layout(cuvs::preprocessing::quantize::bbq::bbq_code_layout layout) const noexcept
  {
    for (uint32_t i = 0; i < quantizers.size(); i++) {
      if (quantizers[i].layout == layout) { return true; }
    }
    return false;
  }
  view_storage_type get_quantizer(cuvs::preprocessing::quantize::bbq::bbq_code_layout layout) const
  {
    for (uint32_t i = 0; i < quantizers.size(); i++) {
      if (quantizers[i].layout == layout) { return quantizers[i]; }
    }
    throw std::runtime_error("No quantizer found with layout.");
  }
};

template <typename DataT, typename IdxT>
using device_bbq_dataset =
  dataset<bbq_dataset_container, DataT, IdxT, detail::device_owning_accessor<DataT>>;

template <typename DataT, typename IdxT>
using device_bbq_dataset_view =
  dataset_view<bbq_dataset_container, DataT, IdxT, detail::device_view_accessor<DataT>>;

template <typename DataT, typename IdxT>
struct owning_dataset_for_view<device_bbq_dataset_view<DataT, IdxT>> {
  using type = device_bbq_dataset<DataT, IdxT>;
};

template <typename DatasetT>
struct is_bbq_dataset : std::false_type {};

template <typename DataT, typename IdxT, typename Accessor>
struct is_bbq_dataset<dataset<bbq_dataset_container, DataT, IdxT, Accessor>> : std::true_type {};

template <typename DatasetT>
inline constexpr bool is_bbq_dataset_v = is_bbq_dataset<DatasetT>::value;

template <typename DataT, typename IdxT, typename Accessor>
struct dataset_view_kind_of<dataset_view<bbq_dataset_container, DataT, IdxT, Accessor>> {
  static constexpr dataset_view_kind value = dataset_view_kind::bbq;
};

template <typename DataT, typename IdxT>
struct cagra_view_element_type<device_bbq_dataset_view<DataT, IdxT>> {
  using type = DataT;
};

template <typename V>
inline constexpr bool is_device_bbq_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::bbq && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_bbq_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::bbq && !dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_bbq_dataset_view_v =
  is_device_bbq_dataset_view_v<V> || is_host_bbq_dataset_view_v<V>;

}  // namespace neighbors

}  // namespace CUVS_EXPORT cuvs
