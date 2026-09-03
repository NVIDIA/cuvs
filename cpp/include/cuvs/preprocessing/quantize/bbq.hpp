/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdspan.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace CUVS_EXPORT cuvs {

namespace preprocessing::quantize::bbq {

/**
 * @defgroup bbq Better Binary Quantization utilities
 * @{
 */

/**
 * Layout of BBQ quantized codes in each dataset row.
 */
enum class bbq_code_layout {
  single_bit, /** Each dimension is quantized to a single bit and packed into bytes. Reflects
               * OptimizedScalarQuantizer.packAsBinary. */
  dibit,      /** Each dimension is quantized to 2 bits (dibit) and transposed for bitwise operations.
               * Same principle as transpose_half_byte, but for 2 bits.
               * Reflects OptimizedScalarQuantizer.transposeDibit. */
  transpose_half_byte, /** Each dimension is quantized to 4 bits, optimized for bitwise operations.
                        * Reflects OptimizedScalarQuantizer.transposeHalfByte. the first bit of
                        * every dimension is in the first set dimensions bits, or (dimensions/8)
                        * bytes. The second, third, and fourth bits are in the second, third, and
                        * fourth set of dimensions bits, respectively. Format used for queries. */
  packed_nibble, /** Each dimension is quantized to 4 bits, two values are packed into each output
                  * byte. Reflects OffHeapScalarQuantizedVectorValues.packNibbles. */
  seven_bit,     /** Each dimension is quantized to 7 bits and treated as a signed value. */
  unsigned_byte, /** Each dimension is quantized to 8 bits and treated as an unsigned value. */

};

/** Owning structure for BBQ quantizer data. */
template <typename DataT, typename IdxT, typename Accessor>
struct bbq_quantizer {
  template <typename T, typename IdT, typename Acc>
  using dense_owning_matrix = cuvs::neighbors::detail::dense_owning_matrix<T, IdT, Acc>;
  template <typename T, typename IdT, typename Acc>
  using dense_owning_vector = cuvs::neighbors::detail::dense_owning_vector<T, IdT, Acc>;
  dense_owning_matrix<uint8_t, IdxT, Accessor> codes;
  dense_owning_vector<float, IdxT, Accessor> lower_intervals;
  dense_owning_vector<float, IdxT, Accessor> upper_intervals;
  dense_owning_vector<float, IdxT, Accessor> additional_corrections;
  dense_owning_vector<int32_t, IdxT, Accessor> quantized_component_sums;
  dense_owning_vector<DataT, IdxT, Accessor> centroid;

  uint32_t bits{};
  bbq_code_layout layout{bbq_code_layout::single_bit};
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};
  float centroid_norm_sq{};

  bbq_quantizer(dense_owning_matrix<uint8_t, IdxT, Accessor>&& codes,
                dense_owning_vector<float, IdxT, Accessor>&& lower_intervals,
                dense_owning_vector<float, IdxT, Accessor>&& upper_intervals,
                dense_owning_vector<float, IdxT, Accessor>&& additional_corrections,
                dense_owning_vector<int32_t, IdxT, Accessor>&& quantized_component_sums,
                dense_owning_vector<DataT, IdxT, Accessor>&& centroid,
                uint32_t bits,
                bbq_code_layout layout,
                cuvs::distance::DistanceType metric,
                float centroid_norm_sq) noexcept
    : codes{std::move(codes)},
      lower_intervals{std::move(lower_intervals)},
      upper_intervals{std::move(upper_intervals)},
      additional_corrections{std::move(additional_corrections)},
      quantized_component_sums{std::move(quantized_component_sums)},
      centroid{std::move(centroid)},
      bits{bits},
      layout{layout},
      metric{metric},
      centroid_norm_sq{centroid_norm_sq}
  {
    const auto n_rows = static_cast<int64_t>(codes.extent(0));
    RAFT_EXPECTS(bits >= 1 && bits <= 8, "BBQ bits must be in [1, 8].");
    RAFT_EXPECTS(codes.extent(1) == static_cast<int64_t>(encoded_row_length()),
                 "BBQ code row length does not match dim, bits, and layout.");
    RAFT_EXPECTS(lower_intervals.extent(0) == n_rows && upper_intervals.extent(0) == n_rows &&
                   additional_corrections.extent(0) == n_rows &&
                   quantized_component_sums.extent(0) == n_rows,
                 "Every BBQ correction array must contain one value per row.");
  }

  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return codes.extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(centroid.extent(0));
  }
  [[nodiscard]] constexpr auto encoded_row_length() const noexcept -> uint32_t
  {
    auto const d = dim();
    switch (layout) {
      case bbq_code_layout::single_bit: return (d * bits + 7) / 8;
      case bbq_code_layout::dibit: return bits * ((d + 7) / 8);
      case bbq_code_layout::packed_nibble: return (d + 1) / 2;
      case bbq_code_layout::seven_bit: return d;
      case bbq_code_layout::unsigned_byte: return d;
      case bbq_code_layout::transpose_half_byte: return 4 * ((d + 7) / 8);
    }
    return 0;
  }
};

/** View structure for BBQ quantizer data. */
template <typename DataT, typename IdxT, typename Accessor>
struct bbq_quantizer_view {
  using owning_accessor =
    cuvs::neighbors::detail::dataset_owning_accessor_for_view<DataT, Accessor>;
  using owning_storage = bbq_quantizer<DataT, IdxT, owning_accessor>;
  template <typename T, typename IdT, typename Acc>
  using dense_view_matrix = cuvs::neighbors::detail::dense_view_matrix<T, IdT, Acc>;
  template <typename T, typename IdT, typename Acc>
  using dense_view_vector = cuvs::neighbors::detail::dense_view_vector<T, IdT, Acc>;
  dense_view_matrix<const uint8_t, IdxT, Accessor> codes;
  dense_view_vector<const float, IdxT, Accessor> lower_intervals;
  dense_view_vector<const float, IdxT, Accessor> upper_intervals;
  dense_view_vector<const float, IdxT, Accessor> additional_corrections;
  dense_view_vector<const int32_t, IdxT, Accessor> quantized_component_sums;
  dense_view_vector<const DataT, IdxT, Accessor> centroid;

  uint32_t bits{};
  bbq_code_layout layout{bbq_code_layout::single_bit};
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};
  float centroid_norm_sq{};

  bbq_quantizer_view(const owning_storage& quantizer) noexcept
    : codes{quantizer.codes.view()},
      lower_intervals{quantizer.lower_intervals.view()},
      upper_intervals{quantizer.upper_intervals.view()},
      additional_corrections{quantizer.additional_corrections.view()},
      quantized_component_sums{quantizer.quantized_component_sums.view()},
      centroid{quantizer.centroid.view()},
      bits{quantizer.bits},
      layout{quantizer.layout},
      metric{quantizer.metric},
      centroid_norm_sq{quantizer.centroid_norm_sq}
  {
  }

  [[nodiscard]] constexpr auto n_rows() const noexcept -> IdxT { return codes.extent(0); }
  [[nodiscard]] constexpr auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(centroid.extent(0));
  }
};

/** @} */  // end of bbq group

}  // namespace preprocessing::quantize::bbq

namespace neighbors {
struct bbq_dataset_container {
  template <typename DataT, typename IdxT, typename Accessor>
  using owning_storage = cuvs::preprocessing::quantize::bbq::bbq_quantizer<DataT, IdxT, Accessor>;
  template <typename DataT, typename IdxT, typename Accessor>
  using view_storage =
    cuvs::preprocessing::quantize::bbq::bbq_quantizer_view<DataT, IdxT, Accessor>;
};

template <typename DataT, typename IdxT, typename Accessor>
struct dataset<bbq_dataset_container, DataT, IdxT, Accessor> {
  using owning_storage_type = bbq_dataset_container::owning_storage<DataT, IdxT, Accessor>;
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
    RAFT_EXPECTS(!has_bit_and_layout(quantizer.bits, quantizer.layout),
                 "Quantizer already exists with bits and layout.");
    this->quantizers.push_back(std::move(quantizer));
  }
  bool has_bit_and_layout(uint32_t bits,
                          cuvs::preprocessing::quantize::bbq::bbq_code_layout layout) const noexcept
  {
    for (uint32_t i = 0; i < quantizers.size(); i++) {
      if (quantizers[i].bits == bits && quantizers[i].layout == layout) { return true; }
    }
    return false;
  }
};

template <typename DataT, typename IdxT, typename Accessor>
struct dataset_view<bbq_dataset_container, DataT, IdxT, Accessor> {
  using owning_storage_type = bbq_dataset_container::
    owning_storage<DataT, IdxT, detail::dataset_owning_accessor_for_view<DataT, Accessor>>;
  using view_storage_type = bbq_dataset_container::view_storage<DataT, IdxT, Accessor>;
  std::vector<view_storage_type> quantizers;

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
    RAFT_EXPECTS(!has_bit_and_layout(quantizer.bits, quantizer.layout),
                 "Quantizer already exists with bits and layout.");
    this->quantizers.push_back(quantizer);
  }
  void add_quantizer(const owning_storage_type& quantizer)
  {
    RAFT_EXPECTS(!has_bit_and_layout(quantizer.bits, quantizer.layout),
                 "Quantizer already exists with bits and layout.");
    this->quantizers.push_back(view_storage_type(quantizer));
  }
  bool has_bit_and_layout(uint32_t bits,
                          cuvs::preprocessing::quantize::bbq::bbq_code_layout layout) const noexcept
  {
    for (uint32_t i = 0; i < quantizers.size(); i++) {
      if (quantizers[i].bits == bits && quantizers[i].layout == layout) { return true; }
    }
    return false;
  }
  view_storage_type get_quantizer(uint32_t bits,
                                  cuvs::preprocessing::quantize::bbq::bbq_code_layout layout) const
  {
    for (uint32_t i = 0; i < quantizers.size(); i++) {
      if (quantizers[i].bits == bits && quantizers[i].layout == layout) { return quantizers[i]; }
    }
    throw std::runtime_error("No quantizer found with bits and layout.");
  }
};

template <typename IdxT>
using device_bbq_dataset =
  dataset<bbq_dataset_container, float, IdxT, detail::device_owning_accessor<float>>;

template <typename IdxT>
using device_bbq_dataset_view =
  dataset_view<bbq_dataset_container, float, IdxT, detail::device_view_accessor<float>>;

template <typename IdxT>
using host_bbq_dataset =
  dataset<bbq_dataset_container, float, IdxT, detail::host_owning_accessor<float>>;

template <typename IdxT>
using host_bbq_dataset_view =
  dataset_view<bbq_dataset_container, float, IdxT, detail::host_view_accessor<float>>;

template <typename IdxT>
struct owning_dataset_for_view<device_bbq_dataset_view<IdxT>> {
  using type = device_bbq_dataset<IdxT>;
};

template <typename DatasetT>
struct is_bbq_dataset : std::false_type {};

template <typename DataT, typename IdxT, typename Accessor>
struct is_bbq_dataset<dataset<bbq_dataset_container, DataT, IdxT, Accessor>> : std::true_type {};

template <typename DatasetT>
inline constexpr bool is_bbq_dataset_v = is_bbq_dataset<DatasetT>::value;

template <typename DataT, typename IdxT, typename Accessor>
struct dataset_view_kind_of<dataset<bbq_dataset_container, DataT, IdxT, Accessor>> {
  static constexpr dataset_view_kind value = dataset_view_kind::bbq;
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
