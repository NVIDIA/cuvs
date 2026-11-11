/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/type.hpp>
#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace CUVS_EXPORT cuvs {

namespace preprocessing::quantize::bbq {

/**
 * @addtogroup bbq-dataset
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

template <typename T>
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
 *
 * The quantizer is the manifest of a `cuvs::device_bbq_dataset`: what every encoding of the rows
 * shares. The layouts and the metric belong to the encoded data.
 *
 * @tparam T data element type
 */
template <typename T>
struct quantizer {
  static_assert(!std::is_const_v<T>, "An owning quantizer cannot hold const elements.");

  using view_type       = quantizer_view<T>;
  using const_view_type = quantizer_view<const T>;

  /** Centroid shared by every row; rows are encoded relative to it. */
  raft::device_vector<T, int64_t> centroid;
  /** Squared L2 norm of the centroid, ||c||^2. */
  raft::device_scalar<float, int64_t> centroid_norm_sq;

  quantizer(raft::resources const& res, uint32_t dim)
    : centroid{raft::make_device_vector<T, int64_t>(res, static_cast<int64_t>(dim))},
      centroid_norm_sq{raft::make_device_scalar<float, int64_t>(res)}
  {
  }

  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(centroid.extent(0));
  }
  [[nodiscard]] auto view() noexcept -> view_type
  {
    return view_type{centroid.view(), centroid_norm_sq.view()};
  }
  [[nodiscard]] auto view() const noexcept -> const_view_type
  {
    return const_view_type{centroid.view(), centroid_norm_sq.view()};
  }
};

/**
 * @brief Non-owning view of a BBQ quantizer.
 *
 * The view owns neither the centroid nor its norm, so the quantizer it was taken from has to
 * outlive it.
 *
 * @tparam T data element type; const for a read-only view
 */
template <typename T>
struct quantizer_view {
  using norm_type = std::conditional_t<std::is_const_v<T>, const float, float>;

  /** Centroid shared by every row; rows are encoded relative to it. */
  raft::device_vector_view<T, int64_t> centroid;
  /** Squared L2 norm of the centroid, ||c||^2. */
  // NOTE: a scalar mdspan has a static extent, so it has no default constructor of its own.
  raft::device_scalar_view<norm_type, int64_t> centroid_norm_sq{static_cast<norm_type*>(nullptr)};

  [[nodiscard]] constexpr auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(centroid.extent(0));
  }

  /** A mutable view decays to the read-only one, the way an mdspan does. */
  template <typename U>
  operator quantizer_view<U>() const noexcept  // NOLINT(google-explicit-constructor)
    requires(!std::is_const_v<T> && std::is_same_v<U, const T>)
  {
    return quantizer_view<U>{raft::make_const_mdspan(centroid),
                             raft::make_const_mdspan(centroid_norm_sq)};
  }
};

/** @} */  // end of bbq group

}  // namespace preprocessing::quantize::bbq

/**
 * One encoding of the rows of a BBQ dataset: the codes in one layout, together with the per-row
 * terms of the distance estimator and the encoder outputs those terms are derived from.
 *
 * It holds arrays for an owning dataset and spans for a view of it.
 *
 * @tparam CodesT the type of the code matrix [n_rows, the encoded row length]
 * @tparam VectorT the type of the per-row vectors [n_rows]
 * @tparam SumsT the type of the per-row code sums [n_rows]
 */
template <typename CodesT, typename VectorT, typename SumsT>
struct bbq_encoding {
  /** Encoded rows, `get_encoded_row_length(dim, layout)` bytes per row. */
  CodesT codes;
  /** Per-row lower bound of the quantization interval, relative to the centroid. */
  VectorT lower_intervals;
  /** Per-row upper bound of the quantization interval, relative to the centroid. */
  VectorT upper_intervals;
  /** Per-row correction: ||x - c||^2 for L2 metrics, x.c otherwise. */
  VectorT additional_corrections;
  /** Per-row sum of the codes. */
  SumsT quantized_component_sums;
  /** Per-row step between consecutive codes: (upper - lower) / (2^bits - 1). */
  VectorT dequant_delta;
  /** Per-row dequant_delta * quantized_component_sums. */
  VectorT dequant_sum_delta;
  /** Per-row squared norm in the original (un-centered) vector space, ||x||^2. */
  VectorT row_norm;
  /** Code layout of `codes`. */
  preprocessing::quantize::bbq::bbq_code_layout layout{
    preprocessing::quantize::bbq::bbq_code_layout::packed_1b};

  /** The spans over the arrays. */
  [[nodiscard]] auto view() noexcept
  {
    return bbq_encoding<typename CodesT::view_type,
                        typename VectorT::view_type,
                        typename SumsT::view_type>{codes.view(),
                                                   lower_intervals.view(),
                                                   upper_intervals.view(),
                                                   additional_corrections.view(),
                                                   quantized_component_sums.view(),
                                                   dequant_delta.view(),
                                                   dequant_sum_delta.view(),
                                                   row_norm.view(),
                                                   layout};
  }

  /** The read-only spans over the arrays. */
  [[nodiscard]] auto view() const noexcept
  {
    return bbq_encoding<typename CodesT::const_view_type,
                        typename VectorT::const_view_type,
                        typename SumsT::const_view_type>{codes.view(),
                                                         lower_intervals.view(),
                                                         upper_intervals.view(),
                                                         additional_corrections.view(),
                                                         quantized_component_sums.view(),
                                                         dequant_delta.view(),
                                                         dequant_sum_delta.view(),
                                                         row_norm.view(),
                                                         layout};
  }
};

/**
 * The data of a BBQ dataset view: the encoding read on the document side of a distance and the one
 * read on its query side. Both encode the same rows, in the same order, against the centroid of the
 * manifest, and their corrections were computed for `metric`. A symmetric dataset refers to one
 * encoding on both sides.
 *
 * @tparam CodesT the type of the code matrix span [n_rows, the encoded row length]
 * @tparam VectorT the type of the per-row vector spans [n_rows]
 * @tparam SumsT the type of the per-row code sum span [n_rows]
 */
template <typename CodesT, typename VectorT, typename SumsT>
struct bbq_data_view {
  using encoding_type = bbq_encoding<CodesT, VectorT, SumsT>;

  /** Metric the additional corrections of both encodings were computed for. */
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};
  /** Encoding read on the document side of a distance. */
  encoding_type document;
  /** Encoding read on the query side of a distance; the document encoding when symmetric. */
  encoding_type query;
};

/**
 * The data of an owning BBQ dataset: the rows encoded in the document layout and, when the query
 * side of a distance reads a different layout, encoded a second time in that one.
 *
 * @tparam CodesT the type of the code matrix [n_rows, the encoded row length]
 * @tparam VectorT the type of the per-row vectors [n_rows]
 * @tparam SumsT the type of the per-row code sums [n_rows]
 */
template <typename CodesT, typename VectorT, typename SumsT>
struct bbq_data {
  using encoding_type   = bbq_encoding<CodesT, VectorT, SumsT>;
  using view_type       = bbq_data_view<typename CodesT::view_type,
                                        typename VectorT::view_type,
                                        typename SumsT::view_type>;
  using const_view_type = bbq_data_view<typename CodesT::const_view_type,
                                        typename VectorT::const_view_type,
                                        typename SumsT::const_view_type>;

  /** Metric the additional corrections of both encodings were computed for. */
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};
  /** Encoding read on the document side of a distance. */
  encoding_type document;
  /** Encoding read on the query side of a distance; empty when the dataset is symmetric. */
  std::optional<encoding_type> query;

  /** The spans over the arrays; a symmetric dataset hands out its document encoding twice. */
  [[nodiscard]] auto view() noexcept -> view_type
  {
    auto document_view = document.view();
    return {metric, document_view, query.has_value() ? query->view() : document_view};
  }

  /** The read-only spans over the arrays. */
  [[nodiscard]] auto view() const noexcept -> const_view_type
  {
    auto document_view = document.view();
    return {metric, document_view, query.has_value() ? query->view() : document_view};
  }
};

/**
 * BBQ dataset specification: device-side, as the kernels read the codes in the device memory, with
 * `float` per-row terms, as the kernels use. The manifest is the BBQ quantizer that produced the
 * codes (the centroid is stored in the value type of the dataset), so the logical dimension never
 * has to be guessed from the padded byte width of the encoded rows: the quantizer knows it.
 */
struct bbq_spec {
  template <typename T, typename IdxT>
  struct apply {
    using value_type = std::remove_const_t<T>;
    using index_type = std::remove_const_t<IdxT>;
    using code_type  = uint8_t;
    using math_type  = float;
    using sum_type   = int32_t;

    static constexpr bool is_device_accessible = true;

    using data_type            = bbq_data<raft::device_matrix<code_type, int64_t>,
                                          raft::device_vector<math_type, int64_t>,
                                          raft::device_vector<sum_type, int64_t>>;
    using data_view_type       = typename data_type::view_type;
    using const_data_view_type = typename data_type::const_view_type;

    using manifest_type            = preprocessing::quantize::bbq::quantizer<value_type>;
    using manifest_view_type       = preprocessing::quantize::bbq::quantizer_view<value_type>;
    using const_manifest_view_type = preprocessing::quantize::bbq::quantizer_view<const value_type>;

    [[nodiscard]] static auto get_data_view(data_type& d) noexcept -> data_view_type
    {
      return d.view();
    }
    [[nodiscard]] static auto get_data_const_view(const data_type& d) noexcept
      -> const_data_view_type
    {
      return d.view();
    }
    [[nodiscard]] static auto get_data_const_view(data_view_type d) noexcept -> const_data_view_type
    {
      return {d.metric, get_encoding_const_view(d.document), get_encoding_const_view(d.query)};
    }

    [[nodiscard]] static auto get_manifest_view(manifest_type& m) noexcept -> manifest_view_type
    {
      return m.view();
    }
    [[nodiscard]] static auto get_manifest_const_view(const manifest_type& m) noexcept
      -> const_manifest_view_type
    {
      return m.view();
    }
    [[nodiscard]] static auto get_manifest_const_view(manifest_view_type m) noexcept
      -> const_manifest_view_type
    {
      return m;
    }

    // NOTE: the shape accessors take the data in either form - the storage an owning dataset holds,
    // or the view a `dataset_view` holds.
    [[nodiscard]] static constexpr auto get_n_rows(const auto& d) noexcept -> index_type
    {
      return static_cast<index_type>(d.document.codes.extent(0));
    }
    // NOTE: the dimension is the quantizer's; the encoded rows only know their byte length.
    [[nodiscard]] static constexpr auto get_dim(const auto&, const auto& manifest) noexcept
      -> uint32_t
    {
      return manifest.dim();
    }

   private:
    [[nodiscard]] static auto get_encoding_const_view(
      const typename data_view_type::encoding_type& e) noexcept ->
      typename const_data_view_type::encoding_type
    {
      return {raft::make_const_mdspan(e.codes),
              raft::make_const_mdspan(e.lower_intervals),
              raft::make_const_mdspan(e.upper_intervals),
              raft::make_const_mdspan(e.additional_corrections),
              raft::make_const_mdspan(e.quantized_component_sums),
              raft::make_const_mdspan(e.dequant_delta),
              raft::make_const_mdspan(e.dequant_sum_delta),
              raft::make_const_mdspan(e.row_norm),
              e.layout};
    }
  };
};

template <typename T, typename IdxT>
using bbq_dataset = dataset<T, IdxT, bbq_spec>;
/** BBQ is a device-side dataset: the kernels read the codes in the device memory. */
template <typename T, typename IdxT>
using device_bbq_dataset = bbq_dataset<T, IdxT>;

}  // namespace CUVS_EXPORT cuvs
