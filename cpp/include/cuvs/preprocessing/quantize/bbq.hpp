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
#include <type_traits>

namespace CUVS_EXPORT cuvs {

// NOTE: the BBQ dataset is defined in <cuvs/core/dataset/bbq.hpp>, which includes this header for
// the quantizer it holds as its manifest; the declarations below only need to name it.
template <typename T, typename IdxT, typename SpecT>
struct dataset;
template <typename T, typename IdxT, typename SpecT>
struct dataset_view;
struct bbq_spec;

namespace preprocessing::quantize::bbq {

/**
 * @defgroup bbq-preprocessing Better Binary Quantization utilities
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
  /** Code layout of the encoded rows. */
  bbq_code_layout layout{bbq_code_layout::packed_1b};
  /** Metric the per-row additional corrections are computed for. */
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};

  [[nodiscard]] constexpr auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(centroid.extent(0));
  }

  /** A mutable view decays to the read-only one, the way an mdspan does. */
  template <typename U>
  operator quantizer_view<U>() const noexcept  // NOLINT(google-explicit-constructor)
    requires(!std::is_const_v<T> && std::is_same_v<U, const T>)
  {
    return quantizer_view<U>{
      raft::make_const_mdspan(centroid), raft::make_const_mdspan(centroid_norm_sq), layout, metric};
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
 * additional per-vector information so search is better than naïve sign-bit comparison. They differ
 * in transformation and scale representation. RaBitQ commonly separates residual magnitude from
 * direction, then applies a random orthogonal rotation before binary coding; BBQ uses per-vector
 * scalar intervals to interpret the compressed residual codes.
 *
 * The quantizer is the manifest of a `cuvs::device_bbq_dataset`: it states how the rows of the
 * dataset are encoded. Two datasets encoding the same rows in different layouts against the same
 * centroid form the document and query sides of an asymmetric distance.
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
  /** Code layout of the encoded rows. */
  bbq_code_layout layout{bbq_code_layout::packed_1b};
  /** Metric the per-row additional corrections are computed for. */
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};

  quantizer(raft::resources const& res,
            uint32_t dim,
            bbq_code_layout layout              = bbq_code_layout::packed_1b,
            cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
    : centroid{raft::make_device_vector<T, int64_t>(res, static_cast<int64_t>(dim))},
      centroid_norm_sq{raft::make_device_scalar<float, int64_t>(res)},
      layout{layout},
      metric{metric}
  {
  }

  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(centroid.extent(0));
  }
  [[nodiscard]] auto view() noexcept -> view_type
  {
    return view_type{centroid.view(), centroid_norm_sq.view(), layout, metric};
  }
  [[nodiscard]] auto view() const noexcept -> const_view_type
  {
    return const_view_type{centroid.view(), centroid_norm_sq.view(), layout, metric};
  }
};

/**
 * layout: The code layout of the encoded rows.
 * metric: Used for selecting the per-row additional correction: ||x - c||^2 for L2, x.c otherwise.
 * lambda: Relative interval weight of the quantization error against the norm-preserving term of
 * the loss.
 * iters: Coordinate descent iterations per row.
 */
struct params {
  bbq_code_layout layout              = bbq_code_layout::packed_1b;
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
  float lambda                        = 0.1f;
  int iters                           = 5;
};

/**
 * @brief Builds a BBQ quantizer from a dataset: computes the centroid shared by every row, and its
 * squared norm, into @p quantizer.
 *
 * @p quantizer is typically the manifest of a BBQ dataset from `make_device_bbq_dataset`.
 * Its dimension has to match the one of @p dense_dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::bbq::params params;
 * auto bbq_dataset = cuvs::preprocessing::quantize::bbq::make_device_bbq_dataset(
 *   handle, params, dense_dataset.extent(0), dense_dataset.extent(1));
 * cuvs::preprocessing::quantize::bbq::build(handle, dense_dataset, bbq_dataset.manifest_view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] dense_dataset a row-major matrix view on device
 * @param[out] quantizer a view of the BBQ quantizer to write the centroid into
 */
void build(raft::resources const& res,
           raft::device_matrix_view<const float, int64_t> dense_dataset,
           quantizer_view<float> quantizer);

/**
 * @brief Allocates a BBQ dataset of the shape `transform` writes, leaving its contents
 * uninitialized.
 *
 * The rows are allocated in `params.layout`. The quantizer records `params.layout` and
 * `params.metric`, and holds room for a centroid of @p dim components.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::bbq::params params;
 * auto bbq_dataset =
 *   cuvs::preprocessing::quantize::bbq::make_device_bbq_dataset(handle, params, n_rows, dim);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configuration for BBQ encoding; only layout and metric are used
 * @param[in] n_rows number of rows
 * @param[in] dim number of components of each row before encoding
 *
 * @return an uninitialized BBQ dataset
 */
cuvs::dataset<float, int64_t, cuvs::bbq_spec> make_device_bbq_dataset(raft::resources const& res,
                                                                      const params params,
                                                                      int64_t n_rows,
                                                                      uint32_t dim);

/**
 * @brief Encodes a dataset into a BBQ dataset, against the centroid in the manifest of the BBQ
 * dataset.
 *
 * @p out receives the rows encoded in the layout of its quantizer. The manifest of @p out has to
 * hold the quantizer already, from a previous call to `build`, and its layout and metric have to
 * be the ones of @p params.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::bbq::params params;
 * auto bbq_dataset = cuvs::preprocessing::quantize::bbq::make_device_bbq_dataset(
 *   handle, params, dense_dataset.extent(0), dense_dataset.extent(1));
 * cuvs::preprocessing::quantize::bbq::build(handle, dense_dataset, bbq_dataset.manifest_view());
 * cuvs::preprocessing::quantize::bbq::transform(handle, params, dense_dataset, bbq_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configuration for BBQ encoding, e.g. layout, metric, lambda, iters
 * @param[in] dense_dataset a row-major matrix view on device
 * @param[inout] out a view of the BBQ dataset whose manifest holds the quantizer, to write the
 * encoding into; a `cuvs::device_bbq_dataset<float, int64_t>::view_type`
 */
void transform(raft::resources const& res,
               const params params,
               raft::device_matrix_view<const float, int64_t> dense_dataset,
               cuvs::dataset_view<float, int64_t, cuvs::bbq_spec> out);

/** @} */  // end of bbq group

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

}  // namespace preprocessing::quantize::bbq
}  // namespace CUVS_EXPORT cuvs
