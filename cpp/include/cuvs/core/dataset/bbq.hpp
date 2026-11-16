/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/type.hpp>
#include <cuvs/core/export.hpp>
#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan.hpp>

#include <cstdint>
#include <type_traits>

namespace CUVS_EXPORT cuvs {

/**
 * The data of a BBQ dataset: the rows encoded in the layout of its quantizer.
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
                                                   row_norm.view()};
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
                                                         row_norm.view()};
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

    using codes_type  = raft::device_matrix<code_type, int64_t>;
    using vector_type = raft::device_vector<math_type, int64_t>;
    using sums_type   = raft::device_vector<sum_type, int64_t>;

    using data_type            = bbq_encoding<codes_type, vector_type, sums_type>;
    using data_view_type       = bbq_encoding<typename codes_type::view_type,
                                              typename vector_type::view_type,
                                              typename sums_type::view_type>;
    using const_data_view_type = bbq_encoding<typename codes_type::const_view_type,
                                              typename vector_type::const_view_type,
                                              typename sums_type::const_view_type>;

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
      return {raft::make_const_mdspan(d.codes),
              raft::make_const_mdspan(d.lower_intervals),
              raft::make_const_mdspan(d.upper_intervals),
              raft::make_const_mdspan(d.additional_corrections),
              raft::make_const_mdspan(d.quantized_component_sums),
              raft::make_const_mdspan(d.dequant_delta),
              raft::make_const_mdspan(d.dequant_sum_delta),
              raft::make_const_mdspan(d.row_norm)};
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
      return static_cast<index_type>(d.codes.extent(0));
    }
    // NOTE: the dimension is the quantizer's; the encoded rows only know their byte length.
    [[nodiscard]] static constexpr auto get_dim(const auto&, const auto& manifest) noexcept
      -> uint32_t
    {
      return manifest.dim();
    }
  };
};

template <typename T, typename IdxT>
using bbq_dataset = dataset<T, IdxT, bbq_spec>;
/** BBQ is a device-side dataset: the kernels read the codes in the device memory. */
template <typename T, typename IdxT>
using device_bbq_dataset = bbq_dataset<T, IdxT>;

}  // namespace CUVS_EXPORT cuvs
