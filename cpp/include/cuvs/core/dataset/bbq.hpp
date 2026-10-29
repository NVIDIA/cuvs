/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/affine.hpp>
#include <cuvs/core/dataset/type.hpp>
#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/mdarray.hpp>

#include <cstdint>
#include <type_traits>

namespace CUVS_EXPORT cuvs {

/**
 * The data of a BBQ dataset, per row: the `codes` in the layout of the manifest, together with the
 * per-row terms of the distance estimator
 *
 *   estimate = codes.shift + codes.scale * <code, query> + ...
 *
 * where `codes.scale` is the `dequant_delta` and `codes.shift` the `additional_corrections`; then
 * the `lower_intervals`, the `dequant_sum_delta` (the row side of the cross term with a quantized
 * query) and the `row_norm` (the squared norm of the row, read by the cosine and the
 * inner-product metrics). The encoder-only columns (the upper intervals and the quantized
 * component sums) are not part of the dataset: they only feed `helpers::resolve_dequant_factors`.
 *
 * It holds arrays for an owning dataset and spans for a view of it.
 *
 * @tparam CodesT the type of the code matrix [n_rows, the encoded row length]
 * @tparam VectorT the type of the per-row vectors [n_rows]
 */
template <typename CodesT, typename VectorT>
struct bbq_data {
  affine_data<CodesT, VectorT> codes;
  VectorT lower_intervals;
  VectorT dequant_sum_delta;
  VectorT row_norm;

  [[nodiscard]] auto view() noexcept
  {
    return bbq_data<typename CodesT::view_type, typename VectorT::view_type>{
      codes.view(), lower_intervals.view(), dequant_sum_delta.view(), row_norm.view()};
  }

  [[nodiscard]] auto const_view() const noexcept
  {
    return bbq_data<decltype(as_const_view(codes.rows)), decltype(as_const_view(codes.scale))>{
      as_const_view(codes),
      as_const_view(lower_intervals),
      as_const_view(dequant_sum_delta),
      as_const_view(row_norm)};
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

    static constexpr bool is_device_accessible = true;

    using data_type =
      bbq_data<raft::device_matrix<code_type, int64_t>, raft::device_vector<math_type, int64_t>>;
    using data_view_type = decltype(std::declval<data_type&>().view());

    using manifest_type            = preprocessing::quantize::bbq::quantizer<value_type>;
    using manifest_view_type       = preprocessing::quantize::bbq::quantizer_view<value_type>;
    using const_manifest_view_type = preprocessing::quantize::bbq::quantizer_view<const value_type>;

    [[nodiscard]] static auto get_data_view(data_type& d) noexcept -> data_view_type
    {
      return d.view();
    }

    [[nodiscard]] static auto get_manifest_view(manifest_type& m) noexcept -> manifest_view_type
    {
      return m.view();
    }

    [[nodiscard]] static constexpr auto get_n_rows(const auto& d) noexcept -> index_type
    {
      return static_cast<index_type>(d.codes.rows.extent(0));
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
