/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/type.hpp>
#include <cuvs/core/export.hpp>
#include <cuvs/preprocessing/quantize/rabitq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>

#include <cstdint>
#include <type_traits>

namespace CUVS_EXPORT cuvs {

/**
 * RaBitQ-quantized dataset specification.
 *
 * The data holds everything that scales with the number of rows:
 *   - `codes` [n_rows, padded_dim]: one `CodeT` per padded dimension, each holding an
 *     unsigned `ex_bits + 1` bit value;
 *   - `factors` [n_rows, 3]: the per-row `(f_add, f_rescale, f_error)` triplet.
 * Both are what the full-factor `preprocessing::quantize::rabitq::transform()` writes when it is
 * called without a centroid.
 *
 * The manifest is the RaBitQ quantizer (rotator and encoder) that produced them. It also knows the
 * logical dimension `dim`, which the codes cannot tell: their width is padded to a multiple of 32.
 *
 * @tparam MathT the element type of the rotation matrix of the quantizer
 * @tparam CodeT the element type of the codes (`uint8_t` or `uint16_t`)
 * @tparam CodePolicy the container policy of the codes (elements of `CodeT`)
 * @tparam FactorPolicy the container policy of the factors (elements of `float`)
 */
template <typename MathT, typename CodeT, typename CodePolicy, typename FactorPolicy>
struct rabitq_spec {
  using codes_type =
    raft::mdarray<CodeT, raft::matrix_extent<int64_t>, raft::layout_c_contiguous, CodePolicy>;
  using factors_type =
    raft::mdarray<float, raft::matrix_extent<int64_t>, raft::layout_c_contiguous, FactorPolicy>;

  // The codes and the factors of the same rows, owning or viewed alike.
  template <typename CodesT, typename FactorsT>
  struct data_of {
    CodesT codes;
    FactorsT factors;
  };
  using data_type      = data_of<codes_type, factors_type>;
  using data_view_type = data_of<typename codes_type::view_type, typename factors_type::view_type>;
  using const_data_view_type =
    data_of<typename codes_type::const_view_type, typename factors_type::const_view_type>;

  // The element type of the rotation matrix; the input and the factors are always `float`.
  using math_type = MathT;

  using manifest_type            = preprocessing::quantize::rabitq::quantizer<math_type>;
  using manifest_view_type       = preprocessing::quantize::rabitq::quantizer_view<math_type>;
  using const_manifest_view_type = preprocessing::quantize::rabitq::quantizer_view<const math_type>;

  template <typename T, typename IdxT>
  struct apply {
    // The dataset stores codes, but it still represents vectors of `T`.
    using value_type = std::remove_cv_t<T>;
    using index_type = std::remove_cv_t<IdxT>;
    using code_type  = CodeT;
    // NOTE: re-exported rather than declared here, so that they do not belong to one instantiation
    // of `apply`; qualified lookup into `apply` does not reach the enclosing spec on its own.
    using math_type                = rabitq_spec::math_type;
    using codes_type               = rabitq_spec::codes_type;
    using factors_type             = rabitq_spec::factors_type;
    using data_type                = rabitq_spec::data_type;
    using data_view_type           = rabitq_spec::data_view_type;
    using const_data_view_type     = rabitq_spec::const_data_view_type;
    using manifest_type            = rabitq_spec::manifest_type;
    using manifest_view_type       = rabitq_spec::manifest_view_type;
    using const_manifest_view_type = rabitq_spec::const_manifest_view_type;

    [[nodiscard]] static auto get_data_view(data_type& data) noexcept -> data_view_type
    {
      return {data.codes.view(), data.factors.view()};
    }
    [[nodiscard]] static auto get_data_const_view(const data_type& data) noexcept
      -> const_data_view_type
    {
      return {data.codes.view(), data.factors.view()};
    }
    [[nodiscard]] static auto get_data_const_view(data_view_type data) noexcept
      -> const_data_view_type
    {
      return {raft::make_const_mdspan(data.codes), raft::make_const_mdspan(data.factors)};
    }
    // NOTE: the shape accessors take the data and the manifest in either form - the storage an
    // owning dataset holds, or the views a `dataset_view` holds.
    [[nodiscard]] static auto get_n_rows(const auto& data) noexcept -> index_type
    {
      return static_cast<index_type>(data.codes.extent(0));
    }
    [[nodiscard]] static auto get_dim(const auto&, const auto& manifest) noexcept -> uint32_t
    {
      return static_cast<uint32_t>(manifest.encoder.dim);
    }
    [[nodiscard]] static auto get_manifest_view(manifest_type& manifest) noexcept
      -> manifest_view_type
    {
      return manifest.view();
    }
    [[nodiscard]] static auto get_manifest_const_view(const manifest_type& manifest) noexcept
      -> const_manifest_view_type
    {
      return manifest.view();
    }
    [[nodiscard]] static auto get_manifest_const_view(manifest_view_type manifest) noexcept
      -> const_manifest_view_type
    {
      return manifest;
    }
  };
};

template <typename T,
          typename IdxT,
          typename MathT,
          typename CodeT,
          typename CodePolicy,
          typename FactorPolicy>
using rabitq_dataset = dataset<T, IdxT, rabitq_spec<MathT, CodeT, CodePolicy, FactorPolicy>>;
template <typename T, typename IdxT, typename MathT = float, typename CodeT = uint8_t>
using device_rabitq_dataset =
  rabitq_dataset<T,
                 IdxT,
                 MathT,
                 CodeT,
                 raft::device_accessor<raft::device_container_policy<CodeT>>,
                 raft::device_accessor<raft::device_container_policy<float>>>;
/** The codes and the factors are on host; the manifest (the quantizer) is on device. */
template <typename T, typename IdxT, typename MathT = float, typename CodeT = uint8_t>
using host_rabitq_dataset = rabitq_dataset<T,
                                           IdxT,
                                           MathT,
                                           CodeT,
                                           raft::host_accessor<raft::host_container_policy<CodeT>>,
                                           raft::host_accessor<raft::host_container_policy<float>>>;

/** A read-only non-owning view of a RaBitQ-quantized dataset in device memory. */
template <typename T, typename IdxT, typename MathT = float, typename CodeT = uint8_t>
using device_rabitq_dataset_view =
  typename device_rabitq_dataset<T, IdxT, MathT, CodeT>::const_view_type;

}  // namespace CUVS_EXPORT cuvs
