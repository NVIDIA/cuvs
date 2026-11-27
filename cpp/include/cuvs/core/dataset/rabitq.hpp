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
 *   - `signs` [padded_dim / 32, n_rows]: transposed packed uint32 sign words;
 *   - `extra_bits` [n_rows, padded_dim * ex_bits / 8]: packed uint8 extra-bit bytes;
 *   - `factors` [n_rows, 3]: the per-row `(f_add, f_rescale, f_error)` triplet.
 * These are what the full-factor `preprocessing::quantize::rabitq::transform()` writes when it is
 * called without a centroid.
 *
 * The manifest is the RaBitQ quantizer (rotator and encoder) that produced them. It also knows the
 * logical dimension `dim`, which the codes cannot tell: their width is padded to a multiple of 32.
 *
 * @tparam MathT the element type of the rotation matrix of the quantizer
 * @tparam SignPolicy the container policy of the sign words (elements of `uint32_t`)
 * @tparam ExtraPolicy the container policy of the extra-bit bytes (elements of `uint8_t`)
 * @tparam FactorPolicy the container policy of the factors (elements of `float`)
 */
template <typename MathT, typename SignPolicy, typename ExtraPolicy, typename FactorPolicy>
struct rabitq_spec {
  using signs_type =
    raft::mdarray<uint32_t, raft::matrix_extent<int64_t>, raft::layout_c_contiguous, SignPolicy>;
  using extra_bits_type =
    raft::mdarray<uint8_t, raft::matrix_extent<int64_t>, raft::layout_c_contiguous, ExtraPolicy>;
  using factors_type =
    raft::mdarray<float, raft::matrix_extent<int64_t>, raft::layout_c_contiguous, FactorPolicy>;

  // Packed codes and factors for the same rows, owning or viewed alike.
  template <typename SignsT, typename ExtraT, typename FactorsT>
  struct data_of {
    SignsT signs;
    ExtraT extra_bits;
    FactorsT factors;
  };
  using data_type            = data_of<signs_type, extra_bits_type, factors_type>;
  using data_view_type       = data_of<typename signs_type::view_type,
                                       typename extra_bits_type::view_type,
                                       typename factors_type::view_type>;
  using const_data_view_type = data_of<typename signs_type::const_view_type,
                                       typename extra_bits_type::const_view_type,
                                       typename factors_type::const_view_type>;

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
    using code_type  = uint8_t;
    // NOTE: re-exported rather than declared here, so that they do not belong to one instantiation
    // of `apply`; qualified lookup into `apply` does not reach the enclosing spec on its own.
    using math_type                = rabitq_spec::math_type;
    using signs_type               = rabitq_spec::signs_type;
    using extra_bits_type          = rabitq_spec::extra_bits_type;
    using factors_type             = rabitq_spec::factors_type;
    using data_type                = rabitq_spec::data_type;
    using data_view_type           = rabitq_spec::data_view_type;
    using const_data_view_type     = rabitq_spec::const_data_view_type;
    using manifest_type            = rabitq_spec::manifest_type;
    using manifest_view_type       = rabitq_spec::manifest_view_type;
    using const_manifest_view_type = rabitq_spec::const_manifest_view_type;

    [[nodiscard]] static auto get_data_view(data_type& data) noexcept -> data_view_type
    {
      return {data.signs.view(), data.extra_bits.view(), data.factors.view()};
    }
    [[nodiscard]] static auto get_data_const_view(const data_type& data) noexcept
      -> const_data_view_type
    {
      return {data.signs.view(), data.extra_bits.view(), data.factors.view()};
    }
    [[nodiscard]] static auto get_data_const_view(data_view_type data) noexcept
      -> const_data_view_type
    {
      return {raft::make_const_mdspan(data.signs),
              raft::make_const_mdspan(data.extra_bits),
              raft::make_const_mdspan(data.factors)};
    }
    // NOTE: the shape accessors take the data and the manifest in either form - the storage an
    // owning dataset holds, or the views a `dataset_view` holds.
    [[nodiscard]] static auto get_n_rows(const auto& data) noexcept -> index_type
    {
      return static_cast<index_type>(data.signs.extent(1));
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
          typename SignPolicy,
          typename ExtraPolicy,
          typename FactorPolicy>
using rabitq_dataset = dataset<T, IdxT, rabitq_spec<MathT, SignPolicy, ExtraPolicy, FactorPolicy>>;
template <typename T, typename IdxT, typename MathT = float>
using device_rabitq_dataset =
  rabitq_dataset<T,
                 IdxT,
                 MathT,
                 raft::device_accessor<raft::device_container_policy<uint32_t>>,
                 raft::device_accessor<raft::device_container_policy<uint8_t>>,
                 raft::device_accessor<raft::device_container_policy<float>>>;
/** The codes and factors are on host; the manifest (the quantizer) is on device. */
template <typename T, typename IdxT, typename MathT = float>
using host_rabitq_dataset =
  rabitq_dataset<T,
                 IdxT,
                 MathT,
                 raft::host_accessor<raft::host_container_policy<uint32_t>>,
                 raft::host_accessor<raft::host_container_policy<uint8_t>>,
                 raft::host_accessor<raft::host_container_policy<float>>>;

/** A read-only non-owning view of a RaBitQ-quantized dataset in device memory. */
template <typename T, typename IdxT, typename MathT = float>
using device_rabitq_dataset_view = typename device_rabitq_dataset<T, IdxT, MathT>::const_view_type;

}  // namespace CUVS_EXPORT cuvs
