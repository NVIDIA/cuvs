/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/dense.hpp>
#include <cuvs/core/dataset/type.hpp>
#include <cuvs/core/export.hpp>
#include <cuvs/preprocessing/quantize/binary.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>

#include <cstdint>
#include <type_traits>

namespace CUVS_EXPORT cuvs {

/**
 * Binary-quantized dataset specification.
 *
 * The data are bit-packed codes, `ceil(dim / 8)` bytes per row; bit `i` of a row is stored in bit
 * `i % 8` of byte `i / 8`, and the unused bits of the last byte are zero. The manifest is the
 * binary quantizer that produced the codes, so the logical dimension `dim` (the number of bits)
 * never has to be guessed from the padded byte width: the quantizer knows it.
 *
 * @tparam MathT the type of the source (unquantized) values the thresholds are compared with
 * @tparam CodePolicy the container policy of the codes (elements of `uint8_t`)
 * @tparam ThresholdPolicy the container policy of the thresholds of the quantizer (elements of
 *   `MathT`)
 */
template <typename MathT, typename CodePolicy, typename ThresholdPolicy>
struct binary_spec {
  // The codes are stored like the rows of any other dense dataset; only where the dimension comes
  // from is specific to the binarization.
  template <typename IdxT>
  using storage_spec =
    typename mdarray_spec<raft::layout_c_contiguous, CodePolicy>::template apply<uint8_t, IdxT>;

  // The type of the source (unquantized) values the thresholds are compared with.
  using math_type = MathT;

  using manifest_type = preprocessing::quantize::binary::quantizer<math_type, ThresholdPolicy>;
  using manifest_view_type =
    preprocessing::quantize::binary::quantizer_view<math_type, ThresholdPolicy>;
  using const_manifest_view_type =
    preprocessing::quantize::binary::quantizer_view<const math_type, ThresholdPolicy>;

  template <typename T, typename IdxT>
  struct apply : storage_spec<IdxT> {
    // The dataset stores codes, but it still represents vectors of `T`.
    using value_type = std::remove_cv_t<T>;
    using code_type  = uint8_t;
    // NOTE: re-exported rather than declared here, so that they do not belong to one instantiation
    // of `apply`; qualified lookup into `apply` does not reach the enclosing spec on its own.
    using math_type                = binary_spec::math_type;
    using manifest_type            = binary_spec::manifest_type;
    using manifest_view_type       = binary_spec::manifest_view_type;
    using const_manifest_view_type = binary_spec::const_manifest_view_type;

    // NOTE: the data accessors are inherited; only `get_dim` differs from a dense dataset. Like the
    // inherited ones, it reads the manifest in either form. The dimension is the number of bits,
    // which the encoded rows cannot tell: their width is padded to whole bytes.
    [[nodiscard]] static auto get_dim(const auto&, const auto& manifest) noexcept -> uint32_t
    {
      return manifest.dim();
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

template <typename T, typename IdxT, typename CodePolicy, typename ThresholdPolicy>
using binary_dataset = dataset<T, IdxT, binary_spec<T, CodePolicy, ThresholdPolicy>>;
template <typename T, typename IdxT>
using device_binary_dataset =
  binary_dataset<T,
                 IdxT,
                 raft::device_accessor<raft::device_container_policy<uint8_t>>,
                 raft::device_accessor<raft::device_container_policy<T>>>;
template <typename T, typename IdxT>
using host_binary_dataset =
  binary_dataset<T,
                 IdxT,
                 raft::host_accessor<raft::host_container_policy<uint8_t>>,
                 raft::host_accessor<raft::host_container_policy<T>>>;

/** A read-only non-owning view of a binary-quantized dataset in device memory. */
template <typename T, typename IdxT>
using device_binary_dataset_view = typename device_binary_dataset<T, IdxT>::const_view_type;

}  // namespace CUVS_EXPORT cuvs
