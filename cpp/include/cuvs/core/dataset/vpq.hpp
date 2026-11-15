/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/dense.hpp>
#include <cuvs/core/dataset/type.hpp>
#include <cuvs/core/export.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/util/integer_utils.hpp>

#include <bit>
#include <cstdint>
#include <type_traits>

namespace CUVS_EXPORT cuvs {

/**
 * VQ+PQ compound dataset specification, mirroring the VPQ dataset of cuvs/neighbors/common.hpp:
 * the two codebooks form the manifest, while the data slot keeps the encoded rows - the VQ label
 * in the row prefix, followed by the packed PQ codes.
 *
 * `BookPolicy` is the container policy of the codebooks (elements of `MathT`) and `CodePolicy` the
 * one of the encoded rows (elements of `uint8_t`); a bound policy cannot be rebound to another
 * element type, hence the two parameters.
 */
template <typename MathT, typename BookPolicy, typename CodePolicy>
struct vpq_spec {
  // The encoded rows are stored like the rows of any other dense dataset; only how a row is read
  // and where the dimension comes from are specific to the compression.
  template <typename IdxT>
  using storage_spec =
    typename mdarray_spec<raft::layout_c_contiguous, CodePolicy>::template apply<uint8_t, IdxT>;

  // The type the codebooks are stored in; the rows themselves are always packed codes.
  using math_type = MathT;

  // [vq_n_centers, dim] for the VQ codebook, [pq_n_centers, pq_len] for the PQ one.
  using book_type =
    raft::mdarray<math_type, raft::matrix_extent<uint32_t>, raft::layout_c_contiguous, BookPolicy>;

  struct manifest_type {
    book_type vq_code_book;
    book_type pq_code_book;
  };

  // The views of the codebooks, writable or read-only alike.
  template <typename BookViewT>
  struct manifest_view_of {
    BookViewT vq_code_book;
    BookViewT pq_code_book;

    // NOTE: the dimension of the dataset is a property of the VQ codebook: the encoded rows tell
    // only how many bytes it took to compress a vector, which the padding makes ambiguous.
    [[nodiscard]] auto dim() const noexcept -> uint32_t { return vq_code_book.extent(1); }
    [[nodiscard]] auto vq_n_centers() const noexcept -> uint32_t { return vq_code_book.extent(0); }
    [[nodiscard]] auto pq_n_centers() const noexcept -> uint32_t { return pq_code_book.extent(0); }
    [[nodiscard]] auto pq_len() const noexcept -> uint32_t { return pq_code_book.extent(1); }
    [[nodiscard]] auto pq_bits() const noexcept -> uint32_t
    {
      return static_cast<uint32_t>(std::countr_zero(pq_n_centers()));
    }
    [[nodiscard]] auto pq_dim() const noexcept -> uint32_t
    {
      return raft::div_rounding_up_safe(dim(), pq_len());
    }
  };
  using manifest_view_type       = manifest_view_of<typename book_type::view_type>;
  using const_manifest_view_type = manifest_view_of<typename book_type::const_view_type>;

  template <typename T, typename IdxT>
  struct apply : storage_spec<IdxT> {
    // The dataset stores codes, but it still represents vectors of `T`.
    using value_type = std::remove_cv_t<T>;
    using code_type  = uint8_t;
    // NOTE: re-exported rather than declared here, so that they do not belong to one instantiation
    // of `apply`; qualified lookup into `apply` does not reach the enclosing spec on its own.
    using math_type                = vpq_spec::math_type;
    using book_type                = vpq_spec::book_type;
    using manifest_type            = vpq_spec::manifest_type;
    using manifest_view_type       = vpq_spec::manifest_view_type;
    using const_manifest_view_type = vpq_spec::const_manifest_view_type;

    // NOTE: the data accessors are inherited; only `get_dim` differs from a dense dataset. Like the
    // inherited ones, it reads the manifest in either form: the codebooks an owning dataset
    // holds, or the views a `dataset_view` holds.
    [[nodiscard]] static auto get_dim(const auto&, const auto& manifest) noexcept -> uint32_t
    {
      return static_cast<uint32_t>(manifest.vq_code_book.extent(1));
    }
    [[nodiscard]] static auto get_manifest_view(manifest_type& manifest) noexcept
      -> manifest_view_type
    {
      return {manifest.vq_code_book.view(), manifest.pq_code_book.view()};
    }
    [[nodiscard]] static auto get_manifest_const_view(const manifest_type& manifest) noexcept
      -> const_manifest_view_type
    {
      return {manifest.vq_code_book.view(), manifest.pq_code_book.view()};
    }
    [[nodiscard]] static auto get_manifest_const_view(manifest_view_type manifest) noexcept
      -> const_manifest_view_type
    {
      return {raft::make_const_mdspan(manifest.vq_code_book),
              raft::make_const_mdspan(manifest.pq_code_book)};
    }
    // The length of an encoded row in bytes, including the inlined VQ label.
    [[nodiscard]] static auto get_encoded_row_length(const auto& data) noexcept -> uint32_t
    {
      return static_cast<uint32_t>(data.extent(1));
    }
  };
};

template <typename T, typename IdxT, typename MathT, typename BookPolicy, typename CodePolicy>
using vpq_dataset = dataset<T, IdxT, vpq_spec<MathT, BookPolicy, CodePolicy>>;
template <typename T, typename IdxT, typename MathT>
using device_vpq_dataset =
  vpq_dataset<T,
              IdxT,
              MathT,
              raft::device_accessor<raft::device_container_policy<MathT>>,
              raft::device_accessor<raft::device_container_policy<uint8_t>>>;
template <typename T, typename IdxT, typename MathT>
using host_vpq_dataset = vpq_dataset<T,
                                     IdxT,
                                     MathT,
                                     raft::host_accessor<raft::host_container_policy<MathT>>,
                                     raft::host_accessor<raft::host_container_policy<uint8_t>>>;

}  // namespace CUVS_EXPORT cuvs
