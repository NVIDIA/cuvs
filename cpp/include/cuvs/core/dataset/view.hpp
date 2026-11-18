/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <variant>

namespace CUVS_EXPORT cuvs {

/**
 * A spec is compound if it defines a manifest: the codebooks, the quantization range, or
 * whatever else is needed to interpret the data, so the dataset consists of the data and the
 * manifest. Plain specs define none and the dataset substitutes `std::monostate`, so "there is no
 * manifest" and "the manifest slot is empty" are the same vocabulary type.
 *
 * NOTE: a spec declares its vocabulary types - the manifest above, and any storage representation
 * of its own - at spec scope and only re-exports them from `apply`. A type defined inside `apply`
 * belongs to one instantiation of it, and the read-only and the mutable flavor of a view apply the
 * spec to element types that differ in constness; the two would then disagree about a nested
 * manifest and could no longer be built from one another.
 */
template <typename SpecT>
concept compound_dataset_spec = requires { typename SpecT::manifest_type; };

/**
 * A non-owning view of a dataset: a view of the data and, for a compound dataset, a view of the
 * manifest that gives it meaning. Cheap to copy and to pass by value, like an mdspan.
 *
 * The dataset consists of two components: the data and the manifest.
 *
 * The data member is the main storage of the dataset: it contains the raw (maybe
 * encoded/compressed) data. The data grows linearly with the number of rows in the dataset.
 * Examples of the data content: floating point vectors, encoded byte representations, per-point
 * labels scaling factors, cluster ids, etc - everything that is associated with every data row.
 *
 * The manifest member is an optional component that contains everything to give meaning to the
 * data. Its size is bounded and doesn't depend directly on the number of rows in the dataset.
 * Examples of the manifest content: codebooks, quantization ranges, global scaling factors,
 * extra data transforms (such as rotation matrices) etc. - everything that is relevant for the
 * entire dataset, not just individual rows.
 */
template <typename T, typename IdxT, typename SpecT>
struct dataset_view {
  using spec_type = typename SpecT::template apply<T, IdxT>;
  // The read-only flavor, in which each view type collapses onto its const counterpart.
  static constexpr bool is_const_view = std::is_const_v<T>;

  using value_type = typename spec_type::value_type;
  using index_type = typename spec_type::index_type;
  // What one element of the data slot is
  //   - for compound datasets it is the raw data element type; often is uint8_t but may be
  //     a larger type to enforce better alignment.
  //   - for plain datasets it is the value type of the dataset.
  using code_type = typename decltype([] {
    if constexpr (requires { typename spec_type::code_type; }) {
      return std::type_identity<typename spec_type::code_type>{};
    } else {
      return std::type_identity<value_type>{};
    }
  }())::type;

  using data_type            = typename spec_type::data_type;
  using const_data_view_type = typename spec_type::const_data_view_type;
  using data_view_type =
    std::conditional_t<is_const_view, const_data_view_type, typename spec_type::data_view_type>;

  // Optional slot that contains everything to give meaning to the encoded data;
  // `std::monostate` for a plain dataset.
  using manifest_type = typename decltype([] {
    if constexpr (requires { typename spec_type::manifest_type; }) {
      return std::type_identity<typename spec_type::manifest_type>{};
    } else {
      return std::type_identity<std::monostate>{};
    }
  }())::type;
  // Optional, so that the specs don't have to declare it; a spec that declares neither view uses
  // the manifest itself, and one that declares only the mutable view uses it for both.
  using const_manifest_view_type = typename decltype([] {
    if constexpr (requires { typename spec_type::const_manifest_view_type; }) {
      return std::type_identity<typename spec_type::const_manifest_view_type>{};
    } else if constexpr (requires { typename spec_type::manifest_view_type; }) {
      return std::type_identity<typename spec_type::manifest_view_type>{};
    } else {
      return std::type_identity<manifest_type>{};
    }
  }())::type;
  // Optional, so that the specs don't have to declare it
  using manifest_view_type = typename decltype([] {
    if constexpr (is_const_view) {
      return std::type_identity<const_manifest_view_type>{};
    } else if constexpr (requires { typename spec_type::manifest_view_type; }) {
      return std::type_identity<typename spec_type::manifest_view_type>{};
    } else {
      return std::type_identity<manifest_type>{};
    }
  }())::type;

  static constexpr bool is_compound = compound_dataset_spec<spec_type>;

  /** An empty view, like a default-constructed mdspan: no rows, no data. */
  dataset_view() noexcept = default;

  explicit dataset_view(data_view_type data_view)
    requires(!is_compound)
    : data_view_{data_view}
  {
  }

  dataset_view(data_view_type data_view, manifest_view_type manifest_view)
    requires(is_compound)
    : data_view_{data_view}, manifest_view_{manifest_view}
  {
  }

  /**
   * @brief A mutable view decays to a read-only one, the way an mdspan does.
   *
   * NOTE: a constructor template, because in the mutable flavor the parameter would spell the very
   * same type; a non-template would be a copy constructor there, and declaring one - even a
   * constrained-away one - suppresses the implicitly declared copy constructor.
   *
   * NOTE: the manifest is assigned in the body rather than in the init-list, because the
   * accessor it comes from is constrained on `is_compound`.
   */
  template <typename OtherT>
    requires(is_const_view && std::same_as<OtherT, std::remove_const_t<T>>)
  dataset_view(const dataset_view<OtherT, IdxT, SpecT>& other) noexcept
    : data_view_{other.data_const_view()}
  {
    if constexpr (is_compound) { manifest_view_ = other.const_manifest_view(); }
  }

  /**
   * @brief The number of rows in the dataset.
   *
   * @return The number of rows.
   */
  [[nodiscard]] auto n_rows() const noexcept -> index_type
  {
    return spec_type::get_n_rows(data_view_);
  }
  /**
   * @brief The dimension of the vectors in the dataset.
   *
   * NOTE: a compound dataset may not be able to tell its dimension from the encoded data alone.
   *
   * @return The dimension.
   */
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return spec_type::get_dim(data_view_, manifest_view_);
  }

  /**
   * @brief A non-owning view of the data, such as mdspan.
   *
   * @return A view of the data.
   */
  [[nodiscard]] auto data_view() const noexcept -> data_view_type { return data_view_; }
  /**
   * @brief A read-only view of the data.
   *
   * NOTE: giving up write access is not a plain cast for most view types - a mutable mdspan, for
   * one, has to be rebuilt over a const accessor - hence the spec does the conversion.
   *
   * @return A read-only view of the data.
   */
  [[nodiscard]] auto data_const_view() const noexcept -> const_data_view_type
  {
    if constexpr (is_const_view) {
      return data_view_;
    } else {
      return spec_type::get_data_const_view(data_view_);
    }
  }

  /**
   * @brief A non-owning view of the manifest, such as the two codebooks of a VPQ dataset.
   *
   * @return A view of the manifest.
   */
  [[nodiscard]] auto manifest_view() const noexcept -> manifest_view_type
    requires(is_compound)
  {
    return manifest_view_;
  }
  /**
   * @brief A read-only view of the manifest.
   *
   * @return A read-only view of the manifest.
   */
  [[nodiscard]] auto const_manifest_view() const noexcept -> const_manifest_view_type
    requires(is_compound)
  {
    if constexpr (is_const_view) {
      return manifest_view_;
    } else {
      return spec_type::get_manifest_const_view(manifest_view_);
    }
  }

 private:
  data_view_type data_view_;
  [[no_unique_address]] manifest_view_type manifest_view_;
};

}  // namespace CUVS_EXPORT cuvs
