/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/view.hpp>
#include <cuvs/core/export.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace CUVS_EXPORT cuvs {

/**
 * An owning dataset: the data and, for a compound dataset, the manifest that gives it meaning.
 * Ownership is unique, so the type is move-only whenever the underlying storage is.
 *
 * See the dataset_view documentation for more details on the dataset's components.
 */
template <typename T, typename IdxT, typename SpecT>
struct dataset {
  // NOTE: element type for an owning dataset cannot be const,
  // the constness of the dataset itself selects the flavor of the view it hands out.
  static_assert(!std::is_const_v<T>, "Element type for an owning dataset must not be const.");

  using view_type       = dataset_view<T, IdxT, SpecT>;
  using const_view_type = dataset_view<const T, IdxT, SpecT>;

  using spec_type  = typename view_type::spec_type;
  using value_type = typename view_type::value_type;
  using index_type = typename view_type::index_type;
  using code_type  = typename view_type::code_type;

  using data_type            = typename view_type::data_type;
  using data_view_type       = typename view_type::data_view_type;
  using const_data_view_type = typename view_type::const_data_view_type;

  /**
    Optional slot that contains everything to give meaning to the encoded data;
    `std::monostate` for a plain dataset.
   */
  using manifest_type            = typename view_type::manifest_type;
  using manifest_view_type       = typename view_type::manifest_view_type;
  using const_manifest_view_type = typename view_type::const_manifest_view_type;

  static constexpr bool is_compound = view_type::is_compound;

  explicit dataset(data_type&& data)
    requires(!is_compound)
    : data_{std::move(data)}
  {
  }

  dataset(data_type&& data, manifest_type&& manifest)
    requires(is_compound)
    : data_{std::move(data)}, manifest_{std::move(manifest)}
  {
  }

  /**
   * @brief A non-owning view of the whole dataset, data and manifest alike.
   *
   * @return A mutable view of the dataset.
   */
  [[nodiscard]] auto view() noexcept -> view_type
  {
    if constexpr (is_compound) {
      return view_type{data_view(), manifest_view()};
    } else {
      return view_type{data_view()};
    }
  }
  /**
   * @brief A read-only view of the whole dataset; the constness of the dataset selects this flavor.
   *
   * @return A read-only view of the dataset.
   */
  [[nodiscard]] auto view() const noexcept -> const_view_type
  {
    if constexpr (is_compound) {
      return const_view_type{data_const_view(), const_manifest_view()};
    } else {
      return const_view_type{data_const_view()};
    }
  }

  /**
   * @brief The number of rows in the dataset.
   *
   * @return The number of rows.
   */
  [[nodiscard]] auto n_rows() const noexcept -> index_type { return spec_type::get_n_rows(data_); }
  /**
   * @brief The dimension of the vectors in the dataset.
   *
   * @return The dimension.
   */
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    // NOTE: a compound dataset may not be able to tell its dimension from the encoded data alone
    return spec_type::get_dim(data_, manifest_);
  }

  /**
   * @brief A non-owning view of the data, such as mdspan.
   *
   * @return A mutable view of the data.
   */
  [[nodiscard]] auto data_view() noexcept -> data_view_type
  {
    return spec_type::get_data_view(data_);
  }
  /**
   * @brief A read-only view of the data.
   *
   * @return A read-only view of the data.
   */
  [[nodiscard]] auto data_const_view() const noexcept -> const_data_view_type
  {
    return spec_type::get_data_const_view(data_);
  }

  /**
   * @brief A non-owning view of the manifest, such as the two codebooks of a VPQ dataset.
   *
   * @return A mutable view of the manifest.
   */
  [[nodiscard]] auto manifest_view() noexcept -> manifest_view_type
    requires(is_compound)
  {
    return spec_type::get_manifest_view(manifest_);
  }
  /**
   * @brief A read-only view of the manifest.
   *
   * @return A read-only view of the manifest.
   */
  [[nodiscard]] auto const_manifest_view() const noexcept -> const_manifest_view_type
    requires(is_compound)
  {
    return spec_type::get_manifest_const_view(manifest_);
  }

 private:
  data_type data_;
  [[no_unique_address]] manifest_type manifest_;
};

}  // namespace CUVS_EXPORT cuvs
