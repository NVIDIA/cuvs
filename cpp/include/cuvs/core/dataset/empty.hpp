/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/type.hpp>
#include <cuvs/core/export.hpp>

#include <cstdint>
#include <type_traits>

namespace CUVS_EXPORT cuvs {

/**
 * Empty dataset specification: contains no data, but keeps the dimension of the dataset implement
 * the common interface.
 */
struct empty_spec {
  struct empty_dataset_rep {
    uint32_t dim;
  };

  template <typename T, typename IdxT>
  struct apply {
    using data_type      = empty_dataset_rep;
    using data_view_type = empty_dataset_rep;
    // NOTE: not `const empty_dataset_rep`: a read-only view holds this as a member, and a const
    // member would delete its assignment operator.
    using const_data_view_type = empty_dataset_rep;
    using value_type           = std::remove_cv_t<T>;
    using index_type           = std::remove_cv_t<IdxT>;

    [[nodiscard]] static auto get_data_view(data_type& data) noexcept -> data_view_type
    {
      return data;
    }
    // NOTE: the view of an empty dataset is the storage itself, so one overload serves both the
    // owning dataset and the view.
    [[nodiscard]] static auto get_data_const_view(const data_type& data) noexcept
      -> const_data_view_type
    {
      return data;
    }
    // NOTE: the shape accessors take the data in either form - the storage an owning dataset holds,
    // or the view a `dataset_view` holds. A plain spec defines no dictionary either, so it ignores
    // whatever the dataset puts in that slot instead.
    [[nodiscard]] static auto get_n_rows(const auto&) noexcept -> index_type { return 0; }
    [[nodiscard]] static auto get_dim(const auto& data, const auto&) noexcept -> uint32_t
    {
      return data.dim;
    }
  };
};

template <typename T, typename IdxT>
using empty_dataset = dataset<T, IdxT, empty_spec>;

}  // namespace CUVS_EXPORT cuvs
