/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/type.hpp>
#include <cuvs/core/export.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan.hpp>

#include <cuda/std/mdspan>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <type_traits>

namespace CUVS_EXPORT cuvs {

/**
 * Dense plain/padded dataset implemented via raft::mdarray.
 */
template <typename LayoutPolicy, typename ContainerPolicy>
struct mdarray_spec {
  template <typename T, typename IdxT>
  struct apply {
    // NOTE: index type != extents type
    // The index type can vary depending on the use case; we often may want to store indices in
    // 32-bit slots to save memory on large datasets. The extents are always fixed to 64-bit signed
    // integers to simplify the conversion and avoid integer overflow issues.
    using value_type = std::remove_cv_t<T>;
    using index_type = std::remove_cv_t<IdxT>;
    using data_type =
      raft::mdarray<value_type, raft::matrix_extent<int64_t>, LayoutPolicy, ContainerPolicy>;
    using data_view_type       = typename data_type::view_type;
    using const_data_view_type = typename data_type::const_view_type;

    [[nodiscard]] static auto get_data_view(data_type& data) noexcept -> data_view_type
    {
      return data.view();
    }

    [[nodiscard]] static auto get_data_const_view(const data_type& data) noexcept
      -> const_data_view_type
    {
      return data.view();
    }
    // A mutable mdspan does not decay to a read-only one on its own; it has to be rebuilt over an
    // accessor of `const T`.
    [[nodiscard]] static auto get_data_const_view(data_view_type data) noexcept
      -> const_data_view_type
    {
      return raft::make_const_mdspan(data);
    }
    // NOTE: the shape accessors take the data in either form - the storage an owning dataset holds,
    // or the view a `dataset_view` holds. A plain spec defines no dictionary either, so it ignores
    // whatever the dataset puts in that slot instead.
    [[nodiscard]] static auto get_n_rows(const auto& data) noexcept -> index_type
    {
      return static_cast<index_type>(data.extent(0));
    }
    [[nodiscard]] static auto get_dim(const auto& data, const auto&) noexcept -> uint32_t
    {
      return static_cast<uint32_t>(data.extent(1));
    }
  };
};

/**
 * Row-major rows padded to 16 bytes, the alignment the CAGRA kernels require. A padded matrix keeps
 * the logical dimension as its second extent; the row stride follows from the layout.
 */
template <typename T>
using padded_layout = cuda::std::layout_right_padded<16 / std::gcd<int, int>(16, sizeof(T))>;

template <typename T, typename IdxT, typename ContainerPolicy>
using contiguous_dataset =
  dataset<T, IdxT, mdarray_spec<raft::layout_c_contiguous, ContainerPolicy>>;
template <typename T, typename IdxT, typename ContainerPolicy>
using padded_dataset = dataset<T, IdxT, mdarray_spec<padded_layout<T>, ContainerPolicy>>;
template <typename T, typename IdxT>
using device_contiguous_dataset =
  contiguous_dataset<T, IdxT, raft::device_accessor<raft::device_container_policy<T>>>;
template <typename T, typename IdxT>
using device_padded_dataset =
  padded_dataset<T, IdxT, raft::device_accessor<raft::device_container_policy<T>>>;
template <typename T, typename IdxT>
using host_contiguous_dataset =
  contiguous_dataset<T, IdxT, raft::host_accessor<raft::host_container_policy<T>>>;
template <typename T, typename IdxT>
using host_padded_dataset =
  padded_dataset<T, IdxT, raft::host_accessor<raft::host_container_policy<T>>>;

}  // namespace CUVS_EXPORT cuvs
