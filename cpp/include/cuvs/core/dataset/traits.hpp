/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * Compile-time questions about dataset types that do not depend on any particular algorithm: what
 * a dataset view must look like to be accepted as one, which spec it was built from, and which
 * owning dataset a read-only view belongs to.
 */

#include <cuvs/core/dataset/dense.hpp>
#include <cuvs/core/dataset/empty.hpp>
#include <cuvs/core/dataset/type.hpp>
#include <cuvs/core/dataset/view.hpp>
#include <cuvs/core/export.hpp>

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace CUVS_EXPORT cuvs {

/** Any non-owning dataset view exposing row count and logical dimension. */
template <typename V, typename IdxT = int64_t>
concept dataset_like = requires(V const& v) {
  { v.n_rows() } -> std::convertible_to<IdxT>;
  { v.dim() } -> std::convertible_to<uint32_t>;
};

namespace detail {

template <typename V>
struct spec_of {
  using type = void;
};
template <typename T, typename IdxT, typename SpecT>
struct spec_of<dataset<T, IdxT, SpecT>> {
  using type = SpecT;
};
template <typename T, typename IdxT, typename SpecT>
struct spec_of<dataset_view<T, IdxT, SpecT>> {
  using type = SpecT;
};

template <typename SpecT>
struct is_dense_spec : std::false_type {};
template <typename LayoutPolicy, typename ContainerPolicy>
struct is_dense_spec<mdarray_spec<LayoutPolicy, ContainerPolicy>> : std::true_type {};

}  // namespace detail

/** True for the empty dataset (view): no rows, only a dimension. */
template <typename V>
inline constexpr bool is_empty_dataset_view_v =
  std::is_same_v<typename detail::spec_of<std::remove_cvref_t<V>>::type, empty_spec>;

/** True for the dense datasets (views), contiguous and padded alike. */
template <typename V>
inline constexpr bool is_dense_dataset_view_v =
  detail::is_dense_spec<typename detail::spec_of<std::remove_cvref_t<V>>::type>::value;

/**
 * Maps a dataset view type to its owning (allocating) dataset counterpart. Used by serialization to
 * type the out_dataset output parameter; a kind that is not a plain spec-based dataset adds a
 * specialization.
 */
template <typename DatasetViewT>
struct owning_dataset_for_view;

template <typename T, typename IdxT, typename SpecT>
struct owning_dataset_for_view<dataset_view<const T, IdxT, SpecT>> {
  using type = dataset<T, IdxT, SpecT>;
};

template <typename DatasetViewT>
using owning_dataset_for_view_t = typename owning_dataset_for_view<DatasetViewT>::type;

}  // namespace CUVS_EXPORT cuvs
