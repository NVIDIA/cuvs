/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>   // get_device_for_address, copy_matrix
#include <raft/util/integer_utils.hpp>  // rounding up

#include <cuvs/core/export.hpp>
#include <raft/core/detail/macros.hpp>

#include <cuda_fp16.h>

#include <concepts>
#include <cstring>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>

namespace CUVS_EXPORT cuvs {
namespace core {

/**
 * @brief Spec-based `dataset` / `dataset_view`.
 *
 * `dataset<T,IdxT,SpecT>` and `dataset_view<T,IdxT,SpecT>` are single generic templates that know
 * nothing about any particular kind of dataset. They hold exactly one payload (`data_type` /
 * `view_type`, chosen by the spec) and expose only what every dataset has: `n_rows()`, `dim()`,
 * `as_matrix_view()`, `as_dataset_view()` and `data()`. Each is a one-line forward to one of the
 * three spec functions `get_data_view()`, `get_n_rows()` and `get_dim()`. Anything else a kind
 * needs (e.g. codebooks or quantizers of a compressed dataset) is state and methods of that
 * kind's payload type, reached through `data()`; `dataset`/`dataset_view` never name or branch on
 * it. Compressed kinds define their payloads and specs in their own headers (quantize/pq.hpp,
 * quantize/bbq.hpp) as children of these two structs. `dataset` and
 * `dataset_view` are deliberately two independent, non-inheriting types (no shared_ptr, no
 * "sometimes owning" object): `dataset` holds the owning payload, `dataset_view` the
 * corresponding non-owning payload.
 */

namespace detail {

// Default owning/view accessors for public dataset aliases.
template <typename T>
using device_owning_accessor = raft::device_accessor<raft::device_container_policy<T>>;

template <typename T>
using host_owning_accessor = raft::host_accessor<raft::host_container_policy<T>>;

template <typename T>
using device_view_accessor = raft::device_accessor<cuda::std::default_accessor<const T>>;

template <typename T>
using host_view_accessor = raft::host_accessor<cuda::std::default_accessor<const T>>;

/** View accessor paired with an owning dataset accessor (same residency). */
template <typename DataT, typename Accessor>
using dataset_view_accessor_for_owning = std::conditional_t<Accessor::is_device_accessible,
                                                            device_view_accessor<DataT>,
                                                            host_view_accessor<DataT>>;

/** Owning accessor paired with a view accessor (same residency). */
template <typename DataT, typename Accessor>
using dataset_owning_accessor_for_view = std::conditional_t<Accessor::is_device_accessible,
                                                            device_owning_accessor<DataT>,
                                                            host_owning_accessor<DataT>>;

// Accessor here is already device_owning_accessor<DataT> / host_owning_accessor<DataT> at every
// call site -- exactly the container policy raft::device_mdarray/host_mdarray default to for
// element type DataT -- so pass it straight through instead of re-deriving a
// raft::device_matrix/host_matrix from scratch.
template <typename DataT, typename IdxT, typename Accessor>
using dense_owning_matrix =
  raft::mdarray<DataT, raft::matrix_extent<IdxT>, raft::row_major, Accessor>;

template <typename DataT, typename IdxT, typename Accessor>
using dense_view_matrix = raft::mdspan<const DataT,
                                       raft::matrix_extent<IdxT>,
                                       raft::row_major,
                                       dataset_view_accessor_for_owning<DataT, Accessor>>;

// -----------------------------------------------------------------------------
// empty
// -----------------------------------------------------------------------------

template <typename IdxT>
struct empty_dataset_storage {
  uint32_t suggested_dim{};
  empty_dataset_storage() noexcept = default;
  explicit empty_dataset_storage(uint32_t dim) noexcept : suggested_dim(dim) {}
  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return 0; }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return suggested_dim; }
};

// -----------------------------------------------------------------------------
// dense row-major (logical dim may differ from row pitch; shared by padded & standard)
// -----------------------------------------------------------------------------

/**
 * Dense row-major payload shared by padded and standard dataset specs, owning and non-owning
 * alike: the owning payload passes its `raft::mdarray` as `BaseT`, the view payload passes the
 * `raft::mdspan`. Publicly inherits from `BaseT` so `view()`/`data_handle()`/`extent()` etc. are
 * reused as-is rather than hand-forwarded; `logical_dim_` is the only state this struct adds.
 *
 * Template parameters:
 * - BaseT: the owning matrix (`raft::mdarray`) or the non-owning row-major view (`raft::mdspan`).
 */
template <typename BaseT>
struct dense_row_major_storage : public BaseT {
  using index_type = typename BaseT::index_type;

  uint32_t logical_dim_{};

  // BaseT (mdarray/mdspan) also has its own stride(size_t); pull it back into scope since
  // declaring our own no-arg stride() below would otherwise hide it entirely (C++ name hiding),
  // and the body of that stride() itself needs to call the inherited one.
  using BaseT::stride;

  dense_row_major_storage() noexcept = default;

  // Takes BaseT by value so an owning matrix is moved in and a view is simply copied.
  explicit dense_row_major_storage(BaseT base) noexcept
    : BaseT(std::move(base)), logical_dim_(static_cast<uint32_t>(this->extent(1)))
  {
  }

  dense_row_major_storage(BaseT base, uint32_t logical_dim) noexcept
    : BaseT(std::move(base)), logical_dim_(logical_dim)
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type { return this->extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return logical_dim_; }
  [[nodiscard]] auto stride() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(BaseT::stride(0) > 0 ? BaseT::stride(0) : this->extent(1));
  }
};

/** Spec-side implementation shared by `padded_dataset_spec`/`standard_dataset_spec`; those two
 * stay distinct top-level types (identical bodies) purely so classification traits can tell them
 * apart -- exactly mirroring today's `padded_dataset_container`/`standard_dataset_container`,
 * which are likewise two differently-named tags over one shared storage implementation. */
template <typename ContainerPolicy>
struct dense_dataset_spec_impl {
  template <typename T, typename IdxT>
  struct apply {
    using value_type = std::remove_cv_t<T>;
    using index_type = std::remove_cv_t<IdxT>;
    using MatrixT    = dense_owning_matrix<T, IdxT, ContainerPolicy>;
    using ViewT      = dense_view_matrix<T, IdxT, ContainerPolicy>;
    using data_type  = dense_row_major_storage<MatrixT>;
    using view_type  = dense_row_major_storage<ViewT>;

    [[nodiscard]] static auto get_data_view(data_type const& data) noexcept -> view_type
    {
      return view_type(data.view(), data.dim());
    }
    template <typename AnyStorage>
    [[nodiscard]] static auto get_n_rows(AnyStorage const& data) noexcept -> index_type
    {
      return data.n_rows();
    }
    template <typename AnyStorage>
    [[nodiscard]] static auto get_dim(AnyStorage const& data) noexcept -> uint32_t
    {
      return data.dim();
    }
  };
};

}  // namespace detail

// -----------------------------------------------------------------------------
// Public specs -- the only place per-kind logic lives.
// -----------------------------------------------------------------------------

template <typename Accessor>
struct empty_dataset_spec {
  using accessor_type = Accessor;
  template <typename NewAccessor>
  using rebind_accessor = empty_dataset_spec<NewAccessor>;

  template <typename T, typename IdxT>
  struct apply {
    using value_type = std::remove_cv_t<T>;
    using index_type = std::remove_cv_t<IdxT>;
    using data_type  = detail::empty_dataset_storage<IdxT>;
    using view_type  = detail::empty_dataset_storage<IdxT>;

    [[nodiscard]] static auto get_data_view(data_type const& data) noexcept -> view_type
    {
      return data;
    }
    [[nodiscard]] static auto get_n_rows(data_type const& data) noexcept -> index_type
    {
      return static_cast<index_type>(data.n_rows());
    }
    [[nodiscard]] static auto get_dim(data_type const& data) noexcept -> uint32_t
    {
      return data.dim();
    }
  };
};

template <typename ContainerPolicy>
struct padded_dataset_spec {
  using accessor_type = ContainerPolicy;
  template <typename NewAccessor>
  using rebind_accessor = padded_dataset_spec<NewAccessor>;
  template <typename T, typename IdxT>
  struct apply : detail::dense_dataset_spec_impl<ContainerPolicy>::template apply<T, IdxT> {};
};

template <typename ContainerPolicy>
struct standard_dataset_spec {
  using accessor_type = ContainerPolicy;
  template <typename NewAccessor>
  using rebind_accessor = standard_dataset_spec<NewAccessor>;
  template <typename T, typename IdxT>
  struct apply : detail::dense_dataset_spec_impl<ContainerPolicy>::template apply<T, IdxT> {};
};

// -----------------------------------------------------------------------------
// dataset / dataset_view
// -----------------------------------------------------------------------------

/** Non-owning dataset view: holds only the view-shaped payload. Deliberately not derived from
 * `dataset` -- a view type holds "all view state" with no inheritance and no shared ownership tying
 * it to the owning type. Reuses the same `get_n_rows`/`get_dim` spec functions as `dataset`, fed
 * the view payload instead of the owning one. */
template <typename T, typename IdxT, typename SpecT>
struct dataset_view {
  using spec_type  = typename SpecT::template apply<T, IdxT>;
  using value_type = typename spec_type::value_type;
  using index_type = typename spec_type::index_type;
  using view_type  = typename spec_type::view_type;

  dataset_view() noexcept = default;

  // Already-constructed view payload -- the shape `as_dataset_view()` always constructs with, for
  // every kind. Not a template, so it's preferred over the forwarding constructor below whenever
  // both could apply.
  explicit dataset_view(view_type data_view) noexcept : data_view_{data_view} {}

  // Forward raw constructor args straight to view_type's own constructor (e.g. (ViewT, uint32_t
  // logical_dim) for dense, (uint32_t dim) for empty) -- preserves today's direct-construction
  // call sites (e.g. `device_padded_dataset_view<T,IdxT>(raw_mdspan, dim)`) unchanged. Explicit
  // only for a single argument, so that a raw matrix view never converts to a dataset view
  // implicitly (which would let it silently bind to overloads meant for dataset views), while
  // multi-argument brace initialization such as `return {view, dim};` keeps working.
  template <typename... Args>
  explicit(sizeof...(Args) == 1) dataset_view(Args&&... args)
    requires(std::is_constructible_v<view_type, Args...>)
    : data_view_(std::forward<Args>(args)...)
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type
  {
    return spec_type::get_n_rows(data_view_);
  }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return spec_type::get_dim(data_view_); }
  [[nodiscard]] auto as_matrix_view() const noexcept -> view_type { return data_view_; }

  /** The view payload; kind-specific state and methods are reached through it. */
  [[nodiscard]] auto data() const noexcept -> view_type const& { return data_view_; }
  [[nodiscard]] auto data() noexcept -> view_type& { return data_view_; }

 private:
  view_type data_view_{};
};

/** Owning dataset: value-held payload (no shared_ptr -- exclusive ownership). Every member is a
 * one-line forward to `spec_type::get_*` or to the payload; all per-kind state and logic lives in
 * the spec's `data_type`, never inside this struct. */
template <typename T, typename IdxT, typename SpecT>
struct dataset {
  using spec_type  = typename SpecT::template apply<T, IdxT>;
  using value_type = typename spec_type::value_type;
  using index_type = typename spec_type::index_type;
  using data_type  = typename spec_type::data_type;

  // Forward constructor args straight to data_type's own constructor (e.g. (MatrixT&&, uint32_t
  // logical_dim) for dense, (uint32_t dim) for empty, or whatever a compressed kind's payload
  // takes). Explicit only for a single argument, so that a lone value never converts to a dataset
  // implicitly, while multi-argument brace initialization such as `return {matrix, dim};` keeps
  // working as it did before datasets were generic.
  template <typename... Args>
  explicit(sizeof...(Args) == 1) dataset(Args&&... args)
    requires(std::is_constructible_v<data_type, Args...>)
    : data_(std::forward<Args>(args)...)
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type { return spec_type::get_n_rows(data_); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return spec_type::get_dim(data_); }
  /** The spec-defined non-owning view of the payload (an mdspan derivative for dense kinds). */
  [[nodiscard]] auto as_matrix_view() const noexcept { return spec_type::get_data_view(data_); }
  [[nodiscard]] auto as_dataset_view() const noexcept -> dataset_view<T, IdxT, SpecT>
  {
    return dataset_view<T, IdxT, SpecT>(as_matrix_view());
  }

  /** The owning payload; kind-specific state and methods are reached through it. */
  [[nodiscard]] auto data() const noexcept -> data_type const& { return data_; }
  [[nodiscard]] auto data() noexcept -> data_type& { return data_; }

 private:
  data_type data_;
};

/**
 * @brief Aliases for concrete `dataset` / `dataset_view` layouts.
 */
template <typename IdxT>
using device_empty_dataset =
  dataset<void, IdxT, empty_dataset_spec<detail::device_view_accessor<char>>>;

template <typename IdxT>
using device_empty_dataset_view =
  dataset_view<void, IdxT, empty_dataset_spec<detail::device_view_accessor<char>>>;

template <typename IdxT>
using host_empty_dataset =
  dataset<void, IdxT, empty_dataset_spec<detail::host_view_accessor<char>>>;

template <typename IdxT>
using host_empty_dataset_view =
  dataset_view<void, IdxT, empty_dataset_spec<detail::host_view_accessor<char>>>;

template <typename DataT, typename IdxT>
using device_padded_dataset =
  dataset<DataT, IdxT, padded_dataset_spec<detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using device_padded_dataset_view =
  dataset_view<DataT, IdxT, padded_dataset_spec<detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_padded_dataset =
  dataset<DataT, IdxT, padded_dataset_spec<detail::host_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_padded_dataset_view =
  dataset_view<DataT, IdxT, padded_dataset_spec<detail::host_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using device_standard_dataset =
  dataset<DataT, IdxT, standard_dataset_spec<detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using device_standard_dataset_view =
  dataset_view<DataT, IdxT, standard_dataset_spec<detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_standard_dataset =
  dataset<DataT, IdxT, standard_dataset_spec<detail::host_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_standard_dataset_view =
  dataset_view<DataT, IdxT, standard_dataset_spec<detail::host_owning_accessor<DataT>>>;

// Maps a dataset view type to its owning (allocating) dataset counterpart. Trivial and total under
// the Spec design: the owning type for `dataset_view<T,IdxT,SpecT>` is always
// `dataset<T,IdxT,SpecT>`
// -- no per-kind specialization table needed (unlike the old Container-tagged design).
template <typename DatasetViewT>
struct owning_dataset_for_view;

template <typename T, typename IdxT, typename SpecT>
struct owning_dataset_for_view<dataset_view<T, IdxT, SpecT>> {
  using type = dataset<T, IdxT, SpecT>;
};

template <typename DatasetViewT>
using owning_dataset_for_view_t = typename owning_dataset_for_view<DatasetViewT>::type;

// -----------------------------------------------------------------------------
// Spec-kind classification. Only the kinds that live in this header are named here; every other
// kind (e.g. the compressed kinds in quantize/pq.hpp and quantize/bbq.hpp) defines its own spec
// predicate next to its own spec and reuses `dataset_view_has_spec_v` below.
// -----------------------------------------------------------------------------

template <typename SpecT>
struct is_empty_spec : std::false_type {};
template <typename Accessor>
struct is_empty_spec<empty_dataset_spec<Accessor>> : std::true_type {};
template <typename SpecT>
inline constexpr bool is_empty_spec_v = is_empty_spec<SpecT>::value;

template <typename SpecT>
struct is_padded_spec : std::false_type {};
template <typename ContainerPolicy>
struct is_padded_spec<padded_dataset_spec<ContainerPolicy>> : std::true_type {};
template <typename SpecT>
inline constexpr bool is_padded_spec_v = is_padded_spec<SpecT>::value;

template <typename SpecT>
struct is_standard_spec : std::false_type {};
template <typename ContainerPolicy>
struct is_standard_spec<standard_dataset_spec<ContainerPolicy>> : std::true_type {};
template <typename SpecT>
inline constexpr bool is_standard_spec_v = is_standard_spec<SpecT>::value;

/** Owning-side kind traits (true for both `dataset<...>` and `dataset_view<...>` of that kind). */
template <typename DatasetT>
struct is_padded_dataset : std::false_type {};
template <typename T, typename IdxT, typename SpecT>
struct is_padded_dataset<dataset<T, IdxT, SpecT>> : std::bool_constant<is_padded_spec_v<SpecT>> {};
template <typename T, typename IdxT, typename SpecT>
struct is_padded_dataset<dataset_view<T, IdxT, SpecT>>
  : std::bool_constant<is_padded_spec_v<SpecT>> {};
template <typename DatasetT>
inline constexpr bool is_padded_dataset_v = is_padded_dataset<DatasetT>::value;

template <typename DatasetT>
struct is_standard_dataset : std::false_type {};
template <typename T, typename IdxT, typename SpecT>
struct is_standard_dataset<dataset<T, IdxT, SpecT>>
  : std::bool_constant<is_standard_spec_v<SpecT>> {};
template <typename T, typename IdxT, typename SpecT>
struct is_standard_dataset<dataset_view<T, IdxT, SpecT>>
  : std::bool_constant<is_standard_spec_v<SpecT>> {};
template <typename DatasetT>
inline constexpr bool is_standard_dataset_v = is_standard_dataset<DatasetT>::value;

// -----------------------------------------------------------------------------
// Dataset view compile-time classification (replaces runtime std::variant dispatch).
// -----------------------------------------------------------------------------

/** Any type that behaves like a dataset: it exposes a row count (`n_rows()`) and a logical
 * dimension (`dim()`). This is a structural check, so owning datasets and dataset views both
 * satisfy it. To ask whether a type is literally a `dataset_view<...>`, use `is_dataset_view_v`. */
template <typename V, typename IdxT = int64_t>
concept dataset_like = requires(V const& v) {
  { v.n_rows() } -> std::convertible_to<IdxT>;
  { v.dim() } -> std::convertible_to<uint32_t>;
};

template <typename V>
using dataset_view_type_t = std::remove_cvref_t<V>;

/** True for any `dataset_view<...>` specialization. Evaluates to `false` (never a hard error) for
 * everything else, e.g. a plain mdspan passed to a deprecated `build(matrix_view)` overload. */
template <typename V>
struct is_dataset_view : std::false_type {};
template <typename T, typename IdxT, typename SpecT>
struct is_dataset_view<dataset_view<T, IdxT, SpecT>> : std::true_type {};
template <typename V>
inline constexpr bool is_dataset_view_v = is_dataset_view<dataset_view_type_t<V>>::value;

/** True when `V` is a `dataset_view` whose spec satisfies the predicate `SpecPred<SpecT>::value`.
 * This is how a kind that lives outside this header classifies its own views. */
template <typename V, template <typename> typename SpecPred>
struct dataset_view_has_spec : std::false_type {};
template <typename T, typename IdxT, typename SpecT, template <typename> typename SpecPred>
struct dataset_view_has_spec<dataset_view<T, IdxT, SpecT>, SpecPred>
  : std::bool_constant<SpecPred<SpecT>::value> {};
template <typename V, template <typename> typename SpecPred>
inline constexpr bool dataset_view_has_spec_v =
  dataset_view_has_spec<dataset_view_type_t<V>, SpecPred>::value;

/** True when the dataset view accessor is device-accessible. */
template <typename V>
struct dataset_view_is_device_accessible : std::false_type {};

template <typename T, typename IdxT, typename SpecT>
struct dataset_view_is_device_accessible<dataset_view<T, IdxT, SpecT>>
  : std::bool_constant<SpecT::accessor_type::is_device_accessible> {};

template <typename V>
inline constexpr bool dataset_view_is_device_accessible_v =
  dataset_view_is_device_accessible<dataset_view_type_t<V>>::value;

template <typename V>
inline constexpr bool is_device_empty_dataset_view_v =
  dataset_view_has_spec_v<V, is_empty_spec> && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_empty_dataset_view_v =
  dataset_view_has_spec_v<V, is_empty_spec> && !dataset_view_is_device_accessible_v<V>;

/** True for any empty dataset view (device or host). */
template <typename V>
inline constexpr bool is_empty_dataset_view_v =
  is_device_empty_dataset_view_v<V> || is_host_empty_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_padded_dataset_view_v =
  dataset_view_has_spec_v<V, is_padded_spec> && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_padded_dataset_view_v =
  dataset_view_has_spec_v<V, is_padded_spec> && !dataset_view_is_device_accessible_v<V>;

/** True for either `device_padded_dataset_view` or `host_padded_dataset_view`. */
template <typename V>
inline constexpr bool is_padded_dataset_view_v =
  is_device_padded_dataset_view_v<V> || is_host_padded_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_standard_dataset_view_v =
  dataset_view_has_spec_v<V, is_standard_spec> && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_standard_dataset_view_v =
  dataset_view_has_spec_v<V, is_standard_spec> && !dataset_view_is_device_accessible_v<V>;

/** True for either `device_standard_dataset_view` or `host_standard_dataset_view`. */
template <typename V>
inline constexpr bool is_standard_dataset_view_v =
  is_device_standard_dataset_view_v<V> || is_host_standard_dataset_view_v<V>;

/** True for any device-resident dataset view. */
template <typename V>
inline constexpr bool is_device_dataset_view_v =
  is_dataset_view_v<V> && dataset_view_is_device_accessible_v<V>;

/** True for any host-resident dataset view. */
template <typename V>
inline constexpr bool is_host_dataset_view_v =
  is_dataset_view_v<V> && !dataset_view_is_device_accessible_v<V>;

/**
 * Generic accessor retargeting while preserving the spec kind and value/index types:
 * `dataset<T, IdxT, SpecT<..., OldAccessor>>      -> dataset<T, IdxT, SpecT<..., NewAccessor>>`
 * `dataset_view<T, IdxT, SpecT<..., OldAccessor>> -> dataset_view<T, IdxT, SpecT<...,
 * NewAccessor>>`
 * Every spec provides `rebind_accessor<NewAccessor>` for this, so this header does not need to
 * know about any particular kind.
 */
template <typename DatasetLikeT, typename NewAccessor>
struct with_accessor;

template <typename T, typename IdxT, typename SpecT, typename NewAccessor>
struct with_accessor<dataset<T, IdxT, SpecT>, NewAccessor> {
  using type = dataset<T, IdxT, typename SpecT::template rebind_accessor<NewAccessor>>;
};

template <typename T, typename IdxT, typename SpecT, typename NewAccessor>
struct with_accessor<dataset_view<T, IdxT, SpecT>, NewAccessor> {
  using type = dataset_view<T, IdxT, typename SpecT::template rebind_accessor<NewAccessor>>;
};

template <typename DatasetLikeT, typename NewAccessor>
using with_accessor_t =
  typename with_accessor<dataset_view_type_t<DatasetLikeT>, NewAccessor>::type;

/** Map any host accessor to its device counterpart (same payload policy). */
template <typename Accessor>
struct to_device_accessor {
  using type = Accessor;
};

template <typename T>
struct to_device_accessor<detail::host_view_accessor<T>> {
  using type = detail::device_view_accessor<T>;
};

template <typename T>
struct to_device_accessor<detail::host_owning_accessor<T>> {
  using type = detail::device_owning_accessor<T>;
};

template <typename Accessor>
using to_device_accessor_t = typename to_device_accessor<Accessor>::type;

/** Maps a host dataset view type to its device-resident counterpart. */
template <typename HostViewT>
struct device_counterpart;

template <typename T, typename IdxT, typename SpecT>
struct device_counterpart<dataset_view<T, IdxT, SpecT>> {
  using type = with_accessor_t<dataset_view<T, IdxT, SpecT>,
                               to_device_accessor_t<typename SpecT::accessor_type>>;
};

template <typename HostViewT>
using device_counterpart_t = typename device_counterpart<dataset_view_type_t<HostViewT>>::type;

/**
 * True when a host view `H` and device view `D` represent the same storage kind and differ
 * only in residency (host vs. device). Used by host/device conversion helpers.
 */
template <typename HostViewT, typename DeviceViewT>
inline constexpr bool compatible_host_device_dataset_views_v =
  is_host_dataset_view_v<HostViewT> && is_device_dataset_view_v<DeviceViewT> &&
  std::is_same_v<device_counterpart_t<HostViewT>, dataset_view_type_t<DeviceViewT>>;

/** True for device padded or standard (dense row-major) dataset views. */
template <typename V>
inline constexpr bool is_dense_row_major_device_dataset_view_v =
  is_device_padded_dataset_view_v<V> || is_device_standard_dataset_view_v<V>;

/** True for host or device padded or standard (dense row-major) dataset views. */
template <typename V>
inline constexpr bool is_dense_row_major_dataset_view_v =
  is_padded_dataset_view_v<V> || is_standard_dataset_view_v<V>;

/** Element type `T` of a dataset view, deduced from the view. Trivial under the Spec design: every
 * `dataset_view<T,IdxT,SpecT>` already carries `T` directly. */
template <typename V>
using dataset_view_value_t = typename dataset_view_type_t<V>::value_type;

// -----------------------------------------------------------------------------
// Padded row width in elements (shared by the make_*_padded_dataset* factories and row-width
// checks).
// -----------------------------------------------------------------------------

/**
 * @brief Minimum row width in elements (the leading dimension) for `logical_columns` feature
 *        columns, such that each row occupies a whole multiple of `align_bytes` bytes (default 16,
 *        combined with `sizeof` of the element type).
 */
[[nodiscard]] inline uint32_t padded_row_width(uint32_t logical_columns,
                                               std::size_t sizeof_value,
                                               uint32_t align_bytes = 16)
{
  return static_cast<uint32_t>(
    raft::round_up_safe<std::size_t>(static_cast<std::size_t>(logical_columns) * sizeof_value,
                                     std::lcm(align_bytes, static_cast<uint32_t>(sizeof_value))) /
    sizeof_value);
}

template <typename ValueT>
[[nodiscard]] inline uint32_t padded_row_width(uint32_t logical_columns, uint32_t align_bytes = 16)
{
  return padded_row_width(logical_columns, sizeof(ValueT), align_bytes);
}

/** Actual row width in elements (leading dimension) of a 2D row-major matrix view. */
template <typename T, typename I, typename L>
[[nodiscard]] inline uint32_t matrix_actual_row_width(raft::device_matrix_view<T, I, L> m)
{
  return m.stride(0) > 0 ? static_cast<uint32_t>(m.stride(0)) : static_cast<uint32_t>(m.extent(1));
}

template <typename T, typename I, typename L>
[[nodiscard]] inline uint32_t matrix_actual_row_width(raft::host_matrix_view<T, I, L> m)
{
  return m.stride(0) > 0 ? static_cast<uint32_t>(m.stride(0)) : static_cast<uint32_t>(m.extent(1));
}

/**
 * @brief True if the matrix's row width in elements equals `padded_row_width` for `m.extent(1)`
 *        and element type `T`, i.e. its rows are already padded.
 */
template <typename T, typename I, typename L>
[[nodiscard]] inline bool matrix_has_padded_row_width(raft::device_matrix_view<T, I, L> m,
                                                      uint32_t align_bytes = 16)
{
  using value_type = std::remove_const_t<T>;
  const uint32_t need =
    padded_row_width<value_type>(static_cast<uint32_t>(m.extent(1)), align_bytes);
  return matrix_actual_row_width(m) == need;
}

template <typename T, typename I, typename L>
[[nodiscard]] inline bool matrix_has_padded_row_width(raft::host_matrix_view<T, I, L> m,
                                                      uint32_t align_bytes = 16)
{
  using value_type = std::remove_const_t<T>;
  const uint32_t need =
    padded_row_width<value_type>(static_cast<uint32_t>(m.extent(1)), align_bytes);
  return matrix_actual_row_width(m) == need;
}

namespace detail {

template <typename SrcT>
[[nodiscard]] inline uint32_t mdspan_row_stride_elements(SrcT const& src)
{
  return src.stride(0) > 0 ? static_cast<uint32_t>(src.stride(0))
                           : static_cast<uint32_t>(src.extent(1));
}

template <typename ValueT, typename SrcT>
[[nodiscard]] inline ValueT* expect_device_accessible_data_handle(SrcT const& src,
                                                                  char const* error_msg)
{
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  // `devicePointer` is relative to the *current* device: it is null for an allocation owned by
  // another device without peer access, even though that allocation is perfectly usable once the
  // caller switches to the owning device (as the multi-GPU paths do). Accept device and managed
  // allocations on their own merit and only consult `devicePointer` for host memory, which needs a
  // mapping to be reachable at all.
  if (ptr_attrs.type == cudaMemoryTypeDevice || ptr_attrs.type == cudaMemoryTypeManaged) {
    return const_cast<ValueT*>(src.data_handle());
  }
  auto* device_ptr = reinterpret_cast<ValueT*>(ptr_attrs.devicePointer);
  RAFT_EXPECTS(device_ptr != nullptr, "%s", error_msg);
  return device_ptr;
}

template <typename ValueT, typename IndexT, typename ViewT, typename SrcT>
[[nodiscard]] inline ViewT make_device_dense_row_major_view_from_src(SrcT const& src,
                                                                     uint32_t logical_dim)
{
  auto* device_ptr = expect_device_accessible_data_handle<ValueT>(
    src, "make_device_*_dataset_view: source must be device-accessible.");
  auto v = raft::make_device_matrix_view(
    device_ptr, src.extent(0), static_cast<IndexT>(mdspan_row_stride_elements(src)));
  return ViewT(v, logical_dim);
}

template <typename ValueT, typename IndexT, typename ViewT, typename SrcT>
[[nodiscard]] inline ViewT make_host_dense_row_major_view_from_src(SrcT const& src,
                                                                   uint32_t logical_dim)
{
  RAFT_EXPECTS(raft::get_device_for_address(src.data_handle()) == -1,
               "make_host_*_dataset_view: source must be host-accessible.");
  auto v = raft::make_host_matrix_view(const_cast<ValueT*>(src.data_handle()),
                                       src.extent(0),
                                       static_cast<IndexT>(mdspan_row_stride_elements(src)));
  return ViewT(v, logical_dim);
}

template <typename DatasetT, typename ValueT, typename IndexT, typename SrcT>
auto make_device_dense_row_major_dataset_from_src(raft::resources const& res,
                                                  SrcT const& src,
                                                  uint32_t logical_dim,
                                                  uint32_t target_stride,
                                                  char const* view_factory_name)
  -> std::unique_ptr<DatasetT>
{
  uint32_t const src_stride = mdspan_row_stride_elements(src);
  RAFT_EXPECTS(logical_dim <= target_stride,
               "logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(logical_dim),
               static_cast<unsigned>(target_stride));
  RAFT_EXPECTS(static_cast<uint32_t>(src.extent(1)) <= target_stride,
               "Source row length must not exceed required stride.");
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  bool const device_src =
    (ptr_attrs.type == cudaMemoryTypeDevice) || (ptr_attrs.type == cudaMemoryTypeManaged);
  if (device_src && src_stride == target_stride) {
    RAFT_EXPECTS(false,
                 "source is device and stride is already correct. "
                 "Use %s() to get a view instead.",
                 view_factory_name);
  }
  auto out_array = raft::make_device_matrix<ValueT, IndexT>(res, src.extent(0), target_stride);
  RAFT_CUDA_TRY(cudaMemsetAsync(out_array.data_handle(),
                                0,
                                out_array.size() * sizeof(ValueT),
                                raft::resource::get_cuda_stream(res).get()));
  raft::copy_matrix(out_array.data_handle(),
                    target_stride,
                    src.data_handle(),
                    src_stride,
                    logical_dim,
                    src.extent(0),
                    raft::resource::get_cuda_stream(res));
  return std::make_unique<DatasetT>(std::move(out_array), logical_dim);
}

template <typename DatasetT, typename ValueT, typename IndexT, typename SrcT>
auto make_host_dense_row_major_dataset_from_src(raft::resources const& res,
                                                SrcT const& src,
                                                uint32_t logical_dim,
                                                uint32_t target_stride,
                                                char const* view_factory_name)
  -> std::unique_ptr<DatasetT>
{
  uint32_t const src_stride = mdspan_row_stride_elements(src);
  constexpr bool device_src = SrcT::accessor_type::is_device_accessible;
  RAFT_EXPECTS(logical_dim <= target_stride,
               "logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(logical_dim),
               static_cast<unsigned>(target_stride));
  if (!device_src && src_stride == target_stride) {
    RAFT_EXPECTS(false,
                 "source stride is already correct. Use %s() to get a view instead.",
                 view_factory_name);
  }
  RAFT_EXPECTS(static_cast<uint32_t>(src.extent(1)) <= target_stride,
               "Source row length must not exceed required stride.");
  auto out_array = raft::make_host_matrix<ValueT, IndexT>(src.extent(0), target_stride);
  std::memset(out_array.data_handle(), 0, out_array.size() * sizeof(ValueT));
  raft::copy_matrix(out_array.data_handle(),
                    target_stride,
                    src.data_handle(),
                    src_stride,
                    logical_dim,
                    src.extent(0),
                    raft::resource::get_cuda_stream(res));
  if (device_src) { raft::resource::sync_stream(res); }
  return std::make_unique<DatasetT>(std::move(out_array), logical_dim);
}

}  // namespace detail

template <typename SrcT>
auto make_device_padded_dataset_view(const raft::resources& res,
                                     SrcT const& src,
                                     uint32_t align_bytes = 16)
  -> device_padded_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  uint32_t required_stride =
    padded_row_width<value_type>(static_cast<uint32_t>(src.extent(1)), align_bytes);
  RAFT_EXPECTS(
    detail::mdspan_row_stride_elements(src) == required_stride,
    "make_device_padded_dataset_view: stride is incorrect (required stride for alignment). "
    "Use make_device_padded_dataset() to get an owning padded copy.");
  return detail::make_device_dense_row_major_view_from_src<
    value_type,
    index_type,
    device_padded_dataset_view<value_type, index_type>>(src, static_cast<uint32_t>(src.extent(1)));
}

template <typename SrcT>
auto make_device_padded_dataset(const raft::resources& res,
                                SrcT const& src,
                                uint32_t align_bytes = 16)
  -> std::unique_ptr<device_padded_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type               = typename SrcT::value_type;
  using index_type               = typename SrcT::index_type;
  uint32_t const logical_dim     = static_cast<uint32_t>(src.extent(1));
  uint32_t const required_stride = padded_row_width<value_type>(logical_dim, align_bytes);
  return detail::make_device_dense_row_major_dataset_from_src<
    device_padded_dataset<value_type, index_type>,
    value_type,
    index_type>(res, src, logical_dim, required_stride, "make_device_padded_dataset_view");
}

template <typename SrcT>
auto make_host_padded_dataset_view(SrcT const& src, uint32_t align_bytes = 16)
  -> host_padded_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  uint32_t required_stride =
    padded_row_width<value_type>(static_cast<uint32_t>(src.extent(1)), align_bytes);
  RAFT_EXPECTS(
    detail::mdspan_row_stride_elements(src) == required_stride,
    "make_host_padded_dataset_view: stride is incorrect (required stride for alignment). "
    "Use make_host_padded_dataset() to get an owning padded copy.");
  return detail::make_host_dense_row_major_view_from_src<
    value_type,
    index_type,
    host_padded_dataset_view<value_type, index_type>>(src, static_cast<uint32_t>(src.extent(1)));
}

template <typename SrcT>
auto make_host_padded_dataset(const raft::resources& res,
                              SrcT const& src,
                              uint32_t align_bytes = 16)
  -> std::unique_ptr<host_padded_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type               = typename SrcT::value_type;
  using index_type               = typename SrcT::index_type;
  uint32_t const logical_dim     = static_cast<uint32_t>(src.extent(1));
  uint32_t const required_stride = padded_row_width<value_type>(logical_dim, align_bytes);
  return detail::make_host_dense_row_major_dataset_from_src<
    host_padded_dataset<value_type, index_type>,
    value_type,
    index_type>(res, src, logical_dim, required_stride, "make_host_padded_dataset_view");
}

template <typename SrcT>
auto make_device_standard_dataset_view(SrcT const& src)
  -> device_standard_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  return detail::make_device_dense_row_major_view_from_src<
    value_type,
    index_type,
    device_standard_dataset_view<value_type, index_type>>(src,
                                                          static_cast<uint32_t>(src.extent(1)));
}

/**
 * @brief Create an owning device standard dataset with explicit row layout.
 *
 * Internal use only: the sole caller today deserializes a dataset from disk and must pass
 * wire-format `(logical_dim, stride)` because the deserialized host buffer is tight `[n_rows x
 * dim]` while the on-disk stride may be larger. Do not call from user code; prefer
 * `make_device_standard_dataset_view()` when wrapping existing correctly-strided storage.
 */
template <typename SrcT>
auto make_device_standard_dataset(const raft::resources& res,
                                  SrcT const& src,
                                  uint32_t logical_dim,
                                  uint32_t target_stride)
  -> std::unique_ptr<device_standard_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  return detail::make_device_dense_row_major_dataset_from_src<
    device_standard_dataset<value_type, index_type>,
    value_type,
    index_type>(res, src, logical_dim, target_stride, "make_device_standard_dataset_view");
}

template <typename SrcT>
auto make_host_standard_dataset_view(SrcT const& src)
  -> host_standard_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  return detail::make_host_dense_row_major_view_from_src<
    value_type,
    index_type,
    host_standard_dataset_view<value_type, index_type>>(src, static_cast<uint32_t>(src.extent(1)));
}

}  // namespace core
}  // namespace CUVS_EXPORT cuvs
