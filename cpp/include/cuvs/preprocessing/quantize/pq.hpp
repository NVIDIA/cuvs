/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/core/dataset.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/util/cuda_data_type.hpp>

#include <cuda_runtime.h>
#include <cuvs/core/export.hpp>
#include <type_traits>
#include <variant>
#ifdef __cpp_lib_bitops
#include <bit>
#endif

namespace CUVS_EXPORT cuvs {
namespace preprocessing {
namespace quantize {
namespace pq {

/**
 * @defgroup pq Product Quantizer utilities
 * @{
 */

/** Alias for the variant holding either balanced or regular k-means parameters. */
using kmeans_params_variant =
  std::variant<cuvs::cluster::kmeans::balanced_params, cuvs::cluster::kmeans::params>;

/**
 * @brief Product Quantizer parameters.
 */
struct params {
  /**
   * Simplified constructor that will build an appropriate kmeans params object.
   */
  params(uint32_t pq_bits,
         uint32_t pq_dim,
         bool use_subspaces,
         bool use_vq,
         uint32_t vq_n_centers,
         uint32_t kmeans_n_iters,
         cuvs::cluster::kmeans::kmeans_type pq_kmeans_type =
           cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
         uint32_t max_train_points_per_pq_code    = 256,
         uint32_t max_train_points_per_vq_cluster = 1024)
    : pq_bits(pq_bits),
      pq_dim(pq_dim),
      use_subspaces(use_subspaces),
      use_vq(use_vq),
      vq_n_centers(vq_n_centers),
      kmeans_params(
        pq_kmeans_type == cuvs::cluster::kmeans::kmeans_type::KMeansBalanced
          ? kmeans_params_variant{cuvs::cluster::kmeans::balanced_params{.n_iters = kmeans_n_iters}}
          : kmeans_params_variant{cuvs::cluster::kmeans::params{
              .n_clusters = 1 << pq_bits, .max_iter = static_cast<int>(kmeans_n_iters)}}),
      max_train_points_per_pq_code(max_train_points_per_pq_code),
      max_train_points_per_vq_cluster(max_train_points_per_vq_cluster)
  {
  }

  params(uint32_t pq_bits,
         uint32_t pq_dim,
         bool use_subspaces,
         bool use_vq,
         uint32_t vq_n_centers,
         kmeans_params_variant kmeans_params,
         uint32_t max_train_points_per_pq_code    = 256,
         uint32_t max_train_points_per_vq_cluster = 1024)
    : pq_bits(pq_bits),
      pq_dim(pq_dim),
      use_subspaces(use_subspaces),
      use_vq(use_vq),
      vq_n_centers(vq_n_centers),
      kmeans_params(kmeans_params),
      max_train_points_per_pq_code(max_train_points_per_pq_code),
      max_train_points_per_vq_cluster(max_train_points_per_vq_cluster)
  {
  }

  params() = default;

  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible value range: [4-16].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the faster the
   * fit/transform time, but the lower the recall.
   */
  uint32_t pq_bits = 8;
  /**
   * The dimensionality of the vector after compression by PQ.
   * When zero, dim / 4 is used as default.
   *
   * TODO: at the moment `dim` must be a multiple `pq_dim`.
   */
  uint32_t pq_dim = 0;
  /**
   * Whether to use subspaces for product quantization (PQ).
   * When true, one PQ codebook is used for each subspace. Otherwise, a single
   * PQ codebook is used.
   */
  bool use_subspaces = true;
  /**
   * Whether to use Vector Quantization (KMeans) before product quantization (PQ).
   * When true, VQ is used and PQ is trained on the residuals.
   */
  bool use_vq = false;
  /**
   * Vector Quantization (VQ) codebook size - number of "coarse cluster centers".
   * When zero, an optimal value is selected using a heuristic. (sqrt(n_rows))
   */
  uint32_t vq_n_centers = 0;
  /**
   * K-means parameters for PQ codebook training.
   *
   * Set to cuvs::cluster::kmeans::balanced_params for balanced k-means (default),
   * or cuvs::cluster::kmeans::params for regular k-means.
   * The active variant type selects the algorithm; balanced k-means tends to be faster
   * for PQ training where cluster sizes are approximately equal.
   * Only L2Expanded metric is supported. The number of clusters is always set to 1 << pq_bits.
   */
  kmeans_params_variant kmeans_params = cuvs::cluster::kmeans::balanced_params{};
  /**
   * The max number of data points to use per PQ code during PQ codebook training. Using more data
   * points per PQ code may increase the quality of PQ codebook but may also increase the build
   * time. We will use `pq_n_centers * max_train_points_per_pq_code` training
   * points to train each PQ codebook.
   */
  uint32_t max_train_points_per_pq_code = 256;
  /**
   * The max number of data points to use per VQ cluster during training.
   */
  uint32_t max_train_points_per_vq_cluster = 1024;
};

/** Parameters for VPQ compression. */
struct vpq_params {
  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible values: [4, 5, 6, 7, 8].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the better the search
   * performance, but the lower the recall.
   */
  uint32_t pq_bits = 8;
  /**
   * The dimensionality of the vector after compression by PQ.
   * When zero, an optimal value is selected using a heuristic.
   *
   * TODO: at the moment `dim` must be a multiple `pq_dim`.
   */
  uint32_t pq_dim = 0;
  /**
   * Vector Quantization (VQ) codebook size - number of "coarse cluster centers".
   * When zero, an optimal value is selected using a heuristic.
   */
  uint32_t vq_n_centers = 0;
  /** The number of iterations searching for kmeans centers (both VQ & PQ phases). */
  uint32_t kmeans_n_iters = 25;
  /**
   * The fraction of data to use during iterative kmeans building (VQ phase).
   * When zero, an optimal value is selected using a heuristic.
   * @deprecated Prefer using `max_train_points_per_vq_cluster` instead.
   */
  double vq_kmeans_trainset_fraction = 0;
  /**
   * The fraction of data to use during iterative kmeans building (PQ phase).
   * When zero, an optimal value is selected using a heuristic.
   * @deprecated Prefer using `max_train_points_per_pq_code` instead.
   */
  double pq_kmeans_trainset_fraction = 0;
  /**
   * Type of k-means algorithm for PQ training.
   * Balanced k-means tends to be faster than regular k-means for PQ training, for
   * problem sets where the number of points per cluster are approximately equal.
   * Regular k-means may be better for skewed cluster distributions.
   */
  cuvs::cluster::kmeans::kmeans_type pq_kmeans_type =
    cuvs::cluster::kmeans::kmeans_type::KMeansBalanced;
  /**
   * The max number of data points to use per PQ code during PQ codebook training. Using more data
   * points per PQ code may increase the quality of PQ codebook but may also increase the build
   * time. We will use `pq_n_centers * max_train_points_per_pq_code` training
   * points to train each PQ codebook.
   */
  uint32_t max_train_points_per_pq_code = 256;
  /**
   * The max number of data points to use per VQ cluster during training.
   */
  uint32_t max_train_points_per_vq_cluster = 1024;
};

// -----------------------------------------------------------------------------
// VPQ dataset: a child of `cuvs::core::dataset` / `dataset_view`. All VPQ-specific state
// (codebooks) and methods live in the payload types below; `dataset` itself knows nothing of them.
// -----------------------------------------------------------------------------

namespace detail {

// The accessor aliases are shared with every dataset kind and live next to `dataset`.
using cuvs::core::detail::device_owning_accessor;
using cuvs::core::detail::host_owning_accessor;

// VPQ codes are always uint8_t regardless of MathT, so retarget the owning accessor's element
// type instead of re-deriving a device/host matrix; residency is still driven by Accessor.
template <typename NewT, typename Accessor>
using owning_accessor_with_value_type = std::conditional_t<Accessor::is_device_accessible,
                                                           device_owning_accessor<NewT>,
                                                           host_owning_accessor<NewT>>;

// The two matrix types of a VPQ dataset. Both are owning (`mdarray`); the non-owning form of each
// is its `const_view_type`, which the view payload below uses directly.

/** Encoded rows (`uint8_t`): each row holds the VQ code followed by the PQ codes. Owning. */
template <typename IdxT, typename Accessor>
using vpq_data_matrix = raft::mdarray<uint8_t,
                                      raft::matrix_extent<IdxT>,
                                      raft::row_major,
                                      owning_accessor_with_value_type<uint8_t, Accessor>>;

/** A codebook (used for both the VQ and the PQ codebook). Owning. */
template <typename MathT, typename IdxT, typename Accessor>
using vpq_codebook_matrix =
  raft::mdarray<MathT, raft::matrix_extent<uint32_t>, raft::row_major, Accessor>;

/** Read-only helpers derived from the codebook shapes; shared by the owning and view payloads.
 * `Derived` provides `vq_code_book`, `pq_code_book` and the codes' `extent(r)`. */
template <typename Derived>
struct vpq_codebook_helpers {
  /** Logical dimension: it comes from the VQ codebook, not from the encoded rows (row padding
   * makes the encoded-row width ambiguous as a dimension). */
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(self().vq_code_book.extent(1));
  }
  [[nodiscard]] auto vq_n_centers() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(self().vq_code_book.extent(0));
  }
  [[nodiscard]] auto pq_n_centers() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(self().pq_code_book.extent(0));
  }
  [[nodiscard]] auto pq_len() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(self().pq_code_book.extent(1));
  }
  [[nodiscard]] auto pq_bits() const noexcept -> uint32_t
  {
    auto pq_width = pq_n_centers();
#ifdef __cpp_lib_bitops
    return std::countr_zero(pq_width);
#else
    uint32_t bits = 0;
    while (pq_width > 1) {
      bits++;
      pq_width >>= 1;
    }
    return bits;
#endif
  }
  [[nodiscard]] auto pq_dim() const noexcept -> uint32_t
  {
    return raft::div_rounding_up_unsafe(dim(), pq_len());
  }
  [[nodiscard]] auto encoded_row_length() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(self().extent(1));
  }

 private:
  [[nodiscard]] auto self() const noexcept -> Derived const&
  {
    return static_cast<Derived const&>(*this);
  }
};

/** The VPQ payload, defined once. `CodesT` is the encoded-rows matrix and `BookT` the type of each
 * codebook: raft `mdarray`s for an owning dataset, `mdspan`s for a non-owning view. The payload
 * *is* the `uint8_t` codes matrix (it derives from `CodesT`) and additionally holds the VQ and PQ
 * codebooks, so `vq_code_book`, `pq_code_book` and the helpers below mean the same thing in both
 * forms and only the ownership of the arrays differs. Use the aliases below rather than naming
 * this template directly. */
template <typename CodesT, typename BookT>
struct vpq_storage : public CodesT, public vpq_codebook_helpers<vpq_storage<CodesT, BookT>> {
  BookT vq_code_book;
  BookT pq_code_book;

  // Only usable when every member is default-constructible, i.e. for the view form.
  vpq_storage() noexcept = default;

  vpq_storage(CodesT&& codes, BookT&& vq_codes, BookT&& pq_codes) noexcept
    : CodesT(std::move(codes)), vq_code_book(std::move(vq_codes)), pq_code_book(std::move(pq_codes))
  {
  }
};

/** Owning VPQ payload: `mdarray` codes and codebooks. `Accessor` drives both codebook and code
 * residency. */
template <typename MathT, typename IdxT, typename Accessor>
using vpq_owning_storage =
  vpq_storage<vpq_data_matrix<IdxT, Accessor>, vpq_codebook_matrix<MathT, IdxT, Accessor>>;

/** Non-owning VPQ payload: `mdspan` views of the codes and of both codebooks. */
template <typename MathT, typename IdxT, typename Accessor>
using vpq_view_storage =
  vpq_storage<typename vpq_data_matrix<IdxT, Accessor>::const_view_type,
              typename vpq_codebook_matrix<MathT, IdxT, Accessor>::const_view_type>;

}  // namespace detail

/** `Accessor` drives both codebook and code residency, mirroring today's
 * single-`Accessor`-per-VPQ-dataset design. The payload (`detail::vpq_owning_storage` /
 * `detail::vpq_view_storage`) holds the encoded rows and the VQ/PQ codebooks. */
template <typename MathT, typename Accessor>
struct vpq_dataset_spec {
  using accessor_type = Accessor;
  template <typename NewAccessor>
  using rebind_accessor = vpq_dataset_spec<MathT, NewAccessor>;

  template <typename T, typename IdxT>
  struct apply {
    using value_type = std::remove_cv_t<T>;
    using index_type = std::remove_cv_t<IdxT>;
    using math_type  = MathT;

    using data_type = detail::vpq_owning_storage<MathT, IdxT, Accessor>;
    using view_type = detail::vpq_view_storage<MathT, IdxT, Accessor>;

    [[nodiscard]] static auto get_data_view(data_type const& data) noexcept -> view_type
    {
      return view_type(data.view(), data.vq_code_book.view(), data.pq_code_book.view());
    }
    template <typename AnyStorage>
    [[nodiscard]] static auto get_n_rows(AnyStorage const& data) noexcept -> index_type
    {
      return static_cast<index_type>(data.extent(0));
    }
    template <typename AnyStorage>
    [[nodiscard]] static auto get_dim(AnyStorage const& data) noexcept -> uint32_t
    {
      return data.dim();
    }
  };
};

template <typename DataT, typename IdxT>
using device_vpq_dataset =
  cuvs::core::dataset<DataT, IdxT, vpq_dataset_spec<DataT, detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using device_vpq_dataset_view = cuvs::core::
  dataset_view<DataT, IdxT, vpq_dataset_spec<DataT, detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_vpq_dataset =
  cuvs::core::dataset<DataT, IdxT, vpq_dataset_spec<DataT, detail::host_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_vpq_dataset_view = cuvs::core::
  dataset_view<DataT, IdxT, vpq_dataset_spec<DataT, detail::host_owning_accessor<DataT>>>;

/** Spec predicate for `cuvs::core::dataset_view_has_spec_v`. */
template <typename SpecT>
struct is_vpq_spec : std::false_type {};
template <typename MathT, typename Accessor>
struct is_vpq_spec<vpq_dataset_spec<MathT, Accessor>> : std::true_type {};
template <typename SpecT>
inline constexpr bool is_vpq_spec_v = is_vpq_spec<SpecT>::value;

template <typename SpecT>
struct vpq_spec_math_type {};
template <typename MathT, typename Accessor>
struct vpq_spec_math_type<vpq_dataset_spec<MathT, Accessor>> {
  using type = MathT;
};
template <typename SpecT>
using vpq_spec_math_type_t = typename vpq_spec_math_type<SpecT>::type;

/** True for an owning `dataset<...>` of the VPQ kind. */
template <typename DatasetT>
struct is_vpq_dataset : std::false_type {};
template <typename T, typename IdxT, typename SpecT>
struct is_vpq_dataset<cuvs::core::dataset<T, IdxT, SpecT>>
  : std::bool_constant<is_vpq_spec_v<SpecT>> {};
template <typename DatasetT>
inline constexpr bool is_vpq_dataset_v = is_vpq_dataset<DatasetT>::value;

/** True when `V` is a VPQ `dataset_view` whose codebooks have element type `MathT`. */
template <typename V, typename MathT>
struct is_vpq_dataset_view_with_math : std::false_type {};
template <typename T, typename IdxT, typename VpqMathT, typename Accessor, typename MathT>
struct is_vpq_dataset_view_with_math<
  cuvs::core::dataset_view<T, IdxT, vpq_dataset_spec<VpqMathT, Accessor>>,
  MathT> : std::is_same<VpqMathT, MathT> {};
template <typename V, typename MathT>
inline constexpr bool is_vpq_dataset_view_with_math_v =
  is_vpq_dataset_view_with_math<cuvs::core::dataset_view_type_t<V>, MathT>::value;

template <typename V>
inline constexpr bool is_device_vpq_f16_dataset_view_v =
  is_vpq_dataset_view_with_math_v<V, half> && cuvs::core::dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_vpq_f16_dataset_view_v =
  is_vpq_dataset_view_with_math_v<V, half> && !cuvs::core::dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_vpq_f16_dataset_view_v =
  is_device_vpq_f16_dataset_view_v<V> || is_host_vpq_f16_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_vpq_f32_dataset_view_v =
  is_vpq_dataset_view_with_math_v<V, float> && cuvs::core::dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_vpq_f32_dataset_view_v =
  is_vpq_dataset_view_with_math_v<V, float> && !cuvs::core::dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_vpq_f32_dataset_view_v =
  is_device_vpq_f32_dataset_view_v<V> || is_host_vpq_f32_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_vpq_dataset_view_v =
  is_device_vpq_f16_dataset_view_v<V> || is_device_vpq_f32_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_host_vpq_dataset_view_v =
  is_host_vpq_f16_dataset_view_v<V> || is_host_vpq_f32_dataset_view_v<V>;

/** True for any VPQ dataset view (host or device, f16 or f32 codebooks). */
template <typename V>
inline constexpr bool is_vpq_dataset_view_v =
  is_device_vpq_dataset_view_v<V> || is_host_vpq_dataset_view_v<V>;

/**
 * @brief Defines and stores VPQ codebooks upon training
 *
 * @tparam T data element type
 *
 */
template <typename T>
struct quantizer {
  /** Parameters used to build this quantizer. */
  params params_quantizer;
  /** VPQ codebooks produced during training. */
  device_vpq_dataset<T, int64_t> vpq_codebooks;
};

/**
 * @brief Initializes a product quantizer to be used later for quantizing the dataset.
 *
 * The use of a pool memory resource is recommended for more consistent training performance.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * // Set the workspace memory resource to a pool with 2 GiB upper limit.
 * raft::resource::set_workspace_to_pool_resource(handle, 2 * 1024 * 1024 * 1024ull);
 * cuvs::preprocessing::quantize::pq::params params;
 * auto quantizer = cuvs::preprocessing::quantize::pq::build(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure product quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on device or host
 *
 * @return quantizer
 */
quantizer<float> build(raft::resources const& res,
                       const params params,
                       raft::device_matrix_view<const float, int64_t> dataset);

/** @copydoc build */
quantizer<float> build(raft::resources const& res,
                       const params params,
                       raft::host_matrix_view<const float, int64_t> dataset);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::pq::params params;
 * auto quantizer = cuvs::preprocessing::quantize::pq::build(handle, params, dataset);
 * auto quantized_dim = get_quantized_dim(quantizer.params_quantizer);
 * auto quantized_dataset =
 *   raft::make_device_matrix<uint8_t, int64_t>(handle, samples, quantized_dim);
 * cuvs::preprocessing::quantize::pq::transform(handle, quantizer, dataset,
 *   quantized_dataset.view());
 *
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quant a product quantizer
 * @param[in] dataset a row-major matrix view on device or host
 * @param[out] codes_out a row-major matrix view on device containing the PQ codes
 * @param[out] vq_labels a vector view on device containing the VQ labels when VQ is
 * used, optional
 */
void transform(raft::resources const& res,
               const quantizer<float>& quant,
               raft::device_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> codes_out,
               std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels = std::nullopt);

/** @copydoc transform */
void transform(raft::resources const& res,
               const quantizer<float>& quant,
               raft::host_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> codes_out,
               std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels = std::nullopt);

/**
 * @brief Get the dimension of the quantized dataset (in bytes)
 *
 * @param[in] config product quantizer parameters
 * @return the dimension of the quantized dataset
 */
inline int64_t get_quantized_dim(const params& config)
{
  return raft::div_rounding_up_safe<int64_t>(config.pq_dim * config.pq_bits, 8);
}

/**
 * @brief Applies inverse quantization transform to given dataset
 *
 * @param[in] res raft resource
 * @param[in] quant a product quantizer
 * @param[in] pq_codes a row-major matrix view on device containing the PQ codes
 * @param[out] out a row-major matrix view on device
 * @param[in] vq_labels a vector view on device containing the VQ labels when VQ is used, optional
 *
 */
void inverse_transform(
  raft::resources const& res,
  const quantizer<float>& quant,
  raft::device_matrix_view<const uint8_t, int64_t> pq_codes,
  raft::device_matrix_view<float, int64_t> out,
  std::optional<raft::device_vector_view<const uint32_t, int64_t>> vq_labels = std::nullopt);

namespace detail {

// Trains from `n_rows` rows of `stride` elements each, whether they are device-accessible or
// host-resident; the residency is detected from the pointer.
//
// NB: the element type is erased into `dtype` so that this stays a plain function: under hidden
// default visibility, an instantiation cannot be exported from the shared library when one of its
// template arguments (`half`, or any mdspan type) is itself hidden, because the visibility of an
// instantiation is capped by that of its template arguments.
[[nodiscard]] CUVS_EXPORT device_vpq_dataset<half, int64_t> vpq_train_from_rows(
  raft::resources const& res,
  vpq_params const& params,
  void const* src_ptr,
  cudaDataType_t dtype,
  int64_t n_rows,
  int64_t dim,
  int64_t stride);

}  // namespace detail

/**
 * @brief Train VPQ storage (codebooks + encoded rows) from a row-major mdspan/mdarray/dataset.
 *
 * Accepts either a row-major mdspan with `value_type`, `extent`, `stride`, and `data_handle` (same
 * pattern as `cuvs::core::make_device_padded_dataset`), or any cuVS dense dataset / dataset
 * view exposing `view`, `dim` and `stride`, in which case the logical `dim()` is quantized and the
 * row padding is skipped. The rows may be device-accessible or host-resident. Device-accessible
 * rows (device, managed or pinned) with tight row-major storage (logical stride equals dimension)
 * are passed through to training as they are; a wider row pitch triggers a contiguous dense copy
 * first. Host-resident rows are subsampled for training and encoded in bounded batches, so the
 * dense dataset is never staged on the device in full; they must be tightly packed. Empty sources
 * are rejected. The element type must be `float`, `half`, `int8_t` or `uint8_t`.
 *
 * Typical **CAGRA-Q** usage: compress the source rows, then build the graph directly from the VPQ
 * dataset (the metric must be `L2Expanded`). Keep the `device_vpq_dataset` alive because the index
 * holds a non-owning view of it.
 *
 * @code{.cpp}
 * #include <cuvs/neighbors/cagra.hpp>
 * #include <cuvs/preprocessing/quantize/pq.hpp>
 *
 * // `padded` is a `device_padded_dataset_view<float, int64_t>` over the source rows.
 * cuvs::preprocessing::quantize::pq::vpq_params vpq_params{};
 * auto vpq = cuvs::preprocessing::quantize::pq::make_vpq_dataset(res, vpq_params, padded);
 * auto idx = cuvs::neighbors::cagra::build(res, cagra_params, vpq.as_dataset_view());
 * @endcode
 */
template <typename SrcT>
[[nodiscard]] auto make_vpq_dataset(raft::resources const& res,
                                    vpq_params const& params,
                                    SrcT const& src) -> device_vpq_dataset<half, int64_t>
{
  // A cuVS dataset keeps its logical width in `dim()` while `as_matrix_view()` spans the full row
  // pitch.
  if constexpr (requires {
                  src.as_matrix_view();
                  src.dim();
                }) {
    auto const rows    = src.as_matrix_view();
    using value_type   = typename decltype(rows)::value_type;
    using extents_type = raft::matrix_extent<int64_t>;
    return make_vpq_dataset(
      res,
      params,
      raft::mdspan<const value_type, extents_type, raft::layout_stride>{
        rows.data_handle(),
        raft::make_strided_layout(extents_type{rows.extent(0), int64_t{src.dim()}},
                                  cuda::std::array<int64_t, 2>{int64_t{rows.stride()}, 1})});
  } else {
    using value_type = typename SrcT::value_type;
    static_assert(std::is_same_v<value_type, float> || std::is_same_v<value_type, half> ||
                    std::is_same_v<value_type, int8_t> || std::is_same_v<value_type, uint8_t>,
                  "make_vpq_dataset: element type must be float, half, int8_t or uint8_t");
    const int64_t n_rows = src.extent(0);
    const int64_t dim    = src.extent(1);
    const int64_t stride = src.stride(0) > 0 ? src.stride(0) : dim;
    RAFT_EXPECTS(n_rows > 0, "make_vpq_dataset: dataset is empty");
    return detail::vpq_train_from_rows(
      res, params, src.data_handle(), raft::get_cuda_data_type<value_type>(), n_rows, dim, stride);
  }
}

/** @} */  // end of group product

}  // namespace pq
}  // namespace quantize
}  // namespace preprocessing
}  // namespace CUVS_EXPORT cuvs
