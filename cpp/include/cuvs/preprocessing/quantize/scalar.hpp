/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>

#include <cuvs/core/cuda_fp16.hpp>
#include <cuvs/core/export.hpp>

#include <type_traits>
#include <utility>
#include <variant>

namespace CUVS_EXPORT cuvs {
namespace preprocessing {
namespace quantize {
namespace scalar {

/**
 * @defgroup scalar Scalar quantizer utilities
 * @{
 */

/** Select global or per-dimension quantization bounds. */
enum class quantization_mode { GLOBAL, PER_DIMENSION };

/**
 * @brief quantizer parameters.
 */
struct params {
  /**
   * Specifies how many outliers at top & bottom will be ignored.
   * Needs to be within range of (0, 1].
   */
  float quantile = 0.99;
  /**
   * Per-dimension training uses every row of a nonempty, finite dataset.
   * It applies the quantile independently to each dimension to select lower and upper bounds.
   * Transforms map each dimension's bounds to the quantized range exactly as global bounds are
   * mapped.
   */
  quantization_mode range = quantization_mode::GLOBAL;
  /**
   * Maximum rows staged on device for host training and host dataset build.
   * Default: 65536. Zero processes the entire host input in one batch. Ignored for device input.
   * Multiple batches use two input buffers when res has an auxiliary stream pool, and one
   * otherwise. Each input buffer uses at most device_buffer_samples * dim * sizeof(T) bytes, in
   * addition to reduction and encoding workspace. Substitute n_rows when it is zero.
   *
   * Per-dimension host training with quantile < 1 selects exact quantiles using one column of
   * host scratch (n_rows * sizeof(T) bytes). Global training retains its sampled quantiles.
   */
  int64_t device_buffer_samples = 1 << 16;
};

/**
 * Non-owning scalar quantizer bounds, with the constness semantics of an mdspan:
 * quantizer_view<T> can modify per-dimension bounds; quantizer_view<const T> cannot.
 * Global bounds are copied. The owner must outlive stream work using the vectors.
 */
template <typename T>
struct quantizer_view {
  using value_type       = std::remove_const_t<T>;
  using scale_type       = std::conditional_t<std::is_same_v<value_type, half>, float, value_type>;
  using vector_view_type = raft::device_vector_view<T, int64_t>;
  /** Global bounds. Transforms derive an affine mapping in double precision.
   * Encode (x - shift) / scale, rounding to nearest with ties away from zero and clamping
   * to the full quantized range. Decode shift + scale * code, saturating to finite input limits.
   * Equal bounds represent a constant range; every code decodes to that value. */
  struct global_type {
    /** Lower bound shared by all dimensions. */
    value_type min = value_type{0};
    /** Upper bound shared by all dimensions. */
    value_type max = value_type{0};
  };
  /** Per-dimension bounds. Transforms map each dimension as global_type does, with that
   * dimension's bounds. */
  struct per_dim_type {
    /** One finite lower bound per dimension, with mins[j] <= maxs[j]. */
    vector_view_type mins;
    /** One finite upper bound per dimension; must have the same extent as mins. */
    vector_view_type maxs;
  };
  /** Global bounds or borrowed per-dimension bounds. */
  std::variant<global_type, per_dim_type> value;

  // A mutable view decays to the read-only one, the way an mdspan does.
  template <typename U>
    requires(!std::is_const_v<T> && std::is_same_v<U, const T>)
  operator quantizer_view<U>() const noexcept  // NOLINT(google-explicit-constructor)
  {
    using const_view_type = quantizer_view<U>;
    if (auto const* range = std::get_if<global_type>(&value)) {
      return {typename const_view_type::global_type{range->min, range->max}};
    }
    auto const& bounds = std::get<per_dim_type>(value);
    return {typename const_view_type::per_dim_type{raft::make_const_mdspan(bounds.mins),
                                                   raft::make_const_mdspan(bounds.maxs)}};
  }
};

/**
 * Owns global bounds or device bounds for each dimension.
 * Move-only; copy view() to borrow the parameters.
 */
template <typename T>
struct quantizer {
  static_assert(!std::is_const_v<T>, "An owning quantizer cannot hold const elements.");

  using view_type       = quantizer_view<T>;
  using const_view_type = quantizer_view<const T>;
  using scale_type      = typename view_type::scale_type;
  using global_type     = typename view_type::global_type;
  /** Owning per-dimension bounds. */
  struct per_dim_type {
    raft::device_vector<T, int64_t> mins;
    raft::device_vector<T, int64_t> maxs;
  };
  /** Global bounds or owning per-dimension bounds. */
  std::variant<global_type, per_dim_type> value;
  /** Construct default global bounds. */
  quantizer() : value{global_type{}} {}
  /** Construct from finite global bounds with min <= max. */
  quantizer(T min, T max);
  /** Copy nonempty device bounds of equal length, with finite min <= max. */
  quantizer(raft::resources const& res,
            raft::device_vector_view<const T, int64_t> mins,
            raft::device_vector_view<const T, int64_t> maxs);

  /** Borrow mutable per-dimension bounds; the owner must outlive stream work using them. */
  [[nodiscard]] auto view() noexcept -> view_type
  {
    if (auto const* range = std::get_if<global_type>(&value)) { return {*range}; }
    auto& bounds = std::get<per_dim_type>(value);
    return {typename view_type::per_dim_type{bounds.mins.view(), bounds.maxs.view()}};
  }
  /** Borrow read-only bounds; the owner must outlive stream work using per-dimension bounds. */
  [[nodiscard]] auto view() const noexcept -> const_view_type
  {
    if (auto const* range = std::get_if<global_type>(&value)) {
      return {typename const_view_type::global_type{range->min, range->max}};
    }
    auto const& bounds = std::get<per_dim_type>(value);
    return {typename const_view_type::per_dim_type{bounds.mins.view(), bounds.maxs.view()}};
  }
};

/**
 * @brief Initializes a scalar quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on device
 *
 * @return quantizer
 */
quantizer<double> train(raft::resources const& res,
                        const params params,
                        raft::device_matrix_view<const double, int64_t> dataset);

/**
 * @brief Initializes a scalar quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on host
 *
 * @return quantizer
 */
quantizer<double> train(raft::resources const& res,
                        const params params,
                        raft::host_matrix_view<const double, int64_t> dataset);

/**
 * @brief Measures the maximum L2 reconstruction error of a trained quantizer over a dataset.
 *
 * Measures every row, including clipped outliers, and rounds the maximum up with a relative
 * 1e-6 margin. The result bounds only the measured rows, and is infinity when the error is not
 * finite. make_sq_dataset can measure it while encoding the same dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params, dataset);
 * auto max_error =
 *   cuvs::preprocessing::quantize::scalar::compute_max_error(handle, quantizer, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 *
 * @return upper bound on the L2 error between a row and its int8 encode/decode result
 */
double compute_max_error(raft::resources const& res,
                         const quantizer<double>& quantizer,
                         raft::device_matrix_view<const double, int64_t> dataset);

/**
 * @brief Measures the maximum L2 reconstruction error of a trained quantizer over a dataset.
 *
 * Measures every row in GPU batches, including clipped outliers, and rounds the maximum up
 * with a relative 1e-6 margin. The result bounds only the measured rows, and is infinity when
 * the error is not finite. make_sq_dataset can measure it while encoding the same dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params, dataset);
 * auto max_error =
 *   cuvs::preprocessing::quantize::scalar::compute_max_error(handle, quantizer, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[in] device_buffer_samples maximum rows staged on device per batch; zero stages the
 * entire input in one batch. See params::device_buffer_samples.
 *
 * @return upper bound on the L2 error between a row and its int8 encode/decode result
 */
double compute_max_error(raft::resources const& res,
                         const quantizer<double>& quantizer,
                         raft::host_matrix_view<const double, int64_t> dataset,
                         int64_t device_buffer_samples = 1 << 16);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<double, int8_t>(handle, params,
 * dataset); auto quantized_dataset = raft::make_device_matrix<int8_t, int64_t>(handle, samples,
 * features); cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer<double>& quantizer,
               raft::device_matrix_view<const double, int64_t> dataset,
               raft::device_matrix_view<int8_t, int64_t> out);

/**
 * @brief Applies global quantization to a host dataset
 *
 * Throws raft::logic_error for per-dimension parameters. Use device transforms or host build
 * for per-dimension quantization.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params,
 * dataset); auto quantized_dataset = raft::make_host_matrix<int8_t, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer with global parameters
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const quantizer<double>& quantizer,
               raft::host_matrix_view<const double, int64_t> dataset,
               raft::host_matrix_view<int8_t, int64_t> out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_device_matrix<int8_t, int64_t>(handle, samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_device_matrix<double,
 * int64_t>(handle, samples, features);
 * cuvs::preprocessing::quantize::scalar::inverse_transform(handle, quantizer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void inverse_transform(raft::resources const& res,
                       const quantizer<double>& quantizer,
                       raft::device_matrix_view<const int8_t, int64_t> dataset,
                       raft::device_matrix_view<double, int64_t> out);

/**
 * @brief Applies inverse global quantization to a host dataset
 *
 * Throws raft::logic_error for per-dimension parameters. Use device inverse_transform
 * for per-dimension quantization.
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_host_matrix<int8_t, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_host_matrix<double, int64_t>(samples,
 * features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, quantizer,
 * raft::make_const_mdspan(quantized_dataset.view()), dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer with global parameters
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void inverse_transform(raft::resources const& res,
                       const quantizer<double>& quantizer,
                       raft::host_matrix_view<const int8_t, int64_t> dataset,
                       raft::host_matrix_view<double, int64_t> out);

/**
 * @brief Initializes a scalar quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on device
 *
 * @return quantizer
 */
quantizer<float> train(raft::resources const& res,
                       const params params,
                       raft::device_matrix_view<const float, int64_t> dataset);

/**
 * @brief Initializes a scalar quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on host
 *
 * @return quantizer
 */
quantizer<float> train(raft::resources const& res,
                       const params params,
                       raft::host_matrix_view<const float, int64_t> dataset);

/**
 * @brief Measures the maximum L2 reconstruction error of a trained quantizer over a dataset.
 *
 * Measures every row, including clipped outliers, and rounds the maximum up with a relative
 * 1e-6 margin. The result bounds only the measured rows, and is infinity when the error is not
 * finite. make_sq_dataset can measure it while encoding the same dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params, dataset);
 * auto max_error =
 *   cuvs::preprocessing::quantize::scalar::compute_max_error(handle, quantizer, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 *
 * @return upper bound on the L2 error between a row and its int8 encode/decode result
 */
float compute_max_error(raft::resources const& res,
                        const quantizer<float>& quantizer,
                        raft::device_matrix_view<const float, int64_t> dataset);

/**
 * @brief Measures the maximum L2 reconstruction error of a trained quantizer over a dataset.
 *
 * Measures every row in GPU batches, including clipped outliers, and rounds the maximum up
 * with a relative 1e-6 margin. The result bounds only the measured rows, and is infinity when
 * the error is not finite. make_sq_dataset can measure it while encoding the same dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params, dataset);
 * auto max_error =
 *   cuvs::preprocessing::quantize::scalar::compute_max_error(handle, quantizer, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[in] device_buffer_samples maximum rows staged on device per batch; zero stages the
 * entire input in one batch. See params::device_buffer_samples.
 *
 * @return upper bound on the L2 error between a row and its int8 encode/decode result
 */
float compute_max_error(raft::resources const& res,
                        const quantizer<float>& quantizer,
                        raft::host_matrix_view<const float, int64_t> dataset,
                        int64_t device_buffer_samples = 1 << 16);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<float, int8_t>(handle, params,
 * dataset); auto quantized_dataset = raft::make_device_matrix<int8_t, int64_t>(handle, samples,
 * features); cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer<float>& quantizer,
               raft::device_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<int8_t, int64_t> out);

/**
 * @brief Applies global quantization to a host dataset
 *
 * Throws raft::logic_error for per-dimension parameters. Use device transforms or host build
 * for per-dimension quantization.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params,
 * dataset); auto quantized_dataset = raft::make_host_matrix<int8_t, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer with global parameters
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const quantizer<float>& quantizer,
               raft::host_matrix_view<const float, int64_t> dataset,
               raft::host_matrix_view<int8_t, int64_t> out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_device_matrix<int8_t, int64_t>(handle, samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_device_matrix<float, int64_t>(handle,
 * samples, features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, quantizer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void inverse_transform(raft::resources const& res,
                       const quantizer<float>& quantizer,
                       raft::device_matrix_view<const int8_t, int64_t> dataset,
                       raft::device_matrix_view<float, int64_t> out);

/**
 * @brief Applies inverse global quantization to a host dataset
 *
 * Throws raft::logic_error for per-dimension parameters. Use device inverse_transform
 * for per-dimension quantization.
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_host_matrix<int8_t, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_host_matrix<float, int64_t>(samples,
 * features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, quantizer,
 * raft::make_const_mdspan(quantized_dataset.view()), dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer with global parameters
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void inverse_transform(raft::resources const& res,
                       const quantizer<float>& quantizer,
                       raft::host_matrix_view<const int8_t, int64_t> dataset,
                       raft::host_matrix_view<float, int64_t> out);

/**
 * @brief Initializes a scalar quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on device
 *
 * @return quantizer
 */
quantizer<half> train(raft::resources const& res,
                      const params params,
                      raft::device_matrix_view<const half, int64_t> dataset);

/**
 * @brief Initializes a scalar quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure scalar quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on host
 *
 * @return quantizer
 */
quantizer<half> train(raft::resources const& res,
                      const params params,
                      raft::host_matrix_view<const half, int64_t> dataset);

/**
 * @brief Measures the maximum L2 reconstruction error of a trained quantizer over a dataset.
 *
 * Measures every row, including clipped outliers, and rounds the maximum up with a relative
 * 1e-6 margin. The result bounds only the measured rows, and is infinity when the error is not
 * finite. make_sq_dataset can measure it while encoding the same dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params, dataset);
 * auto max_error =
 *   cuvs::preprocessing::quantize::scalar::compute_max_error(handle, quantizer, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 *
 * @return upper bound on the L2 error between a row and its int8 encode/decode result
 */
float compute_max_error(raft::resources const& res,
                        const quantizer<half>& quantizer,
                        raft::device_matrix_view<const half, int64_t> dataset);

/**
 * @brief Measures the maximum L2 reconstruction error of a trained quantizer over a dataset.
 *
 * Measures every row in GPU batches, including clipped outliers, and rounds the maximum up
 * with a relative 1e-6 margin. The result bounds only the measured rows, and is infinity when
 * the error is not finite. make_sq_dataset can measure it while encoding the same dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params, dataset);
 * auto max_error =
 *   cuvs::preprocessing::quantize::scalar::compute_max_error(handle, quantizer, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[in] device_buffer_samples maximum rows staged on device per batch; zero stages the
 * entire input in one batch. See params::device_buffer_samples.
 *
 * @return upper bound on the L2 error between a row and its int8 encode/decode result
 */
float compute_max_error(raft::resources const& res,
                        const quantizer<half>& quantizer,
                        raft::host_matrix_view<const half, int64_t> dataset,
                        int64_t device_buffer_samples = 1 << 16);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train<half, int8_t>(handle, params,
 * dataset); auto quantized_dataset = raft::make_device_matrix<int8_t, int64_t>(handle, samples,
 * features); cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer<half>& quantizer,
               raft::device_matrix_view<const half, int64_t> dataset,
               raft::device_matrix_view<int8_t, int64_t> out);

/**
 * @brief Applies global quantization to a host dataset
 *
 * Throws raft::logic_error for per-dimension parameters. Use device transforms or host build
 * for per-dimension quantization.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::scalar::params params;
 * auto quantizer = cuvs::preprocessing::quantize::scalar::train(handle, params,
 * dataset); auto quantized_dataset = raft::make_host_matrix<int8_t, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer with global parameters
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const quantizer<half>& quantizer,
               raft::host_matrix_view<const half, int64_t> dataset,
               raft::host_matrix_view<int8_t, int64_t> out);

/**
 * @brief Perform inverse quantization step on previously quantized dataset
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_device_matrix<int8_t, int64_t>(handle, samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_device_matrix<half, int64_t>(handle,
 * samples, features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, quantizer,
 * dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void inverse_transform(raft::resources const& res,
                       const quantizer<half>& quantizer,
                       raft::device_matrix_view<const int8_t, int64_t> dataset,
                       raft::device_matrix_view<half, int64_t> out);

/**
 * @brief Applies inverse global quantization to a host dataset
 *
 * Throws raft::logic_error for per-dimension parameters. Use device inverse_transform
 * for per-dimension quantization.
 *
 * Note that depending on the chosen data types train dataset the conversion is
 * not lossless.
 *
 * Usage example:
 * @code{.cpp}
 * auto quantized_dataset = raft::make_host_matrix<int8_t, int64_t>(samples, features);
 * cuvs::preprocessing::quantize::scalar::transform(handle, quantizer, dataset,
 * quantized_dataset.view()); auto dataset_revert = raft::make_host_matrix<half, int64_t>(samples,
 * features); cuvs::preprocessing::quantize::scalar::inverse_transform(handle, quantizer,
 * raft::make_const_mdspan(quantized_dataset.view()), dataset_revert.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a scalar quantizer with global parameters
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void inverse_transform(raft::resources const& res,
                       const quantizer<half>& quantizer,
                       raft::host_matrix_view<const int8_t, int64_t> dataset,
                       raft::host_matrix_view<half, int64_t> out);

/** @} */  // end of group scalar

}  // namespace scalar
}  // namespace quantize
}  // namespace preprocessing
}  // namespace CUVS_EXPORT cuvs
