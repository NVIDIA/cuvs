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

#include <cmath>
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

/** Select global or per-dimension shifts and scales. */
enum class range_mode { global, per_dimension };

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
   * Each dimension's shift is the midpoint of its bounds, rounded to the input type.
   * Each dimension's scale fits its bounds around its stored shift into the symmetric range
   * of quantized values.
   *
   * For a zero-width interval, its scale is one. Otherwise, each scale is at least
   * the smallest positive normal value of the scale type.
   */
  range_mode range = range_mode::global;
  /**
   * Maximum rows staged on device for host training and host dataset build.
   * Default: 65536. Zero processes the entire host input in one batch. Ignored for device input.
   * Multiple batches use two input buffers when res has an auxiliary stream pool, and one
   * otherwise. Pageable input uses batch-sized pinned staging buffers, without registering
   * the full input. Each input buffer uses at most device_buffer_samples * dim * sizeof(T)
   * bytes, in addition to reduction and encoding workspace. Substitute n_rows when it is zero.
   *
   * Per-dimension host training with quantile < 1 selects exact quantiles using one column of
   * host scratch (n_rows * sizeof(T) bytes). Global training retains its sampled quantiles.
   */
  int64_t device_buffer_samples = 1 << 16;
};

/** Read-only scalar quantizer parameters. The owner must outlive stream work using shifts and
 * scales. */
template <typename T>
struct quantizer_const_view {
  using scale_type = std::conditional_t<std::is_same_v<T, half>, float, T>;
  /** Global affine parameters using double precision for encoding and decoding.
   * Encode (x - shift) / scale, rounding to nearest with ties away from zero and clamping
   * to the full quantized range. Decode shift + scale * code, saturating to finite input limits.
   * A zero scale represents constant training bounds; every code decodes to shift. */
  struct global_type {
    /** Shift shared by all dimensions. */
    double shift = 0;
    /** Step size shared by all dimensions. */
    double scale = 0;
  };
  /** Encode (x - shifts[j]) / scales[j], rounding to nearest with ties to even and
   * clamping to the symmetric range of quantized values.
   * Decode shifts[j] + scales[j] * code, saturating to finite limits of the input type. */
  struct per_dim_type {
    /** One midpoint shift per dimension. */
    raft::device_vector_view<const T, int64_t> shifts;
    /** One finite, positive scale per dimension, with a finite reciprocal. */
    raft::device_vector_view<const scale_type, int64_t> scales;
  };
  /** Global parameters or borrowed per-dimension shifts and scales. */
  std::variant<global_type, per_dim_type> value;
  /** Maximum training-row L2 reconstruction error; see quantizer::max_error. */
  scale_type max_error = INFINITY;
};

/**
 * Owns a global shift and scale or device shifts and scales for each dimension.
 * Move-only; copy view() to borrow the parameters.
 */
template <typename T>
struct quantizer {
  using view_type   = quantizer_const_view<T>;
  using scale_type  = typename view_type::scale_type;
  using global_type = typename view_type::global_type;
  /** Owning per-dimension parameters. */
  struct per_dim_type {
    raft::device_vector<T, int64_t> shifts;
    raft::device_vector<scale_type, int64_t> scales;
  };
  /** Global parameters or owning per-dimension shifts and scales. */
  std::variant<global_type, per_dim_type> value;
  /** Original lower global bound; unused in per-dimension mode. Does not track changes to value. */
  T min_ = T{0};
  /** Original upper global bound; unused in per-dimension mode. Does not track changes to value. */
  T max_ = T{0};
  /**
   * Upper bound on the L2 error between a training row and its int8 encode/decode result.
   * train measures every input row, including clipped outliers, and rounds the maximum up
   * with a relative 1e-6 margin. This bound applies only to the training data, not later queries.
   * It is infinity when constructed from parameters alone or when the error is not finite.
   */
  scale_type max_error = INFINITY;

  /** Construct default global parameters. */
  quantizer() : value{global_type{}} {}
  /** Derive a global shift and scale from finite bounds with min <= max. */
  quantizer(T min, T max);
  /** Construct from nonempty device bounds of equal length, with finite min <= max.
   * Derives midpoint shifts and a scale for each dimension. */
  quantizer(raft::resources const& res,
            raft::device_vector_view<const T, int64_t> mins,
            raft::device_vector_view<const T, int64_t> maxs);

  /** Borrow read-only parameters; the owner must outlive stream work using shifts and scales. */
  [[nodiscard]] auto view() const noexcept -> view_type
  {
    if (auto const* range = std::get_if<global_type>(&value)) { return {*range, max_error}; }
    auto const& shifts = std::get<per_dim_type>(value);
    return {typename view_type::per_dim_type{shifts.shifts.view(), shifts.scales.view()},
            max_error};
  }
  /** An owning quantizer can be passed wherever a read-only view is accepted. */
  operator view_type() const noexcept { return view(); }  // NOLINT(google-explicit-constructor)
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
               quantizer_const_view<double> quantizer,
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
               quantizer_const_view<double> quantizer,
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
                       quantizer_const_view<double> quantizer,
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
                       quantizer_const_view<double> quantizer,
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
               quantizer_const_view<float> quantizer,
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
               quantizer_const_view<float> quantizer,
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
                       quantizer_const_view<float> quantizer,
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
                       quantizer_const_view<float> quantizer,
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
               quantizer_const_view<half> quantizer,
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
               quantizer_const_view<half> quantizer,
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
                       quantizer_const_view<half> quantizer,
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
                       quantizer_const_view<half> quantizer,
                       raft::host_matrix_view<const int8_t, int64_t> dataset,
                       raft::host_matrix_view<half, int64_t> out);

/** @} */  // end of group scalar

}  // namespace scalar
}  // namespace quantize
}  // namespace preprocessing
}  // namespace CUVS_EXPORT cuvs
