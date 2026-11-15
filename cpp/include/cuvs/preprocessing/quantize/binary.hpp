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

#include <cstdint>
#include <type_traits>
#include <utility>
#include <variant>

namespace CUVS_EXPORT cuvs {
namespace preprocessing {
namespace quantize {
namespace binary {

/**
 * @defgroup binary Binary quantizer utilities
 * @{
 */

/**
 * @brief quantizer algorithms. The mean and sampling_median thresholds are calculated separately
 * for each dimension.
 */
enum class bit_threshold { zero, mean, sampling_median };

/**
 * @brief quantizer parameters.
 */
struct params {
  /** Threshold method for binarization. */
  bit_threshold threshold = bit_threshold::mean;

  /** Specifies the sampling ratio. */
  float sampling_ratio = 0.1;
};

/**
 * @brief Non-owning view of a binary quantizer, with the constness semantics of an mdspan:
 * `quantizer_view<T>` can modify the thresholds, `quantizer_view<const T>` cannot.
 *
 * The threshold is either the vector of per-dimension thresholds (whose extent is the dimension),
 * or just the dimension of the vectors, which stands for the zero quantizer: no thresholds are
 * stored and zero is the threshold in every dimension. These are different quantizers even when
 * the vector is empty.
 *
 * @tparam T data element type, possibly const
 * @tparam ContainerPolicy the container policy of the owning quantizer (bound to the element type
 *   without const); the device memory by default
 */
template <typename T,
          typename ContainerPolicy =
            raft::device_accessor<raft::device_container_policy<std::remove_const_t<T>>>>
struct quantizer_view {
  /** The owning vector of the thresholds, which defines the memory space of the view. */
  using vector_type      = raft::mdarray<std::remove_const_t<T>,
                                         raft::vector_extent<int64_t>,
                                         raft::layout_c_contiguous,
                                         ContainerPolicy>;
  using vector_view_type = std::conditional_t<std::is_const_v<T>,
                                              typename vector_type::const_view_type,
                                              typename vector_type::view_type>;

  /** The thresholds [dim], or the dimension for the zero quantizer. */
  std::variant<uint32_t, vector_view_type> value;

  /** The zero quantizer of dimension zero. */
  quantizer_view() = default;

  /** The zero quantizer of the given dimension (number of bits). */
  explicit quantizer_view(uint32_t dim) noexcept : value{dim} {}

  /** The quantizer with the given thresholds; its dimension is the extent of the vector. */
  explicit quantizer_view(vector_view_type thresholds) noexcept : value{thresholds} {}

  /** The number of dimensions (bits) of the quantized vectors. */
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return std::visit(
      [](const auto& v) -> uint32_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(v)>, uint32_t>) {
          return v;
        } else {
          return static_cast<uint32_t>(v.extent(0));
        }
      },
      value);
  }

  /** A mutable view decays to the read-only one, the way an mdspan does. */
  template <typename U>
    requires(!std::is_const_v<T> && std::is_same_v<U, const T>)
  operator quantizer_view<U, ContainerPolicy>()
    const noexcept  // NOLINT(google-explicit-constructor)
  {
    return std::visit(
      [](const auto& v) -> quantizer_view<U, ContainerPolicy> {
        if constexpr (std::is_same_v<std::decay_t<decltype(v)>, uint32_t>) {
          return quantizer_view<U, ContainerPolicy>{v};
        } else {
          return quantizer_view<U, ContainerPolicy>{raft::make_const_mdspan(v)};
        }
      },
      value);
  }
};

/**
 * @brief Store the threshold vector for quantization. In the binary::transform function, a bit is
 * set if the corresponding element in the dataset vector is greater than the corresponding element
 * in the threshold vector.
 *
 * The threshold is either the vector of per-dimension thresholds (whose extent is the dimension),
 * or just the dimension of the vectors, which stands for the zero quantizer: no thresholds are
 * stored and zero is the threshold in every dimension. These are different quantizers even when
 * the vector is empty. A default-constructed quantizer is the zero quantizer of dimension zero.
 *
 * @tparam T data element type
 * @tparam ContainerPolicy the container policy of the thresholds (bound to `T`); the device memory
 *   by default
 *
 */
template <typename T,
          typename ContainerPolicy = raft::device_accessor<raft::device_container_policy<T>>>
struct quantizer {
  static_assert(!std::is_const_v<T>, "An owning quantizer cannot hold const elements.");

  using threshold_type =
    raft::mdarray<T, raft::vector_extent<int64_t>, raft::layout_c_contiguous, ContainerPolicy>;
  using view_type       = quantizer_view<T, ContainerPolicy>;
  using const_view_type = quantizer_view<const T, ContainerPolicy>;

  /** The thresholds [dim], or the dimension for the zero quantizer. */
  std::variant<uint32_t, threshold_type> value;

  /** The zero quantizer of dimension zero. */
  quantizer() = default;

  /** The zero quantizer of the given dimension (number of bits). */
  explicit quantizer(uint32_t dim) noexcept : value{dim} {}

  /** The quantizer owning the given thresholds; its dimension is the extent of the vector. */
  explicit quantizer(threshold_type&& thresholds) : value{std::move(thresholds)} {}

  /** The number of dimensions (bits) of the quantized vectors. */
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return std::visit(
      [](const auto& v) -> uint32_t {
        if constexpr (std::is_same_v<std::decay_t<decltype(v)>, uint32_t>) {
          return v;
        } else {
          return static_cast<uint32_t>(v.extent(0));
        }
      },
      value);
  }

  /** A non-owning view of the quantizer. */
  [[nodiscard]] auto view() noexcept -> view_type
  {
    return std::visit(
      [](auto& v) -> view_type {
        if constexpr (std::is_same_v<std::decay_t<decltype(v)>, uint32_t>) {
          return view_type{v};
        } else {
          return view_type{v.view()};
        }
      },
      value);
  }

  /** A read-only non-owning view of the quantizer. */
  [[nodiscard]] auto view() const noexcept -> const_view_type
  {
    return std::visit(
      [](const auto& v) -> const_view_type {
        if constexpr (std::is_same_v<std::decay_t<decltype(v)>, uint32_t>) {
          return const_view_type{v};
        } else {
          return const_view_type{v.view()};
        }
      },
      value);
  }
};

/**
 * @brief Initializes a binary quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure binary quantizer, e.g. threshold
 * @param[in] dataset a row-major matrix view on device
 *
 * @return quantizer
 */
quantizer<double> train(raft::resources const& res,
                        const params params,
                        raft::device_matrix_view<const double, int64_t> dataset);

/**
 * @brief Initializes a binary quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure binary quantizer, e.g. threshold
 * @param[in] dataset a row-major matrix view on host
 *
 * @return quantizer
 */
quantizer<double> train(raft::resources const& res,
                        const params params,
                        raft::host_matrix_view<const double, int64_t> dataset);

/**
 * @brief Applies binary quantization transform to given dataset. If a dataset element is positive,
 * set the corresponding bit to 1.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantized_dataset = raft::make_device_matrix<uint8_t, int64_t>(handle, samples,
 * features);
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params,
 * raft::make_const_mdspan(dataset.view()));
 * cuvs::preprocessing::quantize::binary::transform(handle, quantizer,
 * raft::make_const_mdspan(dataset.view()), quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a binary quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer<double>& quantizer,
               raft::device_matrix_view<const double, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> out);

/**
 * @brief Applies binary quantization transform to given dataset. If a dataset element is positive,
 * set the corresponding bit to 1.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantized_dataset = raft::make_host_matrix<uint8_t, int64_t>(handle, samples,
 * features);
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params,
 * raft::make_const_mdspan(dataset.view()));
 * cuvs::preprocessing::quantize::binary::transform(handle, quantizer,
 * raft::make_const_mdspan(dataset.view()), quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a binary quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const quantizer<double>& quantizer,
               raft::host_matrix_view<const double, int64_t> dataset,
               raft::host_matrix_view<uint8_t, int64_t> out);

/**
 * @brief Initializes a binary quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure binary quantizer, e.g. threshold
 * @param[in] dataset a row-major matrix view on device
 *
 * @return quantizer
 */
quantizer<float> train(raft::resources const& res,
                       const params params,
                       raft::device_matrix_view<const float, int64_t> dataset);

/**
 * @brief Initializes a binary quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure binary quantizer, e.g. threshold
 * @param[in] dataset a row-major matrix view on host
 *
 * @return quantizer
 */
quantizer<float> train(raft::resources const& res,
                       const params params,
                       raft::host_matrix_view<const float, int64_t> dataset);

/**
 * @brief Applies binary quantization transform to given dataset. If a dataset element is positive,
 * set the corresponding bit to 1.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * raft::device_matrix<float, uint64_t> dataset = read_dataset(filename);
 * int64_t quantized_dim = raft::div_rounding_up_safe(dataset.extent(1), sizeof(uint8_t) * 8);
 * auto quantized_dataset = raft::make_device_matrix<uint8_t, int64_t>(
 *    handle, dataset.extent(0), quantized_dim);
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params,
 * raft::make_const_mdspan(dataset.view()));
 * cuvs::preprocessing::quantize::binary::transform(handle, quantizer,
 * raft::make_const_mdspan(dataset.view()), quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a binary quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer<float>& quantizer,
               raft::device_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> out);

/**
 * @brief Applies binary quantization transform to given dataset. If a dataset element is positive,
 * set the corresponding bit to 1.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * raft::host_matrix<float, uint64_t> dataset = read_dataset(filename);
 * int64_t quantized_dim = raft::div_rounding_up_safe(dataset.extent(1), sizeof(uint8_t) * 8);
 * auto quantized_dataset = raft::make_host_matrix<uint8_t, int64_t>(
 *    handle, params, dataset.extent(0), quantized_dim);
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params,
 * raft::make_const_mdspan(dataset.view()));
 * cuvs::preprocessing::quantize::binary::transform(handle, quantizer,
 * raft::make_const_mdspan(dataset.view()), quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a binary quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const quantizer<float>& quantizer,
               raft::host_matrix_view<const float, int64_t> dataset,
               raft::host_matrix_view<uint8_t, int64_t> out);

/**
 * @brief Initializes a binary quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure binary quantizer, e.g. threshold
 * @param[in] dataset a row-major matrix view on device
 *
 * @return quantizer
 */
quantizer<half> train(raft::resources const& res,
                      const params params,
                      raft::device_matrix_view<const half, int64_t> dataset);

/**
 * @brief Initializes a binary quantizer to be used later for quantizing the dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure binary quantizer, e.g. threshold
 * @param[in] dataset a row-major matrix view on host
 *
 * @return quantizer
 */
quantizer<half> train(raft::resources const& res,
                      const params params,
                      raft::host_matrix_view<const half, int64_t> dataset);

/**
 * @brief Applies binary quantization transform to given dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * raft::device_matrix<half, uint64_t> dataset = read_dataset(filename);
 * int64_t quantized_dim = raft::div_rounding_up_safe(dataset.extent(1), sizeof(uint8_t) * 8);
 * auto quantized_dataset = raft::make_device_matrix<uint8_t, int64_t>(
 *    handle, params, dataset.extent(0), quantized_dim);
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params,
 * raft::make_const_mdspan(dataset.view()));
 * cuvs::preprocessing::quantize::binary::transform(handle, quantizer,
 * raft::make_const_mdspan(dataset.view()), quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a binary quantizer
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
void transform(raft::resources const& res,
               const quantizer<half>& quantizer,
               raft::device_matrix_view<const half, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> out);

/**
 * @brief Applies binary quantization transform to given dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::binary::params params;
 * raft::host_matrix<half, uint64_t> dataset = read_dataset(filename);
 * int64_t quantized_dim = raft::div_rounding_up_safe(dataset.extent(1), sizeof(uint8_t) * 8);
 * auto quantized_dataset = raft::make_host_matrix<uint8_t, int64_t>(
 *    handle, params, dataset.extent(0), quantized_dim);
 * auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, params,
 * raft::make_const_mdspan(dataset.view()));
 * cuvs::preprocessing::quantize::binary::transform(handle, quantizer,
 * raft::make_const_mdspan(dataset.view()), quantized_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a binary quantizer
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
void transform(raft::resources const& res,
               const quantizer<half>& quantizer,
               raft::host_matrix_view<const half, int64_t> dataset,
               raft::host_matrix_view<uint8_t, int64_t> out);

/**
 * @brief [deprecated] Applies binary quantization transform to given dataset. If a dataset element
 * is positive, set the corresponding bit to 1.
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
[[deprecated("please create and specify a quantizer")]] void transform(
  raft::resources const& res,
  raft::device_matrix_view<const double, int64_t> dataset,
  raft::device_matrix_view<uint8_t, int64_t> out);

/**
 * @brief [deprecated] Applies binary quantization transform to given dataset. If a dataset element
 * is positive, set the corresponding bit to 1.
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
[[deprecated("please create and specify a quantizer")]] void transform(
  raft::resources const& res,
  raft::host_matrix_view<const double, int64_t> dataset,
  raft::host_matrix_view<uint8_t, int64_t> out);

/**
 * @brief [deprecated] Applies binary quantization transform to given dataset. If a dataset element
 * is positive, set the corresponding bit to 1.
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
[[deprecated("please create and specify a quantizer")]] void transform(
  raft::resources const& res,
  raft::device_matrix_view<const float, int64_t> dataset,
  raft::device_matrix_view<uint8_t, int64_t> out);

/**
 * @brief [deprecated] Applies binary quantization transform to given dataset. If a dataset element
 * is positive, set the corresponding bit to 1.
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
[[deprecated("please create and specify a quantizer")]] void transform(
  raft::resources const& res,
  raft::host_matrix_view<const float, int64_t> dataset,
  raft::host_matrix_view<uint8_t, int64_t> out);

/**
 * @brief [deprecated] Applies binary quantization transform to given dataset. If a dataset element
 * is positive, set the corresponding bit to 1.
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major matrix view on device
 * @param[out] out a row-major matrix view on device
 *
 */
[[deprecated("please create and specify a quantizer")]] void transform(
  raft::resources const& res,
  raft::device_matrix_view<const half, int64_t> dataset,
  raft::device_matrix_view<uint8_t, int64_t> out);

/**
 * @brief [deprecated] Applies binary quantization transform to given dataset. If a dataset element
 * is positive, set the corresponding bit to 1.
 *
 * @param[in] res raft resource
 * @param[in] dataset a row-major matrix view on host
 * @param[out] out a row-major matrix view on host
 *
 */
[[deprecated("please create and specify a quantizer")]] void transform(
  raft::resources const& res,
  raft::host_matrix_view<const half, int64_t> dataset,
  raft::host_matrix_view<uint8_t, int64_t> out);

/** @} */  // end of group binary

}  // namespace binary
}  // namespace quantize
}  // namespace preprocessing
}  // namespace CUVS_EXPORT cuvs
