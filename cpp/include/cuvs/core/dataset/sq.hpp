/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/dense.hpp>
#include <cuvs/core/dataset/type.hpp>
#include <cuvs/core/export.hpp>
#include <cuvs/preprocessing/quantize/scalar.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace CUVS_EXPORT cuvs {

/**
 * Scalar-quantized dataset specification: every component of a vector becomes one code, so the
 * manifest holds the quantizer and the dimension is still a property
 * of the data. The manifest also holds the bound on the reconstruction error of the codes, which
 * is a property of the dataset rather than of the quantizer.
 */
template <typename MathT, typename CodePolicy>
struct sq_spec {
  // One code per component, so the codes are laid out exactly like a dense dataset.
  template <typename IdxT>
  using storage_spec =
    typename mdarray_spec<raft::layout_c_contiguous, CodePolicy>::template apply<int8_t, IdxT>;

  using math_type = MathT;

  using quantizer_type = cuvs::preprocessing::quantize::scalar::quantizer;
  using error_type     = double;

  /**
   * Borrowed quantizer parameters and the error bound of the dataset's codes.
   * The quantizer view base keeps the parameters directly accessible.
   */
  template <typename QuantizerViewT>
  struct manifest_view : QuantizerViewT {
    /**
     * Upper bound on the L2 error between a dataset row and its int8 encode/decode result.
     * Infinity when it was not measured or when the error is not finite. Modifying the codes or
     * the quantizer parameters invalidates a measured bound.
     */
    error_type max_error = std::numeric_limits<error_type>::infinity();

    /** Construct default quantizer parameters with no measured error bound. */
    manifest_view() = default;
    /** Borrow quantizer parameters together with the error bound of the codes they produced. */
    manifest_view(  // NOLINT(google-explicit-constructor)
      QuantizerViewT quantizer,
      error_type error = std::numeric_limits<error_type>::infinity())
      : QuantizerViewT{quantizer}, max_error{error}
    {
    }
  };
  using manifest_view_type       = manifest_view<typename quantizer_type::view_type>;
  using const_manifest_view_type = manifest_view<typename quantizer_type::const_view_type>;

  /**
   * Owns the quantizer, with either global or per-dimension bounds, and the error bound of the
   * dataset's codes.
   */
  struct manifest_type : quantizer_type {
    /** See manifest_view::max_error. */
    error_type max_error = std::numeric_limits<error_type>::infinity();

    /** Construct default global bounds with no measured error bound. */
    manifest_type() = default;
    /** Take ownership of a quantizer and, optionally, the measured error bound of its codes. */
    manifest_type(  // NOLINT(google-explicit-constructor)
      quantizer_type&& quantizer,
      error_type error = std::numeric_limits<error_type>::infinity())
      : quantizer_type{std::move(quantizer)}, max_error{error}
    {
    }

    /** Borrow mutable per-dimension bounds; the owner must outlive stream work using them. */
    [[nodiscard]] auto view() noexcept -> manifest_view_type
    {
      return {quantizer_type::view(), max_error};
    }
    /** Borrow read-only bounds; the owner must outlive stream work using per-dimension bounds. */
    [[nodiscard]] auto view() const noexcept -> const_manifest_view_type
    {
      return {quantizer_type::view(), max_error};
    }
  };

  template <typename T, typename IdxT>
  struct apply : storage_spec<IdxT> {
    // The dataset stores codes, but it still represents vectors of `T`.
    using value_type = std::remove_cv_t<T>;
    using code_type  = int8_t;
    // NOTE: re-exported rather than declared here, so that they do not belong to one instantiation
    // of `apply`; qualified lookup into `apply` does not reach the enclosing spec on its own.
    using math_type                = sq_spec::math_type;
    using manifest_type            = sq_spec::manifest_type;
    using manifest_view_type       = sq_spec::manifest_view_type;
    using const_manifest_view_type = sq_spec::const_manifest_view_type;

    // NOTE: every accessor is inherited; the dimension is still one code per component, so even
    // `get_dim` is the dense one.
    /** Borrow mutable per-dimension bounds from the owning manifest. */
    [[nodiscard]] static auto get_manifest_view(manifest_type& manifest) noexcept
      -> manifest_view_type
    {
      return manifest.view();
    }
    /** Borrow read-only quantizer parameters; the manifest must outlive use of the view. */
    [[nodiscard]] static auto get_manifest_const_view(const manifest_type& manifest) noexcept
      -> const_manifest_view_type
    {
      return manifest.view();
    }
    /** Convert a mutable view to read-only; its owner must outlive stream work using it. */
    [[nodiscard]] static auto get_manifest_const_view(manifest_view_type manifest) noexcept
      -> const_manifest_view_type
    {
      return {manifest, manifest.max_error};
    }
  };
};

template <typename T, typename IdxT, typename MathT, typename CodePolicy>
using sq_dataset = dataset<T, IdxT, sq_spec<MathT, CodePolicy>>;
template <typename T, typename IdxT, typename MathT>
using device_sq_dataset =
  sq_dataset<T, IdxT, MathT, raft::device_accessor<raft::device_container_policy<int8_t>>>;
template <typename T, typename IdxT, typename MathT>
using host_sq_dataset =
  sq_dataset<T, IdxT, MathT, raft::host_accessor<raft::host_container_policy<int8_t>>>;

namespace preprocessing::quantize::scalar {

/**
 * @defgroup scalar_dataset Scalar-quantized dataset construction
 * @{
 */
/**
 * @brief Train and encode a device corpus with the selected quantization mode.
 * For separate training and encoding, use train and transform.
 *
 * @param[in] res Resources used for training and allocation.
 * @param[in] params Scalar quantizer parameters.
 * @param[in] dataset Device corpus to train on and encode.
 * @param[in] compute_max_error Measure the dataset's maximum reconstruction error over every row.
 * Default: false, which leaves the manifest's max_error infinite.
 * @return Device dataset owning the quantized codes, the quantizer and the error bound.
 */
template <typename T>
inline device_sq_dataset<T, int64_t, T> make_sq_dataset(
  raft::resources const& res,
  params params,
  raft::device_matrix_view<const T, int64_t> dataset,
  bool compute_max_error = false)
{
  using result_type = device_sq_dataset<T, int64_t, T>;
  auto quantizer    = train(res, params, dataset);
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, dataset.extent(0), dataset.extent(1));
  transform(res, quantizer, dataset, codes.view());
  typename result_type::manifest_type manifest{std::move(quantizer)};
  // Qualified: the parameter hides the function of the same name.
  if (compute_max_error) { manifest.max_error = scalar::compute_max_error(res, manifest, dataset); }
  return result_type{std::move(codes), std::move(manifest)};
}

/**
 * @brief Train and encode a host corpus in GPU batches, returning device-resident int8 codes.
 * Uses one quantizer across all batches.
 * Configure params.device_buffer_samples to bound input staging and an auxiliary stream pool
 * on res to enable double buffering. All int8 codes must fit in device memory.
 *
 * @param[in] res Resources used for training, allocation, and transfer streams.
 * @param[in] params Scalar quantizer and host batching parameters.
 * @param[in] dataset Row-major host corpus; pageable and pinned input are supported.
 * @param[in] compute_max_error Measure the dataset's maximum reconstruction error over every row
 * during the encoding pass. Default: false, which leaves the manifest's max_error infinite.
 * @return Device dataset owning the quantized codes, the quantizer and the error bound.
 */
device_sq_dataset<float, int64_t, float> make_sq_dataset(
  raft::resources const& res,
  params params,
  raft::host_matrix_view<const float, int64_t> dataset,
  bool compute_max_error = false);

/**
 * @brief Train and encode a host corpus in GPU batches, returning device-resident int8 codes.
 * Uses one quantizer across all batches.
 * Configure params.device_buffer_samples to bound input staging and an auxiliary stream pool
 * on res to enable double buffering. All int8 codes must fit in device memory.
 *
 * @param[in] res Resources used for training, allocation, and transfer streams.
 * @param[in] params Scalar quantizer and host batching parameters.
 * @param[in] dataset Row-major host corpus; pageable and pinned input are supported.
 * @param[in] compute_max_error Measure the dataset's maximum reconstruction error over every row
 * during the encoding pass. Default: false, which leaves the manifest's max_error infinite.
 * @return Device dataset owning the quantized codes, the quantizer and the error bound.
 */
device_sq_dataset<double, int64_t, double> make_sq_dataset(
  raft::resources const& res,
  params params,
  raft::host_matrix_view<const double, int64_t> dataset,
  bool compute_max_error = false);

/**
 * @brief Train and encode a host corpus in GPU batches, returning device-resident int8 codes.
 * Uses one quantizer across all batches.
 * Configure params.device_buffer_samples to bound input staging and an auxiliary stream pool
 * on res to enable double buffering. All int8 codes must fit in device memory.
 *
 * @param[in] res Resources used for training, allocation, and transfer streams.
 * @param[in] params Scalar quantizer and host batching parameters.
 * @param[in] dataset Row-major host corpus; pageable and pinned input are supported.
 * @param[in] compute_max_error Measure the dataset's maximum reconstruction error over every row
 * during the encoding pass. Default: false, which leaves the manifest's max_error infinite.
 * @return Device dataset owning the quantized codes, the quantizer and the error bound.
 */
device_sq_dataset<half, int64_t, half> make_sq_dataset(
  raft::resources const& res,
  params params,
  raft::host_matrix_view<const half, int64_t> dataset,
  bool compute_max_error = false);

/** @} */

}  // namespace preprocessing::quantize::scalar

}  // namespace CUVS_EXPORT cuvs
