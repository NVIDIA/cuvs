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
#include <type_traits>
#include <utility>

namespace CUVS_EXPORT cuvs {

/**
 * Scalar-quantized dataset specification: every component of a vector becomes one code, so the
 * manifest holds the quantizer and the dimension is still a property
 * of the data.
 */
template <typename MathT, typename CodePolicy>
struct sq_spec {
  // One code per component, so the codes are laid out exactly like a dense dataset.
  template <typename IdxT>
  using storage_spec =
    typename mdarray_spec<raft::layout_c_contiguous, CodePolicy>::template apply<int8_t, IdxT>;

  using math_type = MathT;

  // The manifest owns either global bounds or per-dimension bounds.
  using manifest_type            = cuvs::preprocessing::quantize::scalar::quantizer<MathT>;
  using manifest_view_type       = typename manifest_type::view_type;
  using const_manifest_view_type = typename manifest_type::const_view_type;

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
      return manifest;
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
 * @return Device dataset owning the quantized codes and quantizer.
 */
template <typename T>
inline device_sq_dataset<T, int64_t, T> make_sq_dataset(
  raft::resources const& res, params params, raft::device_matrix_view<const T, int64_t> dataset)
{
  auto quantizer = train(res, params, dataset);
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, dataset.extent(0), dataset.extent(1));
  transform(res, quantizer, dataset, codes.view());
  return device_sq_dataset<T, int64_t, T>{std::move(codes), std::move(quantizer)};
}

/**
 * @brief Train and encode a host corpus in GPU batches, returning device-resident int8 codes.
 * Uses one quantizer across all batches and measures the maximum error over every row.
 * Configure params.device_buffer_samples to bound input staging and an auxiliary stream pool
 * on res to enable double buffering. All int8 codes must fit in device memory.
 *
 * @param[in] res Resources used for training, allocation, and transfer streams.
 * @param[in] params Scalar quantizer and host batching parameters.
 * @param[in] dataset Row-major host corpus; pageable and pinned input are supported.
 * @return Device dataset owning the quantized codes and quantizer.
 */
device_sq_dataset<float, int64_t, float> make_sq_dataset(
  raft::resources const& res, params params, raft::host_matrix_view<const float, int64_t> dataset);

/**
 * @brief Train and encode a host corpus in GPU batches, returning device-resident int8 codes.
 * Uses one quantizer across all batches and measures the maximum error over every row.
 * Configure params.device_buffer_samples to bound input staging and an auxiliary stream pool
 * on res to enable double buffering. All int8 codes must fit in device memory.
 *
 * @param[in] res Resources used for training, allocation, and transfer streams.
 * @param[in] params Scalar quantizer and host batching parameters.
 * @param[in] dataset Row-major host corpus; pageable and pinned input are supported.
 * @return Device dataset owning the quantized codes and quantizer.
 */
device_sq_dataset<double, int64_t, double> make_sq_dataset(
  raft::resources const& res, params params, raft::host_matrix_view<const double, int64_t> dataset);

/**
 * @brief Train and encode a host corpus in GPU batches, returning device-resident int8 codes.
 * Uses one quantizer across all batches and measures the maximum error over every row.
 * Configure params.device_buffer_samples to bound input staging and an auxiliary stream pool
 * on res to enable double buffering. All int8 codes must fit in device memory.
 *
 * @param[in] res Resources used for training, allocation, and transfer streams.
 * @param[in] params Scalar quantizer and host batching parameters.
 * @param[in] dataset Row-major host corpus; pageable and pinned input are supported.
 * @return Device dataset owning the quantized codes and quantizer.
 */
device_sq_dataset<half, int64_t, half> make_sq_dataset(
  raft::resources const& res, params params, raft::host_matrix_view<const half, int64_t> dataset);

/** @} */

}  // namespace preprocessing::quantize::scalar

}  // namespace CUVS_EXPORT cuvs
