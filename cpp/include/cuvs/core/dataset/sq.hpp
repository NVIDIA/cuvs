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
template <typename MathT,
          typename CodePolicy,
          typename QuantizerT = cuvs::preprocessing::quantize::scalar::quantizer<MathT>>
struct sq_spec {
  // One code per component, so the codes are laid out exactly like a dense dataset.
  template <typename IdxT>
  using storage_spec =
    typename mdarray_spec<raft::layout_c_contiguous, CodePolicy>::template apply<int8_t, IdxT>;

  using math_type = MathT;

  // The manifest is either a global interval or owning per-dimension offsets and a shared scale.
  using manifest_type      = QuantizerT;
  using manifest_view_type = typename manifest_type::view_type;

  template <typename T, typename IdxT>
  struct apply : storage_spec<IdxT> {
    // The dataset stores codes, but it still represents vectors of `T`.
    using value_type = std::remove_cv_t<T>;
    using code_type  = int8_t;
    // NOTE: re-exported rather than declared here, so that they do not belong to one instantiation
    // of `apply`; qualified lookup into `apply` does not reach the enclosing spec on its own.
    using math_type          = sq_spec::math_type;
    using manifest_type      = sq_spec::manifest_type;
    using manifest_view_type = sq_spec::manifest_view_type;

    // NOTE: every accessor is inherited; the dimension is still one code per component, so even
    // `get_dim` is the dense one.
    [[nodiscard]] static auto get_manifest_view(const manifest_type& manifest) noexcept
      -> manifest_view_type
    {
      return manifest.view();
    }
    /** Borrow read-only quantizer parameters; the manifest must outlive use of the view. */
    [[nodiscard]] static auto get_manifest_const_view(const manifest_type& manifest) noexcept
      -> manifest_view_type
    {
      return manifest.view();
    }
    /** Return the read-only view; its owner must outlive stream work using it. */
    [[nodiscard]] static auto get_manifest_const_view(manifest_view_type manifest) noexcept
      -> manifest_view_type
      requires(!std::is_same_v<manifest_type, manifest_view_type>)
    {
      return manifest;
    }
  };
};

template <typename T,
          typename IdxT,
          typename MathT,
          typename CodePolicy,
          typename QuantizerT = cuvs::preprocessing::quantize::scalar::quantizer<MathT>>
using sq_dataset = dataset<T, IdxT, sq_spec<MathT, CodePolicy, QuantizerT>>;
template <typename T, typename IdxT, typename MathT>
using device_sq_dataset =
  sq_dataset<T, IdxT, MathT, raft::device_accessor<raft::device_container_policy<int8_t>>>;
template <typename T, typename IdxT, typename MathT>
using host_sq_dataset =
  sq_dataset<T, IdxT, MathT, raft::host_accessor<raft::host_container_policy<int8_t>>>;

/** Owns row-major INT8 codes and their per-dimension offsets/shared scale. */
using device_per_dim_sq_dataset =
  sq_dataset<float,
             int64_t,
             float,
             raft::device_accessor<raft::device_container_policy<int8_t>>,
             cuvs::preprocessing::quantize::scalar::per_dim_quantizer>;

namespace preprocessing::quantize::scalar {

/**
 * Train and encode a device FP32 corpus. Reuse result.const_manifest_view() with transform
 * for query batches. For separate training/encoding, use train_per_dim and transform.
 */
inline device_per_dim_sq_dataset build_per_dim(
  raft::resources const& res,
  raft::device_matrix_view<const float, int64_t> dataset,
  params params = {})
{
  auto quantizer = train_per_dim(res, dataset, params);
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, dataset.extent(0), dataset.extent(1));
  transform(res, quantizer.view(), dataset, codes.view());
  return device_per_dim_sq_dataset{std::move(codes), std::move(quantizer)};
}

}  // namespace preprocessing::quantize::scalar

}  // namespace CUVS_EXPORT cuvs
