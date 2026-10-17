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

namespace CUVS_EXPORT cuvs {

/**
 * Scalar-quantized dataset specification: every component of a vector becomes one code, so the
 * dictionary is only the range the codes are mapped back onto and the dimension is still a property
 * of the data.
 */
template <typename MathT, typename CodePolicy>
struct sq_spec {
  // One code per component, so the codes are laid out exactly like a dense dataset.
  template <typename IdxT>
  using storage_spec =
    typename mdarray_spec<raft::layout_c_contiguous, CodePolicy>::template apply<int8_t, IdxT>;

  using math_type = MathT;

  // Nothing to allocate: the whole dictionary is the interval the codes are dequantized into,
  // which is exactly what the scalar quantizer holds.
  using dictionary_type      = cuvs::preprocessing::quantize::scalar::quantizer<math_type>;
  using dictionary_view_type = dictionary_type;

  template <typename T, typename IdxT>
  struct apply : storage_spec<IdxT> {
    // The dataset stores codes, but it still represents vectors of `T`.
    using value_type = std::remove_cv_t<T>;
    using code_type  = int8_t;
    // NOTE: re-exported rather than declared here, so that they do not belong to one instantiation
    // of `apply`; qualified lookup into `apply` does not reach the enclosing spec on its own.
    using math_type            = sq_spec::math_type;
    using dictionary_type      = sq_spec::dictionary_type;
    using dictionary_view_type = sq_spec::dictionary_view_type;

    // NOTE: every accessor is inherited; the dimension is still one code per component, so even
    // `get_dim` is the dense one.
    [[nodiscard]] static auto get_dictionary_view(const dictionary_type& dict) noexcept
      -> dictionary_view_type
    {
      return dict;
    }
    // NOTE: the dictionary is its own view here, read-only or not, so one overload serves both the
    // owning dataset and the view.
    [[nodiscard]] static auto get_dictionary_const_view(const dictionary_type& dict) noexcept
      -> dictionary_view_type
    {
      return dict;
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

}  // namespace CUVS_EXPORT cuvs
