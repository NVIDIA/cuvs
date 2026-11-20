/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cuda/std/limits>
#include <memory>
#include <dlpack/dlpack.h>

#include <cuvs/core/c_api.h>
#include <cuvs/preprocessing/quantize/scalar.h>
#include <cuvs/preprocessing/quantize/scalar.hpp>

#include "../../core/exceptions.hpp"
#include "../../core/interop.hpp"

namespace {

namespace scalar = cuvs::preprocessing::quantize::scalar;

// Keep the public min/max layout while owning per-dimension bounds behind the factory handle.
struct scalar_quantizer : cuvsScalarQuantizer {
  uintptr_t addr = 0;
  DLDataType dtype{};
};

template <typename T>
scalar::quantizer<T> _train(cuvsResources_t res,
                            scalar::params params,
                            DLManagedTensor* dataset_tensor)
{
  auto dataset = dataset_tensor->dl_tensor;
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    return scalar::train(*res_ptr, params, cuvs::core::from_dlpack<mdspan_type>(dataset_tensor));
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    return scalar::train(*res_ptr, params, cuvs::core::from_dlpack<mdspan_type>(dataset_tensor));
  } else {
    RAFT_FAIL("dataset must be accessible on host or device memory");
  }
}

template <typename T, typename OutputT = int8_t>
void _transform(cuvsResources_t res,
                const scalar::quantizer<T>& quantizer,
                DLManagedTensor* dataset_tensor,
                DLManagedTensor* out_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto dataset = dataset_tensor->dl_tensor;
  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type     = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    using out_mdspan_type = raft::device_matrix_view<OutputT, int64_t, raft::row_major>;

    cuvs::preprocessing::quantize::scalar::transform(
      *res_ptr,
      quantizer,
      cuvs::core::from_dlpack<mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(out_tensor));

  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type     = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    using out_mdspan_type = raft::host_matrix_view<OutputT, int64_t, raft::row_major>;

    cuvs::preprocessing::quantize::scalar::transform(
      *res_ptr,
      quantizer,
      cuvs::core::from_dlpack<mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(out_tensor));
  } else {
    RAFT_FAIL("dataset must be accessible on host or device memory");
  }
}

template <typename OutputT, typename InputT = int8_t>
void _inverse_transform(cuvsResources_t res,
                        const scalar::quantizer<OutputT>& quantizer,
                        DLManagedTensor* dataset_tensor,
                        DLManagedTensor* out_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto dataset = dataset_tensor->dl_tensor;
  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type     = raft::device_matrix_view<InputT const, int64_t, raft::row_major>;
    using out_mdspan_type = raft::device_matrix_view<OutputT, int64_t, raft::row_major>;

    cuvs::preprocessing::quantize::scalar::inverse_transform(
      *res_ptr,
      quantizer,
      cuvs::core::from_dlpack<mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(out_tensor));

  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type     = raft::host_matrix_view<InputT const, int64_t, raft::row_major>;
    using out_mdspan_type = raft::host_matrix_view<OutputT, int64_t, raft::row_major>;

    cuvs::preprocessing::quantize::scalar::inverse_transform(
      *res_ptr,
      quantizer,
      cuvs::core::from_dlpack<mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(out_tensor));
  } else {
    RAFT_FAIL("dataset must be accessible on host or device memory");
  }
}

template <typename T>
void _transform(cuvsResources_t res,
                cuvsScalarQuantizer_t quantizer,
                DLManagedTensor* dataset,
                DLManagedTensor* out)
{
  _transform(res,
             scalar::quantizer<T>{static_cast<T>(quantizer->min_), static_cast<T>(quantizer->max_)},
             dataset,
             out);
}

template <typename T>
void _inverse_transform(cuvsResources_t res,
                        cuvsScalarQuantizer_t quantizer,
                        DLManagedTensor* dataset,
                        DLManagedTensor* out)
{
  _inverse_transform(res,
                     scalar::quantizer<T>{static_cast<T>(quantizer->min_), static_cast<T>(quantizer->max_)},
                     dataset,
                     out);
}

template <typename Fn>
void dispatch_float(DLDataType dtype, Fn&& fn)
{
  RAFT_EXPECTS(dtype.code == kDLFloat && dtype.lanes == 1, "Expected a scalar floating-point dtype");
  switch (dtype.bits) {
    case 16: fn.template operator()<half>(); break;
    case 32: fn.template operator()<float>(); break;
    case 64: fn.template operator()<double>(); break;
    default: RAFT_FAIL("Unsupported floating-point dtype bits: %d", dtype.bits);
  }
}

void destroy_per_dim(scalar_quantizer* quantizer)
{
  if (quantizer->addr) {
    dispatch_float(quantizer->dtype, [&]<typename T>() {
      delete reinterpret_cast<scalar::quantizer<T>*>(quantizer->addr);
    });
    quantizer->addr = 0;
  }
}

}  // namespace

extern "C" cuvsError_t cuvsScalarQuantizerParamsCreate(cuvsScalarQuantizerParams_t* params)
{
  return cuvs::core::translate_exceptions(
    [=] { *params = new cuvsScalarQuantizerParams{.quantile = 0.99, .mode = CUVS_SCALAR_GLOBAL}; });
}

extern "C" cuvsError_t cuvsScalarQuantizerParamsDestroy(cuvsScalarQuantizerParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsScalarQuantizerCreate(cuvsScalarQuantizer_t* quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(quantizer != nullptr, "quantizer must not be null");
    *quantizer = new scalar_quantizer{};
  });
}

extern "C" cuvsError_t cuvsScalarQuantizerDestroy(cuvsScalarQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer) {
      auto q = static_cast<scalar_quantizer*>(quantizer);
      destroy_per_dim(q);
      delete q;
    }
  });
}

extern "C" cuvsError_t cuvsScalarQuantizerTrain(cuvsResources_t res,
                                                cuvsScalarQuantizerParams_t params,
                                                DLManagedTensor* dataset_tensor,
                                                cuvsScalarQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(params && dataset_tensor && quantizer,
                 "params, dataset, and quantizer must not be null");
    scalar::params cpp_params;
    cpp_params.quantile = params->quantile;
    switch (params->mode) {
      case CUVS_SCALAR_GLOBAL: cpp_params.range = scalar::quantization_mode::GLOBAL; break;
      case CUVS_SCALAR_PER_DIMENSION:
        cpp_params.range = scalar::quantization_mode::PER_DIMENSION;
        break;
      default: RAFT_FAIL("Unsupported scalar quantization mode");
    }
    auto q = static_cast<scalar_quantizer*>(quantizer);
    dispatch_float(dataset_tensor->dl_tensor.dtype, [&]<typename T>() {
      auto trained = _train<T>(res, cpp_params, dataset_tensor);
      if (auto const* bounds =
            std::get_if<typename scalar::quantizer<T>::global_type>(&trained.value)) {
        destroy_per_dim(q);
        q->min_ = bounds->min;
        q->max_ = bounds->max;
      } else {
        auto owned = std::make_unique<scalar::quantizer<T>>(std::move(trained));
        destroy_per_dim(q);
        q->addr = reinterpret_cast<uintptr_t>(owned.release());
        q->dtype = dataset_tensor->dl_tensor.dtype;
        q->min_ = static_cast<double>(cuda::std::numeric_limits<T>::min());
        q->max_ = static_cast<double>(cuda::std::numeric_limits<T>::max());
      }
    });
  });
}

extern "C" cuvsError_t cuvsScalarQuantizerTransform(cuvsResources_t res,
                                                    cuvsScalarQuantizer_t quantizer,
                                                    DLManagedTensor* dataset_tensor,
                                                    DLManagedTensor* out_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(quantizer && dataset_tensor && out_tensor,
                 "quantizer, dataset, and out must not be null");
    auto q = static_cast<scalar_quantizer*>(quantizer);
    dispatch_float(q->addr ? q->dtype : dataset_tensor->dl_tensor.dtype, [&]<typename T>() {
      if (q->addr) {
        _transform(res, *reinterpret_cast<scalar::quantizer<T>*>(q->addr), dataset_tensor, out_tensor);
      } else {
        _transform<T>(res, quantizer, dataset_tensor, out_tensor);
      }
    });
  });
}

cuvsError_t cuvsScalarQuantizerInverseTransform(cuvsResources_t res,
                                                cuvsScalarQuantizer_t quantizer,
                                                DLManagedTensor* dataset,
                                                DLManagedTensor* out)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(quantizer && dataset && out, "quantizer, dataset, and out must not be null");
    auto q = static_cast<scalar_quantizer*>(quantizer);
    dispatch_float(q->addr ? q->dtype : out->dl_tensor.dtype, [&]<typename T>() {
      if (q->addr) {
        _inverse_transform(res, *reinterpret_cast<scalar::quantizer<T>*>(q->addr), dataset, out);
      } else {
        _inverse_transform<T>(res, quantizer, dataset, out);
      }
    });
  });
}
