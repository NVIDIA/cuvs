/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>
#include <memory>
#include <optional>
#include <utility>

#include <cuvs/core/c_api.h>
#include <cuvs/preprocessing/quantize/scalar.h>
#include <cuvs/preprocessing/quantize/scalar.hpp>

#include "../../core/exceptions.hpp"
#include "../../core/interop.hpp"

namespace {

namespace scalar = cuvs::preprocessing::quantize::scalar;

// The handle's addr points to a scalar::quantizer<T>, where dtype selects T.
template <typename Fn>
void with_held_type(cuvsScalarQuantizer_t quantizer, Fn&& fn)
{
  RAFT_EXPECTS(quantizer->addr != 0, "quantizer must be allocated by cuvsScalarQuantizerCreate");
  switch (quantizer->dtype.bits) {
    case 16: fn.template operator()<half>(); break;
    case 32: fn.template operator()<float>(); break;
    case 64: fn.template operator()<double>(); break;
    default: RAFT_FAIL("Unsupported quantizer dtype bits: %d", quantizer->dtype.bits);
  }
}

void destroy(cuvsScalarQuantizer_t quantizer)
{
  if (quantizer->addr == 0) { return; }
  with_held_type(quantizer, [&]<typename T>() {
    delete reinterpret_cast<scalar::quantizer<T>*>(quantizer->addr);
  });
  quantizer->addr = 0;
}

// Replaces the quantizer the handle owns.
template <typename T>
void reset(cuvsScalarQuantizer_t quantizer, scalar::quantizer<T>&& value)
{
  auto owned = std::make_unique<scalar::quantizer<T>>(std::move(value));
  destroy(quantizer);
  quantizer->addr  = reinterpret_cast<uintptr_t>(owned.release());
  quantizer->dtype = DLDataType{kDLFloat, sizeof(T) * 8, 1};
}

// The global bounds held by the quantizer; empty for per-dimension bounds.
std::optional<std::pair<double, double>> global_bounds(cuvsScalarQuantizer_t quantizer)
{
  std::optional<std::pair<double, double>> result;
  with_held_type(quantizer, [&]<typename T>() {
    auto const& value = reinterpret_cast<scalar::quantizer<T>*>(quantizer->addr)->value;
    if (auto const* bounds = std::get_if<typename scalar::quantizer<T>::global_type>(&value)) {
      result = std::pair{static_cast<double>(bounds->min), static_cast<double>(bounds->max)};
    }
  });
  return result;
}

// The quantizer for data of type T. Global bounds apply to every floating-point type and are
// converted into `converted`; per-dimension bounds require the type they were trained on.
template <typename T>
const scalar::quantizer<T>& typed_quantizer(cuvsScalarQuantizer_t quantizer,
                                            scalar::quantizer<T>& converted)
{
  auto const bounds = global_bounds(quantizer);
  if (quantizer->dtype.bits == sizeof(T) * 8) {
    return *reinterpret_cast<scalar::quantizer<T>*>(quantizer->addr);
  }
  RAFT_EXPECTS(bounds.has_value(),
               "The data type must match the one the per-dimension quantizer was trained on");
  converted = scalar::quantizer<T>{static_cast<T>(bounds->first), static_cast<T>(bounds->second)};
  return converted;
}

template <typename T>
void _train(cuvsResources_t res,
            cuvsScalarQuantizerParams params,
            DLManagedTensor* dataset_tensor,
            cuvsScalarQuantizer_t quantizer)
{
  auto dataset = dataset_tensor->dl_tensor;

  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto quantizer_params     = cuvs::preprocessing::quantize::scalar::params();
  quantizer_params.quantile = params.quantile;
  switch (params.mode) {
    case CUVS_SCALAR_GLOBAL: quantizer_params.range = scalar::quantization_mode::GLOBAL; break;
    case CUVS_SCALAR_PER_DIMENSION:
      quantizer_params.range = scalar::quantization_mode::PER_DIMENSION;
      break;
    default: RAFT_FAIL("Unsupported scalar quantization mode");
  }

  cuvs::preprocessing::quantize::scalar::quantizer<T> ret;

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    ret = cuvs::preprocessing::quantize::scalar::train(*res_ptr, quantizer_params, mds);
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    ret = cuvs::preprocessing::quantize::scalar::train(*res_ptr, quantizer_params, mds);
  } else {
    RAFT_FAIL("dataset must be accessible on host or device memory");
  }

  reset(quantizer, std::move(ret));
}

template <typename T, typename OutputT = int8_t>
void _transform(cuvsResources_t res,
                cuvsScalarQuantizer_t quantizer_,
                DLManagedTensor* dataset_tensor,
                DLManagedTensor* out_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  cuvs::preprocessing::quantize::scalar::quantizer<T> converted;
  auto const& quantizer = typed_quantizer(quantizer_, converted);

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
                        cuvsScalarQuantizer_t quantizer_,
                        DLManagedTensor* dataset_tensor,
                        DLManagedTensor* out_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  cuvs::preprocessing::quantize::scalar::quantizer<OutputT> converted;
  auto const& quantizer = typed_quantizer(quantizer_, converted);

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
    auto created = std::make_unique<cuvsScalarQuantizer>();
    reset(created.get(), scalar::quantizer<double>{});
    *quantizer = created.release();
  });
}

extern "C" cuvsError_t cuvsScalarQuantizerDestroy(cuvsScalarQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer != nullptr) {
      destroy(quantizer);
      delete quantizer;
    }
  });
}

extern "C" cuvsError_t cuvsScalarQuantizerGetBounds(cuvsScalarQuantizer_t quantizer,
                                                    double* min,
                                                    double* max)
{
  return cuvs::core::translate_exceptions([=] {
    auto const bounds = global_bounds(quantizer);
    RAFT_EXPECTS(bounds.has_value(), "A per-dimension quantizer has no global bounds");
    *min = bounds->first;
    *max = bounds->second;
  });
}

extern "C" cuvsError_t cuvsScalarQuantizerSetBounds(cuvsScalarQuantizer_t quantizer,
                                                    double min,
                                                    double max)
{
  return cuvs::core::translate_exceptions(
    [=] { reset(quantizer, scalar::quantizer<double>{min, max}); });
}

extern "C" cuvsError_t cuvsScalarQuantizerTrain(cuvsResources_t res,
                                                cuvsScalarQuantizerParams_t params,
                                                DLManagedTensor* dataset_tensor,
                                                cuvsScalarQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      _train<float>(res, *params, dataset_tensor, quantizer);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      _train<half>(res, *params, dataset_tensor, quantizer);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      _train<double>(res, *params, dataset_tensor, quantizer);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsScalarQuantizerTransform(cuvsResources_t res,
                                                    cuvsScalarQuantizer_t quantizer,
                                                    DLManagedTensor* dataset_tensor,
                                                    DLManagedTensor* out_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      _transform<float>(res, quantizer, dataset_tensor, out_tensor);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      _transform<half>(res, quantizer, dataset_tensor, out_tensor);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      _transform<double>(res, quantizer, dataset_tensor, out_tensor);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

cuvsError_t cuvsScalarQuantizerInverseTransform(cuvsResources_t res,
                                                cuvsScalarQuantizer_t quantizer,
                                                DLManagedTensor* dataset,
                                                DLManagedTensor* out)
{
  return cuvs::core::translate_exceptions([=] {
    auto dtype = out->dl_tensor.dtype;
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      _inverse_transform<float>(res, quantizer, dataset, out);
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      _inverse_transform<half>(res, quantizer, dataset, out);
    } else if (dtype.code == kDLFloat && dtype.bits == 64) {
      _inverse_transform<double>(res, quantizer, dataset, out);
    } else {
      RAFT_FAIL(
        "Unsupported output dataset DLtensor dtype: %d and bits: %d", dtype.code, dtype.bits);
    }
  });
}
