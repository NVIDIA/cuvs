/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>
#include <optional>
#include <utility>
#include <variant>

#include <cuvs/core/c_api.h>
#include <cuvs/preprocessing/quantize/scalar.h>
#include <cuvs/preprocessing/quantize/scalar.hpp>

#include "../../core/exceptions.hpp"
#include "../../core/interop.hpp"

namespace scalar = cuvs::preprocessing::quantize::scalar;

// The definition of the public handle: the C++ quantizer for the type it was trained on. Its
// global bounds also apply to the other floating-point types; per-dimension bounds do not.
struct cuvsScalarQuantizer {
  std::variant<scalar::quantizer<double>, scalar::quantizer<float>, scalar::quantizer<half>> value;
};

namespace {

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

// The global bounds held by the quantizer; empty for per-dimension bounds.
std::optional<std::pair<double, double>> global_bounds(const cuvsScalarQuantizer& quantizer)
{
  return std::visit(
    []<typename T>(const scalar::quantizer<T>& q) -> std::optional<std::pair<double, double>> {
      if (auto const* bounds = std::get_if<typename scalar::quantizer<T>::global_type>(&q.value)) {
        return std::pair{static_cast<double>(bounds->min), static_cast<double>(bounds->max)};
      }
      return std::nullopt;
    },
    quantizer.value);
}

// Calls fn with the quantizer for input type T.
template <typename T, typename Fn>
void with_quantizer(const cuvsScalarQuantizer& quantizer, Fn&& fn)
{
  if (auto const* typed = std::get_if<scalar::quantizer<T>>(&quantizer.value)) {
    fn(*typed);
    return;
  }
  auto const bounds = global_bounds(quantizer);
  RAFT_EXPECTS(bounds.has_value(),
               "The data type must match the one the per-dimension quantizer was trained on");
  fn(scalar::quantizer<T>{static_cast<T>(bounds->first), static_cast<T>(bounds->second)});
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
    *quantizer = new cuvsScalarQuantizer{};
  });
}

extern "C" cuvsError_t cuvsScalarQuantizerDestroy(cuvsScalarQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] { delete quantizer; });
}

extern "C" cuvsError_t cuvsScalarQuantizerGetBounds(cuvsScalarQuantizer_t quantizer,
                                                    double* min,
                                                    double* max)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(quantizer && min && max, "quantizer, min, and max must not be null");
    auto const bounds = global_bounds(*quantizer);
    RAFT_EXPECTS(bounds.has_value(), "A per-dimension quantizer has no global bounds");
    *min = bounds->first;
    *max = bounds->second;
  });
}

extern "C" cuvsError_t cuvsScalarQuantizerSetBounds(cuvsScalarQuantizer_t quantizer,
                                                    double min,
                                                    double max)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(quantizer != nullptr, "quantizer must not be null");
    quantizer->value = scalar::quantizer<double>{min, max};
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
    dispatch_float(dataset_tensor->dl_tensor.dtype, [&]<typename T>() {
      quantizer->value = _train<T>(res, cpp_params, dataset_tensor);
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
    dispatch_float(dataset_tensor->dl_tensor.dtype, [&]<typename T>() {
      with_quantizer<T>(*quantizer, [&](const scalar::quantizer<T>& q) {
        _transform(res, q, dataset_tensor, out_tensor);
      });
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
    dispatch_float(out->dl_tensor.dtype, [&]<typename T>() {
      with_quantizer<T>(*quantizer, [&](const scalar::quantizer<T>& q) {
        _inverse_transform(res, q, dataset, out);
      });
    });
  });
}
