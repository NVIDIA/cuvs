/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>

#include <cuvs/core/c_api.h>
#include <cuvs/preprocessing/quantize/binary.h>
#include <cuvs/preprocessing/quantize/binary.hpp>

#include "../../core/exceptions.hpp"
#include "../../core/interop.hpp"

namespace {

template <typename T, typename OutputT = uint8_t>
void _transform(cuvsResources_t res,
                cuvsBinaryQuantizer_t quantizer,
                DLManagedTensor* dataset_tensor,
                DLManagedTensor* out_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto q = reinterpret_cast<cuvs::preprocessing::quantize::binary::quantizer<T>*>(quantizer->addr);

  auto dataset = dataset_tensor->dl_tensor;
  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type     = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    using out_mdspan_type = raft::device_matrix_view<OutputT, int64_t, raft::row_major>;

    cuvs::preprocessing::quantize::binary::transform(
      *res_ptr,
      *q,
      cuvs::core::from_dlpack<mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(out_tensor));

  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type     = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    using out_mdspan_type = raft::host_matrix_view<OutputT, int64_t, raft::row_major>;

    cuvs::preprocessing::quantize::binary::transform(
      *res_ptr,
      *q,
      cuvs::core::from_dlpack<mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(out_tensor));
  } else {
    RAFT_FAIL("dataset must be accessible on host or device memory");
  }
}

template <typename T>
void* _train(cuvsResources_t res,
             cuvsBinaryQuantizerParams_t params,
             DLManagedTensor* dataset_tensor)
{
  auto dataset = dataset_tensor->dl_tensor;

  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto quantizer_params           = cuvs::preprocessing::quantize::binary::params();
  quantizer_params.sampling_ratio = params->sampling_ratio;
  switch (params->threshold) {
    case ZERO:
      quantizer_params.threshold = cuvs::preprocessing::quantize::binary::bit_threshold::zero;
      break;
    case MEAN:
      quantizer_params.threshold = cuvs::preprocessing::quantize::binary::bit_threshold::mean;
      break;
    case SAMPLING_MEDIAN:
      quantizer_params.threshold =
        cuvs::preprocessing::quantize::binary::bit_threshold::sampling_median;
      break;
    default: RAFT_FAIL("Unsupported threshold");
  }

  auto ret = new cuvs::preprocessing::quantize::binary::quantizer<T>();

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    *ret = cuvs::preprocessing::quantize::binary::train(*res_ptr, quantizer_params, mds);
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    *ret = cuvs::preprocessing::quantize::binary::train(*res_ptr, quantizer_params, mds);
  } else {
    RAFT_FAIL("dataset must be accessible on host or device memory");
  }
  return ret;
}

}  // namespace

extern "C" cuvsError_t cuvsBinaryQuantizerParamsCreate(cuvsBinaryQuantizerParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params =
        new cuvsBinaryQuantizerParams{.threshold = MEAN, .sampling_ratio = 0.1};
  });
}

extern "C" cuvsError_t cuvsBinaryQuantizerParamsDestroy(cuvsBinaryQuantizerParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsBinaryQuantizerCreate(cuvsBinaryQuantizer_t* quantizer)
{
  return cuvs::core::translate_exceptions([=] { *quantizer = new cuvsBinaryQuantizer; });
}

extern "C" cuvsError_t cuvsBinaryQuantizerDestroy(cuvsBinaryQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] { delete quantizer; });
}

extern "C" cuvsError_t cuvsBinaryQuantizerTrain(cuvsResources_t res,
                                                cuvsBinaryQuantizerParams_t params,
                                                DLManagedTensor* dataset_tensor,
                                                cuvsBinaryQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset     = dataset_tensor->dl_tensor;
    quantizer->dtype = dataset.dtype;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      quantizer->addr = reinterpret_cast<uintptr_t>(_train<double>(res, params, dataset_tensor));
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      quantizer->addr = reinterpret_cast<uintptr_t>(_train<float>(res, params, dataset_tensor));
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      quantizer->addr = reinterpret_cast<uintptr_t>(_train<half>(res, params, dataset_tensor));
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsBinaryQuantizerTransformWithParams(cuvsResources_t res,
                                                              cuvsBinaryQuantizer_t quantizer,
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

extern "C" cuvsError_t cuvsBinaryQuantizerTransform(cuvsResources_t res,
                                                    DLManagedTensor* dataset_tensor,
                                                    DLManagedTensor* out_tensor)
{
  cuvsBinaryQuantizerParams_t params;
  cuvsBinaryQuantizerParamsCreate(&params);
  params->threshold = ZERO;

  cuvsBinaryQuantizer_t quantizer;
  cuvsBinaryQuantizerCreate(&quantizer);
  cuvsBinaryQuantizerTrain(res, params, dataset_tensor, quantizer);

  const auto result =
    cuvsBinaryQuantizerTransformWithParams(res, quantizer, dataset_tensor, out_tensor);

  cuvsBinaryQuantizerDestroy(quantizer);
  cuvsBinaryQuantizerParamsDestroy(params);
  return result;
}
