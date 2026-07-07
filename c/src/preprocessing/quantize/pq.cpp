/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>

#include <cuvs/core/c_api.h>
#include <cuvs/preprocessing/quantize/pq.h>
#include <cuvs/preprocessing/quantize/pq.hpp>
#include "../../core/exceptions.hpp"
#include "../../core/interop.hpp"

namespace {

template <typename T, typename OutputT = uint8_t>
void _transform(cuvsResources_t res,
                cuvsProductQuantizer_t quantizer,
                DLManagedTensor* dataset_tensor,
                DLManagedTensor* codes_out_tensor,
                DLManagedTensor* vq_labels_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto q = reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<T>*>(quantizer->addr);

  auto dataset = dataset_tensor->dl_tensor;
  using mdspan_type     = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  using out_mdspan_type = raft::device_matrix_view<OutputT, int64_t, raft::row_major>;
  using vq_labels_mdspan_type = raft::device_vector_view<uint32_t, int64_t>;
  std::optional<vq_labels_mdspan_type> vq_labels = std::nullopt;
  if (vq_labels_tensor != NULL) {
    vq_labels = cuvs::core::from_dlpack<vq_labels_mdspan_type>(vq_labels_tensor);
  }
  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    cuvs::preprocessing::quantize::pq::transform(
      *res_ptr,
      *q,
      cuvs::core::from_dlpack<mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(codes_out_tensor),
      vq_labels);

  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using host_mdspan_type     = raft::host_matrix_view<T const, int64_t, raft::row_major>;

    cuvs::preprocessing::quantize::pq::transform(
      *res_ptr,
      *q,
      cuvs::core::from_dlpack<host_mdspan_type>(dataset_tensor),
      cuvs::core::from_dlpack<out_mdspan_type>(codes_out_tensor),
      vq_labels);
  } else {
    RAFT_FAIL("dataset must be accessible on host or device memory");
  }
}

template <typename T>
void* _build(cuvsResources_t res,
             cuvsProductQuantizerParams_t params,
             DLManagedTensor* dataset_tensor)
{
  auto dataset = dataset_tensor->dl_tensor;

  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  cuvs::preprocessing::quantize::pq::params quantizer_params(
    params->pq_bits, params->pq_dim, params->use_subspaces, params->use_vq, params->vq_n_centers,
    params->kmeans_n_iters,  static_cast<cuvs::cluster::kmeans::kmeans_type>(params->pq_kmeans_type), params->max_train_points_per_pq_code,
     params->max_train_points_per_vq_cluster);
  cuvs::preprocessing::quantize::pq::quantizer<T>* ret = nullptr;

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    ret = new cuvs::preprocessing::quantize::pq::quantizer<T>{
      cuvs::preprocessing::quantize::pq::build(*res_ptr, quantizer_params, mds)};
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using host_mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<host_mdspan_type>(dataset_tensor);
    ret = new cuvs::preprocessing::quantize::pq::quantizer<T>{
      cuvs::preprocessing::quantize::pq::build(*res_ptr, quantizer_params, mds)};
  } else {
    RAFT_FAIL("dataset must be accessible on host or device memory");
  }
  return ret;
}

template <typename DataT, typename QuantT = uint8_t>
void _inverse_transform(cuvsResources_t res,
                cuvsProductQuantizer_t quantizer,
                DLManagedTensor* pq_codes_tensor,
                DLManagedTensor* out_tensor,
                DLManagedTensor* vq_labels_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto q = reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<DataT>*>(quantizer->addr);

  auto codes = pq_codes_tensor->dl_tensor;
  if (cuvs::core::is_dlpack_device_compatible(codes)) {
    using codes_mdspan_type     = raft::device_matrix_view<QuantT const, int64_t, raft::row_major>;
    using data_mdspan_type = raft::device_matrix_view<DataT, int64_t, raft::row_major>;
    using vq_labels_mdspan_type = raft::device_vector_view<const uint32_t, int64_t>;
    std::optional<vq_labels_mdspan_type> vq_labels = std::nullopt;
    if (vq_labels_tensor != NULL) {
      vq_labels = cuvs::core::from_dlpack<vq_labels_mdspan_type>(vq_labels_tensor);
    }
    cuvs::preprocessing::quantize::pq::inverse_transform(
      *res_ptr,
      *q,
      cuvs::core::from_dlpack<codes_mdspan_type>(pq_codes_tensor),
      cuvs::core::from_dlpack<data_mdspan_type>(out_tensor),
      vq_labels);

  } else {
    RAFT_FAIL("codes must be accessible on device memory");
  }
}
}  // namespace

extern "C" cuvsError_t cuvsProductQuantizerParamsCreate(cuvsProductQuantizerParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsProductQuantizerParams{
      .pq_bits = 8, .pq_dim = 0, .use_subspaces = true, .use_vq = false, .vq_n_centers = 0,
      .kmeans_n_iters = 25, .pq_kmeans_type = CUVS_KMEANS_TYPE_KMEANS_BALANCED,
      .max_train_points_per_pq_code = 256, .max_train_points_per_vq_cluster = 1024}; });
}

extern "C" cuvsError_t cuvsProductQuantizerParamsDestroy(cuvsProductQuantizerParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsProductQuantizerCreate(cuvsProductQuantizer_t* quantizer)
{
  return cuvs::core::translate_exceptions([=] { *quantizer = new cuvsProductQuantizer; });
}

extern "C" cuvsError_t cuvsProductQuantizerDestroy(cuvsProductQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] { delete quantizer; });
}

extern "C" cuvsError_t cuvsProductQuantizerBuild(cuvsResources_t res,
                                                 cuvsProductQuantizerParams_t params,
                                                 DLManagedTensor* dataset_tensor,
                                                 cuvsProductQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset     = dataset_tensor->dl_tensor;
    quantizer->dtype = dataset.dtype;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      quantizer->addr = reinterpret_cast<uintptr_t>(_build<float>(res, params, dataset_tensor));
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsProductQuantizerTransform(cuvsResources_t res,
                                                     cuvsProductQuantizer_t quantizer,
                                                     DLManagedTensor* dataset_tensor,
                                                     DLManagedTensor* codes_out_tensor,
                                                     DLManagedTensor* vq_labels_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      _transform<float>(res, quantizer, dataset_tensor, codes_out_tensor, vq_labels_tensor);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsProductQuantizerInverseTransform(cuvsResources_t res,
                                                            cuvsProductQuantizer_t quantizer,
                                                            DLManagedTensor* codes_tensor,
                                                            DLManagedTensor* out_tensor,
                                                            DLManagedTensor* vq_labels_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto out_dtype = out_tensor->dl_tensor.dtype;
    if (out_dtype.code == kDLFloat && out_dtype.bits == 32) {
      _inverse_transform<float>(res, quantizer, codes_tensor, out_tensor, vq_labels_tensor);
    } else {
      RAFT_FAIL("Unsupported out DLtensor dtype: %d and bits: %d",
                out_dtype.code,
                out_dtype.bits);
    }
  });
}
extern "C" cuvsError_t cuvsProductQuantizerGetPqBits(cuvsProductQuantizer_t quantizer,
                                                     uint32_t* pq_bits)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer != nullptr) {
      auto quant_addr = quantizer->addr;
      if (quantizer->dtype.code == kDLFloat && quantizer->dtype.bits == 32) {
        *pq_bits =
          (reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<float>*>(quant_addr))
            ->params_quantizer.pq_bits;
      } else {
        RAFT_FAIL("Unsupported quantizer dtype: %d and bits: %d",
                  quantizer->dtype.code,
                  quantizer->dtype.bits);
      }
    } else {
      RAFT_FAIL("quantizer is not initialized");
    }
  });
}

extern "C" cuvsError_t cuvsProductQuantizerGetPqDim(cuvsProductQuantizer_t quantizer,
                                                    uint32_t* pq_dim)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer != nullptr) {
      auto quant_addr = quantizer->addr;
      if (quantizer->dtype.code == kDLFloat && quantizer->dtype.bits == 32) {
        *pq_dim =
          (reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<float>*>(quant_addr))
            ->params_quantizer.pq_dim;
      } else {
        RAFT_FAIL("Unsupported quantizer dtype: %d and bits: %d",
                  quantizer->dtype.code,
                  quantizer->dtype.bits);
      }
    } else {
      RAFT_FAIL("quantizer is not initialized");
    }
  });
}

extern "C" cuvsError_t cuvsProductQuantizerGetPqCodebook(cuvsProductQuantizer_t quantizer,
                                                         DLManagedTensor* pq_codebook)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer != nullptr) {
      auto quant_addr = quantizer->addr;
      if (quantizer->dtype.code == kDLFloat && quantizer->dtype.bits == 32) {
        auto pq_mdspan =
          (reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<float>*>(quant_addr))
            ->codebooks.pq_code_book();
        cuvs::core::to_dlpack(pq_mdspan, pq_codebook);
      } else {
        RAFT_FAIL("Unsupported quantizer dtype: %d and bits: %d",
                  quantizer->dtype.code,
                  quantizer->dtype.bits);
      }
    } else {
      RAFT_FAIL("quantizer is not initialized");
    }
  });
}

extern "C" cuvsError_t cuvsProductQuantizerGetVqCodebook(cuvsProductQuantizer_t quantizer,
                                                         DLManagedTensor* vq_codebook)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer != nullptr) {
      auto quant_addr = quantizer->addr;
      if (quantizer->dtype.code == kDLFloat && quantizer->dtype.bits == 32) {
        auto vq_opt =
          (reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<float>*>(quant_addr))
            ->codebooks.vq_code_book();
        RAFT_EXPECTS(vq_opt.has_value(),
                     "quantizer has no VQ codebook (build with use_vq=true to enable)");
        cuvs::core::to_dlpack(vq_opt.value(), vq_codebook);
      } else {
        RAFT_FAIL("Unsupported quantizer dtype: %d and bits: %d",
                  quantizer->dtype.code,
                  quantizer->dtype.bits);
      }
    } else {
      RAFT_FAIL("quantizer is not initialized");
    }
  });
}

extern "C" cuvsError_t cuvsProductQuantizerGetEncodedDim(cuvsProductQuantizer_t quantizer,
                                                        uint32_t* encoded_dim)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer != nullptr) {
      auto quant_addr = quantizer->addr;
      if (quantizer->dtype.code == kDLFloat && quantizer->dtype.bits == 32) {
        *encoded_dim = cuvs::preprocessing::quantize::pq::get_quantized_dim(
          reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<float>*>(quant_addr)->params_quantizer);
      } else {
        RAFT_FAIL("Unsupported quantizer dtype: %d and bits: %d",
                  quantizer->dtype.code,
                  quantizer->dtype.bits);
      }
    } else {
      RAFT_FAIL("quantizer is not initialized");
    }
  });
}

extern "C" cuvsError_t cuvsProductQuantizerGetUseVq(
  cuvsProductQuantizer_t quantizer, bool* use_vq)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer != nullptr) {
      auto quant_addr = quantizer->addr;
      if (quantizer->dtype.code == kDLFloat && quantizer->dtype.bits == 32) {
        *use_vq = (reinterpret_cast<cuvs::preprocessing::quantize::pq::quantizer<float>*>(quant_addr))->params_quantizer.use_vq;
      } else {
        RAFT_FAIL("Unsupported quantizer dtype: %d and bits: %d",
                  quantizer->dtype.code,
                  quantizer->dtype.bits);
      }
    } else {
      RAFT_FAIL("quantizer is not initialized");
    }
  });
}
