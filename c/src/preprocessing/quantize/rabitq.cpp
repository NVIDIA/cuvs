/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>
#include <optional>
#include <utility>

#include <cuvs/core/c_api.h>
#include <cuvs/preprocessing/quantize/rabitq.h>
#include <cuvs/preprocessing/quantize/rabitq.hpp>

#include "../../core/exceptions.hpp"
#include "../../core/interop.hpp"

namespace {

namespace rabitq = cuvs::preprocessing::quantize::rabitq;

/** Read-only `n_rows x dim` float input: `dataset`, `residuals`, `in`. */
using in_matrix_mdspan_type = raft::device_matrix_view<float const, int64_t, raft::row_major>;
/** Read-only `dim` float vector: `centroid`, `rotated_centroid`. */
using centroid_mdspan_type = raft::device_vector_view<float const, int64_t>;
/** Writable float matrix output: `factors` (`n_rows x 3`) and `rotate`'s `out`. */
using out_matrix_mdspan_type = raft::device_matrix_view<float, int64_t, raft::row_major>;
/** Writable `n_rows` float vector output: `delta`, `vl`. */
using out_vector_mdspan_type = raft::device_vector_view<float, int64_t>;

template <typename CodeT>
using codes_mdspan_type = raft::device_matrix_view<CodeT, int64_t, raft::row_major>;

/** Translate the C params struct into its C++ peer. */
rabitq::params _convert_params(cuvsRabitqQuantizerParams_t params)
{
  if (params == nullptr) { RAFT_FAIL("params is not allocated"); }

  auto out    = rabitq::params();
  out.ex_bits = params->ex_bits;
  switch (params->rotator) {
    case CUVS_RABITQ_ROTATOR_KIND_MATMUL: out.rotator = rabitq::rotator_kind::matmul; break;
    case CUVS_RABITQ_ROTATOR_KIND_FHT_KAC: out.rotator = rabitq::rotator_kind::fht_kac; break;
    default: RAFT_FAIL("Unsupported rotator kind: %d", static_cast<int>(params->rotator));
  }
  out.seed = params->seed;
  switch (params->delta_mode) {
    case CUVS_RABITQ_DELTA_KIND_RECONSTRUCTION:
      out.delta_mode = rabitq::delta_kind::reconstruction;
      break;
    case CUVS_RABITQ_DELTA_KIND_UNBIASED: out.delta_mode = rabitq::delta_kind::unbiased; break;
    case CUVS_RABITQ_DELTA_KIND_PLAIN: out.delta_mode = rabitq::delta_kind::plain; break;
    default: RAFT_FAIL("Unsupported delta mode: %d", static_cast<int>(params->delta_mode));
  }
  out.use_fast       = params->use_fast;
  out.coarse_samples = params->coarse_samples;
  out.fine_samples   = params->fine_samples;
  return out;
}

rabitq::rotator* _rotator_ptr(cuvsRabitqRotator_t rotator)
{
  if (rotator == nullptr || rotator->addr == 0) { RAFT_FAIL("rotator is not initialized"); }
  return reinterpret_cast<rabitq::rotator*>(rotator->addr);
}

rabitq::quantizer* _quantizer_ptr(cuvsRabitqQuantizer_t quantizer)
{
  if (quantizer == nullptr || quantizer->addr == 0) { RAFT_FAIL("quantizer is not initialized"); }
  return reinterpret_cast<rabitq::quantizer*>(quantizer->addr);
}

/** Every RaBitQ entry point is device-only; give a uniform diagnostic for host tensors. */
void _expect_device(DLManagedTensor* tensor, char const* name)
{
  if (tensor == nullptr) { RAFT_FAIL("%s must not be NULL", name); }
  if (!cuvs::core::is_dlpack_device_compatible(tensor->dl_tensor)) {
    RAFT_FAIL("%s must be accessible on device memory", name);
  }
}

void _expect_float32(DLManagedTensor* tensor, char const* name)
{
  _expect_device(tensor, name);
  auto dtype = tensor->dl_tensor.dtype;
  if (dtype.code != kDLFloat || dtype.bits != 32) {
    RAFT_FAIL("Unsupported %s DLtensor dtype: %d and bits: %d", name, dtype.code, dtype.bits);
  }
}

/** NULL maps to std::nullopt, which both transform() families treat as "no centroid". */
std::optional<centroid_mdspan_type> _optional_centroid(DLManagedTensor* centroid_tensor,
                                                       char const* name)
{
  if (centroid_tensor == nullptr) { return std::nullopt; }
  _expect_float32(centroid_tensor, name);
  return cuvs::core::from_dlpack<centroid_mdspan_type>(centroid_tensor);
}

template <typename CodeT>
void _transform(cuvsResources_t res,
                cuvsRabitqQuantizer_t quantizer,
                cuvsRabitqRotator_t rotator,
                DLManagedTensor* dataset_tensor,
                DLManagedTensor* centroid_tensor,
                DLManagedTensor* codes_tensor,
                DLManagedTensor* delta_tensor,
                DLManagedTensor* vl_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto* q      = _quantizer_ptr(quantizer);
  auto* r      = _rotator_ptr(rotator);

  _expect_float32(dataset_tensor, "dataset");
  _expect_float32(delta_tensor, "delta");
  _expect_float32(vl_tensor, "vl");
  _expect_device(codes_tensor, "codes");

  rabitq::transform(*res_ptr,
                    *q,
                    *r,
                    cuvs::core::from_dlpack<in_matrix_mdspan_type>(dataset_tensor),
                    _optional_centroid(centroid_tensor, "centroid"),
                    cuvs::core::from_dlpack<codes_mdspan_type<CodeT>>(codes_tensor),
                    cuvs::core::from_dlpack<out_vector_mdspan_type>(delta_tensor),
                    cuvs::core::from_dlpack<out_vector_mdspan_type>(vl_tensor));
}

template <typename CodeT>
void _transform_full(cuvsResources_t res,
                     cuvsRabitqQuantizer_t quantizer,
                     cuvsRabitqRotator_t rotator,
                     DLManagedTensor* dataset_tensor,
                     DLManagedTensor* centroid_tensor,
                     DLManagedTensor* codes_tensor,
                     DLManagedTensor* factors_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto* q      = _quantizer_ptr(quantizer);
  auto* r      = _rotator_ptr(rotator);

  _expect_float32(dataset_tensor, "dataset");
  _expect_float32(factors_tensor, "factors");
  _expect_device(codes_tensor, "codes");

  rabitq::transform(*res_ptr,
                    *q,
                    *r,
                    cuvs::core::from_dlpack<in_matrix_mdspan_type>(dataset_tensor),
                    _optional_centroid(centroid_tensor, "centroid"),
                    cuvs::core::from_dlpack<codes_mdspan_type<CodeT>>(codes_tensor),
                    cuvs::core::from_dlpack<out_matrix_mdspan_type>(factors_tensor));
}

template <typename CodeT>
void _transform_residuals(cuvsResources_t res,
                          cuvsRabitqQuantizer_t quantizer,
                          DLManagedTensor* residuals_tensor,
                          DLManagedTensor* codes_tensor,
                          DLManagedTensor* delta_tensor,
                          DLManagedTensor* vl_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto* q      = _quantizer_ptr(quantizer);

  _expect_float32(residuals_tensor, "residuals");
  _expect_float32(delta_tensor, "delta");
  _expect_float32(vl_tensor, "vl");
  _expect_device(codes_tensor, "codes");

  rabitq::transform_residuals(*res_ptr,
                              *q,
                              cuvs::core::from_dlpack<in_matrix_mdspan_type>(residuals_tensor),
                              cuvs::core::from_dlpack<codes_mdspan_type<CodeT>>(codes_tensor),
                              cuvs::core::from_dlpack<out_vector_mdspan_type>(delta_tensor),
                              cuvs::core::from_dlpack<out_vector_mdspan_type>(vl_tensor));
}

template <typename CodeT>
void _transform_residuals_full(cuvsResources_t res,
                               cuvsRabitqQuantizer_t quantizer,
                               DLManagedTensor* residuals_tensor,
                               DLManagedTensor* rotated_centroid_tensor,
                               DLManagedTensor* codes_tensor,
                               DLManagedTensor* factors_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto* q      = _quantizer_ptr(quantizer);

  _expect_float32(residuals_tensor, "residuals");
  _expect_float32(factors_tensor, "factors");
  _expect_device(codes_tensor, "codes");

  rabitq::transform_residuals(
    *res_ptr,
    *q,
    cuvs::core::from_dlpack<in_matrix_mdspan_type>(residuals_tensor),
    _optional_centroid(rotated_centroid_tensor, "rotated_centroid"),
    cuvs::core::from_dlpack<codes_mdspan_type<CodeT>>(codes_tensor),
    cuvs::core::from_dlpack<out_matrix_mdspan_type>(factors_tensor));
}

}  // namespace

extern "C" cuvsError_t cuvsRabitqQuantizerParamsCreate(cuvsRabitqQuantizerParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsRabitqQuantizerParams{.ex_bits        = 3,
                                            .rotator        = CUVS_RABITQ_ROTATOR_KIND_FHT_KAC,
                                            .seed           = 12345ULL,
                                            .delta_mode     = CUVS_RABITQ_DELTA_KIND_RECONSTRUCTION,
                                            .use_fast       = true,
                                            .coarse_samples = 64,
                                            .fine_samples   = 64};
  });
}

extern "C" cuvsError_t cuvsRabitqQuantizerParamsDestroy(cuvsRabitqQuantizerParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsRabitqRotatorCreate(cuvsRabitqRotator_t* rotator)
{
  return cuvs::core::translate_exceptions([=] { *rotator = new cuvsRabitqRotator{}; });
}

extern "C" cuvsError_t cuvsRabitqRotatorDestroy(cuvsRabitqRotator_t rotator)
{
  return cuvs::core::translate_exceptions([=] {
    if (rotator == nullptr) { return; }
    delete reinterpret_cast<rabitq::rotator*>(rotator->addr);
    delete rotator;
  });
}

extern "C" cuvsError_t cuvsRabitqQuantizerCreate(cuvsRabitqQuantizer_t* quantizer)
{
  return cuvs::core::translate_exceptions([=] { *quantizer = new cuvsRabitqQuantizer{}; });
}

extern "C" cuvsError_t cuvsRabitqQuantizerDestroy(cuvsRabitqQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer == nullptr) { return; }
    delete reinterpret_cast<rabitq::quantizer*>(quantizer->addr);
    delete quantizer;
  });
}

extern "C" cuvsError_t cuvsRabitqRotatorMake(cuvsResources_t res,
                                             cuvsRabitqQuantizerParams_t params,
                                             int64_t dim,
                                             cuvsRabitqRotator_t rotator)
{
  return cuvs::core::translate_exceptions([=] {
    if (rotator == nullptr) { RAFT_FAIL("rotator is not allocated"); }
    auto res_ptr          = reinterpret_cast<raft::resources*>(res);
    auto quantizer_params = _convert_params(params);
    auto* ret = new rabitq::rotator(rabitq::make_rotator(*res_ptr, quantizer_params, dim));
    delete reinterpret_cast<rabitq::rotator*>(rotator->addr);
    rotator->addr = reinterpret_cast<uintptr_t>(ret);
  });
}

extern "C" cuvsError_t cuvsRabitqQuantizerMake(cuvsResources_t res,
                                               cuvsRabitqQuantizerParams_t params,
                                               int64_t dim,
                                               cuvsRabitqQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    if (quantizer == nullptr) { RAFT_FAIL("quantizer is not allocated"); }
    auto res_ptr          = reinterpret_cast<raft::resources*>(res);
    auto quantizer_params = _convert_params(params);
    auto* ret = new rabitq::quantizer(rabitq::make_quantizer(*res_ptr, quantizer_params, dim));
    delete reinterpret_cast<rabitq::quantizer*>(quantizer->addr);
    quantizer->addr = reinterpret_cast<uintptr_t>(ret);
  });
}

extern "C" cuvsError_t cuvsRabitqPipelineMake(cuvsResources_t res,
                                              cuvsRabitqQuantizerParams_t params,
                                              int64_t dim,
                                              cuvsRabitqRotator_t rotator,
                                              cuvsRabitqQuantizer_t quantizer)
{
  return cuvs::core::translate_exceptions([=] {
    if (rotator == nullptr) { RAFT_FAIL("rotator is not allocated"); }
    if (quantizer == nullptr) { RAFT_FAIL("quantizer is not allocated"); }
    auto res_ptr          = reinterpret_cast<raft::resources*>(res);
    auto quantizer_params = _convert_params(params);
    auto built            = rabitq::make_pipeline(*res_ptr, quantizer_params, dim);

    auto* new_rotator   = new rabitq::rotator(std::move(built.rotator));
    auto* new_quantizer = new rabitq::quantizer(built.quantizer);

    delete reinterpret_cast<rabitq::rotator*>(rotator->addr);
    delete reinterpret_cast<rabitq::quantizer*>(quantizer->addr);
    rotator->addr   = reinterpret_cast<uintptr_t>(new_rotator);
    quantizer->addr = reinterpret_cast<uintptr_t>(new_quantizer);
  });
}

extern "C" cuvsError_t cuvsRabitqRotate(cuvsResources_t res,
                                        cuvsRabitqRotator_t rotator,
                                        DLManagedTensor* in_tensor,
                                        DLManagedTensor* out_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    auto* r      = _rotator_ptr(rotator);

    _expect_float32(in_tensor, "in");
    _expect_float32(out_tensor, "out");

    rabitq::rotate(*res_ptr,
                   *r,
                   cuvs::core::from_dlpack<in_matrix_mdspan_type>(in_tensor),
                   cuvs::core::from_dlpack<out_matrix_mdspan_type>(out_tensor));
  });
}

extern "C" cuvsError_t cuvsRabitqTransform(cuvsResources_t res,
                                           cuvsRabitqQuantizer_t quantizer,
                                           cuvsRabitqRotator_t rotator,
                                           DLManagedTensor* dataset_tensor,
                                           DLManagedTensor* centroid_tensor,
                                           DLManagedTensor* codes_tensor,
                                           DLManagedTensor* delta_tensor,
                                           DLManagedTensor* vl_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    if (codes_tensor == nullptr) { RAFT_FAIL("codes must not be NULL"); }
    auto codes = codes_tensor->dl_tensor;
    if (codes.dtype.code == kDLUInt && codes.dtype.bits == 8) {
      _transform<uint8_t>(res,
                          quantizer,
                          rotator,
                          dataset_tensor,
                          centroid_tensor,
                          codes_tensor,
                          delta_tensor,
                          vl_tensor);
    } else if (codes.dtype.code == kDLUInt && codes.dtype.bits == 16) {
      _transform<uint16_t>(res,
                           quantizer,
                           rotator,
                           dataset_tensor,
                           centroid_tensor,
                           codes_tensor,
                           delta_tensor,
                           vl_tensor);
    } else {
      RAFT_FAIL(
        "Unsupported codes DLtensor dtype: %d and bits: %d", codes.dtype.code, codes.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsRabitqTransformFull(cuvsResources_t res,
                                               cuvsRabitqQuantizer_t quantizer,
                                               cuvsRabitqRotator_t rotator,
                                               DLManagedTensor* dataset_tensor,
                                               DLManagedTensor* centroid_tensor,
                                               DLManagedTensor* codes_tensor,
                                               DLManagedTensor* factors_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    if (codes_tensor == nullptr) { RAFT_FAIL("codes must not be NULL"); }
    auto codes = codes_tensor->dl_tensor;
    if (codes.dtype.code == kDLUInt && codes.dtype.bits == 8) {
      _transform_full<uint8_t>(
        res, quantizer, rotator, dataset_tensor, centroid_tensor, codes_tensor, factors_tensor);
    } else if (codes.dtype.code == kDLUInt && codes.dtype.bits == 16) {
      _transform_full<uint16_t>(
        res, quantizer, rotator, dataset_tensor, centroid_tensor, codes_tensor, factors_tensor);
    } else {
      RAFT_FAIL(
        "Unsupported codes DLtensor dtype: %d and bits: %d", codes.dtype.code, codes.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsRabitqTransformResiduals(cuvsResources_t res,
                                                    cuvsRabitqQuantizer_t quantizer,
                                                    DLManagedTensor* residuals_tensor,
                                                    DLManagedTensor* codes_tensor,
                                                    DLManagedTensor* delta_tensor,
                                                    DLManagedTensor* vl_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    if (codes_tensor == nullptr) { RAFT_FAIL("codes must not be NULL"); }
    auto codes = codes_tensor->dl_tensor;
    if (codes.dtype.code == kDLUInt && codes.dtype.bits == 8) {
      _transform_residuals<uint8_t>(
        res, quantizer, residuals_tensor, codes_tensor, delta_tensor, vl_tensor);
    } else if (codes.dtype.code == kDLUInt && codes.dtype.bits == 16) {
      _transform_residuals<uint16_t>(
        res, quantizer, residuals_tensor, codes_tensor, delta_tensor, vl_tensor);
    } else {
      RAFT_FAIL(
        "Unsupported codes DLtensor dtype: %d and bits: %d", codes.dtype.code, codes.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsRabitqTransformResidualsFull(cuvsResources_t res,
                                                        cuvsRabitqQuantizer_t quantizer,
                                                        DLManagedTensor* residuals_tensor,
                                                        DLManagedTensor* rotated_centroid_tensor,
                                                        DLManagedTensor* codes_tensor,
                                                        DLManagedTensor* factors_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    if (codes_tensor == nullptr) { RAFT_FAIL("codes must not be NULL"); }
    auto codes = codes_tensor->dl_tensor;
    if (codes.dtype.code == kDLUInt && codes.dtype.bits == 8) {
      _transform_residuals_full<uint8_t>(res,
                                         quantizer,
                                         residuals_tensor,
                                         rotated_centroid_tensor,
                                         codes_tensor,
                                         factors_tensor);
    } else if (codes.dtype.code == kDLUInt && codes.dtype.bits == 16) {
      _transform_residuals_full<uint16_t>(res,
                                          quantizer,
                                          residuals_tensor,
                                          rotated_centroid_tensor,
                                          codes_tensor,
                                          factors_tensor);
    } else {
      RAFT_FAIL(
        "Unsupported codes DLtensor dtype: %d and bits: %d", codes.dtype.code, codes.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsRabitqRotatorGetKind(cuvsRabitqRotator_t rotator,
                                                enum cuvsRabitqRotatorKind* kind)
{
  return cuvs::core::translate_exceptions([=] {
    auto* r = _rotator_ptr(rotator);
    switch (r->kind) {
      case rabitq::rotator_kind::matmul: *kind = CUVS_RABITQ_ROTATOR_KIND_MATMUL; break;
      case rabitq::rotator_kind::fht_kac: *kind = CUVS_RABITQ_ROTATOR_KIND_FHT_KAC; break;
      default: RAFT_FAIL("Unsupported rotator kind: %d", static_cast<int>(r->kind));
    }
  });
}

extern "C" cuvsError_t cuvsRabitqRotatorGetPaddedDim(cuvsRabitqRotator_t rotator,
                                                     int64_t* padded_dim)
{
  return cuvs::core::translate_exceptions([=] { *padded_dim = _rotator_ptr(rotator)->padded_dim; });
}

extern "C" cuvsError_t cuvsRabitqRotatorGetState(cuvsRabitqRotator_t rotator,
                                                 DLManagedTensor* state)
{
  return cuvs::core::translate_exceptions([=] {
    auto* r = _rotator_ptr(rotator);
    switch (r->kind) {
      case rabitq::rotator_kind::matmul:
        cuvs::core::to_dlpack(r->rotation_matrix.view(), state);
        break;
      case rabitq::rotator_kind::fht_kac: cuvs::core::to_dlpack(r->flip_bits.view(), state); break;
      default: RAFT_FAIL("Unsupported rotator kind: %d", static_cast<int>(r->kind));
    }
  });
}

extern "C" cuvsError_t cuvsRabitqQuantizerGetDim(cuvsRabitqQuantizer_t quantizer, int64_t* dim)
{
  return cuvs::core::translate_exceptions([=] { *dim = _quantizer_ptr(quantizer)->dim; });
}

extern "C" cuvsError_t cuvsRabitqQuantizerGetPaddedDim(cuvsRabitqQuantizer_t quantizer,
                                                       int64_t* padded_dim)
{
  return cuvs::core::translate_exceptions(
    [=] { *padded_dim = _quantizer_ptr(quantizer)->padded_dim; });
}

extern "C" cuvsError_t cuvsRabitqQuantizerGetConstScalingFactor(cuvsRabitqQuantizer_t quantizer,
                                                                float* const_scaling_factor)
{
  return cuvs::core::translate_exceptions(
    [=] { *const_scaling_factor = _quantizer_ptr(quantizer)->const_scaling_factor; });
}
