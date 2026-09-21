/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>

#include <dlpack/dlpack.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup preprocessing_c_bbq C API for Better Binary Quantization datasets
 * @{
 */

/** Storage layout of the quantized component codes in each dataset row. */
typedef enum {
  CUVS_BBQ_CODE_LAYOUT_PACKED_1B = 0,
  CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B,
  CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B,
  CUVS_BBQ_CODE_LAYOUT_PACKED_4B,
  CUVS_BBQ_CODE_LAYOUT_PACKED_7B,
  CUVS_BBQ_CODE_LAYOUT_PACKED_8B
} cuvsBbqCodeLayout_t;

typedef struct cuvsBbqQuantizer {
  uintptr_t addr;
  void (*destroy_addr)(void*);
  DLDataType dtype;
  bool is_owning;
} cuvsBbqQuantizer;
typedef cuvsBbqQuantizer* cuvsBbqQuantizer_t;

/**
 * @brief Create a BBQ quantizer view from caller-owned device tensors.
 *
 * Tensors are not copied and must remain valid while a derived dataset is in use.
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerCreateView(
  DLManagedTensor* codes,
  DLManagedTensor* lower_intervals,
  DLManagedTensor* upper_intervals,
  DLManagedTensor* additional_corrections,
  DLManagedTensor* quantized_component_sums,
  DLManagedTensor* centroid,
  DLManagedTensor* dequant_delta,
  DLManagedTensor* dequant_sum_delta,
  DLManagedTensor* row_norm,
  cuvsBbqCodeLayout_t layout,
  cuvsDistanceType metric,
  float centroid_norm_sq,
  cuvsBbqQuantizer_t* quantizer);

/** Destroy a BBQ quantizer without destroying its caller-owned tensors. */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerDestroy(cuvsBbqQuantizer_t quantizer);

/** @} */

#ifdef __cplusplus
}
#endif
