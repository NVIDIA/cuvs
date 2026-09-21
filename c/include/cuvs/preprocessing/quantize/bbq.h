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

/**
 * Storage layout of BBQ/OSQ quantized component codes in each dataset row.
 */
typedef enum {
  CUVS_BBQ_CODE_LAYOUT_PACKED_1B = 0,  ///< One bit per dimension packed into bytes.
  CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B,  ///< Two bits per dimension stored as two bitplanes.
  CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B,  ///< Four bits per dimension stored as four bitplanes.
  CUVS_BBQ_CODE_LAYOUT_PACKED_4B,      ///< Four bits per dimension with two values per byte.
  CUVS_BBQ_CODE_LAYOUT_PACKED_7B,      ///< Seven-bit values stored one per byte.
  CUVS_BBQ_CODE_LAYOUT_PACKED_8B       ///< Eight-bit values stored one per byte.
} cuvsBbqCodeLayout_t;

/**
 * @brief Better Binary Quantization
 * ([BBQ](https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch))
 * is a vector-quantization approach used in Elasticsearch and Apache Lucene. It builds on ideas
 * introduced in RaBitQ([Gao and Long](https://arxiv.org/pdf/2405.12497, [Gao et
 * al.](https://arxiv.org/pdf/2409.09913)): residual binary codes around a centroid, corrective
 * factors, and efficient bitwise comparison of codes at different bit widths. Lucene implements
 * this as optimized scalar quantization (OSQ) with packed and bit-plane layouts; Elasticsearch
 * exposes it as BBQ.
 *
 * BBQ in cuVS designed to be compatible with the Lucene/Elasticsearch dataset: a single shared
 * centroid, no random rotation, and OSQ codes.
 *
 * RaBitQ and BBQ in cuVS both compress centroid-relative vectors to low-bit codes and retain
 * additional per-vector information so search is better than naïve sign-bit comparison. They differ
 * in transformation and scale representation. RaBitQ commonly separates residual magnitude from
 * direction, then applies a random orthogonal rotation before binary coding; BBQ uses per-vector
 * scalar intervals to interpret the compressed residual codes.
 */
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
 *
 * @param[in] codes uint8 device matrix containing encoded rows
 * @param[in] lower_intervals float32 device vector with one lower interval per row
 * @param[in] upper_intervals float32 device vector with one upper interval per row
 * @param[in] additional_corrections float32 device vector with one correction per row
 * @param[in] quantized_component_sums int32 device vector with one component sum per row
 * @param[in] centroid device vector containing the dataset centroid
 * @param[in] dequant_delta float32 device vector with one dequantization delta per row
 * @param[in] dequant_sum_delta float32 device vector with one delta-times-sum value per row
 * @param[in] row_norm float32 device vector with one original-space squared norm per row
 * @param[in] layout encoded code layout
 * @param[in] metric distance metric associated with the encoded dataset
 * @param[in] centroid_norm_sq squared norm of the centroid
 * @param[out] quantizer newly allocated non-owning quantizer handle
 * @return cuvsError_t
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

/**
 * @brief Destroy a BBQ quantizer without destroying its caller-owned tensors.
 *
 * @param[in] quantizer quantizer handle to destroy
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerDestroy(cuvsBbqQuantizer_t quantizer);

/** @} */

#ifdef __cplusplus
}
#endif
