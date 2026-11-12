/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/core/dataset.h>
#include <cuvs/distance/distance.h>

#include <dlpack/dlpack.h>
#include <stddef.h>
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
 * CUVS_BBQ_CODE_LAYOUT_PACKED_1B: Each dimension is quantized to a single bit and packed into bytes. Reflects
 * Lucene's OptimizedScalarQuantizer.packAsBinary.
 * CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B: Each dimension is quantized to 2 bits, stored as 2 bitplanes.
 * Reflects Lucene's OptimizedScalarQuantizer.transposeDibit. SIMT popc path only
 * (paired with a transposed_4b or packed_1b operand);
 * CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B: Each dimension is quantized to 4 bits, optimized for bitwise operations.
 * Reflects Lucene's OptimizedScalarQuantizer.transposeHalfByte. the first bit of
 * every dimension is in the first set dimensions bits, or (dimensions/8)
 * bytes. The second, third, and fourth bits are in the second, third, and
 * fourth set of dimensions bits, respectively. Format used for queries.
 * CUVS_BBQ_CODE_LAYOUT_PACKED_4B: Each dimension is quantized to 4 bits, two values are packed into each output
 * byte.
 * CUVS_BBQ_CODE_LAYOUT_PACKED_7B: Each dimension is quantized to 7 bits and treated as a signed value.
 * CUVS_BBQ_CODE_LAYOUT_PACKED_8B: Each dimension is quantized to 8 bits and treated as an unsigned value.
 */
typedef enum {
  CUVS_BBQ_CODE_LAYOUT_PACKED_1B = 0,
  CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_2B,
  CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B,
  CUVS_BBQ_CODE_LAYOUT_PACKED_4B,
  CUVS_BBQ_CODE_LAYOUT_PACKED_7B,
  CUVS_BBQ_CODE_LAYOUT_PACKED_8B
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
 * `centroid_norm_sq` is copied to device memory owned by the quantizer handle, and datasets and
 * indexes derived from the handle refer to that copy. The handle must therefore not be destroyed
 * while any dataset or index derived from it is in use.
 *
 * @code{.c}
 * cuvsBbqQuantizer_t quantizer;
 * cuvsBbqQuantizerCreateView(&codes, &lower, &upper, &corrections, &sums, &centroid, &delta,
 *                            &sum_delta, &row_norm, CUVS_BBQ_CODE_LAYOUT_PACKED_1B, InnerProduct,
 *                            centroid_norm_sq, &quantizer);
 * cuvsDataset_t dataset;
 * cuvsDatasetMakeBbqView(res, &quantizer, 1, &dataset);
 * cuvsCagraIndex_t index;
 * cuvsCagraIndexCreate(&index);
 * cuvsCagraBuild(res, index_params, dataset, index);  // builds a graph-only index
 * // Attach a device padded or PQ dataset before searching.
 * cuvsCagraUpdateDataset(res, search_dataset, index);
 * cuvsCagraSearch(res, search_params, index, &queries, &neighbors, &distances, filter);
 * cuvsCagraIndexDestroy(index);
 * cuvsDatasetDestroy(dataset);         // destroy derived datasets and indexes first,
 * cuvsBbqQuantizerDestroy(quantizer);  // then the quantizer, then the tensors
 * @endcode
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
 * Frees the quantizer's device copy of `centroid_norm_sq`. Datasets and indexes derived from the
 * quantizer must be destroyed first.
 *
 * @param[in] quantizer quantizer handle to destroy
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerDestroy(cuvsBbqQuantizer_t quantizer);

/**
 * @brief Create a non-owning device BBQ dataset view.
 *
 * Accepts one symmetric quantizer or two compatible asymmetric quantizers. Both asymmetric
 * quantizers must encode the same rows, in the same order, against the same centroid. The dataset,
 * and any index built from it, refers to the quantizers' data, so the quantizer handles must
 * outlive them.
 *
 * @param[in] res cuVS resources
 * @param[in] quantizers array containing one or two BBQ quantizer handles
 * @param[in] num_quantizers number of elements in `quantizers`
 * @param[out] dataset newly allocated non-owning BBQ dataset handle
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakeBbqView(cuvsResources_t res,
                                               cuvsBbqQuantizer_t* quantizers,
                                               size_t num_quantizers,
                                               cuvsDataset_t* dataset);

/**
 * @brief Parameters for encoding a dataset with `cuvsBbqQuantizerTransform`.
 */
struct cuvsBbqQuantizerParams {
  /** Code layout of the encoded rows on the document side of a distance. */
  cuvsBbqCodeLayout_t doc_layout;
  /**
   * Code layout of the encoded rows on the query side of a distance. When it differs from
   * `doc_layout`, the dataset is encoded a second time in this layout.
   */
  cuvsBbqCodeLayout_t query_layout;
  /** Selects the per-row additional correction: ||x - c||^2 for L2, x.c otherwise. */
  cuvsDistanceType metric;
  /**
   * Relative interval weight of the quantization error against the norm-preserving term of the
   * loss.
   */
  float lambda;
  /** Coordinate descent iterations per row. */
  int iters;
};

typedef struct cuvsBbqQuantizerParams* cuvsBbqQuantizerParams_t;

/**
 * @brief Allocate BBQ quantizer params, and populate with default values
 *
 * Defaults: `CUVS_BBQ_CODE_LAYOUT_PACKED_1B` documents, `CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B`
 * queries, `L2Expanded`, `lambda` 0.1 and 5 `iters`.
 *
 * @param[out] params cuvsBbqQuantizerParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerParamsCreate(cuvsBbqQuantizerParams_t* params);

/**
 * @brief De-allocate BBQ quantizer params
 *
 * @param[in] params cuvsBbqQuantizerParams_t to de-allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerParamsDestroy(cuvsBbqQuantizerParams_t params);

/**
 * @brief Build a BBQ quantizer from a dataset: computes the centroid shared by every row.
 *
 * The returned handle owns the centroid and its squared norm, and only serves to encode datasets
 * with `cuvsBbqQuantizerTransform`: it holds no encoded rows, so `cuvsDatasetMakeBbqView` rejects
 * it. Destroy it with `cuvsBbqQuantizerDestroy`.
 *
 * @code{.c}
 * cuvsBbqQuantizerParams_t params;
 * cuvsBbqQuantizerParamsCreate(&params);
 * cuvsBbqQuantizer_t quantizer;
 * cuvsBbqQuantizerBuild(res, &dataset, &quantizer);
 * cuvsDataset_t bbq_dataset;
 * cuvsBbqQuantizerTransform(res, params, quantizer, &dataset, &bbq_dataset);
 * cuvsBbqQuantizerDestroy(quantizer);  // `bbq_dataset` holds its own copy of the centroid
 * cuvsCagraBuild(res, index_params, bbq_dataset, index);
 * @endcode
 *
 * @param[in] res cuVS resources
 * @param[in] dataset float32 row-major device matrix
 * @param[out] quantizer newly allocated owning quantizer handle
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerBuild(cuvsResources_t res,
                                              DLManagedTensor* dataset,
                                              cuvsBbqQuantizer_t* quantizer);

/**
 * @brief Encode a dataset against the centroid of a BBQ quantizer into an owning BBQ dataset.
 *
 * The dataset holds one storage in `params->doc_layout`, plus one in `params->query_layout` when it
 * differs, and its own copy of the centroid, so it does not depend on `quantizer` or `dataset` once
 * returned. `quantizer` is either built by `cuvsBbqQuantizerBuild`, or a float32 view from
 * `cuvsBbqQuantizerCreateView`, whose centroid is then reused. Destroy the result with
 * `cuvsDatasetDestroy`.
 *
 * @param[in] res cuVS resources
 * @param[in] params encoding parameters
 * @param[in] quantizer BBQ quantizer whose centroid the rows are encoded against
 * @param[in] in_dataset float32 row-major device matrix with the quantizer's dimension
 * @param[out] bbq_dataset newly allocated owning `CUVS_DATASET_LAYOUT_BBQ` device dataset handle
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsBbqQuantizerTransform(cuvsResources_t res,
                                                  cuvsBbqQuantizerParams_t params,
                                                  cuvsBbqQuantizer_t quantizer,
                                                  DLManagedTensor* in_dataset,
                                                  cuvsDataset_t* bbq_dataset);

/** @} */

#ifdef __cplusplus
}
#endif
