/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/core/filter.h>

#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generic dataset layout kind for C API dataset handles.
 */
typedef enum {
  CUVS_DATASET_LAYOUT_STANDARD = 0,
  CUVS_DATASET_LAYOUT_PADDED   = 1,
  CUVS_DATASET_LAYOUT_PQ       = 2,
  CUVS_DATASET_LAYOUT_BBQ      = 3
} cuvsDatasetLayout_t;

/**
 * @brief Memory space holding a C API dataset handle's data.
 */
typedef enum {
  CUVS_DATASET_MEM_TYPE_HOST   = 0,
  CUVS_DATASET_MEM_TYPE_DEVICE = 1
} cuvsDatasetMemType_t;

/**
 * @brief Dataset handle representing owning storage or a non-owning view.
 *
 * `addr` points to C++ dataset storage or view metadata managed by the C API. `mem_type`
 * identifies the memory space, `layout` identifies the data layout, and `is_owning` indicates
 * whether the handle owns its backing data.
 */
typedef struct {
  uintptr_t addr;
  void (*destroy_addr)(void*);
  DLDataType dtype;
  cuvsDatasetMemType_t mem_type;
  cuvsDatasetLayout_t layout;
  bool is_owning;
} cuvsDataset;
typedef cuvsDataset* cuvsDataset_t;

struct cuvsCagraCompressionParams;
typedef struct cuvsCagraCompressionParams cuvsPqParams;
typedef cuvsPqParams* cuvsPqParams_t;

/**
 * @brief Compatibility name for PQ dataset parameters; planned for removal in the 27.02 ABI-breaking release.
 *
 * Use `cuvsPqParams_t` in new code.
 */
typedef struct cuvsCagraCompressionParams* cuvsCagraCompressionParams_t;

/** Allocate generic PQ dataset parameters with default values. */
CUVS_EXPORT cuvsError_t cuvsPqParamsCreate(cuvsPqParams_t* params);

/** De-allocate generic PQ dataset parameters. */
CUVS_EXPORT cuvsError_t cuvsPqParamsDestroy(cuvsPqParams_t params);

/**
 * @brief Create an empty owning dataset handle.
 *
 * The dataset storage, memory type, layout, and dtype are populated by the operation that fills
 * this handle.
 */
CUVS_EXPORT cuvsError_t cuvsDatasetCreate(cuvsDataset_t* dataset);

/**
 * @brief Create an owning padded dataset in the requested memory space.
 *
 * The source tensor may reside in host- or device-accessible memory. Its contents are copied into
 * newly allocated padded storage in `target_mem_type`.
 *
 * @param[in] res cuVS resources
 * @param[in] dataset source tensor
 * @param[in] target_mem_type memory space in which to allocate the padded dataset
 * @param[out] padded_dataset newly allocated owning padded dataset
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakePadded(cuvsResources_t res,
                                              DLManagedTensor* dataset,
                                              cuvsDatasetMemType_t target_mem_type,
                                              cuvsDataset_t* padded_dataset);

/**
 * @brief Compress a dense dataset into a device PQ dataset.
 *
 * Only device output is currently supported.
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakePQ(cuvsResources_t res,
                                          cuvsPqParams_t params,
                                          cuvsDataset_t dataset,
                                          cuvsDatasetMemType_t target_mem_type,
                                          cuvsDataset_t* pq_dataset);

/**
 * @brief Create a non-owning padded dataset view from a host- or device-resident tensor.
 *
 * Memory residency is inferred from the tensor.
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakePaddedView(cuvsResources_t res,
                                                  DLManagedTensor* dataset,
                                                  cuvsDataset_t* padded_dataset);

/**
 * @brief Create a non-owning standard dataset view from a host- or device-resident tensor.
 *
 * Memory residency is inferred from the tensor.
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakeStandardView(cuvsResources_t res,
                                                    DLManagedTensor* dataset,
                                                    cuvsDataset_t* standard_dataset);

/** @brief Destroy a dataset handle created by a `cuvsDatasetMake*` function. */
CUVS_EXPORT cuvsError_t cuvsDatasetDestroy(cuvsDataset_t dataset);

/** @brief Get the memory residency of a dataset handle. */
CUVS_EXPORT cuvsError_t cuvsDatasetGetMemType(cuvsDataset_t dataset,
                                              cuvsDatasetMemType_t* mem_type);

/** @brief Get the layout of a dataset handle. */
CUVS_EXPORT cuvsError_t cuvsDatasetGetLayout(cuvsDataset_t dataset, cuvsDatasetLayout_t* layout);

/** @brief Get whether a dataset handle owns its backing storage. */
CUVS_EXPORT cuvsError_t cuvsDatasetGetIsOwning(cuvsDataset_t dataset, bool* is_owning);

/** @brief Get the element dtype of a dataset handle. */
CUVS_EXPORT cuvsError_t cuvsDatasetGetDtype(cuvsDataset_t dataset, DLDataType* dtype);

/**
 * @defgroup dataset_c_index_concat Dataset concatenation helpers
 * @{
 */

/**
 * @brief Compute per-dataset write offsets for a merged dataset buffer.
 *
 * `cuvsCagraMerge_v2`/`cuvsCagraMergeWithParams_v2` require the caller to have already
 * concatenated every input dataset (in `datasets` order, applying `filter` if any) into a single
 * buffer and to know each dataset's starting row within it. For `filter.type == NO_FILTER`, those
 * offsets are just the cumulative sizes of `datasets` and this function is not needed. For
 * `BITSET`, the number of surviving rows per dataset cannot be derived any other way, so call this
 * first.
 *
 * Use `cuvsCagraIndexGetCuvsDataset` to obtain a `cuvsDataset_t` handle from each CAGRA index
 * before calling this function.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] datasets Array of input cuvsDataset_t handles (one per index, in merge order)
 * @param[in] num_datasets Number of input datasets
 * @param[in] filter Filter that will be passed to merge. Only `NO_FILTER` and `BITSET` supported.
 * @param[out] offsets Caller-allocated array of `num_datasets + 1` int64_t. Entry `i` is the row
 *                     at which `datasets[i]`'s surviving rows must start in the merged buffer; the
 *                     last entry is the total row count of the merged buffer.
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraMergedDatasetOffsets(cuvsResources_t res,
                                                      cuvsDataset_t* datasets,
                                                      size_t num_datasets,
                                                      cuvsFilter filter,
                                                      int64_t* offsets);

/**
 * @brief Concatenate every input dataset (in `datasets` order) into a freshly allocated,
 * CAGRA-padded, owning device dataset, optionally retaining only the rows selected by `filter`.
 *
 * This is an optional convenience function for building `cuvsCagraMerge_v2`'s `merged_dataset`
 * argument. Callers that already have their own concatenated buffer are not required to use it.
 * Call `cuvsCagraMergedDatasetOffsets` with the same `filter` to get the matching `offsets`.
 *
 * Use `cuvsCagraIndexGetCuvsDataset` to obtain a `cuvsDataset_t` handle from each CAGRA index
 * before calling this function.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] datasets Array of input cuvsDataset_t handles to concatenate, in the order they will
 *                     be passed to `cuvsCagraMerge_v2`
 * @param[in] num_datasets Number of input datasets
 * @param[in] filter Row filter. Pass `{NO_FILTER, 0}` (or a zero-initialized `cuvsFilter`) to
 *                   concatenate all rows. Pass a `BITSET` filter to retain only the surviving rows.
 * @param[out] merged_dataset Newly allocated owning dataset handle containing the concatenated
 *                            (and optionally filtered) rows in `datasets` order. Caller must
 *                            destroy it with `cuvsDatasetDestroy` when done.
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraConcatenateDatasets(cuvsResources_t res,
                                                     cuvsDataset_t* datasets,
                                                     size_t num_datasets,
                                                     cuvsFilter filter,
                                                     cuvsDataset_t* merged_dataset);

/** @} */

#ifdef __cplusplus
}
#endif
