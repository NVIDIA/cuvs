/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dlpack/dlpack.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generic dataset layout kind for C API dataset handles.
 */
typedef enum {
  CUVS_DATASET_LAYOUT_STANDARD = 0,
  CUVS_DATASET_LAYOUT_PADDED   = 1
} cuvsDatasetLayout_t;

/**
 * @brief Generic opaque dataset handle used by C APIs.
 *
 * `addr` points to implementation-owned storage.
 * `dtype` carries dataset element type in DLPack format.
 * `layout` differentiates standard vs padded layouts.
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
  cuvsDatasetLayout_t layout;
} cuvsDataset;
typedef cuvsDataset* cuvsDataset_t;

/**
 * @brief Layout-specific aliases that sort well lexicographically in docs.
 */
typedef cuvsDataset cuvsDatasetPadded;
typedef cuvsDataset* cuvsDatasetPadded_t;
typedef cuvsDataset cuvsDatasetStandard;
typedef cuvsDataset* cuvsDatasetStandard_t;

/**
 * @brief Generic storage kind for operation-specific dataset storage.
 */
typedef enum {
  CUVS_DATASET_STORAGE_KIND_EXTENDED = 0,
  CUVS_DATASET_STORAGE_KIND_MERGED   = 1
} cuvsDatasetStorageKind_t;

/**
 * @brief Generic opaque storage handle for dataset-backed operation outputs.
 *
 * Used by operations like CAGRA extend/merge where storage shape is identical
 * but semantics differ by operation kind.
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
  cuvsDatasetStorageKind_t kind;
} cuvsDatasetStorage;
typedef cuvsDatasetStorage* cuvsDatasetStorage_t;

#ifdef __cplusplus
}
#endif
