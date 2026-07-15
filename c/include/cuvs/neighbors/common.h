/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup filters Filters APIs
 * @brief APIs related to filter functionality.
 * @{
 */

/**
 * @brief Enum to denote filter type.
 */
enum cuvsFilterType {
  /* No filter */
  NO_FILTER = 0,
  /* Filter an index with a bitset */
  BITSET = 1,
  /* Filter an index with a bitmap */
  BITMAP = 2
};

/**
 * @brief Struct to hold address of cuvs::neighbors::prefilter and its type
 *
 */
typedef struct {
  uintptr_t addr;
  enum cuvsFilterType type;
} cuvsFilter;

/**
 * @}
 */

/**
 * @defgroup index_merge Index Merge
 * @brief Common definitions related to index merging.
 * @{
 */

/**
 * @brief Strategy for merging indices.
 */
typedef enum {
  MERGE_STRATEGY_PHYSICAL = 0,  ///< Merge indices physically
  MERGE_STRATEGY_LOGICAL  = 1   ///< Merge indices logically
} cuvsMergeStrategy;

/**
 * @}
 */
#ifdef __cplusplus
}
#endif
