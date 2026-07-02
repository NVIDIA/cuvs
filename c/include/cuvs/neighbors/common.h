/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dlpack/dlpack.h>
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
  BITMAP = 2,
  /* Filter multiple index partitions with a single concatenated bitset plus per-partition offsets */
  MULTI_PARTITION_BITSET = 3
};

/**
 * @brief Filter parameters for multi-partition search.
 *
 * Holds a single device bitset that is the concatenation of per-partition bitsets,
 * together with a device array of per-partition bit offsets.  Pass a pointer to
 * this struct (cast to uintptr_t) in cuvsFilter::addr with
 * cuvsFilter::type == MULTI_PARTITION_BITSET.
 */
typedef struct {
  /** Device tensor (uint32, flat) of packed bitset words for all partitions concatenated. */
  DLManagedTensor* combined_bitset;
  /** Total number of logical bits in combined_bitset. */
  int64_t total_bitset_bits;
  /** Device tensor (int64, [num_partitions]) of per-partition bit offsets into combined_bitset. */
  DLManagedTensor* partition_offsets;
} cuvsMultiPartitionBitsetFilter;

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
