/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.h>
#include <cuvs/core/filter.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
