/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup mg_c_common_types Multi-GPU common types and enums
 * @{
 */

/**
 * @brief Distribution mode for multi-GPU indexes
 */
typedef enum {
  /** Index is replicated on each device, favors throughput */
  CUVS_NEIGHBORS_MG_REPLICATED = 0,
  /** Index is split on several devices, favors scaling */
  CUVS_NEIGHBORS_MG_SHARDED = 1
} cuvsMultiGpuDistributionMode;

/**
 * @brief Search mode when using a replicated index
 */
typedef enum {
  /** Search queries are split to maintain equal load on GPUs */
  CUVS_NEIGHBORS_MG_LOAD_BALANCER = 0,
  /** Each search query is processed by a single GPU in a round-robin fashion */
  CUVS_NEIGHBORS_MG_ROUND_ROBIN = 1
} cuvsMultiGpuReplicatedSearchMode;

/**
 * @brief Merge mode when using a sharded index
 */
typedef enum {
  /** Search batches are merged on the root rank */
  CUVS_NEIGHBORS_MG_MERGE_ON_ROOT_RANK = 0,
  /** Search batches are merged in a tree reduction fashion */
  CUVS_NEIGHBORS_MG_TREE_MERGE = 1
} cuvsMultiGpuShardedMergeMode;

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
