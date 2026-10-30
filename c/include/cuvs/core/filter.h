/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

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
 * @brief Struct to hold a filter address and its type.
 *
 * The meaning of `addr` depends on `type`:
 *   - `NO_FILTER`: `addr` is unused (set to 0).
 *   - `BITSET`:    `addr` is the address of a `DLManagedTensor*` holding the bitset on device.
 *                  A set bit means the corresponding row survives the filter.
 *
 * The filter object pointed to by `addr` is owned by the caller; the library performs no caching
 * of the underlying bitset across search calls. Allocating and populating the device bitset may
 * be more expensive than a single filtered search, so callers that issue repeated searches against
 * the same filter (e.g. many queries over one index) should build the bitset once and reuse the
 * same cuvsFilter across those calls rather than rebuild it per search. Reusing the bitset is
 * essential for realizing the full throughput of filtered search.
 */
typedef struct {
  uintptr_t addr;
  enum cuvsFilterType type;
} cuvsFilter;

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
