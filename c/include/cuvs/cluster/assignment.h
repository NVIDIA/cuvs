/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup assignment_c Nearest-centroid assignment backends
 * @{
 */

/** @brief The backend that assigns vectors to their nearest centroid. */
enum cuvsAssignmentKind {
  /**
   * The IVF-based (approximate) assignment, an IVF-Flat style search over the
   * centroids with k = 1: the centroids are grouped into
   * `n_lists` lists; each vector probes its `n_probes` best lists and takes the
   * best centroid among them. Probing every list is exhaustive. The data
   * dimension must be divisible by 4.
   */
  CUVS_ASSIGNMENT_IVF = 0,
};

/** @brief Parameters of the IVF assignment (see `CUVS_ASSIGNMENT_IVF`).
 */
struct cuvsIvfAssignmentParams {
  /** Number of lists; 0 means round(sqrt(n_centroids)). Only used when building
   * an index. */
  uint32_t n_lists;
  /** Balanced k-means iterations used to group the centroids. Only used when
   * building an index. */
  uint32_t kmeans_n_iters;
  /** Lists probed per vector (clamped to n_lists); 0 selects 16. */
  uint32_t n_probes;
};

/**
 * @brief Index of an assignment backend, built once from a fixed set of
 * centroids and reused across calls. Opaque; `kind` records which backend built
 * it.
 */
typedef struct {
  uintptr_t addr;
  enum cuvsAssignmentKind kind;
} cuvsAssignmentIndex;

typedef cuvsAssignmentIndex *cuvsAssignmentIndex_t;

/** @brief Allocate an assignment index (empty until built). */
CUVS_EXPORT cuvsError_t cuvsAssignmentIndexCreate(cuvsAssignmentIndex_t *index);

/** @brief De-allocate an assignment index. */
CUVS_EXPORT cuvsError_t cuvsAssignmentIndexDestroy(cuvsAssignmentIndex_t index);

/**
 * @brief Selects the assignment backend and holds its parameters.
 *
 * Allocate with `cuvsAssignmentParamsCreate`, which selects the IVF backend
 * with default parameters. Operations that take an optional
 * `cuvsAssignmentParams_t` use their default (exhaustive) assignment when it is
 * NULL. Only the member of `kind` is read.
 */
struct cuvsAssignmentParams {
  enum cuvsAssignmentKind kind;
  /** Parameters of `CUVS_ASSIGNMENT_IVF`. */
  struct cuvsIvfAssignmentParams ivf;
  /**
   * Optional prebuilt index, not owned (must outlive the call); NULL builds one
   * per call and discards it. When set, it must have been built for `kind`, the
   * index-building parameters (e.g. `ivf.n_lists`) are ignored, and its
   * centroid count, dimension and metric must match the call.
   */
  cuvsAssignmentIndex_t index;
};

typedef struct cuvsAssignmentParams *cuvsAssignmentParams_t;

/** @brief Allocate assignment params (IVF backend, default parameters, no
 * index). */
CUVS_EXPORT cuvsError_t
cuvsAssignmentParamsCreate(cuvsAssignmentParams_t *params);

/** @brief De-allocate assignment params. */
CUVS_EXPORT cuvsError_t
cuvsAssignmentParamsDestroy(cuvsAssignmentParams_t params);

/**
 * @brief Build an assignment index from centroids.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params backend and its index-building parameters; `params->index`
 * is ignored
 * @param[in] centroids DLManagedTensor* float32 device matrix [n_centroids,
 * dim]
 * @param[in] metric L2Expanded, L2SqrtExpanded, CosineExpanded or InnerProduct
 * @param[out] index index to build, allocated with `cuvsAssignmentIndexCreate`;
 * a previously built index is replaced
 */
CUVS_EXPORT cuvsError_t cuvsAssignmentIndexBuild(cuvsResources_t res,
                                                 cuvsAssignmentParams_t params,
                                                 DLManagedTensor *centroids,
                                                 cuvsDistanceType metric,
                                                 cuvsAssignmentIndex_t index);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
