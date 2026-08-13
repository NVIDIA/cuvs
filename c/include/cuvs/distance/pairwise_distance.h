/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <dlpack/dlpack.h>

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup pairwise_distance_c C pairwise distance
 * @{
 */

/**
 * @brief Compute pairwise distances for two matrices
 *
 *
 * Usage example:
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/distance/pairwise_distance.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor x;
 * DLManagedTensor y;
 * DLManagedTensor dist;
 *
 * cuvsPairwiseDistance(res, &x, &y, &dist, L2SqrtUnexpanded, 2.0);
 * @endcode
 *
 * @param[in] res cuvs resources object for managing expensive resources
 * @param[in] x first set of points (size n*k)
 * @param[in] y second set of points (size m*k)
 * @param[out] dist output distance matrix (size n*m)
 * @param[in] metric distance to evaluate
 * @param[in] metric_arg metric argument (used for Minkowski distance)
 */
CUVS_EXPORT cuvsError_t cuvsPairwiseDistance(cuvsResources_t res,
                                 DLManagedTensor* x,
                                 DLManagedTensor* y,
                                 DLManagedTensor* dist,
                                 cuvsDistanceType metric,
                                 float metric_arg);

/** @} */

#ifdef __cplusplus
}
#endif
