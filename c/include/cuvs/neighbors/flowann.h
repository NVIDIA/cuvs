/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/core/dataset.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/cagra.h>

#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Parameters for building a dense FlowANN index. */
typedef struct {
  cuvsDistanceType metric;
  size_t intermediate_graph_degree;
  size_t graph_degree;
  enum cuvsCagraGraphBuildAlgo build_algo;
  size_t nn_descent_niter;
  size_t device_graph_budget_bytes;
  uint16_t node_per_cacheline;
  bool grouping_enabled;
  uint32_t n_groups;
  uint16_t n_bits;
  double balance_tolerance;
  size_t training_rows;
  size_t assignment_batch_rows;
  uint32_t kmeans_n_iters;
  bool validate;
  uint32_t num_seeds;
  size_t seed_training_rows;
  uint64_t seed;
} cuvsFlowannIndexParams;
typedef cuvsFlowannIndexParams* cuvsFlowannIndexParams_t;

/** Search and host queue parameters for FlowANN. */
typedef struct {
  struct cuvsCagraSearchParams cagra;
  uint32_t num_seeds;
  float sync_window_scale;
  uint32_t sync_drop_threshold;
  uint32_t num_queues;
  uint32_t empty_pause;
  bool collect_statistics;
} cuvsFlowannSearchParams;
typedef cuvsFlowannSearchParams* cuvsFlowannSearchParams_t;

/** Opaque owning FlowANN index. */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsFlowannIndex;
typedef cuvsFlowannIndex* cuvsFlowannIndex_t;

CUVS_EXPORT cuvsError_t cuvsFlowannIndexParamsCreate(cuvsFlowannIndexParams_t* params);
CUVS_EXPORT cuvsError_t cuvsFlowannIndexParamsDestroy(cuvsFlowannIndexParams_t params);
CUVS_EXPORT cuvsError_t cuvsFlowannSearchParamsCreate(cuvsFlowannSearchParams_t* params);
CUVS_EXPORT cuvsError_t cuvsFlowannSearchParamsDestroy(cuvsFlowannSearchParams_t params);
CUVS_EXPORT cuvsError_t cuvsFlowannIndexCreate(cuvsFlowannIndex_t* index);
CUVS_EXPORT cuvsError_t cuvsFlowannIndexDestroy(cuvsFlowannIndex_t index);

CUVS_EXPORT cuvsError_t cuvsFlowannIndexGetDims(cuvsFlowannIndex_t index, int64_t* dim);
CUVS_EXPORT cuvsError_t cuvsFlowannIndexGetSize(cuvsFlowannIndex_t index, int64_t* size);
CUVS_EXPORT cuvsError_t cuvsFlowannIndexGetGraphDegree(cuvsFlowannIndex_t index, int64_t* degree);

/** Build an owning dense FlowANN index from a float32, int8, or uint8 device-padded dataset. */
CUVS_EXPORT cuvsError_t cuvsFlowannBuild(cuvsResources_t res,
                                         cuvsFlowannIndexParams_t params,
                                         cuvsDataset_t dataset,
                                         cuvsFlowannIndex_t index);

/** Search an owning dense FlowANN index. Output neighbors must use uint32. */
CUVS_EXPORT cuvsError_t cuvsFlowannSearch(cuvsResources_t res,
                                          cuvsFlowannSearchParams_t params,
                                          cuvsFlowannIndex_t index,
                                          DLManagedTensor* queries,
                                          DLManagedTensor* neighbors,
                                          DLManagedTensor* distances);

CUVS_EXPORT cuvsError_t cuvsFlowannSerialize(cuvsResources_t res,
                                             const char* filename,
                                             cuvsFlowannIndex_t index);

/** Deserialize a dense index. dtype must match the dtype stored in the file. */
CUVS_EXPORT cuvsError_t cuvsFlowannDeserialize(cuvsResources_t res,
                                               const char* filename,
                                               DLDataType dtype,
                                               cuvsFlowannIndex_t index);

#ifdef __cplusplus
}
#endif
