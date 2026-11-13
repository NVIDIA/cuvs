/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/brute_force.h>
#include <stdint.h>

static const char dataset[] = {0.74021935f,
                               0.9209938f,
                               0.03902049f,
                               0.9689629f,
                               0.92514056f,
                               0.4463501f,
                               0.6673192f,
                               0.10993068f};

static const char queries[] = {0.48216683f,
                               0.0428398f,
                               0.5084142f,
                               0.6545497f,
                               0.51260436f,
                               0.2643005f,
                               0.05198065f,
                               0.5789965f};

void index_and_search()
{
  int64_t n_rows       = 4;
  int64_t n_queries    = 4;
  int64_t n_dim        = 2;
  uint32_t n_neighbors = 2;

  float* index_data;
  float* query_data;

  long indexBytes     = sizeof(float) * n_rows * n_dim;
  long queriesBytes   = sizeof(float) * n_queries * n_dim;
  long neighborsBytes = sizeof(long) * n_queries * n_neighbors;
  long distanceBytes  = sizeof(float) * n_queries * n_neighbors;

  uint32_t* prefilter_data           = NULL;
  enum cuvsFilterType prefilter_type = NO_FILTER;

  float* distances_data;
  int64_t* neighbors_data;

  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  cuvsRMMAlloc(res, (void**)&index_data, indexBytes);
  cuvsRMMAlloc(res, (void**)&query_data, queriesBytes);
  cuvsRMMAlloc(res, (void**)&distances_data, distanceBytes);
  cuvsRMMAlloc(res, (void**)&neighbors_data, neighborsBytes);

  cudaMemcpy(index_data, dataset, indexBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(query_data, queries, queriesBytes, cudaMemcpyHostToDevice);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor              = {0};
  dataset_tensor.dl_tensor.data               = index_data;
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {n_rows, n_dim};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;

  // create index
  cuvsBruteForceIndex_t index;
  cuvsBruteForceIndexCreate(&index);

  // build index
  cuvsBruteForceBuild(res, &dataset_tensor, 0, 0.0f, index);

  // create queries DLTensor
  DLManagedTensor queries_tensor              = {0};
  queries_tensor.dl_tensor.data               = (void*)query_data;
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {n_queries, n_dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = NULL;

  // create neighbors DLTensor
  DLManagedTensor neighbors_tensor              = {0};
  neighbors_tensor.dl_tensor.data               = (void*)neighbors_data;
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {n_queries, n_neighbors};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = NULL;

  // create distances DLTensor
  DLManagedTensor distances_tensor              = {0};
  distances_tensor.dl_tensor.data               = (void*)distances_data;
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {n_queries, n_neighbors};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = NULL;

  cuvsFilter prefilter;

  DLManagedTensor prefilter_tensor = {0};
  if (prefilter_data == NULL || prefilter_type == NO_FILTER) {
    prefilter.type = NO_FILTER;
    prefilter.addr = (uintptr_t)NULL;
  } else {
    prefilter_tensor.dl_tensor.data               = (void*)prefilter_data;
    prefilter_tensor.dl_tensor.device.device_type = kDLCUDA;
    prefilter_tensor.dl_tensor.ndim               = 1;
    prefilter_tensor.dl_tensor.dtype.code         = kDLUInt;
    prefilter_tensor.dl_tensor.dtype.bits         = 32;
    prefilter_tensor.dl_tensor.dtype.lanes        = 1;

    int64_t prefilter_bits_num = (prefilter_type == BITMAP) ? n_queries * n_rows : n_rows;
    int64_t prefilter_shape[1] = {(prefilter_bits_num + 31) / 32};

    prefilter_tensor.dl_tensor.shape   = prefilter_shape;
    prefilter_tensor.dl_tensor.strides = NULL;

    prefilter.type = prefilter_type;
    prefilter.addr = (uintptr_t)&prefilter_tensor;
  }

  // search index
  cuvsBruteForceSearch(
    res, index, &queries_tensor, &neighbors_tensor, &distances_tensor, prefilter);

  // de-allocate index and res
  cuvsBruteForceIndexDestroy(index);

  cuvsRMMFree(res, index_data, indexBytes);
  cuvsRMMFree(res, query_data, queriesBytes);
  cuvsRMMFree(res, distances_data, distanceBytes);
  cuvsRMMFree(res, neighbors_data, neighborsBytes);

  cuvsResourcesDestroy(res);
}

int main()
{
  // Perform indexing and search with pooled resources
  cuvsRMMPoolMemoryResourceEnable(10, 60, false);
  index_and_search();

  // Perform indexing and search with the default memory resources
  cuvsRMMMemoryResourceReset();
  index_and_search();

  return 0;
}
