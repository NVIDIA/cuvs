/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>

#include <dlpack/dlpack.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

float dataset[4][2] = {{0.74021935, 0.9209938},
                       {0.03902049, 0.9689629},
                       {0.92514056, 0.4463501},
                       {0.6673192, 0.10993068}};
float queries[4][2] = {{0.48216683, 0.0428398},
                       {0.5084142, 0.6545497},
                       {0.51260436, 0.2643005},
                       {0.05198065, 0.5789965}};

void cagra_build_search_simple()
{
  int64_t n_rows    = 4;
  int64_t n_cols    = 2;
  int64_t topk      = 2;
  int64_t n_queries = 4;

  // Create a cuvsResources_t object
  cuvsResources_t res;
  CHECK_CUVS(cuvsResourcesCreate(&res));

  // Use DLPack to represent `dataset` as a tensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = dataset;
  dataset_tensor.dl_tensor.device.device_type = kDLCPU;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {n_rows, n_cols};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;

  // Build the CAGRA index
  cuvsCagraIndexParams_t index_params;
  CHECK_CUVS(cuvsCagraIndexParamsCreate(&index_params));

  cuvsCagraIndex_t index;
  CHECK_CUVS(cuvsCagraIndexCreate(&index));

  cuvsDatasetStandardView_t host_dataset_view = NULL;
  CHECK_CUVS(cuvsDatasetMakeHostStandardView(res, &dataset_tensor, &host_dataset_view));
  CHECK_CUVS(cuvsCagraBuildHostStandard(res, index_params, host_dataset_view, index));

  // Allocate memory for `queries`, `neighbors` and `distances` output
  uint32_t* neighbors;
  float *distances, *queries_d;
  CHECK_CUVS(cuvsRMMAlloc(res, (void**)&queries_d, sizeof(float) * n_queries * n_cols));
  CHECK_CUVS(cuvsRMMAlloc(res, (void**)&neighbors, sizeof(uint32_t) * n_queries * topk));
  CHECK_CUVS(cuvsRMMAlloc(res, (void**)&distances, sizeof(float) * n_queries * topk));

  // Use DLPack to represent `queries`, `neighbors` and `distances` as tensors
  CHECK_CUDA(cudaMemcpy(queries_d, queries, sizeof(float) * 4 * 2, cudaMemcpyDefault));

  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries_d;
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {n_queries, n_cols};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = NULL;

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors;
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 32;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {n_queries, topk};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = NULL;

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances;
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {n_queries, topk};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = NULL;

  // Attach a caller-provided device padded dataset to get a search-ready index.
  DLManagedTensor device_dataset_tensor = dataset_tensor;
  float* dataset_d                      = NULL;
  CHECK_CUVS(cuvsRMMAlloc(res, (void**)&dataset_d, sizeof(float) * n_rows * n_cols));
  CHECK_CUDA(cudaMemcpy(dataset_d, dataset, sizeof(float) * n_rows * n_cols, cudaMemcpyDefault));
  device_dataset_tensor.dl_tensor.data               = dataset_d;
  device_dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  device_dataset_tensor.dl_tensor.device.device_id   = 0;
  cuvsDatasetPadded_t padded_owner                   = NULL;
  CHECK_CUVS(cuvsDatasetMakeDevicePadded(res, &device_dataset_tensor, &padded_owner));
  cuvsDatasetPaddedView_t padded_view = NULL;
  CHECK_CUVS(cuvsDatasetMakeViewFromOwningPadded(padded_owner, &padded_view));
  CHECK_CUVS(cuvsCagraAttachDataset(res, padded_view, index));

  // Search the CAGRA index
  cuvsCagraSearchParams_t search_params;
  CHECK_CUVS(cuvsCagraSearchParamsCreate(&search_params));

  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = (uintptr_t)NULL;

  CHECK_CUVS(cuvsCagraSearch(
    res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor, filter));

  // print results
  uint32_t* neighbors_h = (uint32_t*)malloc(sizeof(uint32_t) * n_queries * topk);
  float* distances_h    = (float*)malloc(sizeof(float) * n_queries * topk);
  CHECK_CUDA(
    cudaMemcpy(neighbors_h, neighbors, sizeof(uint32_t) * n_queries * topk, cudaMemcpyDefault));
  CHECK_CUDA(
    cudaMemcpy(distances_h, distances, sizeof(float) * n_queries * topk, cudaMemcpyDefault));
  printf("Query 0 neighbor indices: =[%d, %d]\n", neighbors_h[0], neighbors_h[1]);
  printf("Query 0 neighbor distances: =[%f, %f]\n", distances_h[0], distances_h[1]);

  // Free or destroy all allocations
  free(neighbors_h);
  free(distances_h);

  CHECK_CUVS(cuvsCagraSearchParamsDestroy(search_params));
  CHECK_CUVS(cuvsDatasetPaddedViewDestroy(padded_view));
  CHECK_CUVS(cuvsDatasetPaddedDestroy(padded_owner));
  CHECK_CUVS(cuvsDatasetStandardViewDestroy(host_dataset_view));

  CHECK_CUVS(cuvsRMMFree(res, distances, sizeof(float) * n_queries * topk));
  CHECK_CUVS(cuvsRMMFree(res, neighbors, sizeof(uint32_t) * n_queries * topk));
  CHECK_CUVS(cuvsRMMFree(res, queries_d, sizeof(float) * n_queries * n_cols));
  CHECK_CUVS(cuvsRMMFree(res, dataset_d, sizeof(float) * n_rows * n_cols));

  CHECK_CUVS(cuvsCagraIndexDestroy(index));
  CHECK_CUVS(cuvsCagraIndexParamsDestroy(index_params));
  CHECK_CUVS(cuvsResourcesDestroy(res));
}

int main()
{
  // Simple build and search example.
  cagra_build_search_simple();
}
