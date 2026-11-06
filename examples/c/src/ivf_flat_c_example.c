/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/ivf_flat.h>

#include "common.h"
#include <cuda_runtime.h>

void ivf_flat_build_search_simple(cuvsResources_t* res,
                                  DLManagedTensor* dataset_tensor,
                                  DLManagedTensor* queries_tensor)
{
  // Create default index params
  cuvsIvfFlatIndexParams_t index_params;
  CHECK_CUVS(cuvsIvfFlatIndexParamsCreate(&index_params));
  index_params->n_lists                  = 1024;  // default value
  index_params->kmeans_n_iters           = 20;    // default value
  index_params->kmeans_trainset_fraction = 0.1;
  // index_params->metric default is L2Expanded

  // Create IVF-Flat index
  cuvsIvfFlatIndex_t index;
  CHECK_CUVS(cuvsIvfFlatIndexCreate(&index));

  printf("Building IVF-Flat index\n");
  // Build the IVF-Flat Index
  CHECK_CUVS(cuvsIvfFlatBuild(*res, index_params, dataset_tensor, index));

  // Create output arrays.
  int64_t topk      = 10;
  int64_t n_queries = queries_tensor->dl_tensor.shape[0];

  // Allocate memory for `neighbors` and `distances` output
  int64_t* neighbors_d;
  float* distances_d;
  CHECK_CUVS(cuvsRMMAlloc(*res, (void**)&neighbors_d, sizeof(int64_t) * n_queries * topk));
  CHECK_CUVS(cuvsRMMAlloc(*res, (void**)&distances_d, sizeof(float) * n_queries * topk));

  DLManagedTensor neighbors_tensor = {0};
  int64_t neighbors_shape[2]       = {n_queries, topk};
  int_tensor_initialize(neighbors_d, neighbors_shape, &neighbors_tensor);

  DLManagedTensor distances_tensor = {0};
  int64_t distances_shape[2]       = {n_queries, topk};
  float_tensor_initialize(distances_d, distances_shape, &distances_tensor);

  // Create default search params
  cuvsIvfFlatSearchParams_t search_params;
  CHECK_CUVS(cuvsIvfFlatSearchParamsCreate(&search_params));
  search_params->n_probes = 50;

  // Search the `index` built using `ivfFlatBuild`
  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = (uintptr_t)NULL;

  CHECK_CUVS(cuvsIvfFlatSearch(
    *res, search_params, index, queries_tensor, &neighbors_tensor, &distances_tensor, filter));

  int64_t* neighbors = (int64_t*)malloc(n_queries * topk * sizeof(int64_t));
  float* distances   = (float*)malloc(n_queries * topk * sizeof(float));
  memset(neighbors, 0, n_queries * topk * sizeof(int64_t));
  memset(distances, 0, n_queries * topk * sizeof(float));

  CHECK_CUDA(
    cudaMemcpy(neighbors, neighbors_d, sizeof(int64_t) * n_queries * topk, cudaMemcpyDefault));
  CHECK_CUDA(
    cudaMemcpy(distances, distances_d, sizeof(float) * n_queries * topk, cudaMemcpyDefault));

  print_results(neighbors, distances, 2, topk);

  free(distances);
  free(neighbors);

  CHECK_CUVS(cuvsRMMFree(*res, neighbors_d, sizeof(int64_t) * n_queries * topk));
  CHECK_CUVS(cuvsRMMFree(*res, distances_d, sizeof(float) * n_queries * topk));

  CHECK_CUVS(cuvsIvfFlatSearchParamsDestroy(search_params));
  CHECK_CUVS(cuvsIvfFlatIndexDestroy(index));
  CHECK_CUVS(cuvsIvfFlatIndexParamsDestroy(index_params));
}

void ivf_flat_build_extend_search(cuvsResources_t* res,
                                  DLManagedTensor* trainset_tensor,
                                  DLManagedTensor* dataset_tensor,
                                  DLManagedTensor* queries_tensor)
{
  int64_t* data_indices_d;
  int64_t n_dataset = dataset_tensor->dl_tensor.shape[0];
  CHECK_CUVS(cuvsRMMAlloc(*res, (void**)&data_indices_d, sizeof(int64_t) * n_dataset));
  DLManagedTensor data_indices_tensor = {0};
  int64_t data_indices_shape[1]       = {n_dataset};
  int_tensor_initialize(data_indices_d, data_indices_shape, &data_indices_tensor);
  data_indices_tensor.dl_tensor.ndim = 1;

  printf("\nRun k-means clustering using the training set\n");

  int64_t* data_indices = (int64_t*)malloc(n_dataset * sizeof(int64_t));
  int64_t* ptr          = data_indices;
  for (int i = 0; i < n_dataset; i++) {
    *ptr = i;
    ptr++;
  }
  ptr = NULL;
  cudaMemcpy(data_indices_d, data_indices, sizeof(int64_t) * n_dataset, cudaMemcpyDefault);

  // Create default index params
  cuvsIvfFlatIndexParams_t index_params;
  CHECK_CUVS(cuvsIvfFlatIndexParamsCreate(&index_params));
  index_params->n_lists           = 100;
  index_params->add_data_on_build = false;
  // index_params->metric default is L2Expanded

  // Create IVF-Flat index
  cuvsIvfFlatIndex_t index;
  CHECK_CUVS(cuvsIvfFlatIndexCreate(&index));

  // Build the IVF-Flat Index
  CHECK_CUVS(cuvsIvfFlatBuild(*res, index_params, trainset_tensor, index));

  printf("Filling index with the dataset vectors\n");
  CHECK_CUVS(cuvsIvfFlatExtend(*res, dataset_tensor, &data_indices_tensor, index));

  // Create output arrays.
  int64_t topk      = 10;
  int64_t n_queries = queries_tensor->dl_tensor.shape[0];

  // Allocate memory for `neighbors` and `distances` output
  int64_t* neighbors_d;
  float* distances_d;
  CHECK_CUVS(cuvsRMMAlloc(*res, (void**)&neighbors_d, sizeof(int64_t) * n_queries * topk));
  CHECK_CUVS(cuvsRMMAlloc(*res, (void**)&distances_d, sizeof(float) * n_queries * topk));

  DLManagedTensor neighbors_tensor = {0};
  int64_t neighbors_shape[2]       = {n_queries, topk};
  int_tensor_initialize(neighbors_d, neighbors_shape, &neighbors_tensor);

  DLManagedTensor distances_tensor = {0};
  int64_t distances_shape[2]       = {n_queries, topk};
  float_tensor_initialize(distances_d, distances_shape, &distances_tensor);

  // Create default search params
  cuvsIvfFlatSearchParams_t search_params;
  CHECK_CUVS(cuvsIvfFlatSearchParamsCreate(&search_params));
  search_params->n_probes = 10;

  // Search the `index` built using `ivfFlatBuild`
  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = (uintptr_t)NULL;
  CHECK_CUVS(cuvsIvfFlatSearch(
    *res, search_params, index, queries_tensor, &neighbors_tensor, &distances_tensor, filter));

  int64_t* neighbors = (int64_t*)malloc(n_queries * topk * sizeof(int64_t));
  float* distances   = (float*)malloc(n_queries * topk * sizeof(float));
  memset(neighbors, 0, n_queries * topk * sizeof(int64_t));
  memset(distances, 0, n_queries * topk * sizeof(float));

  CHECK_CUDA(
    cudaMemcpy(neighbors, neighbors_d, sizeof(int64_t) * n_queries * topk, cudaMemcpyDefault));
  CHECK_CUDA(
    cudaMemcpy(distances, distances_d, sizeof(float) * n_queries * topk, cudaMemcpyDefault));

  print_results(neighbors, distances, 2, topk);

  free(distances);
  free(neighbors);
  free(data_indices);
  CHECK_CUVS(cuvsRMMFree(*res, data_indices_d, sizeof(int64_t) * n_dataset));
  CHECK_CUVS(cuvsRMMFree(*res, neighbors_d, sizeof(int64_t) * n_queries * topk));
  CHECK_CUVS(cuvsRMMFree(*res, distances_d, sizeof(float) * n_queries * topk));

  CHECK_CUVS(cuvsIvfFlatSearchParamsDestroy(search_params));
  CHECK_CUVS(cuvsIvfFlatIndexDestroy(index));

  CHECK_CUVS(cuvsIvfFlatIndexParamsDestroy(index_params));
}

int main()
{
  // Create input arrays.
  int64_t n_samples = 10000;
  int64_t n_dim     = 3;
  int64_t n_queries = 10;
  float* dataset    = (float*)malloc(n_samples * n_dim * sizeof(float));
  float* queries    = (float*)malloc(n_queries * n_dim * sizeof(float));
  generate_dataset(dataset, n_samples, n_dim, -10.0, 10.0);
  generate_dataset(queries, n_queries, n_dim, -1.0, 1.0);

  // Create a cuvsResources_t object
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // Allocate memory for `queries`
  float* dataset_d;
  CHECK_CUVS(cuvsRMMAlloc(res, (void**)&dataset_d, sizeof(float) * n_samples * n_dim));
  // Use DLPack to represent `dataset_d` as a tensor
  CHECK_CUDA(cudaMemcpy(dataset_d, dataset, sizeof(float) * n_samples * n_dim, cudaMemcpyDefault));

  DLManagedTensor dataset_tensor = {0};
  int64_t dataset_shape[2]       = {n_samples, n_dim};
  float_tensor_initialize(dataset_d, dataset_shape, &dataset_tensor);

  // Allocate memory for `queries`
  float* queries_d;
  CHECK_CUVS(cuvsRMMAlloc(res, (void**)&queries_d, sizeof(float) * n_queries * n_dim));

  // Use DLPack to represent `queries` as tensors
  cudaMemcpy(queries_d, queries, sizeof(float) * n_queries * n_dim, cudaMemcpyDefault);

  DLManagedTensor queries_tensor = {0};
  int64_t queries_shape[2]       = {n_queries, n_dim};
  float_tensor_initialize(queries_d, queries_shape, &queries_tensor);

  // Simple build and search example.
  ivf_flat_build_search_simple(&res, &dataset_tensor, &queries_tensor);

  float* trainset_d;
  int64_t n_trainset = n_samples * 0.1;
  float* trainset    = (float*)malloc(n_trainset * n_dim * sizeof(float));
  for (int i = 0; i < n_trainset; i++) {
    for (int j = 0; j < n_dim; j++) {
      *(trainset + i * n_dim + j) = *(dataset + i * n_dim + j);
    }
  }
  CHECK_CUVS(cuvsRMMAlloc(res, (void**)&trainset_d, sizeof(float) * n_trainset * n_dim));
  CHECK_CUDA(
    cudaMemcpy(trainset_d, trainset, sizeof(float) * n_trainset * n_dim, cudaMemcpyDefault));
  DLManagedTensor trainset_tensor = {0};
  int64_t trainset_shape[2]       = {n_trainset, n_dim};
  float_tensor_initialize(trainset_d, trainset_shape, &trainset_tensor);

  // Build and extend example.
  ivf_flat_build_extend_search(&res, &trainset_tensor, &dataset_tensor, &queries_tensor);

  CHECK_CUVS(cuvsRMMFree(res, trainset_d, sizeof(float) * n_trainset * n_dim));
  CHECK_CUVS(cuvsRMMFree(res, queries_d, sizeof(float) * n_queries * n_dim));
  CHECK_CUVS(cuvsRMMFree(res, dataset_d, sizeof(float) * n_samples * n_dim));
  CHECK_CUVS(cuvsResourcesDestroy(res));
  free(trainset);
  free(dataset);
  free(queries);
}
