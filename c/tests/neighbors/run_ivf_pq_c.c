/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_pq.h>

void run_ivf_pq(int64_t n_rows,
                int64_t n_queries,
                int64_t n_dim,
                uint32_t n_neighbors,
                float* index_data,
                float* query_data,
                float* distances_data,
                int64_t* neighbors_data,
                cuvsDistanceType metric,
                size_t n_probes,
                size_t n_lists)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor = {0};
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
  cuvsIvfPqIndex_t index;
  cuvsIvfPqIndexCreate(&index);

  // build index
  cuvsIvfPqIndexParams_t build_params;
  cuvsIvfPqIndexParamsCreate(&build_params);
  build_params->metric  = metric;
  build_params->n_lists = n_lists;
  cuvsIvfPqBuild(res, build_params, &dataset_tensor, index);

  // create queries DLTensor
  DLManagedTensor queries_tensor = {0};
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
  DLManagedTensor neighbors_tensor = {0};
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
  DLManagedTensor distances_tensor = {0};
  distances_tensor.dl_tensor.data               = (void*)distances_data;
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {n_queries, n_neighbors};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = NULL;

  // search index
  cuvsIvfPqSearchParams_t search_params;
  cuvsIvfPqSearchParamsCreate(&search_params);
  search_params->n_probes = n_probes;
  cuvsIvfPqSearch(res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor);

  // de-allocate index and res
  cuvsIvfPqSearchParamsDestroy(search_params);
  cuvsIvfPqIndexParamsDestroy(build_params);
  cuvsIvfPqIndexDestroy(index);
  cuvsResourcesDestroy(res);
}
