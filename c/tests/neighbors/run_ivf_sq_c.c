/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_sq.h>

void run_ivf_sq(cuvsResources_t res,
                int64_t n_rows,
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
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = index_data;
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {n_rows, n_dim};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;

  cuvsIvfSqIndex_t index;
  cuvsIvfSqIndexCreate(&index);

  cuvsIvfSqIndexParams_t build_params;
  cuvsIvfSqIndexParamsCreate(&build_params);
  build_params->metric  = metric;
  build_params->n_lists = n_lists;
  cuvsIvfSqBuild(res, build_params, &dataset_tensor, index);

  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = (void*)query_data;
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {n_queries, n_dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = NULL;

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = (void*)neighbors_data;
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {n_queries, n_neighbors};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = NULL;

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = (void*)distances_data;
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {n_queries, n_neighbors};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = NULL;

  cuvsIvfSqSearchParams_t search_params;
  cuvsIvfSqSearchParamsCreate(&search_params);
  search_params->n_probes = n_probes;
  cuvsIvfSqSearch(res,
                  search_params,
                  index,
                  &queries_tensor,
                  &neighbors_tensor,
                  &distances_tensor,
                  (cuvsFilter){.type = NO_FILTER});
  cuvsStreamSync(res);

  cuvsIvfSqSearchParamsDestroy(search_params);
  cuvsIvfSqIndexParamsDestroy(build_params);
  cuvsIvfSqIndexDestroy(index);
}
