/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/all_neighbors.h>
#include <stdint.h>

void run_all_neighbors(int64_t n_rows,
                       int64_t n_dim,
                       int64_t k,
                       float* dataset_data,
                       int64_t* indices_data,
                       float* distances_data,
                       float* core_distances_data,
                       cuvsDistanceType metric,
                       cuvsAllNeighborsAlgo algo,
                       float alpha)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // create all neighbors index parameters
  cuvsAllNeighborsIndexParams_t params;
  cuvsAllNeighborsIndexParamsCreate(&params);

  // configure parameters
  params->algo           = algo;
  params->metric         = metric;
  params->overlap_factor = 1;
  params->n_clusters     = 1;

  // create dataset DLTensor
  DLManagedTensor dataset_tensor = {0};
  dataset_tensor.dl_tensor.data               = dataset_data;
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.device.device_id   = 0;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {n_rows, n_dim};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = NULL;
  dataset_tensor.dl_tensor.byte_offset        = 0;
  dataset_tensor.manager_ctx                  = NULL;
  dataset_tensor.deleter                      = NULL;

  // create indices DLTensor
  DLManagedTensor indices_tensor = {0};
  indices_tensor.dl_tensor.data               = (void*)indices_data;
  indices_tensor.dl_tensor.device.device_type = kDLCUDA;
  indices_tensor.dl_tensor.device.device_id   = 0;
  indices_tensor.dl_tensor.ndim               = 2;
  indices_tensor.dl_tensor.dtype.code         = kDLInt;
  indices_tensor.dl_tensor.dtype.bits         = 64;
  indices_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t indices_shape[2]                    = {n_rows, k};
  indices_tensor.dl_tensor.shape              = indices_shape;
  indices_tensor.dl_tensor.strides            = NULL;
  indices_tensor.dl_tensor.byte_offset        = 0;
  indices_tensor.manager_ctx                  = NULL;
  indices_tensor.deleter                      = NULL;

  // create distances DLTensor (optional)
  DLManagedTensor* distances_ptr = NULL;
  DLManagedTensor distances_tensor = {0};
  int64_t distances_shape[2] = {n_rows, k};  // Moved outside if block
  if (distances_data != NULL) {
    distances_tensor.dl_tensor.data               = (void*)distances_data;
    distances_tensor.dl_tensor.device.device_type = kDLCUDA;
    distances_tensor.dl_tensor.device.device_id   = 0;
    distances_tensor.dl_tensor.ndim               = 2;
    distances_tensor.dl_tensor.dtype.code         = kDLFloat;
    distances_tensor.dl_tensor.dtype.bits         = 32;
    distances_tensor.dl_tensor.dtype.lanes        = 1;
    distances_tensor.dl_tensor.shape              = distances_shape;
    distances_tensor.dl_tensor.strides            = NULL;
    distances_tensor.dl_tensor.byte_offset        = 0;
    distances_tensor.manager_ctx                  = NULL;
    distances_tensor.deleter                      = NULL;
    distances_ptr                                 = &distances_tensor;
  }

  // create core_distances DLTensor (optional)
  DLManagedTensor* core_distances_ptr = NULL;
  DLManagedTensor core_distances_tensor = {0};
  int64_t core_distances_shape[1] = {n_rows};  // Moved outside if block
  if (core_distances_data != NULL) {
    core_distances_tensor.dl_tensor.data               = (void*)core_distances_data;
    core_distances_tensor.dl_tensor.device.device_type = kDLCUDA;
    core_distances_tensor.dl_tensor.device.device_id   = 0;
    core_distances_tensor.dl_tensor.ndim               = 1;
    core_distances_tensor.dl_tensor.dtype.code         = kDLFloat;
    core_distances_tensor.dl_tensor.dtype.bits         = 32;
    core_distances_tensor.dl_tensor.dtype.lanes        = 1;
    core_distances_tensor.dl_tensor.shape              = core_distances_shape;
    core_distances_tensor.dl_tensor.strides            = NULL;
    core_distances_tensor.dl_tensor.byte_offset        = 0;
    core_distances_tensor.manager_ctx                  = NULL;
    core_distances_tensor.deleter                      = NULL;
    core_distances_ptr                                 = &core_distances_tensor;
  }

  // build all neighbors graph
  cuvsError_t result = cuvsAllNeighborsBuild(
    res, params, &dataset_tensor, &indices_tensor, distances_ptr, core_distances_ptr, alpha);

  // cleanup
  cuvsAllNeighborsIndexParamsDestroy(params);
  cuvsResourcesDestroy(res);
}
