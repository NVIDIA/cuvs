/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/vamana.h>

cuvsVamanaGraphInitAlgo default_vamana_graph_init(void)
{
  cuvsVamanaIndexParams_t params;
  cuvsVamanaIndexParamsCreate(&params);
  cuvsVamanaGraphInitAlgo graph_init = params->graph_init;
  cuvsVamanaIndexParamsDestroy(params);
  return graph_init;
}

cuvsError_t run_vamana(cuvsResources_t res,
                       int64_t n_rows,
                       int64_t n_dim,
                       float* index_data,
                       uint32_t graph_degree,
                       uint32_t visited_size,
                       cuvsVamanaGraphInitAlgo graph_init,
                       const char* graph_file)
{
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = index_data;
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

  cuvsVamanaIndexParams_t build_params;
  cuvsVamanaIndexParamsCreate(&build_params);
  build_params->graph_degree = graph_degree;
  build_params->visited_size = visited_size;
  build_params->graph_init   = graph_init;

  cuvsVamanaIndex_t index;
  cuvsVamanaIndexCreate(&index);

  cuvsError_t status = cuvsVamanaBuild(res, build_params, &dataset_tensor, index);
  if (status == CUVS_SUCCESS) { status = cuvsVamanaSerialize(res, graph_file, index, false); }

  cuvsVamanaIndexDestroy(index);
  cuvsVamanaIndexParamsDestroy(build_params);
  return status;
}
