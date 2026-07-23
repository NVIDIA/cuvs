/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.cuh"
#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/hnsw.h>
#include <dlpack/dlpack.h>

#include <cstdint>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <vector>

float dataset[4][2] = {{0.74021935, 0.9209938},
                       {0.03902049, 0.9689629},
                       {0.92514056, 0.4463501},
                       {0.6673192, 0.10993068}};
float queries[4][2] = {{0.48216683, 0.0428398},
                       {0.5084142, 0.6545497},
                       {0.51260436, 0.2643005},
                       {0.05198065, 0.5789965}};

std::vector<uint64_t> neighbors_exp = {3, 0, 3, 1};
std::vector<float> distances_exp    = {0.03878258, 0.12472608, 0.04776672, 0.15224178};

TEST(CagraHnswC, BuildSearch)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = dataset;
  dataset_tensor.dl_tensor.device.device_type = kDLCPU;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {4, 2};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = nullptr;

  // create index
  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);

  // build index
  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);
  cuvsCagraBuild(res, build_params, &dataset_tensor, index);
  cuvsCagraSerializeToHnswlib(res, "/tmp/cagra_hnswlib.index", index);

  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries;
  queries_tensor.dl_tensor.device.device_type = kDLCPU;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {4, 2};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  // create neighbors DLTensor
  std::vector<uint64_t> neighbors(4);

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCPU;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {4, 1};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  // create distances DLTensor
  std::vector<float> distances(4);

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances.data();
  distances_tensor.dl_tensor.device.device_type = kDLCPU;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {4, 1};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  // create hnsw index
  cuvsHnswIndex_t hnsw_index;
  cuvsHnswIndexCreate(&hnsw_index);
  hnsw_index->dtype = index->dtype;
  cuvsHnswIndexParams_t hnsw_params;
  cuvsHnswIndexParamsCreate(&hnsw_params);
  // Use NONE hierarchy since cuvsCagraSerializeToHnswlib creates a base-layer-only index
  hnsw_params->hierarchy = NONE;
  cuvsHnswDeserialize(res, hnsw_params, "/tmp/cagra_hnswlib.index", 2, L2Expanded, hnsw_index);

  // search index
  cuvsHnswSearchParams_t search_params;
  cuvsHnswSearchParamsCreate(&search_params);
  cuvsHnswSearch(
    res, search_params, hnsw_index, &queries_tensor, &neighbors_tensor, &distances_tensor);

  // verify output
  ASSERT_TRUE(cuvs::hostVecMatch(neighbors_exp, neighbors, cuvs::Compare<uint64_t>()));
  ASSERT_TRUE(cuvs::hostVecMatch(distances_exp, distances, cuvs::CompareApprox<float>(0.001f)));

  cuvsCagraIndexParamsDestroy(build_params);
  cuvsCagraIndexDestroy(index);
  cuvsHnswSearchParamsDestroy(search_params);
  cuvsHnswIndexDestroy(hnsw_index);
  cuvsResourcesDestroy(res);
}

TEST(HnswC, BuildWithoutAce)
{
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = dataset;
  dataset_tensor.dl_tensor.device.device_type = kDLCPU;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {4, 2};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = nullptr;

  cuvsHnswIndexParams_t hnsw_params;
  cuvsHnswIndexParamsCreate(&hnsw_params);
  hnsw_params->M               = 2;
  hnsw_params->ef_construction = 100;

  cuvsHnswIndex_t hnsw_index;
  cuvsHnswIndexCreate(&hnsw_index);

  ASSERT_EQ(cuvsHnswBuild(res, hnsw_params, &dataset_tensor, hnsw_index), CUVS_SUCCESS);

  cuvsHnswIndexParamsDestroy(hnsw_params);
  cuvsHnswIndexDestroy(hnsw_index);
  cuvsResourcesDestroy(res);
}

TEST(HnswAceC, BuildSearch)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = dataset;
  dataset_tensor.dl_tensor.device.device_type = kDLCPU;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {4, 2};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = nullptr;

  // create ACE params
  cuvsHnswAceParams_t ace_params;
  cuvsHnswAceParamsCreate(&ace_params);
  ace_params->npartitions = 2;
  ace_params->build_dir   = "/tmp/hnsw_ace_test";
  ace_params->use_disk    = false;

  // create index params
  cuvsHnswIndexParams_t hnsw_params;
  cuvsHnswIndexParamsCreate(&hnsw_params);
  hnsw_params->hierarchy       = GPU;
  hnsw_params->ef_construction = 100;
  hnsw_params->ace_params      = ace_params;
  hnsw_params->metric          = L2Expanded;
  hnsw_params->M               = 16;

  // create HNSW index
  cuvsHnswIndex_t hnsw_index;
  cuvsHnswIndexCreate(&hnsw_index);

  // build index using ACE
  cuvsError_t build_status = cuvsHnswBuild(res, hnsw_params, &dataset_tensor, hnsw_index);
  ASSERT_EQ(build_status, CUVS_SUCCESS) << "cuvsHnswBuild failed";

  // create queries DLTensor
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries;
  queries_tensor.dl_tensor.device.device_type = kDLCPU;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {4, 2};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  // create neighbors DLTensor
  std::vector<uint64_t> neighbors(4);
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCPU;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {4, 1};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  // create distances DLTensor
  std::vector<float> distances(4);
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances.data();
  distances_tensor.dl_tensor.device.device_type = kDLCPU;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {4, 1};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  // search index
  cuvsHnswSearchParams_t search_params;
  cuvsHnswSearchParamsCreate(&search_params);
  search_params->ef = 100;

  cuvsError_t search_status = cuvsHnswSearch(
    res, search_params, hnsw_index, &queries_tensor, &neighbors_tensor, &distances_tensor);
  ASSERT_EQ(search_status, CUVS_SUCCESS) << "cuvsHnswSearch failed";

  // verify output
  ASSERT_TRUE(cuvs::hostVecMatch(neighbors_exp, neighbors, cuvs::Compare<uint64_t>()));
  ASSERT_TRUE(cuvs::hostVecMatch(distances_exp, distances, cuvs::CompareApprox<float>(0.001f)));

  // cleanup
  cuvsHnswAceParamsDestroy(ace_params);
  cuvsHnswIndexParamsDestroy(hnsw_params);
  cuvsHnswSearchParamsDestroy(search_params);
  cuvsHnswIndexDestroy(hnsw_index);
  cuvsResourcesDestroy(res);
}

TEST(HnswAceDiskC, BuildSerializeDeserializeSearch)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = dataset;
  dataset_tensor.dl_tensor.device.device_type = kDLCPU;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {4, 2};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = nullptr;

  // create ACE params with use_disk = true
  cuvsHnswAceParams_t ace_params;
  cuvsHnswAceParamsCreate(&ace_params);
  ace_params->npartitions = 2;
  ace_params->build_dir   = "/tmp/hnsw_ace_disk_test";
  ace_params->use_disk    = true;

  // create index params
  cuvsHnswIndexParams_t hnsw_params;
  cuvsHnswIndexParamsCreate(&hnsw_params);
  hnsw_params->hierarchy       = GPU;
  hnsw_params->ef_construction = 100;
  hnsw_params->ace_params      = ace_params;
  hnsw_params->metric          = L2Expanded;
  hnsw_params->M               = 16;

  // create HNSW index
  cuvsHnswIndex_t hnsw_index;
  cuvsHnswIndexCreate(&hnsw_index);

  // build index using ACE with disk mode (index is serialized to disk by the build function)
  cuvsError_t build_status = cuvsHnswBuild(res, hnsw_params, &dataset_tensor, hnsw_index);
  ASSERT_EQ(build_status, CUVS_SUCCESS) << "cuvsHnswBuild failed";

  // create a new HNSW index for deserialization
  cuvsHnswIndex_t deserialized_index;
  cuvsHnswIndexCreate(&deserialized_index);
  deserialized_index->dtype = hnsw_index->dtype;

  // deserialize from disk
  cuvsError_t deserialize_status =
    cuvsHnswDeserialize(res, hnsw_params, "/tmp/hnsw_ace_disk_test/hnsw_index.bin", 2, L2Expanded, deserialized_index);
  ASSERT_EQ(deserialize_status, CUVS_SUCCESS) << "cuvsHnswDeserialize failed";

  // create queries DLTensor
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries;
  queries_tensor.dl_tensor.device.device_type = kDLCPU;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {4, 2};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  // create neighbors DLTensor
  std::vector<uint64_t> neighbors(4);
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCPU;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {4, 1};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  // create distances DLTensor
  std::vector<float> distances(4);
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances.data();
  distances_tensor.dl_tensor.device.device_type = kDLCPU;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {4, 1};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  // search the deserialized index
  cuvsHnswSearchParams_t search_params;
  cuvsHnswSearchParamsCreate(&search_params);
  search_params->ef = 100;

  cuvsError_t search_status = cuvsHnswSearch(
    res, search_params, deserialized_index, &queries_tensor, &neighbors_tensor, &distances_tensor);
  ASSERT_EQ(search_status, CUVS_SUCCESS) << "cuvsHnswSearch failed";

  // verify output
  ASSERT_TRUE(cuvs::hostVecMatch(neighbors_exp, neighbors, cuvs::Compare<uint64_t>()));
  ASSERT_TRUE(cuvs::hostVecMatch(distances_exp, distances, cuvs::CompareApprox<float>(0.001f)));

  // cleanup
  cuvsHnswAceParamsDestroy(ace_params);
  cuvsHnswIndexParamsDestroy(hnsw_params);
  cuvsHnswSearchParamsDestroy(search_params);
  cuvsHnswIndexDestroy(hnsw_index);
  cuvsHnswIndexDestroy(deserialized_index);
  cuvsResourcesDestroy(res);
}
