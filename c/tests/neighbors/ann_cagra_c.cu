/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.cuh"
#include <array>
#include <cstddef>
#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.hpp>
#include <dlpack/dlpack.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/hnsw.h>
#include <string>
#include <type_traits>
#include <unistd.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/math.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/matrix/argmin.cuh>
#include <raft/matrix/linewise_op.cuh>
#include <sys/types.h>

#include <raft/random/make_blobs.cuh>

float dataset[4][2] = {{0.74021935, 0.9209938},
                       {0.03902049, 0.9689629},
                       {0.92514056, 0.4463501},
                       {0.6673192, 0.10993068}};
float queries[4][2] = {{0.48216683, 0.0428398},
                       {0.5084142, 0.6545497},
                       {0.51260436, 0.2643005},
                       {0.05198065, 0.5789965}};

uint32_t filter[1] = {0b1001};  // index 1 and 2 are removed

uint32_t neighbors_exp[4] = {3, 0, 3, 1};
float distances_exp[4]    = {0.03878258, 0.12472608, 0.04776672, 0.15224178};

uint32_t neighbors_exp_filtered[4] = {3, 0, 3, 0};
float distances_exp_filtered[4]    = {0.03878258, 0.12472608, 0.04776672, 0.59063464};

std::vector<uint64_t> neighbors_exp_disk = {3, 0, 3, 1};
std::vector<float> distances_exp_disk    = {0.03878258, 0.12472608, 0.04776672, 0.15224178};

TEST(CagraC, BuildSearch)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

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
  cuvsDataset_t dataset_view;
  ASSERT_EQ(cuvsDatasetMakeStandardView(res, &dataset_tensor, &dataset_view), CUVS_SUCCESS);
  {
    bool is_owning = true;
    ASSERT_EQ(cuvsDatasetGetIsOwning(dataset_view, &is_owning), CUVS_SUCCESS);
    EXPECT_FALSE(is_owning);
  }
  ASSERT_EQ(cuvsCagraBuild(res, build_params, dataset_view, index), CUVS_SUCCESS);
  EXPECT_EQ(cuvsCagraUpdateDataset(res, dataset_view, index), CUVS_ERROR);

  // Host build yields a host index. Copy the host tensor into caller-owned device-padded storage
  // to produce a search-ready device-padded index.
  cuvsDataset_t padded_dataset_owner;
  ASSERT_EQ(cuvsDatasetMakePadded(
              res, &dataset_tensor, CUVS_DATASET_MEM_TYPE_DEVICE, &padded_dataset_owner),
            CUVS_SUCCESS);
  {
    bool is_owning = false;
    ASSERT_EQ(cuvsDatasetGetIsOwning(padded_dataset_owner, &is_owning), CUVS_SUCCESS);
    EXPECT_TRUE(is_owning);
  }
  cuvsCagraIndex_t owner_built_index;
  ASSERT_EQ(cuvsCagraIndexCreate(&owner_built_index), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraBuild(res, build_params, padded_dataset_owner, owner_built_index), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraIndexDestroy(owner_built_index), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraUpdateDataset(res, padded_dataset_owner, index), CUVS_SUCCESS);

  // create queries DLTensor
  rmm::device_uvector<float> queries_d(4 * 2, stream);
  raft::copy(queries_d.data(), (float*)queries, 4 * 2, stream);

  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries_d.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {4, 2};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  // create neighbors DLTensor
  rmm::device_uvector<uint32_t> neighbors_d(4, stream);

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_d.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 32;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {4, 1};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  // create distances DLTensor
  rmm::device_uvector<float> distances_d(4, stream);

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_d.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {4, 1};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = (uintptr_t)NULL;

  // search index
  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);
  ASSERT_EQ(cuvsCagraSearch(
              res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor, filter),
            CUVS_SUCCESS);

  // verify output
  ASSERT_TRUE(
    cuvs::devArrMatchHost(neighbors_exp, neighbors_d.data(), 4, cuvs::Compare<uint32_t>()));
  ASSERT_TRUE(cuvs::devArrMatchHost(
    distances_exp, distances_d.data(), 4, cuvs::CompareApprox<float>(0.001f)));

  // de-allocate index and res
  cuvsCagraSearchParamsDestroy(search_params);
  cuvsDatasetDestroy(dataset_view);
  cuvsDatasetDestroy(padded_dataset_owner);
  cuvsCagraIndexParamsDestroy(build_params);
  cuvsCagraIndexDestroy(index);
  cuvsResourcesDestroy(res);
}

// CAGRA operations that need a search-ready index must reject host / non-device-padded
// datasets rather than succeeding and producing undefined behavior.
TEST(CagraC, DatasetContractFailures)
{
  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);
  cudaStream_t stream;
  ASSERT_EQ(cuvsStreamGet(res, &stream), CUVS_SUCCESS);

  DLManagedTensor host_tensor{};
  host_tensor.dl_tensor.data               = dataset;
  host_tensor.dl_tensor.device.device_type = kDLCPU;
  host_tensor.dl_tensor.ndim               = 2;
  host_tensor.dl_tensor.dtype              = {kDLFloat, 32, 1};
  int64_t host_shape[2]                    = {4, 2};
  host_tensor.dl_tensor.shape              = host_shape;

  cuvsCagraIndexParams_t build_params;
  ASSERT_EQ(cuvsCagraIndexParamsCreate(&build_params), CUVS_SUCCESS);
  cuvsCagraIndex_t host_index;
  ASSERT_EQ(cuvsCagraIndexCreate(&host_index), CUVS_SUCCESS);
  cuvsDataset_t host_standard_view;
  ASSERT_EQ(cuvsDatasetMakeStandardView(res, &host_tensor, &host_standard_view), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraBuild(res, build_params, host_standard_view, host_index), CUVS_SUCCESS);

  // UpdateDataset requires a device-padded dataset.
  EXPECT_EQ(cuvsCagraUpdateDataset(res, host_standard_view, host_index), CUVS_ERROR);

  cuvsDataset_t host_padded_owner;
  ASSERT_EQ(cuvsDatasetMakePadded(
              res, &host_tensor, CUVS_DATASET_MEM_TYPE_HOST, &host_padded_owner),
            CUVS_SUCCESS);
  EXPECT_EQ(cuvsCagraUpdateDataset(res, host_padded_owner, host_index), CUVS_ERROR);

  rmm::device_uvector<float> queries_d(8, stream);
  raft::copy(queries_d.data(), reinterpret_cast<float*>(queries), 8, stream);
  DLManagedTensor queries_tensor{};
  queries_tensor.dl_tensor.data               = queries_d.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype              = {kDLFloat, 32, 1};
  int64_t queries_shape[2]                    = {4, 2};
  queries_tensor.dl_tensor.shape              = queries_shape;

  rmm::device_uvector<uint32_t> neighbors_d(4, stream);
  DLManagedTensor neighbors_tensor{};
  neighbors_tensor.dl_tensor.data               = neighbors_d.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype              = {kDLUInt, 32, 1};
  int64_t neighbors_shape[2]                    = {4, 1};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;

  rmm::device_uvector<float> distances_d(4, stream);
  DLManagedTensor distances_tensor{};
  distances_tensor.dl_tensor.data               = distances_d.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype              = {kDLFloat, 32, 1};
  int64_t distances_shape[2]                    = {4, 1};
  distances_tensor.dl_tensor.shape              = distances_shape;

  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = 0;
  cuvsCagraSearchParams_t search_params;
  ASSERT_EQ(cuvsCagraSearchParamsCreate(&search_params), CUVS_SUCCESS);

  EXPECT_EQ(cuvsCagraSearch(res,
                            search_params,
                            host_index,
                            &queries_tensor,
                            &neighbors_tensor,
                            &distances_tensor,
                            filter),
            CUVS_ERROR);
  // hnswlib search runs on the host, so a host index serializes straight from its own host
  // vectors -- no device-padded dataset required.
  EXPECT_EQ(cuvsCagraSerializeToHnswlib(res, "/tmp/cagra_host_index.hnsw", host_index),
            CUVS_SUCCESS);
  std::remove("/tmp/cagra_host_index.hnsw");

  // Device-standard indexes are mergeable but not extendable; extend needs device-padded.
  rmm::device_uvector<float> device_dataset(8, stream);
  raft::copy(device_dataset.data(), reinterpret_cast<float*>(dataset), 8, stream);
  DLManagedTensor device_tensor       = host_tensor;
  device_tensor.dl_tensor.data        = device_dataset.data();
  device_tensor.dl_tensor.device.device_type = kDLCUDA;

  cuvsCagraIndex_t device_standard_index;
  ASSERT_EQ(cuvsCagraIndexCreate(&device_standard_index), CUVS_SUCCESS);
  cuvsDataset_t device_standard_view;
  ASSERT_EQ(cuvsDatasetMakeStandardView(res, &device_tensor, &device_standard_view), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraBuild(res, build_params, device_standard_view, device_standard_index),
            CUVS_SUCCESS);

  cuvsDataset_t device_padded_owner;
  ASSERT_EQ(cuvsDatasetMakePadded(
              res, &device_tensor, CUVS_DATASET_MEM_TYPE_DEVICE, &device_padded_owner),
            CUVS_SUCCESS);
  cuvsCagraExtendParams_t extend_params;
  ASSERT_EQ(cuvsCagraExtendParamsCreate(&extend_params), CUVS_SUCCESS);
  EXPECT_EQ(
    cuvsCagraExtend(res, extend_params, device_padded_owner, 4, device_standard_index),
    CUVS_ERROR);

  // Host indexes are not mergeable.
  cuvsCagraIndex_t host_index_2;
  ASSERT_EQ(cuvsCagraIndexCreate(&host_index_2), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraBuild(res, build_params, host_standard_view, host_index_2), CUVS_SUCCESS);
  cuvsCagraIndex_t merge_out;
  ASSERT_EQ(cuvsCagraIndexCreate(&merge_out), CUVS_SUCCESS);
  cuvsDataset_t merged_dataset;
  ASSERT_EQ(cuvsDatasetCreate(&merged_dataset), CUVS_SUCCESS);
  cuvsCagraIndex_t host_indices[2] = {host_index, host_index_2};
  EXPECT_EQ(
    cuvsCagraMerge(res, build_params, host_indices, 2, filter, merged_dataset, merge_out),
    CUVS_ERROR);

  ASSERT_EQ(cuvsCagraExtendParamsDestroy(extend_params), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraSearchParamsDestroy(search_params), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetDestroy(host_standard_view), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetDestroy(host_padded_owner), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetDestroy(device_standard_view), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetDestroy(device_padded_owner), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetDestroy(merged_dataset), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraIndexParamsDestroy(build_params), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraIndexDestroy(host_index), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraIndexDestroy(host_index_2), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraIndexDestroy(device_standard_index), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraIndexDestroy(merge_out), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

TEST(CagraC, UpdateHostPadded)
{
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

  float host_dataset[16]  = {0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 3, 0, 0};
  int64_t dataset_shape[] = {4, 4};
  DLManagedTensor host_tensor{};
  host_tensor.dl_tensor.data               = host_dataset;
  host_tensor.dl_tensor.device.device_type = kDLCPU;
  host_tensor.dl_tensor.ndim               = 2;
  host_tensor.dl_tensor.dtype              = {kDLFloat, 32, 1};
  host_tensor.dl_tensor.shape              = dataset_shape;

  cuvsDataset_t host_view;
  ASSERT_EQ(cuvsDatasetMakePaddedView(res, &host_tensor, &host_view), CUVS_SUCCESS);
  float host_standard_dataset[8] = {0, 0, 1, 1, 2, 2, 3, 3};
  int64_t standard_shape[]       = {4, 2};
  DLManagedTensor host_standard_tensor = host_tensor;
  host_standard_tensor.dl_tensor.data  = host_standard_dataset;
  host_standard_tensor.dl_tensor.shape = standard_shape;
  cuvsDataset_t host_owner;
  ASSERT_EQ(cuvsDatasetMakePadded(
              res, &host_standard_tensor, CUVS_DATASET_MEM_TYPE_HOST, &host_owner),
            CUVS_SUCCESS);
  {
    cuvsDatasetMemType_t mem_type{};
    ASSERT_EQ(cuvsDatasetGetMemType(host_owner, &mem_type), CUVS_SUCCESS);
    EXPECT_EQ(mem_type, CUVS_DATASET_MEM_TYPE_HOST);
  }
  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);
  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);
  ASSERT_EQ(cuvsCagraBuild(res, build_params, host_view, index), CUVS_SUCCESS);

  auto serialized_path =
    "/tmp/cuvs-cagra-host-padded-" + std::to_string(getpid()) + ".bin";
  ASSERT_EQ(cuvsCagraSerializeGraphAndDataset(res, serialized_path.c_str(), index), CUVS_SUCCESS);
  cuvsCagraIndex_t loaded_index;
  ASSERT_EQ(cuvsCagraIndexCreate(&loaded_index), CUVS_SUCCESS);
  cuvsDataset_t loaded_dataset = nullptr;
  ASSERT_EQ(cuvsCagraDeserializeGraphAndDataset(
              res, serialized_path.c_str(), loaded_index, &loaded_dataset),
            CUVS_SUCCESS)
    << cuvsGetLastErrorText();
  ASSERT_NE(loaded_dataset, nullptr);
  {
    cuvsDatasetMemType_t mem_type{};
    cuvsDatasetLayout_t layout{};
    ASSERT_EQ(cuvsDatasetGetMemType(loaded_dataset, &mem_type), CUVS_SUCCESS);
    ASSERT_EQ(cuvsDatasetGetLayout(loaded_dataset, &layout), CUVS_SUCCESS);
    EXPECT_EQ(mem_type, CUVS_DATASET_MEM_TYPE_HOST);
    EXPECT_EQ(layout, CUVS_DATASET_LAYOUT_PADDED);
  }
  cuvsCagraIndexDestroy(loaded_index);
  cuvsDatasetDestroy(loaded_dataset);
  std::filesystem::remove(serialized_path);

  rmm::device_uvector<float> device_dataset(16, stream);
  raft::copy(device_dataset.data(), host_dataset, 16, stream);
  DLManagedTensor device_tensor              = host_tensor;
  device_tensor.dl_tensor.data               = device_dataset.data();
  device_tensor.dl_tensor.device.device_type = kDLCUDA;
  cuvsDataset_t device_to_host_owner;
  ASSERT_EQ(cuvsDatasetMakePadded(
              res, &device_tensor, CUVS_DATASET_MEM_TYPE_HOST, &device_to_host_owner),
            CUVS_SUCCESS);
  {
    cuvsDatasetMemType_t mem_type{};
    cuvsDatasetLayout_t layout{};
    ASSERT_EQ(cuvsDatasetGetMemType(device_to_host_owner, &mem_type), CUVS_SUCCESS);
    ASSERT_EQ(cuvsDatasetGetLayout(device_to_host_owner, &layout), CUVS_SUCCESS);
    EXPECT_EQ(mem_type, CUVS_DATASET_MEM_TYPE_HOST);
    EXPECT_EQ(layout, CUVS_DATASET_LAYOUT_PADDED);
  }
  cuvsDataset_t device_view;
  ASSERT_EQ(cuvsDatasetMakePaddedView(res, &device_tensor, &device_view), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraUpdateDataset(res, device_view, index), CUVS_SUCCESS);

  auto device_serialized_path =
    "/tmp/cuvs-cagra-device-padded-" + std::to_string(getpid()) + ".bin";
  ASSERT_EQ(
    cuvsCagraSerializeGraphAndDataset(res, device_serialized_path.c_str(), index), CUVS_SUCCESS);
  cuvsCagraIndex_t loaded_device_index;
  ASSERT_EQ(cuvsCagraIndexCreate(&loaded_device_index), CUVS_SUCCESS);
  cuvsDataset_t loaded_device_dataset = nullptr;
  ASSERT_EQ(cuvsCagraDeserializeGraphAndDataset(
              res, device_serialized_path.c_str(), loaded_device_index, &loaded_device_dataset),
            CUVS_SUCCESS)
    << cuvsGetLastErrorText();
  ASSERT_NE(loaded_device_dataset, nullptr);
  {
    cuvsDatasetMemType_t mem_type{};
    cuvsDatasetLayout_t layout{};
    ASSERT_EQ(cuvsDatasetGetMemType(loaded_device_dataset, &mem_type), CUVS_SUCCESS);
    ASSERT_EQ(cuvsDatasetGetLayout(loaded_device_dataset, &layout), CUVS_SUCCESS);
    EXPECT_EQ(mem_type, CUVS_DATASET_MEM_TYPE_DEVICE);
    EXPECT_EQ(layout, CUVS_DATASET_LAYOUT_PADDED);
  }

  cuvsCagraIndexDestroy(loaded_device_index);
  cuvsDatasetDestroy(loaded_device_dataset);
  std::filesystem::remove(device_serialized_path);

  cuvsDatasetDestroy(device_view);
  cuvsDatasetDestroy(host_view);
  cuvsDatasetDestroy(host_owner);
  cuvsDatasetDestroy(device_to_host_owner);
  cuvsCagraIndexDestroy(index);
  cuvsCagraIndexParamsDestroy(build_params);
  cuvsResourcesDestroy(res);
}

TEST(CagraC, BuildExtendSearch)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

  raft::resources handle;

  const int32_t dimensions = 16;
  // main_data_size needs to be >= 128 (see issue #486)
  const int32_t main_data_size       = 1024;
  const int32_t additional_data_size = 64;
  const int32_t num_queries          = 4;

  // create random data for datasets and queries
  rmm::device_uvector<float> random_data_d(
    (main_data_size + additional_data_size + num_queries) * dimensions, stream);
  rmm::device_uvector<int32_t> random_labels_d(
    (main_data_size + additional_data_size + num_queries) * dimensions, stream);

  raft::random::make_blobs<float, int32_t>(
    random_data_d.data(),
    random_labels_d.data(),
    main_data_size + additional_data_size + num_queries,
    dimensions,
    static_cast<int32_t>(10),
    stream,
    true,
    nullptr,
    nullptr,
    1.0f,
    true,
    -10.0f,
    10.0f,
    42ULL,
    raft::random::GenPC);

  // create  dataset DLTensor
  rmm::device_uvector<float> main_d(main_data_size * dimensions, stream);
  rmm::device_uvector<int32_t> main_labels_d(main_data_size, stream);
  raft::copy(main_d.data(), random_data_d.data(), main_data_size * dimensions, stream);
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = main_d.data();
  dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {main_data_size, dimensions};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = nullptr;

  // create additional dataset (concatenated into extended_dataset below)
  rmm::device_uvector<float> additional_d(additional_data_size * dimensions, stream);
  raft::copy(additional_d.data(),
             random_data_d.data() + main_d.size(),
             additional_data_size * dimensions,
             stream);

  // create index
  cuvsCagraIndex_t index;
  cuvsCagraIndexCreate(&index);

  // build index
  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);
  cuvsDataset_t dataset_view;
  ASSERT_EQ(cuvsDatasetMakePaddedView(res, &dataset_tensor, &dataset_view), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraBuild(res, build_params, dataset_view, index), CUVS_SUCCESS);

  cuvsStreamSync(res);

  // Extend requires a device-padded extended dataset; a standard view must fail.
  {
    cuvsDataset_t standard_extended;
    ASSERT_EQ(cuvsDatasetMakeStandardView(res, &dataset_tensor, &standard_extended), CUVS_SUCCESS);
    cuvsCagraExtendParams_t bad_extend_params;
    ASSERT_EQ(cuvsCagraExtendParamsCreate(&bad_extend_params), CUVS_SUCCESS);
    EXPECT_EQ(cuvsCagraExtend(res, bad_extend_params, standard_extended, main_data_size, index),
              CUVS_ERROR);
    ASSERT_EQ(cuvsCagraExtendParamsDestroy(bad_extend_params), CUVS_SUCCESS);
    ASSERT_EQ(cuvsDatasetDestroy(standard_extended), CUVS_SUCCESS);
  }

  // extend index — caller concatenates old || new into extended_dataset first
  cuvsCagraExtendParams_t extend_params;
  cuvsCagraExtendParamsCreate(&extend_params);
  rmm::device_uvector<float> extended_d((main_data_size + additional_data_size) * dimensions, stream);
  raft::copy(extended_d.data(), main_d.data(), main_data_size * dimensions, stream);
  raft::copy(extended_d.data() + main_data_size * dimensions,
             additional_d.data(),
             additional_data_size * dimensions,
             stream);
  DLManagedTensor extended_dataset_tensor;
  extended_dataset_tensor.dl_tensor.data               = extended_d.data();
  extended_dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  extended_dataset_tensor.dl_tensor.ndim               = 2;
  extended_dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  extended_dataset_tensor.dl_tensor.dtype.bits         = 32;
  extended_dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t extended_dataset_shape[2]                    = {main_data_size + additional_data_size,
                                                           dimensions};
  extended_dataset_tensor.dl_tensor.shape              = extended_dataset_shape;
  extended_dataset_tensor.dl_tensor.strides            = nullptr;
  cuvsDataset_t extended_dataset_view;
  ASSERT_EQ(cuvsDatasetMakePaddedView(res, &extended_dataset_tensor, &extended_dataset_view),
            CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraExtend(res, extend_params, extended_dataset_view, main_data_size, index),
            CUVS_SUCCESS);

  // create queries DLTensor
  rmm::device_uvector<float> queries_d(num_queries * dimensions, stream);
  raft::copy(queries_d.data(),
             random_data_d.data() + (main_data_size + additional_data_size) * dimensions,
             num_queries * dimensions,
             stream);
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries_d.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {4, dimensions};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  // create pairwise distance matrix for dataset and queries
  auto pairwise_distance_dataset_input =
    raft::make_device_matrix<float>(handle, main_data_size + additional_data_size, dimensions);

  raft::copy(pairwise_distance_dataset_input.data_handle(), main_d.data(), main_d.size(), stream);
  raft::copy(pairwise_distance_dataset_input.data_handle() + main_d.size(),
             additional_d.data(),
             additional_d.size(),
             stream);

  auto pairwise_distance_queries_input =
    raft::make_device_matrix<float>(handle, num_queries, dimensions);

  raft::copy(pairwise_distance_queries_input.data_handle(),
             (float*)queries_d.data(),
             num_queries * dimensions,
             stream);

  auto pairwise_distances =
    raft::make_device_matrix<float>(handle, num_queries, (main_data_size + additional_data_size));
  auto metric = cuvs::distance::DistanceType::L2Expanded;

  cuvs::distance::pairwise_distance(handle,
                                    pairwise_distance_queries_input.view(),
                                    pairwise_distance_dataset_input.view(),

                                    pairwise_distances.view(),
                                    metric);

  auto min_cols =
    raft::make_device_vector<uint32_t, uint32_t>(handle, pairwise_distances.extent(0));

  auto distances_const_view = raft::make_device_matrix_view<const float, uint32_t>(
    pairwise_distances.data_handle(), pairwise_distances.extent(0), pairwise_distances.extent(1));

  raft::matrix::argmin(handle, distances_const_view, min_cols.view());

  float min_cols_distances[num_queries];

  for (uint32_t i = 0; i < min_cols.extent(0); i++) {
    uint32_t mc           = min_cols(i);
    min_cols_distances[i] = pairwise_distances(i, mc);
  }

  // create neighbors DLTensor
  rmm::device_uvector<uint32_t> neighbors_d(4, stream);

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_d.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 32;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {num_queries, 1};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  // create distances DLTensor
  rmm::device_uvector<float> distances_d(4, stream);

  distances_d.resize(4, stream);

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_d.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {num_queries, 1};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = (uintptr_t)NULL;

  // search index
  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);
  ASSERT_EQ(cuvsCagraSearch(
              res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor, filter),
            CUVS_SUCCESS);

  ASSERT_TRUE(
    cuvs::devArrMatch(min_cols.data_handle(), neighbors_d.data(), 4, cuvs::Compare<uint32_t>()));

  // check distances
  ASSERT_TRUE(cuvs::devArrMatchHost(
    min_cols_distances, distances_d.data(), 4, cuvs::CompareApprox<float>(0.001f)));

  // de-allocate index and res
  cuvsCagraSearchParamsDestroy(search_params);
  cuvsDatasetDestroy(dataset_view);
  cuvsDatasetDestroy(extended_dataset_view);
  cuvsCagraExtendParamsDestroy(extend_params);
  cuvsCagraIndexParamsDestroy(build_params);
  cuvsCagraIndexDestroy(index);
  cuvsResourcesDestroy(res);
}

TEST(CagraC, BuildSearchFiltered)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

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
  cuvsDataset_t dataset_view;
  ASSERT_EQ(cuvsDatasetMakeStandardView(res, &dataset_tensor, &dataset_view), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraBuild(res, build_params, dataset_view, index), CUVS_SUCCESS);

  // Host build yields a host index. Attach a caller-provided device padded dataset
  // to produce a search-ready device padded index.
  rmm::device_uvector<float> dataset_d(4 * 2, stream);
  raft::copy(dataset_d.data(), (float*)dataset, 4 * 2, stream);
  DLManagedTensor device_dataset_tensor = dataset_tensor;
  device_dataset_tensor.dl_tensor.data               = dataset_d.data();
  device_dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  device_dataset_tensor.dl_tensor.device.device_id   = 0;
  cuvsDataset_t padded_dataset_owner;
  ASSERT_EQ(cuvsDatasetMakePadded(
              res, &device_dataset_tensor, CUVS_DATASET_MEM_TYPE_DEVICE, &padded_dataset_owner),
            CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraUpdateDataset(res, padded_dataset_owner, index), CUVS_SUCCESS);

  // create queries DLTensor
  rmm::device_uvector<float> queries_d(4 * 2, stream);
  raft::copy(queries_d.data(), (float*)queries, 4 * 2, stream);

  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries_d.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {4, 2};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  // create neighbors DLTensor
  rmm::device_uvector<uint32_t> neighbors_d(4, stream);

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_d.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 32;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {4, 1};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  // create distances DLTensor
  rmm::device_uvector<float> distances_d(4, stream);

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_d.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {4, 1};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  // create filter DLTensor
  rmm::device_uvector<uint32_t> filter_d(1, stream);
  raft::copy(filter_d.data(), filter, 1, stream);

  cuvsFilter filter;

  DLManagedTensor filter_tensor;
  filter_tensor.dl_tensor.data               = filter_d.data();
  filter_tensor.dl_tensor.device.device_type = kDLCUDA;
  filter_tensor.dl_tensor.ndim               = 1;
  filter_tensor.dl_tensor.dtype.code         = kDLUInt;
  filter_tensor.dl_tensor.dtype.bits         = 32;
  filter_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t filter_shape[1]                    = {1};
  filter_tensor.dl_tensor.shape              = filter_shape;
  filter_tensor.dl_tensor.strides            = nullptr;

  filter.type = BITSET;
  filter.addr = (uintptr_t)&filter_tensor;

  // search index
  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);
  ASSERT_EQ(cuvsCagraSearch(
              res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor, filter),
            CUVS_SUCCESS);
  // verify output
  ASSERT_TRUE(cuvs::devArrMatchHost(
    neighbors_exp_filtered, neighbors_d.data(), 4, cuvs::Compare<uint32_t>()));
  ASSERT_TRUE(cuvs::devArrMatchHost(
    distances_exp_filtered, distances_d.data(), 4, cuvs::CompareApprox<float>(0.001f)));

  // de-allocate index and res
  cuvsCagraSearchParamsDestroy(search_params);
  cuvsDatasetDestroy(dataset_view);
  cuvsDatasetDestroy(padded_dataset_owner);
  cuvsCagraIndexParamsDestroy(build_params);
  cuvsCagraIndexDestroy(index);
  cuvsResourcesDestroy(res);
}

TEST(CagraC, BuildMergeSearch)
{
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

  float dataset[7][2] = {{0.74021935f, 0.92099380f},
                         {0.03902049f, 0.96896291f},
                         {0.92514056f, 0.44635010f},
                         {0.12345678f, 0.87654321f},
                         {0.50112233f, 0.33221100f},
                         {0.66731918f, 0.10993068f},
                         {0.77777777f, 0.88888888f}};

  float* main_data_ptr       = &dataset[0][0];
  float* additional_data_ptr = &dataset[4][0];
  float* query_data_ptr      = &dataset[6][0];

  rmm::device_uvector<float> main_d(8, stream);
  rmm::device_uvector<float> additional_d(6, stream);
  rmm::device_uvector<float> queries_d(2, stream);
  raft::copy(main_d.data(), main_data_ptr, 8, stream);
  raft::copy(additional_d.data(), additional_data_ptr, 6, stream);
  raft::copy(queries_d.data(), query_data_ptr, 2, stream);

  DLManagedTensor main_dataset_tensor;
  int64_t main_shape[2]                            = {4, 2};
  main_dataset_tensor.dl_tensor.data               = main_d.data();
  main_dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  main_dataset_tensor.dl_tensor.device.device_id   = 0;
  main_dataset_tensor.dl_tensor.ndim               = 2;
  main_dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  main_dataset_tensor.dl_tensor.dtype.bits         = 32;
  main_dataset_tensor.dl_tensor.dtype.lanes        = 1;
  main_dataset_tensor.dl_tensor.shape              = main_shape;
  main_dataset_tensor.dl_tensor.strides            = nullptr;

  DLManagedTensor additional_dataset_tensor = main_dataset_tensor;
  int64_t additional_shape[2]               = {3, 2};
  additional_dataset_tensor.dl_tensor.data  = additional_d.data();
  additional_dataset_tensor.dl_tensor.shape = additional_shape;

  DLManagedTensor query_tensor = main_dataset_tensor;
  int64_t query_shape[2]       = {1, 2};
  query_tensor.dl_tensor.data  = queries_d.data();
  query_tensor.dl_tensor.shape = query_shape;

  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);
  cuvsCagraIndex_t index_main, index_add;
  cuvsCagraIndexCreate(&index_main);
  cuvsCagraIndexCreate(&index_add);
  cuvsDataset_t main_dataset_view;
  cuvsDataset_t additional_dataset_view;
  ASSERT_EQ(cuvsDatasetMakeStandardView(res, &main_dataset_tensor, &main_dataset_view),
            CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetMakeStandardView(
              res, &additional_dataset_tensor, &additional_dataset_view),
            CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraBuild(res, build_params, main_dataset_view, index_main),
            CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraBuild(
              res, build_params, additional_dataset_view, index_add),
            CUVS_SUCCESS);
  EXPECT_EQ(cuvsCagraUpdateDataset(res, main_dataset_view, index_main), CUVS_ERROR);

  cuvsCagraIndex_t index_merged;
  cuvsCagraIndexCreate(&index_merged);

  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = 0;

  cuvsCagraIndex_t index_array[2] = {index_main, index_add};
  cuvsDataset_t merged_dataset;
  ASSERT_EQ(cuvsDatasetCreate(&merged_dataset), CUVS_SUCCESS);
  cuvsCagraMergeParams_t merge_params;
  ASSERT_EQ(cuvsCagraMergeParamsCreate(&merge_params), CUVS_SUCCESS);
  EXPECT_EQ(merge_params->algo, CUVS_CAGRA_MERGE_AUTO);
  merge_params->algo = CUVS_CAGRA_MERGE_REBUILD;
  ASSERT_EQ(cuvsCagraMergeWithParams(
              res, build_params, merge_params, index_array, 2, filter, merged_dataset, index_merged),
            CUVS_SUCCESS);
  {
    cuvsDatasetMemType_t mem_type{};
    cuvsDatasetLayout_t layout{};
    ASSERT_EQ(cuvsDatasetGetMemType(merged_dataset, &mem_type), CUVS_SUCCESS);
    ASSERT_EQ(cuvsDatasetGetLayout(merged_dataset, &layout), CUVS_SUCCESS);
    EXPECT_EQ(layout, CUVS_DATASET_LAYOUT_STANDARD);
    EXPECT_EQ(mem_type, CUVS_DATASET_MEM_TYPE_DEVICE);
  }

  // Merge of standard-layout device inputs yields a standard index. Under the explicit C API
  // contract, attach a padded dataset before calling search.
  rmm::device_uvector<float> merged_d(14, stream);
  raft::copy(merged_d.data(), main_d.data(), main_d.size(), stream);
  raft::copy(merged_d.data() + main_d.size(), additional_d.data(), additional_d.size(), stream);

  DLManagedTensor merged_dataset_tensor;
  int64_t merged_shape[2]                            = {7, 2};
  merged_dataset_tensor.dl_tensor.data               = merged_d.data();
  merged_dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  merged_dataset_tensor.dl_tensor.device.device_id   = 0;
  merged_dataset_tensor.dl_tensor.ndim               = 2;
  merged_dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  merged_dataset_tensor.dl_tensor.dtype.bits         = 32;
  merged_dataset_tensor.dl_tensor.dtype.lanes        = 1;
  merged_dataset_tensor.dl_tensor.shape              = merged_shape;
  merged_dataset_tensor.dl_tensor.strides            = nullptr;

  cuvsDataset_t padded_dataset_owner;
  ASSERT_EQ(cuvsDatasetMakePadded(
              res, &merged_dataset_tensor, CUVS_DATASET_MEM_TYPE_DEVICE, &padded_dataset_owner),
            CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraUpdateDataset(res, padded_dataset_owner, index_merged), CUVS_SUCCESS);

  int64_t merged_dim = -1;
  ASSERT_EQ(cuvsCagraIndexGetDims(index_merged, &merged_dim), CUVS_SUCCESS);
  EXPECT_EQ(merged_dim, 2);

  DLManagedTensor neighbors_tensor, distances_tensor;
  rmm::device_uvector<int64_t> neighbors_d(1, stream);
  rmm::device_uvector<float> distances_d(1, stream);
  int64_t neighbors_shape[2]             = {1, 1};
  int64_t distances_shape[2]             = {1, 1};
  neighbors_tensor.dl_tensor.data        = neighbors_d.data();
  neighbors_tensor.dl_tensor.device      = main_dataset_tensor.dl_tensor.device;
  neighbors_tensor.dl_tensor.ndim        = 2;
  neighbors_tensor.dl_tensor.dtype.code  = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits  = 64;
  neighbors_tensor.dl_tensor.dtype.lanes = 1;
  neighbors_tensor.dl_tensor.shape       = neighbors_shape;
  neighbors_tensor.dl_tensor.strides     = nullptr;
  distances_tensor.dl_tensor.data        = distances_d.data();
  distances_tensor.dl_tensor.device      = main_dataset_tensor.dl_tensor.device;
  distances_tensor.dl_tensor.ndim        = 2;
  distances_tensor.dl_tensor.dtype.code  = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits  = 32;
  distances_tensor.dl_tensor.dtype.lanes = 1;
  distances_tensor.dl_tensor.shape       = distances_shape;
  distances_tensor.dl_tensor.strides     = nullptr;

  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);
  (*search_params).itopk_size = 1;

  ASSERT_EQ(cuvsCagraSearch(res,
                            search_params,
                            index_merged,
                            &query_tensor,
                            &neighbors_tensor,
                            &distances_tensor,
                            filter),
            CUVS_SUCCESS);

  int64_t neighbor_host = -1;
  float distance_host   = 1.0f;
  raft::copy(&neighbor_host, neighbors_d.data(), 1, stream);
  raft::copy(&distance_host, distances_d.data(), 1, stream);
  cudaStreamSynchronize(stream);

  EXPECT_EQ(neighbor_host, 6);
  EXPECT_NEAR(distance_host, 0.0f, 1e-6);

  cuvsCagraSearchParamsDestroy(search_params);
  cuvsCagraMergeParamsDestroy(merge_params);
  cuvsCagraIndexParamsDestroy(build_params);
  cuvsCagraIndexDestroy(index_merged);
  cuvsCagraIndexDestroy(index_add);
  cuvsCagraIndexDestroy(index_main);
  cuvsDatasetDestroy(padded_dataset_owner);
  cuvsDatasetDestroy(additional_dataset_view);
  cuvsDatasetDestroy(main_dataset_view);
  cuvsDatasetDestroy(merged_dataset);
  cuvsResourcesDestroy(res);
}

TEST(CagraC, BuildSearchACEMemory)
{
  // create cuvsResources_t
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

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

  // build index with ACE memory mode
  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);
  build_params->build_algo = ACE;

  cuvsAceParams_t ace_params;
  cuvsAceParamsCreate(&ace_params);
  ace_params->npartitions = 2;
  ace_params->ef_construction = 120;
  ace_params->use_disk = false;

  build_params->graph_build_params = ace_params;
  cuvsDataset_t dataset_view;
  ASSERT_EQ(cuvsDatasetMakeStandardView(res, &dataset_tensor, &dataset_view), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraBuild(res, build_params, dataset_view, index), CUVS_SUCCESS);

  // Host build yields a host index. Attach a caller-provided device padded dataset
  // to produce a search-ready device padded index.
  rmm::device_uvector<float> dataset_d(4 * 2, stream);
  raft::copy(dataset_d.data(), (float*)dataset, 4 * 2, stream);
  DLManagedTensor device_dataset_tensor = dataset_tensor;
  device_dataset_tensor.dl_tensor.data               = dataset_d.data();
  device_dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
  device_dataset_tensor.dl_tensor.device.device_id   = 0;
  cuvsDataset_t padded_dataset_owner;
  ASSERT_EQ(cuvsDatasetMakePadded(
              res, &device_dataset_tensor, CUVS_DATASET_MEM_TYPE_DEVICE, &padded_dataset_owner),
            CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraUpdateDataset(res, padded_dataset_owner, index), CUVS_SUCCESS);

  // create queries DLTensor
  rmm::device_uvector<float> queries_d(4 * 2, stream);
  raft::copy(queries_d.data(), (float*)queries, 4 * 2, stream);

  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries_d.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {4, 2};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  // create neighbors DLTensor
  rmm::device_uvector<uint32_t> neighbors_d(4, stream);

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_d.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 32;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {4, 1};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  // create distances DLTensor
  rmm::device_uvector<float> distances_d(4, stream);

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_d.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {4, 1};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = (uintptr_t)NULL;

  // search index
  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);
  ASSERT_EQ(cuvsCagraSearch(
              res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor, filter),
            CUVS_SUCCESS);

  // verify output
  ASSERT_TRUE(
    cuvs::devArrMatchHost(neighbors_exp, neighbors_d.data(), 4, cuvs::Compare<uint32_t>()));
  ASSERT_TRUE(cuvs::devArrMatchHost(
    distances_exp, distances_d.data(), 4, cuvs::CompareApprox<float>(0.001f)));

  // de-allocate index and res
  cuvsCagraSearchParamsDestroy(search_params);
  cuvsDatasetDestroy(dataset_view);
  cuvsDatasetDestroy(padded_dataset_owner);
  cuvsCagraIndexParamsDestroy(build_params);
  cuvsCagraIndexDestroy(index);
  cuvsResourcesDestroy(res);
}

TEST(CagraC, BuildSearchACEDisk)
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

  // build index with ACE memory mode
  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);
  build_params->build_algo = ACE;

  cuvsAceParams_t ace_params;
  cuvsAceParamsCreate(&ace_params);
  ace_params->npartitions = 2;
  ace_params->ef_construction = 120;
  ace_params->use_disk = true;
  ace_params->build_dir = strdup("/tmp/cagra_ace_test_disk");

  build_params->graph_build_params = ace_params;
  cuvsDataset_t dataset_view;
  ASSERT_EQ(cuvsDatasetMakeStandardView(res, &dataset_tensor, &dataset_view), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraBuild(res, build_params, dataset_view, index), CUVS_SUCCESS);

  // Convert CAGRA index to HNSW (automatically serializes to disk for ACE)
  cuvsHnswIndex_t hnsw_index_ser;
  cuvsHnswIndexCreate(&hnsw_index_ser);
  cuvsHnswIndexParams_t hnsw_params;
  cuvsHnswIndexParamsCreate(&hnsw_params);

  cuvsHnswFromCagra(res, hnsw_params, index, hnsw_index_ser);
  ASSERT_NE(hnsw_index_ser->addr, 0);
  cuvsHnswIndexDestroy(hnsw_index_ser);

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

  // Deserialize the HNSW index from disk for search
  cuvsHnswIndex_t hnsw_index;
  cuvsHnswIndexCreate(&hnsw_index);
  hnsw_index->dtype = index->dtype;

  // Use the actual dimension from the dataset
  int dim = dataset_tensor.dl_tensor.shape[1];
  cuvsHnswDeserialize(res, hnsw_params, "/tmp/cagra_ace_test_disk/hnsw_index.bin", dim, L2Expanded, hnsw_index);
  ASSERT_NE(hnsw_index->addr, 0);

  // Search the HNSW index
  cuvsHnswSearchParams_t search_params;
  cuvsHnswSearchParamsCreate(&search_params);
  cuvsHnswSearch(
    res, search_params, hnsw_index, &queries_tensor, &neighbors_tensor, &distances_tensor);

  // Verify output
  ASSERT_TRUE(cuvs::hostVecMatch(neighbors_exp_disk, neighbors, cuvs::Compare<uint64_t>()));
  ASSERT_TRUE(cuvs::hostVecMatch(distances_exp_disk, distances, cuvs::CompareApprox<float>(0.001f)));

  cuvsCagraIndexParamsDestroy(build_params);
  cuvsDatasetDestroy(dataset_view);
  cuvsCagraIndexDestroy(index);
  cuvsHnswSearchParamsDestroy(search_params);
  cuvsHnswIndexParamsDestroy(hnsw_params);
  cuvsHnswIndexDestroy(hnsw_index);
  cuvsResourcesDestroy(res);
}

TEST(CagraC, SerializeHostStandardAllDtypes) {
  auto round_trip = [](auto type_tag, DLDataType dtype, const char *suffix) {
    using T = decltype(type_tag);
    cuvsResources_t res;
    ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);

    std::array<T, 8> values{};
    for (std::size_t i = 0; i < values.size(); ++i) {
      if constexpr (std::is_same_v<T, half>) {
        values[i] = __float2half(static_cast<float>(i));
      } else {
        values[i] = static_cast<T>(i);
      }
    }
    int64_t shape[2] = {4, 2};
    DLManagedTensor tensor{};
    tensor.dl_tensor.data = values.data();
    tensor.dl_tensor.device.device_type = kDLCPU;
    tensor.dl_tensor.ndim = 2;
    tensor.dl_tensor.dtype = dtype;
    tensor.dl_tensor.shape = shape;

    cuvsDataset_t dataset_view;
    ASSERT_EQ(cuvsDatasetMakeStandardView(res, &tensor, &dataset_view),
              CUVS_SUCCESS);
    cuvsCagraIndexParams_t params;
    ASSERT_EQ(cuvsCagraIndexParamsCreate(&params), CUVS_SUCCESS);
    cuvsCagraIndex_t index;
    ASSERT_EQ(cuvsCagraIndexCreate(&index), CUVS_SUCCESS);
    ASSERT_EQ(cuvsCagraBuild(res, params, dataset_view, index), CUVS_SUCCESS);

    auto path = "/tmp/cuvs-cagra-host-standard-" + std::string(suffix) + "-" +
                std::to_string(getpid()) + ".bin";
    ASSERT_EQ(cuvsCagraSerializeGraphAndDataset(res, path.c_str(), index),
              CUVS_SUCCESS);

    cuvsCagraIndex_t loaded;
    ASSERT_EQ(cuvsCagraIndexCreate(&loaded), CUVS_SUCCESS);
    cuvsDataset_t loaded_dataset = nullptr;
    ASSERT_EQ(cuvsCagraDeserializeGraphAndDataset(res, path.c_str(), loaded,
                                                  &loaded_dataset),
              CUVS_SUCCESS)
      << cuvsGetLastErrorText();
    ASSERT_NE(loaded_dataset, nullptr);
    {
      DLDataType loaded_dtype{};
      cuvsDatasetMemType_t mem_type{};
      cuvsDatasetLayout_t layout{};
      ASSERT_EQ(cuvsDatasetGetDtype(loaded_dataset, &loaded_dtype), CUVS_SUCCESS);
      ASSERT_EQ(cuvsDatasetGetMemType(loaded_dataset, &mem_type), CUVS_SUCCESS);
      ASSERT_EQ(cuvsDatasetGetLayout(loaded_dataset, &layout), CUVS_SUCCESS);
      EXPECT_EQ(loaded_dtype.code, dtype.code);
      EXPECT_EQ(loaded_dtype.bits, dtype.bits);
      EXPECT_EQ(loaded_dtype.lanes, 1);
      EXPECT_EQ(mem_type, CUVS_DATASET_MEM_TYPE_HOST);
      EXPECT_EQ(layout, CUVS_DATASET_LAYOUT_STANDARD);
    }
    int64_t size = 0;
    int64_t dim = 0;
    EXPECT_EQ(cuvsCagraIndexGetSize(loaded, &size), CUVS_SUCCESS);
    EXPECT_EQ(cuvsCagraIndexGetDims(loaded, &dim), CUVS_SUCCESS);
    EXPECT_EQ(size, 4);
    EXPECT_EQ(dim, 2);

    // The index is non-owning: both objects are independently destroyable.
    cuvsCagraIndexDestroy(loaded);
    cuvsDatasetDestroy(loaded_dataset);
    cuvsCagraIndexDestroy(index);
    cuvsCagraIndexParamsDestroy(params);
    cuvsDatasetDestroy(dataset_view);
    cuvsResourcesDestroy(res);
    std::filesystem::remove(path);
  };

  round_trip(float{}, {kDLFloat, 32, 1}, "f32");
  round_trip(half{}, {kDLFloat, 16, 1}, "f16");
  round_trip(int8_t{}, {kDLInt, 8, 1}, "i8");
  round_trip(uint8_t{}, {kDLUInt, 8, 1}, "u8");
}

TEST(CagraC, ExplicitSerializationSemantics) {
  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);
  cudaStream_t stream;
  ASSERT_EQ(cuvsStreamGet(res, &stream), CUVS_SUCCESS);

  int64_t dataset_shape[2] = {4, 2};
  DLManagedTensor host_tensor{};
  host_tensor.dl_tensor.data = dataset;
  host_tensor.dl_tensor.device.device_type = kDLCPU;
  host_tensor.dl_tensor.ndim = 2;
  host_tensor.dl_tensor.dtype = {kDLFloat, 32, 1};
  host_tensor.dl_tensor.shape = dataset_shape;

  cuvsDataset_t host_view;
  ASSERT_EQ(cuvsDatasetMakeStandardView(res, &host_tensor, &host_view),
            CUVS_SUCCESS);
  cuvsCagraIndexParams_t params;
  ASSERT_EQ(cuvsCagraIndexParamsCreate(&params), CUVS_SUCCESS);
  cuvsCagraIndex_t source;
  ASSERT_EQ(cuvsCagraIndexCreate(&source), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraBuild(res, params, host_view, source), CUVS_SUCCESS);

  auto prefix = "/tmp/cuvs-cagra-explicit-" + std::to_string(getpid());
  auto full_path = prefix + "-full.bin";
  auto graph_path = prefix + "-graph.bin";
  ASSERT_EQ(cuvsCagraSerializeGraphAndDataset(res, full_path.c_str(), source),
            CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraSerializeGraph(res, graph_path.c_str(), source),
            CUVS_SUCCESS);

  auto expect_search = [&](cuvsCagraIndex_t index) {
    rmm::device_uvector<float> queries_d(8, stream);
    raft::copy(queries_d.data(), reinterpret_cast<float *>(queries), 8, stream);
    int64_t queries_shape[2] = {4, 2};
    DLManagedTensor queries_tensor{};
    queries_tensor.dl_tensor.data = queries_d.data();
    queries_tensor.dl_tensor.device.device_type = kDLCUDA;
    queries_tensor.dl_tensor.ndim = 2;
    queries_tensor.dl_tensor.dtype = {kDLFloat, 32, 1};
    queries_tensor.dl_tensor.shape = queries_shape;

    rmm::device_uvector<uint32_t> neighbors_d(4, stream);
    int64_t result_shape[2] = {4, 1};
    DLManagedTensor neighbors_tensor{};
    neighbors_tensor.dl_tensor.data = neighbors_d.data();
    neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
    neighbors_tensor.dl_tensor.ndim = 2;
    neighbors_tensor.dl_tensor.dtype = {kDLUInt, 32, 1};
    neighbors_tensor.dl_tensor.shape = result_shape;

    rmm::device_uvector<float> distances_d(4, stream);
    DLManagedTensor distances_tensor{};
    distances_tensor.dl_tensor.data = distances_d.data();
    distances_tensor.dl_tensor.device.device_type = kDLCUDA;
    distances_tensor.dl_tensor.ndim = 2;
    distances_tensor.dl_tensor.dtype = {kDLFloat, 32, 1};
    distances_tensor.dl_tensor.shape = result_shape;

    cuvsCagraSearchParams_t search_params;
    ASSERT_EQ(cuvsCagraSearchParamsCreate(&search_params), CUVS_SUCCESS);
    cuvsFilter no_filter{0, NO_FILTER};
    ASSERT_EQ(cuvsCagraSearch(res, search_params, index, &queries_tensor,
                              &neighbors_tensor, &distances_tensor, no_filter),
              CUVS_SUCCESS);
    EXPECT_TRUE(cuvs::devArrMatchHost(neighbors_exp, neighbors_d.data(), 4,
                                      cuvs::Compare<uint32_t>()));
    cuvsCagraSearchParamsDestroy(search_params);
  };

  cuvsCagraIndex_t loaded;
  ASSERT_EQ(cuvsCagraIndexCreate(&loaded), CUVS_SUCCESS);
  cuvsDataset_t loaded_dataset = nullptr;
  ASSERT_EQ(cuvsCagraDeserializeGraphAndDataset(res, full_path.c_str(), loaded,
                                                &loaded_dataset),
            CUVS_SUCCESS)
    << cuvsGetLastErrorText();
  ASSERT_NE(loaded_dataset, nullptr);
  {
    cuvsDatasetMemType_t mem_type{};
    cuvsDatasetLayout_t layout{};
    ASSERT_EQ(cuvsDatasetGetMemType(loaded_dataset, &mem_type), CUVS_SUCCESS);
    ASSERT_EQ(cuvsDatasetGetLayout(loaded_dataset, &layout), CUVS_SUCCESS);
    EXPECT_EQ(mem_type, CUVS_DATASET_MEM_TYPE_HOST);
    EXPECT_EQ(layout, CUVS_DATASET_LAYOUT_STANDARD);
  }

  // Dataset-requiring failures leave a populated destination and output
  // unchanged.
  auto loaded_addr = loaded->addr;
  cuvsDataset_t no_dataset = nullptr;
  EXPECT_EQ(cuvsCagraDeserializeGraphAndDataset(res, graph_path.c_str(), loaded,
                                                &no_dataset),
            CUVS_ERROR);
  EXPECT_EQ(loaded->addr, loaded_addr);
  EXPECT_EQ(no_dataset, nullptr);
  EXPECT_EQ(cuvsCagraDeserializeGraphAndDataset(res, full_path.c_str(), loaded,
                                                nullptr),
            CUVS_ERROR);
  EXPECT_EQ(loaded->addr, loaded_addr);
  auto populated_output = loaded_dataset;
  EXPECT_EQ(cuvsCagraDeserializeGraphAndDataset(res, full_path.c_str(), loaded,
                                                &populated_output),
            CUVS_ERROR);
  EXPECT_EQ(populated_output, loaded_dataset);
  EXPECT_EQ(loaded->addr, loaded_addr);

  auto truncated_path = prefix + "-truncated.bin";
  auto truncated_payload_path = prefix + "-truncated-payload.bin";
  auto bad_dtype_path = prefix + "-bad-dtype.bin";
  auto bad_kind_path = prefix + "-bad-kind.bin";
  std::filesystem::copy_file(full_path, truncated_payload_path);
  auto const full_size = std::filesystem::file_size(truncated_payload_path);
  std::filesystem::resize_file(truncated_payload_path, full_size - 1);
  {
    std::ofstream truncated(truncated_path, std::ios::binary);
    truncated.write("xx", 2);
    std::ofstream bad_dtype(bad_dtype_path, std::ios::binary);
    char unsupported_dtype[4] = {'<', 'i', '2', '\0'};
    bad_dtype.write(unsupported_dtype, 4);

    std::filesystem::copy_file(full_path, bad_kind_path);
    std::fstream bad_kind(
      bad_kind_path, std::ios::in | std::ios::out | std::ios::binary);
    raft::resources parse_res;
    bad_kind.seekg(4);
    static_cast<void>(raft::deserialize_scalar<int>(parse_res, bad_kind));
    auto const kind_position = bad_kind.tellg();
    uint32_t unsupported_kind = 99;
    bad_kind.seekp(kind_position);
    raft::serialize_scalar(parse_res, bad_kind, unsupported_kind);
  }
  EXPECT_EQ(cuvsCagraDeserializeGraph(res, truncated_path.c_str(), loaded),
            CUVS_ERROR);
  EXPECT_EQ(loaded->addr, loaded_addr);
  EXPECT_EQ(cuvsCagraDeserializeGraph(res, truncated_payload_path.c_str(), loaded),
            CUVS_ERROR);
  EXPECT_EQ(loaded->addr, loaded_addr);
  EXPECT_EQ(cuvsCagraDeserializeGraph(res, bad_dtype_path.c_str(), loaded),
            CUVS_ERROR);
  EXPECT_EQ(loaded->addr, loaded_addr);
  EXPECT_EQ(cuvsCagraDeserializeGraph(res, bad_kind_path.c_str(), loaded),
            CUVS_ERROR);
  EXPECT_EQ(loaded->addr, loaded_addr);

  // Graph-only mode also accepts a graph-and-dataset file and discards its
  // payload safely.
  cuvsCagraIndex_t graph_from_full;
  ASSERT_EQ(cuvsCagraIndexCreate(&graph_from_full), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraDeserializeGraph(res, full_path.c_str(), graph_from_full),
            CUVS_SUCCESS);
  auto sentinel_path = prefix + "-sentinel.bin";
  {
    std::ofstream sentinel(sentinel_path, std::ios::binary);
    sentinel << "sentinel";
  }
  EXPECT_EQ(cuvsCagraSerializeGraphAndDataset(res, sentinel_path.c_str(),
                                              graph_from_full),
            CUVS_ERROR);
  std::ifstream sentinel(sentinel_path, std::ios::binary);
  std::string sentinel_contents;
  sentinel >> sentinel_contents;
  EXPECT_EQ(sentinel_contents, "sentinel");

  // A graph-only file becomes search-ready after attaching a caller-owned
  // padded dataset.
  cuvsCagraIndex_t graph_only;
  ASSERT_EQ(cuvsCagraIndexCreate(&graph_only), CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraDeserializeGraph(res, graph_path.c_str(), graph_only),
            CUVS_SUCCESS);
  rmm::device_uvector<float> device_dataset(8, stream);
  raft::copy(device_dataset.data(), reinterpret_cast<float *>(dataset), 8,
             stream);
  DLManagedTensor device_tensor = host_tensor;
  device_tensor.dl_tensor.data = device_dataset.data();
  device_tensor.dl_tensor.device.device_type = kDLCUDA;
  cuvsDataset_t external_owner;
  ASSERT_EQ(cuvsDatasetMakePadded(
              res, &device_tensor, CUVS_DATASET_MEM_TYPE_DEVICE, &external_owner),
            CUVS_SUCCESS);
  ASSERT_EQ(cuvsCagraUpdateDataset(res, external_owner, loaded), CUVS_SUCCESS);
  expect_search(loaded);

  ASSERT_EQ(cuvsCagraUpdateDataset(res, external_owner, graph_only),
            CUVS_SUCCESS);
  expect_search(graph_only);

  cuvsCagraIndexDestroy(graph_only);
  cuvsDatasetDestroy(external_owner);
  cuvsCagraIndexDestroy(graph_from_full);
  cuvsCagraIndexDestroy(loaded);
  cuvsDatasetDestroy(loaded_dataset);
  cuvsCagraIndexDestroy(source);
  cuvsCagraIndexParamsDestroy(params);
  cuvsDatasetDestroy(host_view);
  cuvsResourcesDestroy(res);
  std::filesystem::remove(full_path);
  std::filesystem::remove(graph_path);
  std::filesystem::remove(truncated_path);
  std::filesystem::remove(truncated_payload_path);
  std::filesystem::remove(bad_dtype_path);
  std::filesystem::remove(sentinel_path);
  std::filesystem::remove(bad_kind_path);
}

// Multi-partition search splits the known 4-row dataset into two contiguous 2-row partitions
// (partition p holds global rows [2*p, 2*p+2)). The result is reported as a per-partition
// (partition_id, local ordinal) pair, which decodes to the global index as 2*partition_id +
// neighbor. So the global answers stay neighbors_exp = {3, 0, 3, 1}, i.e.:
//   partition_ids = {1, 0, 1, 0}, local neighbors = {1, 0, 1, 1}.
TEST(CagraC, BuildSearchMultiPartition)
{
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

  constexpr uint32_t num_partitions = 2;
  constexpr int part_rows = 2, dim = 2, n_queries = 4, k = 1;

  // Build one index per contiguous 2-row slice of the host dataset.
  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);

  cuvsCagraIndex_t indices[num_partitions];
  cuvsDataset_t part_views[num_partitions]         = {};
  cuvsDataset_t part_padded_owners[num_partitions] = {};
  for (uint32_t p = 0; p < num_partitions; p++) {
    DLManagedTensor part_tensor;
    part_tensor.dl_tensor.data               = &dataset[p * part_rows][0];
    part_tensor.dl_tensor.device.device_type = kDLCPU;
    part_tensor.dl_tensor.ndim               = 2;
    part_tensor.dl_tensor.dtype.code         = kDLFloat;
    part_tensor.dl_tensor.dtype.bits         = 32;
    part_tensor.dl_tensor.dtype.lanes        = 1;
    int64_t part_shape[2]                    = {part_rows, dim};
    part_tensor.dl_tensor.shape              = part_shape;
    part_tensor.dl_tensor.strides            = nullptr;

    cuvsCagraIndexCreate(&indices[p]);
    ASSERT_EQ(cuvsDatasetMakeStandardView(res, &part_tensor, &part_views[p]), CUVS_SUCCESS);
    ASSERT_EQ(cuvsCagraBuild(res, build_params, part_views[p], indices[p]), CUVS_SUCCESS);

    // The host build yields a host index; multi-partition search needs every partition to carry a
    // device-padded dataset.
    ASSERT_EQ(cuvsDatasetMakePadded(
                res, &part_tensor, CUVS_DATASET_MEM_TYPE_DEVICE, &part_padded_owners[p]),
              CUVS_SUCCESS);
    ASSERT_EQ(cuvsCagraUpdateDataset(res, part_padded_owners[p], indices[p]), CUVS_SUCCESS);
  }

  // queries (device)
  rmm::device_uvector<float> queries_d(n_queries * dim, stream);
  raft::copy(queries_d.data(), (float*)queries, n_queries * dim, stream);
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries_d.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {n_queries, dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  // partition_ids output (device, uint32)
  rmm::device_uvector<uint32_t> partition_ids_d(n_queries * k, stream);
  DLManagedTensor partition_ids_tensor;
  partition_ids_tensor.dl_tensor.data               = partition_ids_d.data();
  partition_ids_tensor.dl_tensor.device.device_type = kDLCUDA;
  partition_ids_tensor.dl_tensor.ndim               = 2;
  partition_ids_tensor.dl_tensor.dtype.code         = kDLUInt;
  partition_ids_tensor.dl_tensor.dtype.bits         = 32;
  partition_ids_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t out_shape[2]                              = {n_queries, k};
  partition_ids_tensor.dl_tensor.shape              = out_shape;
  partition_ids_tensor.dl_tensor.strides            = nullptr;

  // neighbors output (device, uint32 local ordinal)
  rmm::device_uvector<uint32_t> neighbors_d(n_queries * k, stream);
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_d.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 32;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  neighbors_tensor.dl_tensor.shape              = out_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  // distances output (device, float)
  rmm::device_uvector<float> distances_d(n_queries * k, stream);
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_d.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  distances_tensor.dl_tensor.shape              = out_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);
  ASSERT_EQ(cuvsCagraSearchMultiPartition(res,
                                          search_params,
                                          num_partitions,
                                          indices,
                                          &queries_tensor,
                                          &partition_ids_tensor,
                                          &neighbors_tensor,
                                          &distances_tensor,
                                          /* filters = */ nullptr),
            CUVS_SUCCESS);

  uint32_t partition_ids_exp[4] = {1, 0, 1, 0};
  uint32_t neighbors_exp_mp[4]  = {1, 0, 1, 1};
  ASSERT_TRUE(cuvs::devArrMatchHost(
    partition_ids_exp, partition_ids_d.data(), 4, cuvs::Compare<uint32_t>()));
  ASSERT_TRUE(
    cuvs::devArrMatchHost(neighbors_exp_mp, neighbors_d.data(), 4, cuvs::Compare<uint32_t>()));
  ASSERT_TRUE(cuvs::devArrMatchHost(
    distances_exp, distances_d.data(), 4, cuvs::CompareApprox<float>(0.001f)));

  cuvsCagraSearchParamsDestroy(search_params);
  cuvsCagraIndexParamsDestroy(build_params);
  for (uint32_t p = 0; p < num_partitions; p++) {
    cuvsCagraIndexDestroy(indices[p]);
    cuvsDatasetDestroy(part_padded_owners[p]);
    cuvsDatasetDestroy(part_views[p]);
  }
  cuvsResourcesDestroy(res);
}

// Same as BuildSearchMultiPartition, but requesting int64 neighbor ordinals to exercise the
// int64 neighbor dispatch (matching the single-partition search coverage).
TEST(CagraC, BuildSearchMultiPartitionInt64Neighbors)
{
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

  constexpr uint32_t num_partitions = 2;
  constexpr int part_rows = 2, dim = 2, n_queries = 4, k = 1;

  // Build one index per contiguous 2-row slice of the host dataset.
  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);

  cuvsCagraIndex_t indices[num_partitions];
  cuvsDataset_t part_views[num_partitions]         = {};
  cuvsDataset_t part_padded_owners[num_partitions] = {};
  for (uint32_t p = 0; p < num_partitions; p++) {
    DLManagedTensor part_tensor;
    part_tensor.dl_tensor.data               = &dataset[p * part_rows][0];
    part_tensor.dl_tensor.device.device_type = kDLCPU;
    part_tensor.dl_tensor.ndim               = 2;
    part_tensor.dl_tensor.dtype.code         = kDLFloat;
    part_tensor.dl_tensor.dtype.bits         = 32;
    part_tensor.dl_tensor.dtype.lanes        = 1;
    int64_t part_shape[2]                    = {part_rows, dim};
    part_tensor.dl_tensor.shape              = part_shape;
    part_tensor.dl_tensor.strides            = nullptr;

    cuvsCagraIndexCreate(&indices[p]);
    ASSERT_EQ(cuvsDatasetMakeStandardView(res, &part_tensor, &part_views[p]), CUVS_SUCCESS);
    ASSERT_EQ(cuvsCagraBuild(res, build_params, part_views[p], indices[p]), CUVS_SUCCESS);

    // The host build yields a host index; multi-partition search needs every partition to carry a
    // device-padded dataset.
    ASSERT_EQ(cuvsDatasetMakePadded(
                res, &part_tensor, CUVS_DATASET_MEM_TYPE_DEVICE, &part_padded_owners[p]),
              CUVS_SUCCESS);
    ASSERT_EQ(cuvsCagraUpdateDataset(res, part_padded_owners[p], indices[p]), CUVS_SUCCESS);
  }

  // queries (device)
  rmm::device_uvector<float> queries_d(n_queries * dim, stream);
  raft::copy(queries_d.data(), (float*)queries, n_queries * dim, stream);
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries_d.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {n_queries, dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  // partition_ids output (device, uint32)
  rmm::device_uvector<uint32_t> partition_ids_d(n_queries * k, stream);
  DLManagedTensor partition_ids_tensor;
  partition_ids_tensor.dl_tensor.data               = partition_ids_d.data();
  partition_ids_tensor.dl_tensor.device.device_type = kDLCUDA;
  partition_ids_tensor.dl_tensor.ndim               = 2;
  partition_ids_tensor.dl_tensor.dtype.code         = kDLUInt;
  partition_ids_tensor.dl_tensor.dtype.bits         = 32;
  partition_ids_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t out_shape[2]                              = {n_queries, k};
  partition_ids_tensor.dl_tensor.shape              = out_shape;
  partition_ids_tensor.dl_tensor.strides            = nullptr;

  // neighbors output (device, int64 local ordinal)
  rmm::device_uvector<int64_t> neighbors_d(n_queries * k, stream);
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_d.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 64;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  neighbors_tensor.dl_tensor.shape              = out_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  // distances output (device, float)
  rmm::device_uvector<float> distances_d(n_queries * k, stream);
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_d.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  distances_tensor.dl_tensor.shape              = out_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);
  ASSERT_EQ(cuvsCagraSearchMultiPartition(res,
                                          search_params,
                                          num_partitions,
                                          indices,
                                          &queries_tensor,
                                          &partition_ids_tensor,
                                          &neighbors_tensor,
                                          &distances_tensor,
                                          /* filters = */ nullptr),
            CUVS_SUCCESS);

  uint32_t partition_ids_exp[4] = {1, 0, 1, 0};
  int64_t neighbors_exp_mp[4]   = {1, 0, 1, 1};
  ASSERT_TRUE(cuvs::devArrMatchHost(
    partition_ids_exp, partition_ids_d.data(), 4, cuvs::Compare<uint32_t>()));
  ASSERT_TRUE(
    cuvs::devArrMatchHost(neighbors_exp_mp, neighbors_d.data(), 4, cuvs::Compare<int64_t>()));
  ASSERT_TRUE(cuvs::devArrMatchHost(
    distances_exp, distances_d.data(), 4, cuvs::CompareApprox<float>(0.001f)));

  cuvsCagraSearchParamsDestroy(search_params);
  cuvsCagraIndexParamsDestroy(build_params);
  for (uint32_t p = 0; p < num_partitions; p++) {
    cuvsCagraIndexDestroy(indices[p]);
    cuvsDatasetDestroy(part_padded_owners[p]);
    cuvsDatasetDestroy(part_views[p]);
  }
  cuvsResourcesDestroy(res);
}

// Filtered multi-partition search: the combined bitset is addressed by the global index
// (partition_offset[p] + local), so filtering global rows 1 and 2 (bitset 0b1001) matches the
// single-index filtered case. Global answers stay neighbors_exp_filtered = {3, 0, 3, 0}:
//   partition_ids = {1, 0, 1, 0}, local neighbors = {1, 0, 1, 0}.
TEST(CagraC, BuildSearchMultiPartitionFiltered)
{
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

  constexpr uint32_t num_partitions = 2;
  constexpr int part_rows = 2, dim = 2, n_queries = 4, k = 1;

  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);

  cuvsCagraIndex_t indices[num_partitions];
  cuvsDataset_t part_views[num_partitions]         = {};
  cuvsDataset_t part_padded_owners[num_partitions] = {};
  for (uint32_t p = 0; p < num_partitions; p++) {
    DLManagedTensor part_tensor;
    part_tensor.dl_tensor.data               = &dataset[p * part_rows][0];
    part_tensor.dl_tensor.device.device_type = kDLCPU;
    part_tensor.dl_tensor.ndim               = 2;
    part_tensor.dl_tensor.dtype.code         = kDLFloat;
    part_tensor.dl_tensor.dtype.bits         = 32;
    part_tensor.dl_tensor.dtype.lanes        = 1;
    int64_t part_shape[2]                    = {part_rows, dim};
    part_tensor.dl_tensor.shape              = part_shape;
    part_tensor.dl_tensor.strides            = nullptr;

    cuvsCagraIndexCreate(&indices[p]);
    ASSERT_EQ(cuvsDatasetMakeStandardView(res, &part_tensor, &part_views[p]), CUVS_SUCCESS);
    ASSERT_EQ(cuvsCagraBuild(res, build_params, part_views[p], indices[p]), CUVS_SUCCESS);

    // The host build yields a host index; multi-partition search needs every partition to carry a
    // device-padded dataset.
    ASSERT_EQ(cuvsDatasetMakePadded(
                res, &part_tensor, CUVS_DATASET_MEM_TYPE_DEVICE, &part_padded_owners[p]),
              CUVS_SUCCESS);
    ASSERT_EQ(cuvsCagraUpdateDataset(res, part_padded_owners[p], indices[p]), CUVS_SUCCESS);
  }

  rmm::device_uvector<float> queries_d(n_queries * dim, stream);
  raft::copy(queries_d.data(), (float*)queries, n_queries * dim, stream);
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries_d.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {n_queries, dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  rmm::device_uvector<uint32_t> partition_ids_d(n_queries * k, stream);
  int64_t out_shape[2] = {n_queries, k};
  DLManagedTensor partition_ids_tensor;
  partition_ids_tensor.dl_tensor.data               = partition_ids_d.data();
  partition_ids_tensor.dl_tensor.device.device_type = kDLCUDA;
  partition_ids_tensor.dl_tensor.ndim               = 2;
  partition_ids_tensor.dl_tensor.dtype.code         = kDLUInt;
  partition_ids_tensor.dl_tensor.dtype.bits         = 32;
  partition_ids_tensor.dl_tensor.dtype.lanes        = 1;
  partition_ids_tensor.dl_tensor.shape              = out_shape;
  partition_ids_tensor.dl_tensor.strides            = nullptr;

  rmm::device_uvector<uint32_t> neighbors_d(n_queries * k, stream);
  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_d.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 32;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  neighbors_tensor.dl_tensor.shape              = out_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  rmm::device_uvector<float> distances_d(n_queries * k, stream);
  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_d.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  distances_tensor.dl_tensor.shape              = out_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  // Each partition supplies its OWN bitset (one bit per row in that partition). Keeping global rows
  // 0 and 3 (removing 1 and 2) => partition 0 keeps local row 0 (0b01), partition 1 keeps local
  // row 1 (0b10).
  uint32_t part0_bits[1] = {0b01u};
  uint32_t part1_bits[1] = {0b10u};
  rmm::device_uvector<uint32_t> part0_bitset_d(1, stream);
  rmm::device_uvector<uint32_t> part1_bitset_d(1, stream);
  raft::copy(part0_bitset_d.data(), part0_bits, 1, stream);
  raft::copy(part1_bitset_d.data(), part1_bits, 1, stream);

  int64_t part_bitset_shape[1] = {1};
  DLManagedTensor part0_bitset_tensor;
  part0_bitset_tensor.dl_tensor.data               = part0_bitset_d.data();
  part0_bitset_tensor.dl_tensor.device.device_type = kDLCUDA;
  part0_bitset_tensor.dl_tensor.ndim               = 1;
  part0_bitset_tensor.dl_tensor.dtype.code         = kDLUInt;
  part0_bitset_tensor.dl_tensor.dtype.bits         = 32;
  part0_bitset_tensor.dl_tensor.dtype.lanes        = 1;
  part0_bitset_tensor.dl_tensor.shape              = part_bitset_shape;
  part0_bitset_tensor.dl_tensor.strides            = nullptr;
  DLManagedTensor part1_bitset_tensor              = part0_bitset_tensor;
  part1_bitset_tensor.dl_tensor.data               = part1_bitset_d.data();

  // One filter per partition, each over that partition's own bitset.
  cuvsFilter filters[num_partitions];
  filters[0].type = BITSET;
  filters[0].addr = (uintptr_t)&part0_bitset_tensor;
  filters[1].type = BITSET;
  filters[1].addr = (uintptr_t)&part1_bitset_tensor;

  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);
  ASSERT_EQ(cuvsCagraSearchMultiPartition(res,
                                          search_params,
                                          num_partitions,
                                          indices,
                                          &queries_tensor,
                                          &partition_ids_tensor,
                                          &neighbors_tensor,
                                          &distances_tensor,
                                          filters),
            CUVS_SUCCESS);

  uint32_t partition_ids_exp[4] = {1, 0, 1, 0};
  uint32_t neighbors_exp_mp[4]  = {1, 0, 1, 0};
  ASSERT_TRUE(cuvs::devArrMatchHost(
    partition_ids_exp, partition_ids_d.data(), 4, cuvs::Compare<uint32_t>()));
  ASSERT_TRUE(
    cuvs::devArrMatchHost(neighbors_exp_mp, neighbors_d.data(), 4, cuvs::Compare<uint32_t>()));
  ASSERT_TRUE(cuvs::devArrMatchHost(
    distances_exp_filtered, distances_d.data(), 4, cuvs::CompareApprox<float>(0.001f)));

  cuvsCagraSearchParamsDestroy(search_params);
  cuvsCagraIndexParamsDestroy(build_params);
  for (uint32_t p = 0; p < num_partitions; p++) {
    cuvsCagraIndexDestroy(indices[p]);
    cuvsDatasetDestroy(part_padded_owners[p]);
    cuvsDatasetDestroy(part_views[p]);
  }
  cuvsResourcesDestroy(res);
}

// MULTI_KERNEL is intentionally unsupported in the multi-partition path; the call must return an
// error rather than silently falling back. (cuvsError_t: CUVS_SUCCESS == 1, CUVS_ERROR == 0.)
TEST(CagraC, SearchMultiPartitionMultiKernelRejected)
{
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

  constexpr uint32_t num_partitions = 2;
  constexpr int part_rows = 2, dim = 2, n_queries = 4, k = 1;

  cuvsCagraIndexParams_t build_params;
  cuvsCagraIndexParamsCreate(&build_params);

  cuvsCagraIndex_t indices[num_partitions];
  cuvsDataset_t part_views[num_partitions]         = {};
  cuvsDataset_t part_padded_owners[num_partitions] = {};
  for (uint32_t p = 0; p < num_partitions; p++) {
    DLManagedTensor part_tensor;
    part_tensor.dl_tensor.data               = &dataset[p * part_rows][0];
    part_tensor.dl_tensor.device.device_type = kDLCPU;
    part_tensor.dl_tensor.ndim               = 2;
    part_tensor.dl_tensor.dtype.code         = kDLFloat;
    part_tensor.dl_tensor.dtype.bits         = 32;
    part_tensor.dl_tensor.dtype.lanes        = 1;
    int64_t part_shape[2]                    = {part_rows, dim};
    part_tensor.dl_tensor.shape              = part_shape;
    part_tensor.dl_tensor.strides            = nullptr;

    cuvsCagraIndexCreate(&indices[p]);
    ASSERT_EQ(cuvsDatasetMakeStandardView(res, &part_tensor, &part_views[p]), CUVS_SUCCESS);
    ASSERT_EQ(cuvsCagraBuild(res, build_params, part_views[p], indices[p]), CUVS_SUCCESS);

    // The host build yields a host index; multi-partition search needs every partition to carry a
    // device-padded dataset.
    ASSERT_EQ(cuvsDatasetMakePadded(
                res, &part_tensor, CUVS_DATASET_MEM_TYPE_DEVICE, &part_padded_owners[p]),
              CUVS_SUCCESS);
    ASSERT_EQ(cuvsCagraUpdateDataset(res, part_padded_owners[p], indices[p]), CUVS_SUCCESS);
  }

  rmm::device_uvector<float> queries_d(n_queries * dim, stream);
  raft::copy(queries_d.data(), (float*)queries, n_queries * dim, stream);
  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries_d.data();
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {n_queries, dim};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  rmm::device_uvector<uint32_t> partition_ids_d(n_queries * k, stream);
  rmm::device_uvector<uint32_t> neighbors_d(n_queries * k, stream);
  rmm::device_uvector<float> distances_d(n_queries * k, stream);
  int64_t out_shape[2] = {n_queries, k};

  DLManagedTensor partition_ids_tensor;
  partition_ids_tensor.dl_tensor.data               = partition_ids_d.data();
  partition_ids_tensor.dl_tensor.device.device_type = kDLCUDA;
  partition_ids_tensor.dl_tensor.ndim               = 2;
  partition_ids_tensor.dl_tensor.dtype.code         = kDLUInt;
  partition_ids_tensor.dl_tensor.dtype.bits         = 32;
  partition_ids_tensor.dl_tensor.dtype.lanes        = 1;
  partition_ids_tensor.dl_tensor.shape              = out_shape;
  partition_ids_tensor.dl_tensor.strides            = nullptr;

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_d.data();
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 32;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  neighbors_tensor.dl_tensor.shape              = out_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_d.data();
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  distances_tensor.dl_tensor.shape              = out_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  cuvsCagraSearchParams_t search_params;
  cuvsCagraSearchParamsCreate(&search_params);
  search_params->algo = MULTI_KERNEL;

  ASSERT_EQ(cuvsCagraSearchMultiPartition(res,
                                          search_params,
                                          num_partitions,
                                          indices,
                                          &queries_tensor,
                                          &partition_ids_tensor,
                                          &neighbors_tensor,
                                          &distances_tensor,
                                          nullptr),
            CUVS_ERROR);

  cuvsCagraSearchParamsDestroy(search_params);
  cuvsCagraIndexParamsDestroy(build_params);
  for (uint32_t p = 0; p < num_partitions; p++) {
    cuvsCagraIndexDestroy(indices[p]);
    cuvsDatasetDestroy(part_padded_owners[p]);
    cuvsDatasetDestroy(part_views[p]);
  }
  cuvsResourcesDestroy(res);
}
