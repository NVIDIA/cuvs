/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>

#include <gtest/gtest.h>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resources.hpp>

#include <cuda_runtime.h>

#include <cstdint>

namespace {

struct resource_handle {
  cuvsResources_t value{};
  cuvsError_t (*destroy)(cuvsResources_t);

  explicit resource_handle(cuvsError_t (*destroy_fn)(cuvsResources_t)) : destroy(destroy_fn) {}
  ~resource_handle()
  {
    if (value != 0) { destroy(value); }
  }
};

TEST(ResourcesStreamPool, SingleGpuSetterInstallsPool)
{
  resource_handle handle{cuvsResourcesDestroy};
  ASSERT_EQ(cuvsResourcesCreate(&handle.value), CUVS_SUCCESS);

  auto& resources = *reinterpret_cast<raft::resources*>(handle.value);
  EXPECT_FALSE(resources.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL));

  ASSERT_EQ(cuvsResourcesSetStreamPool(handle.value, 3), CUVS_SUCCESS);
  EXPECT_EQ(raft::resource::get_stream_pool_size(resources), 3);
}

TEST(ResourcesStreamPool, MultiGpuSetterInstallsPoolOnEveryDevice)
{
  int device_count{};
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  ASSERT_GT(device_count, 0);

  // Use two managed resources even when CI has only one visible GPU.
  int32_t device_ids[2] = {0, device_count > 1 ? 1 : 0};
  int64_t shape[1]      = {2};
  DLManagedTensor device_ids_tensor{};
  device_ids_tensor.dl_tensor.data         = device_ids;
  device_ids_tensor.dl_tensor.device       = {kDLCPU, 0};
  device_ids_tensor.dl_tensor.ndim         = 1;
  device_ids_tensor.dl_tensor.dtype        = {kDLInt, 32, 1};
  device_ids_tensor.dl_tensor.shape        = shape;

  resource_handle handle{cuvsMultiGpuResourcesDestroy};
  ASSERT_EQ(cuvsMultiGpuResourcesCreateWithDeviceIds(&handle.value, &device_ids_tensor),
            CUVS_SUCCESS);

  auto& devices =
    raft::resource::get_multi_gpu_resource(*reinterpret_cast<raft::resources*>(handle.value));
  ASSERT_EQ(devices.size(), 2);

  ASSERT_EQ(cuvsMultiGpuResourcesSetStreamPool(handle.value, 2), CUVS_SUCCESS);
  for (auto const& device : devices) {
    EXPECT_EQ(raft::resource::get_stream_pool_size(device), 2);
  }
}

}  // namespace
