/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>

#include <gtest/gtest.h>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resources.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <optional>
#include <utility>

namespace {

struct resource_handle {
  cuvsResources_t value{};

  resource_handle() = default;
  resource_handle(resource_handle const&) = delete;
  resource_handle& operator=(resource_handle const&) = delete;

  ~resource_handle()
  {
    if (value != 0) { cuvsResourcesDestroy(value); }
  }

  cuvsError_t destroy()
  {
    auto handle = std::exchange(value, cuvsResources_t{});
    return cuvsResourcesDestroy(handle);
  }
};

std::optional<rmm::mr::pool_memory_resource> current_pool()
{
  auto resource = rmm::mr::get_current_device_resource_ref();
  auto* pool    = cuda::mr::resource_cast<rmm::mr::pool_memory_resource>(&resource);
  if (pool == nullptr) { return std::nullopt; }
  return *pool;
}

void check_pool_restoration(bool destroy_older_first)
{
  ASSERT_FALSE(current_pool().has_value());
  resource_handle older;
  resource_handle newer;

  ASSERT_EQ(cuvsResourcesCreate(&older.value), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesCreate(&newer.value), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesSetMemoryPool(older.value, 1), CUVS_SUCCESS);
  auto older_pool = current_pool();
  ASSERT_TRUE(older_pool.has_value());

  ASSERT_EQ(cuvsResourcesSetMemoryPool(newer.value, 1), CUVS_SUCCESS);
  auto newer_pool = current_pool();
  ASSERT_TRUE(newer_pool.has_value());
  EXPECT_NE(*newer_pool, *older_pool);

  if (destroy_older_first) {
    EXPECT_EQ(older.destroy(), CUVS_SUCCESS);
    auto active_pool = current_pool();
    ASSERT_TRUE(active_pool.has_value());
    EXPECT_EQ(*active_pool, *newer_pool);
    EXPECT_EQ(newer.destroy(), CUVS_SUCCESS);
  } else {
    EXPECT_EQ(newer.destroy(), CUVS_SUCCESS);
    auto active_pool = current_pool();
    ASSERT_TRUE(active_pool.has_value());
    EXPECT_EQ(*active_pool, *older_pool);
    EXPECT_EQ(older.destroy(), CUVS_SUCCESS);
  }
  EXPECT_FALSE(current_pool().has_value());
}

TEST(ResourcesMemoryPool, DestroyOlderHandleFirst) { check_pool_restoration(true); }

TEST(ResourcesMemoryPool, DestroyNewerHandleFirst) { check_pool_restoration(false); }

TEST(ResourcesMemoryPool, RejectsConfigurationAfterWorkspaceUse)
{
  resource_handle handle;
  ASSERT_EQ(cuvsResourcesCreate(&handle.value), CUVS_SUCCESS);

  void* ptr{};
  ASSERT_EQ(cuvsRMMAlloc(handle.value, &ptr, 256), CUVS_SUCCESS);
  ASSERT_EQ(cuvsRMMFree(handle.value, ptr, 256), CUVS_SUCCESS);
  EXPECT_TRUE(reinterpret_cast<raft::resources*>(handle.value)->has_resource_factory(
    raft::resource::resource_type::WORKSPACE_RESOURCE));

  EXPECT_EQ(cuvsResourcesSetMemoryPool(handle.value, 1), CUVS_ERROR);
  EXPECT_FALSE(current_pool().has_value());
}

TEST(ResourcesMemoryPool, RejectsConfigurationAfterLargeWorkspaceUse)
{
  resource_handle handle;
  ASSERT_EQ(cuvsResourcesCreate(&handle.value), CUVS_SUCCESS);

  auto& raft_res = *reinterpret_cast<raft::resources*>(handle.value);
  (void)raft::resource::get_large_workspace_resource_ref(raft_res);
  EXPECT_TRUE(raft_res.has_resource_factory(
    raft::resource::resource_type::LARGE_WORKSPACE_RESOURCE));

  EXPECT_EQ(cuvsResourcesSetMemoryPool(handle.value, 1), CUVS_ERROR);
  EXPECT_FALSE(current_pool().has_value());
}

TEST(ResourcesMemoryPool, RejectsReconfigurationAfterWorkspaceUse)
{
  resource_handle handle;
  ASSERT_EQ(cuvsResourcesCreate(&handle.value), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesSetMemoryPool(handle.value, 1), CUVS_SUCCESS);
  auto original_pool = current_pool();
  ASSERT_TRUE(original_pool.has_value());

  void* ptr{};
  ASSERT_EQ(cuvsRMMAlloc(handle.value, &ptr, 256), CUVS_SUCCESS);
  ASSERT_EQ(cuvsRMMFree(handle.value, ptr, 256), CUVS_SUCCESS);

  EXPECT_EQ(cuvsResourcesSetMemoryPool(handle.value, 1), CUVS_ERROR);
  auto active_pool = current_pool();
  ASSERT_TRUE(active_pool.has_value());
  EXPECT_EQ(*active_pool, *original_pool);
}

TEST(ResourcesMemoryPool, RejectsMultiGpuConfigurationAfterWorkspaceUse)
{
  cuvsResources_t handle{};
  ASSERT_EQ(cuvsMultiGpuResourcesCreate(&handle), CUVS_SUCCESS);

  auto& devices = raft::resource::get_multi_gpu_resource(
    *reinterpret_cast<raft::resources*>(handle));
  ASSERT_FALSE(devices.empty());
  (void)raft::resource::get_workspace_resource_ref(devices.front());
  EXPECT_TRUE(devices.front().has_resource_factory(
    raft::resource::resource_type::WORKSPACE_RESOURCE));

  EXPECT_EQ(cuvsMultiGpuResourcesSetMemoryPool(handle, 1), CUVS_ERROR);
  EXPECT_EQ(cuvsMultiGpuResourcesDestroy(handle), CUVS_SUCCESS);
}

}  // namespace
