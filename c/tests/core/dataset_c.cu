/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>
#include <cuvs/core/dataset.h>
#include <dlpack/dlpack.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <vector>

namespace {

struct MatrixTensor {
  int64_t shape[2];
  DLManagedTensor tensor{};

  MatrixTensor(void* data, int64_t n_rows, int64_t n_cols, DLDeviceType device, uint8_t bits)
  {
    shape[0]                            = n_rows;
    shape[1]                            = n_cols;
    tensor.dl_tensor.data               = data;
    tensor.dl_tensor.device.device_type = device;
    tensor.dl_tensor.ndim               = 2;
    tensor.dl_tensor.dtype.code         = kDLFloat;
    tensor.dl_tensor.dtype.bits         = bits;
    tensor.dl_tensor.dtype.lanes        = 1;
    tensor.dl_tensor.shape              = shape;
    tensor.dl_tensor.strides            = nullptr;
  }
};

}  // namespace

TEST(DatasetC, CreateDestroy)
{
  cuvsDataset_t dataset;
  ASSERT_EQ(cuvsDatasetCreate(&dataset), CUVS_SUCCESS);
  ASSERT_NE(dataset, nullptr);
  ASSERT_EQ(cuvsDatasetDestroy(dataset), CUVS_SUCCESS);
}

TEST(DatasetC, MakePaddedFromHost)
{
  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);

  constexpr int64_t n_rows = 128;
  constexpr int64_t n_cols = 50;
  std::vector<float> host(n_rows * n_cols, 1.0f);
  MatrixTensor matrix(host.data(), n_rows, n_cols, kDLCPU, 32);

  cuvsDataset_t padded;
  ASSERT_EQ(
    cuvsDatasetMakePadded(res, &matrix.tensor, CUVS_DATASET_MEM_TYPE_DEVICE, &padded),
    CUVS_SUCCESS);
  ASSERT_NE(padded, nullptr);

  ASSERT_EQ(cuvsDatasetDestroy(padded), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

TEST(DatasetC, MakePaddedFromDeviceUnalignedOwnsCopy)
{
  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);
  cudaStream_t stream;
  ASSERT_EQ(cuvsStreamGet(res, &stream), CUVS_SUCCESS);

  constexpr int64_t n_rows = 64;
  constexpr int64_t n_cols = 50;
  std::vector<float> host(n_rows * n_cols, 2.0f);
  rmm::device_uvector<float> device(host.size(), stream);
  raft::copy(device.data(), host.data(), host.size(), stream);

  MatrixTensor matrix(device.data(), n_rows, n_cols, kDLCUDA, 32);
  cuvsDataset_t padded;
  ASSERT_EQ(
    cuvsDatasetMakePadded(res, &matrix.tensor, CUVS_DATASET_MEM_TYPE_DEVICE, &padded),
    CUVS_SUCCESS);
  ASSERT_NE(padded, nullptr);

  ASSERT_EQ(cuvsDatasetDestroy(padded), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

TEST(DatasetC, MakePaddedFromDeviceAlignedFailsUseView)
{
  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);
  cudaStream_t stream;
  ASSERT_EQ(cuvsStreamGet(res, &stream), CUVS_SUCCESS);

  constexpr int64_t n_rows = 64;
  constexpr int64_t n_cols = 32;
  std::vector<float> host(n_rows * n_cols, 3.0f);
  rmm::device_uvector<float> device(host.size(), stream);
  raft::copy(device.data(), host.data(), host.size(), stream);

  MatrixTensor matrix(device.data(), n_rows, n_cols, kDLCUDA, 32);
  cuvsDataset_t padded;
  EXPECT_EQ(
    cuvsDatasetMakePadded(res, &matrix.tensor, CUVS_DATASET_MEM_TYPE_DEVICE, &padded),
    CUVS_ERROR);
  EXPECT_EQ(padded, nullptr);

  cuvsDataset_t view;
  ASSERT_EQ(cuvsDatasetMakePaddedView(res, &matrix.tensor, &view), CUVS_SUCCESS);
  ASSERT_NE(view, nullptr);

  ASSERT_EQ(cuvsDatasetDestroy(view), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

TEST(DatasetC, MakeStandardViewHostAndDevice)
{
  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);
  cudaStream_t stream;
  ASSERT_EQ(cuvsStreamGet(res, &stream), CUVS_SUCCESS);

  constexpr int64_t n_rows = 32;
  constexpr int64_t n_cols = 16;
  std::vector<float> host(n_rows * n_cols, 4.0f);
  MatrixTensor host_matrix(host.data(), n_rows, n_cols, kDLCPU, 32);

  cuvsDataset_t host_view;
  ASSERT_EQ(cuvsDatasetMakeStandardView(res, &host_matrix.tensor, &host_view), CUVS_SUCCESS);
  ASSERT_NE(host_view, nullptr);

  rmm::device_uvector<float> device(host.size(), stream);
  raft::copy(device.data(), host.data(), host.size(), stream);
  MatrixTensor device_matrix(device.data(), n_rows, n_cols, kDLCUDA, 32);

  cuvsDataset_t device_view;
  ASSERT_EQ(cuvsDatasetMakeStandardView(res, &device_matrix.tensor, &device_view), CUVS_SUCCESS);
  ASSERT_NE(device_view, nullptr);

  ASSERT_EQ(cuvsDatasetDestroy(host_view), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetDestroy(device_view), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

TEST(DatasetC, MakeHostPadded)
{
  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);

  constexpr int64_t n_rows = 64;
  constexpr int64_t n_cols = 50;
  std::vector<float> host(n_rows * n_cols, 5.0f);
  MatrixTensor matrix(host.data(), n_rows, n_cols, kDLCPU, 32);

  cuvsDataset_t padded;
  ASSERT_EQ(cuvsDatasetMakePadded(res, &matrix.tensor, CUVS_DATASET_MEM_TYPE_HOST, &padded),
            CUVS_SUCCESS);
  ASSERT_NE(padded, nullptr);

  ASSERT_EQ(cuvsDatasetDestroy(padded), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

TEST(DatasetC, MakeHostPaddedFromDevice)
{
  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);
  cudaStream_t stream;
  ASSERT_EQ(cuvsStreamGet(res, &stream), CUVS_SUCCESS);

  constexpr int64_t n_rows = 64;
  constexpr int64_t n_cols = 50;
  std::vector<float> host(n_rows * n_cols, 6.0f);
  rmm::device_uvector<float> device(host.size(), stream);
  raft::copy(device.data(), host.data(), host.size(), stream);

  MatrixTensor matrix(device.data(), n_rows, n_cols, kDLCUDA, 32);
  cuvsDataset_t padded = nullptr;
  ASSERT_EQ(cuvsDatasetMakePadded(res, &matrix.tensor, CUVS_DATASET_MEM_TYPE_HOST, &padded),
            CUVS_SUCCESS);
  ASSERT_NE(padded, nullptr);

  cuvsDatasetMemType_t mem_type;
  cuvsDatasetLayout_t layout;
  bool is_owning;
  DLDataType dtype;
  ASSERT_EQ(cuvsDatasetGetMemType(padded, &mem_type), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetGetLayout(padded, &layout), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetGetIsOwning(padded, &is_owning), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetGetDtype(padded, &dtype), CUVS_SUCCESS);
  EXPECT_EQ(mem_type, CUVS_DATASET_MEM_TYPE_HOST);
  EXPECT_EQ(layout, CUVS_DATASET_LAYOUT_PADDED);
  EXPECT_TRUE(is_owning);
  EXPECT_EQ(dtype.code, kDLFloat);
  EXPECT_EQ(dtype.bits, 32);
  EXPECT_EQ(dtype.lanes, 1);

  ASSERT_EQ(cuvsDatasetDestroy(padded), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}
