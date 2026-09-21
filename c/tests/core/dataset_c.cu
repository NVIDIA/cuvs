/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/core/interop.hpp"

#include <cuvs/core/c_api.h>
#include <cuvs/core/dataset.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/preprocessing/quantize/bbq.h>
#include <dlpack/dlpack.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <raft/core/device_mdspan.hpp>
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

template <typename T>
auto make_device_matrix_tensor(T* data, int64_t rows, int64_t columns) -> DLManagedTensor
{
  DLManagedTensor tensor{};
  cuvs::core::to_dlpack(raft::make_device_matrix_view<T, int64_t>(data, rows, columns), &tensor);
  return tensor;
}

template <typename T>
auto make_device_vector_tensor(T* data, int64_t size) -> DLManagedTensor
{
  DLManagedTensor tensor{};
  cuvs::core::to_dlpack(raft::make_device_vector_view<T, int64_t>(data, size), &tensor);
  return tensor;
}

void free_tensor(DLManagedTensor& tensor)
{
  if (tensor.deleter != nullptr) { tensor.deleter(&tensor); }
}

}  // namespace

TEST(DatasetC, CreateDestroy)
{
  cuvsDataset_t dataset;
  ASSERT_EQ(cuvsDatasetCreate(&dataset), CUVS_SUCCESS);
  ASSERT_NE(dataset, nullptr);
  ASSERT_EQ(cuvsDatasetDestroy(dataset), CUVS_SUCCESS);
}

TEST(DatasetC, PQDatasetParamsCreateDestroy)
{
  cuvsPqParams_t params;
  ASSERT_EQ(cuvsPqParamsCreate(&params), CUVS_SUCCESS);
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->pq_bits, 8);
  ASSERT_EQ(cuvsPqParamsDestroy(params), CUVS_SUCCESS);
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

TEST(DatasetC, MakeBbqView)
{
  constexpr int64_t n_rows = 4;
  constexpr int64_t dim    = 8;

  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);
  cudaStream_t stream;
  ASSERT_EQ(cuvsStreamGet(res, &stream), CUVS_SUCCESS);

  rmm::device_uvector<uint8_t> codes_1b(n_rows, stream);
  rmm::device_uvector<float> lower(n_rows, stream);
  rmm::device_uvector<float> upper(n_rows, stream);
  rmm::device_uvector<float> corrections(n_rows, stream);
  rmm::device_uvector<int32_t> sums(n_rows, stream);
  rmm::device_uvector<float> centroid(dim, stream);
  rmm::device_uvector<float> delta(n_rows, stream);
  rmm::device_uvector<float> sum_delta(n_rows, stream);
  rmm::device_uvector<float> row_norm(n_rows, stream);

  auto codes_tensor       = make_device_matrix_tensor(codes_1b.data(), n_rows, 1);
  auto lower_tensor       = make_device_vector_tensor(lower.data(), n_rows);
  auto upper_tensor       = make_device_vector_tensor(upper.data(), n_rows);
  auto corrections_tensor = make_device_vector_tensor(corrections.data(), n_rows);
  auto sums_tensor        = make_device_vector_tensor(sums.data(), n_rows);
  auto centroid_tensor    = make_device_vector_tensor(centroid.data(), dim);
  auto delta_tensor       = make_device_vector_tensor(delta.data(), n_rows);
  auto sum_delta_tensor   = make_device_vector_tensor(sum_delta.data(), n_rows);
  auto row_norm_tensor    = make_device_vector_tensor(row_norm.data(), n_rows);

  cuvsBbqQuantizer_t quantizer;
  ASSERT_EQ(cuvsBbqQuantizerCreateView(&codes_tensor,
                                       &lower_tensor,
                                       &upper_tensor,
                                       &corrections_tensor,
                                       &sums_tensor,
                                       &centroid_tensor,
                                       &delta_tensor,
                                       &sum_delta_tensor,
                                       &row_norm_tensor,
                                       CUVS_BBQ_CODE_LAYOUT_PACKED_1B,
                                       L2Expanded,
                                       0.0f,
                                       &quantizer),
            CUVS_SUCCESS);

  cuvsDataset_t dataset;
  ASSERT_EQ(cuvsDatasetMakeBbqView(res, &quantizer, 1, &dataset), CUVS_SUCCESS);

  cuvsDatasetLayout_t layout;
  cuvsDatasetMemType_t mem_type;
  bool is_owning;
  DLDataType dtype;
  ASSERT_EQ(cuvsDatasetGetLayout(dataset, &layout), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetGetMemType(dataset, &mem_type), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetGetIsOwning(dataset, &is_owning), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetGetDtype(dataset, &dtype), CUVS_SUCCESS);
  EXPECT_EQ(layout, CUVS_DATASET_LAYOUT_BBQ);
  EXPECT_EQ(mem_type, CUVS_DATASET_MEM_TYPE_DEVICE);
  EXPECT_FALSE(is_owning);
  EXPECT_EQ(dtype.code, kDLFloat);
  EXPECT_EQ(dtype.bits, 32);

  ASSERT_EQ(cuvsDatasetDestroy(dataset), CUVS_SUCCESS);

  rmm::device_uvector<uint8_t> codes_4t(n_rows * 4, stream);
  auto codes_4t_tensor = make_device_matrix_tensor(codes_4t.data(), n_rows, 4);
  cuvsBbqQuantizer_t quantizer_4t;
  ASSERT_EQ(cuvsBbqQuantizerCreateView(&codes_4t_tensor,
                                       &lower_tensor,
                                       &upper_tensor,
                                       &corrections_tensor,
                                       &sums_tensor,
                                       &centroid_tensor,
                                       &delta_tensor,
                                       &sum_delta_tensor,
                                       &row_norm_tensor,
                                       CUVS_BBQ_CODE_LAYOUT_TRANSPOSED_4B,
                                       L2Expanded,
                                       0.0f,
                                       &quantizer_4t),
            CUVS_SUCCESS);

  cuvsBbqQuantizer_t quantizers[] = {quantizer, quantizer_4t};
  cuvsDataset_t asymmetric_dataset;
  ASSERT_EQ(cuvsDatasetMakeBbqView(res, quantizers, 2, &asymmetric_dataset), CUVS_SUCCESS);
  ASSERT_EQ(cuvsDatasetGetLayout(asymmetric_dataset, &layout), CUVS_SUCCESS);
  EXPECT_EQ(layout, CUVS_DATASET_LAYOUT_BBQ);

  ASSERT_EQ(cuvsDatasetDestroy(asymmetric_dataset), CUVS_SUCCESS);
  ASSERT_EQ(cuvsBbqQuantizerDestroy(quantizer_4t), CUVS_SUCCESS);
  ASSERT_EQ(cuvsBbqQuantizerDestroy(quantizer), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);

  free_tensor(codes_tensor);
  free_tensor(codes_4t_tensor);
  free_tensor(lower_tensor);
  free_tensor(upper_tensor);
  free_tensor(corrections_tensor);
  free_tensor(sums_tensor);
  free_tensor(centroid_tensor);
  free_tensor(delta_tensor);
  free_tensor(sum_delta_tensor);
  free_tensor(row_norm_tensor);
}
