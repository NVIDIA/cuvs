/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.cuh"

#include <cuvs/core/c_api.h>
#include <cuvs/selection/select_k.h>
#include <dlpack/dlpack.h>

#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <raft/core/device_mdarray.hpp>

// cuvsSelectK treats the input as a flat [1, n] matrix and returns the k smallest values (and
// their int64 column positions), sorted ascending. Distinct values keep the expected output
// unambiguous: from {5, 2, 8, 1, 3} the 3 smallest are 1, 2, 3 at positions 3, 1, 4.
TEST(SelectKC, SelectKSmallest)
{
  cuvsResources_t res;
  cuvsResourcesCreate(&res);
  cudaStream_t stream;
  cuvsStreamGet(res, &stream);

  constexpr int n = 5, k = 3;
  float in_val[n] = {5.0f, 2.0f, 8.0f, 1.0f, 3.0f};

  // input values (device, [1, n], float32)
  rmm::device_uvector<float> in_d(n, stream);
  raft::copy(in_d.data(), in_val, n, stream);
  DLManagedTensor in_tensor{};
  in_tensor.dl_tensor.data               = in_d.data();
  in_tensor.dl_tensor.device.device_type = kDLCUDA;
  in_tensor.dl_tensor.ndim               = 2;
  in_tensor.dl_tensor.dtype.code         = kDLFloat;
  in_tensor.dl_tensor.dtype.bits         = 32;
  in_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t in_shape[2]                    = {1, n};
  in_tensor.dl_tensor.shape              = in_shape;
  in_tensor.dl_tensor.strides            = nullptr;

  // output values (device, [1, k], float32)
  rmm::device_uvector<float> out_val_d(k, stream);
  DLManagedTensor out_val_tensor{};
  out_val_tensor.dl_tensor.data               = out_val_d.data();
  out_val_tensor.dl_tensor.device.device_type = kDLCUDA;
  out_val_tensor.dl_tensor.ndim               = 2;
  out_val_tensor.dl_tensor.dtype.code         = kDLFloat;
  out_val_tensor.dl_tensor.dtype.bits         = 32;
  out_val_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t out_shape[2]                        = {1, k};
  out_val_tensor.dl_tensor.shape              = out_shape;
  out_val_tensor.dl_tensor.strides            = nullptr;

  // output indices (device, [1, k], int64)
  rmm::device_uvector<int64_t> out_idx_d(k, stream);
  DLManagedTensor out_idx_tensor{};
  out_idx_tensor.dl_tensor.data               = out_idx_d.data();
  out_idx_tensor.dl_tensor.device.device_type = kDLCUDA;
  out_idx_tensor.dl_tensor.ndim               = 2;
  out_idx_tensor.dl_tensor.dtype.code         = kDLInt;
  out_idx_tensor.dl_tensor.dtype.bits         = 64;
  out_idx_tensor.dl_tensor.dtype.lanes        = 1;
  out_idx_tensor.dl_tensor.shape              = out_shape;
  out_idx_tensor.dl_tensor.strides            = nullptr;

  ASSERT_EQ(cuvsSelectK(res, &in_tensor, &out_val_tensor, &out_idx_tensor), CUVS_SUCCESS);

  float out_val_exp[k]   = {1.0f, 2.0f, 3.0f};
  int64_t out_idx_exp[k] = {3, 1, 4};
  ASSERT_TRUE(
    cuvs::devArrMatchHost(out_val_exp, out_val_d.data(), k, cuvs::CompareApprox<float>(0.001f)));
  ASSERT_TRUE(cuvs::devArrMatchHost(out_idx_exp, out_idx_d.data(), k, cuvs::Compare<int64_t>()));

  cuvsResourcesDestroy(res);
}
