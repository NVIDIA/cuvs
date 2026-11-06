/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/cuda_fp16.hpp>
#include <cuvs/preprocessing/quantize/scalar.h>

#include <gtest/gtest.h>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <type_traits>
#include <vector>

namespace {

template <typename T>
DLManagedTensor matrix_tensor(T* data, DLDeviceType device, int64_t* shape)
{
  DLManagedTensor tensor{};
  tensor.dl_tensor.data = data;
  tensor.dl_tensor.device = {device, 0};
  tensor.dl_tensor.ndim = 2;
  tensor.dl_tensor.dtype = {std::is_same_v<T, int8_t> ? uint8_t{kDLInt} : uint8_t{kDLFloat},
                            uint8_t{sizeof(T) * 8}, 1};
  tensor.dl_tensor.shape = shape;
  return tensor;
}

template <typename T>
class ScalarQuantizerC : public ::testing::Test {
 protected:
  cuvsResources_t res{};
  cudaStream_t stream{};
  cuvsScalarQuantizerParams params{1.f};
  cuvsScalarQuantizerPerDim_t quantizer{};

  void SetUp() override
  {
    ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);
    ASSERT_EQ(cuvsStreamGet(res, &stream), CUVS_SUCCESS);
    ASSERT_EQ(cuvsScalarQuantizerPerDimCreate(&quantizer), CUVS_SUCCESS);
  }

  void TearDown() override
  {
    EXPECT_EQ(cuvsStreamSync(res), CUVS_SUCCESS);
    EXPECT_EQ(cuvsScalarQuantizerPerDimDestroy(quantizer), CUVS_SUCCESS);
    EXPECT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
  }

  void check_round_trip(bool host_training)
  {
    int64_t shape[]{2, 3};
    std::vector<T> values{0.f, 10.f, 9.f, 254.f, 137.f, 9.f};
    rmm::device_uvector<T> input(values.size(), stream);
    rmm::device_uvector<int8_t> codes(values.size(), stream);
    raft::update_device(input.data(), values.data(), values.size(), stream);
    auto host = matrix_tensor(values.data(), kDLCPU, shape);
    auto device = matrix_tensor(input.data(), kDLCUDA, shape);
    auto encoded = matrix_tensor(codes.data(), kDLCUDA, shape);
    ASSERT_EQ(cuvsScalarQuantizerPerDimTrain(res, &params, host_training ? &host : &device, quantizer),
              CUVS_SUCCESS);
    ASSERT_EQ(cuvsScalarQuantizerPerDimTransform(res, quantizer, &device, &encoded), CUVS_SUCCESS);
    ASSERT_EQ(cuvsScalarQuantizerPerDimInverseTransform(res, quantizer, &encoded, &device), CUVS_SUCCESS);
    std::vector<int8_t> actual_codes(values.size());
    std::vector<T> decoded(values.size());
    raft::update_host(actual_codes.data(), codes.data(), actual_codes.size(), stream);
    raft::update_host(decoded.data(), input.data(), decoded.size(), stream);
    ASSERT_EQ(cuvsStreamSync(res), CUVS_SUCCESS);
    EXPECT_EQ(actual_codes, (std::vector<int8_t>{-127, -127, 0, 127, 127, 0}));
    for (size_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(double(decoded[i]), double(values[i])) << i;
    }
  }
};

}  // namespace

using ScalarTypes = ::testing::Types<half, float, double>;
TYPED_TEST_SUITE(ScalarQuantizerC, ScalarTypes);

TYPED_TEST(ScalarQuantizerC, PerDimensionTrainAndRetrain)
{
  this->check_round_trip(false);
  this->check_round_trip(true);
}

TYPED_TEST(ScalarQuantizerC, GlobalBoundsAndRoundTrip)
{
  using T = TypeParam;
  int64_t shape[]{2, 1};
  std::vector<T> values{T{-3}, T{7}}, decoded(2);
  std::vector<int8_t> codes(2);
  auto input = matrix_tensor(values.data(), kDLCPU, shape);
  auto encoded = matrix_tensor(codes.data(), kDLCPU, shape);
  auto output = matrix_tensor(decoded.data(), kDLCPU, shape);
  cuvsScalarQuantizer global{};
  ASSERT_EQ(cuvsScalarQuantizerTrain(this->res, &this->params, &input, &global), CUVS_SUCCESS);
  EXPECT_EQ(global.min_, -3);
  EXPECT_EQ(global.max_, 7);
  ASSERT_EQ(cuvsScalarQuantizerTransform(this->res, &global, &input, &encoded), CUVS_SUCCESS);
  ASSERT_EQ(cuvsScalarQuantizerInverseTransform(this->res, &global, &encoded, &output), CUVS_SUCCESS);
  EXPECT_EQ(codes, (std::vector<int8_t>{-128, 127}));
  EXPECT_EQ(double(decoded[0]), -3);
  EXPECT_EQ(double(decoded[1]), 7);
}

using ScalarQuantizerCFloat = ScalarQuantizerC<float>;
TEST_F(ScalarQuantizerCFloat, InvalidArgumentsPreserveTrainedQuantizer)
{
  int64_t shape[]{2, 3};
  std::vector<float> values{0, 10, 9, 254, 137, 9};
  rmm::device_uvector<float> input(values.size(), stream);
  rmm::device_uvector<int8_t> codes(values.size(), stream);
  raft::update_device(input.data(), values.data(), values.size(), stream);
  auto host = matrix_tensor(values.data(), kDLCPU, shape);
  auto device = matrix_tensor(input.data(), kDLCUDA, shape);
  auto encoded = matrix_tensor(codes.data(), kDLCUDA, shape);
  EXPECT_EQ(cuvsScalarQuantizerPerDimTransform(res, quantizer, &device, &encoded), CUVS_ERROR);
  ASSERT_EQ(cuvsScalarQuantizerPerDimTrain(res, &params, &host, quantizer), CUVS_SUCCESS);
  auto address = quantizer->addr;
  params.quantile = 0;
  EXPECT_EQ(cuvsScalarQuantizerPerDimTrain(res, &params, &host, quantizer), CUVS_ERROR);
  EXPECT_EQ(quantizer->addr, address);
  EXPECT_EQ(cuvsScalarQuantizerPerDimTransform(res, quantizer, &device, &encoded), CUVS_SUCCESS);
  device.dl_tensor.dtype.bits = 64;
  EXPECT_EQ(cuvsScalarQuantizerPerDimTransform(res, quantizer, &device, &encoded), CUVS_ERROR);
  device.dl_tensor.dtype.bits = 32;
  encoded.dl_tensor.dtype.code = kDLUInt;
  EXPECT_EQ(cuvsScalarQuantizerPerDimTransform(res, quantizer, &device, &encoded), CUVS_ERROR);
  encoded.dl_tensor.dtype.code = kDLInt;
  int64_t wrong_shape[]{2, 2};
  device.dl_tensor.shape = wrong_shape;
  EXPECT_EQ(cuvsScalarQuantizerPerDimTransform(res, quantizer, &device, &encoded), CUVS_ERROR);
  device.dl_tensor.shape = shape;
  std::vector<int8_t> host_codes(values.size());
  auto host_encoded = matrix_tensor(host_codes.data(), kDLCPU, shape);
  EXPECT_EQ(cuvsScalarQuantizerPerDimTransform(res, quantizer, &host, &host_encoded), CUVS_ERROR);
  EXPECT_EQ(cuvsScalarQuantizerPerDimInverseTransform(res, quantizer, &host_encoded, &host), CUVS_ERROR);
}
