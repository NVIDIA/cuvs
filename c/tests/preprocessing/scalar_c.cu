/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/core/interop.hpp"
#include <cuvs/core/cuda_fp16.hpp>
#include <cuvs/preprocessing/quantize/scalar.h>

#include <cuda/std/limits>
#include <gtest/gtest.h>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <vector>

namespace {

template <typename T>
class ScalarQuantizerC : public ::testing::Test {
 protected:
  cuvsResources_t res{};
  cudaStream_t stream{};
  cuvsScalarQuantizerParams_t params{};
  cuvsScalarQuantizer_t quantizer{};
  std::vector<T> values{0.f, 10.f, 9.f, 254.f, 137.f, 9.f};
  std::vector<int8_t> host_codes = std::vector<int8_t>(values.size());
  rmm::device_uvector<T> input{0, stream};
  rmm::device_uvector<int8_t> codes{0, stream};
  DLManagedTensor host{}, device{}, encoded{}, host_encoded{};

  void SetUp() override
  {
    ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);
    ASSERT_EQ(cuvsStreamGet(res, &stream), CUVS_SUCCESS);
    ASSERT_EQ(cuvsScalarQuantizerParamsCreate(&params), CUVS_SUCCESS);
    EXPECT_EQ(params->mode, CUVS_SCALAR_GLOBAL);
    params->quantile = 1.f;
    ASSERT_EQ(cuvsScalarQuantizerCreate(&quantizer), CUVS_SUCCESS);
    input.resize(values.size(), stream);
    codes.resize(values.size(), stream);
    raft::update_device(input.data(), values.data(), values.size(), stream);
    cuvs::core::to_dlpack(raft::make_host_matrix_view<T, int64_t>(values.data(), 2, 3), &host);
    cuvs::core::to_dlpack(raft::make_device_matrix_view<T, int64_t>(input.data(), 2, 3), &device);
    cuvs::core::to_dlpack(raft::make_device_matrix_view<int8_t, int64_t>(codes.data(), 2, 3), &encoded);
    cuvs::core::to_dlpack(raft::make_host_matrix_view<int8_t, int64_t>(host_codes.data(), 2, 3), &host_encoded);
    ASSERT_EQ(cuvsStreamSync(res), CUVS_SUCCESS);
  }

  void TearDown() override
  {
    EXPECT_EQ(cuvsStreamSync(res), CUVS_SUCCESS);
    EXPECT_EQ(cuvsScalarQuantizerDestroy(quantizer), CUVS_SUCCESS);
    EXPECT_EQ(cuvsScalarQuantizerParamsDestroy(params), CUVS_SUCCESS);
    for (auto tensor : {&host, &device, &encoded, &host_encoded}) {
      if (tensor->deleter) { tensor->deleter(tensor); }
    }
    EXPECT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
  }

  cuvsError_t train(cuvsScalarQuantizationMode mode, DLManagedTensor& data)
  {
    params->mode = mode;
    return cuvsScalarQuantizerTrain(res, params, &data, quantizer);
  }

  void check_per_dimension_round_trip()
  {
    ASSERT_EQ(cuvsScalarQuantizerTransform(res, quantizer, &device, &encoded), CUVS_SUCCESS);
    ASSERT_EQ(cuvsScalarQuantizerInverseTransform(res, quantizer, &encoded, &device), CUVS_SUCCESS);
    std::vector<T> decoded(values.size());
    raft::update_host(host_codes.data(), codes.data(), host_codes.size(), stream);
    raft::update_host(decoded.data(), input.data(), decoded.size(), stream);
    ASSERT_EQ(cuvsStreamSync(res), CUVS_SUCCESS);
    EXPECT_EQ(host_codes, (std::vector<int8_t>{-127, -127, 0, 127, 127, 0}));
    EXPECT_EQ(decoded, values);
  }
};

}  // namespace

using ScalarTypes = ::testing::Types<half, float, double>;
TYPED_TEST_SUITE(ScalarQuantizerC, ScalarTypes);

TYPED_TEST(ScalarQuantizerC, PerDimensionRoundTrip)
{
  ASSERT_EQ(this->train(CUVS_SCALAR_PER_DIMENSION, this->device), CUVS_SUCCESS);
  EXPECT_EQ(this->quantizer->min_, double(cuda::std::numeric_limits<TypeParam>::min()));
  EXPECT_EQ(this->quantizer->max_, double(cuda::std::numeric_limits<TypeParam>::max()));
  this->check_per_dimension_round_trip();
}

TYPED_TEST(ScalarQuantizerC, GlobalBoundsAndRoundTrip)
{
  std::fill_n(this->values.begin(), 3, TypeParam{-3});
  std::fill_n(this->values.begin() + 3, 3, TypeParam{7});
  auto original = this->values;
  ASSERT_EQ(this->train(CUVS_SCALAR_GLOBAL, this->host), CUVS_SUCCESS);
  EXPECT_EQ(this->quantizer->min_, -3);
  EXPECT_EQ(this->quantizer->max_, 7);
  ASSERT_EQ(cuvsScalarQuantizerTransform(this->res, this->quantizer, &this->host, &this->host_encoded), CUVS_SUCCESS);
  EXPECT_EQ(this->host_codes, (std::vector<int8_t>{-128, -128, -128, 127, 127, 127}));
  ASSERT_EQ(cuvsScalarQuantizerInverseTransform(this->res, this->quantizer, &this->host_encoded, &this->host), CUVS_SUCCESS);
  EXPECT_EQ(this->values, original);
  this->quantizer->min_ = 1;
  this->quantizer->max_ = 5;
  ASSERT_EQ(cuvsScalarQuantizerInverseTransform(this->res, this->quantizer, &this->host_encoded, &this->host), CUVS_SUCCESS);
  EXPECT_EQ(this->values, (std::vector<TypeParam>{1.f, 1.f, 1.f, 5.f, 5.f, 5.f}));
}

using ScalarQuantizerCFloat = ScalarQuantizerC<float>;
TEST_F(ScalarQuantizerCFloat, HostRetraining)
{
  ASSERT_EQ(train(CUVS_SCALAR_PER_DIMENSION, device), CUVS_SUCCESS);
  ASSERT_EQ(train(CUVS_SCALAR_PER_DIMENSION, host), CUVS_SUCCESS);
  check_per_dimension_round_trip();
}

TEST_F(ScalarQuantizerCFloat, RetrainingChangesMode)
{
  ASSERT_EQ(train(CUVS_SCALAR_GLOBAL, host), CUVS_SUCCESS);
  ASSERT_EQ(train(CUVS_SCALAR_PER_DIMENSION, host), CUVS_SUCCESS);
  check_per_dimension_round_trip();
  ASSERT_EQ(train(CUVS_SCALAR_GLOBAL, host), CUVS_SUCCESS);
  EXPECT_EQ(quantizer->min_, 0);
  EXPECT_EQ(quantizer->max_, 254);
  EXPECT_EQ(cuvsScalarQuantizerTransform(res, quantizer, &host, &host_encoded), CUVS_SUCCESS);
}

TEST_F(ScalarQuantizerCFloat, InvalidTrainingPreservesQuantizer)
{
  ASSERT_EQ(train(CUVS_SCALAR_PER_DIMENSION, host), CUVS_SUCCESS);
  params->quantile = 0;
  EXPECT_EQ(train(CUVS_SCALAR_PER_DIMENSION, host), CUVS_ERROR);
  params->quantile = 1;
  EXPECT_EQ(train(static_cast<cuvsScalarQuantizationMode>(2), host), CUVS_ERROR);
  check_per_dimension_round_trip();
}

TEST_F(ScalarQuantizerCFloat, RejectsMismatchedTypes)
{
  ASSERT_EQ(train(CUVS_SCALAR_PER_DIMENSION, host), CUVS_SUCCESS);
  device.dl_tensor.dtype.bits = 64;
  EXPECT_EQ(cuvsScalarQuantizerTransform(res, quantizer, &device, &encoded), CUVS_ERROR);
  device.dl_tensor.dtype.bits = 32;
  encoded.dl_tensor.dtype.code = kDLUInt;
  EXPECT_EQ(cuvsScalarQuantizerTransform(res, quantizer, &device, &encoded), CUVS_ERROR);
}

TEST_F(ScalarQuantizerCFloat, RejectsMismatchedWidth)
{
  ASSERT_EQ(train(CUVS_SCALAR_PER_DIMENSION, host), CUVS_SUCCESS);
  device.dl_tensor.shape[1] = 2;
  EXPECT_EQ(cuvsScalarQuantizerTransform(res, quantizer, &device, &encoded), CUVS_ERROR);
}

TEST_F(ScalarQuantizerCFloat, RejectsHostPerDimensionTransforms)
{
  ASSERT_EQ(train(CUVS_SCALAR_PER_DIMENSION, host), CUVS_SUCCESS);
  EXPECT_EQ(cuvsScalarQuantizerTransform(res, quantizer, &host, &host_encoded), CUVS_ERROR);
  EXPECT_EQ(cuvsScalarQuantizerInverseTransform(res, quantizer, &host_encoded, &host), CUVS_ERROR);
}
