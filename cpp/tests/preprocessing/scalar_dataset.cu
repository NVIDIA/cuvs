/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <cuvs/core/dataset/sq.hpp>
#include <gtest/gtest.h>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace cuvs::preprocessing::quantize::scalar {

using shifts_view = quantizer_const_view<float>::shifts_type;

struct ScalarDatasetInputs {
  int64_t n_samples;
  int64_t n_features;
  std::vector<float> dataset;
  std::vector<int8_t> codes;
  float tolerance;
};

std::ostream& operator<<(std::ostream& os, const ScalarDatasetInputs& inputs)
{
  return os << "n_samples:" << inputs.n_samples << " n_features:" << inputs.n_features
            << " tolerance:" << inputs.tolerance;
}

class ScalarDatasetTest : public ::testing::TestWithParam<ScalarDatasetInputs> {
 public:
  ScalarDatasetTest()
    : params_(GetParam()),
      stream(raft::resource::get_cuda_stream(handle).get()),
      dataset_(
        raft::make_device_matrix<float, int64_t>(handle, params_.n_samples, params_.n_features))
  {
  }

 protected:
  void SetUp() override
  {
    raft::update_device(
      dataset_.data_handle(), params_.dataset.data(), params_.dataset.size(), stream);
  }

  void testScalarDataset()
  {
    auto encoded = build(
      handle, params{.range = range_mode::per_dimension}, raft::make_const_mdspan(dataset_.view()));
    EXPECT_TRUE(devArrMatchHost(params_.codes.data(),
                                encoded.data_const_view().data_handle(),
                                params_.codes.size(),
                                cuvs::Compare<int8_t>(),
                                stream));
    if (params_.tolerance == 0.f) {
      EXPECT_FLOAT_EQ(std::get<shifts_view>(encoded.const_manifest_view().value).scale, 1.f);
    }
    inverse_transform(
      handle, encoded.const_manifest_view(), encoded.data_const_view(), dataset_.view());
    auto reconstructed =
      raft::make_host_matrix<float, int64_t>(params_.n_samples, params_.n_features);
    raft::update_host(reconstructed.data_handle(), dataset_.data_handle(), dataset_.size(), stream);
    raft::resource::sync_stream(handle);
    for (size_t i = 0; i < params_.dataset.size(); ++i) {
      EXPECT_TRUE(std::isfinite(reconstructed.data_handle()[i]));
      EXPECT_NEAR(reconstructed.data_handle()[i], params_.dataset[i], params_.tolerance);
    }
  }

 private:
  raft::resources handle;
  ScalarDatasetInputs params_;
  cudaStream_t stream;
  raft::device_matrix<float, int64_t> dataset_;
};

TEST(ScalarDatasetTest, CorpusAndQueriesShareQuantizer)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<float> corpus{0.f, 10.f, 9.f, 254.f, 14.f, 9.f};
  auto input = raft::make_device_matrix<float, int64_t>(res, 2, 3);
  raft::update_device(input.data_handle(), corpus.data(), corpus.size(), stream);
  auto encoded =
    build(res, params{.range = range_mode::per_dimension}, raft::make_const_mdspan(input.view()));
  ASSERT_EQ(encoded.n_rows(), 2);
  ASSERT_EQ(encoded.dim(), 3u);
  auto quantizer = encoded.const_manifest_view();
  EXPECT_FLOAT_EQ(std::get<shifts_view>(quantizer.value).scale, 1.f);
  EXPECT_TRUE(devArrMatchHost(std::vector<float>{127.f, 12.f, 9.f}.data(),
                              std::get<shifts_view>(quantizer.value).shifts.data_handle(),
                              3,
                              cuvs::Compare<float>(),
                              raft::resource::get_cuda_stream(res).get()));
  EXPECT_TRUE(devArrMatchHost(std::vector<int8_t>{-127, -2, 0, 127, 2, 0}.data(),
                              encoded.data_const_view().data_handle(),
                              6,
                              cuvs::Compare<int8_t>(),
                              raft::resource::get_cuda_stream(res).get()));

  // The corpus manifest, rather than query extrema, controls ties and saturation.
  std::vector<float> queries{127.5f, 11.5f, 1000.f, 128.5f, 9.5f, -1000.f};
  raft::update_device(input.data_handle(), queries.data(), queries.size(), stream);
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 2, 3);
  transform(res, quantizer, raft::make_const_mdspan(input.view()), codes.view());
  EXPECT_TRUE(devArrMatchHost(std::vector<int8_t>{0, 0, 127, 2, -2, -127}.data(),
                              codes.data_handle(),
                              6,
                              cuvs::Compare<int8_t>(),
                              raft::resource::get_cuda_stream(res).get()));
  inverse_transform(res, quantizer, raft::make_const_mdspan(codes.view()), input.view());
  EXPECT_TRUE(devArrMatchHost(std::vector<float>{127.f, 12.f, 136.f, 129.f, 10.f, -118.f}.data(),
                              input.data_handle(),
                              6,
                              cuvs::Compare<float>(),
                              raft::resource::get_cuda_stream(res).get()));
}

TEST(ScalarDatasetTest, PerDimensionQuantiles)
{
  raft::resources res;
  constexpr int64_t rows = 1000, dim = 3;
  std::vector<float> corpus(rows * dim);
  for (int64_t i = 0; i < rows; ++i) {
    corpus[i * dim]     = static_cast<float>(i);
    corpus[i * dim + 1] = 10000.f + 2.f * static_cast<float>(rows - i - 1);
    corpus[i * dim + 2] = 9.f;
  }
  auto input = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  raft::update_device(
    input.data_handle(), corpus.data(), corpus.size(), raft::resource::get_cuda_stream(res));
  auto encoded =
    build(res, params{.range = range_mode::per_dimension}, raft::make_const_mdspan(input.view()));
  auto quantizer = encoded.const_manifest_view();
  EXPECT_TRUE(devArrMatchHost(std::vector<float>{499.5f, 10999.f, 9.f}.data(),
                              std::get<shifts_view>(quantizer.value).shifts.data_handle(),
                              dim,
                              cuvs::Compare<float>(),
                              raft::resource::get_cuda_stream(res).get()));
  EXPECT_FLOAT_EQ(std::get<shifts_view>(quantizer.value).scale, 991.f / 127.f);
  std::vector<int8_t> codes(rows * dim);
  raft::update_host(codes.data(),
                    encoded.data_const_view().data_handle(),
                    codes.size(),
                    raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  EXPECT_EQ(codes[1], 127);
  EXPECT_EQ(codes[(rows - 1) * dim + 1], -127);
  EXPECT_EQ(codes[2], 0);
  auto full =
    train(res, params{1.f, range_mode::per_dimension}, raft::make_const_mdspan(input.view()));
  EXPECT_FLOAT_EQ(std::get<shifts_view>(full.view().value).scale, 999.f / 127.f);
  auto trimmed =
    train(res, params{0.5f, range_mode::per_dimension}, raft::make_const_mdspan(input.view()));
  EXPECT_FLOAT_EQ(std::get<shifts_view>(trimmed.view().value).scale, 499.f / 127.f);
  for (float q : {0.f, -1.f, 1.1f, std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_THROW(
      train(res, params{q, range_mode::per_dimension}, raft::make_const_mdspan(input.view())),
      raft::logic_error);
  }
}

TEST(ScalarDatasetTest, CustomParametersAndValidation)
{
  raft::resources res;
  auto mins = raft::make_device_vector<float, int64_t>(res, 2);
  auto maxs = raft::make_device_vector<float, int64_t>(res, 2);
  std::vector<float> lower{-154.f, -104.f}, upper{354.f, -96.f};
  auto stream = raft::resource::get_cuda_stream(res);
  raft::update_device(mins.data_handle(), lower.data(), lower.size(), stream);
  raft::update_device(maxs.data_handle(), upper.data(), upper.size(), stream);
  scalar::quantizer<float> owner{
    res, raft::make_const_mdspan(mins.view()), raft::make_const_mdspan(maxs.view())};
  auto quantizer = owner.view();
  EXPECT_FLOAT_EQ(std::get<shifts_view>(quantizer.value).scale, 2.f);
  EXPECT_TRUE(devArrMatchHost(std::vector<float>{100.f, -100.f}.data(),
                              std::get<shifts_view>(quantizer.value).shifts.data_handle(),
                              2,
                              cuvs::Compare<float>(),
                              raft::resource::get_cuda_stream(res).get()));
  auto input = raft::make_device_matrix<float, int64_t>(res, 1, 2);
  std::vector<float> values{104.f, -106.f};
  raft::update_device(
    input.data_handle(), values.data(), values.size(), raft::resource::get_cuda_stream(res));
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 1, 2);
  transform(res, quantizer, raft::make_const_mdspan(input.view()), codes.view());
  EXPECT_TRUE(devArrMatchHost(std::vector<int8_t>{2, -3}.data(),
                              codes.data_handle(),
                              2,
                              cuvs::Compare<int8_t>(),
                              raft::resource::get_cuda_stream(res).get()));
  auto wrong_shape = raft::make_device_matrix<int8_t, int64_t>(res, 2, 1);
  EXPECT_THROW(transform(res, quantizer, raft::make_const_mdspan(input.view()), wrong_shape.view()),
               raft::logic_error);
  EXPECT_THROW((scalar::quantizer<float>{
                 res,
                 raft::make_const_mdspan(mins.view()),
                 raft::make_device_vector_view<const float, int64_t>(maxs.data_handle(), 1)}),
               raft::logic_error);
  EXPECT_THROW((scalar::quantizer<float>{
                 res, raft::make_const_mdspan(maxs.view()), raft::make_const_mdspan(mins.view())}),
               raft::logic_error);
  EXPECT_THROW(train(res,
                     params{.range = range_mode::per_dimension},
                     raft::make_device_matrix_view<const float, int64_t>(nullptr, 0, 2)),
               raft::logic_error);
}

TEST(ScalarDatasetTest, GlobalDatasetStillWorks)
{
  raft::resources res;
  auto input = raft::make_device_matrix<float, int64_t>(res, 1, 2);
  std::vector<float> values{-1.f, 1.f};
  raft::update_device(
    input.data_handle(), values.data(), values.size(), raft::resource::get_cuda_stream(res));
  auto encoded = build(res, params{1.f}, raft::make_const_mdspan(input.view()));
  EXPECT_TRUE(devArrMatchHost(std::vector<int8_t>{-128, 127}.data(),
                              encoded.data_const_view().data_handle(),
                              2,
                              cuvs::Compare<int8_t>(),
                              raft::resource::get_cuda_stream(res).get()));
  inverse_transform(res, encoded.const_manifest_view(), encoded.data_const_view(), input.view());
  EXPECT_TRUE(devArrMatchHost(values.data(),
                              input.data_handle(),
                              2,
                              cuvs::Compare<float>(),
                              raft::resource::get_cuda_stream(res).get()));
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 1, 2);
  device_sq_dataset<float, int64_t, float> dataset{std::move(codes), quantizer<float>{-1.f, 1.f}};
  decltype(dataset)::const_view_type view = dataset.view();
  EXPECT_EQ(view.dim(), 2u);
  EXPECT_FLOAT_EQ(
    std::get<quantizer_const_view<float>::range_type>(view.const_manifest_view().value).min_, -1.f);
  EXPECT_FLOAT_EQ(
    std::get<quantizer_const_view<float>::range_type>(view.const_manifest_view().value).max_, 1.f);
}

TEST_P(ScalarDatasetTest, Result) { this->testScalarDataset(); }

const std::vector<ScalarDatasetInputs> dataset_inputs = {
  {2, 2, {10.f, -7.f, 10.f, -7.f}, {0, 0, 0, 0}, 0.f},
  {2,
   1,
   {-std::numeric_limits<float>::max(), std::numeric_limits<float>::max()},
   {-127, 127},
   std::numeric_limits<float>::max() / 254.f},
  {2,
   1,
   {-std::ldexp(124.f, 104), std::numeric_limits<float>::max()},
   {-127, 127},
   (std::ldexp(124.f, 103) + std::numeric_limits<float>::max() * 0.5f) / 127.f * 0.5f}};

INSTANTIATE_TEST_CASE_P(ScalarDatasetTests, ScalarDatasetTest, ::testing::ValuesIn(dataset_inputs));

}  // namespace cuvs::preprocessing::quantize::scalar
