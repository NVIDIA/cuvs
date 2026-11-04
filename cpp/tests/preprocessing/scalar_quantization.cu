/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include <cuvs/core/dataset/sq.hpp>
#include <cuvs/preprocessing/quantize/scalar.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/init.cuh>
#include <raft/stats/stddev.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace cuvs::preprocessing::quantize::scalar {

template <typename T>
struct QuantizationInputs {
  cuvs::preprocessing::quantize::scalar::params quantization_params;
  int rows;
  int cols;
  T min            = T(-1.0);
  T max            = T(1.0);
  double threshold = 2e-2;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const QuantizationInputs<T>& inputs)
{
  return os << "quantization_quantile:<" << inputs.quantization_params.quantile
            << "> rows:" << inputs.rows << " cols:" << inputs.cols << " min:" << (double)inputs.min
            << " max:" << (double)inputs.max;
}

template <typename T, typename QuantI>
class QuantizationTest : public ::testing::TestWithParam<QuantizationInputs<T>> {
 public:
  QuantizationTest()
    : params_(::testing::TestWithParam<QuantizationInputs<T>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle).get()),
      input_(0, stream)
  {
  }

  double getRelativeErrorStddev(const T* array_a, const T* array_b, size_t size, float quantile)
  {
    // relative error elementwise
    rmm::device_uvector<double> relative_error(size, stream);
    raft::linalg::binaryOp(
      relative_error.data(),
      array_a,
      array_b,
      size,
      [] __device__(double a, double b) {
        return a != b ? (raft::abs(a - b) / raft::max(raft::abs(a), raft::abs(b))) : 0;
      },
      stream);

    // sort by size --> remove largest errors to account for quantile chosen
    thrust::sort(raft::resource::get_thrust_policy(handle),
                 relative_error.data(),
                 relative_error.data() + size);
    int elements_to_consider =
      std::ceil(double(params_.quantization_params.quantile) * double(size));

    rmm::device_uvector<double> mu(1, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(mu.data(), 0, sizeof(double), stream));

    rmm::device_uvector<double> error_stddev(1, stream);
    raft::stats::stddev<true>(error_stddev.data(),
                              relative_error.data(),
                              mu.data(),
                              1,
                              elements_to_consider,
                              false,
                              stream);

    double error_stddev_h;
    raft::update_host(&error_stddev_h, error_stddev.data(), 1, stream);
    raft::resource::sync_stream(handle, stream);
    return error_stddev_h;
  }

 protected:
  void testScalarQuantization()
  {
    // dataset identical on host / device
    auto dataset = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
      (const T*)(input_.data()), rows_, cols_);
    auto dataset_h = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
      (const T*)(host_input_.data()), rows_, cols_);

    size_t print_size = std::min(input_.size(), 20ul);

    // `train` samples this many rows into a device buffer that stays alive for the whole call, so
    // whatever else it allocates, the peak cannot be below the subsample. Mirrors the sizing in
    // quantile_min_max, integer division included.
    constexpr size_t max_num_samples = 1000000;
    size_t n_sample_rows =
      std::min<size_t>(max_num_samples / static_cast<size_t>(cols_), static_cast<size_t>(rows_));
    size_t min_train_alloc = n_sample_rows * static_cast<size_t>(cols_) * sizeof(T);

    // train quantizer_1 on device
    // `train` sizes its allocations from the dataset extents and the queried sort workspace, so
    // the prediction is exact once the sort workspace dominates the peak. On very small inputs it
    // is not: the peak then falls inside raft::matrix::sample_rows, where skipping the gather
    // keeps the index buffer alive across the peak, and the dry run reports 8 bytes per sampled
    // row more than the real run. Hence the upper-bound check rather than an exact match.
    cuvs::preprocessing::quantize::scalar::quantizer<T> quantizer_1;
    cuvs::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        quantizer_1 =
          cuvs::preprocessing::quantize::scalar::train(h, params_.quantization_params, dataset);
      },
      cuvs::alloc_behavior::DATA_DRIVEN,
      min_train_alloc);
    std::cerr
      << "Q1: min = "
      << (double)std::get<typename decltype(quantizer_1)::range_type>(quantizer_1.value).min_
      << ", max = "
      << (double)std::get<typename decltype(quantizer_1)::range_type>(quantizer_1.value).max_
      << std::endl;

    {
      auto quantized_input_h = raft::make_host_matrix<QuantI, int64_t>(rows_, cols_);
      auto quantized_input_d = raft::make_device_matrix<QuantI, int64_t>(handle, rows_, cols_);
      cuvs::execute_with_dry_run_check(
        handle,
        [&](raft::resources const& h) {
          cuvs::preprocessing::quantize::scalar::transform(
            h, quantizer_1, dataset, quantized_input_d.view());
        },
        cuvs::alloc_behavior::NO_ALLOCATIONS);
      cuvs::execute_with_dry_run_check(
        handle,
        [&](raft::resources const& h) {
          cuvs::preprocessing::quantize::scalar::transform(
            h, quantizer_1, dataset_h, quantized_input_h.view());
        },
        cuvs::alloc_behavior::NO_ALLOCATIONS);

      {
        raft::print_device_vector("Input array: ", input_.data(), print_size, std::cerr);

        rmm::device_uvector<int> quantization_for_print(print_size, stream);
        raft::linalg::unaryOp(quantization_for_print.data(),
                              quantized_input_d.data_handle(),
                              print_size,
                              raft::cast_op<int>{},
                              stream);
        raft::resource::sync_stream(handle, stream);
        raft::print_device_vector(
          "Quantized array 1: ", quantization_for_print.data(), print_size, std::cerr);
      }

      // test (inverse) transform host/device equal
      ASSERT_TRUE(devArrMatchHost(quantized_input_h.data_handle(),
                                  quantized_input_d.data_handle(),
                                  input_.size(),
                                  cuvs::Compare<QuantI>(),
                                  stream));

      auto quantized_input_h_const_view = raft::make_host_matrix_view<const QuantI, int64_t>(
        quantized_input_h.data_handle(), rows_, cols_);
      auto re_transformed_input_h = raft::make_host_matrix<T, int64_t>(rows_, cols_);
      cuvs::execute_with_dry_run_check(
        handle,
        [&](raft::resources const& h) {
          cuvs::preprocessing::quantize::scalar::inverse_transform(
            h, quantizer_1, quantized_input_h_const_view, re_transformed_input_h.view());
        },
        cuvs::alloc_behavior::NO_ALLOCATIONS);
      // Also covers the rows outside the range-fitting sample on the large input.
      for (int64_t row = 0; row < rows_; ++row) {
        double error = 0;
        for (int64_t j = 0; j < cols_; ++j) {
          auto const i = row * cols_ + j;
          error        = std::hypot(error,
                             static_cast<double>(host_input_[i]) -
                               static_cast<double>(re_transformed_input_h.data_handle()[i]));
        }
        EXPECT_GE(static_cast<double>(quantizer_1.max_error), error);
      }

      auto quantized_input_d_const_view = raft::make_device_matrix_view<const QuantI, int64_t>(
        quantized_input_d.data_handle(), rows_, cols_);
      auto re_transformed_input_d = raft::make_device_matrix<T, int64_t>(handle, rows_, cols_);
      cuvs::execute_with_dry_run_check(
        handle,
        [&](raft::resources const& h) {
          cuvs::preprocessing::quantize::scalar::inverse_transform(
            h, quantizer_1, quantized_input_d_const_view, re_transformed_input_d.view());
        },
        cuvs::alloc_behavior::NO_ALLOCATIONS);
      raft::print_device_vector(
        "re-transformed array: ", re_transformed_input_d.data_handle(), print_size, std::cerr);

      {
        double l2_error = getRelativeErrorStddev(dataset.data_handle(),
                                                 re_transformed_input_d.data_handle(),
                                                 input_.size(),
                                                 params_.quantization_params.quantile);
        std::cerr << "error stddev = " << l2_error << ", threshold = " << params_.threshold
                  << std::endl;
        // test (inverse) transform close to original dataset
        ASSERT_TRUE(l2_error < params_.threshold);
      }
    }

    // train quantizer_2 on host
    cuvs::preprocessing::quantize::scalar::quantizer<T> quantizer_2;
    cuvs::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        quantizer_2 =
          cuvs::preprocessing::quantize::scalar::train(h, params_.quantization_params, dataset_h);
      },
      cuvs::alloc_behavior::DATA_DRIVEN,
      min_train_alloc);
    std::cerr
      << "Q2: min = "
      << (double)std::get<typename decltype(quantizer_2)::range_type>(quantizer_2.value).min_
      << ", max = "
      << (double)std::get<typename decltype(quantizer_2)::range_type>(quantizer_2.value).max_
      << std::endl;

    // check both quantizers are the same (valid if sampling is identical)
    if (input_.size() <= 1000000) {
      ASSERT_TRUE(
        (double)std::get<typename decltype(quantizer_1)::range_type>(quantizer_1.value).min_ ==
        (double)std::get<typename decltype(quantizer_2)::range_type>(quantizer_2.value).min_);
      ASSERT_TRUE(
        (double)std::get<typename decltype(quantizer_1)::range_type>(quantizer_1.value).max_ ==
        (double)std::get<typename decltype(quantizer_2)::range_type>(quantizer_2.value).max_);
    }

    {
      // test transform host/device equal
      auto quantized_input_h = raft::make_host_matrix<QuantI, int64_t>(rows_, cols_);
      auto quantized_input_d = raft::make_device_matrix<QuantI, int64_t>(handle, rows_, cols_);
      cuvs::preprocessing::quantize::scalar::transform(
        handle, quantizer_2, dataset, quantized_input_d.view());
      cuvs::preprocessing::quantize::scalar::transform(
        handle, quantizer_2, dataset_h, quantized_input_h.view());

      {
        rmm::device_uvector<int> quantization_for_print(print_size, stream);
        raft::linalg::unaryOp(quantization_for_print.data(),
                              quantized_input_d.data_handle(),
                              print_size,
                              raft::cast_op<int>{},
                              stream);
        raft::resource::sync_stream(handle, stream);
        raft::print_device_vector(
          "Quantized array 2: ", quantization_for_print.data(), print_size, std::cerr);
      }

      ASSERT_TRUE(devArrMatchHost(quantized_input_h.data_handle(),
                                  quantized_input_d.data_handle(),
                                  input_.size(),
                                  cuvs::Compare<QuantI>(),
                                  stream));
    }

    // sort_by_key (input, quantization) -- check <= on result
    {
      auto quantized_input = raft::make_device_matrix<QuantI, int64_t>(handle, rows_, cols_);
      cuvs::preprocessing::quantize::scalar::transform(
        handle, quantizer_1, dataset, quantized_input.view());
      thrust::sort_by_key(raft::resource::get_thrust_policy(handle),
                          input_.data(),
                          input_.data() + input_.size(),
                          quantized_input.data_handle());
      std::vector<QuantI> quantized_input_sorted_host(input_.size());
      raft::update_host(
        quantized_input_sorted_host.data(), quantized_input.data_handle(), input_.size(), stream);
      raft::resource::sync_stream(handle, stream);

      for (size_t i = 0; i < input_.size() - 1; ++i) {
        ASSERT_TRUE(quantized_input_sorted_host[i] <= quantized_input_sorted_host[i + 1]);
      }
    }
  }

  void SetUp() override
  {
    rows_ = params_.rows;
    cols_ = params_.cols;

    int n_elements = rows_ * cols_;
    input_.resize(n_elements, stream);
    host_input_.resize(n_elements);

    // random input
    unsigned long long int seed = 1234ULL;
    raft::random::RngState r(seed);
    uniform(handle, r, input_.data(), input_.size(), params_.min, params_.max);

    raft::update_host(host_input_.data(), input_.data(), input_.size(), stream);

    raft::resource::sync_stream(handle, stream);
  }

 private:
  raft::resources handle;
  cudaStream_t stream;

  QuantizationInputs<T> params_;
  int rows_;
  int cols_;
  rmm::device_uvector<T> input_;
  std::vector<T> host_input_;
};

using shifts_view = quantizer_const_view<float>::shifts_type;

struct PerDimensionInputs {
  int64_t n_samples;
  int64_t n_features;
  std::vector<float> dataset;
  std::vector<int8_t> codes;
  float tolerance;
};

std::ostream& operator<<(std::ostream& os, const PerDimensionInputs& inputs)
{
  return os << "n_samples:" << inputs.n_samples << " n_features:" << inputs.n_features
            << " tolerance:" << inputs.tolerance;
}

class PerDimensionRangeTest : public ::testing::TestWithParam<PerDimensionInputs> {
 public:
  PerDimensionRangeTest()
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

  void testPerDimensionQuantization()
  {
    auto quantizer = train(
      handle, params{.range = range_mode::per_dimension}, raft::make_const_mdspan(dataset_.view()));
    auto codes =
      raft::make_device_matrix<int8_t, int64_t>(handle, params_.n_samples, params_.n_features);
    transform(handle, quantizer, raft::make_const_mdspan(dataset_.view()), codes.view());
    EXPECT_TRUE(devArrMatchHost(params_.codes.data(),
                                codes.data_handle(),
                                params_.codes.size(),
                                cuvs::Compare<int8_t>(),
                                stream));
    if (params_.tolerance == 0.f) {
      EXPECT_FLOAT_EQ(std::get<shifts_view>(quantizer.view().value).scale, 1.f);
    }
    inverse_transform(handle, quantizer, raft::make_const_mdspan(codes.view()), dataset_.view());
    auto reconstructed =
      raft::make_host_matrix<float, int64_t>(params_.n_samples, params_.n_features);
    raft::update_host(reconstructed.data_handle(), dataset_.data_handle(), dataset_.size(), stream);
    raft::resource::sync_stream(handle);
    for (size_t i = 0; i < params_.dataset.size(); ++i) {
      EXPECT_TRUE(std::isfinite(reconstructed.data_handle()[i]));
      EXPECT_NEAR(reconstructed.data_handle()[i], params_.dataset[i], params_.tolerance);
    }
    for (int64_t row = 0; row < params_.n_samples; ++row) {
      double error = 0;
      for (int64_t j = 0; j < params_.n_features; ++j) {
        auto const i = row * params_.n_features + j;
        error        = std::hypot(error,
                           static_cast<double>(params_.dataset[i]) -
                             static_cast<double>(reconstructed.data_handle()[i]));
      }
      EXPECT_GE(static_cast<double>(quantizer.max_error), error);
      if (params_.tolerance == 0.f) { EXPECT_EQ(quantizer.max_error, 0.f); }
    }
  }

 private:
  raft::resources handle;
  PerDimensionInputs params_;
  cudaStream_t stream;
  raft::device_matrix<float, int64_t> dataset_;
};

template <typename T>
class PerDimensionQuantizationTest : public ::testing::Test {};
using PerDimensionTypes = ::testing::Types<half, float, double>;
TYPED_TEST_SUITE(PerDimensionQuantizationTest, PerDimensionTypes);

TYPED_TEST(PerDimensionQuantizationTest, TrainingReconstructionError)
{
  using T = TypeParam;
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  // A partial warp plus a second column per lane, with clipped outliers in different rows.
  constexpr int64_t rows = 9, dim = 35;
  std::vector<T> values(rows * dim);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t j = 0; j < dim; ++j) {
      values[row * dim + j] = static_cast<T>((row - 4) * 0.17 + (j % 5) * 20.0);
    }
  }
  values.front() = T{-300.f};
  values.back()  = T{500.f};
  auto input     = raft::make_device_matrix<T, int64_t>(res, rows, dim);
  raft::update_device(input.data_handle(), values.data(), values.size(), stream);
  auto codes   = raft::make_device_matrix<int8_t, int64_t>(res, rows, dim);
  auto decoded = raft::make_device_matrix<T, int64_t>(res, rows, dim);
  std::vector<T> reconstructed(values.size());
  for (auto mode : {range_mode::global, range_mode::per_dimension}) {
    for (float quantile : {1.f, 0.5f}) {
      SCOPED_TRACE(::testing::Message()
                   << "range=" << static_cast<int>(mode) << " quantile=" << quantile);
      auto owner = train(res, params{quantile, mode}, raft::make_const_mdspan(input.view()));
      auto view  = owner.view();
      EXPECT_EQ(owner.max_error, view.max_error);
      transform(res, view, raft::make_const_mdspan(input.view()), codes.view());
      inverse_transform(res, view, raft::make_const_mdspan(codes.view()), decoded.view());
      raft::update_host(reconstructed.data(), decoded.data_handle(), reconstructed.size(), stream);
      raft::resource::sync_stream(res);
      long double maximum = 0;
      for (int64_t row = 0; row < rows; ++row) {
        long double error = 0;
        for (int64_t j = 0; j < dim; ++j) {
          auto const i     = row * dim + j;
          auto const delta = static_cast<long double>(static_cast<double>(values[i])) -
                             static_cast<long double>(static_cast<double>(reconstructed[i]));
          error = std::hypot(error, delta);
        }
        EXPECT_GE(static_cast<long double>(owner.max_error), error);
        maximum = std::max(maximum, error);
      }
      EXPECT_GT(maximum, 0);
      EXPECT_NEAR(static_cast<double>(owner.max_error),
                  static_cast<double>(maximum),
                  static_cast<double>(maximum) * 2e-6);
      if (quantile < 1.f) { EXPECT_GT(owner.max_error, 100); }
      if (mode == range_mode::global) {
        auto host = train(res,
                          params{quantile},
                          raft::make_host_matrix_view<const T, int64_t>(values.data(), rows, dim));
        EXPECT_GE(static_cast<long double>(host.max_error), maximum);
        EXPECT_NEAR(static_cast<double>(host.max_error),
                    static_cast<double>(owner.max_error),
                    static_cast<double>(maximum) * 1e-6);
      }
    }
  }
}

TEST(PerDimensionRangeTest, DoubleReconstructionErrorExtremes)
{
  raft::resources res;
  auto stream  = raft::resource::get_cuda_stream(res);
  auto input   = raft::make_device_matrix<double, int64_t>(res, 3, 2);
  auto codes   = raft::make_device_matrix<int8_t, int64_t>(res, 3, 2);
  auto decoded = raft::make_device_matrix<double, int64_t>(res, 3, 2);
  // Squaring these residuals would underflow or overflow, although their L2 norm is finite.
  for (double scale : {1e-300, 1e200}) {
    SCOPED_TRACE(scale);
    std::vector<double> values{-scale, -scale, scale * 0.3, scale * 0.3, scale, scale};
    raft::update_device(input.data_handle(), values.data(), values.size(), stream);
    auto owner =
      train(res, params{1.f, range_mode::per_dimension}, raft::make_const_mdspan(input.view()));
    transform(res, owner, raft::make_const_mdspan(input.view()), codes.view());
    inverse_transform(res, owner, raft::make_const_mdspan(codes.view()), decoded.view());
    std::vector<double> reconstructed(values.size());
    raft::update_host(reconstructed.data(), decoded.data_handle(), reconstructed.size(), stream);
    raft::resource::sync_stream(res);
    double maximum = 0;
    for (int64_t row = 0; row < 3; ++row) {
      auto const i = row * 2;
      maximum      = std::max(
        maximum, std::hypot(values[i] - reconstructed[i], values[i + 1] - reconstructed[i + 1]));
    }
    EXPECT_GT(maximum, 0);
    EXPECT_TRUE(std::isfinite(owner.max_error));
    EXPECT_GE(owner.max_error, maximum);
    EXPECT_NEAR(owner.max_error, maximum, maximum * 2e-6);
  }
}

TYPED_TEST(PerDimensionQuantizationTest, CorpusAndQueriesShareQuantizer)
{
  using T           = TypeParam;
  using shifts_view = typename quantizer_const_view<T>::shifts_type;
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<T> corpus{0.f, 10.f, 9.f, 254.f, 14.f, 9.f};
  auto input = raft::make_device_matrix<T, int64_t>(res, 2, 3);
  raft::update_device(input.data_handle(), corpus.data(), corpus.size(), stream);
  auto owner =
    train(res, params{.range = range_mode::per_dimension}, raft::make_const_mdspan(input.view()));
  auto quantizer = owner.view();
  EXPECT_EQ(owner.max_error, 0);
  EXPECT_EQ(quantizer.max_error, 0);
  EXPECT_FLOAT_EQ(std::get<shifts_view>(quantizer.value).scale, 1.f);
  {
    std::vector<T> actual(3);
    std::vector<double> expected{127.f, 12.f, 9.f};
    raft::update_host(actual.data(),
                      std::get<shifts_view>(quantizer.value).shifts.data_handle(),
                      actual.size(),
                      stream);
    raft::resource::sync_stream(res);
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_EQ(static_cast<double>(actual[i]), expected[i]);
    }
  }
  // The corpus manifest, rather than query extrema, controls ties and saturation.
  std::vector<T> queries{127.5f, 11.5f, 1000.f, 128.5f, 9.5f, -1000.f};
  raft::update_device(input.data_handle(), queries.data(), queries.size(), stream);
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 2, 3);
  transform(res, quantizer, raft::make_const_mdspan(input.view()), codes.view());
  EXPECT_TRUE(devArrMatchHost(std::vector<int8_t>{0, 0, 127, 2, -2, -127}.data(),
                              codes.data_handle(),
                              6,
                              cuvs::Compare<int8_t>(),
                              raft::resource::get_cuda_stream(res).get()));
  inverse_transform(res, quantizer, raft::make_const_mdspan(codes.view()), input.view());
  {
    std::vector<T> actual(6);
    std::vector<double> expected{127.f, 12.f, 136.f, 129.f, 10.f, -118.f};
    raft::update_host(actual.data(), input.data_handle(), actual.size(), stream);
    raft::resource::sync_stream(res);
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_EQ(static_cast<double>(actual[i]), expected[i]);
    }
  }
}

TEST(PerDimensionRangeTest, AdjacentHalfBounds)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<half> values{half{1.f}, half{1.f + 1.f / 1024.f}};
  auto input = raft::make_device_matrix<half, int64_t>(res, 2, 1);
  raft::update_device(input.data_handle(), values.data(), values.size(), stream);
  auto quantizer =
    train(res, params{1.f, range_mode::per_dimension}, raft::make_const_mdspan(input.view()));
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 2, 1);
  transform(res, quantizer, raft::make_const_mdspan(input.view()), codes.view());
  inverse_transform(res, quantizer, raft::make_const_mdspan(codes.view()), input.view());
  std::vector<half> reconstructed(2);
  raft::update_host(reconstructed.data(), input.data_handle(), reconstructed.size(), stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(static_cast<float>(reconstructed[0]), static_cast<float>(values[0]));
  EXPECT_EQ(static_cast<float>(reconstructed[1]), static_cast<float>(values[1]));
}

TEST(PerDimensionRangeTest, PerDimensionQuantiles)
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
  auto owner =
    train(res, params{.range = range_mode::per_dimension}, raft::make_const_mdspan(input.view()));
  auto quantizer = owner.view();
  EXPECT_TRUE(devArrMatchHost(std::vector<float>{499.5f, 10999.f, 9.f}.data(),
                              std::get<shifts_view>(quantizer.value).shifts.data_handle(),
                              dim,
                              cuvs::Compare<float>(),
                              raft::resource::get_cuda_stream(res).get()));
  EXPECT_FLOAT_EQ(std::get<shifts_view>(quantizer.value).scale, 991.f / 127.f);
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

TYPED_TEST(PerDimensionQuantizationTest, CustomParametersAndValidation)
{
  using T           = TypeParam;
  using shifts_view = typename quantizer_const_view<T>::shifts_type;
  raft::resources res;
  auto mins = raft::make_device_vector<T, int64_t>(res, 2);
  auto maxs = raft::make_device_vector<T, int64_t>(res, 2);
  std::vector<T> lower{-154.f, -104.f}, upper{354.f, -96.f};
  auto stream = raft::resource::get_cuda_stream(res);
  raft::update_device(mins.data_handle(), lower.data(), lower.size(), stream);
  raft::update_device(maxs.data_handle(), upper.data(), upper.size(), stream);
  scalar::quantizer<T> owner{
    res, raft::make_const_mdspan(mins.view()), raft::make_const_mdspan(maxs.view())};
  auto quantizer = owner.view();
  EXPECT_TRUE(std::isinf(owner.max_error));
  EXPECT_TRUE(std::isinf(quantizer.max_error));
  EXPECT_TRUE(std::isinf((scalar::quantizer<T>{T{-1}, T{1}}.view().max_error)));
  EXPECT_FLOAT_EQ(std::get<shifts_view>(quantizer.value).scale, 2.f);
  {
    std::vector<T> actual(2);
    std::vector<double> expected{100.f, -100.f};
    raft::update_host(actual.data(),
                      std::get<shifts_view>(quantizer.value).shifts.data_handle(),
                      actual.size(),
                      stream);
    raft::resource::sync_stream(res);
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_EQ(static_cast<double>(actual[i]), expected[i]);
    }
  }
  auto input = raft::make_device_matrix<T, int64_t>(res, 1, 2);
  std::vector<T> values{104.f, -106.f};
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
  EXPECT_THROW(
    (scalar::quantizer<T>{res,
                          raft::make_const_mdspan(mins.view()),
                          raft::make_device_vector_view<const T, int64_t>(maxs.data_handle(), 1)}),
    raft::logic_error);
  EXPECT_THROW((scalar::quantizer<T>{
                 res, raft::make_const_mdspan(maxs.view()), raft::make_const_mdspan(mins.view())}),
               raft::logic_error);
  EXPECT_THROW(train(res,
                     params{.range = range_mode::per_dimension},
                     raft::make_device_matrix_view<const T, int64_t>(nullptr, 0, 2)),
               raft::logic_error);
}

TEST(QuantizationDatasetTest, GlobalRangeDataset)
{
  raft::resources res;
  auto input = raft::make_device_matrix<float, int64_t>(res, 1, 2);
  std::vector<float> values{-1.f, 1.f};
  auto stream = raft::resource::get_cuda_stream(res);
  raft::update_device(input.data_handle(), values.data(), values.size(), stream);
  auto encoded = build(res, params{1.f}, raft::make_const_mdspan(input.view()));
  EXPECT_EQ(encoded.const_manifest_view().max_error, 0.f);
  EXPECT_EQ(encoded.n_rows(), 1);
  EXPECT_EQ(encoded.dim(), 2u);
  EXPECT_TRUE(devArrMatchHost(std::vector<int8_t>{-128, 127}.data(),
                              encoded.data_const_view().data_handle(),
                              2,
                              cuvs::Compare<int8_t>(),
                              stream.get()));
  auto range =
    std::get<quantizer_const_view<float>::range_type>(encoded.const_manifest_view().value);
  EXPECT_FLOAT_EQ(range.min_, -1.f);
  EXPECT_FLOAT_EQ(range.max_, 1.f);
}

TEST(QuantizationDatasetTest, PerDimensionDataset)
{
  raft::resources res;
  auto input = raft::make_device_matrix<float, int64_t>(res, 2, 2);
  std::vector<float> values{0.f, 10.f, 254.f, 14.f};
  auto stream = raft::resource::get_cuda_stream(res);
  raft::update_device(input.data_handle(), values.data(), values.size(), stream);
  auto encoded =
    build(res, params{.range = range_mode::per_dimension}, raft::make_const_mdspan(input.view()));
  EXPECT_EQ(encoded.const_manifest_view().max_error, 0.f);
  EXPECT_EQ(encoded.n_rows(), 2);
  EXPECT_EQ(encoded.dim(), 2u);
  EXPECT_TRUE(devArrMatchHost(std::vector<int8_t>{-127, -2, 127, 2}.data(),
                              encoded.data_const_view().data_handle(),
                              4,
                              cuvs::Compare<int8_t>(),
                              stream.get()));
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 2, 2);
  transform(
    res, encoded.const_manifest_view(), raft::make_const_mdspan(input.view()), codes.view());
  EXPECT_TRUE(devArrMatch(encoded.data_const_view().data_handle(),
                          codes.data_handle(),
                          4,
                          cuvs::Compare<int8_t>(),
                          stream.get()));
}

TYPED_TEST(PerDimensionQuantizationTest, BatchedHostDataset)
{
  using T                = TypeParam;
  constexpr int64_t rows = 37, dim = 35;
  std::vector<T> values(rows * dim);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < dim; ++j) {
      values[i * dim + j] = T((i - 18) * .17 + (j % 5) * 20.0);
    }
  }
  // Put the largest clipping error before the final batch to exercise the running maximum.
  values.front() = T{-1200.f};
  values.back()  = T{500.f};
  for (bool prefetch : {false, true}) {
    raft::resources res;
    if (prefetch) {
      raft::resource::set_cuda_stream_pool(res, std::make_shared<rmm::cuda_stream_pool>(1));
    }
    auto stream = raft::resource::get_cuda_stream(res);
    auto input  = raft::make_device_matrix<T, int64_t>(res, rows, dim);
    raft::update_device(input.data_handle(), values.data(), values.size(), stream);
    auto pinned = raft::make_pinned_matrix<T, int64_t>(res, rows, dim);
    std::copy(values.begin(), values.end(), pinned.data_handle());
    for (auto mode : {range_mode::global, range_mode::per_dimension}) {
      for (float quantile : {1.f, .5f, .99f}) {
        auto expected = build(res, params{quantile, mode}, raft::make_const_mdspan(input.view()));
        for (int64_t batch_rows : {int64_t{1}, int64_t{7}, rows, int64_t{0}}) {
          for (bool use_pinned : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                         << "range=" << static_cast<int>(mode) << " quantile=" << quantile
                         << " batch=" << batch_rows << " prefetch=" << prefetch
                         << " pinned=" << use_pinned);
            auto host_input = raft::make_host_matrix_view<const T, int64_t>(
              use_pinned ? pinned.data_handle() : values.data(), rows, dim);
            auto actual = build(res, params{quantile, mode, batch_rows}, host_input);
            EXPECT_EQ(actual.n_rows(), rows);
            EXPECT_EQ(actual.dim(), dim);
            EXPECT_TRUE(devArrMatch(expected.data_const_view().data_handle(),
                                    actual.data_const_view().data_handle(),
                                    values.size(),
                                    cuvs::Compare<int8_t>(),
                                    stream.get()));
            auto a = actual.const_manifest_view();
            auto e = expected.const_manifest_view();
            EXPECT_NEAR(double(a.max_error), double(e.max_error), double(e.max_error) * 1e-6);
            if (mode == range_mode::per_dimension) {
              auto actual_shifts = std::get<typename quantizer_const_view<T>::shifts_type>(a.value);
              auto expected_shifts =
                std::get<typename quantizer_const_view<T>::shifts_type>(e.value);
              EXPECT_EQ(actual_shifts.scale, expected_shifts.scale);
              EXPECT_TRUE(devArrMatch(actual_shifts.shifts.data_handle(),
                                      expected_shifts.shifts.data_handle(),
                                      dim,
                                      cuvs::Compare<T>(),
                                      stream.get()));
            }
            // Standalone training must give the same manifest without allocating corpus codes.
            if (batch_rows == 7 && !use_pinned) {
              auto trained = train(res, params{quantile, mode, batch_rows}, host_input);
              EXPECT_NEAR(
                double(trained.max_error), double(a.max_error), double(a.max_error) * 1e-6);
            }
          }
        }
      }
    }
  }
}

TEST(QuantizationDatasetTest, HostBuildBatchWorkspace)
{
  constexpr int64_t rows = 4097, dim = 35, batch_rows = 128;
  std::vector<float> values(rows * dim, 1.f);
  auto input = raft::make_host_matrix_view<const float, int64_t>(values.data(), rows, dim);
  for (bool prefetch : {false, true}) {
    raft::resources res;
    if (prefetch) {
      raft::resource::set_cuda_stream_pool(res, std::make_shared<rmm::cuda_stream_pool>(1));
    }
    for (float quantile : {1.f, .5f}) {
      params p{quantile, range_mode::per_dimension, batch_rows};
      auto action = [&](raft::resources const& h) {
        auto encoded = build(h, p, input);
        if (!raft::resource::get_dry_run_flag(h)) {
          EXPECT_EQ(encoded.const_manifest_view().max_error, 0.f);
        }
      };
      execute_with_dry_run_check(res, action, alloc_behavior::ARGUMENT_DRIVEN);
      raft::memory_stats_resources stats(res);
      action(stats);
      auto peak         = stats.get_bytes_peak();
      auto device_bytes = peak.device_workspace + peak.device_large_workspace + peak.device_global +
                          peak.device_managed;
      // Includes all int8 codes; excludes enough memory for even one full FP32 corpus copy.
      EXPECT_LT(device_bytes, values.size() * sizeof(float));
    }
  }
  EXPECT_THROW(build(raft::resources{}, params{1.f, range_mode::per_dimension, -1}, input),
               raft::logic_error);
}

TEST_P(PerDimensionRangeTest, Result) { this->testPerDimensionQuantization(); }

typedef QuantizationTest<float, int8_t> QuantizationTest_float_int8t;
TEST_P(QuantizationTest_float_int8t, ScalarQuantizationTest) { this->testScalarQuantization(); }

typedef QuantizationTest<double, int8_t> QuantizationTest_double_int8t;
TEST_P(QuantizationTest_double_int8t, ScalarQuantizationTest) { this->testScalarQuantization(); }

typedef QuantizationTest<half, int8_t> QuantizationTest_half_int8t;
TEST_P(QuantizationTest_half_int8t, ScalarQuantizationTest) { this->testScalarQuantization(); }

template <typename T>
const std::vector<QuantizationInputs<T>> inputs = {
  {{1.0}, 5, 5, T(0.0), T(1.0)},
  {{0.98}, 10, 20, T(0.0), T(1.0)},
  {{0.90}, 1000, 1500, T(-500.0), T(100.0)},
  {{0.59}, 100, 200},
  {{0.1}, 1, 1, T(0.0), T(1.0)},
  {{0.01}, 50, 50, T(0.0), T(1.0)},
  {{0.94}, 10, 20, T(-1.0), T(0.0)},
  {{0.95}, 10, 2, T(50.0), T(100.0)},
  {{0.95}, 10, 20, T(-500.0), T(-100.0)},
  {{0.95}, 10, 20, T(5.0), T(5.0)},
};

const std::vector<PerDimensionInputs> dataset_inputs = {
  {2, 2, {10.f, -7.f, 10.f, -7.f}, {0, 0, 0, 0}, 0.f},
  // Equal subnormal bounds must retain their value and reconstruct exactly.
  {1,
   2,
   {std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min()},
   {0, 0},
   0.f},
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

INSTANTIATE_TEST_CASE_P(PerDimensionRangeTests,
                        PerDimensionRangeTest,
                        ::testing::ValuesIn(dataset_inputs));

INSTANTIATE_TEST_CASE_P(QuantizationTest,
                        QuantizationTest_float_int8t,
                        ::testing::ValuesIn(inputs<float>));
INSTANTIATE_TEST_CASE_P(QuantizationTest,
                        QuantizationTest_double_int8t,
                        ::testing::ValuesIn(inputs<double>));
INSTANTIATE_TEST_CASE_P(QuantizationTest,
                        QuantizationTest_half_int8t,
                        ::testing::ValuesIn(inputs<half>));

}  // namespace cuvs::preprocessing::quantize::scalar
