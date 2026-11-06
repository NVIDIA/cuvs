/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/preprocessing/quantize/detail/scalar.cuh"
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

// Compute row norms in extended precision without squaring extreme residuals.
template <typename T>
long double expectReconstructionErrorBound(
  const T* original, const T* reconstructed, int64_t rows, int64_t dim, long double bound)
{
  long double maximum = 0;
  for (int64_t row = 0; row < rows; ++row) {
    SCOPED_TRACE(row);
    long double error = 0;
    for (int64_t col = 0; col < dim; ++col) {
      auto i     = row * dim + col;
      auto delta = static_cast<long double>(static_cast<double>(original[i])) -
                   static_cast<long double>(static_cast<double>(reconstructed[i]));
      error = std::hypot(error, delta);
    }
    EXPECT_GE(bound, error);
    maximum = std::max(maximum, error);
  }
  return maximum;
}

template <typename T>
void expectDeviceValues(raft::resources const& res,
                        const T* device_values,
                        std::vector<T> const& expected)
{
  std::vector<T> actual(expected.size());
  raft::update_host(
    actual.data(), device_values, actual.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(static_cast<double>(actual[i]), static_cast<double>(expected[i])) << i;
  }
}

template <typename T>
std::vector<T> makeShiftedDataset(int64_t rows, int64_t dim)
{
  std::vector<T> values(rows * dim);
  for (int64_t i = 0; i < rows * dim; ++i) {
    values[i] = static_cast<T>((i / dim - rows / 2) * .17 + (i % dim % 5) * 20.0);
  }
  return values;
}

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
      << (double)std::get<typename decltype(quantizer_1)::global_type>(quantizer_1.value).min
      << ", max = "
      << (double)std::get<typename decltype(quantizer_1)::global_type>(quantizer_1.value).max
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
      expectReconstructionErrorBound(host_input_.data(),
                                     re_transformed_input_h.data_handle(),
                                     rows_,
                                     cols_,
                                     quantizer_1.max_error);

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
      << (double)std::get<typename decltype(quantizer_2)::global_type>(quantizer_2.value).min
      << ", max = "
      << (double)std::get<typename decltype(quantizer_2)::global_type>(quantizer_2.value).max
      << std::endl;

    // check both quantizers are the same (valid if sampling is identical)
    if (input_.size() <= 1000000) {
      ASSERT_TRUE(
        (double)std::get<typename decltype(quantizer_1)::global_type>(quantizer_1.value).min ==
        (double)std::get<typename decltype(quantizer_2)::global_type>(quantizer_2.value).min);
      ASSERT_TRUE(
        (double)std::get<typename decltype(quantizer_1)::global_type>(quantizer_1.value).max ==
        (double)std::get<typename decltype(quantizer_2)::global_type>(quantizer_2.value).max);
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

using bounds_view = quantizer_view<const float>::per_dim_type;

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
    auto quantizer = train(handle,
                           params{.range = quantization_mode::PER_DIMENSION},
                           raft::make_const_mdspan(dataset_.view()));
    auto codes =
      raft::make_device_matrix<int8_t, int64_t>(handle, params_.n_samples, params_.n_features);
    transform(handle, quantizer, raft::make_const_mdspan(dataset_.view()), codes.view());
    expectDeviceValues(handle, codes.data_handle(), params_.codes);
    if (params_.tolerance == 0.f) {
      auto bounds = std::get<bounds_view>(std::as_const(quantizer).view().value);
      EXPECT_TRUE(devArrMatch(bounds.mins.data_handle(),
                              bounds.maxs.data_handle(),
                              params_.n_features,
                              cuvs::Compare<float>(),
                              stream));
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
    expectReconstructionErrorBound(params_.dataset.data(),
                                   reconstructed.data_handle(),
                                   params_.n_samples,
                                   params_.n_features,
                                   quantizer.max_error);
    if (params_.tolerance == 0.f) { EXPECT_EQ(quantizer.max_error, 0.f); }
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

template <typename T>
void testTrainingReconstructionError(params p)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  // A partial warp plus a second column per lane, with clipped outliers in different rows.
  constexpr int64_t rows = 9, dim = 35;
  auto values    = makeShiftedDataset<T>(rows, dim);
  values.front() = T{-300.f};
  values.back()  = T{500.f};
  auto input     = raft::make_device_matrix<T, int64_t>(res, rows, dim);
  raft::update_device(input.data_handle(), values.data(), values.size(), stream);
  auto codes   = raft::make_device_matrix<int8_t, int64_t>(res, rows, dim);
  auto decoded = raft::make_device_matrix<T, int64_t>(res, rows, dim);
  std::vector<T> reconstructed(values.size());
  auto owner = train(res, p, raft::make_const_mdspan(input.view()));
  auto view  = std::as_const(owner).view();
  EXPECT_EQ(owner.max_error, view.max_error);
  transform(res, owner, raft::make_const_mdspan(input.view()), codes.view());
  inverse_transform(res, owner, raft::make_const_mdspan(codes.view()), decoded.view());
  raft::update_host(reconstructed.data(), decoded.data_handle(), reconstructed.size(), stream);
  raft::resource::sync_stream(res);
  auto maximum =
    expectReconstructionErrorBound(values.data(), reconstructed.data(), rows, dim, owner.max_error);
  EXPECT_GT(maximum, 0);
  EXPECT_NEAR(static_cast<double>(owner.max_error),
              static_cast<double>(maximum),
              static_cast<double>(maximum) * 2e-6);
  if (p.quantile < 1.f) { EXPECT_GT(owner.max_error, 100); }
  if (p.range == quantization_mode::GLOBAL) {
    auto host              = train(res,
                      params{p.quantile},
                      raft::make_host_matrix_view<const T, int64_t>(values.data(), rows, dim));
    auto const host_bounds = std::get<typename quantizer<T>::global_type>(host.value);
    auto const bounds      = std::get<typename quantizer<T>::global_type>(owner.value);
    EXPECT_EQ(host_bounds.min, bounds.min);
    EXPECT_EQ(host_bounds.max, bounds.max);
    if (p.quantile == 1.f) {
      EXPECT_EQ(bounds.min, values.front());
      EXPECT_EQ(bounds.max, values.back());
    }
    EXPECT_GE(static_cast<long double>(host.max_error), maximum);
    EXPECT_NEAR(static_cast<double>(host.max_error),
                static_cast<double>(owner.max_error),
                static_cast<double>(maximum) * 1e-6);
  }
}
TYPED_TEST(PerDimensionQuantizationTest, GlobalTrainingReconstructionError)
{
  testTrainingReconstructionError<TypeParam>(params{1.f, quantization_mode::GLOBAL});
}

TYPED_TEST(PerDimensionQuantizationTest, GlobalClippedReconstructionError)
{
  testTrainingReconstructionError<TypeParam>(params{.5f, quantization_mode::GLOBAL});
}

TYPED_TEST(PerDimensionQuantizationTest, PerDimensionTrainingReconstructionError)
{
  testTrainingReconstructionError<TypeParam>(params{1.f, quantization_mode::PER_DIMENSION});
}

TYPED_TEST(PerDimensionQuantizationTest, PerDimensionClippedReconstructionError)
{
  testTrainingReconstructionError<TypeParam>(params{.5f, quantization_mode::PER_DIMENSION});
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
    auto owner = train(
      res, params{1.f, quantization_mode::PER_DIMENSION}, raft::make_const_mdspan(input.view()));
    transform(res, owner, raft::make_const_mdspan(input.view()), codes.view());
    inverse_transform(res, owner, raft::make_const_mdspan(codes.view()), decoded.view());
    std::vector<double> reconstructed(values.size());
    raft::update_host(reconstructed.data(), decoded.data_handle(), reconstructed.size(), stream);
    raft::resource::sync_stream(res);
    auto maximum =
      expectReconstructionErrorBound(values.data(), reconstructed.data(), 3, 2, owner.max_error);
    EXPECT_GT(maximum, 0);
    EXPECT_TRUE(std::isfinite(owner.max_error));
    EXPECT_GE(owner.max_error, maximum);
    EXPECT_NEAR(owner.max_error, maximum, maximum * 2e-6);
  }
}

TYPED_TEST(PerDimensionQuantizationTest, CorpusAndQueriesShareQuantizer)
{
  using T           = TypeParam;
  using bounds_view = typename quantizer_view<const T>::per_dim_type;
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<T> corpus{0.f, 10.f, 9.f, 254.f, 137.f, 9.f};
  auto input = raft::make_device_matrix<T, int64_t>(res, 2, 3);
  raft::update_device(input.data_handle(), corpus.data(), corpus.size(), stream);
  auto owner = train(
    res, params{.range = quantization_mode::PER_DIMENSION}, raft::make_const_mdspan(input.view()));
  auto quantizer = std::as_const(owner).view();
  EXPECT_EQ(owner.max_error, 0);
  EXPECT_EQ(quantizer.max_error, 0);
  expectDeviceValues(
    res, std::get<bounds_view>(quantizer.value).mins.data_handle(), std::vector<T>{0.f, 10.f, 9.f});
  expectDeviceValues(res,
                     std::get<bounds_view>(quantizer.value).maxs.data_handle(),
                     std::vector<T>{254.f, 137.f, 9.f});
  // The corpus manifest, rather than query extrema, controls ties and saturation.
  std::vector<T> queries{127.5f, 73.75f, 1000.f, 128.5f, 72.75f, -1000.f};
  raft::update_device(input.data_handle(), queries.data(), queries.size(), stream);
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 2, 3);
  transform(res, owner, raft::make_const_mdspan(input.view()), codes.view());
  expectDeviceValues(res, codes.data_handle(), {0, 0, 127, 2, -2, -127});
  inverse_transform(res, owner, raft::make_const_mdspan(codes.view()), input.view());
  expectDeviceValues(res, input.data_handle(), {127.f, 73.5f, 136.f, 129.f, 72.5f, -118.f});
}

TEST(PerDimensionRangeTest, AdjacentHalfBounds)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<half> values{half{1.f}, half{1.f + 1.f / 1024.f}};
  auto input = raft::make_device_matrix<half, int64_t>(res, 2, 1);
  raft::update_device(input.data_handle(), values.data(), values.size(), stream);
  auto quantizer = train(
    res, params{1.f, quantization_mode::PER_DIMENSION}, raft::make_const_mdspan(input.view()));
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 2, 1);
  transform(res, quantizer, raft::make_const_mdspan(input.view()), codes.view());
  inverse_transform(res, quantizer, raft::make_const_mdspan(codes.view()), input.view());
  expectDeviceValues(res, input.data_handle(), values);
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
  auto owner = train(
    res, params{.range = quantization_mode::PER_DIMENSION}, raft::make_const_mdspan(input.view()));
  auto quantizer = std::as_const(owner).view();
  expectDeviceValues(
    res, std::get<bounds_view>(quantizer.value).mins.data_handle(), {4.f, 10008.f, 9.f});
  expectDeviceValues(
    res, std::get<bounds_view>(quantizer.value).maxs.data_handle(), {995.f, 11990.f, 9.f});
  auto full = train(
    res, params{1.f, quantization_mode::PER_DIMENSION}, raft::make_const_mdspan(input.view()));
  expectDeviceValues(res,
                     std::get<bounds_view>(std::as_const(full).view().value).maxs.data_handle(),
                     {999.f, 11998.f, 9.f});
  expectDeviceValues(res,
                     std::get<bounds_view>(std::as_const(full).view().value).mins.data_handle(),
                     {0.f, 10000.f, 9.f});
  auto trimmed = train(
    res, params{0.5f, quantization_mode::PER_DIMENSION}, raft::make_const_mdspan(input.view()));
  expectDeviceValues(res,
                     std::get<bounds_view>(std::as_const(trimmed).view().value).maxs.data_handle(),
                     {749.f, 11498.f, 9.f});
  expectDeviceValues(res,
                     std::get<bounds_view>(std::as_const(trimmed).view().value).mins.data_handle(),
                     {250.f, 10500.f, 9.f});
  for (float q : {0.f, -1.f, 1.1f, std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_THROW(
      train(
        res, params{q, quantization_mode::PER_DIMENSION}, raft::make_const_mdspan(input.view())),
      raft::logic_error);
  }
}

TYPED_TEST(PerDimensionQuantizationTest, CustomParametersAndValidation)
{
  using T           = TypeParam;
  using bounds_view = typename quantizer_view<const T>::per_dim_type;
  raft::resources res;
  auto mins = raft::make_device_vector<T, int64_t>(res, 2);
  auto maxs = raft::make_device_vector<T, int64_t>(res, 2);
  std::vector<T> lower{-154.f, -104.f}, upper{354.f, -96.f};
  auto stream = raft::resource::get_cuda_stream(res);
  raft::update_device(mins.data_handle(), lower.data(), lower.size(), stream);
  raft::update_device(maxs.data_handle(), upper.data(), upper.size(), stream);
  scalar::quantizer<T> owner{
    res, raft::make_const_mdspan(mins.view()), raft::make_const_mdspan(maxs.view())};
  auto quantizer = std::as_const(owner).view();
  EXPECT_TRUE(std::isinf(owner.max_error));
  EXPECT_TRUE(std::isinf(quantizer.max_error));
  EXPECT_TRUE(std::isinf((scalar::quantizer<T>{T{-1}, T{1}}.view().max_error)));
  expectDeviceValues(res, std::get<bounds_view>(quantizer.value).mins.data_handle(), lower);
  expectDeviceValues(res, std::get<bounds_view>(quantizer.value).maxs.data_handle(), upper);
  auto input = raft::make_device_matrix<T, int64_t>(res, 1, 2);
  std::vector<T> values{104.f, -106.f};
  raft::update_device(
    input.data_handle(), values.data(), values.size(), raft::resource::get_cuda_stream(res));
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 1, 2);
  transform(res, owner, raft::make_const_mdspan(input.view()), codes.view());
  expectDeviceValues(res, codes.data_handle(), {2, -127});
  scalar::quantizer<T> wrong_bounds;
  wrong_bounds.value = typename scalar::quantizer<T>::per_dim_type{
    raft::make_device_vector<T, int64_t>(res, 2), raft::make_device_vector<T, int64_t>(res, 1)};
  EXPECT_THROW(transform(res, wrong_bounds, raft::make_const_mdspan(input.view()), codes.view()),
               raft::logic_error);
  auto wrong_shape = raft::make_device_matrix<int8_t, int64_t>(res, 2, 1);
  EXPECT_THROW(transform(res, owner, raft::make_const_mdspan(input.view()), wrong_shape.view()),
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
                     params{.range = quantization_mode::PER_DIMENSION},
                     raft::make_device_matrix_view<const T, int64_t>(nullptr, 0, 2)),
               raft::logic_error);
}

TYPED_TEST(PerDimensionQuantizationTest, DirectPerDimensionBounds)
{
  using T = TypeParam;
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  auto mins   = raft::make_device_vector<T, int64_t>(res, 2);
  auto maxs   = raft::make_device_vector<T, int64_t>(res, 2);
  std::vector<T> lower{-127.f, -254.f}, upper{127.f, 254.f}, values{1.f, 2.f};
  raft::update_device(mins.data_handle(), lower.data(), lower.size(), stream);
  raft::update_device(maxs.data_handle(), upper.data(), upper.size(), stream);
  quantizer<T> owner;
  owner.value = typename quantizer<T>::per_dim_type{std::move(mins), std::move(maxs)};
  auto input  = raft::make_device_matrix<T, int64_t>(res, 1, 2);
  auto codes  = raft::make_device_matrix<int8_t, int64_t>(res, 1, 2);
  raft::update_device(input.data_handle(), values.data(), values.size(), stream);
  transform(res, owner, raft::make_const_mdspan(input.view()), codes.view());
  expectDeviceValues(res, codes.data_handle(), {1, 1});
  inverse_transform(res, owner, raft::make_const_mdspan(codes.view()), input.view());
  expectDeviceValues(res, input.data_handle(), values);

  // Transforms use the owner's current bounds.
  lower                             = {T{-254.f}, T{-508.f}};
  upper                             = {T{254.f}, T{508.f}};
  auto mutable_view                 = owner.view();
  quantizer_view<const T> read_only = mutable_view;
  auto bounds       = std::get<typename quantizer_view<T>::per_dim_type>(mutable_view.value);
  auto const_bounds = std::get<typename quantizer_view<const T>::per_dim_type>(read_only.value);
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(bounds.mins.data_handle())>, T*>);
  static_assert(
    std::is_same_v<std::remove_cvref_t<decltype(const_bounds.mins.data_handle())>, const T*>);
  static_assert(!std::is_convertible_v<quantizer_view<const T>, quantizer_view<T>>);
  EXPECT_EQ(bounds.mins.data_handle(), const_bounds.mins.data_handle());
  EXPECT_EQ(bounds.maxs.data_handle(), const_bounds.maxs.data_handle());
  raft::update_device(bounds.mins.data_handle(), lower.data(), lower.size(), stream);
  raft::update_device(bounds.maxs.data_handle(), upper.data(), upper.size(), stream);
  transform(res, owner, raft::make_const_mdspan(input.view()), codes.view());
  expectDeviceValues(res, codes.data_handle(), {0, 0});
}

TYPED_TEST(PerDimensionQuantizationTest, DirectGlobalBounds)
{
  using T = TypeParam;
  raft::resources res;
  auto input  = raft::make_host_matrix<T, int64_t>(1, 2);
  input(0, 0) = T{1};
  input(0, 1) = T{2};
  auto codes  = raft::make_host_matrix<int8_t, int64_t>(1, 2);
  quantizer<T> owner;
  owner.value                       = typename quantizer<T>::global_type{T{-128}, T{127}};
  quantizer_view<const T> read_only = owner.view();
  auto copied_bounds = std::get<typename quantizer_view<const T>::global_type>(read_only.value);
  EXPECT_EQ(double(copied_bounds.min), -128.0);
  EXPECT_EQ(double(copied_bounds.max), 127.0);
  transform(res, owner, raft::make_const_mdspan(input.view()), codes.view());
  EXPECT_EQ(codes(0, 0), 1);
  EXPECT_EQ(codes(0, 1), 2);
  auto& bounds = std::get<typename quantizer<T>::global_type>(owner.value);
  bounds.min   = T{-256};
  bounds.max   = T{254};
  transform(res, owner, raft::make_const_mdspan(input.view()), codes.view());
  EXPECT_EQ(codes(0, 0), 1);
  EXPECT_EQ(codes(0, 1), 1);
  inverse_transform(res, owner, raft::make_const_mdspan(codes.view()), input.view());
  EXPECT_EQ(static_cast<double>(input(0, 0)), 2);
  EXPECT_EQ(static_cast<double>(input(0, 1)), 2);
}

TYPED_TEST(PerDimensionQuantizationTest, HostTransformsRejectPerDimensionQuantizer)
{
  using T = TypeParam;
  raft::resources res;
  auto values    = raft::make_host_matrix<T, int64_t>(1, 1);
  values(0, 0)   = T{1};
  auto quantizer = train(
    res, params{1.f, quantization_mode::PER_DIMENSION}, raft::make_const_mdspan(values.view()));
  auto codes = raft::make_host_matrix<int8_t, int64_t>(1, 1);
  EXPECT_THROW(transform(res, quantizer, raft::make_const_mdspan(values.view()), codes.view()),
               raft::logic_error);
  EXPECT_THROW(
    inverse_transform(res, quantizer, raft::make_const_mdspan(codes.view()), values.view()),
    raft::logic_error);
}

TEST(PerDimensionRangeTest, QuantizedTypeLimits)
{
  using QuantI = int16_t;
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<float> values{-100000.f, -32767.f, 0.f, 32767.f, 100000.f};
  auto input = raft::make_device_matrix<float, int64_t>(res, values.size(), 1);
  raft::update_device(input.data_handle(), values.data(), values.size(), stream);
  auto codes = raft::make_device_matrix<QuantI, int64_t>(res, values.size(), 1);
  auto owner = detail::train<float, QuantI>(res,
                                            params{.5f, quantization_mode::PER_DIMENSION},
                                            raft::make_const_mdspan(input.view()),
                                            codes.view());
  expectDeviceValues(
    res, std::get<bounds_view>(std::as_const(owner).view().value).maxs.data_handle(), {32767.f});
  std::vector<QuantI> expected{-32767, -32767, 0, 32767, 32767};
  expectDeviceValues(res, codes.data_handle(), expected);
  detail::apply_quantizer<float, QuantI>(
    res, std::as_const(owner).view(), raft::make_const_mdspan(codes.view()), input.view());
  std::vector<float> decoded(values.size());
  raft::update_host(decoded.data(), input.data_handle(), decoded.size(), stream);
  raft::resource::sync_stream(res);
  double maximum = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_FLOAT_EQ(decoded[i], static_cast<float>(expected[i]));
    maximum = std::max(maximum, std::abs(double(values[i]) - decoded[i]));
  }
  EXPECT_GE(owner.max_error, maximum);
  EXPECT_NEAR(owner.max_error, maximum, maximum * 2e-6);
}

TYPED_TEST(PerDimensionQuantizationTest, GlobalBounds)
{
  using T = TypeParam;
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  for (auto bounds : {std::pair<T, T>{T{-3}, T{7}}, std::pair<T, T>{T{5}, T{5}}}) {
    auto owner      = quantizer<T>{bounds.first, bounds.second};
    auto parameters = std::get<typename quantizer<T>::global_type>(owner.value);
    EXPECT_EQ(parameters.min, bounds.first);
    EXPECT_EQ(parameters.max, bounds.second);
    std::vector<T> values{T{-100}, bounds.first, bounds.second, T{100}};
    auto input = raft::make_device_matrix<T, int64_t>(res, 1, values.size());
    auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 1, values.size());
    raft::update_device(input.data_handle(), values.data(), values.size(), stream);
    transform(res, owner, raft::make_const_mdspan(input.view()), codes.view());
    inverse_transform(res, owner, raft::make_const_mdspan(codes.view()), input.view());
    std::vector<T> decoded(values.size());
    raft::update_host(decoded.data(), input.data_handle(), decoded.size(), stream);
    raft::resource::sync_stream(res);
    EXPECT_EQ(decoded.front(), bounds.first);
    EXPECT_EQ(decoded.back(), bounds.second);
    EXPECT_EQ(decoded[1], bounds.first);
    EXPECT_EQ(decoded[2], bounds.second);
  }
}

TEST(QuantizationDatasetTest, GlobalRangeDataset)
{
  raft::resources res;
  auto input = raft::make_device_matrix<float, int64_t>(res, 1, 2);
  std::vector<float> values{-1.f, 1.f};
  auto stream = raft::resource::get_cuda_stream(res);
  raft::update_device(input.data_handle(), values.data(), values.size(), stream);
  auto encoded = make_sq_dataset(res, params{1.f}, raft::make_const_mdspan(input.view()));
  typename decltype(encoded)::const_view_type read_only = encoded.view();
  EXPECT_EQ(read_only.const_manifest_view().max_error, 0.f);
  EXPECT_EQ(encoded.n_rows(), 1);
  EXPECT_EQ(encoded.dim(), 2u);
  expectDeviceValues(res, encoded.data_const_view().data_handle(), {-128, 127});
  auto range =
    std::get<quantizer_view<const float>::global_type>(encoded.const_manifest_view().value);
  EXPECT_EQ(range.min, -1.f);
  EXPECT_EQ(range.max, 1.f);
}

TEST(QuantizationDatasetTest, PerDimensionDataset)
{
  raft::resources res;
  auto input = raft::make_device_matrix<float, int64_t>(res, 2, 2);
  std::vector<float> values{0.f, 10.f, 254.f, 14.f};
  auto stream = raft::resource::get_cuda_stream(res);
  raft::update_device(input.data_handle(), values.data(), values.size(), stream);
  auto encoded = make_sq_dataset(
    res, params{.range = quantization_mode::PER_DIMENSION}, raft::make_const_mdspan(input.view()));
  typename decltype(encoded)::const_view_type read_only = encoded.view();
  EXPECT_EQ(read_only.const_manifest_view().max_error, 0.f);
  EXPECT_EQ(encoded.n_rows(), 2);
  EXPECT_EQ(encoded.dim(), 2u);
  expectDeviceValues(res, encoded.data_const_view().data_handle(), {-127, -127, 127, 127});
  auto codes     = raft::make_device_matrix<int8_t, int64_t>(res, 2, 2);
  auto quantizer = train(
    res, params{.range = quantization_mode::PER_DIMENSION}, raft::make_const_mdspan(input.view()));
  transform(res, quantizer, raft::make_const_mdspan(input.view()), codes.view());
  EXPECT_TRUE(devArrMatch(encoded.data_const_view().data_handle(),
                          codes.data_handle(),
                          4,
                          cuvs::Compare<int8_t>(),
                          stream.get()));
}

template <typename T>
void testHostDataset(params p, bool prefetch = false, bool use_pinned = false)
{
  constexpr int64_t rows = 37, dim = 35;
  auto values = makeShiftedDataset<T>(rows, dim);
  // The largest clipping error precedes the final batch to check the running maximum.
  values.front() = T{-1200.f};
  values.back()  = T{500.f};
  raft::resources res;
  if (prefetch) {
    raft::resource::set_cuda_stream_pool(res, std::make_shared<rmm::cuda_stream_pool>(1));
  }
  auto stream = raft::resource::get_cuda_stream(res);
  auto input  = raft::make_device_matrix<T, int64_t>(res, rows, dim);
  raft::update_device(input.data_handle(), values.data(), values.size(), stream);
  auto pinned = raft::make_pinned_matrix<T, int64_t>(res, rows, dim);
  if (use_pinned) { std::copy(values.begin(), values.end(), pinned.data_handle()); }
  // A negative staging limit is ignored for device input.
  auto expected =
    make_sq_dataset(res, params{p.quantile, p.range, -1}, raft::make_const_mdspan(input.view()));
  auto host_input = raft::make_host_matrix_view<const T, int64_t>(
    use_pinned ? pinned.data_handle() : values.data(), rows, dim);
  auto actual = make_sq_dataset(res, p, host_input);
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
  if (p.range == quantization_mode::PER_DIMENSION) {
    auto actual_bounds   = std::get<typename quantizer_view<const T>::per_dim_type>(a.value);
    auto expected_bounds = std::get<typename quantizer_view<const T>::per_dim_type>(e.value);
    EXPECT_TRUE(devArrMatch(actual_bounds.maxs.data_handle(),
                            expected_bounds.maxs.data_handle(),
                            dim,
                            cuvs::Compare<T>(),
                            stream.get()));
    EXPECT_TRUE(devArrMatch(actual_bounds.mins.data_handle(),
                            expected_bounds.mins.data_handle(),
                            dim,
                            cuvs::Compare<T>(),
                            stream.get()));
  }
  // Standalone training must agree without allocating corpus codes.
  auto trained = train(res, p, host_input);
  EXPECT_NEAR(double(trained.max_error), double(a.max_error), double(a.max_error) * 1e-6);
}

TYPED_TEST(PerDimensionQuantizationTest, HostDatasetPartialBatch)
{
  testHostDataset<TypeParam>(params{1.f, quantization_mode::PER_DIMENSION, 7});
}

TYPED_TEST(PerDimensionQuantizationTest, HostDatasetSingleRowBatch)
{
  testHostDataset<TypeParam>(params{1.f, quantization_mode::PER_DIMENSION, 1});
}

TYPED_TEST(PerDimensionQuantizationTest, HostDatasetFullBatch)
{
  testHostDataset<TypeParam>(params{1.f, quantization_mode::PER_DIMENSION, 37});
}

TYPED_TEST(PerDimensionQuantizationTest, HostDatasetUnlimitedBatch)
{
  testHostDataset<TypeParam>(params{1.f, quantization_mode::PER_DIMENSION, 0});
}

TYPED_TEST(PerDimensionQuantizationTest, HostDatasetPinned)
{
  testHostDataset<TypeParam>(params{1.f, quantization_mode::PER_DIMENSION, 7}, false, true);
}

TYPED_TEST(PerDimensionQuantizationTest, HostDatasetPrefetch)
{
  testHostDataset<TypeParam>(params{1.f, quantization_mode::PER_DIMENSION, 7}, true);
}

TYPED_TEST(PerDimensionQuantizationTest, HostDatasetPinnedPrefetch)
{
  testHostDataset<TypeParam>(params{1.f, quantization_mode::PER_DIMENSION, 7}, true, true);
}

TYPED_TEST(PerDimensionQuantizationTest, HostDatasetClippedQuantile)
{
  testHostDataset<TypeParam>(params{.5f, quantization_mode::PER_DIMENSION, 7});
}

TYPED_TEST(PerDimensionQuantizationTest, HostDatasetDefaultQuantile)
{
  testHostDataset<TypeParam>(params{.99f, quantization_mode::PER_DIMENSION, 7});
}

TYPED_TEST(PerDimensionQuantizationTest, HostDatasetGlobalRange)
{
  testHostDataset<TypeParam>(params{1.f, quantization_mode::GLOBAL, 7});
}

TYPED_TEST(PerDimensionQuantizationTest, HostDatasetGlobalClippedQuantile)
{
  testHostDataset<TypeParam>(params{.5f, quantization_mode::GLOBAL, 7});
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
      params p{quantile, quantization_mode::PER_DIMENSION, batch_rows};
      auto action = [&](raft::resources const& h) {
        auto encoded = make_sq_dataset(h, p, input);
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
  EXPECT_THROW(
    make_sq_dataset(raft::resources{}, params{1.f, quantization_mode::PER_DIMENSION, -1}, input),
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
  // Each nonconstant dimension uses the full code range despite different input ranges.
  {3,
   3,
   {-127.f, -127.f / 128.f, 9.f, 1.f, 1.f / 128.f, 9.f, 127.f, 127.f / 128.f, 9.f},
   {-127, -127, 0, 1, 1, 0, 127, 127, 0},
   1e-6f},
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
