/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include <cuvs/preprocessing/quantize/binary.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/init.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/util/itertools.hpp>
#include <thrust/reduce.h>

#include <type_traits>
#include <variant>
#include <vector>

namespace cuvs::preprocessing::quantize::binary {

template <typename T>
struct BinaryQuantizationInputs {
  int rows;
  int cols;
  cuvs::preprocessing::quantize::binary::bit_threshold threshold;
  bool train_host;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const BinaryQuantizationInputs<T>& inputs)
{
  os << "> dataset_size:" << inputs.rows << " dataset_dim:" << inputs.cols;
  os << " threshold: ";
  switch (inputs.threshold) {
    case bit_threshold::zero: os << "zero"; break;
    case bit_threshold::mean: os << "mean"; break;
    case bit_threshold::sampling_median: os << "sampling_median"; break;
    default: os << "unknown"; break;
  }
  os << " train_host_dataset: " << inputs.train_host;
  return os;
}

template <typename T, typename QuantI>
class BinaryQuantizationTest : public ::testing::TestWithParam<BinaryQuantizationInputs<T>> {
 public:
  BinaryQuantizationTest()
    : params_(::testing::TestWithParam<BinaryQuantizationInputs<T>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle).get()),
      input_(0, stream)
  {
  }

 protected:
  void testBinaryQuantization()
  {
    // dataset identical on host / device
    auto dataset = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
      (const T*)(input_.data()), rows_, cols_);
    auto dataset_h = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
      (const T*)(host_input_.data()), rows_, cols_);

    {
      static_assert(std::is_same_v<QuantI, uint8_t>);

      cuvs::preprocessing::quantize::binary::params params;
      params.threshold = params_.threshold;

      const auto col_quantized = raft::div_rounding_up_safe(cols_, 8);
      auto quantized_input_h   = raft::make_host_matrix<QuantI, int64_t>(rows_, col_quantized);
      auto quantized_input_d =
        raft::make_device_matrix<QuantI, int64_t>(handle, rows_, col_quantized);

      cuvs::preprocessing::quantize::binary::quantizer<T> quantizer;
      if (train_host_) {
        quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset_h);
      } else {
        quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset);
      }
      ASSERT_EQ(quantizer.dim(), static_cast<uint32_t>(cols_));
      // The zero threshold is implied by the dimension alone
      ASSERT_EQ(std::holds_alternative<uint32_t>(quantizer.value),
                params.threshold == cuvs::preprocessing::quantize::binary::bit_threshold::zero);

      cuvs::preprocessing::quantize::binary::transform(
        handle, quantizer, dataset, quantized_input_d.view());
      cuvs::preprocessing::quantize::binary::transform(
        handle, quantizer, dataset_h, quantized_input_h.view());

      ASSERT_TRUE(devArrMatchHost(quantized_input_h.data_handle(),
                                  quantized_input_d.data_handle(),
                                  quantized_input_h.size(),
                                  cuvs::Compare<QuantI>(),
                                  stream));
    }
  }

  void SetUp() override
  {
    rows_ = params_.rows;
    cols_ = params_.cols;

    int n_elements = rows_ * cols_;
    input_.resize(n_elements, stream);
    host_input_.resize(n_elements);

    train_host_ = params_.train_host;

    // random input
    unsigned long long int seed = 1234ULL;
    raft::random::RngState r(seed);
    uniform(handle, r, input_.data(), input_.size(), static_cast<T>(-1), static_cast<T>(1));

    raft::update_host(host_input_.data(), input_.data(), input_.size(), stream);

    raft::resource::sync_stream(handle, stream);
  }

 private:
  raft::resources handle;
  cudaStream_t stream;

  BinaryQuantizationInputs<T> params_;
  int rows_;
  int cols_;
  rmm::device_uvector<T> input_;
  std::vector<T> host_input_;
  bool train_host_;
};

template <typename T>
const std::vector<BinaryQuantizationInputs<T>> generate_inputs()
{
  const auto inputs = raft::util::itertools::product<BinaryQuantizationInputs<T>>(
    {5, 100, 1000},
    {7, 128, 1999},
    {cuvs::preprocessing::quantize::binary::bit_threshold::zero,
     cuvs::preprocessing::quantize::binary::bit_threshold::mean,
     cuvs::preprocessing::quantize::binary::bit_threshold::sampling_median},
    {true, false});
  return inputs;
}

typedef BinaryQuantizationTest<float, uint8_t> QuantizationTest_float_uint8t;
TEST_P(QuantizationTest_float_uint8t, BinaryQuantizationTest) { this->testBinaryQuantization(); }

typedef BinaryQuantizationTest<double, uint8_t> QuantizationTest_double_uint8t;
TEST_P(QuantizationTest_double_uint8t, BinaryQuantizationTest) { this->testBinaryQuantization(); }

typedef BinaryQuantizationTest<half, uint8_t> QuantizationTest_half_uint8t;
TEST_P(QuantizationTest_half_uint8t, BinaryQuantizationTest) { this->testBinaryQuantization(); }

INSTANTIATE_TEST_CASE_P(BinaryQuantizationTest,
                        QuantizationTest_float_uint8t,
                        ::testing::ValuesIn(generate_inputs<float>()));
INSTANTIATE_TEST_CASE_P(BinaryQuantizationTest,
                        QuantizationTest_double_uint8t,
                        ::testing::ValuesIn(generate_inputs<double>()));
INSTANTIATE_TEST_CASE_P(BinaryQuantizationTest,
                        QuantizationTest_half_uint8t,
                        ::testing::ValuesIn(generate_inputs<half>()));

// Train a sampling-median quantizer on the same data on the device and on the host and compare the
// thresholds. With sampling_ratio = 1 and an odd row count every row is sampled, so both
// thresholds are the exact median of each column.
template <typename T>
void testSamplingMedianDeviceMatchesHost()
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  constexpr int64_t rows = 1001;
  constexpr int64_t cols = 12;
  // Column j holds a permutation of 2 * j + k / rows for k in [0, rows), so every column has its
  // own median and a threshold taken from another column is easy to spot.
  std::vector<T> host_data(rows * cols);
  for (int64_t i = 0; i < rows; i++) {
    for (int64_t j = 0; j < cols; j++) {
      const int64_t k         = (i * 7919 + j * 104729) % rows;
      host_data[i * cols + j] = static_cast<T>(2.0 * j + static_cast<double>(k) / rows);
    }
  }
  auto device_data = raft::make_device_matrix<T, int64_t>(handle, rows, cols);
  raft::update_device(device_data.data_handle(), host_data.data(), host_data.size(), stream);

  params p;
  p.threshold         = bit_threshold::sampling_median;
  p.sampling_ratio    = 1.0;
  auto device_trained = train(handle, p, raft::make_const_mdspan(device_data.view()));
  auto host_trained   = train(
    handle,
    p,
    raft::make_host_matrix_view<const T, int64_t, raft::row_major>(host_data.data(), rows, cols));

  using threshold_type = typename quantizer<T>::threshold_type;
  ASSERT_TRUE(std::holds_alternative<threshold_type>(device_trained.value));
  ASSERT_TRUE(std::holds_alternative<threshold_type>(host_trained.value));
  std::vector<T> device_thresholds(cols);
  std::vector<T> host_thresholds(cols);
  raft::update_host(device_thresholds.data(),
                    std::get<threshold_type>(device_trained.value).data_handle(),
                    cols,
                    stream);
  raft::update_host(host_thresholds.data(),
                    std::get<threshold_type>(host_trained.value).data_handle(),
                    cols,
                    stream);
  raft::resource::sync_stream(handle, stream);

  for (int64_t j = 0; j < cols; j++) {
    if constexpr (std::is_same_v<T, half>) {
      EXPECT_EQ(static_cast<float>(device_thresholds[j]), static_cast<float>(host_thresholds[j]))
        << "dim " << j;
    } else {
      EXPECT_EQ(device_thresholds[j], host_thresholds[j]) << "dim " << j;
    }
  }
}

TEST(BinaryQuantizationSamplingMedian, DeviceMatchesHostFloat)
{
  testSamplingMedianDeviceMatchesHost<float>();
}
TEST(BinaryQuantizationSamplingMedian, DeviceMatchesHostDouble)
{
  testSamplingMedianDeviceMatchesHost<double>();
}
TEST(BinaryQuantizationSamplingMedian, DeviceMatchesHostHalf)
{
  testSamplingMedianDeviceMatchesHost<half>();
}

}  // namespace cuvs::preprocessing::quantize::binary
