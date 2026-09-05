/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
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
      auto quantized_input_h   = raft::make_host_matrix<QuantI, int64_t>(rows_, cols_);
      auto quantized_input_d   = raft::make_device_matrix<QuantI, int64_t>(handle, rows_, cols_);

      cuvs::preprocessing::quantize::binary::quantizer<T> quantizer(handle);
      if (train_host_) {
        quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset_h);
      } else {
        quantizer = cuvs::preprocessing::quantize::binary::train(handle, params, dataset);
      }

      cuvs::preprocessing::quantize::binary::transform(
        handle, quantizer, dataset, quantized_input_d.view());
      cuvs::preprocessing::quantize::binary::transform(
        handle, quantizer, dataset_h, quantized_input_h.view());

      ASSERT_TRUE(devArrMatchHost(quantized_input_h.data_handle(),
                                  quantized_input_d.data_handle(),
                                  input_.size(),
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

}  // namespace cuvs::preprocessing::quantize::binary
