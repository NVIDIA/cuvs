/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include <cuvs/preprocessing/quantize/bbq.hpp>
#include <cuvs_internal/preprocessing/bbq_cpu_quantize.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <vector>

namespace cuvs::preprocessing::quantize::bbq {

struct BbqQuantizationInputs {
  int64_t rows;
  int64_t cols;
  bbq_code_layout doc_layout;
  bbq_code_layout query_layout;
  cuvs::distance::DistanceType metric;
};

std::ostream& operator<<(std::ostream& os, const BbqQuantizationInputs& inputs)
{
  return os << " n_rows:" << inputs.rows << " n_cols:" << inputs.cols
            << " doc_layout:" << static_cast<int>(inputs.doc_layout)
            << " query_layout:" << static_cast<int>(inputs.query_layout)
            << " metric:" << static_cast<int>(inputs.metric);
}

template <typename T>
auto to_host(raft::resources const& res, const T* device_ptr, size_t len) -> std::vector<T>
{
  std::vector<T> host(len);
  raft::copy(host.data(), device_ptr, len, raft::resource::get_cuda_stream(res).get());
  raft::resource::sync_stream(res);
  return host;
}

/**
 * Values on a 1/16 grid, that makes the shared centroid come out the same whether
 * it is summed serially on the host or by the device's tree reduction, which is what lets the two
 * quantizers be compared byte for byte: everything downstream of the centroid is per-row, and
 * there the device follows the host reference exactly. A grid this coarse also lands plenty of
 * components exactly halfway between two codes, which is where Java's round-half-up matters.
 */
auto make_summable_dataset(int64_t n_rows, int64_t dim, uint64_t seed) -> std::vector<float>
{
  std::mt19937 gen{seed};
  std::uniform_int_distribution<int> dist{-32, 127};
  std::vector<float> data(static_cast<size_t>(n_rows * dim));
  for (float& x : data) {
    x = static_cast<float>(dist(gen)) / 16.0f;
  }
  return data;
}

/**
 * Decodes one row back to a code per component. Written from the layout documentation in bbq.hpp
 * rather than by running the packing backwards, so that a row which reconstructs accurately is
 * evidence the codes are readable by an independent decoder -- a search kernel, or Lucene -- and
 * not just that they round-trip through the helper that wrote them.
 */
auto unpack_row(const uint8_t* row, int64_t dim, bbq_code_layout layout) -> std::vector<uint8_t>
{
  const uint32_t bits = get_bit_width(layout);
  std::vector<uint8_t> codes(static_cast<size_t>(dim), 0);
  switch (layout) {
    case bbq_code_layout::packed_7b:
    case bbq_code_layout::packed_8b: std::copy(row, row + dim, codes.begin()); break;
    case bbq_code_layout::packed_4b:
      // Dims 2k and 2k+1 share byte k, high nibble first. An odd dim has no byte for its last
      // component, which is why the tests below quantize an even number of dimensions.
      for (int64_t d = 0; d + 1 < dim; d += 2) {
        codes[d]     = row[d / 2] >> 4;
        codes[d + 1] = row[d / 2] & 0x0f;
      }
      break;
    default: {
      // One bit plane per code bit, least significant plane first, dimension d held in bit
      // 7 - d % 8 of the plane's (d / 8)-th byte. packed_1b is the single-plane case.
      const int64_t stripe = (dim + 7) / 8;
      for (int64_t d = 0; d < dim; ++d) {
        for (uint32_t bit = 0; bit < bits; ++bit) {
          const uint8_t plane_byte = row[bit * stripe + d / 8];
          codes[d] |= static_cast<uint8_t>(((plane_byte >> (7 - d % 8)) & 1u) << bit);
        }
      }
      break;
    }
  }
  return codes;
}

/** Every row of @p q decoded back to one code per component, row-major. */
auto unpack_codes(raft::resources const& res,
                  const bbq_dataset_owning_storage<float, int64_t>& q,
                  int64_t n_rows,
                  int64_t dim) -> std::vector<uint8_t>
{
  const auto packed     = to_host(res, q.codes.data_handle(), q.codes.size());
  const auto row_length = static_cast<size_t>(q.codes.extent(1));
  std::vector<uint8_t> codes(static_cast<size_t>(n_rows * dim));
  for (int64_t i = 0; i < n_rows; ++i) {
    const auto row = unpack_row(packed.data() + i * row_length, dim, q.layout);
    std::copy(row.begin(), row.end(), codes.begin() + i * dim);
  }
  return codes;
}

/**
 * Mean squared reconstruction error of the quantizer over @p host_dataset, relative to the energy
 * of what it is asked to encode: the rows after the shared centroid is subtracted. Rebuilding a
 * component is what search does with the codes -- take the row's interval floor and step up by
 * the code -- so this exercises the codes, the intervals and the derived step together.
 */
auto relative_reconstruction_error(raft::resources const& res,
                                   const bbq_dataset_owning_storage<float, int64_t>& q,
                                   const std::vector<uint8_t>& codes,
                                   const std::vector<float>& host_dataset,
                                   int64_t n_rows,
                                   int64_t dim) -> float
{
  const auto centroid = to_host(res, q.quantizer.centroid.data_handle(), dim);
  const auto lower    = to_host(res, q.lower_intervals.data_handle(), n_rows);
  const auto delta    = to_host(res, q.dequant_delta.data_handle(), n_rows);

  double squared_error   = 0.0;
  double residual_energy = 0.0;
  for (int64_t i = 0; i < n_rows; ++i) {
    for (int64_t d = 0; d < dim; ++d) {
      const double residual = host_dataset[i * dim + d] - centroid[d];
      const double restored = lower[i] + delta[i] * codes[i * dim + d];
      squared_error += (residual - restored) * (residual - restored);
      residual_energy += residual * residual;
    }
  }
  return float(squared_error / residual_energy);
}

class BbqQuantizationTest : public ::testing::TestWithParam<BbqQuantizationInputs> {
 public:
  BbqQuantizationTest()
    : params_(::testing::TestWithParam<BbqQuantizationInputs>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle).get()),
      input_(0, stream)
  {
  }

 protected:
  // Test against host reference quantizer, field by field.
  void testBbqQuantization()
  {
    auto dataset = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      input_.data(), rows_, cols_);

    const params p{params_.doc_layout, params_.query_layout, params_.metric};
    const auto quantizer = build(handle, dataset);
    const auto encoded   = transform(handle, p, quantizer.view(), dataset);
    ASSERT_EQ(quantizer.dim(), cols_);

    std::vector<bbq_code_layout> layouts{params_.doc_layout};
    if (params_.query_layout != params_.doc_layout) { layouts.push_back(params_.query_layout); }
    ASSERT_EQ(encoded.storages.size(), layouts.size());

    const auto encoded_view = encoded.as_dataset_view();
    for (const auto layout : layouts) {
      ASSERT_TRUE(encoded.has_layout(layout));
      const auto expected =
        cuvs_internal::bbq::quantize(host_input_.data(), rows_, cols_, params_.metric, layout);
      const auto actual = encoded_view.get_storage(layout);
      checkStorage(expected, quantizer, encoded, actual);
    }
  }

  void checkStorage(const cuvs_internal::bbq::host_quantizer_storage& expected,
                    const quantizer<float>& quantizer,
                    const cuvs::neighbors::device_bbq_dataset<float, int64_t>& encoded,
                    const bbq_dataset_view_storage<float, int64_t>& actual)
  {
    ASSERT_EQ(actual.n_rows(), rows_);
    ASSERT_EQ(actual.dim(), cols_);
    ASSERT_EQ(actual.codes.extent(1), expected.codes.extent(1));
    ASSERT_EQ(actual.quantizer.centroid.data_handle(), encoded.quantizer.centroid.data_handle());
    ASSERT_EQ(actual.layout, expected.layout);
    ASSERT_EQ(actual.metric, expected.metric);

    ASSERT_TRUE(devArrMatchHost(expected.centroid.data_handle(),
                                quantizer.centroid.data_handle(),
                                cols_,
                                cuvs::Compare<float>(),
                                stream));
    ASSERT_TRUE(devArrMatchHost(expected.centroid.data_handle(),
                                encoded.quantizer.centroid.data_handle(),
                                cols_,
                                cuvs::Compare<float>(),
                                stream));
    ASSERT_FLOAT_EQ(expected.centroid_norm_sq, quantizer.centroid_norm_sq);
    ASSERT_FLOAT_EQ(expected.centroid_norm_sq, actual.quantizer.centroid_norm_sq);

    // A row sitting on the threshold of interval search stop can end up with a different interval.
    // Check that at least 99.5% of the rows match within 1e-4 of the reference.
    constexpr float kMinMatchFraction = 0.995f;
    const auto compare_rows           = cuvs::CompareApprox<float>(1e-4f);
    ASSERT_TRUE(devArrMatchHostFraction(expected.lower_intervals.data_handle(),
                                        actual.lower_intervals.data_handle(),
                                        rows_,
                                        compare_rows,
                                        kMinMatchFraction,
                                        stream));
    ASSERT_TRUE(devArrMatchHostFraction(expected.upper_intervals.data_handle(),
                                        actual.upper_intervals.data_handle(),
                                        rows_,
                                        compare_rows,
                                        kMinMatchFraction,
                                        stream));
    ASSERT_TRUE(devArrMatchHostFraction(expected.additional_corrections.data_handle(),
                                        actual.additional_corrections.data_handle(),
                                        rows_,
                                        compare_rows,
                                        kMinMatchFraction,
                                        stream));
    ASSERT_TRUE(devArrMatchHostFraction(expected.row_norm.data_handle(),
                                        actual.row_norm.data_handle(),
                                        rows_,
                                        compare_rows,
                                        kMinMatchFraction,
                                        stream));
    ASSERT_TRUE(devArrMatchHostFraction(expected.dequant_delta.data_handle(),
                                        actual.dequant_delta.data_handle(),
                                        rows_,
                                        compare_rows,
                                        kMinMatchFraction,
                                        stream));
    ASSERT_TRUE(devArrMatchHostFraction(expected.dequant_sum_delta.data_handle(),
                                        actual.dequant_sum_delta.data_handle(),
                                        rows_,
                                        compare_rows,
                                        kMinMatchFraction,
                                        stream));
    ASSERT_TRUE(devArrMatchHostFraction(expected.quantized_component_sums.data_handle(),
                                        actual.quantized_component_sums.data_handle(),
                                        rows_,
                                        cuvs::Compare<int32_t>(),
                                        kMinMatchFraction,
                                        stream));
    ASSERT_TRUE(devArrMatchHostFraction(expected.codes.data_handle(),
                                        actual.codes.data_handle(),
                                        actual.codes.size(),
                                        cuvs::Compare<uint8_t>(),
                                        kMinMatchFraction,
                                        stream));
  }

  void SetUp() override
  {
    rows_ = params_.rows;
    cols_ = params_.cols;

    host_input_ = make_summable_dataset(rows_, cols_, 137ULL);
    input_.resize(host_input_.size(), stream);
    raft::update_device(input_.data(), host_input_.data(), host_input_.size(), stream);

    raft::resource::sync_stream(handle, stream);
  }

 private:
  raft::resources handle;
  cudaStream_t stream;

  BbqQuantizationInputs params_;
  int64_t rows_;
  int64_t cols_;
  rmm::device_uvector<float> input_;
  std::vector<float> host_input_;
};

using cuvs::distance::DistanceType;

const std::vector<BbqQuantizationInputs> inputs = {
  // Every layout on the same dimensionality, as a single storage.
  {512, 32, bbq_code_layout::packed_1b, bbq_code_layout::packed_1b, DistanceType::L2Expanded},
  {512,
   32,
   bbq_code_layout::transposed_2b,
   bbq_code_layout::transposed_2b,
   DistanceType::L2Expanded},
  {512,
   32,
   bbq_code_layout::transposed_4b,
   bbq_code_layout::transposed_4b,
   DistanceType::L2Expanded},
  {512, 64, bbq_code_layout::packed_4b, bbq_code_layout::packed_4b, DistanceType::L2Expanded},
  {512, 32, bbq_code_layout::packed_7b, bbq_code_layout::packed_7b, DistanceType::L2Expanded},
  {512, 32, bbq_code_layout::packed_8b, bbq_code_layout::packed_8b, DistanceType::L2Expanded},
  // The asymmetric (document, query) pairs search supports, as two storages.
  {512, 32, bbq_code_layout::packed_1b, bbq_code_layout::transposed_4b, DistanceType::L2Expanded},
  {512, 32, bbq_code_layout::packed_1b, bbq_code_layout::packed_4b, DistanceType::L2Expanded},
  {512, 32, bbq_code_layout::packed_1b, bbq_code_layout::transposed_2b, DistanceType::L2Expanded},
  {512,
   32,
   bbq_code_layout::transposed_2b,
   bbq_code_layout::transposed_4b,
   DistanceType::L2Expanded},
  // Non-euclidean metrics.
  {512, 64, bbq_code_layout::packed_1b, bbq_code_layout::transposed_4b, DistanceType::InnerProduct},
  {512, 64, bbq_code_layout::packed_8b, bbq_code_layout::packed_8b, DistanceType::CosineExpanded},
  // Rows shorter than a warp, and tails in the middle of a byte, a group and a nibble pair.
  {64, 5, bbq_code_layout::packed_1b, bbq_code_layout::transposed_4b, DistanceType::L2Expanded},
  {64,
   12,
   bbq_code_layout::transposed_2b,
   bbq_code_layout::transposed_2b,
   DistanceType::L2Expanded},
  {64,
   33,
   bbq_code_layout::transposed_2b,
   bbq_code_layout::transposed_4b,
   DistanceType::L2Expanded},
  {64, 41, bbq_code_layout::packed_1b, bbq_code_layout::packed_4b, DistanceType::L2Expanded},
  {64, 41, bbq_code_layout::packed_1b, bbq_code_layout::packed_1b, DistanceType::L2Expanded},
  {64, 7, bbq_code_layout::packed_8b, bbq_code_layout::packed_8b, DistanceType::L2Expanded},
  // A single row, and enough rows to span several blocks.
  {1, 128, bbq_code_layout::packed_1b, bbq_code_layout::transposed_4b, DistanceType::L2Expanded},
  {1,
   33,
   bbq_code_layout::transposed_2b,
   bbq_code_layout::transposed_2b,
   DistanceType::InnerProduct},
  {4096, 96, bbq_code_layout::packed_1b, bbq_code_layout::transposed_4b, DistanceType::L2Expanded},
};

TEST_P(BbqQuantizationTest, BbqQuantizationTest) { this->testBbqQuantization(); }

INSTANTIATE_TEST_CASE_P(BbqQuantizationTest, BbqQuantizationTest, ::testing::ValuesIn(inputs));

// Decode the output codes and check they reconstruct the input rows within an error budget.
TEST(BbqQuantization, ReconstructsRowsWithinErrorBudget)
{
  constexpr int64_t kRows = 2048;
  constexpr int64_t kDim  = 128;
  constexpr auto kMetric  = cuvs::distance::DistanceType::L2Expanded;
  const std::vector<std::pair<bbq_code_layout, float>> budgets{
    {bbq_code_layout::packed_1b, 0.5f},       // measured 0.4953
    {bbq_code_layout::transposed_2b, 0.15f},  // measured 0.1269
    {bbq_code_layout::packed_4b, 0.01f},      // measured 0.00997
    {bbq_code_layout::transposed_4b, 0.01f},  // same codes as packed_4b
    {bbq_code_layout::packed_7b, 2.0e-4f},    // measured 1.44e-4
    {bbq_code_layout::packed_8b, 4.0e-5f}};   // measured 3.52e-5

  raft::resources res;
  auto dataset = raft::make_device_matrix<float, int64_t>(res, kRows, kDim);
  raft::random::RngState rng{137ULL};
  raft::random::normal(res, rng, dataset.data_handle(), dataset.size(), 0.5f, 2.0f);
  const auto host_dataset = to_host(res, dataset.data_handle(), static_cast<size_t>(kRows * kDim));

  const auto view      = raft::make_const_mdspan(dataset.view());
  const auto quantizer = build(res, view);

  std::vector<double> errors;
  std::vector<std::vector<uint8_t>> codes;
  for (const auto& [layout, budget] : budgets) {
    const auto encoded = transform(res, params{layout, layout, kMetric}, quantizer.view(), view);
    const auto& q      = encoded.storages[0];
    codes.push_back(unpack_codes(res, q, kRows, kDim));
    errors.push_back(
      relative_reconstruction_error(res, q, codes.back(), host_dataset, kRows, kDim));
    std::cout << "[ RECON    ] " << get_bit_width(layout) << " bits, layout "
              << static_cast<int>(layout) << ": " << errors.back() << std::endl;
    EXPECT_LT(errors.back(), budget)
      << get_bit_width(layout) << " bits, layout " << static_cast<int>(layout);
  }

  // The two 4-bit layouts differ only in which byte a code lands in, so they have to carry the
  // same codes; a decoder that disagrees has misread one of the two.
  int64_t four_bit_mismatches = 0;
  for (size_t i = 0; i < codes[2].size(); ++i) {
    if (codes[2][i] != codes[3][i]) { ++four_bit_mismatches; }
  }
  EXPECT_EQ(four_bit_mismatches, 0) << four_bit_mismatches << " of " << codes[2].size()
                                    << " codes differ between packed_4b and transposed_4b";
  // Wider codes have to be worth their bytes.
  EXPECT_LT(errors[1], errors[0]) << "2 bits is no better than 1";
  EXPECT_LT(errors[2], errors[1]) << "4 bits is no better than 2";
  EXPECT_LT(errors[4], errors[2]) << "7 bits is no better than 4";
  EXPECT_LT(errors[5], errors[4]) << "8 bits is no better than 7";
}
}  // namespace cuvs::preprocessing::quantize::bbq
