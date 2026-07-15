/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_nn_helper.cuh"

#include "../../src/distance/fused_distance_nn.cuh"
#include "../../src/distance/unfused_distance_nn.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/init.cuh>

namespace cuvs::neighbors {

enum class ImplType { fused, unfused };

template <typename IdxT>
struct NNInputs {
  IdxT m;
  IdxT n;
  IdxT k;
  DistanceType metric;
  bool sqrt;
  uint64_t rng_seed;
  double tol;
};

__global__ void fill_int8(int8_t* buff, int len, int seed_offset)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // Fill the buffer with pseudo-random int8_t values using a simple LCG
  for (int i = tid; i < len; i += blockDim.x * gridDim.x) {
    int hash = (i + seed_offset) * 1103515245 + 12345;
    buff[i]  = static_cast<int8_t>((hash >> 16) & 0xFF);
  }
}

template <typename DataT, typename AccT, typename IdxT, ImplType impl>
class NNTest : public ::testing::TestWithParam<NNInputs<IdxT>> {
 public:
  using OutT = raft::KeyValuePair<IdxT, AccT>;
  NNTest()
    : params_{::testing::TestWithParam<NNInputs<IdxT>>::GetParam()},
      m{params_.m},
      n{params_.n},
      k{params_.k},
      metric{params_.metric},
      sqrt{params_.sqrt},
      stream{raft::resource::get_cuda_stream(handle)},
      x{raft::make_device_matrix<DataT, IdxT>(handle, m, k)},
      y{raft::make_device_matrix<DataT, IdxT>(handle, n, k)},
      x_norm{raft::make_device_vector<AccT, IdxT>(handle, m)},
      y_norm{raft::make_device_vector<AccT, IdxT>(handle, n)},
      out{raft::make_device_vector<OutT, IdxT>(handle, m)},
      ref_out{raft::make_device_vector<OutT, IdxT>(handle, m)}
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState rng{params_.rng_seed};
    if constexpr (std::is_same_v<DataT, int8_t>) {
      fill_int8<<<1000, 256, 0, stream>>>(x.data_handle(), m * k, 0);
      RAFT_CUDA_TRY(cudaGetLastError());
      fill_int8<<<1000, 256, 0, stream>>>(y.data_handle(), n * k, m * k);
      RAFT_CUDA_TRY(cudaGetLastError());
    } else {
      raft::random::uniform(handle, rng, x.data_handle(), m * k, DataT(-1.0), DataT(1.0));
      raft::random::uniform(handle, rng, y.data_handle(), n * k, DataT(-1.0), DataT(1.0));
    }

    // Pre-compute norms
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      x_norm.data_handle(), x.data_handle(), k, m, stream);
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      y_norm.data_handle(), y.data_handle(), k, n, stream);

    // CosineExpanded expects ||x|| not ||x||^2
    if (metric == DistanceType::CosineExpanded) {
      raft::linalg::unaryOp(x_norm.data_handle(), x_norm.data_handle(), m, raft::sqrt_op{}, stream);
      raft::linalg::unaryOp(y_norm.data_handle(), y_norm.data_handle(), n, raft::sqrt_op{}, stream);
    }

    if constexpr (impl == ImplType::fused) {
      workspace_size = m * sizeof(IdxT);
    } else if constexpr (impl == ImplType::unfused) {
      workspace_size = m * n * sizeof(AccT);
    }

    // Reset buffer
    if constexpr (std::is_same_v<OutT, raft::KeyValuePair<IdxT, AccT>>) {
      // OutT is a RAFT KeyValuePair
      raft::matrix::fill(
        handle, raft::make_device_matrix_view(out.data_handle(), m, 1), OutT{0, 0});
    } else {
      // OutT is a scalar type
      raft::matrix::fill(handle, raft::make_device_matrix_view(out.data_handle(), m, 1), OutT{0});
    }
    raft::resource::sync_stream(handle, stream);
  }

  void compute_1nn()
  {
    raft::device_vector<char, IdxT> workspace =
      raft::make_device_vector<char, IdxT>(handle, workspace_size);

    ref_nn<DataT, AccT, OutT, IdxT>(
      ref_out.data_handle(), x.data_handle(), y.data_handle(), m, n, k, sqrt, metric, stream);

    if constexpr (impl == ImplType::fused) {
      if constexpr (std::is_same_v<DataT, float>) {
        cuvs::distance::fusedDistanceNNMinReduce<DataT, OutT, IdxT>(out.data_handle(),
                                                                    x.data_handle(),
                                                                    y.data_handle(),
                                                                    x_norm.data_handle(),
                                                                    y_norm.data_handle(),
                                                                    m,
                                                                    n,
                                                                    k,
                                                                    (void*)workspace.data_handle(),
                                                                    sqrt,
                                                                    true,
                                                                    true,
                                                                    metric,
                                                                    0.0,
                                                                    stream);
      } else {
        static_assert(sizeof(DataT) == 0,
                      "fusedDistanceNNMinReduce is not implemented for datatype other than float");
      }
    } else if constexpr (impl == ImplType::unfused) {
      cuvs::distance::unfusedDistanceNNMinReduce<DataT, AccT, OutT, IdxT>(
        handle,
        out.data_handle(),
        x.data_handle(),
        y.data_handle(),
        x_norm.data_handle(),
        y_norm.data_handle(),
        m,
        n,
        k,
        (AccT*)workspace.data_handle(),
        sqrt,
        true,
        true,
        metric,
        0.0,
        stream);
    }
  }

  void compare()
  {
    vector_compare(handle, ref_out.data_handle(), out.data_handle(), m, summary);
    ASSERT_TRUE(summary.max_diff < params_.tol) << summary;
  }

 private:
  raft::resources handle;
  rmm::cuda_stream_view stream;
  NNInputs<IdxT> params_;
  ComparisonSummary summary;
  IdxT m;
  IdxT n;
  IdxT k;
  DistanceType metric;
  bool sqrt;
  raft::device_matrix<DataT, IdxT> x;
  raft::device_matrix<DataT, IdxT> y;
  raft::device_vector<AccT, IdxT> x_norm;
  raft::device_vector<AccT, IdxT> y_norm;
  raft::device_vector<OutT, IdxT> out;
  raft::device_vector<OutT, IdxT> ref_out;
  size_t workspace_size;
};

template <typename IdxT>
const std::vector<NNInputs<IdxT>> input_fp32 = {
  {4096, 4096, 64, DistanceType::L2Expanded, false, uint64_t(31415926), 0.1},
  {16384, 4096, 64, DistanceType::L2Expanded, false, uint64_t(31415926), 0.1},
  {4096, 4096, 128, DistanceType::L2Expanded, true, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2Expanded, true, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::CosineExpanded, false, uint64_t(31415926), 0.1},
  {8192, 4096, 64, DistanceType::CosineExpanded, false, uint64_t(31415926), 0.1},
  // Fused implementation for cosine distance ignores the sqrt parameter, therefore
  // commenting the following two tests
  // {4096, 4096, 128, DistanceType::CosineExpanded, true, uint64_t(31415926), 0.1},
  // {4096, 8192, 128, DistanceType::CosineExpanded, true, uint64_t(31415926), 0.1},
};

// Test fused implementation with single-precision
typedef NNTest<float, float, int32_t, ImplType::fused> NNTest_fp32_fused;
TEST_P(NNTest_fp32_fused, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_fp32_fused, ::testing::ValuesIn(input_fp32<int>));

// Test unfused implementation with single-precision
typedef NNTest<float, float, int32_t, ImplType::unfused> NNTest_fp32_unfused;
TEST_P(NNTest_fp32_unfused, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_fp32_unfused, ::testing::ValuesIn(input_fp32<int>));

template <typename IdxT>
const std::vector<NNInputs<IdxT>> input_fp16 = {
  {4096, 4096, 64, DistanceType::L2Expanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2Expanded, true, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::CosineExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::CosineExpanded, true, uint64_t(31415926), 0.1},
};

// Test unfused implementation with fp16, int8
// Fused implementation has no support for fp16, int8 so no test for it
typedef NNTest<half, float, int32_t, ImplType::unfused> NNTest_fp16_unfused;
TEST_P(NNTest_fp16_unfused, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_fp16_unfused, ::testing::ValuesIn(input_fp16<int>));

template <typename IdxT>
const std::vector<NNInputs<IdxT>> input_int8 = {
  {4096, 4096, 64, DistanceType::L2Expanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2Expanded, true, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::CosineExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::CosineExpanded, true, uint64_t(31415926), 0.1},
};

// DataT = int8_t, AccT = int32_t
typedef NNTest<int8_t, int32_t, int32_t, ImplType::unfused> NNTest_int8_unfused;
TEST_P(NNTest_int8_unfused, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_int8_unfused, ::testing::ValuesIn(input_int8<int>));

// DataT = int8_t, AccT = float
typedef NNTest<int8_t, float, int32_t, ImplType::unfused> NNTest_int8_unfused2;
TEST_P(NNTest_int8_unfused2, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_int8_unfused2, ::testing::ValuesIn(input_int8<int>));
}  // namespace cuvs::neighbors
