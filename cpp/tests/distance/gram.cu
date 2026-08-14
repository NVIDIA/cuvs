/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "gram_base.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/distance/grammian.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <memory>

namespace cuvs::distance::kernels {

struct GramMatrixInputs {
  int n1;      // feature vectors in matrix 1
  int n2;      // feature vectors in matrix 2
  int n_cols;  // number of elements in a feature vector
  bool is_row_major;
  KernelParams kernel;
  int ld1;
  int ld2;
  int ld_out;
  // We will generate random input using the dimensions given here.
  // The reference output is calculated by a custom kernel.
};

std::ostream& operator<<(std::ostream& os, const GramMatrixInputs& p)
{
  std::vector<std::string> kernel_names{"linear", "poly", "rbf", "tanh"};
  os << "/" << p.n1 << "x" << p.n2 << "x" << p.n_cols << "/"
     << (p.is_row_major ? "RowMajor/" : "ColMajor/") << kernel_names[p.kernel.kernel] << "/ld_"
     << p.ld1 << "x" << p.ld2 << "x" << p.ld_out;
  return os;
}

const std::vector<GramMatrixInputs> inputs = {
  {42, 137, 2, false, {KernelType::LINEAR}},
  {42, 137, 2, true, {KernelType::LINEAR}},
  {42, 137, 2, false, {KernelType::LINEAR}, 64, 179, 181},
  {42, 137, 2, true, {KernelType::LINEAR}, 64, 179, 181},
  {137, 42, 2, false, {KernelType::POLYNOMIAL, 2, 0.5, 2.4}},
  {137, 42, 2, true, {KernelType::POLYNOMIAL, 2, 0.5, 2.4}},
  {137, 42, 2, false, {KernelType::POLYNOMIAL, 2, 0.5, 2.4}, 159, 73, 144},
  {137, 42, 2, true, {KernelType::POLYNOMIAL, 2, 0.5, 2.4}, 159, 73, 144},
  {42, 137, 2, false, {KernelType::TANH, 0, 0.5, 2.4}},
  {42, 137, 2, true, {KernelType::TANH, 0, 0.5, 2.4}},
  {42, 137, 2, false, {KernelType::TANH, 0, 0.5, 2.4}, 64, 155, 49},
  {42, 137, 2, true, {KernelType::TANH, 0, 0.5, 2.4}, 64, 155, 143},
  {3, 4, 2, false, {KernelType::RBF, 0, 0.5}},
  {42, 137, 2, false, {KernelType::RBF, 0, 0.5}},
  {42, 137, 2, true, {KernelType::RBF, 0, 0.5}},
  // Distance kernel does not support LD parameter yet.
  //{42, 137, 2, false, {KernelType::RBF, 0, 0.5}, 64, 155, 49},
  // {42, 137, 2, true, {KernelType::RBF, 0, 0.5}, 64, 155, 143},
};

template <typename math_t>
class GramMatrixTest : public ::testing::TestWithParam<GramMatrixInputs> {
 protected:
  GramMatrixTest()
    : params(GetParam()),
      handle(),
      x1(0, raft::resource::get_cuda_stream(handle)),
      x2(0, raft::resource::get_cuda_stream(handle)),
      gram(0, raft::resource::get_cuda_stream(handle)),
      gram_host(0)
  {
    auto stream = raft::resource::get_cuda_stream(handle);

    if (params.ld1 == 0) { params.ld1 = params.is_row_major ? params.n_cols : params.n1; }
    if (params.ld2 == 0) { params.ld2 = params.is_row_major ? params.n_cols : params.n2; }
    if (params.ld_out == 0) { params.ld_out = params.is_row_major ? params.n2 : params.n1; }
    // Derive the size of the output from the offset of the last element.
    size_t size = get_offset(params.n1 - 1, params.n_cols - 1, params.ld1, params.is_row_major) + 1;
    x1.resize(size, stream);
    size = get_offset(params.n2 - 1, params.n_cols - 1, params.ld2, params.is_row_major) + 1;
    x2.resize(size, stream);
    size = get_offset(params.n1 - 1, params.n2 - 1, params.ld_out, params.is_row_major) + 1;

    gram.resize(size, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(gram.data(), 0, gram.size() * sizeof(math_t), stream));
    gram_host.resize(gram.size());
    std::fill(gram_host.begin(), gram_host.end(), 0);

    raft::random::RngState rng(42137ULL);
    raft::random::uniform(handle, rng, x1.data(), x1.size(), math_t(0), math_t(1));
    raft::random::uniform(handle, rng, x2.data(), x2.size(), math_t(0), math_t(1));
  }

  ~GramMatrixTest() override {}

  void runTest()
  {
    std::unique_ptr<GramMatrixBase<math_t>> kernel =
      std::unique_ptr<GramMatrixBase<math_t>>(KernelFactory<math_t>::create(params.kernel));

    auto x1_span =
      params.is_row_major
        ? raft::make_device_strided_matrix_view<const math_t, int, raft::layout_c_contiguous>(
            x1.data(), params.n1, params.n_cols, params.ld1)
        : raft::make_device_strided_matrix_view<const math_t, int, raft::layout_f_contiguous>(
            x1.data(), params.n1, params.n_cols, params.ld1);
    auto x2_span =
      params.is_row_major
        ? raft::make_device_strided_matrix_view<const math_t, int, raft::layout_c_contiguous>(
            x2.data(), params.n2, params.n_cols, params.ld2)
        : raft::make_device_strided_matrix_view<const math_t, int, raft::layout_f_contiguous>(
            x2.data(), params.n2, params.n_cols, params.ld2);
    auto out_span =
      params.is_row_major
        ? raft::make_device_strided_matrix_view<math_t, int, raft::layout_c_contiguous>(
            gram.data(), params.n1, params.n2, params.ld_out)
        : raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
            gram.data(), params.n1, params.n2, params.ld_out);

    (*kernel)(handle, x1_span, x2_span, out_span);

    auto stream = raft::resource::get_cuda_stream(handle);
    naiveGramMatrixKernel(params.n1,
                          params.n2,
                          params.n_cols,
                          x1,
                          x2,
                          gram_host.data(),
                          params.ld1,
                          params.ld2,
                          params.ld_out,
                          params.is_row_major,
                          params.kernel,
                          stream,
                          handle);

    ASSERT_TRUE(cuvs::devArrMatchHost(
      gram_host.data(), gram.data(), gram.size(), cuvs::CompareApprox<math_t>(1e-6f), stream));
  }

  GramMatrixInputs params;
  raft::resources handle;

  rmm::device_uvector<math_t> x1;
  rmm::device_uvector<math_t> x2;
  rmm::device_uvector<math_t> gram;

  std::vector<math_t> gram_host;
};

typedef GramMatrixTest<float> GramMatrixTestFloat;
typedef GramMatrixTest<double> GramMatrixTestDouble;

TEST_P(GramMatrixTestFloat, Gram) { runTest(); }

INSTANTIATE_TEST_SUITE_P(GramMatrixTests, GramMatrixTestFloat, ::testing::ValuesIn(inputs));
};  // namespace cuvs::distance::kernels
