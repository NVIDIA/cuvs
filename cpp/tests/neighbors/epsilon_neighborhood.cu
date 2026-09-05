/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <cuvs/neighbors/ball_cover.hpp>
#include <cuvs/neighbors/epsilon_neighborhood.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/init.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <memory>

namespace cuvs::neighbors::epsilon_neighborhood {

template <typename T, typename IdxT>
struct EpsInputs {
  IdxT n_row, n_col, n_centers, n_batches;
  T eps;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const EpsInputs<T, IdxT>& p)
{
  return os;
}

template <typename T, typename IdxT>
class EpsNeighTest : public ::testing::TestWithParam<EpsInputs<T, IdxT>> {
 protected:
  EpsNeighTest()
    : data(0, raft::resource::get_cuda_stream(handle)),
      adj(0, raft::resource::get_cuda_stream(handle)),
      labels(0, raft::resource::get_cuda_stream(handle)),
      vd(0, raft::resource::get_cuda_stream(handle))
  {
  }

  void SetUp() override
  {
    auto stream = raft::resource::get_cuda_stream(handle);
    param       = ::testing::TestWithParam<EpsInputs<T, IdxT>>::GetParam();
    data.resize(param.n_row * param.n_col, stream);
    labels.resize(param.n_row, stream);
    batchSize = param.n_row / param.n_batches;
    adj.resize(param.n_row * batchSize, stream);
    vd.resize(batchSize + 1, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(vd.data(), 0, vd.size() * sizeof(IdxT), stream.get()));
    raft::random::make_blobs<T, IdxT>(data.data(),
                                      labels.data(),
                                      param.n_row,
                                      param.n_col,
                                      param.n_centers,
                                      stream.get(),
                                      true,
                                      nullptr,
                                      nullptr,
                                      T(0.01),
                                      false);
  }

  const raft::resources handle;
  EpsInputs<T, IdxT> param;
  cudaStream_t stream = 0;
  rmm::device_uvector<T> data;
  rmm::device_uvector<bool> adj;
  rmm::device_uvector<IdxT> labels, vd;
  IdxT batchSize;
};  // class EpsNeighTest

const std::vector<EpsInputs<float, int64_t>> inputsfi = {
  {100, 16, 5, 2, 2.f},
  {1500, 16, 5, 3, 2.f},
  {15000, 16, 5, 1, 2.f},
  {15000, 3, 5, 1, 2.f},
  {14000, 16, 5, 1, 2.f},
  {15000, 17, 5, 1, 2.f},
  {14000, 17, 5, 1, 2.f},
  {15000, 18, 5, 1, 2.f},
  {14000, 18, 5, 1, 2.f},
  {15000, 32, 5, 1, 2.f},
  {14000, 32, 5, 1, 2.f},
  {14000, 32, 5, 10, 2.f},
  {20000, 10000, 10, 1, 2.f},
  {20000, 10000, 10, 2, 2.f},
};

typedef EpsNeighTest<float, int64_t> EpsNeighTestFI;

TEST_P(EpsNeighTestFI, ResultBruteForce)
{
  for (int i = 0; i < param.n_batches; ++i) {
    RAFT_CUDA_TRY(cudaMemsetAsync(adj.data(), 0, sizeof(bool) * param.n_row * batchSize, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(vd.data(), 0, sizeof(int64_t) * (batchSize + 1), stream));

    auto adj_view =
      raft::make_device_matrix_view<bool, int64_t>(adj.data(), batchSize, param.n_row);
    auto vd_view = raft::make_device_vector_view<int64_t, int64_t>(vd.data(), batchSize + 1);
    auto x_view  = raft::make_device_matrix_view<float, int64_t>(
      data.data() + (i * batchSize * param.n_col), batchSize, param.n_col);
    auto y_view =
      raft::make_device_matrix_view<float, int64_t>(data.data(), param.n_row, param.n_col);

    cuvs::neighbors::epsilon_neighborhood::compute<float, int64_t, int64_t>(
      handle,
      x_view,
      y_view,
      adj_view,
      vd_view,
      param.eps * param.eps,
      cuvs::distance::DistanceType::L2Unexpanded);

    ASSERT_TRUE(cuvs::devArrMatch(
      param.n_row / param.n_centers, vd.data(), batchSize, cuvs::Compare<int64_t>(), stream));
  }
}

INSTANTIATE_TEST_CASE_P(EpsNeighTests, EpsNeighTestFI, ::testing::ValuesIn(inputsfi));

// rbc examples take fewer points as correctness checks are very costly
const std::vector<EpsInputs<float, int64_t>> inputsfi_rbc = {
  {100, 16, 5, 2, 2.f},
  {1500, 16, 5, 3, 2.f},
  {1500, 16, 5, 1, 2.f},
  {1500, 3, 5, 1, 2.f},
  {1400, 16, 5, 1, 2.f},
  {1500, 17, 5, 1, 2.f},
  {1400, 17, 5, 1, 2.f},
  {1500, 18, 5, 1, 2.f},
  {1400, 18, 5, 1, 2.f},
  {1500, 32, 5, 1, 2.f},
  {1400, 32, 5, 1, 2.f},
  {1400, 32, 5, 10, 2.f},
  {2000, 1000, 10, 1, 2.f},
  {2000, 1000, 10, 2, 2.f},
};

typedef EpsNeighTest<float, int64_t> EpsNeighRbcTestFI;

TEST_P(EpsNeighRbcTestFI, DenseRbc)
{
  auto adj_baseline = raft::make_device_matrix<bool, int64_t>(handle, batchSize, param.n_row);

  auto data_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    data.data(), param.n_row, param.n_col);
  cuvs::neighbors::ball_cover::index<int64_t, float> rbc_index(
    handle, data_view, cuvs::distance::DistanceType::L2SqrtUnexpanded);
  cuvs::neighbors::ball_cover::build(handle, rbc_index);

  for (int i = 0; i < param.n_batches; ++i) {
    // invalidate
    RAFT_CUDA_TRY(cudaMemsetAsync(adj.data(), 1, sizeof(bool) * param.n_row * batchSize, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(vd.data(), 1, sizeof(int64_t) * (batchSize + 1), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      adj_baseline.data_handle(), 1, sizeof(bool) * param.n_row * batchSize, stream));

    float* query = data.data() + (i * batchSize * param.n_col);

    cuvs::neighbors::ball_cover::eps_nn(
      handle,
      rbc_index,
      raft::make_device_matrix_view<bool, int64_t>(adj.data(), batchSize, param.n_row),
      raft::make_device_vector_view<int64_t, int64_t>(vd.data(), batchSize + 1),
      raft::make_device_matrix_view<float, int64_t>(query, batchSize, param.n_col),
      param.eps * param.eps);

    ASSERT_TRUE(cuvs::devArrMatch(
      param.n_row / param.n_centers, vd.data(), batchSize, cuvs::Compare<int64_t>(), stream));

    // compute baseline via brute force + compare
    cuvs::neighbors::epsilon_neighborhood::compute<float, int64_t, int64_t>(
      handle,
      raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
        query, batchSize, param.n_col),
      raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
        data.data(), param.n_row, param.n_col),
      raft::make_device_matrix_view<bool, int64_t, raft::row_major>(
        adj_baseline.data_handle(), batchSize, param.n_row),
      raft::make_device_vector_view<int64_t, int64_t>(vd.data(), batchSize + 1),
      param.eps * param.eps,
      cuvs::distance::DistanceType::L2Unexpanded);

    ASSERT_TRUE(cuvs::devArrMatch(adj_baseline.data_handle(),
                                  adj.data(),
                                  batchSize,
                                  param.n_row,
                                  cuvs::Compare<bool>(),
                                  stream));
  }
}

template <typename T>
testing::AssertionResult assertCsrEqualUnordered(
  T* ia_exp, T* ja_exp, T* ia_act, T* ja_act, size_t rows, size_t cols, cudaStream_t stream)
{
  std::unique_ptr<T[]> ia_exp_h(new T[rows + 1]);
  std::unique_ptr<T[]> ia_act_h(new T[rows + 1]);
  raft::update_host<T>(ia_exp_h.get(), ia_exp, rows + 1, stream);
  raft::update_host<T>(ia_act_h.get(), ia_act, rows + 1, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  size_t nnz = ia_exp_h.get()[rows];
  std::unique_ptr<T[]> ja_exp_h(new T[nnz]);
  std::unique_ptr<T[]> ja_act_h(new T[nnz]);
  raft::update_host<T>(ja_exp_h.get(), ja_exp, nnz, stream);
  raft::update_host<T>(ja_act_h.get(), ja_act, nnz, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  for (size_t i(0); i < rows; ++i) {
    auto row_start = ia_exp_h.get()[i];
    auto row_end   = ia_exp_h.get()[i + 1];

    // sort ja's
    std::sort(ja_exp_h.get() + row_start, ja_exp_h.get() + row_end);
    std::sort(ja_act_h.get() + row_start, ja_act_h.get() + row_end);

    for (size_t idx(row_start); idx < (size_t)row_end; ++idx) {
      auto exp = ja_exp_h.get()[idx];
      auto act = ja_act_h.get()[idx];
      if (exp != act) {
        return testing::AssertionFailure()
               << "actual=" << act << " != expected=" << exp << " @" << i << "," << idx;
      }
    }
  }
  return testing::AssertionSuccess();
}

TEST_P(EpsNeighRbcTestFI, SparseRbc)
{
  auto adj_ia          = raft::make_device_vector<int64_t>(handle, batchSize + 1);
  auto adj_ja          = raft::make_device_vector<int64_t>(handle, param.n_row * batchSize);
  auto vd_baseline     = raft::make_device_vector<int64_t>(handle, batchSize + 1);
  auto adj_ia_baseline = raft::make_device_vector<int64_t>(handle, batchSize + 1);
  auto adj_ja_baseline = raft::make_device_vector<int64_t>(handle, param.n_row * batchSize);

  auto data_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    data.data(), param.n_row, param.n_col);
  cuvs::neighbors::ball_cover::index<int64_t, float> rbc_index(
    handle, data_view, cuvs::distance::DistanceType::L2SqrtUnexpanded);
  cuvs::neighbors::ball_cover::build(handle, rbc_index);

  for (int i = 0; i < param.n_batches; ++i) {
    // reset full array -- that way we can compare the full size
    RAFT_CUDA_TRY(
      cudaMemsetAsync(adj_ja.data_handle(), 0, sizeof(int64_t) * param.n_row * batchSize, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      adj_ja_baseline.data_handle(), 0, sizeof(int64_t) * param.n_row * batchSize, stream));

    float* query = data.data() + (i * batchSize * param.n_col);

    // compute dense baseline and convert adj to csr
    {
      cuvs::neighbors::ball_cover::eps_nn(
        handle,
        rbc_index,
        raft::make_device_matrix_view<bool, int64_t>(adj.data(), batchSize, param.n_row),
        raft::make_device_vector_view<int64_t, int64_t>(vd_baseline.data_handle(), batchSize + 1),
        raft::make_device_matrix_view<float, int64_t>(query, batchSize, param.n_col),
        param.eps * param.eps);
      thrust::exclusive_scan(raft::resource::get_thrust_policy(handle),
                             vd_baseline.data_handle(),
                             vd_baseline.data_handle() + batchSize + 1,
                             adj_ia_baseline.data_handle());
      raft::sparse::convert::adj_to_csr(handle,
                                        adj.data(),
                                        adj_ia_baseline.data_handle(),
                                        batchSize,
                                        param.n_row,
                                        labels.data(),
                                        adj_ja_baseline.data_handle());
    }

    // exact computation with 2 passes
    {
      cuvs::neighbors::ball_cover::eps_nn(
        handle,
        rbc_index,
        raft::make_device_vector_view<int64_t, int64_t>(adj_ia.data_handle(), batchSize + 1),
        raft::make_device_vector_view<int64_t, int64_t>(nullptr, 0),
        raft::make_device_vector_view<int64_t, int64_t>(vd.data(), batchSize + 1),
        raft::make_device_matrix_view<float, int64_t>(query, batchSize, param.n_col),
        param.eps * param.eps);
      cuvs::neighbors::ball_cover::eps_nn(
        handle,
        rbc_index,
        raft::make_device_vector_view<int64_t, int64_t>(adj_ia.data_handle(), batchSize + 1),
        raft::make_device_vector_view<int64_t, int64_t>(adj_ja.data_handle(),
                                                        batchSize * param.n_row),
        raft::make_device_vector_view<int64_t, int64_t>(vd.data(), batchSize + 1),
        raft::make_device_matrix_view<float, int64_t>(query, batchSize, param.n_col),
        param.eps * param.eps);
      ASSERT_TRUE(cuvs::devArrMatch(adj_ia_baseline.data_handle(),
                                    adj_ia.data_handle(),
                                    batchSize + 1,
                                    cuvs::Compare<int64_t>(),
                                    stream));
      ASSERT_TRUE(assertCsrEqualUnordered(adj_ia_baseline.data_handle(),
                                          adj_ja_baseline.data_handle(),
                                          adj_ia.data_handle(),
                                          adj_ja.data_handle(),
                                          batchSize,
                                          param.n_row,
                                          stream));
    }
  }
}

TEST_P(EpsNeighRbcTestFI, SparseRbcMaxK)
{
  auto adj_ia          = raft::make_device_vector<int64_t>(handle, batchSize + 1);
  auto adj_ja          = raft::make_device_vector<int64_t>(handle, param.n_row * batchSize);
  auto vd_baseline     = raft::make_device_vector<int64_t>(handle, batchSize + 1);
  auto adj_ia_baseline = raft::make_device_vector<int64_t>(handle, batchSize + 1);
  auto adj_ja_baseline = raft::make_device_vector<int64_t>(handle, param.n_row * batchSize);

  auto data_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    data.data(), param.n_row, param.n_col);
  cuvs::neighbors::ball_cover::index<int64_t, float> rbc_index(
    handle, data_view, cuvs::distance::DistanceType::L2SqrtUnexpanded);
  cuvs::neighbors::ball_cover::build(handle, rbc_index);

  int64_t expected_max_k = param.n_row / param.n_centers;

  for (int i = 0; i < param.n_batches; ++i) {
    // reset full array -- that way we can compare the full size
    RAFT_CUDA_TRY(
      cudaMemsetAsync(adj_ja.data_handle(), 0, sizeof(int64_t) * param.n_row * batchSize, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      adj_ja_baseline.data_handle(), 0, sizeof(int64_t) * param.n_row * batchSize, stream));

    float* query = data.data() + (i * batchSize * param.n_col);

    // compute dense baseline and convert adj to csr
    {
      cuvs::neighbors::ball_cover::eps_nn(
        handle,
        rbc_index,
        raft::make_device_matrix_view<bool, int64_t>(adj.data(), batchSize, param.n_row),
        raft::make_device_vector_view<int64_t, int64_t>(vd_baseline.data_handle(), batchSize + 1),
        raft::make_device_matrix_view<float, int64_t>(query, batchSize, param.n_col),
        param.eps * param.eps);
      thrust::exclusive_scan(raft::resource::get_thrust_policy(handle),
                             vd_baseline.data_handle(),
                             vd_baseline.data_handle() + batchSize + 1,
                             adj_ia_baseline.data_handle());
      raft::sparse::convert::adj_to_csr(handle,
                                        adj.data(),
                                        adj_ia_baseline.data_handle(),
                                        batchSize,
                                        param.n_row,
                                        labels.data(),
                                        adj_ja_baseline.data_handle());
    }

    // exact computation with 1 pass
    {
      int64_t max_k = expected_max_k;
      cuvs::neighbors::ball_cover::eps_nn(
        handle,
        rbc_index,
        raft::make_device_vector_view<int64_t, int64_t>(adj_ia.data_handle(), batchSize + 1),
        raft::make_device_vector_view<int64_t, int64_t>(adj_ja.data_handle(),
                                                        batchSize * param.n_row),
        raft::make_device_vector_view<int64_t, int64_t>(vd.data(), batchSize + 1),
        raft::make_device_matrix_view<float, int64_t>(query, batchSize, param.n_col),
        param.eps * param.eps,
        raft::make_host_scalar_view<int64_t, int64_t>(&max_k));
      ASSERT_TRUE(cuvs::devArrMatch(adj_ia_baseline.data_handle(),
                                    adj_ia.data_handle(),
                                    batchSize + 1,
                                    cuvs::Compare<int64_t>(),
                                    stream));
      ASSERT_TRUE(assertCsrEqualUnordered(adj_ia_baseline.data_handle(),
                                          adj_ja_baseline.data_handle(),
                                          adj_ia.data_handle(),
                                          adj_ja.data_handle(),
                                          batchSize,
                                          param.n_row,
                                          stream));
      ASSERT_TRUE(cuvs::devArrMatch(
        vd_baseline.data_handle(), vd.data(), batchSize + 1, cuvs::Compare<int64_t>(), stream));
      ASSERT_TRUE(max_k == expected_max_k);
    }

    // k-limited computation with 1 pass
    {
      int64_t max_k = expected_max_k / 2;
      cuvs::neighbors::ball_cover::eps_nn(
        handle,
        rbc_index,
        raft::make_device_vector_view<int64_t, int64_t>(adj_ia.data_handle(), batchSize + 1),
        raft::make_device_vector_view<int64_t, int64_t>(adj_ja.data_handle(),
                                                        batchSize * param.n_row),
        raft::make_device_vector_view<int64_t, int64_t>(vd.data(), batchSize + 1),
        raft::make_device_matrix_view<float, int64_t>(query, batchSize, param.n_col),
        param.eps * param.eps,
        raft::make_host_scalar_view<int64_t, int64_t>(&max_k));
      ASSERT_TRUE(max_k == expected_max_k);
      ASSERT_TRUE(cuvs::devArrMatch(
        expected_max_k / 2, vd.data(), batchSize, cuvs::Compare<int64_t>(), stream));
      ASSERT_TRUE(cuvs::devArrMatch(expected_max_k / 2 * batchSize,
                                    vd.data() + batchSize,
                                    1,
                                    cuvs::Compare<int64_t>(),
                                    stream));
    }
  }
}

INSTANTIATE_TEST_CASE_P(EpsNeighTests, EpsNeighRbcTestFI, ::testing::ValuesIn(inputsfi_rbc));

TEST(EpsNeighborhood, LargeNDimension)
{
  // n just past the grid.y=65535 limit for Nblk=16
  int64_t m = 1, n = 65536 * 16 + 1, k = 4;
  float eps = 1e10f;  // large enough that everything is a neighbor

  raft::resources handle;
  auto x   = raft::make_device_matrix<float, int64_t>(handle, m, k);
  auto y   = raft::make_device_matrix<float, int64_t>(handle, n, k);
  auto adj = raft::make_device_matrix<bool, int64_t>(handle, m, n);
  auto vd  = raft::make_device_vector<int64_t, int64_t>(handle, m + 1);

  // fill x, y with zeros (every pair has distance 0 < eps)
  raft::matrix::fill(handle, x.view(), 0.0f);
  raft::matrix::fill(handle, y.view(), 0.0f);

  cuvs::neighbors::epsilon_neighborhood::compute(handle,
                                                 raft::make_const_mdspan(x.view()),
                                                 raft::make_const_mdspan(y.view()),
                                                 adj.view(),
                                                 vd.view(),
                                                 eps,
                                                 cuvs::distance::DistanceType::L2Unexpanded);

  // Verify: with distance=0 and huge eps, every entry in adj should be true
  // and vd[0] should equal n
  auto adj_expected = raft::make_device_matrix<bool, int64_t>(handle, m, n);
  raft::matrix::fill(handle, adj_expected.view(), true);
  auto stream = raft::resource::get_cuda_stream(handle);
  ASSERT_TRUE(cuvs::devArrMatch(
    adj_expected.data_handle(), adj.data_handle(), m * n, cuvs::Compare<bool>(), stream.get()));

  int64_t expected_vd0 = n;
  ASSERT_TRUE(
    cuvs::devArrMatch(&expected_vd0, vd.data_handle(), 1, cuvs::Compare<int64_t>(), stream.get()));
}

};  // namespace cuvs::neighbors::epsilon_neighborhood
