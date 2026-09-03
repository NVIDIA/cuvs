/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "spatial_data.h"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ball_cover.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <vector>

namespace cuvs::neighbors::ball_cover {
using namespace std;

template <typename value_idx, typename value_t>
RAFT_KERNEL count_discrepancies_kernel(value_idx* actual_idx,
                                       value_idx* expected_idx,
                                       value_t* actual,
                                       value_t* expected,
                                       uint32_t m,
                                       uint32_t n,
                                       uint32_t* out,
                                       float thres = 1e-3)
{
  uint32_t row = blockDim.x * blockIdx.x + threadIdx.x;

  int n_diffs = 0;
  if (row < m) {
    for (uint32_t i = 0; i < n; i++) {
      value_t d    = actual[row * n + i] - expected[row * n + i];
      bool matches = (fabsf(d) <= thres) || (actual_idx[row * n + i] == expected_idx[row * n + i] &&
                                             actual_idx[row * n + i] == row);

      if (!matches) {
        printf(
          "row=%ud, n=%ud, actual_dist=%f, actual_ind=%ld, expected_dist=%f, expected_ind=%ld\n",
          row,
          i,
          actual[row * n + i],
          actual_idx[row * n + i],
          expected[row * n + i],
          expected_idx[row * n + i]);
      }
      n_diffs += !matches;
      out[row] = n_diffs;
    }
  }
}

struct is_nonzero {
  __host__ __device__ bool operator()(uint32_t i) { return i > 0; }
};

template <typename value_idx, typename value_t>
uint32_t count_discrepancies(value_idx* actual_idx,
                             value_idx* expected_idx,
                             value_t* actual,
                             value_t* expected,
                             uint32_t m,
                             uint32_t n,
                             uint32_t* out,
                             cudaStream_t stream)
{
  uint32_t tpb = 256;
  count_discrepancies_kernel<<<raft::ceildiv(m, tpb), tpb, 0, stream>>>(
    actual_idx, expected_idx, actual, expected, m, n, out);

  // Copy results to host
  std::vector<uint32_t> h_out(m);
  raft::copy(h_out.data(), out, m, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  // Count non-zero elements on the host
  uint32_t count = 0;
  for (uint32_t i = 0; i < m; i++) {
    if (h_out[i] > 0) count++;
  }

  return count;
}

template <typename value_t>
void compute_bfknn(const raft::resources& handle,
                   const value_t* X1,
                   const value_t* X2,
                   uint32_t n_rows,
                   uint32_t n_query_rows,
                   uint32_t d,
                   uint32_t k,
                   const cuvs::distance::DistanceType metric,
                   value_t* dists,
                   int64_t* inds)
{
  raft::device_matrix_view<const value_t, int64_t> input_vec =
    raft::make_device_matrix_view(X1, n_rows, d);
  auto bf_index_params   = cuvs::neighbors::brute_force::index_params();
  bf_index_params.metric = metric;
  auto bfindex           = cuvs::neighbors::brute_force::build(handle, bf_index_params, input_vec);

  auto bf_search_params = cuvs::neighbors::brute_force::search_params();
  cuvs::neighbors::brute_force::search(
    handle,
    bf_search_params,
    bfindex,
    raft::make_device_matrix_view<const value_t, int64_t>(X2, n_query_rows, d),
    raft::make_device_matrix_view<int64_t, int64_t>(inds, n_query_rows, k),
    raft::make_device_matrix_view<value_t, int64_t>(dists, n_query_rows, k));
}

struct ToRadians {
  __device__ __host__ float operator()(float a) { return a * (CUDART_PI_F / 180.0); }
};

template <typename value_int = std::int64_t>
struct BallCoverInputs {
  value_int k;
  value_int n_rows;
  value_int n_cols;
  float weight;
  value_int n_query;
  cuvs::distance::DistanceType metric;
};

template <typename value_idx, typename value_t, typename value_int = std::int64_t>
class BallCoverKNNQueryTest : public ::testing::TestWithParam<BallCoverInputs<value_int>> {
 protected:
  void basicTest()
  {
    params = ::testing::TestWithParam<BallCoverInputs<value_int>>::GetParam();
    raft::resources handle;

    uint32_t k         = params.k;
    uint32_t n_centers = 25;
    float weight       = params.weight;
    auto metric        = params.metric;

    rmm::device_uvector<value_t> X(params.n_rows * params.n_cols,
                                   raft::resource::get_cuda_stream(handle));
    rmm::device_uvector<value_idx> Y(params.n_rows, raft::resource::get_cuda_stream(handle));

    // Make sure the train and query sets are completely disjoint
    rmm::device_uvector<value_t> X2(params.n_query * params.n_cols,
                                    raft::resource::get_cuda_stream(handle));
    rmm::device_uvector<value_idx> Y2(params.n_query, raft::resource::get_cuda_stream(handle));

    raft::random::make_blobs<value_t, value_idx>(X.data(),
                                                 Y.data(),
                                                 params.n_rows,
                                                 params.n_cols,
                                                 n_centers,
                                                 raft::resource::get_cuda_stream(handle).get());

    raft::random::make_blobs<value_t, value_idx>(X2.data(),
                                                 Y2.data(),
                                                 params.n_query,
                                                 params.n_cols,
                                                 n_centers,
                                                 raft::resource::get_cuda_stream(handle).get());

    rmm::device_uvector<value_idx> d_ref_I(params.n_query * k,
                                           raft::resource::get_cuda_stream(handle));
    rmm::device_uvector<value_t> d_ref_D(params.n_query * k,
                                         raft::resource::get_cuda_stream(handle));

    if (metric == cuvs::distance::DistanceType::Haversine) {
      raft::linalg::unaryOp(
        X.data(), X.data(), X.size(), ToRadians(), raft::resource::get_cuda_stream(handle).get());
      raft::linalg::unaryOp(X2.data(),
                            X2.data(),
                            X2.size(),
                            ToRadians(),
                            raft::resource::get_cuda_stream(handle).get());
    }

    compute_bfknn(handle,
                  X.data(),
                  X2.data(),
                  params.n_rows,
                  params.n_query,
                  params.n_cols,
                  k,
                  metric,
                  d_ref_D.data(),
                  d_ref_I.data());

    raft::resource::sync_stream(handle);

    // Allocate predicted arrays
    rmm::device_uvector<value_idx> d_pred_I(params.n_query * k,
                                            raft::resource::get_cuda_stream(handle));
    rmm::device_uvector<value_t> d_pred_D(params.n_query * k,
                                          raft::resource::get_cuda_stream(handle));

    auto X_view =
      raft::make_device_matrix_view<value_t, value_int>(X.data(), params.n_rows, params.n_cols);
    auto X2_view = raft::make_device_matrix_view<const value_t, value_int>(
      (const value_t*)X2.data(), params.n_query, params.n_cols);

    auto d_pred_I_view =
      raft::make_device_matrix_view<value_idx, value_int>(d_pred_I.data(), params.n_query, k);
    auto d_pred_D_view =
      raft::make_device_matrix_view<value_t, value_int>(d_pred_D.data(), params.n_query, k);

    cuvs::neighbors::ball_cover::index<value_idx, value_t> index(handle, X_view, metric);
    cuvs::neighbors::ball_cover::build(handle, index);
    cuvs::neighbors::ball_cover::knn_query(
      handle, index, X2_view, d_pred_I_view, d_pred_D_view, true);

    raft::resource::sync_stream(handle);
    // What we really want are for the distances to match exactly. The
    // indices may or may not match exactly, depending upon the ordering which
    // can be nondeterministic.

    rmm::device_uvector<uint32_t> discrepancies(params.n_query,
                                                raft::resource::get_cuda_stream(handle));
    thrust::fill(raft::resource::get_thrust_policy(handle),
                 discrepancies.data(),
                 discrepancies.data() + discrepancies.size(),
                 0);
    //
    int res = count_discrepancies(d_ref_I.data(),
                                  d_pred_I.data(),
                                  d_ref_D.data(),
                                  d_pred_D.data(),
                                  params.n_query,
                                  k,
                                  discrepancies.data(),
                                  raft::resource::get_cuda_stream(handle).get());

    ASSERT_TRUE(res == 0);
  }

  void SetUp() override {}

  void TearDown() override {}

 protected:
  uint32_t d = 2;
  BallCoverInputs<value_int> params;
};

template <typename value_idx, typename value_t, typename value_int = std::int64_t>
class BallCoverAllKNNTest : public ::testing::TestWithParam<BallCoverInputs<value_int>> {
 protected:
  void basicTest()
  {
    params = ::testing::TestWithParam<BallCoverInputs<value_int>>::GetParam();
    raft::resources handle;

    uint32_t k         = params.k;
    uint32_t n_centers = 25;
    float weight       = params.weight;
    auto metric        = params.metric;

    rmm::device_uvector<value_t> X(params.n_rows * params.n_cols,
                                   raft::resource::get_cuda_stream(handle));
    rmm::device_uvector<value_int> Y(params.n_rows, raft::resource::get_cuda_stream(handle));

    raft::random::make_blobs<value_t, value_idx>(X.data(),
                                                 Y.data(),
                                                 params.n_rows,
                                                 params.n_cols,
                                                 n_centers,
                                                 raft::resource::get_cuda_stream(handle).get());

    rmm::device_uvector<value_idx> d_ref_I(params.n_rows * k,
                                           raft::resource::get_cuda_stream(handle));
    rmm::device_uvector<value_t> d_ref_D(params.n_rows * k,
                                         raft::resource::get_cuda_stream(handle));

    auto X_view = raft::make_device_matrix_view<const value_t, value_int>(
      (const value_t*)X.data(), params.n_rows, params.n_cols);

    if (metric == cuvs::distance::DistanceType::Haversine) {
      raft::linalg::unaryOp(
        X.data(), X.data(), X.size(), ToRadians(), raft::resource::get_cuda_stream(handle).get());
    }

    compute_bfknn(handle,
                  X.data(),
                  X.data(),
                  params.n_rows,
                  params.n_rows,
                  params.n_cols,
                  k,
                  metric,
                  d_ref_D.data(),
                  d_ref_I.data());

    raft::resource::sync_stream(handle);

    // Allocate predicted arrays
    rmm::device_uvector<value_idx> d_pred_I(params.n_rows * k,
                                            raft::resource::get_cuda_stream(handle));
    rmm::device_uvector<value_t> d_pred_D(params.n_rows * k,
                                          raft::resource::get_cuda_stream(handle));

    auto d_pred_I_view =
      raft::make_device_matrix_view<value_idx, value_int>(d_pred_I.data(), params.n_rows, k);
    auto d_pred_D_view =
      raft::make_device_matrix_view<value_t, value_int>(d_pred_D.data(), params.n_rows, k);

    cuvs::neighbors::ball_cover::index<value_idx, value_t> index(handle, X_view, metric);

    cuvs::neighbors::ball_cover::all_knn_query(handle, index, d_pred_I_view, d_pred_D_view, true);

    raft::resource::sync_stream(handle);
    // What we really want are for the distances to match exactly. The
    // indices may or may not match exactly, depending upon the ordering which
    // can be nondeterministic.

    rmm::device_uvector<uint32_t> discrepancies(params.n_rows,
                                                raft::resource::get_cuda_stream(handle));
    thrust::fill(raft::resource::get_thrust_policy(handle),
                 discrepancies.data(),
                 discrepancies.data() + discrepancies.size(),
                 0);
    //
    uint32_t res = count_discrepancies(d_ref_I.data(),
                                       d_pred_I.data(),
                                       d_ref_D.data(),
                                       d_pred_D.data(),
                                       params.n_rows,
                                       k,
                                       discrepancies.data(),
                                       raft::resource::get_cuda_stream(handle).get());

    // TODO: There seem to be discrepancies here only when
    // the entire test suite is executed.
    // Ref: https://github.com/rapidsai/raft/issues/
    // 1-5 mismatches in 8000 samples is 0.0125% - 0.0625%
    ASSERT_TRUE(res <= 5);
  }

  void SetUp() override {}

  void TearDown() override {}

 protected:
  BallCoverInputs<value_int> params;
};

typedef BallCoverAllKNNTest<int64_t, float> BallCoverAllKNNTestF;
typedef BallCoverKNNQueryTest<int64_t, float> BallCoverKNNQueryTestF;

const std::vector<BallCoverInputs<std::int64_t>> ballcover_inputs = {
  {11, 5000, 2, 1.0, 10000, cuvs::distance::DistanceType::Haversine},
  {25, 10000, 2, 1.0, 5000, cuvs::distance::DistanceType::Haversine},
  {2, 10000, 2, 1.0, 5000, cuvs::distance::DistanceType::L2SqrtUnexpanded},
  {2, 5000, 2, 1.0, 10000, cuvs::distance::DistanceType::Haversine},
  {11, 10000, 2, 1.0, 5000, cuvs::distance::DistanceType::L2SqrtUnexpanded},
  {25, 5000, 2, 1.0, 10000, cuvs::distance::DistanceType::L2SqrtUnexpanded},
  {5, 8000, 3, 1.0, 10000, cuvs::distance::DistanceType::L2SqrtUnexpanded},
  {11, 6000, 3, 1.0, 10000, cuvs::distance::DistanceType::L2SqrtUnexpanded},
  {25, 10000, 3, 1.0, 5000, cuvs::distance::DistanceType::L2SqrtUnexpanded}};

INSTANTIATE_TEST_CASE_P(BallCoverAllKNNTest,
                        BallCoverAllKNNTestF,
                        ::testing::ValuesIn(ballcover_inputs));
INSTANTIATE_TEST_CASE_P(BallCoverKNNQueryTest,
                        BallCoverKNNQueryTestF,
                        ::testing::ValuesIn(ballcover_inputs));

TEST_P(BallCoverAllKNNTestF, Fit) { basicTest(); }
TEST_P(BallCoverKNNQueryTestF, Fit) { basicTest(); }

}  // namespace cuvs::neighbors::ball_cover
