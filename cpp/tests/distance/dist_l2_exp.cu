/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_base.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cmath>
#include <vector>

namespace cuvs {
namespace distance {

template <typename DataType, typename OutputType = DataType>
class DistanceEucExpTest
  : public DistanceTest<cuvs::distance::DistanceType::L2Expanded, DataType, OutputType> {};

template <typename DataType, typename OutputType = DataType>
class DistanceEucExpTestXequalY
  : public DistanceTestSameBuffer<cuvs::distance::DistanceType::L2Expanded, DataType, OutputType> {
};

const std::vector<DistanceInputs<float>> inputsf = {
  {0.001f, 128, (65536 + 128) * 128, 8, true, 1234ULL},
  {0.001f, 2048, 4096, 128, true, 1234ULL},
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.003f, 1021, 1021, 1021, true, 1234ULL},
  {0.001f, (65536 + 128) * 128, 128, 8, false, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
  {0.003f, 1021, 1021, 1021, false, 1234ULL},
};

const std::vector<DistanceInputs<float>> inputsXeqYf = {
  {0.01f, 2048, 4096, 128, true, 1234ULL},
  {0.01f, 1024, 1024, 32, true, 1234ULL},
  {0.01f, 1024, 32, 1024, true, 1234ULL},
  {0.01f, 32, 1024, 1024, true, 1234ULL},
  {0.03f, 1024, 1024, 1024, true, 1234ULL},
  {0.03f, 1021, 1021, 1021, true, 1234ULL},
  {0.01f, 1024, 1024, 32, false, 1234ULL},
  {0.01f, 1024, 32, 1024, false, 1234ULL},
  {0.01f, 32, 1024, 1024, false, 1234ULL},
  {0.03f, 1024, 1024, 1024, false, 1234ULL},
  {0.03f, 1021, 1021, 1021, false, 1234ULL},
};

const std::vector<DistanceInputs<half, float>> inputsh = {
  {0.001f, 128, (65536 + 128) * 128, 8, true, 1234ULL},
  {0.001f, 2048, 4096, 128, true, 1234ULL},
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.003f, 1021, 1021, 1021, true, 1234ULL},
  {0.001f, (65536 + 128) * 128, 128, 8, false, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
  {0.003f, 1021, 1021, 1021, false, 1234ULL},
};

const std::vector<DistanceInputs<half, float>> inputsXeqYh = {
  {0.01f, 2048, 4096, 128, true, 1234ULL},
  {0.01f, 1024, 1024, 32, true, 1234ULL},
  {0.01f, 1024, 32, 1024, true, 1234ULL},
  {0.01f, 32, 1024, 1024, true, 1234ULL},
  {0.03f, 1024, 1024, 1024, true, 1234ULL},
  {0.03f, 1021, 1021, 1021, true, 1234ULL},
  {0.01f, 1024, 1024, 32, false, 1234ULL},
  {0.01f, 1024, 32, 1024, false, 1234ULL},
  {0.01f, 32, 1024, 1024, false, 1234ULL},
  {0.03f, 1024, 1024, 1024, false, 1234ULL},
  {0.03f, 1021, 1021, 1021, false, 1234ULL},
};

typedef DistanceEucExpTest<float> DistanceEucExpTestF;
TEST_P(DistanceEucExpTestF, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceEucExpTestF, ::testing::ValuesIn(inputsf));

typedef DistanceEucExpTest<half, float> DistanceEucExpTestH;
TEST_P(DistanceEucExpTestH, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceEucExpTestH, ::testing::ValuesIn(inputsh));

typedef DistanceEucExpTestXequalY<float> DistanceEucExpTestXequalYF;
TEST_P(DistanceEucExpTestXequalYF, Result)
{
  int m = params.m;
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[0].data(),
                                dist[0].data(),
                                m,
                                m,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[1].data(),
                                dist[1].data(),
                                m / 2,
                                m,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests,
                        DistanceEucExpTestXequalYF,
                        ::testing::ValuesIn(inputsXeqYf));

typedef DistanceEucExpTestXequalY<half, float> DistanceEucExpTestXequalYH;
TEST_P(DistanceEucExpTestXequalYH, Result)
{
  int m = params.m;
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[0].data(),
                                dist[0].data(),
                                m,
                                m,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[1].data(),
                                dist[1].data(),
                                m / 2,
                                m,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests,
                        DistanceEucExpTestXequalYH,
                        ::testing::ValuesIn(inputsXeqYh));

const std::vector<DistanceInputs<double>> inputsd = {
  {0.001, 1024, 1024, 32, true, 1234ULL},
  {0.001, 1024, 32, 1024, true, 1234ULL},
  {0.001, 32, 1024, 1024, true, 1234ULL},
  {0.003, 1024, 1024, 1024, true, 1234ULL},
  {0.001, 1024, 1024, 32, false, 1234ULL},
  {0.001, 1024, 32, 1024, false, 1234ULL},
  {0.001, 32, 1024, 1024, false, 1234ULL},
  {0.003, 1024, 1024, 1024, false, 1234ULL},
};
typedef DistanceEucExpTest<double> DistanceEucExpTestD;
TEST_P(DistanceEucExpTestD, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceEucExpTestD, ::testing::ValuesIn(inputsd));

class BigMatrixEucExp : public BigMatrixDistanceTest<cuvs::distance::DistanceType::L2Expanded> {};
TEST_F(BigMatrixEucExp, Result) {}

// Distinct points with equal norms must keep their (small) distances. The clamp of self-distance
// round-off to zero used to apply an absolute tolerance, which collapsed such points when the data
// has a small magnitude (e.g. a spectral embedding).
template <typename T>
void test_small_scale_equal_norms(T scale)
{
  raft::resources handle;
  auto stream         = raft::resource::get_cuda_stream(handle);
  constexpr int64_t n = 8;
  std::vector<T> h_x(n * n, T{0});
  for (int64_t i = 0; i < n; i++) {
    h_x[i * n + i] = scale;
  }
  auto x = raft::make_device_matrix<T, int64_t>(handle, n, n);
  auto d = raft::make_device_matrix<T, int64_t>(handle, n, n);
  raft::update_device(x.data_handle(), h_x.data(), h_x.size(), stream);

  for (auto metric : {DistanceType::L2Expanded, DistanceType::L2SqrtExpanded}) {
    cuvs::distance::pairwise_distance(handle,
                                      raft::make_const_mdspan(x.view()),
                                      raft::make_const_mdspan(x.view()),
                                      d.view(),
                                      metric);
    std::vector<T> h_d(n * n);
    raft::update_host(h_d.data(), d.data_handle(), h_d.size(), stream);
    raft::resource::sync_stream(handle, stream);

    const T expected =
      metric == DistanceType::L2Expanded ? T(2) * scale * scale : std::sqrt(T(2)) * scale;
    for (int64_t i = 0; i < n; i++) {
      for (int64_t j = 0; j < n; j++) {
        EXPECT_NEAR(h_d[i * n + j], i == j ? T(0) : expected, T(1e-3) * expected)
          << "metric=" << static_cast<int>(metric) << " i=" << i << " j=" << j;
      }
    }
  }
}

TEST(DistanceEucExpSmallScale, EqualNormsF) { test_small_scale_equal_norms<float>(1e-2f); }
TEST(DistanceEucExpSmallScale, EqualNormsD) { test_small_scale_equal_norms<double>(1e-5); }
}  // end namespace distance
}  // namespace cuvs
