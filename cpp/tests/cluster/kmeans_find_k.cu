/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.h"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_span.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <vector>

namespace cuvs::cluster::kmeans {

template <typename T>
struct KmeansFindKInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  bool weighted;
};

template <typename T>
class KmeansFindKTest : public ::testing::TestWithParam<KmeansFindKInputs<T>> {
 protected:
  KmeansFindKTest()
    : stream(raft::resource::get_cuda_stream(handle).get()), best_k(raft::make_host_scalar<int>(0))
  {
  }

  void basicTest()
  {
    testparams = ::testing::TestWithParam<KmeansFindKInputs<T>>::GetParam();

    int n_samples  = testparams.n_row;
    int n_features = testparams.n_col;
    int n_clusters = testparams.n_clusters;

    auto X      = raft::make_device_matrix<T, int>(handle, n_samples, n_features);
    auto labels = raft::make_device_vector<int, int>(handle, n_samples);

    raft::random::make_blobs<T, int>(X.data_handle(),
                                     labels.data_handle(),
                                     n_samples,
                                     n_features,
                                     n_clusters,
                                     stream,
                                     true,
                                     nullptr,
                                     nullptr,
                                     T(.001),
                                     false,
                                     (T)-10.0f,
                                     (T)10.0f,
                                     (uint64_t)1234);

    auto inertia = raft::make_host_scalar<T>(0);
    auto n_iter  = raft::make_host_scalar<int>(0);

    auto X_view =
      raft::make_device_matrix_view<const T, int>(X.data_handle(), X.extent(0), X.extent(1));

    cuvs::cluster::kmeans::helpers::find_k(
      handle, X_view, best_k.view(), inertia.view(), n_iter.view(), n_clusters);

    raft::resource::sync_stream(handle, stream);
  }

  void SetUp() override { basicTest(); }

 protected:
  raft::resources handle;
  cudaStream_t stream;
  KmeansFindKInputs<T> testparams;
  raft::host_scalar<int> best_k;
};

const std::vector<KmeansFindKInputs<float>> inputsf2 = {{1000, 32, 8, 0.001f, true},
                                                        {1000, 32, 8, 0.001f, false},
                                                        {1000, 100, 20, 0.001f, true},
                                                        {1000, 100, 20, 0.001f, false},
                                                        {10000, 32, 10, 0.001f, true},
                                                        {10000, 32, 10, 0.001f, false},
                                                        {10000, 100, 50, 0.001f, true},
                                                        {10000, 100, 50, 0.001f, false},
                                                        {10000, 500, 100, 0.001f, true},
                                                        {10000, 500, 100, 0.001f, false}};

typedef KmeansFindKTest<float> KmeansFindKTestF;
TEST_P(KmeansFindKTestF, Result)
{
  if (best_k.view()[0] != testparams.n_clusters) {
    std::cout << best_k.view()[0] << " " << testparams.n_clusters << std::endl;
  }
  ASSERT_TRUE(best_k.view()[0] == testparams.n_clusters);
}

INSTANTIATE_TEST_CASE_P(KmeansFindKTests, KmeansFindKTestF, ::testing::ValuesIn(inputsf2));

}  // namespace cuvs::cluster::kmeans
