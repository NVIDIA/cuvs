/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/kernel_launch.hpp>
#include <raft/util/reduction.cuh>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <ostream>
#include <vector>

namespace cuvs::cluster::kmeans {

enum class Distribution { Uniform, Blobs };

struct ClusterCostInputs {
  Distribution distribution;
  int64_t n_cols;
  int64_t n_clusters;
  bool row_aligned = false;  // 16-byte alignment of row address
};

inline std::ostream& operator<<(std::ostream& os, const ClusterCostInputs& p)
{
  return os << (p.distribution == Distribution::Uniform ? "uniform" : "blobs") << " K=" << p.n_cols
            << " N=" << p.n_clusters << " row_aligned=" << p.row_aligned;
}

__global__ void reference_distances(const float* x,
                                    const float* c,
                                    int64_t M,
                                    int64_t N,
                                    int64_t K,
                                    float* fp32_distance,
                                    double* fp64_distance)
{
  const int64_t row =
    static_cast<int64_t>(blockIdx.x) * (blockDim.x / raft::WarpSize) + threadIdx.x / raft::WarpSize;
  if (row >= M) { return; }

  // Each warps owns 1 row and 32 centroids.
  const float* xr = x + row * K;
  float x_norm    = 0;
  for (int64_t j = 0; j < K; j++) {
    x_norm = fmaf(xr[j], xr[j], x_norm);
  }
  float best_fp32  = std::numeric_limits<float>::max();
  double best_fp64 = std::numeric_limits<double>::max();
  // best centroids per lane
  for (int64_t i = raft::laneId(); i < N; i += raft::WarpSize) {
    const float* cr = c + i * K;
    float dot = 0, c_norm = 0;
    double direct = 0;
    for (int64_t j = 0; j < K; j++) {
      dot               = fmaf(xr[j], cr[j], dot);
      c_norm            = fmaf(cr[j], cr[j], c_norm);
      const double diff = double(xr[j]) - double(cr[j]);
      direct            = fma(diff, diff, direct);
    }
    // FP32 baseline uses expanded formula (|x|^2 + |c|^2 - 2 x.c)
    best_fp32 = fminf(best_fp32, fmaxf(x_norm + c_norm - 2.0f * dot, 0.0f));
    // groundtruth FP64 uses more-precise canonical formula (x - c)^2
    best_fp64 = fmin(best_fp64, direct);
  }
  // best centroid per warp
  best_fp32 = raft::warpReduce(best_fp32, raft::min_op{});
  best_fp64 = raft::warpReduce(best_fp64, raft::min_op{});
  if (raft::laneId() == 0) {
    fp32_distance[row] = best_fp32;
    fp64_distance[row] = best_fp64;
  }
}

/** Pairwise sum in a fixed order (the same addition tree on every GPU), accumulated in T. */
template <typename T>
T pairwise_sum(const T* v, std::size_t n)
{
  if (n <= 8) {
    T total{0};
    for (std::size_t i = 0; i < n; i++) {
      total += v[i];
    }
    return total;
  }
  return pairwise_sum(v, n / 2) + pairwise_sum(v + n / 2, n - n / 2);
}

/** Sum of a device vector, on the host in a fixed order for determinism across GPUs. */
template <typename T>
double sum(raft::resources const& handle, const raft::device_vector<T, int64_t>& v)
{
  std::vector<T> host(v.size());
  raft::copy(host.data(), v.data_handle(), v.size(), raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);
  return double(pairwise_sum(host.data(), host.size()));
}

/** Relative errors of the cluster_cost API and of the native FP32 baseline against FP64. */
struct CostErrors {
  double api;
  double fp32;
};

CostErrors cost_errors(const ClusterCostInputs& p, uint64_t seed)
{
  raft::device_resources handle;
  const int64_t M = 10000, K = p.n_cols, N = p.n_clusters;

  // Initialize X and C, either uniformly or as blobs.
  auto x_buf = raft::make_device_vector<float, int64_t>(handle, M * K + 1);
  auto x     = raft::make_device_matrix_view<float, int64_t>(
    x_buf.data_handle() + (p.row_aligned ? 0 : 1), M, K);
  auto c = raft::make_device_matrix<float, int64_t>(handle, N, K);
  raft::random::RngState rng(seed);
  // Uniform C
  raft::random::uniform(handle, rng, c.data_handle(), N * K, -10.0f, 10.0f);
  if (p.distribution == Distribution::Uniform) {
    // Uniform X
    raft::random::uniform(handle, rng, x.data_handle(), M * K, -10.0f, 10.0f);
  } else {
    // Blobs X
    auto labels = raft::make_device_vector<int64_t, int64_t>(handle, M);
    raft::random::make_blobs<float, int64_t>(handle,
                                             x,
                                             labels.view(),
                                             N,
                                             std::make_optional(c.view()),
                                             std::nullopt,
                                             1.0f,
                                             true,
                                             -10.0f,
                                             10.0f,
                                             seed);
  }

  // Ground truth and the native FP32 baseline
  auto distance         = raft::make_device_vector<float, int64_t>(handle, M);
  auto fp64_distance    = raft::make_device_vector<double, int64_t>(handle, M);
  constexpr int threads = 256;
  raft::launch_kernel(handle,
                      static_cast<unsigned>(raft::ceildiv<int64_t>(M, threads / raft::WarpSize)),
                      threads,
                      reference_distances,
                      x.data_handle(),
                      c.data_handle(),
                      M,
                      N,
                      K,
                      distance.data_handle(),
                      fp64_distance.data_handle());
  const double truth = sum(handle, fp64_distance);
  EXPECT_GT(truth, 0.0);
  const double fp32_cost = sum(handle, distance);
  float api_cost         = 0;
  cluster_cost(handle,
               raft::make_const_mdspan(x),
               raft::make_const_mdspan(c.view()),
               raft::make_host_scalar_view(&api_cost));

  return {std::abs(api_cost - truth) / truth, std::abs(fp32_cost - truth) / truth};
}

class ClusterCostTest : public ::testing::TestWithParam<ClusterCostInputs> {
 protected:
  void run()
  {
    const auto p      = GetParam();
    const auto errors = cost_errors(p, 42);
    // Tolerance dictates the maximum cluster cost error against the native FP32 baseline.
    const double tolerance = p.distribution == Distribution::Uniform ? 6e-7 : 3e-7;
    EXPECT_LE(errors.api, errors.fp32 + tolerance)
      << "cluster_cost: " << errors.api << " vs native FP32 " << errors.fp32;
  }
};

TEST_P(ClusterCostTest, MaxPrecisionLossAgainstNativeFP32) { run(); }

const std::vector<ClusterCostInputs> cluster_cost_inputs = [] {
  std::vector<ClusterCostInputs> inputs;
  for (auto distribution : {Distribution::Uniform, Distribution::Blobs}) {
    for (int64_t K : {4, 16, 64, 256}) {
      for (int64_t N : {100, 1000, 10000}) {
        inputs.push_back({distribution, K, N, true});
      }
    }
  }
  // Misaligned tests: falls back to non-cutile backends. Inefficient GEMM, so keep test small.
  inputs.push_back({Distribution::Blobs, 4, 1000, false});
  inputs.push_back({Distribution::Blobs, 64, 1000, false});
  return inputs;
}();

INSTANTIATE_TEST_SUITE_P(ClusterCostTests,
                         ClusterCostTest,
                         ::testing::ValuesIn(cluster_cost_inputs));

/** Errors of cluster_cost and of the native FP32 baseline over 5 seeds and a grid of K and N. */
std::vector<CostErrors> errors_over_seeds(Distribution distribution)
{
  std::vector<CostErrors> errors;
  for (uint64_t seed = 1; seed <= 5; seed++) {
    for (int64_t K : {4, 16, 64, 256}) {
      for (int64_t N : {100, 1000}) {
        errors.push_back(cost_errors({distribution, K, N, true}, seed));
      }
    }
  }
  return errors;
}

TEST(ClusterCostBlobs, WinsMoreOftenThanNativeFp32)
{
  int wins = 0, losses = 0;
  for (const auto& errors : errors_over_seeds(Distribution::Blobs)) {
    wins += errors.api < errors.fp32;
    losses += errors.api > errors.fp32;
  }
  EXPECT_GE(wins, 3 * losses) << wins << " wins, " << losses << " losses";
}

}  // namespace cuvs::cluster::kmeans
