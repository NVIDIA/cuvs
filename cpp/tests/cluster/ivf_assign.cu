/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include "../../src/cluster/detail/ivf_assign.hpp"
#include "../../src/cluster/kmeans_balanced.cuh"
#include <cuvs/distance/distance.hpp>

#include <cuda_fp16.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/map.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace cuvs::cluster::kmeans::detail::ivf_assign {

using cuvs::distance::DistanceType;

struct inputs {
  int64_t n_centroids;
  int64_t n_queries;
  int64_t dim;
  uint32_t n_lists;  // 0: default
  DistanceType metric;
};

inline auto operator<<(std::ostream& os, const inputs& p) -> std::ostream&
{
  return os << "{K=" << p.n_centroids << " nq=" << p.n_queries << " dim=" << p.dim
            << " n_lists=" << p.n_lists << " metric=" << static_cast<int>(p.metric) << "}";
}

// The assignment multiplies in TF32 (10-bit mantissa), so two centroids whose scores differ by less
// than the TF32 rounding error of the dot product may be swapped. Scores are compared with this
// tolerance instead of requiring identical labels.
constexpr double kTf32Tol = 1.0 / 512;

class IvfAssignTest : public ::testing::TestWithParam<inputs> {
 protected:
  void SetUp() override
  {
    const inputs p = GetParam();
    stream_        = raft::resource::get_cuda_stream(res_).get();
    auto cents     = raft::make_device_matrix<float, int64_t>(res_, p.n_centroids, p.dim);
    auto queries   = raft::make_device_matrix<float, int64_t>(res_, p.n_queries, p.dim);
    raft::random::RngState rng(1234ULL);
    raft::random::normal(res_, rng, cents.data_handle(), p.n_centroids * p.dim, 0.f, 1.f);
    raft::random::normal(res_, rng, queries.data_handle(), p.n_queries * p.dim, 0.f, 1.f);
    h_cents_.resize(p.n_centroids * p.dim);
    h_queries_.resize(p.n_queries * p.dim);
    raft::update_host(h_cents_.data(), cents.data_handle(), h_cents_.size(), stream_);
    raft::update_host(h_queries_.data(), queries.data_handle(), h_queries_.size(), stream_);
    raft::resource::sync_stream(res_);

    // The rows of the centroid matrix are used as is (not normalized) for every metric: the index
    // normalizes what it needs.
    for (int64_t c = 0; c < p.n_centroids; c++) {
      double ny      = 0;
      const float* y = h_cents_.data() + c * p.dim;
      for (int64_t i = 0; i < p.dim; i++)
        ny += double(y[i]) * y[i];
      cmax_ = std::max(cmax_, std::sqrt(ny));
    }
    d_cents_   = std::move(cents);
    d_queries_ = std::move(queries);
    index_params ip;
    ip.metric  = p.metric;
    ip.n_lists = p.n_lists;
    idx_ = std::make_unique<index>(build(res_, ip, raft::make_const_mdspan(d_cents_->view())));
  }

  auto run(uint32_t n_probes, assign_algo algo) -> std::vector<uint32_t>
  {
    const inputs p = GetParam();
    auto labels    = raft::make_device_vector<uint32_t, int64_t>(res_, p.n_queries);
    search_params sp;
    sp.n_probes = n_probes;
    sp.algo     = algo;
    assign(res_, sp, *idx_, raft::make_const_mdspan(d_queries_->view()), labels.view());
    std::vector<uint32_t> out(p.n_queries);
    raft::update_host(out.data(), labels.data_handle(), out.size(), stream_);
    raft::resource::sync_stream(res_);
    return out;
  }

  // Score of centroid c for query q in the metric's "larger is nearer" form, in double precision.
  auto score(int64_t q, int64_t c) const -> double
  {
    const inputs p = GetParam();
    const float* x = h_queries_.data() + q * p.dim;
    const float* y = h_cents_.data() + c * p.dim;
    double dot = 0, ny = 0;
    for (int64_t i = 0; i < p.dim; i++) {
      dot += double(x[i]) * y[i];
      ny += double(y[i]) * y[i];
    }
    switch (p.metric) {
      case DistanceType::CosineExpanded: return dot / std::sqrt(ny);
      case DistanceType::InnerProduct: return dot;
      default: return dot - 0.5 * ny;  // L2: argmin |x-y|^2 = argmax x.y - |y|^2/2
    }
  }

  // Tolerance for comparing two scores of query q.
  auto tol(int64_t q) const -> double
  {
    const inputs p = GetParam();
    double nx      = 0;
    const float* x = h_queries_.data() + q * p.dim;
    for (int64_t i = 0; i < p.dim; i++)
      nx += double(x[i]) * x[i];
    nx = std::sqrt(nx);
    // |error of the TF32 dot product| <= 2^-9 |x||y|; the cosine score divides |y| out
    return kTf32Tol * nx * (p.metric == DistanceType::CosineExpanded ? 1.0 : cmax_) + 1e-6;
  }

  // The best score over all centroids.
  auto best_score(int64_t q) const -> double
  {
    double best = -INFINITY;
    for (int64_t c = 0; c < GetParam().n_centroids; c++)
      best = std::max(best, score(q, c));
    return best;
  }

  auto fused_supported() const -> bool
  {
    int dev = 0, major = 0;
    RAFT_CUDA_TRY(cudaGetDevice(&dev));
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    return major >= 8;  // every tested dim is a multiple of 4; the k-tail is zero-filled
  }

  raft::device_resources res_;
  cudaStream_t stream_ = nullptr;
  std::vector<float> h_cents_, h_queries_;
  double cmax_ = 0;  // largest centroid norm
  std::optional<raft::device_matrix<float, int64_t>> d_cents_, d_queries_;
  std::unique_ptr<index> idx_;
};

// Probing every list is exhaustive: every label must be a (TF32-tolerance) best centroid.
TEST_P(IvfAssignTest, ExhaustiveIsNearest)
{
  const inputs p                 = GetParam();
  std::vector<assign_algo> algos = {assign_algo::AUTO, assign_algo::GEMM_LOOP};
  if (fused_supported()) algos.push_back(assign_algo::FUSED);
  for (auto algo : algos) {
    auto labels = run(idx_->n_lists, algo);
    for (int64_t q = 0; q < p.n_queries; q++) {
      ASSERT_LT(labels[q], p.n_centroids) << "query " << q;
      EXPECT_GE(score(q, labels[q]), best_score(q) - tol(q))
        << "algo " << static_cast<int>(algo) << " query " << q;
    }
  }
}

// The fused kernel and the GEMM loop compute the same TF32 scores.
TEST_P(IvfAssignTest, FusedMatchesLoop)
{
  if (!fused_supported()) GTEST_SKIP() << "fused kernel not supported on this device";
  const inputs p = GetParam();
  for (uint32_t probes : {1u, 3u, 0u, idx_->n_lists}) {
    auto a = run(probes, assign_algo::FUSED);
    auto b = run(probes, assign_algo::GEMM_LOOP);
    for (int64_t q = 0; q < p.n_queries; q++) {
      if (a[q] != b[q]) {
        // only a near-tie may differ
        EXPECT_NEAR(score(q, a[q]), score(q, b[q]), tol(q))
          << "n_probes " << probes << " query " << q;
      }
    }
  }
}

// The probed lists are nested, so more probes never give a worse centroid.
TEST_P(IvfAssignTest, MoreProbesNeverWorse)
{
  const inputs p = GetParam();
  auto one       = run(1, assign_algo::AUTO);
  auto few       = run(std::min<uint32_t>(3, idx_->n_lists), assign_algo::AUTO);
  auto all       = run(idx_->n_lists, assign_algo::AUTO);
  for (int64_t q = 0; q < p.n_queries; q++) {
    const double t = tol(q);
    EXPECT_GE(score(q, few[q]), score(q, one[q]) - t) << "query " << q;
    EXPECT_GE(score(q, all[q]), score(q, few[q]) - t) << "query " << q;
  }
}

// n_probes = 0 selects 16, at most n_lists.
TEST_P(IvfAssignTest, DefaultProbes)
{
  const uint32_t m        = idx_->n_lists;
  const uint32_t expected = std::min<uint32_t>(m, 16);
  EXPECT_EQ(run(0, assign_algo::GEMM_LOOP), run(expected, assign_algo::GEMM_LOOP));
}

// More probes than lists is clamped, i.e. exhaustive.
TEST_P(IvfAssignTest, ProbesAboveListsAreClamped)
{
  EXPECT_EQ(run(idx_->n_lists + 100, assign_algo::GEMM_LOOP),
            run(idx_->n_lists, assign_algo::GEMM_LOOP));
}

// Requiring the fused kernel on an input it cannot handle is an error, not a silent fallback.
TEST_P(IvfAssignTest, FusedRequiredFailsWhenUnsupported)
{
  if (fused_supported()) GTEST_SKIP() << "fused kernel is supported here";
  EXPECT_THROW(run(0, assign_algo::FUSED), raft::exception);
  EXPECT_NO_THROW(run(0, assign_algo::AUTO));  // the fallback still works
}

// Both the fused kernel (cp.async) and the gather of the GEMM loop load query rows 16 bytes at a
// time, so queries that are not 16-byte aligned are rejected for every algorithm.
TEST_P(IvfAssignTest, MisalignedQueriesAreRejected)
{
  const inputs p = GetParam();
  auto buffer    = raft::make_device_vector<float, int64_t>(res_, p.n_queries * p.dim + 4);
  raft::copy(buffer.data_handle() + 1, d_queries_->data_handle(), p.n_queries * p.dim, stream_);
  auto queries = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    buffer.data_handle() + 1, p.n_queries, p.dim);
  auto labels = raft::make_device_vector<uint32_t, int64_t>(res_, p.n_queries);
  for (auto algo : {assign_algo::AUTO, assign_algo::FUSED, assign_algo::GEMM_LOOP}) {
    search_params sp;
    sp.algo = algo;
    EXPECT_THROW(assign(res_, sp, *idx_, queries, labels.view()), raft::exception);
  }
}

// kmeans_balanced::predict with a IVF index runs the same assignment as `assign`.
TEST_P(IvfAssignTest, PredictOverloadMatchesAssign)
{
  const inputs p = GetParam();
  search_params sp;
  sp.n_probes   = 0;
  auto expected = run(0, assign_algo::AUTO);
  auto labels   = raft::make_device_vector<uint32_t, int64_t>(res_, p.n_queries);
  cuvs::cluster::kmeans_balanced::predict(
    res_, *idx_, sp, raft::make_const_mdspan(d_queries_->view()), labels.view());
  std::vector<uint32_t> out(p.n_queries);
  raft::update_host(out.data(), labels.data_handle(), out.size(), stream_);
  raft::resource::sync_stream(res_);
  EXPECT_EQ(out, expected);
}

// Non-float data is mapped to float before the assignment: the labels equal those of `assign` on
// the half-rounded queries.
TEST_P(IvfAssignTest, PredictOverloadHalfData)
{
  const inputs p  = GetParam();
  auto q_half     = raft::make_device_matrix<half, int64_t>(res_, p.n_queries, p.dim);
  auto q_round    = raft::make_device_matrix<float, int64_t>(res_, p.n_queries, p.dim);
  const int64_t n = p.n_queries * p.dim;
  raft::linalg::map(
    res_,
    raft::make_device_vector_view<const float, int64_t>(d_queries_->data_handle(), n),
    raft::make_device_vector_view<half, int64_t>(q_half.data_handle(), n),
    raft::cast_op<half>());
  raft::linalg::map(res_,
                    raft::make_device_vector_view<const half, int64_t>(q_half.data_handle(), n),
                    raft::make_device_vector_view<float, int64_t>(q_round.data_handle(), n),
                    raft::cast_op<float>());
  search_params sp;
  auto expected = raft::make_device_vector<uint32_t, int64_t>(res_, p.n_queries);
  assign(res_, sp, *idx_, raft::make_const_mdspan(q_round.view()), expected.view());
  auto labels = raft::make_device_vector<uint32_t, int64_t>(res_, p.n_queries);
  cuvs::cluster::kmeans_balanced::predict(
    res_, *idx_, sp, raft::make_const_mdspan(q_half.view()), labels.view(), raft::cast_op<float>());
  std::vector<uint32_t> a(p.n_queries), b(p.n_queries);
  raft::update_host(a.data(), expected.data_handle(), a.size(), stream_);
  raft::update_host(b.data(), labels.data_handle(), b.size(), stream_);
  raft::resource::sync_stream(res_);
  EXPECT_EQ(a, b);
}

// A dataset whose dim differs from the index's is rejected.
TEST_P(IvfAssignTest, PredictOverloadDimMismatchThrows)
{
  const inputs p = GetParam();
  auto q         = raft::make_device_matrix<float, int64_t>(res_, 4, p.dim + 4);
  auto labels    = raft::make_device_vector<uint32_t, int64_t>(res_, 4);
  search_params sp;
  EXPECT_THROW(cuvs::cluster::kmeans_balanced::predict(
                 res_, *idx_, sp, raft::make_const_mdspan(q.view()), labels.view()),
               raft::exception);
}

const std::vector<inputs> kInputs = {
  // dim % 32 == 0: the k-loop has no tail
  {2000, 300, 64, 0, DistanceType::CosineExpanded},
  {2000, 300, 64, 0, DistanceType::L2Expanded},
  {2000, 300, 64, 0, DistanceType::InnerProduct},
  {4096, 500, 128, 64, DistanceType::CosineExpanded},
  {1000, 257, 96, 7, DistanceType::L2Expanded},      // n_lists not a divisor, odd batch
  {3000, 200, 32, 3, DistanceType::CosineExpanded},  // few, large lists
  {500, 3, 64, 0, DistanceType::L2Expanded},         // tiny batch
  {500, 1, 64, 0, DistanceType::CosineExpanded},     // batch of one
  {64, 100, 64, 64, DistanceType::L2Expanded},       // one centroid per list
  {300, 100, 64, 1, DistanceType::CosineExpanded},   // a single list
  // dim % 32 != 0: the last k-step of the fused kernel is zero-filled
  {2000, 300, 100, 0, DistanceType::CosineExpanded},
  {1500, 200, 36, 0, DistanceType::L2Expanded},
  {900, 150, 68, 0, DistanceType::L2Expanded},
  {900, 150, 124, 5, DistanceType::CosineExpanded},
  {600, 100, 20, 0, DistanceType::InnerProduct},  // dim < 32: a single partial k-step
  {600, 100, 4, 0, DistanceType::L2Expanded},
};

INSTANTIATE_TEST_CASE_P(IvfAssign, IvfAssignTest, ::testing::ValuesIn(kInputs));

}  // namespace cuvs::cluster::kmeans::detail::ivf_assign
