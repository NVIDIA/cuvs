/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <cuvs/cluster/gmm.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace cuvs::cluster::gmm {

template <typename T>
struct GMMInputs {
  int n_row;
  int n_col;
  int n_components;
  covariance_type cov_type;
};

// Number of elements in a covariance-typed buffer for the given type.
inline int64_t cov_len(covariance_type ct, int d, int K)
{
  switch (ct) {
    case covariance_type::FULL: return (int64_t)K * d * d;
    case covariance_type::TIED: return (int64_t)d * d;
    case covariance_type::DIAG: return (int64_t)K * d;
    case covariance_type::SPHERICAL: return (int64_t)K;
  }
  return 0;
}

template <typename T>
class GMMTest : public ::testing::TestWithParam<GMMInputs<T>> {
 protected:
  GMMTest() : stream(raft::resource::get_cuda_stream(handle)) {}

  void basicTest()
  {
    auto p = ::testing::TestWithParam<GMMInputs<T>>::GetParam();
    int n = p.n_row, d = p.n_col, K = p.n_components;
    int64_t cn = cov_len(p.cov_type, d, K);

    // Well-separated blobs: hard labels should recover the generating clusters.
    auto d_X    = raft::make_device_matrix<T, int64_t>(handle, n, d);
    auto d_yref = raft::make_device_vector<int, int>(handle, n);
    raft::random::make_blobs<T, int>(d_X.data_handle(),
                                     d_yref.data_handle(),
                                     n,
                                     d,
                                     K,
                                     stream,
                                     /* row_major          */ true,
                                     /* centers            */ nullptr,
                                     /* cluster_std        */ nullptr,
                                     /* cluster_std_scalar */ T(1.0),
                                     /* shuffle            */ false,
                                     /* center_box_min     */ static_cast<T>(-10.0f),
                                     /* center_box_max     */ static_cast<T>(10.0f),
                                     /* seed               */ 1234ULL);

    auto weights = raft::make_device_vector<T, int64_t>(handle, K);
    auto means   = raft::make_device_matrix<T, int64_t>(handle, K, d);
    auto covs    = raft::make_device_vector<T, int64_t>(handle, cn);
    auto pchol   = raft::make_device_vector<T, int64_t>(handle, cn);
    auto precs   = raft::make_device_vector<T, int64_t>(handle, cn);
    auto labels  = raft::make_device_vector<int, int64_t>(handle, n);

    params prm;
    prm.n_components = K;
    prm.cov_type     = p.cov_type;
    prm.init         = init_method::KMeans;
    prm.max_iter     = 100;
    prm.seed         = 1234ULL;

    T lower_bound  = 0;
    int n_iter     = 0;
    bool converged = false;

    fit(handle,
        prm,
        raft::make_const_mdspan(d_X.view()),
        weights.view(),
        means.view(),
        covs.view(),
        pchol.view(),
        precs.view(),
        labels.view(),
        raft::make_host_scalar_view(&lower_bound),
        raft::make_host_scalar_view(&n_iter),
        raft::make_host_scalar_view(&converged));

    // Fit diagnostics are sane. Well-separated blobs converge well before
    // max_iter, so the converged flag must be set and EM must have stopped early.
    ASSERT_TRUE(std::isfinite((double)lower_bound));
    ASSERT_GE(n_iter, 1);
    ASSERT_TRUE(converged) << "EM did not converge on well-separated blobs";
    ASSERT_LT(n_iter, prm.max_iter) << "converged flag set but EM ran to max_iter";

    // Hard labels recover the ground-truth clusters on well-separated blobs.
    double ari_fit =
      raft::stats::adjusted_rand_index(d_yref.data_handle(), labels.data_handle(), n, stream);
    raft::resource::sync_stream(handle, stream);
    ASSERT_GT(ari_fit, 0.95) << "fit labels disagree with ground truth";

    // predict() on the same data reproduces the fit labels exactly.
    auto labels2 = raft::make_device_vector<int, int64_t>(handle, n);
    predict(handle,
            prm,
            raft::make_const_mdspan(d_X.view()),
            raft::make_const_mdspan(weights.view()),
            raft::make_const_mdspan(means.view()),
            raft::make_const_mdspan(pchol.view()),
            labels2.view());
    double ari_pred =
      raft::stats::adjusted_rand_index(labels.data_handle(), labels2.data_handle(), n, stream);
    raft::resource::sync_stream(handle, stream);
    ASSERT_NEAR(ari_pred, 1.0, 1e-6) << "predict disagrees with fit labels";

    // predict_proba rows form a valid distribution (sum to 1, non-negative).
    auto resp = raft::make_device_matrix<T, int64_t>(handle, n, K);
    predict_proba(handle,
                  prm,
                  raft::make_const_mdspan(d_X.view()),
                  raft::make_const_mdspan(weights.view()),
                  raft::make_const_mdspan(means.view()),
                  raft::make_const_mdspan(pchol.view()),
                  resp.view());
    std::vector<T> h_resp((size_t)n * K);
    raft::update_host(h_resp.data(), resp.data_handle(), (size_t)n * K, stream);

    // score_samples: per-sample log-likelihood; its mean equals the lower
    // bound returned by fit (both are the average log p(x)).
    auto logp = raft::make_device_vector<T, int64_t>(handle, n);
    score_samples(handle,
                  prm,
                  raft::make_const_mdspan(d_X.view()),
                  raft::make_const_mdspan(weights.view()),
                  raft::make_const_mdspan(means.view()),
                  raft::make_const_mdspan(pchol.view()),
                  logp.view());
    std::vector<T> h_logp(n);
    raft::update_host(h_logp.data(), logp.data_handle(), n, stream);
    raft::resource::sync_stream(handle, stream);

    for (int i = 0; i < n; ++i) {
      double s = 0.0;
      for (int k = 0; k < K; ++k) {
        T r = h_resp[(size_t)i * K + k];
        ASSERT_GE((double)r, -1e-5);
        s += (double)r;
      }
      ASSERT_NEAR(s, 1.0, 1e-3) << "responsibilities row " << i << " not normalized";
    }

    double mean_logp = 0.0;
    for (int i = 0; i < n; ++i)
      mean_logp += (double)h_logp[i];
    mean_logp /= n;
    // tolerance loose for float; lower_bound is the fit-time average log p(x).
    double tol = std::is_same_v<T, float> ? 1e-2 : 1e-5;
    ASSERT_NEAR(mean_logp, (double)lower_bound, std::abs((double)lower_bound) * tol + tol);
  }

  raft::resources handle;
  cudaStream_t stream;
};

const std::vector<GMMInputs<float>> inputsf = {
  {600, 8, 4, covariance_type::FULL},
  {600, 8, 4, covariance_type::TIED},
  {600, 8, 4, covariance_type::DIAG},
  {600, 8, 4, covariance_type::SPHERICAL},
  {2000, 16, 5, covariance_type::FULL},  // fixed-D=16 specialization
  {2000, 16, 5, covariance_type::DIAG},
  {2000, 32, 4, covariance_type::FULL},          // fixed-D=32 specialization
  {2000, 50, 4, covariance_type::FULL},          // fixed-D=50 specialization
  {2000, 64, 4, covariance_type::FULL},          // fixed-D=64 specialization (boundary)
  {3000, 128, 4, covariance_type::FULL},         // 64<d<257 -> tiled thread64 kernel
  {3000, 128, 4, covariance_type::TIED},         // tied, tiled thread64 kernel
  {4000, 300, 4, covariance_type::FULL},         // d>=257 (float) -> cuBLAS E-step route
  {2000, 512, 16, covariance_type::DIAG},        // K*d large -> diag global-mem path
  {2000, 1024, 16, covariance_type::SPHERICAL},  // K*d large -> spherical global-mem path
};

const std::vector<GMMInputs<double>> inputsd = {
  {600, 8, 4, covariance_type::FULL},
  {600, 8, 4, covariance_type::TIED},
  {600, 8, 4, covariance_type::DIAG},
  {600, 8, 4, covariance_type::SPHERICAL},
  {2000, 16, 5, covariance_type::FULL},
  {2000, 50, 4, covariance_type::FULL},   // fixed-D=50 specialization
  {2000, 64, 4, covariance_type::FULL},   // fixed-D=64 specialization (boundary)
  {3000, 128, 4, covariance_type::FULL},  // d>64 (double) -> cuBLAS E-step route
};

using GMMTestF = GMMTest<float>;
TEST_P(GMMTestF, Result) { basicTest(); }
INSTANTIATE_TEST_CASE_P(GMMTests, GMMTestF, ::testing::ValuesIn(inputsf));

using GMMTestD = GMMTest<double>;
TEST_P(GMMTestD, Result) { basicTest(); }
INSTANTIATE_TEST_CASE_P(GMMTests, GMMTestD, ::testing::ValuesIn(inputsd));

// ---------------------------------------------------------------------------
// Standalone tests for behaviors not covered by the parametrized sweep:
// every init method, warm_start, n_init best-of, and the ill-defined-
// covariance error path.
// ---------------------------------------------------------------------------
namespace {

// Generate well-separated blobs into freshly allocated device buffers.
template <typename T>
std::pair<raft::device_matrix<T, int64_t>, raft::device_vector<int, int>> make_gmm_blobs(
  raft::resources const& handle, int n, int d, int K, std::uint64_t seed = 1234ULL)
{
  auto X    = raft::make_device_matrix<T, int64_t>(handle, n, d);
  auto yref = raft::make_device_vector<int, int>(handle, n);
  raft::random::make_blobs<T, int>(X.data_handle(),
                                   yref.data_handle(),
                                   n,
                                   d,
                                   K,
                                   raft::resource::get_cuda_stream(handle),
                                   true,
                                   nullptr,
                                   nullptr,
                                   T(1.0),
                                   false,
                                   static_cast<T>(-10.0f),
                                   static_cast<T>(10.0f),
                                   seed);
  return {std::move(X), std::move(yref)};
}

}  // namespace

// Every init method produces a valid fit; kmeans-family inits recover the
// generating clusters on well-separated blobs.
TEST(GMMExtra, InitMethods)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  const int n = 1500, d = 8, K = 4;
  auto [X, yref] = make_gmm_blobs<float>(handle, n, d, K);

  for (auto im : {init_method::KMeans,
                  init_method::KMeansPlusPlus,
                  init_method::Random,
                  init_method::RandomFromData}) {
    int64_t cn   = cov_len(covariance_type::FULL, d, K);
    auto weights = raft::make_device_vector<float, int64_t>(handle, K);
    auto means   = raft::make_device_matrix<float, int64_t>(handle, K, d);
    auto covs    = raft::make_device_vector<float, int64_t>(handle, cn);
    auto pchol   = raft::make_device_vector<float, int64_t>(handle, cn);
    auto precs   = raft::make_device_vector<float, int64_t>(handle, cn);
    auto labels  = raft::make_device_vector<int, int64_t>(handle, n);

    params prm;
    prm.n_components = K;
    prm.cov_type     = covariance_type::FULL;
    prm.init         = im;
    prm.n_init       = 1;
    // A modest regularizer keeps every init's first covariance well-defined so
    // the test deterministically exercises the init code path (random inits
    // can otherwise collapse a component, which legitimately raises).
    prm.reg_covar = 1e-2;
    prm.max_iter  = 100;
    prm.seed      = 1234ULL;

    float lb = 0;
    int it   = 0;
    bool cv  = false;
    fit(handle,
        prm,
        raft::make_const_mdspan(X.view()),
        weights.view(),
        means.view(),
        covs.view(),
        pchol.view(),
        precs.view(),
        labels.view(),
        raft::make_host_scalar_view(&lb),
        raft::make_host_scalar_view(&it),
        raft::make_host_scalar_view(&cv));

    ASSERT_TRUE(std::isfinite(lb)) << "init " << (int)im;
    if (im == init_method::KMeans || im == init_method::KMeansPlusPlus) {
      double ari =
        raft::stats::adjusted_rand_index(yref.data_handle(), labels.data_handle(), n, stream);
      raft::resource::sync_stream(handle, stream);
      ASSERT_GT(ari, 0.95) << "init " << (int)im;
    }
  }
}

// warm_start reuses the supplied weights/means/covariances as the single
// initialization and refines them to a finite, non-decreasing lower bound.
TEST(GMMExtra, WarmStart)
{
  raft::resources handle;
  const int n = 1500, d = 8, K = 4;
  auto [X, yref] = make_gmm_blobs<double>(handle, n, d, K);

  int64_t cn   = cov_len(covariance_type::FULL, d, K);
  auto weights = raft::make_device_vector<double, int64_t>(handle, K);
  auto means   = raft::make_device_matrix<double, int64_t>(handle, K, d);
  auto covs    = raft::make_device_vector<double, int64_t>(handle, cn);
  auto pchol   = raft::make_device_vector<double, int64_t>(handle, cn);
  auto precs   = raft::make_device_vector<double, int64_t>(handle, cn);
  auto labels  = raft::make_device_vector<int, int64_t>(handle, n);

  params prm;
  prm.n_components = K;
  prm.cov_type     = covariance_type::FULL;
  prm.init         = init_method::KMeans;
  prm.max_iter     = 5;
  prm.seed         = 1234ULL;

  double lb1 = 0;
  int it1    = 0;
  bool cv1   = false;
  auto run   = [&](bool warm, double& lb, int& it, bool& cv) {
    fit(handle,
        prm,
        raft::make_const_mdspan(X.view()),
        weights.view(),
        means.view(),
        covs.view(),
        pchol.view(),
        precs.view(),
        labels.view(),
        raft::make_host_scalar_view(&lb),
        raft::make_host_scalar_view(&it),
        raft::make_host_scalar_view(&cv),
        warm);
  };
  run(false, lb1, it1, cv1);

  // Continue from the fitted parameters; the lower bound should not regress.
  double lb2   = 0;
  int it2      = 0;
  bool cv2     = false;
  prm.max_iter = 20;
  run(true, lb2, it2, cv2);
  ASSERT_TRUE(std::isfinite(lb2));
  ASSERT_GE(lb2, lb1 - 1e-6);
}

// n_init>1 keeps the restart with the largest lower bound. The first restart
// of an N-restart fit uses the same seed as a single-restart fit, so more
// restarts can only match or beat it: lower_bound(n_init=N) >= lower_bound(1).
TEST(GMMExtra, NInitSelectsBest)
{
  raft::resources handle;
  const int n = 1500, d = 8, K = 4;
  auto [X, yref] = make_gmm_blobs<float>(handle, n, d, K);

  int64_t cn   = cov_len(covariance_type::FULL, d, K);
  auto weights = raft::make_device_vector<float, int64_t>(handle, K);
  auto means   = raft::make_device_matrix<float, int64_t>(handle, K, d);
  auto covs    = raft::make_device_vector<float, int64_t>(handle, cn);
  auto pchol   = raft::make_device_vector<float, int64_t>(handle, cn);
  auto precs   = raft::make_device_vector<float, int64_t>(handle, cn);
  auto labels  = raft::make_device_vector<int, int64_t>(handle, n);

  auto run = [&](int n_init) {
    params prm;
    prm.n_components = K;
    prm.cov_type     = covariance_type::FULL;
    prm.init         = init_method::Random;  // restart-sensitive init
    prm.n_init       = n_init;
    prm.reg_covar    = 1e-2;  // keep every restart well-defined
    prm.max_iter     = 50;
    prm.seed         = 1234ULL;
    float lb         = 0;
    int it           = 0;
    bool cv          = false;
    fit(handle,
        prm,
        raft::make_const_mdspan(X.view()),
        weights.view(),
        means.view(),
        covs.view(),
        pchol.view(),
        precs.view(),
        labels.view(),
        raft::make_host_scalar_view(&lb),
        raft::make_host_scalar_view(&it),
        raft::make_host_scalar_view(&cv));
    return lb;
  };

  float lb1  = run(1);
  float lb10 = run(10);
  ASSERT_GE((double)lb10, (double)lb1 - 1e-5);
}

// A zero-weight component is a valid sklearn parameterization (e.g. an
// imported model with a pruned component). Its -inf log-prob must fold
// cleanly into the online log-sum-exp: score_samples must return the same
// values as the equivalent mixture without the dead component, not NaN.
// Regression test for the fused-path running max initialized to -inf.
TEST(GMMExtra, ZeroWeightComponentScoresFinite)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  const int n = 64, d = 2;

  // Two unit-covariance components at (-2, 0) and (2, 0), plus a dead one.
  std::vector<float> h_X(n * d);
  for (int i = 0; i < n; ++i) {
    h_X[i * d + 0] = (i % 2 ? 2.0f : -2.0f) + 0.01f * i;
    h_X[i * d + 1] = 0.1f * (i % 5);
  }
  auto X = raft::make_device_matrix<float, int64_t>(handle, n, d);
  raft::update_device(X.data_handle(), h_X.data(), h_X.size(), stream);

  auto upload = [&](std::vector<float> const& h) {
    auto buf = raft::make_device_vector<float, int64_t>(handle, (int64_t)h.size());
    raft::update_device(buf.data_handle(), h.data(), h.size(), stream);
    return buf;
  };

  // Identity precision Cholesky per component (unit covariance).
  std::vector<float> h_eye = {1, 0, 0, 1};
  std::vector<float> h_w3 = {0.0f, 0.5f, 0.5f}, h_w2 = {0.5f, 0.5f};
  std::vector<float> h_m3 = {9, 9, -2, 0, 2, 0}, h_m2 = {-2, 0, 2, 0};
  std::vector<float> h_p3, h_p2;
  for (int k = 0; k < 3; ++k)
    h_p3.insert(h_p3.end(), h_eye.begin(), h_eye.end());
  for (int k = 0; k < 2; ++k)
    h_p2.insert(h_p2.end(), h_eye.begin(), h_eye.end());

  auto w3 = upload(h_w3), w2 = upload(h_w2), p3 = upload(h_p3), p2 = upload(h_p2);
  auto m3 = raft::make_device_matrix<float, int64_t>(handle, 3, d);
  auto m2 = raft::make_device_matrix<float, int64_t>(handle, 2, d);
  raft::update_device(m3.data_handle(), h_m3.data(), h_m3.size(), stream);
  raft::update_device(m2.data_handle(), h_m2.data(), h_m2.size(), stream);

  auto score = [&](int K, auto& w, auto& m, auto& p, std::vector<float>& out) {
    params prm;
    prm.n_components = K;
    prm.cov_type     = covariance_type::FULL;
    auto logp        = raft::make_device_vector<float, int64_t>(handle, n);
    score_samples(handle,
                  prm,
                  raft::make_const_mdspan(X.view()),
                  raft::make_const_mdspan(w.view()),
                  raft::make_const_mdspan(m.view()),
                  raft::make_const_mdspan(p.view()),
                  logp.view());
    out.resize(n);
    raft::update_host(out.data(), logp.data_handle(), n, stream);
    raft::resource::sync_stream(handle, stream);
  };

  std::vector<float> lp3, lp2;
  score(3, w3, m3, p3, lp3);
  score(2, w2, m2, p2, lp2);
  for (int i = 0; i < n; ++i) {
    ASSERT_TRUE(std::isfinite(lp3[i])) << "NaN/inf log-prob with zero-weight component, row " << i;
    ASSERT_NEAR(lp3[i], lp2[i], 1e-5) << "row " << i;
  }

  // predict must ignore the dead component: labels shift by exactly one.
  auto lab3 = raft::make_device_vector<int, int64_t>(handle, n);
  auto lab2 = raft::make_device_vector<int, int64_t>(handle, n);
  params prm3, prm2;
  prm3.n_components = 3;
  prm2.n_components = 2;
  prm3.cov_type = prm2.cov_type = covariance_type::FULL;
  predict(handle,
          prm3,
          raft::make_const_mdspan(X.view()),
          raft::make_const_mdspan(w3.view()),
          raft::make_const_mdspan(m3.view()),
          raft::make_const_mdspan(p3.view()),
          lab3.view());
  predict(handle,
          prm2,
          raft::make_const_mdspan(X.view()),
          raft::make_const_mdspan(w2.view()),
          raft::make_const_mdspan(m2.view()),
          raft::make_const_mdspan(p2.view()),
          lab2.view());
  std::vector<int> h_lab3(n), h_lab2(n);
  raft::update_host(h_lab3.data(), lab3.data_handle(), n, stream);
  raft::update_host(h_lab2.data(), lab2.data_handle(), n, stream);
  raft::resource::sync_stream(handle, stream);
  for (int i = 0; i < n; ++i)
    ASSERT_EQ(h_lab3[i], h_lab2[i] + 1) << "row " << i;
}

// Invalid hyper-parameters and buffer shapes must be rejected with a clear
// exception at the public API boundary instead of failing inside a kernel.
TEST(GMMExtra, InvalidArgsThrow)
{
  raft::resources handle;
  const int n = 32, d = 4, K = 2;
  auto [X, yref] = make_gmm_blobs<float>(handle, n, d, K);

  int64_t cn   = cov_len(covariance_type::SPHERICAL, d, K);
  auto weights = raft::make_device_vector<float, int64_t>(handle, K);
  auto means   = raft::make_device_matrix<float, int64_t>(handle, K, d);
  auto covs    = raft::make_device_vector<float, int64_t>(handle, cn);
  auto pchol   = raft::make_device_vector<float, int64_t>(handle, cn);
  auto precs   = raft::make_device_vector<float, int64_t>(handle, cn);
  auto labels  = raft::make_device_vector<int, int64_t>(handle, n);

  params base;
  base.n_components = K;
  base.cov_type     = covariance_type::SPHERICAL;

  float lb     = 0;
  int it       = 0;
  bool cv      = false;
  auto try_fit = [&](params const& prm) {
    fit(handle,
        prm,
        raft::make_const_mdspan(X.view()),
        weights.view(),
        means.view(),
        covs.view(),
        pchol.view(),
        precs.view(),
        labels.view(),
        raft::make_host_scalar_view(&lb),
        raft::make_host_scalar_view(&it),
        raft::make_host_scalar_view(&cv));
  };

  {
    params prm    = base;
    prm.reg_covar = -1e-6;
    EXPECT_THROW(try_fit(prm), raft::logic_error);
  }
  {
    params prm = base;
    prm.tol    = -1.0;
    EXPECT_THROW(try_fit(prm), raft::logic_error);
  }
  {
    params prm   = base;
    prm.cov_type = static_cast<covariance_type>(42);
    EXPECT_THROW(try_fit(prm), raft::logic_error);
  }
  {
    params prm = base;
    prm.init   = static_cast<init_method>(42);
    EXPECT_THROW(try_fit(prm), raft::logic_error);
  }
  {  // n_components above the supported limit (checked before K <= n).
    params prm       = base;
    prm.n_components = 70000;
    auto w_big       = raft::make_device_vector<float, int64_t>(handle, 70000);
    auto m_big       = raft::make_device_matrix<float, int64_t>(handle, 70000, d);
    auto p_big       = raft::make_device_vector<float, int64_t>(handle, 70000);
    auto lab         = raft::make_device_vector<int, int64_t>(handle, n);
    EXPECT_THROW(predict(handle,
                         prm,
                         raft::make_const_mdspan(X.view()),
                         raft::make_const_mdspan(w_big.view()),
                         raft::make_const_mdspan(m_big.view()),
                         raft::make_const_mdspan(p_big.view()),
                         lab.view()),
                 raft::logic_error);
  }
  {  // weights buffer with the wrong number of elements.
    auto w_bad = raft::make_device_vector<float, int64_t>(handle, K + 1);
    auto lab   = raft::make_device_vector<int, int64_t>(handle, n);
    EXPECT_THROW(predict(handle,
                         base,
                         raft::make_const_mdspan(X.view()),
                         raft::make_const_mdspan(w_bad.view()),
                         raft::make_const_mdspan(means.view()),
                         raft::make_const_mdspan(pchol.view()),
                         lab.view()),
                 raft::logic_error);
  }
}

// A degenerate component (more components than distinct points) yields an
// ill-defined covariance and must surface as an exception rather than NaNs.
TEST(GMMExtra, IllDefinedCovarianceThrows)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  const int n = 6, d = 4, K = 5;

  // All points identical -> any component covariance collapses to zero.
  auto X = raft::make_device_matrix<float, int64_t>(handle, n, d);
  RAFT_CUDA_TRY(cudaMemsetAsync(X.data_handle(), 0, sizeof(float) * (size_t)n * d, stream));

  int64_t cn   = cov_len(covariance_type::FULL, d, K);
  auto weights = raft::make_device_vector<float, int64_t>(handle, K);
  auto means   = raft::make_device_matrix<float, int64_t>(handle, K, d);
  auto covs    = raft::make_device_vector<float, int64_t>(handle, cn);
  auto pchol   = raft::make_device_vector<float, int64_t>(handle, cn);
  auto precs   = raft::make_device_vector<float, int64_t>(handle, cn);
  auto labels  = raft::make_device_vector<int, int64_t>(handle, n);

  params prm;
  prm.n_components = K;
  prm.cov_type     = covariance_type::FULL;
  prm.init         = init_method::RandomFromData;
  prm.reg_covar    = 0.0;  // disable the regularizer that would otherwise mask it
  prm.max_iter     = 50;
  prm.seed         = 1234ULL;

  float lb = 0;
  int it   = 0;
  bool cv  = false;
  EXPECT_ANY_THROW(fit(handle,
                       prm,
                       raft::make_const_mdspan(X.view()),
                       weights.view(),
                       means.view(),
                       covs.view(),
                       pchol.view(),
                       precs.view(),
                       labels.view(),
                       raft::make_host_scalar_view(&lb),
                       raft::make_host_scalar_view(&it),
                       raft::make_host_scalar_view(&cv)));
}

}  // namespace cuvs::cluster::gmm
