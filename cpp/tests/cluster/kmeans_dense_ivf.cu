/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../../src/cluster/detail/kmeans_dense_ivf.cuh"

#include <gtest/gtest.h>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <random>

namespace cuvs::cluster::kmeans {
namespace dense_ivf_test {

template <typename T, typename I>
void compare_search(bool duplicate, int probes, int batch)
{
  raft::resources handle;
  auto stream    = raft::resource::get_cuda_stream(handle);
  constexpr I nc = 97, nq = 131, dim = 32;
  std::mt19937 rng(42);
  std::vector<T> centers(nc * dim), queries(nq * dim);
  for (auto& x : centers) {
    x = duplicate ? T{3} : T(int(rng() % 31) - 15);
  }
  // Exact centroid queries test the limited-probe path without depending on random ANN recall.
  for (I i = 0; i < nq; ++i) {
    for (I j = 0; j < dim; ++j) {
      queries[i * dim + j] = probes == 7 ? T(int(rng() % 31) - 15) : centers[(i % nc) * dim + j];
    }
  }
  auto dc = raft::make_device_matrix<T, I>(handle, nc, dim);
  auto dq = raft::make_device_matrix<T, I>(handle, nq, dim);
  raft::update_device(dc.data_handle(), centers.data(), centers.size(), stream);
  raft::update_device(dq.data_handle(), queries.data(), queries.size(), stream);
  params p;
  p.ivf_n_lists       = 7;
  p.ivf_n_probes      = probes;
  p.ivf_max_iter      = 4;
  p.ivf_batch_samples = batch;
  detail::dense_ivf_assignment<T, I> index(handle, p);
  for (int rebuild = 0; rebuild < 2; ++rebuild) {
    index.build(raft::make_const_mdspan(dc.view()));
    auto result = index.search(raft::make_const_mdspan(dq.view()));
    std::vector<raft::KeyValuePair<I, T>> found(nq);
    raft::update_host(found.data(), result.key_values(), found.size(), stream);
    raft::resource::sync_stream(handle);
    for (I q = 0; q < nq; ++q) {
      I best_id   = 0;
      double best = std::numeric_limits<double>::infinity();
      for (I c = 0; c < nc; ++c) {
        double d = 0;
        for (I j = 0; j < dim; ++j) {
          double delta = double(queries[q * dim + j]) - double(centers[c * dim + j]);
          d += delta * delta;
        }
        if (d < best) {
          best    = d;
          best_id = c;
        }
      }
      ASSERT_EQ(found[q].key, best_id) << "query " << q << " rebuild " << rebuild;
      ASSERT_NEAR(double(found[q].value), best, 0.02);
    }
    // Change the centroid ordering: the rebuilt original-ID mapping must change accordingly.
    for (I c = 0; c < nc / 2; ++c) {
      for (I j = 0; j < dim; ++j) {
        std::swap(centers[c * dim + j], centers[(nc - 1 - c) * dim + j]);
      }
    }
    raft::update_device(dc.data_handle(), centers.data(), centers.size(), stream);
  }
  auto empty = index.search(raft::make_device_matrix_view<const T, I>(dq.data_handle(), I{0}, dim));
  EXPECT_EQ(empty.size(), 0);
}

TEST(KmeansDenseIvf, AllProbesFloatInt) { compare_search<float, int>(false, 7, 37); }
TEST(KmeansDenseIvf, AllProbesFloatInt64) { compare_search<float, int64_t>(false, 7, 131); }
TEST(KmeansDenseIvf, AllProbesDouble) { compare_search<double, int64_t>(false, 7, 23); }
TEST(KmeansDenseIvf, LimitedProbesAndTail) { compare_search<float, int64_t>(false, 1, 17); }
TEST(KmeansDenseIvf, EmptyListsAndOriginalIdTies) { compare_search<float, int64_t>(true, 7, 31); }
TEST(KmeansDenseIvf, EmptyListsWithOneProbe) { compare_search<float, int>(true, 1, 19); }

void compare_fit(bool host_input, bool weighted)
{
  raft::resources handle;
  raft::resource::set_cuda_stream_pool(
    handle, std::make_shared<rmm::cuda_stream_pool>(1, rmm::cuda_stream::flags::non_blocking));
  auto stream            = raft::resource::get_cuda_stream(handle);
  constexpr int64_t rows = 113, dim = 7, clusters = 9;
  std::vector<float> data(rows * dim), initial(clusters * dim), weights(rows);
  for (int64_t r = 0; r < rows; ++r) {
    weights[r] = float(1 + r % 3);
    for (int64_t d = 0; d < dim; ++d) {
      data[r * dim + d] = float(8 * (r % clusters) + d + (r / clusters) % 3);
    }
  }
  for (int64_t c = 0; c < clusters; ++c) {
    for (int64_t d = 0; d < dim; ++d) {
      initial[c * dim + d] = float(8 * c + d);
    }
  }
  auto x = raft::make_device_matrix<float, int64_t>(handle, rows, dim);
  auto w = raft::make_device_vector<float, int64_t>(handle, rows);
  auto c = raft::make_device_matrix<float, int64_t>(handle, clusters, dim);
  raft::update_device(x.data_handle(), data.data(), data.size(), stream);
  raft::update_device(w.data_handle(), weights.data(), weights.size(), stream);
  params p;
  p.n_clusters            = clusters;
  p.init                  = params::Array;
  p.max_iter              = 3;
  p.tol                   = 1e-10;
  p.device_buffer_samples = 23;
  p.ivf_n_probes          = 3;
  p.ivf_max_iter          = 4;
  p.ivf_batch_samples     = 11;
  float costs[2]{};
  int64_t iters[2]{};
  std::vector<float> actual[2];
  for (int method = 0; method < 2; ++method) {
    p.ivf_n_lists = method == 0 ? 0 : 3;
    raft::update_device(c.data_handle(), initial.data(), initial.size(), stream);
    if (host_input) {
      std::optional<raft::host_vector_view<const float, int64_t>> sw;
      if (weighted) {
        sw = raft::make_host_vector_view<const float, int64_t>(weights.data(), rows);
      }
      fit(handle,
          p,
          raft::make_host_matrix_view<const float, int64_t>(data.data(), rows, dim),
          sw,
          c.view(),
          raft::make_host_scalar_view(costs + method),
          raft::make_host_scalar_view(iters + method));
    } else {
      std::optional<raft::device_vector_view<const float, int64_t>> sw;
      if (weighted) { sw = raft::make_const_mdspan(w.view()); }
      fit(handle,
          p,
          raft::make_const_mdspan(x.view()),
          sw,
          c.view(),
          raft::make_host_scalar_view(costs + method),
          raft::make_host_scalar_view(iters + method));
    }
    actual[method].resize(initial.size());
    raft::update_host(actual[method].data(), c.data_handle(), initial.size(), stream);
    raft::resource::sync_stream(handle);
  }
  // Independent FP64 reference: these separated groups have known assignments. Weighted
  // FP32 atomics can differ by a few ULPs, and expanded FP32 L2 can lose low bits when large
  // coordinate norms cancel. Check both methods against the reference, not bit identity.
  std::vector<double> expected(initial.size(), 0.0), weight_sums(clusters, 0.0);
  double total_weight = 0;
  for (int64_t r = 0; r < rows; ++r) {
    double weight = weighted ? weights[r] : 1.0;
    total_weight += weight;
    weight_sums[r % clusters] += weight;
    for (int64_t d = 0; d < dim; ++d) {
      expected[(r % clusters) * dim + d] += weight * data[r * dim + d];
    }
  }
  for (int64_t c = 0; c < clusters; ++c) {
    for (int64_t d = 0; d < dim; ++d) {
      expected[c * dim + d] /= weight_sums[c];
    }
  }
  double expected_cost = 0;
  for (int64_t r = 0; r < rows; ++r) {
    double distance = 0;
    for (int64_t d = 0; d < dim; ++d) {
      double delta = data[r * dim + d] - expected[(r % clusters) * dim + d];
      distance += delta * delta;
    }
    expected_cost += distance * (weighted ? weights[r] : 1.0) * rows / total_weight;
  }
  for (int method = 0; method < 2; ++method) {
    for (std::size_t i = 0; i < initial.size(); ++i) {
      ASSERT_NEAR(actual[method][i],
                  expected[i],
                  4 * std::numeric_limits<float>::epsilon() * std::max(1.0, std::abs(expected[i])));
    }
    EXPECT_NEAR(costs[method], expected_cost, expected_cost * 1e-3);
  }
  testing::Test::RecordProperty("cpu_reference_inertia", expected_cost);
  testing::Test::RecordProperty("exhaustive_inertia", costs[0]);
  testing::Test::RecordProperty("dense_ivf_inertia", costs[1]);
  EXPECT_EQ(iters[0], iters[1]);
  EXPECT_GT(iters[1], 1);
}
TEST(KmeansDenseIvf, FitDevice) { compare_fit(false, false); }
TEST(KmeansDenseIvf, FitDeviceWeighted) { compare_fit(false, true); }
TEST(KmeansDenseIvf, FitPageableBuffered) { compare_fit(true, false); }
TEST(KmeansDenseIvf, FitPageableBufferedWeighted) { compare_fit(true, true); }

TEST(KmeansDenseIvf, InvalidParameters)
{
  raft::resources handle;
  params p;
  p.ivf_n_lists = 4;
  EXPECT_THROW((detail::dense_ivf_assignment<float, int>(handle, p)), raft::logic_error);
  p.ivf_n_probes      = 1;
  p.ivf_batch_samples = 0;
  EXPECT_THROW((detail::dense_ivf_assignment<float, int>(handle, p)), raft::logic_error);
  p.ivf_batch_samples = 1;
  p.metric            = cuvs::distance::DistanceType::CosineExpanded;
  EXPECT_THROW((detail::dense_ivf_assignment<float, int>(handle, p)), raft::logic_error);
}
}  // namespace dense_ivf_test
}  // namespace cuvs::cluster::kmeans
