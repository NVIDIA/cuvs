/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/cluster/detail/soar.cuh"
#include "../test_utils.cuh"

#include <cuvs/cluster/soar.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace cuvs::cluster::soar {

struct SoarInputs {
  int64_t n_rows;
  int64_t dim;
  int64_t n_clusters;
  float lambda;
};

::std::ostream& operator<<(::std::ostream& os, const SoarInputs& p)
{
  os << "{ " << p.n_rows << ", " << p.dim << ", " << p.n_clusters << ", " << p.lambda << '}';
  return os;
}

namespace {

/** Uniform random matrix in [-1, 1], generated on the host so the tests are reproducible. */
auto random_matrix(int64_t n_rows, int64_t dim, uint64_t seed) -> std::vector<float>
{
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(n_rows * dim);
  std::generate(data.begin(), data.end(), [&]() { return dist(rng); });
  return data;
}

/** Index of the closest centroid in L2, i.e. what k-means prediction would produce. */
auto nearest_centroids(const std::vector<float>& dataset,
                       const std::vector<float>& centroids,
                       int64_t n_rows,
                       int64_t dim,
                       int64_t n_clusters) -> std::vector<uint32_t>
{
  std::vector<uint32_t> labels(n_rows);
  for (int64_t i = 0; i < n_rows; i++) {
    double best_distance = std::numeric_limits<double>::max();
    for (int64_t c = 0; c < n_clusters; c++) {
      double distance = 0.0;
      for (int64_t k = 0; k < dim; k++) {
        double diff = static_cast<double>(dataset[i * dim + k]) - centroids[c * dim + k];
        distance += diff * diff;
      }
      if (distance < best_distance) {
        best_distance = distance;
        labels[i]     = static_cast<uint32_t>(c);
      }
    }
  }
  return labels;
}

auto residuals_host(const std::vector<float>& dataset,
                    const std::vector<float>& centroids,
                    const std::vector<uint32_t>& labels,
                    int64_t n_rows,
                    int64_t dim) -> std::vector<float>
{
  std::vector<float> residuals(n_rows * dim);
  for (int64_t i = 0; i < n_rows; i++) {
    for (int64_t k = 0; k < dim; k++) {
      residuals[i * dim + k] = dataset[i * dim + k] - centroids[labels[i] * dim + k];
    }
  }
  return residuals;
}

/**
 * `||x - c||^2 + lambda * (dot(r / ||r||, x - c))^2`, the loss that the device implementation
 * minimizes over all centroids, up to a per-row constant that does not move the argmin.
 */
auto soar_score(
  const float* x, const float* residual, const float* centroid, int64_t dim, float lambda) -> double
{
  double residual_norm = 0.0;
  for (int64_t k = 0; k < dim; k++) {
    residual_norm += static_cast<double>(residual[k]) * residual[k];
  }
  residual_norm = std::sqrt(residual_norm);

  double squared_distance = 0.0;
  double projection       = 0.0;
  for (int64_t k = 0; k < dim; k++) {
    double diff = static_cast<double>(x[k]) - centroid[k];
    squared_distance += diff * diff;
    projection += diff * (residual[k] / residual_norm);
  }

  return squared_distance + static_cast<double>(lambda) * projection * projection;
}

/** The best achievable loss per row, from an exhaustive host search. */
auto reference_scores(const std::vector<float>& dataset,
                      const std::vector<float>& centroids,
                      const std::vector<float>& residuals,
                      int64_t n_rows,
                      int64_t dim,
                      int64_t n_clusters,
                      float lambda) -> std::vector<double>
{
  std::vector<double> scores(n_rows);
  for (int64_t i = 0; i < n_rows; i++) {
    double best_score = std::numeric_limits<double>::max();
    for (int64_t c = 0; c < n_clusters; c++) {
      best_score = std::min(
        best_score,
        soar_score(&dataset[i * dim], &residuals[i * dim], &centroids[c * dim], dim, lambda));
    }
    scores[i] = best_score;
  }
  return scores;
}

}  // namespace

class SoarTest : public ::testing::TestWithParam<SoarInputs> {
 public:
  SoarTest()
    : params_(GetParam()),
      dataset_(raft::make_device_matrix<float, int64_t>(handle_, params_.n_rows, params_.dim)),
      centroids_(
        raft::make_device_matrix<float, int64_t>(handle_, params_.n_clusters, params_.dim)),
      labels_(raft::make_device_vector<uint32_t, int64_t>(handle_, params_.n_rows)),
      soar_labels_(raft::make_device_vector<uint32_t, int64_t>(handle_, params_.n_rows))
  {
  }

 protected:
  void SetUp() override
  {
    h_dataset_   = random_matrix(params_.n_rows, params_.dim, 1234ULL);
    h_centroids_ = random_matrix(params_.n_clusters, params_.dim, 5678ULL);
    h_labels_ =
      nearest_centroids(h_dataset_, h_centroids_, params_.n_rows, params_.dim, params_.n_clusters);

    auto stream = raft::resource::get_cuda_stream(handle_);
    raft::update_device(dataset_.data_handle(), h_dataset_.data(), h_dataset_.size(), stream);
    raft::update_device(centroids_.data_handle(), h_centroids_.data(), h_centroids_.size(), stream);
    raft::update_device(labels_.data_handle(), h_labels_.data(), h_labels_.size(), stream);
    raft::resource::sync_stream(handle_);
  }

  /** Run the public API and copy the resulting labels back to the host. */
  auto run_predict() -> std::vector<uint32_t>
  {
    cuvs::cluster::soar::params soar_params;
    soar_params.lambda = params_.lambda;

    cuvs::cluster::soar::predict(handle_,
                                 soar_params,
                                 raft::make_const_mdspan(dataset_.view()),
                                 raft::make_const_mdspan(centroids_.view()),
                                 raft::make_const_mdspan(labels_.view()),
                                 soar_labels_.view());

    return to_host(raft::make_const_mdspan(soar_labels_.view()));
  }

  auto to_host(raft::device_vector_view<const uint32_t, int64_t> labels) -> std::vector<uint32_t>
  {
    std::vector<uint32_t> h_labels(labels.extent(0));
    raft::update_host(h_labels.data(),
                      labels.data_handle(),
                      labels.extent(0),
                      raft::resource::get_cuda_stream(handle_));
    raft::resource::sync_stream(handle_);
    return h_labels;
  }

  raft::resources handle_;
  SoarInputs params_;

  std::vector<float> h_dataset_;
  std::vector<float> h_centroids_;
  std::vector<uint32_t> h_labels_;

  raft::device_matrix<float, int64_t> dataset_;
  raft::device_matrix<float, int64_t> centroids_;
  raft::device_vector<uint32_t, int64_t> labels_;
  raft::device_vector<uint32_t, int64_t> soar_labels_;
};

/**
 * Every label must be a valid centroid id achieving the same loss as an exhaustive host search.
 * Comparing losses rather than ids keeps the test from being fragile when two centroids are
 * nearly tied.
 */
TEST_P(SoarTest, MatchesHostReference)
{
  auto soar_labels = run_predict();

  auto h_residuals =
    residuals_host(h_dataset_, h_centroids_, h_labels_, params_.n_rows, params_.dim);
  auto best_scores = reference_scores(h_dataset_,
                                      h_centroids_,
                                      h_residuals,
                                      params_.n_rows,
                                      params_.dim,
                                      params_.n_clusters,
                                      params_.lambda);

  for (int64_t i = 0; i < params_.n_rows; i++) {
    ASSERT_LT(soar_labels[i], static_cast<uint32_t>(params_.n_clusters))
      << "label out of range at row " << i;

    double score = soar_score(&h_dataset_[i * params_.dim],
                              &h_residuals[i * params_.dim],
                              &h_centroids_[soar_labels[i] * params_.dim],
                              params_.dim,
                              params_.lambda);
    ASSERT_NEAR(score, best_scores[i], 1e-4 * (1.0 + std::abs(best_scores[i])))
      << "row " << i << " picked centroid " << soar_labels[i];
  }
}

/**
 * The residuals feeding the SOAR loss must match a plain host `x - c[label]`. Both the public
 * `predict` and the ScaNN builder depend on this, and `MatchesHostReference` only detects
 * residual errors large enough to move an argmin.
 */
TEST_P(SoarTest, ComputeResidualsMatchesHost)
{
  auto h_residuals =
    residuals_host(h_dataset_, h_centroids_, h_labels_, params_.n_rows, params_.dim);

  auto residuals =
    detail::compute_residuals<float, uint32_t>(handle_,
                                               raft::make_const_mdspan(dataset_.view()),
                                               raft::make_const_mdspan(centroids_.view()),
                                               raft::make_const_mdspan(labels_.view()));

  ASSERT_TRUE(cuvs::devArrMatchHost(h_residuals.data(),
                                    residuals.data_handle(),
                                    h_residuals.size(),
                                    cuvs::CompareApprox<float>(1e-6f),
                                    raft::resource::get_cuda_stream(handle_)));
}

const std::vector<SoarInputs> inputs = {{1000, 8, 16, 1.0f},
                                        {1000, 8, 16, 0.0f},
                                        {1000, 8, 16, 4.0f},
                                        {512, 32, 64, 1.5f},
                                        {17, 3, 2, 1.0f}};

INSTANTIATE_TEST_CASE_P(SoarTests, SoarTest, ::testing::ValuesIn(inputs));

/**
 * A hand-checked case with well-separated centroids, covering both outcomes: a row in the
 * interior of its cluster keeps its primary centroid, because no other centroid is close enough
 * to be worth spilling to, while a row near a boundary spills to the neighboring cluster.
 */
TEST(SoarTestSmall, SeparatedClusters)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  constexpr int64_t n_rows = 4, dim = 2, n_clusters = 3;

  std::vector<float> h_centroids{0.0f, 0.0f, 100.0f, 0.0f, 0.0f, 100.0f};
  std::vector<float> h_dataset{1.0f, 0.0f, 48.0f, 20.0f, 20.0f, 48.0f, 99.0f, 0.0f};
  std::vector<uint32_t> h_labels{0, 0, 0, 1};

  // Rows 0 and 3 sit next to their own centroid and keep it. Rows 1 and 2 sit between two
  // centroids, so the second-closest one wins with a ~13% margin in the loss.
  std::vector<uint32_t> expected{0, 1, 2, 1};

  auto dataset     = raft::make_device_matrix<float, int64_t>(handle, n_rows, dim);
  auto centroids   = raft::make_device_matrix<float, int64_t>(handle, n_clusters, dim);
  auto labels      = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows);
  auto soar_labels = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows);

  raft::update_device(dataset.data_handle(), h_dataset.data(), h_dataset.size(), stream);
  raft::update_device(centroids.data_handle(), h_centroids.data(), h_centroids.size(), stream);
  raft::update_device(labels.data_handle(), h_labels.data(), h_labels.size(), stream);

  cuvs::cluster::soar::params params;
  cuvs::cluster::soar::predict(handle,
                               params,
                               raft::make_const_mdspan(dataset.view()),
                               raft::make_const_mdspan(centroids.view()),
                               raft::make_const_mdspan(labels.view()),
                               soar_labels.view());

  std::vector<uint32_t> result(n_rows);
  raft::update_host(result.data(), soar_labels.data_handle(), n_rows, stream);
  raft::resource::sync_stream(handle);

  ASSERT_EQ(expected, result);
}

TEST(SoarTestErrors, RejectsMismatchedShapes)
{
  raft::resources handle;

  constexpr int64_t n_rows = 32, dim = 4, n_clusters = 8;

  auto dataset     = raft::make_device_matrix<float, int64_t>(handle, n_rows, dim);
  auto centroids   = raft::make_device_matrix<float, int64_t>(handle, n_clusters, dim);
  auto labels      = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows);
  auto soar_labels = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows);

  cuvs::cluster::soar::params params;
  auto dataset_view   = raft::make_const_mdspan(dataset.view());
  auto centroids_view = raft::make_const_mdspan(centroids.view());
  auto labels_view    = raft::make_const_mdspan(labels.view());

  // centroid dimension differs from the dataset dimension
  auto narrow_centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, dim - 1);
  ASSERT_THROW(cuvs::cluster::soar::predict(handle,
                                            params,
                                            dataset_view,
                                            raft::make_const_mdspan(narrow_centroids.view()),
                                            labels_view,
                                            soar_labels.view()),
               raft::logic_error);

  // no centroids to choose from
  auto no_centroids = raft::make_device_matrix<float, int64_t>(handle, 0, dim);
  ASSERT_THROW(cuvs::cluster::soar::predict(handle,
                                            params,
                                            dataset_view,
                                            raft::make_const_mdspan(no_centroids.view()),
                                            labels_view,
                                            soar_labels.view()),
               raft::logic_error);

  // one primary label per row is required
  auto short_labels = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows - 1);
  ASSERT_THROW(cuvs::cluster::soar::predict(handle,
                                            params,
                                            dataset_view,
                                            centroids_view,
                                            raft::make_const_mdspan(short_labels.view()),
                                            soar_labels.view()),
               raft::logic_error);

  // one output slot per row is required
  auto short_output = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows - 1);
  ASSERT_THROW(cuvs::cluster::soar::predict(
                 handle, params, dataset_view, centroids_view, labels_view, short_output.view()),
               raft::logic_error);
}

}  // namespace cuvs::cluster::soar
