/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/benchmark.hpp"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace cuvs::bench::cluster {

/** Lloyd's k-means (`cuvs::cluster::kmeans::fit` with `kmeans::params`). */
class cuvs_kmeans : public algo<float> {
 public:
  using param_type = cuvs::cluster::kmeans::params;

  cuvs_kmeans(const raft::resources& res, const param_type& param) : res_(res), param_(param) {}

  void fit(raft::device_matrix_view<const float, int64_t> X) override
  {
    if (!centroids_.has_value() || centroids_->extent(1) != X.extent(1)) {
      centroids_.emplace(
        raft::make_device_matrix<float, int64_t>(res_, param_.n_clusters, X.extent(1)));
    }
    cuvs::cluster::kmeans::fit(res_,
                               param_,
                               X,
                               std::nullopt,
                               centroids_->view(),
                               raft::make_host_scalar_view(&inertia_),
                               raft::make_host_scalar_view(&n_iter_));
  }

  [[nodiscard]] auto centroids() const -> raft::device_matrix_view<const float, int64_t> override
  {
    return raft::make_const_mdspan(centroids_->view());
  }

  [[nodiscard]] auto metrics() const -> std::vector<std::tuple<std::string, double>> override
  {
    return {{"n_iter", static_cast<double>(n_iter_)}};
  }

 private:
  const raft::resources& res_;
  param_type param_;
  std::optional<raft::device_matrix<float, int64_t>> centroids_;
  float inertia_  = 0;  // required output of fit; the bench reports its own, untimed inertia
  int64_t n_iter_ = 0;
};

/** Balanced hierarchical k-means (`cuvs::cluster::kmeans::fit` with `balanced_params`). */
template <typename T>
class cuvs_kmeans_balanced : public algo<T> {
 public:
  using param_type = cuvs::cluster::kmeans::balanced_params;

  cuvs_kmeans_balanced(const raft::resources& res, const param_type& param, int64_t n_clusters)
    : res_(res), param_(param), n_clusters_(n_clusters)
  {
  }

  void fit(raft::device_matrix_view<const T, int64_t> X) override
  {
    if (!centroids_.has_value() || centroids_->extent(1) != X.extent(1)) {
      centroids_.emplace(raft::make_device_matrix<float, int64_t>(res_, n_clusters_, X.extent(1)));
    }
    // No inertia output: it would add a cost pass to the timed fit.
    cuvs::cluster::kmeans::fit(res_, param_, X, centroids_->view());
  }

  [[nodiscard]] auto centroids() const -> raft::device_matrix_view<const float, int64_t> override
  {
    return raft::make_const_mdspan(centroids_->view());
  }

  [[nodiscard]] auto metrics() const -> std::vector<std::tuple<std::string, double>> override
  {
    return {};
  }

 private:
  const raft::resources& res_;
  param_type param_;
  int64_t n_clusters_;
  std::optional<raft::device_matrix<float, int64_t>> centroids_;
};

}  // namespace cuvs::bench::cluster
