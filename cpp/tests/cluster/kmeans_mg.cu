/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "kmeans_test_blobs.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_resources_snmg.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace cuvs {

template <typename T>
struct KmeansSNMGInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  kmeans_weight_mode weight_mode;
  int device_buffer_samples;
  int n_init;
  cuvs::cluster::kmeans::params::InitMethod init = cuvs::cluster::kmeans::params::Array;
  int max_iter                                   = 20;
};

template <typename T>
class KmeansSNMGTest : public ::testing::TestWithParam<KmeansSNMGInputs<T>> {
 protected:
  KmeansSNMGTest() : clique_() { clique_.set_memory_pool(50); }

  void runTest()
  {
    testparams_ = ::testing::TestWithParam<KmeansSNMGInputs<T>>::GetParam();

    int n_samples  = testparams_.n_row;
    int n_features = testparams_.n_col;
    int n_clusters = testparams_.n_clusters;
    int num_ranks  = raft::resource::get_num_ranks(clique_);

    auto stream = raft::resource::get_cuda_stream(clique_);

    auto bi       = make_kmeans_blob_inputs<T>(clique_, n_samples, n_features, n_clusters);
    auto h_X_view = raft::make_const_mdspan(bi.h_X->view());

    auto d_centroids_snmg = raft::make_device_matrix<T, int64_t>(clique_, n_clusters, n_features);
    auto d_centroids_ref  = raft::make_device_matrix<T, int64_t>(clique_, n_clusters, n_features);

    if (testparams_.init == cuvs::cluster::kmeans::params::Array) {
      std::vector<int> h_labels(n_samples);
      raft::update_host(h_labels.data(), bi.d_labels_ref.data_handle(), n_samples, stream);
      raft::resource::sync_stream(clique_, stream);

      const T* h_X_data = bi.h_X->data_handle();
      std::vector<T> h_centroids(n_clusters * n_features, T(0));
      std::vector<int> counts(n_clusters, 0);
      for (int i = 0; i < n_samples; ++i) {
        int c = h_labels[i];
        counts[c]++;
        for (int j = 0; j < n_features; ++j)
          h_centroids[c * n_features + j] += h_X_data[i * n_features + j];
      }
      for (int c = 0; c < n_clusters; ++c) {
        if (counts[c] > 0) {
          for (int j = 0; j < n_features; ++j)
            h_centroids[c * n_features + j] /= T(counts[c]);
        }
      }

      raft::update_device(
        d_centroids_snmg.data_handle(), h_centroids.data(), n_clusters * n_features, stream);
      raft::copy(d_centroids_ref.data_handle(),
                 d_centroids_snmg.data_handle(),
                 n_clusters * n_features,
                 stream);
    }

    auto const mode = testparams_.weight_mode;
    auto h_sample_weight =
      raft::make_host_vector<T, int64_t>(mode != kmeans_weight_mode::none ? n_samples : 0);
    std::optional<raft::host_vector_view<const T, int64_t>> h_sw = std::nullopt;
    if (mode != kmeans_weight_mode::none) {
      fill_kmeans_test_weights(h_sample_weight.view(), mode);
      h_sw = std::make_optional(raft::make_const_mdspan(h_sample_weight.view()));
    }

    cuvs::cluster::kmeans::params snmg_params;
    snmg_params.n_clusters            = n_clusters;
    snmg_params.tol                   = testparams_.tol;
    snmg_params.max_iter              = testparams_.max_iter;
    snmg_params.n_init                = testparams_.n_init;
    snmg_params.rng_state.seed        = 42;
    snmg_params.init                  = testparams_.init;
    snmg_params.device_buffer_samples = testparams_.device_buffer_samples;

    T snmg_inertia      = T{0};
    int64_t snmg_n_iter = 0;

    cuvs::cluster::kmeans::fit(clique_,
                               snmg_params,
                               h_X_view,
                               h_sw,
                               d_centroids_snmg.view(),
                               raft::make_host_scalar_view(&snmg_inertia),
                               raft::make_host_scalar_view(&snmg_n_iter));

    raft::resources sg_handle;
    auto sg_stream = raft::resource::get_cuda_stream(sg_handle);

    auto d_centroids_sg = raft::make_device_matrix<T, int64_t>(sg_handle, n_clusters, n_features);
    if (testparams_.init == cuvs::cluster::kmeans::params::Array) {
      raft::copy(d_centroids_sg.data_handle(),
                 d_centroids_ref.data_handle(),
                 n_clusters * n_features,
                 sg_stream);
    }

    cuvs::cluster::kmeans::params sg_params = snmg_params;

    T sg_inertia      = T{0};
    int64_t sg_n_iter = 0;

    cuvs::cluster::kmeans::fit(sg_handle,
                               sg_params,
                               h_X_view,
                               h_sw,
                               d_centroids_sg.view(),
                               raft::make_host_scalar_view(&sg_inertia),
                               raft::make_host_scalar_view(&sg_n_iter));

    rmm::device_uvector<int> d_labels_snmg(n_samples, sg_stream);
    rmm::device_uvector<int> d_labels_sg(n_samples, sg_stream);
    rmm::device_uvector<int> d_labels_ref(n_samples, sg_stream);

    raft::copy(d_labels_ref.data(), bi.d_labels_ref.data_handle(), n_samples, sg_stream);

    auto X_dev_view =
      raft::make_device_matrix_view<const T, int>(bi.d_X.data_handle(), n_samples, n_features);

    cuvs::cluster::kmeans::params pred_params;
    pred_params.n_clusters = n_clusters;

    auto d_centroids_snmg_copy =
      raft::make_device_matrix<T, int>(sg_handle, n_clusters, n_features);
    raft::copy(d_centroids_snmg_copy.data_handle(),
               d_centroids_snmg.data_handle(),
               n_clusters * n_features,
               sg_stream);

    auto d_centroids_sg_int = raft::make_device_matrix<T, int>(sg_handle, n_clusters, n_features);
    raft::copy(d_centroids_sg_int.data_handle(),
               d_centroids_sg.data_handle(),
               n_clusters * n_features,
               sg_stream);

    T pred_inertia_snmg = T{0};
    cuvs::cluster::kmeans::predict(
      sg_handle,
      pred_params,
      X_dev_view,
      std::nullopt,
      raft::make_device_matrix_view<const T, int>(
        d_centroids_snmg_copy.data_handle(), n_clusters, n_features),
      raft::make_device_vector_view<int, int>(d_labels_snmg.data(), n_samples),
      true,
      raft::make_host_scalar_view(&pred_inertia_snmg));

    T pred_inertia_sg = T{0};
    cuvs::cluster::kmeans::predict(
      sg_handle,
      pred_params,
      X_dev_view,
      std::nullopt,
      raft::make_device_matrix_view<const T, int>(
        d_centroids_sg_int.data_handle(), n_clusters, n_features),
      raft::make_device_vector_view<int, int>(d_labels_sg.data(), n_samples),
      true,
      raft::make_host_scalar_view(&pred_inertia_sg));

    ari_vs_ref_ = raft::stats::adjusted_rand_index(
      d_labels_ref.data(), d_labels_snmg.data(), n_samples, sg_stream);

    ari_vs_sg_ = raft::stats::adjusted_rand_index(
      d_labels_sg.data(), d_labels_snmg.data(), n_samples, sg_stream);

    snmg_inertia_ = snmg_inertia;
    sg_inertia_   = sg_inertia;
    snmg_n_iter_  = snmg_n_iter;
    sg_n_iter_    = sg_n_iter;

    if (testparams_.init == cuvs::cluster::kmeans::params::Array) {
      std::vector<T> h_c_snmg(n_clusters * n_features);
      std::vector<T> h_c_sg(n_clusters * n_features);
      raft::update_host(
        h_c_snmg.data(), d_centroids_snmg_copy.data_handle(), n_clusters * n_features, sg_stream);
      raft::update_host(
        h_c_sg.data(), d_centroids_sg_int.data_handle(), n_clusters * n_features, sg_stream);
      raft::resource::sync_stream(sg_handle, sg_stream);

      double max_rel = 0;
      for (int i = 0; i < n_clusters * n_features; ++i) {
        double denom = std::max(double{1e-8}, std::abs(static_cast<double>(h_c_sg[i])));
        double rel =
          std::abs(static_cast<double>(h_c_snmg[i]) - static_cast<double>(h_c_sg[i])) / denom;
        max_rel = std::max(max_rel, rel);
      }
      max_centroid_rel_diff_   = max_rel;
      has_centroid_comparison_ = true;
    }

    if (ari_vs_ref_ < 0.94 || ari_vs_sg_ < 0.94) {
      std::cout << "SNMG KMeans: ARI vs ref = " << ari_vs_ref_ << ", ARI vs SG = " << ari_vs_sg_
                << ", num_ranks = " << num_ranks << ", snmg_inertia = " << snmg_inertia
                << ", sg_inertia = " << sg_inertia << ", snmg_n_iter = " << snmg_n_iter
                << ", sg_n_iter = " << sg_n_iter << std::endl;
    }
  }

  void SetUp() override { runTest(); }

  void checkResult()
  {
    ASSERT_TRUE(std::isfinite(snmg_inertia_));
    ASSERT_TRUE(std::isfinite(sg_inertia_));
    ASSERT_GE(snmg_inertia_, T{0});
    ASSERT_GE(sg_inertia_, T{0});
    ASSERT_GT(snmg_n_iter_, int64_t{0});
    ASSERT_LE(snmg_n_iter_, static_cast<int64_t>(testparams_.max_iter));

    ASSERT_GE(ari_vs_ref_, 0.94);
    ASSERT_GE(ari_vs_sg_, 0.94);
    if (testparams_.init == cuvs::cluster::kmeans::params::Array) {
      EXPECT_GE(ari_vs_sg_, 0.98);
      if (sg_inertia_ > 0) {
        EXPECT_LT(std::abs(snmg_inertia_ - sg_inertia_) / sg_inertia_, decltype(sg_inertia_){0.02});
      }
    }
    if (has_centroid_comparison_) {
      EXPECT_LT(max_centroid_rel_diff_, 0.02)
        << "SNMG vs SG centroid max relative diff = " << max_centroid_rel_diff_;
    }
  }

  raft::device_resources_snmg clique_;
  KmeansSNMGInputs<T> testparams_;
  double ari_vs_ref_            = 0;
  double ari_vs_sg_             = 0;
  T snmg_inertia_               = T{0};
  T sg_inertia_                 = T{0};
  int64_t snmg_n_iter_          = 0;
  int64_t sg_n_iter_            = 0;
  double max_centroid_rel_diff_ = 0;
  bool has_centroid_comparison_ = false;
};

// ============================================================================
// SNMG float test inputs
// ============================================================================
const std::vector<KmeansSNMGInputs<float>> snmg_inputsf = {
  // n_row, n_col, n_clusters, tol, weight_mode, device_buffer_samples, n_init[, init]
  {1000, 32, 5, 0.0001f, kmeans_weight_mode::none, 1000, 1},
  {1000, 32, 5, 0.0001f, kmeans_weight_mode::uniform, 1000, 1},
  {1000, 32, 5, 0.0001f, kmeans_weight_mode::none, 128, 1},
  {10000, 16, 10, 0.0001f, kmeans_weight_mode::none, 2000, 1},
  {10000, 16, 10, 0.0001f, kmeans_weight_mode::uniform, 2000, 1},
  {10000, 16, 10, 0.0001f, kmeans_weight_mode::none, 500, 1},
  {1001, 32, 5, 0.0001f, kmeans_weight_mode::none, 1001, 1},
  {1000,
   32,
   5,
   0.0001f,
   kmeans_weight_mode::none,
   1000,
   1,
   cuvs::cluster::kmeans::params::KMeansPlusPlus},
  {1001, 32, 5, 0.0001f, kmeans_weight_mode::none, 128, 1},
  // Non-uniform weights: exercises weight_scale = global_n / global_wt normalization
  {1000, 32, 5, 0.0001f, kmeans_weight_mode::mild_nonuniform, 1000, 1},
  {10000, 16, 10, 0.0001f, kmeans_weight_mode::mild_nonuniform, 2000, 1},
  // Extreme non-uniform weights (100:1 ratio): stresses weight normalization
  {1000, 32, 5, 0.0001f, kmeans_weight_mode::extreme_nonuniform, 1000, 1},
  // Extreme batch size = 1: single-element work buffers, many batch iterations
  {100, 8, 3, 0.001f, kmeans_weight_mode::none, 1, 1},
  // Very small dataset: some ranks may get only 2-3 rows with 4+ GPUs
  {10, 4, 3, 0.001f, kmeans_weight_mode::none, 10, 1},
  // Fewer rows than GPUs: exercises empty-rank (has_data=false) partitions on 4+ GPU systems
  {3, 4, 2, 0.001f, kmeans_weight_mode::none, 3, 1},
  // Trivial single cluster: convergence should be immediate
  {1000, 16, 1, 0.0001f, kmeans_weight_mode::none, 1000, 1},
  // Batch size > n_samples: tests per-rank clamping logic
  {1000, 32, 5, 0.0001f, kmeans_weight_mode::none, 5000, 1},
  // n_init > 1 with KMeansPlusPlus: best-of-n seed management across ranks
  {1000,
   32,
   5,
   0.0001f,
   kmeans_weight_mode::none,
   1000,
   3,
   cuvs::cluster::kmeans::params::KMeansPlusPlus},
  // max_iter saturation: algorithm should stop at max_iter without convergence
  {1000,
   32,
   5,
   0.0001f,
   kmeans_weight_mode::none,
   1000,
   1,
   cuvs::cluster::kmeans::params::Array,
   2},
};

// ============================================================================
// SNMG double test inputs
// ============================================================================
const std::vector<KmeansSNMGInputs<double>> snmg_inputsd = {
  {1000, 32, 5, 0.0001, kmeans_weight_mode::none, 1000, 1},
  {1000, 32, 5, 0.0001, kmeans_weight_mode::none, 128, 1},
  {1000, 32, 5, 0.0001, kmeans_weight_mode::uniform, 1000, 1},
  {1000, 32, 5, 0.0001, kmeans_weight_mode::mild_nonuniform, 1000, 1},
  {10000, 16, 10, 0.0001, kmeans_weight_mode::none, 2000, 1},
  {100, 8, 3, 0.001, kmeans_weight_mode::none, 1, 1},
  {10, 4, 3, 0.001, kmeans_weight_mode::none, 10, 1},
  {3, 4, 2, 0.001, kmeans_weight_mode::none, 3, 1},
  {1000, 16, 1, 0.0001, kmeans_weight_mode::none, 1000, 1},
  {1000, 32, 5, 0.0001, kmeans_weight_mode::none, 5000, 1},
  {1000,
   32,
   5,
   0.0001,
   kmeans_weight_mode::none,
   1000,
   1,
   cuvs::cluster::kmeans::params::KMeansPlusPlus},
  {1000, 32, 5, 0.0001, kmeans_weight_mode::none, 1000, 1, cuvs::cluster::kmeans::params::Array, 2},
};

typedef KmeansSNMGTest<float> KmeansSNMGTestF;
typedef KmeansSNMGTest<double> KmeansSNMGTestD;

TEST_P(KmeansSNMGTestF, Result) { checkResult(); }
TEST_P(KmeansSNMGTestD, Result) { checkResult(); }

INSTANTIATE_TEST_SUITE_P(KmeansSNMGTests, KmeansSNMGTestF, ::testing::ValuesIn(snmg_inputsf));
INSTANTIATE_TEST_SUITE_P(KmeansSNMGTests, KmeansSNMGTestD, ::testing::ValuesIn(snmg_inputsd));

}  // namespace cuvs
