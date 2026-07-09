/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// This file contains the multi-node multi-GPU (MNMG) kmeans tests. They depend on the distributed
// (UCXX) dependencies pulled in through raft's std_comms and are only compiled when the build is
// configured with BUILD_MNMG_TESTS=ON.

#include "../test_utils.cuh"
#include "kmeans_test_blobs.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/common/nccl_macros.hpp>
#include <raft/comms/std_comms.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

namespace cuvs {

namespace {

constexpr int kMaxRanksForNcclTest = 4;

template <typename T>
int run_mg_fit_omp(const std::vector<int>& device_ids,
                   const cuvs::cluster::kmeans::params& kp,
                   const T* h_X,
                   const std::vector<T>* h_w,
                   const std::vector<T>* h_initial_centroids,
                   int n_samples,
                   int n_features,
                   int n_clusters,
                   int partitions_per_rank,
                   bool host_data,
                   std::vector<T>& out_h_centroids,
                   T& out_inertia,
                   int64_t& out_n_iter)
{
  const int num_ranks = static_cast<int>(device_ids.size());

  int current_device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&current_device));

  std::vector<std::unique_ptr<raft::device_resources>> rank_resources;
  rank_resources.reserve(static_cast<size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    RAFT_CUDA_TRY(cudaSetDevice(device_ids[static_cast<size_t>(r)]));
    rank_resources.push_back(std::make_unique<raft::device_resources>());
  }

  std::vector<ncclComm_t> nccl_comms(static_cast<size_t>(num_ranks), nullptr);
  ncclUniqueId nccl_id;
  RAFT_NCCL_TRY(ncclGetUniqueId(&nccl_id));
  RAFT_NCCL_TRY(ncclGroupStart());
  for (int r = 0; r < num_ranks; ++r) {
    RAFT_CUDA_TRY(cudaSetDevice(device_ids[static_cast<size_t>(r)]));
    RAFT_NCCL_TRY(ncclCommInitRank(&nccl_comms[static_cast<size_t>(r)], num_ranks, nccl_id, r));
  }
  RAFT_NCCL_TRY(ncclGroupEnd());

  for (int r = 0; r < num_ranks; ++r) {
    RAFT_CUDA_TRY(cudaSetDevice(device_ids[static_cast<size_t>(r)]));
    raft::comms::build_comms_nccl_only(rank_resources[static_cast<size_t>(r)].get(),
                                       nccl_comms[static_cast<size_t>(r)],
                                       num_ranks,
                                       r);
  }

  RAFT_CUDA_TRY(cudaSetDevice(current_device));

  partitions_per_rank = std::max(1, partitions_per_rank);
  out_h_centroids.assign(static_cast<size_t>(n_clusters) * n_features, T{0});
  T inertia          = T{0};
  int64_t n_iter     = 0;
  int actual_threads = 0;

#pragma omp parallel num_threads(num_ranks)
  {
#pragma omp single
    {
      actual_threads = omp_get_num_threads();
    }
    if (actual_threads == num_ranks) {
      const int r = omp_get_thread_num();
      RAFT_CUDA_TRY(cudaSetDevice(device_ids[static_cast<size_t>(r)]));
      auto const& rank_res = *rank_resources[static_cast<size_t>(r)];
      auto rank_stream     = raft::resource::get_cuda_stream(rank_res);

      const int base     = n_samples / num_ranks;
      const int rem      = n_samples % num_ranks;
      const int rank_off = r * base + std::min(r, rem);
      const int rank_n   = base + (r < rem ? 1 : 0);

      const int part_base = rank_n / partitions_per_rank;
      const int part_rem  = rank_n % partitions_per_rank;

      rmm::device_uvector<T> d_rank_centroids(static_cast<size_t>(n_clusters) * n_features,
                                              rank_stream);
      if (h_initial_centroids != nullptr) {
        raft::update_device(d_rank_centroids.data(),
                            h_initial_centroids->data(),
                            d_rank_centroids.size(),
                            rank_stream);
      }

      auto d_rank_centroids_view =
        raft::make_device_matrix_view<T, int64_t>(d_rank_centroids.data(), n_clusters, n_features);

      T local_inertia      = T{0};
      int64_t local_n_iter = 0;

      if (!host_data) {
        // Device-partition path: each partition is its own rmm::device_uvector.
        std::vector<rmm::device_uvector<T>> d_X_parts;
        d_X_parts.reserve(static_cast<size_t>(partitions_per_rank));
        std::optional<std::vector<rmm::device_uvector<T>>> d_w_parts;
        if (h_w != nullptr) {
          d_w_parts.emplace();
          d_w_parts->reserve(static_cast<size_t>(partitions_per_rank));
        }

        std::vector<raft::device_matrix_view<const T, int64_t>> X_parts;
        X_parts.reserve(static_cast<size_t>(partitions_per_rank));
        std::optional<std::vector<raft::device_vector_view<const T, int64_t>>> sw_parts;
        if (h_w != nullptr) {
          sw_parts.emplace();
          sw_parts->reserve(static_cast<size_t>(partitions_per_rank));
        }

        for (int p = 0; p < partitions_per_rank; ++p) {
          int p_off  = p * part_base + std::min(p, part_rem);
          int p_rows = part_base + (p < part_rem ? 1 : 0);

          d_X_parts.emplace_back(static_cast<size_t>(p_rows) * n_features, rank_stream);
          if (p_rows > 0) {
            raft::update_device(d_X_parts.back().data(),
                                h_X + static_cast<size_t>(rank_off + p_off) * n_features,
                                d_X_parts.back().size(),
                                rank_stream);
          }
          X_parts.push_back(raft::make_device_matrix_view<const T, int64_t>(
            d_X_parts.back().data(), p_rows, n_features));

          if (d_w_parts.has_value()) {
            d_w_parts->emplace_back(static_cast<size_t>(p_rows), rank_stream);
            if (p_rows > 0) {
              raft::update_device(d_w_parts->back().data(),
                                  h_w->data() + rank_off + p_off,
                                  static_cast<size_t>(p_rows),
                                  rank_stream);
            }
            sw_parts->push_back(
              raft::make_device_vector_view<const T, int64_t>(d_w_parts->back().data(), p_rows));
          }
        }

        cuvs::cluster::kmeans::mg::fit(rank_res,
                                       kp,
                                       X_parts,
                                       sw_parts,
                                       d_rank_centroids_view,
                                       raft::make_host_scalar_view(&local_inertia),
                                       raft::make_host_scalar_view(&local_n_iter));
      } else {
        // Host-partition path: each partition is a std::vector<T> on host, views
        // are host_*_view, and `mg::fit` streams batches to device internally.
        std::vector<std::vector<T>> h_X_parts_buf;
        h_X_parts_buf.reserve(static_cast<size_t>(partitions_per_rank));
        std::optional<std::vector<std::vector<T>>> h_w_parts_buf;
        if (h_w != nullptr) {
          h_w_parts_buf.emplace();
          h_w_parts_buf->reserve(static_cast<size_t>(partitions_per_rank));
        }

        std::vector<raft::host_matrix_view<const T, int64_t>> X_parts;
        X_parts.reserve(static_cast<size_t>(partitions_per_rank));
        std::optional<std::vector<raft::host_vector_view<const T, int64_t>>> sw_parts;
        if (h_w != nullptr) {
          sw_parts.emplace();
          sw_parts->reserve(static_cast<size_t>(partitions_per_rank));
        }

        for (int p = 0; p < partitions_per_rank; ++p) {
          int p_off  = p * part_base + std::min(p, part_rem);
          int p_rows = part_base + (p < part_rem ? 1 : 0);

          h_X_parts_buf.emplace_back(static_cast<size_t>(p_rows) * n_features, T{0});
          if (p_rows > 0) {
            std::copy_n(h_X + static_cast<size_t>(rank_off + p_off) * n_features,
                        static_cast<size_t>(p_rows) * n_features,
                        h_X_parts_buf.back().data());
          }
          X_parts.push_back(raft::make_host_matrix_view<const T, int64_t>(
            h_X_parts_buf.back().data(), p_rows, n_features));

          if (h_w_parts_buf.has_value()) {
            h_w_parts_buf->emplace_back(static_cast<size_t>(p_rows), T{0});
            if (p_rows > 0) {
              std::copy_n(h_w->data() + rank_off + p_off,
                          static_cast<size_t>(p_rows),
                          h_w_parts_buf->back().data());
            }
            sw_parts->push_back(
              raft::make_host_vector_view<const T, int64_t>(h_w_parts_buf->back().data(), p_rows));
          }
        }

        cuvs::cluster::kmeans::mg::fit(rank_res,
                                       kp,
                                       X_parts,
                                       sw_parts,
                                       d_rank_centroids_view,
                                       raft::make_host_scalar_view(&local_inertia),
                                       raft::make_host_scalar_view(&local_n_iter));
      }

      // Ensure all ranks have completed the fit before writing outputs.
      raft::resource::sync_stream(rank_res);
#pragma omp barrier
      if (r == 0) {
        // Copy rank 0's outputs for comparison.
        raft::update_host(
          out_h_centroids.data(), d_rank_centroids.data(), out_h_centroids.size(), rank_stream);
        raft::resource::sync_stream(rank_res);
        inertia = local_inertia;
        n_iter  = local_n_iter;
      }
    }
  }

  for (int r = 0; r < num_ranks; ++r) {
    RAFT_CUDA_TRY(cudaSetDevice(device_ids[static_cast<size_t>(r)]));
    rank_resources[static_cast<size_t>(r)].reset();
  }
  rank_resources.clear();

  RAFT_NCCL_TRY(ncclGroupStart());
  for (int r = 0; r < num_ranks; ++r) {
    RAFT_CUDA_TRY(cudaSetDevice(device_ids[static_cast<size_t>(r)]));
    auto comm = nccl_comms[static_cast<size_t>(r)];
    if (comm != nullptr) { RAFT_NCCL_TRY(ncclCommDestroy(comm)); }
  }
  RAFT_NCCL_TRY(ncclGroupEnd());
  RAFT_CUDA_TRY(cudaSetDevice(current_device));

  out_inertia = inertia;
  out_n_iter  = n_iter;
  return actual_threads;
}

}  // namespace

template <typename T>
struct KmeansMGNcclInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  kmeans_weight_mode weight_mode;
  int64_t streaming_batch_size;
  int n_init;
  int partitions_per_rank;
  cuvs::cluster::kmeans::params::InitMethod init = cuvs::cluster::kmeans::params::Array;
  int max_iter                                   = 20;
  // When true, partitions are allocated on the host and the host vector-of-mdspan `mg::fit`
  // overload is invoked from inside the OMP region.
  bool host_data = false;
};

template <typename T>
class KmeansMGNcclTest : public ::testing::TestWithParam<KmeansMGNcclInputs<T>> {
 protected:
  KmeansMGNcclTest() : device_ids_(make_nccl_test_device_ids()) {}

  static std::vector<int> make_nccl_test_device_ids()
  {
    int num_devices = 0;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&num_devices));
    int n = std::min(num_devices, kMaxRanksForNcclTest);
    std::vector<int> ids(n);
    for (int i = 0; i < n; ++i) {
      ids[i] = i;
    }
    return ids;
  }

  void runTest()
  {
    testparams_ = ::testing::TestWithParam<KmeansMGNcclInputs<T>>::GetParam();

    const int num_ranks = static_cast<int>(device_ids_.size());
    if (num_ranks < 1) { GTEST_SKIP() << "No CUDA devices available."; }

    const int n_samples           = testparams_.n_row;
    const int n_features          = testparams_.n_col;
    const int n_clusters          = testparams_.n_clusters;
    const int partitions_per_rank = std::max(1, testparams_.partitions_per_rank);
    const bool has_weights        = testparams_.weight_mode != kmeans_weight_mode::none;

    RAFT_CUDA_TRY(cudaSetDevice(0));
    raft::resources gen_handle;
    auto gen_stream = raft::resource::get_cuda_stream(gen_handle);
    auto bi         = make_kmeans_blob_inputs<T>(gen_handle, n_samples, n_features, n_clusters);

    std::vector<int> h_labels_ref(n_samples);
    raft::update_host(h_labels_ref.data(), bi.d_labels_ref.data_handle(), n_samples, gen_stream);
    raft::resource::sync_stream(gen_handle);

    const T* h_X = bi.h_X->data_handle();

    std::vector<T> h_w;
    if (has_weights) {
      h_w.resize(n_samples);
      for (int i = 0; i < n_samples; ++i) {
        h_w[i] = kmeans_test_weight_value<T, int>(i, testparams_.weight_mode);
      }
    }

    std::vector<T> h_initial_centroids;
    if (testparams_.init == cuvs::cluster::kmeans::params::Array) {
      h_initial_centroids.assign(static_cast<size_t>(n_clusters) * n_features, T(0));
      std::vector<int> counts(n_clusters, 0);
      for (int i = 0; i < n_samples; ++i) {
        int c = h_labels_ref[i];
        counts[c]++;
        for (int j = 0; j < n_features; ++j) {
          h_initial_centroids[static_cast<size_t>(c) * n_features + j] +=
            h_X[static_cast<size_t>(i) * n_features + j];
        }
      }
      for (int c = 0; c < n_clusters; ++c) {
        if (counts[c] > 0) {
          for (int j = 0; j < n_features; ++j) {
            h_initial_centroids[static_cast<size_t>(c) * n_features + j] /= T(counts[c]);
          }
        }
      }
    }

    cuvs::cluster::kmeans::params kp;
    kp.n_clusters           = n_clusters;
    kp.tol                  = testparams_.tol;
    kp.max_iter             = testparams_.max_iter;
    kp.n_init               = testparams_.n_init;
    kp.rng_state.seed       = 42;
    kp.init                 = testparams_.init;
    kp.streaming_batch_size = testparams_.streaming_batch_size;

    std::vector<T> h_mg_centroids;
    T mg_inertia      = T{0};
    int64_t mg_n_iter = 0;

    const std::vector<T>* h_w_ptr = has_weights ? &h_w : nullptr;
    const std::vector<T>* h_init_ptr =
      testparams_.init == cuvs::cluster::kmeans::params::Array ? &h_initial_centroids : nullptr;
    const int actual_threads = run_mg_fit_omp<T>(device_ids_,
                                                 kp,
                                                 h_X,
                                                 h_w_ptr,
                                                 h_init_ptr,
                                                 n_samples,
                                                 n_features,
                                                 n_clusters,
                                                 partitions_per_rank,
                                                 testparams_.host_data,
                                                 h_mg_centroids,
                                                 mg_inertia,
                                                 mg_n_iter);
    ASSERT_EQ(actual_threads, num_ranks)
      << "MG NCCL test required " << num_ranks << " OMP threads but got " << actual_threads;

    RAFT_CUDA_TRY(cudaSetDevice(0));
    raft::resources sg_handle;
    auto sg_stream = raft::resource::get_cuda_stream(sg_handle);

    rmm::device_uvector<T> d_X_full(static_cast<size_t>(n_samples) * n_features, sg_stream);
    raft::update_device(d_X_full.data(), h_X, d_X_full.size(), sg_stream);
    auto X_full_view =
      raft::make_device_matrix_view<const T, int>(d_X_full.data(), n_samples, n_features);

    std::optional<raft::device_vector_view<const T, int>> sw_full = std::nullopt;
    rmm::device_uvector<T> d_w_full(0, sg_stream);
    if (has_weights) {
      d_w_full.resize(n_samples, sg_stream);
      raft::update_device(d_w_full.data(), h_w.data(), n_samples, sg_stream);
      sw_full = raft::make_device_vector_view<const T, int>(d_w_full.data(), n_samples);
    }

    rmm::device_uvector<T> d_sg_centroids(static_cast<size_t>(n_clusters) * n_features, sg_stream);
    if (testparams_.init == cuvs::cluster::kmeans::params::Array) {
      raft::update_device(
        d_sg_centroids.data(), h_initial_centroids.data(), d_sg_centroids.size(), sg_stream);
    }

    cuvs::cluster::kmeans::params skp;
    skp.n_clusters           = n_clusters;
    skp.tol                  = testparams_.tol;
    skp.max_iter             = testparams_.max_iter;
    skp.n_init               = testparams_.n_init;
    skp.rng_state.seed       = 42;
    skp.init                 = testparams_.init;
    skp.streaming_batch_size = testparams_.streaming_batch_size;

    T sg_inertia  = T{0};
    int sg_n_iter = 0;
    cuvs::cluster::kmeans::fit(
      sg_handle,
      skp,
      X_full_view,
      sw_full,
      raft::make_device_matrix_view<T, int>(d_sg_centroids.data(), n_clusters, n_features),
      raft::make_host_scalar_view(&sg_inertia),
      raft::make_host_scalar_view(&sg_n_iter));

    rmm::device_uvector<T> d_mg_centroids(static_cast<size_t>(n_clusters) * n_features, sg_stream);
    raft::update_device(
      d_mg_centroids.data(), h_mg_centroids.data(), d_mg_centroids.size(), sg_stream);

    rmm::device_uvector<int> d_labels_mg(n_samples, sg_stream);
    rmm::device_uvector<int> d_labels_sg(n_samples, sg_stream);
    rmm::device_uvector<int> d_labels_ref(n_samples, sg_stream);
    raft::update_device(d_labels_ref.data(), h_labels_ref.data(), n_samples, sg_stream);

    cuvs::cluster::kmeans::params pred_params;
    pred_params.n_clusters = n_clusters;

    T pred_inertia_mg = T{0};
    cuvs::cluster::kmeans::predict(
      sg_handle,
      pred_params,
      X_full_view,
      std::nullopt,
      raft::make_device_matrix_view<const T, int>(d_mg_centroids.data(), n_clusters, n_features),
      raft::make_device_vector_view<int, int>(d_labels_mg.data(), n_samples),
      true,
      raft::make_host_scalar_view(&pred_inertia_mg));

    T pred_inertia_sg = T{0};
    cuvs::cluster::kmeans::predict(
      sg_handle,
      pred_params,
      X_full_view,
      std::nullopt,
      raft::make_device_matrix_view<const T, int>(d_sg_centroids.data(), n_clusters, n_features),
      raft::make_device_vector_view<int, int>(d_labels_sg.data(), n_samples),
      true,
      raft::make_host_scalar_view(&pred_inertia_sg));

    ari_vs_ref_ = raft::stats::adjusted_rand_index(
      d_labels_ref.data(), d_labels_mg.data(), n_samples, sg_stream);
    ari_vs_sg_ = raft::stats::adjusted_rand_index(
      d_labels_sg.data(), d_labels_mg.data(), n_samples, sg_stream);

    mg_inertia_ = mg_inertia;
    mg_n_iter_  = mg_n_iter;
    sg_inertia_ = sg_inertia;
    num_ranks_  = num_ranks;

    if (ari_vs_ref_ < 0.94 || ari_vs_sg_ < 0.94) {
      std::cout << "MG NCCL KMeans: ARI vs ref = " << ari_vs_ref_ << ", ARI vs SG = " << ari_vs_sg_
                << ", num_ranks = " << num_ranks
                << ", partitions_per_rank = " << partitions_per_rank
                << ", mg_inertia = " << mg_inertia << ", sg_inertia = " << sg_inertia
                << ", mg_n_iter = " << mg_n_iter << ", sg_n_iter = " << sg_n_iter << std::endl;
    }
  }

  void SetUp() override { runTest(); }

  void checkResult()
  {
    ASSERT_TRUE(std::isfinite(mg_inertia_));
    ASSERT_TRUE(std::isfinite(sg_inertia_));
    ASSERT_GE(mg_inertia_, T{0});
    ASSERT_GE(sg_inertia_, T{0});
    ASSERT_GT(mg_n_iter_, int64_t{0});
    ASSERT_LE(mg_n_iter_, static_cast<int64_t>(testparams_.max_iter));

    ASSERT_GE(ari_vs_ref_, 0.94);
    ASSERT_GE(ari_vs_sg_, 0.94);
    if (testparams_.init == cuvs::cluster::kmeans::params::Array) {
      EXPECT_GE(ari_vs_sg_, 0.98);
      if (sg_inertia_ > 0) {
        EXPECT_LT(std::abs(mg_inertia_ - sg_inertia_) / sg_inertia_, decltype(sg_inertia_){0.02});
      }
    }
  }

  std::vector<int> device_ids_;
  KmeansMGNcclInputs<T> testparams_;
  double ari_vs_ref_ = 0;
  double ari_vs_sg_  = 0;
  T mg_inertia_      = T{0};
  T sg_inertia_      = T{0};
  int64_t mg_n_iter_ = 0;
  int num_ranks_     = 0;
};

// ============================================================================
// NCCL float test inputs
// ============================================================================
const std::vector<KmeansMGNcclInputs<float>> mg_nccl_inputsf = {
  // n_row, n_col, n_clusters, tol, weight_mode, streaming_batch_size, n_init,
  // partitions_per_rank[, init[, max_iter]]
  {1000, 32, 5, 0.0001f, kmeans_weight_mode::none, 1000, 1, 1},
  {1000, 32, 5, 0.0001f, kmeans_weight_mode::none, 1000, 1, 2},
  // K=4 with batch < partition rows: partition and batch loops both iterate.
  {1000, 32, 5, 0.0001f, kmeans_weight_mode::none, 128, 1, 4},
  {1000, 32, 5, 0.0001f, kmeans_weight_mode::mild_nonuniform, 1000, 1, 3},
  {10000, 16, 10, 0.0001f, kmeans_weight_mode::none, 500, 1, 3},
  {1000,
   32,
   5,
   0.0001f,
   kmeans_weight_mode::none,
   1000,
   1,
   2,
   cuvs::cluster::kmeans::params::KMeansPlusPlus},
  // Empty-partition coverage: <=3 rows/rank split into 4 partitions.
  {10, 4, 3, 0.001f, kmeans_weight_mode::none, 10, 1, 4},
  {10000, 16, 10, 0.0001f, kmeans_weight_mode::mild_nonuniform, 1024, 1, 5},
  {2000,
   16,
   8,
   0.0001f,
   kmeans_weight_mode::none,
   2000,
   1,
   3,
   cuvs::cluster::kmeans::params::KMeansPlusPlus},
  // Host partitions
  {1000,
   32,
   5,
   0.0001f,
   kmeans_weight_mode::none,
   256,
   1,
   3,
   cuvs::cluster::kmeans::params::Array,
   20,
   true},
  // Host-partition with weights + small batch size (multiple batches per
  // partition stress the streaming + per-partition offset interaction).
  {2000,
   16,
   6,
   0.0001f,
   kmeans_weight_mode::mild_nonuniform,
   128,
   1,
   4,
   cuvs::cluster::kmeans::params::Array,
   20,
   true},
  // Host-partition KMeans++ (host out-of-core sample-to-root init path) with 2 parts/rank.
  {1000,
   16,
   5,
   0.0001f,
   kmeans_weight_mode::none,
   512,
   1,
   2,
   cuvs::cluster::kmeans::params::KMeansPlusPlus,
   20,
   true},
};

// ============================================================================
// NCCL double test inputs
// ============================================================================
const std::vector<KmeansMGNcclInputs<double>> mg_nccl_inputsd = {
  {1000, 32, 5, 0.0001, kmeans_weight_mode::none, 1000, 1, 1},
  {1000, 32, 5, 0.0001, kmeans_weight_mode::none, 1000, 1, 2},
  {1000, 32, 5, 0.0001, kmeans_weight_mode::mild_nonuniform, 1000, 1, 3},
  {10000, 16, 10, 0.0001, kmeans_weight_mode::none, 500, 1, 3},
  {1000,
   32,
   5,
   0.0001,
   kmeans_weight_mode::none,
   1000,
   1,
   2,
   cuvs::cluster::kmeans::params::KMeansPlusPlus},
  {10000, 16, 10, 0.0001, kmeans_weight_mode::mild_nonuniform, 1024, 1, 5},
  // Host-partition multi-rank with weights and small batches.
  {2000,
   16,
   6,
   0.0001,
   kmeans_weight_mode::mild_nonuniform,
   128,
   1,
   4,
   cuvs::cluster::kmeans::params::Array,
   20,
   true},
  // Host-partition KMeans++ with 2 parts/rank.
  {1000,
   16,
   5,
   0.0001,
   kmeans_weight_mode::none,
   512,
   1,
   2,
   cuvs::cluster::kmeans::params::KMeansPlusPlus,
   20,
   true},
};

typedef KmeansMGNcclTest<float> KmeansMGNcclTestF;
typedef KmeansMGNcclTest<double> KmeansMGNcclTestD;

TEST_P(KmeansMGNcclTestF, Result) { checkResult(); }
TEST_P(KmeansMGNcclTestD, Result) { checkResult(); }

INSTANTIATE_TEST_SUITE_P(KmeansMGNcclTests,
                         KmeansMGNcclTestF,
                         ::testing::ValuesIn(mg_nccl_inputsf));
INSTANTIATE_TEST_SUITE_P(KmeansMGNcclTests,
                         KmeansMGNcclTestD,
                         ::testing::ValuesIn(mg_nccl_inputsd));

template <typename T>
class KmeansMGOversamplingTest : public ::testing::Test {
 protected:
  KmeansMGOversamplingTest() : device_ids_(make_nccl_test_device_ids()) {}

  static std::vector<int> make_nccl_test_device_ids()
  {
    int num_devices = 0;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&num_devices));
    int n = std::min(num_devices, kMaxRanksForNcclTest);
    std::vector<int> ids(n);
    for (int i = 0; i < n; ++i) {
      ids[i] = i;
    }
    return ids;
  }

  void run_test_body()
  {
    const int num_ranks = static_cast<int>(device_ids_.size());
    if (num_ranks < 1) { GTEST_SKIP() << "No CUDA devices available."; }

    constexpr int n_samples  = 2000;
    constexpr int n_features = 2;
    constexpr int n_clusters = 8;

    RAFT_CUDA_TRY(cudaSetDevice(0));
    raft::resources gen_handle;
    auto bi      = make_kmeans_blob_inputs<T>(gen_handle, n_samples, n_features, n_clusters);
    const T* h_X = bi.h_X->data_handle();
    std::vector<T> h_X_vec(h_X, h_X + static_cast<size_t>(n_samples) * n_features);

    T inertia_clamped      = T{0};
    T inertia_unit         = T{0};
    int64_t n_iter_clamped = 0;
    int64_t n_iter_unit    = 0;
    run_fit(h_X_vec, n_samples, n_features, n_clusters, inertia_clamped, n_iter_clamped, 0.0);
    run_fit(h_X_vec, n_samples, n_features, n_clusters, inertia_unit, n_iter_unit, 1.0);

    ASSERT_TRUE(std::isfinite(inertia_clamped));
    ASSERT_TRUE(std::isfinite(inertia_unit));
    ASSERT_GT(inertia_clamped, T{0});
    ASSERT_GT(inertia_unit, T{0});

    const double rel = std::abs(static_cast<double>(inertia_clamped - inertia_unit)) /
                       static_cast<double>(inertia_unit);
    EXPECT_LT(rel, 0.01) << "oversampling_factor=0 and oversampling_factor=1.0 produced different "
                            "inertias rel diff = "
                         << rel << ", " << inertia_clamped << " vs " << inertia_unit;
  }

  void run_fit(const std::vector<T>& h_X,
               int n_samples,
               int n_features,
               int n_clusters,
               T& inertia,
               int64_t& n_iter,
               double oversampling_factor = 1.0)
  {
    cuvs::cluster::kmeans::params kp;
    kp.n_clusters           = n_clusters;
    kp.tol                  = T(1e-4);
    kp.max_iter             = 30;
    kp.n_init               = 1;
    kp.rng_state.seed       = 42;
    kp.init                 = cuvs::cluster::kmeans::params::KMeansPlusPlus;
    kp.streaming_batch_size = n_samples;
    kp.oversampling_factor  = oversampling_factor;

    std::vector<T> h_centroids;
    const int actual_threads = run_mg_fit_omp<T>(device_ids_,
                                                 kp,
                                                 h_X.data(),
                                                 /*h_w=*/nullptr,
                                                 /*h_initial_centroids=*/nullptr,
                                                 n_samples,
                                                 n_features,
                                                 n_clusters,
                                                 /*partitions_per_rank=*/1,
                                                 /*host_data=*/false,
                                                 h_centroids,
                                                 inertia,
                                                 n_iter);
    ASSERT_EQ(actual_threads, static_cast<int>(device_ids_.size()));
  }

  std::vector<int> device_ids_;
};

typedef KmeansMGOversamplingTest<float> KmeansMGOversamplingTestF;
typedef KmeansMGOversamplingTest<double> KmeansMGOversamplingTestD;

TEST_F(KmeansMGOversamplingTestF, ZeroEquivalentToOne) { run_test_body(); }
TEST_F(KmeansMGOversamplingTestD, ZeroEquivalentToOne) { run_test_body(); }

}  // namespace cuvs
