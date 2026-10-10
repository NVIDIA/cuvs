/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Temporary diagnostics for the flaky SpectralClusteringTestF.Result/7 (DO NOT MERGE).

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/cluster/spectral.hpp>
#include <cuvs/preprocessing/spectral_embedding.hpp>
#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/sparse/linalg/laplacian.cuh>
#include <raft/sparse/solver/lanczos.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cudart_utils.hpp>

#include "diag_lanczos.cuh"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace cuvs {
namespace {

template <typename T>
auto to_host(const T* d, size_t n, cudaStream_t s) -> std::vector<T>
{
  std::vector<T> h(n);
  raft::update_host(h.data(), d, n, s);
  RAFT_CUDA_TRY(cudaStreamSynchronize(s));
  return h;
}

template <typename T>
auto fnv(const std::vector<T>& v) -> uint64_t
{
  uint64_t h        = 1469598103934665603ULL;
  const auto* bytes = reinterpret_cast<const unsigned char*>(v.data());
  for (size_t i = 0; i < v.size() * sizeof(T); i++) {
    h ^= bytes[i];
    h *= 1099511628211ULL;
  }
  return h;
}

auto n_components(int n, const std::vector<int>& rows, const std::vector<int>& cols) -> int
{
  std::vector<int> p(n);
  std::iota(p.begin(), p.end(), 0);
  std::function<int(int)> find = [&](int x) { return p[x] == x ? x : p[x] = find(p[x]); };
  for (size_t i = 0; i < rows.size(); i++) {
    int a = find(rows[i]), b = find(cols[i]);
    if (a != b) p[a] = b;
  }
  int c = 0;
  for (int i = 0; i < n; i++)
    if (find(i) == i) c++;
  return c;
}

struct counter {
  std::map<uint64_t, int> m;
  int add(uint64_t h)
  {
    auto [it, inserted] = m.emplace(h, static_cast<int>(m.size()));
    return it->second;
  }
};

}  // namespace

void run_spectral_diag()
{
  int repeats = 50;
  if (const char* e = std::getenv("SPECTRAL_DIAG_REPEATS")) repeats = std::atoi(e);

  const int n_samples = 1000, n_features = 30, n_clusters = 5, n_comp = 5, n_neighbors = 20,
            n_init        = 3;
  const float cluster_std = 0.3f;
  uint64_t seed           = 444ULL;
  if (const char* e = std::getenv("SPECTRAL_DIAG_SEED")) seed = std::strtoull(e, nullptr, 10);

  raft::resources handle;
  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();

  counter c_x, c_graph, c_lap, c_eig, c_emb, c_kmeans_fixed, c_labels, c_fp;
  int n_bad = 0, n_bad_fp = 0, n_missed = 0;
  std::vector<float> first_embedding;
  std::vector<int> first_r, first_c;
  std::vector<float> first_v;

  for (int r = 0; r < repeats; r++) {
    auto X           = raft::make_device_matrix<float, int>(handle, n_samples, n_features);
    auto true_labels = raft::make_device_vector<int, int>(handle, n_samples);
    raft::random::make_blobs<float, int>(X.data_handle(),
                                         true_labels.data_handle(),
                                         n_samples,
                                         n_features,
                                         n_clusters,
                                         stream,
                                         true,
                                         nullptr,
                                         nullptr,
                                         cluster_std,
                                         false,
                                         -10.0f,
                                         10.0f,
                                         seed);
    int ix = c_x.add(fnv(to_host(X.data_handle(), X.size(), stream)));

    auto graph = raft::make_device_coo_matrix<float, int, int, int>(handle, n_samples, n_samples);
    cuvs::preprocessing::spectral_embedding::params embed_params;
    embed_params.n_neighbors = n_neighbors;
    embed_params.seed        = seed;
    cuvs::preprocessing::spectral_embedding::helpers::create_connectivity_graph(
      handle, embed_params, X.view(), graph);
    int nnz   = graph.structure_view().get_nnz();
    auto g_r  = to_host(graph.structure_view().get_rows().data(), nnz, stream);
    auto g_c  = to_host(graph.structure_view().get_cols().data(), nnz, stream);
    auto g_v  = to_host(graph.get_elements().data(), nnz, stream);
    int ig    = c_graph.add(fnv(g_r) ^ (fnv(g_c) * 3) ^ (fnv(g_v) * 7));
    int ncomp = n_components(n_samples, g_r, g_c);
    if (r == 0) {
      first_r = g_r;
      first_c = g_c;
      first_v = g_v;
    } else if (g_r != first_r || g_c != first_c || g_v != first_v) {
      int nd = 0;
      for (int i = 0; i < nnz && nd < 5; i++) {
        if (g_r[i] != first_r[i] || g_c[i] != first_c[i] || g_v[i] != first_v[i]) {
          printf("[diag]   graph diff at %d: (%d,%d,%g) vs first (%d,%d,%g)\n",
                 i,
                 g_r[i],
                 g_c[i],
                 g_v[i],
                 first_r[i],
                 first_c[i],
                 first_v[i]);
          nd++;
        }
      }
    }

    // Replicate the spectral embedding internals so that the eigenvalues can be inspected.
    auto diagonal = raft::make_device_vector<float, int>(handle, n_samples);
    auto laplacian =
      raft::sparse::linalg::laplacian_normalized(handle, graph.view(), diagonal.view());
    int lnnz = laplacian.structure_view().get_nnz();
    auto lap_vals =
      raft::make_device_vector_view<float, int>(laplacian.get_elements().data(), lnnz);
    raft::linalg::map(
      handle, lap_vals, [] __device__(float x) { return -x; }, raft::make_const_mdspan(lap_vals));
    int il = c_lap.add(fnv(to_host(laplacian.get_elements().data(), lnnz, stream)) ^
                       (fnv(to_host(diagonal.data_handle(), n_samples, stream)) * 5));

    auto config           = raft::sparse::solver::lanczos_solver_config<float>();
    config.n_components   = n_comp;
    config.max_iterations = 10 * n_samples;
    config.ncv            = std::min(n_samples - n_comp, std::max(2 * n_comp + 1, 20));
    config.tolerance      = 0.0f;
    config.which          = raft::sparse::solver::LANCZOS_WHICH::LA;
    config.seed           = seed;
    auto evals            = raft::make_device_vector<float, uint32_t>(handle, n_comp);
    auto evecs =
      raft::make_device_matrix<float, uint32_t, raft::col_major>(handle, n_samples, n_comp);
    raft::sparse::solver::diag_detail::lanczos_trace().clear();
    raft::sparse::solver::diag_detail::lanczos_compute_eigenpairs<int, float>(
      handle,
      config,
      laplacian.view(),
      std::optional<raft::device_vector_view<float, uint32_t>>{},
      evals.view(),
      evecs.view());
    auto h_evals     = to_host(evals.data_handle(), n_comp, stream);
    bool missed_null = std::abs(h_evals[0]) > 1e-4f;
    if (r == 0 || missed_null) {
      printf("[diag] r=%d lanczos trace%s:\n%s",
             r,
             missed_null ? " (MISSED NULL VECTOR)" : "",
             raft::sparse::solver::diag_detail::lanczos_trace().c_str());
    }
    if (missed_null) n_missed++;
    int ie =
      c_eig.add(fnv(h_evals) ^ (fnv(to_host(evecs.data_handle(), evecs.size(), stream)) * 11));

    if (r == 0 && std::getenv("SPECTRAL_DIAG_SEED_SWEEP")) {
      int nsweep     = std::atoi(std::getenv("SPECTRAL_DIAG_SEED_SWEEP"));
      int n_not_null = 0;
      for (int sd = 0; sd < nsweep; sd++) {
        auto cfg = config;
        cfg.seed = 1000 + sd;
        raft::sparse::solver::lanczos_compute_eigenpairs<int, float>(
          handle, cfg, laplacian.view(), std::nullopt, evals.view(), evecs.view());
        auto ev  = to_host(evals.data_handle(), n_comp, stream);
        bool bad = std::abs(ev[0]) > 1e-4f;
        if (bad) n_not_null++;
        printf("[diag-sweep] seed=%d evals=[%.6g %.6g %.6g %.6g %.6g]%s\n",
               1000 + sd,
               ev[0],
               ev[1],
               ev[2],
               ev[3],
               ev[4],
               bad ? " BAD" : "");
      }
      printf(
        "[diag-sweep-summary] %d/%d seeds did not find a 5-dim null space\n", n_not_null, nsweep);
      raft::sparse::solver::lanczos_compute_eigenpairs<int, float>(
        handle, config, laplacian.view(), std::nullopt, evals.view(), evecs.view());
    }

    // The public embedding + kmeans path.
    cuvs::preprocessing::spectral_embedding::params se;
    se.n_components   = n_comp;
    se.n_neighbors    = n_neighbors;
    se.norm_laplacian = true;
    se.drop_first     = false;
    se.seed           = seed;
    se.tolerance      = 0.0f;
    auto emb_cm = raft::make_device_matrix<float, int, raft::col_major>(handle, n_samples, n_comp);
    cuvs::preprocessing::spectral_embedding::transform(handle, se, graph.view(), emb_cm.view());
    auto h_emb = to_host(emb_cm.data_handle(), emb_cm.size(), stream);
    int iemb   = c_emb.add(fnv(h_emb));
    if (r == 0) first_embedding = h_emb;

    auto run_kmeans = [&](const std::vector<float>& emb_col_major, std::vector<int>& out_labels) {
      auto e_cm = raft::make_device_matrix<float, int, raft::col_major>(handle, n_samples, n_comp);
      raft::update_device(e_cm.data_handle(), emb_col_major.data(), emb_col_major.size(), stream);
      auto e_rm = raft::make_device_matrix<float, int, raft::row_major>(handle, n_samples, n_comp);
      raft::linalg::transpose(
        handle, e_cm.data_handle(), e_rm.data_handle(), n_samples, n_comp, stream);
      cuvs::cluster::kmeans::params kp;
      kp.n_clusters          = n_clusters;
      kp.rng_state           = raft::random::RngState(seed);
      kp.n_init              = n_init;
      kp.oversampling_factor = 0.0;
      auto labels            = raft::make_device_vector<int, int>(handle, n_samples);
      float inertia;
      int n_iter;
      cuvs::cluster::kmeans::fit_predict(handle,
                                         kp,
                                         e_rm.view(),
                                         std::nullopt,
                                         std::nullopt,
                                         labels.view(),
                                         raft::make_host_scalar_view(&inertia),
                                         raft::make_host_scalar_view(&n_iter));
      out_labels = to_host(labels.data_handle(), n_samples, stream);
      return raft::stats::adjusted_rand_index(
        true_labels.data_handle(), labels.data_handle(), n_samples, stream);
    };

    std::vector<int> lab_fixed, lab;
    run_kmeans(first_embedding, lab_fixed);
    int ikf    = c_kmeans_fixed.add(fnv(lab_fixed));
    double ari = run_kmeans(h_emb, lab);
    int ilab   = c_labels.add(fnv(lab));

    // Full public fit_predict exactly as the original test does it.
    cluster::spectral::params params;
    params.n_clusters   = n_clusters;
    params.n_components = n_comp;
    params.n_neighbors  = n_neighbors;
    params.n_init       = n_init;
    params.tolerance    = 0.0f;
    params.rng_state    = raft::random::RngState(seed);
    auto fp_labels      = raft::make_device_vector<int, int>(handle, n_samples);
    cluster::spectral::fit_predict(handle, params, graph.view(), fp_labels.view());
    double ari_fp = raft::stats::adjusted_rand_index(
      true_labels.data_handle(), fp_labels.data_handle(), n_samples, stream);
    int ifp = c_fp.add(fnv(to_host(fp_labels.data_handle(), n_samples, stream)));

    if (ari < 0.7 || ari_fp < 0.7 || r == 0) {
      // Summarize the embedding per ground-truth cluster: centroid and max deviation from it.
      auto h_true = to_host(true_labels.data_handle(), n_samples, stream);
      for (int k = 0; k < n_clusters; k++) {
        std::vector<double> mean(n_comp, 0.0);
        int cnt = 0;
        for (int i = 0; i < n_samples; i++) {
          if (h_true[i] != k) continue;
          cnt++;
          for (int j = 0; j < n_comp; j++)
            mean[j] += h_emb[j * n_samples + i];
        }
        for (auto& m : mean)
          m /= std::max(cnt, 1);
        double maxdev = 0;
        for (int i = 0; i < n_samples; i++) {
          if (h_true[i] != k) continue;
          double d = 0;
          for (int j = 0; j < n_comp; j++) {
            double t = h_emb[j * n_samples + i] - mean[j];
            d += t * t;
          }
          maxdev = std::max(maxdev, std::sqrt(d));
        }
        printf(
          "[diag]   r=%d true cluster %d (n=%d): centroid=[%.4g %.4g %.4g %.4g %.4g] maxdev=%.4g\n",
          r,
          k,
          cnt,
          mean[0],
          mean[1],
          mean[2],
          mean[3],
          mean[4],
          maxdev);
      }
    }
    if (ari < 0.7) n_bad++;
    if (ari_fp < 0.7) n_bad_fp++;
    printf(
      "[diag] r=%d X=%d graph=%d(nnz=%d%s) lap=%d eig=%d evals=[%.6g %.6g %.6g %.6g %.6g] emb=%d "
      "kmeans_fixed=%d labels=%d ari=%.4f fp=%d ari_fp=%.4f\n",
      r,
      ix,
      ig,
      nnz,
      ncomp >= 0 ? (std::string(",ncomp=") + std::to_string(ncomp)).c_str() : "",
      il,
      ie,
      h_evals[0],
      h_evals[1],
      h_evals[2],
      h_evals[3],
      h_evals[4],
      iemb,
      ikf,
      ilab,
      ari,
      ifp,
      ari_fp);
  }
  printf(
    "[diag-summary] repeats=%d distinct: X=%zu graph=%zu lap=%zu eig=%zu emb=%zu kmeans_fixed=%zu "
    "labels=%zu fp=%zu; ari<0.7: %d, ari_fp<0.7: %d, lanczos missed null vector: %d\n",
    repeats,
    c_x.m.size(),
    c_graph.m.size(),
    c_lap.m.size(),
    c_eig.m.size(),
    c_emb.m.size(),
    c_kmeans_fixed.m.size(),
    c_labels.m.size(),
    c_fp.m.size(),
    n_bad,
    n_bad_fp,
    n_missed);
}

TEST(SpectralClusteringDiag, Result7) { run_spectral_diag(); }

}  // namespace cuvs
