/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace cuvs {

// ============================================================================
// Blob inputs
// ============================================================================

/**
 * @brief Aggregate of synthetic blob inputs used by the cuVS k-means test
 *        suite.
 *
 * `d_X` and `d_labels_ref` are produced by `raft::random::make_blobs`. `h_X`
 * is an optional host mirror sized for `int64_t` indexing (matching the host
 * APIs exercised by the batched and SNMG fixtures).
 */
template <typename T>
struct kmeans_blob_inputs {
  raft::device_matrix<T, int> d_X;
  raft::device_vector<int, int> d_labels_ref;
  std::optional<raft::host_matrix<T, int64_t>> h_X;
};

/**
 * @brief Generate well-separated synthetic blob data for k-means tests.
 *
 * Centralizes the parameters (cluster spread, center box, seed) shared across
 * the single-GPU, batched, and SNMG tests so any change to the reference
 * distribution applies everywhere.
 *
 * When `with_host_mirror` is true, the helper synchronizes the stream before
 * returning so the host mirror is safe to read immediately. Callers that only
 * need device data can pass `false` to skip the host allocation, copy, and
 * sync.
 *
 * @tparam T               Floating-point element type for X.
 *
 * @param[in] handle           RAFT resources used to allocate device memory.
 * @param[in] n_samples        Number of rows in X.
 * @param[in] n_features       Number of columns in X.
 * @param[in] n_clusters       Number of cluster centers.
 * @param[in] with_host_mirror If true (default), also allocate `h_X` and
 *                             populate it from `d_X`.
 * @param[in] seed             RNG seed; defaults to the value used by the
 *                             cuVS k-means test suite.
 */
template <typename T>
kmeans_blob_inputs<T> make_kmeans_blob_inputs(raft::resources const& handle,
                                              int n_samples,
                                              int n_features,
                                              int n_clusters,
                                              bool with_host_mirror = true,
                                              std::uint64_t seed    = 1234)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  auto d_X          = raft::make_device_matrix<T, int>(handle, n_samples, n_features);
  auto d_labels_ref = raft::make_device_vector<int, int>(handle, n_samples);

  raft::random::make_blobs<T, int>(d_X.data_handle(),
                                   d_labels_ref.data_handle(),
                                   n_samples,
                                   n_features,
                                   n_clusters,
                                   stream,
                                   /* row_major          */ true,
                                   /* centers            */ nullptr,
                                   /* cluster_std        */ nullptr,
                                   /* cluster_std_scalar */ T(1.0),
                                   /* shuffle            */ false,
                                   /* center_box_min     */ static_cast<T>(-10.0f),
                                   /* center_box_max     */ static_cast<T>(10.0f),
                                   seed);

  std::optional<raft::host_matrix<T, int64_t>> h_X;
  if (with_host_mirror) {
    h_X.emplace(raft::make_host_matrix<T, int64_t>(n_samples, n_features));
    raft::update_host(h_X->data_handle(),
                      d_X.data_handle(),
                      static_cast<std::size_t>(n_samples) * n_features,
                      stream);
    raft::resource::sync_stream(handle, stream);
  }

  return {std::move(d_X), std::move(d_labels_ref), std::move(h_X)};
}

// ============================================================================
// Sample-weight modes
// ============================================================================

/**
 * @brief Sample-weight modes shared across the cuVS k-means test suite.
 *
 * The per-index value formula lives in `kmeans_test_weight_value`. Sharing it
 * keeps host- and device-side weight vectors bitwise identical for the same
 * mode, so cross-test comparisons (single-GPU vs batched vs SNMG) remain
 * well-defined under non-uniform weighting.
 */
enum class kmeans_weight_mode { none, uniform, mild_nonuniform, extreme_nonuniform };

/**
 * @brief Per-sample weight value for a given mode.
 *
 * @tparam T     Element type of the weight vector.
 * @tparam IdxT  Index type (templated to allow either `int` or `int64_t`,
 *               depending on the caller's view extents).
 */
template <typename T, typename IdxT>
constexpr T kmeans_test_weight_value(IdxT i, kmeans_weight_mode mode)
{
  switch (mode) {
    case kmeans_weight_mode::uniform: return T(1);
    case kmeans_weight_mode::mild_nonuniform: return T(1) + T(i % 5);
    case kmeans_weight_mode::extreme_nonuniform: return (i % 10 == 0) ? T(100) : T(1);
    case kmeans_weight_mode::none: [[fallthrough]];
    default: return T(0);
  }
}

/**
 * @brief Fill a host weight vector according to `mode`.
 *
 * Callers are expected to gate this on `mode != kmeans_weight_mode::none` and
 * skip allocating the buffer when no weights are wanted.
 */
template <typename T, typename IdxT>
void fill_kmeans_test_weights(raft::host_vector_view<T, IdxT> w, kmeans_weight_mode mode)
{
  for (IdxT i = 0; i < w.extent(0); ++i)
    w(i) = kmeans_test_weight_value<T, IdxT>(i, mode);
}

/**
 * @brief Fill a device weight vector according to `mode`.
 *
 * Builds the weights on host using the same formula as the host overload and
 * copies them to the device buffer in one shot. The host scratch path is
 * intentional: it guarantees that `mode`'s device-side values match the host
 * overload bit-for-bit, so SNMG (host-input) and batched (device-input) runs
 * exercise the same numerical contract.
 *
 * Synchronizes the stream before returning so the local host scratch buffer
 * cannot be destroyed while the asynchronous H2D copy is still in flight.
 */
template <typename T, typename IdxT>
void fill_kmeans_test_weights(raft::resources const& handle,
                              raft::device_vector_view<T, IdxT> w,
                              kmeans_weight_mode mode)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  std::vector<T> tmp(static_cast<std::size_t>(w.extent(0)));
  for (std::size_t i = 0; i < tmp.size(); ++i)
    tmp[i] = kmeans_test_weight_value<T, IdxT>(static_cast<IdxT>(i), mode);
  raft::update_device(w.data_handle(), tmp.data(), tmp.size(), stream);
  raft::resource::sync_stream(handle, stream);
}

}  // namespace cuvs
