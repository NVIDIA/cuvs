/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>

#include <cuvs/core/export.hpp>
#include <variant>

namespace CUVS_EXPORT cuvs {
namespace preprocessing {
namespace quantize {
namespace pq {

/**
 * @defgroup pq Product Quantizer utilities
 * @{
 */

/** Alias for the variant holding either balanced or regular k-means parameters. */
using kmeans_params_variant =
  std::variant<cuvs::cluster::kmeans::balanced_params, cuvs::cluster::kmeans::params>;

/**
 * @brief Product Quantizer parameters.
 */
struct params {
  /**
   * Simplified constructor that will build an appropriate kmeans params object.
   */
  params(uint32_t pq_bits,
         uint32_t pq_dim,
         bool use_subspaces,
         bool use_vq,
         uint32_t vq_n_centers,
         uint32_t kmeans_n_iters,
         cuvs::cluster::kmeans::kmeans_type pq_kmeans_type =
           cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
         uint32_t max_train_points_per_pq_code    = 256,
         uint32_t max_train_points_per_vq_cluster = 1024)
    : pq_bits(pq_bits),
      pq_dim(pq_dim),
      use_subspaces(use_subspaces),
      use_vq(use_vq),
      vq_n_centers(vq_n_centers),
      kmeans_params(
        pq_kmeans_type == cuvs::cluster::kmeans::kmeans_type::KMeansBalanced
          ? kmeans_params_variant{cuvs::cluster::kmeans::balanced_params{.n_iters = kmeans_n_iters}}
          : kmeans_params_variant{cuvs::cluster::kmeans::params{
              .n_clusters = 1 << pq_bits, .max_iter = static_cast<int>(kmeans_n_iters)}}),
      max_train_points_per_pq_code(max_train_points_per_pq_code),
      max_train_points_per_vq_cluster(max_train_points_per_vq_cluster)
  {
  }

  params(uint32_t pq_bits,
         uint32_t pq_dim,
         bool use_subspaces,
         bool use_vq,
         uint32_t vq_n_centers,
         kmeans_params_variant kmeans_params,
         uint32_t max_train_points_per_pq_code    = 256,
         uint32_t max_train_points_per_vq_cluster = 1024)
    : pq_bits(pq_bits),
      pq_dim(pq_dim),
      use_subspaces(use_subspaces),
      use_vq(use_vq),
      vq_n_centers(vq_n_centers),
      kmeans_params(kmeans_params),
      max_train_points_per_pq_code(max_train_points_per_pq_code),
      max_train_points_per_vq_cluster(max_train_points_per_vq_cluster)
  {
  }

  params() = default;

  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible value range: [4-16].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the faster the
   * fit/transform time, but the lower the recall.
   */
  uint32_t pq_bits = 8;
  /**
   * The dimensionality of the vector after compression by PQ.
   * When zero, dim / 4 is used as default.
   *
   * TODO: at the moment `dim` must be a multiple `pq_dim`.
   */
  uint32_t pq_dim = 0;
  /**
   * Whether to use subspaces for product quantization (PQ).
   * When true, one PQ codebook is used for each subspace. Otherwise, a single
   * PQ codebook is used.
   */
  bool use_subspaces = true;
  /**
   * Whether to use Vector Quantization (KMeans) before product quantization (PQ).
   * When true, VQ is used and PQ is trained on the residuals.
   */
  bool use_vq = false;
  /**
   * Vector Quantization (VQ) codebook size - number of "coarse cluster centers".
   * When zero, an optimal value is selected using a heuristic. (sqrt(n_rows))
   */
  uint32_t vq_n_centers = 0;
  /**
   * K-means parameters for PQ codebook training.
   *
   * Set to cuvs::cluster::kmeans::balanced_params for balanced k-means (default),
   * or cuvs::cluster::kmeans::params for regular k-means.
   * The active variant type selects the algorithm; balanced k-means tends to be faster
   * for PQ training where cluster sizes are approximately equal.
   * Only L2Expanded metric is supported. The number of clusters is always set to 1 << pq_bits.
   */
  kmeans_params_variant kmeans_params = cuvs::cluster::kmeans::balanced_params{};
  /**
   * The max number of data points to use per PQ code during PQ codebook training. Using more data
   * points per PQ code may increase the quality of PQ codebook but may also increase the build
   * time. We will use `pq_n_centers * max_train_points_per_pq_code` training
   * points to train each PQ codebook.
   */
  uint32_t max_train_points_per_pq_code = 256;
  /**
   * The max number of data points to use per VQ cluster during training.
   */
  uint32_t max_train_points_per_vq_cluster = 1024;
};

/**
 * @brief Defines and stores VPQ codebooks upon training
 *
 * @tparam T data element type
 *
 */
template <typename T>
struct quantizer {
  /** Parameters used to build this quantizer. */
  params params_quantizer;
  /** VPQ codebooks produced during training. */
  cuvs::neighbors::vpq_dataset<T, int64_t> vpq_codebooks;
};

/**
 * @brief Initializes a product quantizer to be used later for quantizing the dataset.
 *
 * The use of a pool memory resource is recommended for more consistent training performance.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * // Set the workspace memory resource to a pool with 2 GiB upper limit.
 * raft::resource::set_workspace_to_pool_resource(handle, 2 * 1024 * 1024 * 1024ull);
 * cuvs::preprocessing::quantize::pq::params params;
 * auto quantizer = cuvs::preprocessing::quantize::pq::build(handle, params, dataset);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure product quantizer, e.g. quantile
 * @param[in] dataset a row-major matrix view on device or host
 *
 * @return quantizer
 */
quantizer<float> build(raft::resources const& res,
                       const params params,
                       raft::device_matrix_view<const float, int64_t> dataset);

/** @copydoc build */
quantizer<float> build(raft::resources const& res,
                       const params params,
                       raft::host_matrix_view<const float, int64_t> dataset);

/**
 * @brief Applies quantization transform to given dataset
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::pq::params params;
 * auto quantizer = cuvs::preprocessing::quantize::pq::build(handle, params, dataset);
 * auto quantized_dim = get_quantized_dim(quantizer.params_quantizer);
 * auto quantized_dataset =
 *   raft::make_device_matrix<uint8_t, int64_t>(handle, samples, quantized_dim);
 * cuvs::preprocessing::quantize::pq::transform(handle, quantizer, dataset,
 *   quantized_dataset.view());
 *
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quant a product quantizer
 * @param[in] dataset a row-major matrix view on device or host
 * @param[out] codes_out a row-major matrix view on device containing the PQ codes
 * @param[out] vq_labels a vector view on device containing the VQ labels when VQ is
 * used, optional
 */
void transform(raft::resources const& res,
               const quantizer<float>& quant,
               raft::device_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> codes_out,
               std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels = std::nullopt);

/** @copydoc transform */
void transform(raft::resources const& res,
               const quantizer<float>& quant,
               raft::host_matrix_view<const float, int64_t> dataset,
               raft::device_matrix_view<uint8_t, int64_t> codes_out,
               std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels = std::nullopt);

/**
 * @brief Get the dimension of the quantized dataset (in bytes)
 *
 * @param[in] config product quantizer parameters
 * @return the dimension of the quantized dataset
 */
inline int64_t get_quantized_dim(const params& config)
{
  return raft::div_rounding_up_safe<int64_t>(config.pq_dim * config.pq_bits, 8);
}

/**
 * @brief Applies inverse quantization transform to given dataset
 *
 * @param[in] res raft resource
 * @param[in] quant a product quantizer
 * @param[in] pq_codes a row-major matrix view on device containing the PQ codes
 * @param[out] out a row-major matrix view on device
 * @param[in] vq_labels a vector view on device containing the VQ labels when VQ is used, optional
 *
 */
void inverse_transform(
  raft::resources const& res,
  const quantizer<float>& quant,
  raft::device_matrix_view<const uint8_t, int64_t> pq_codes,
  raft::device_matrix_view<float, int64_t> out,
  std::optional<raft::device_vector_view<const uint32_t, int64_t>> vq_labels = std::nullopt);

/** @} */  // end of group product

}  // namespace pq
}  // namespace quantize
}  // namespace preprocessing
}  // namespace CUVS_EXPORT cuvs
