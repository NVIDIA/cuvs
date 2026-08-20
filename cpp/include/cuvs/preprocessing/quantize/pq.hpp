/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/util/cuda_data_type.hpp>

#include <cuda_runtime.h>
#include <cuvs/core/export.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <type_traits>
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
  cuvs::neighbors::device_vpq_dataset<T, int64_t> vpq_codebooks;
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

namespace detail {

// Trains from `n_rows` rows of `stride` elements each, whether they are device-accessible or
// host-resident; the residency is detected from the pointer.
//
// NB: the element type is erased into `dtype` so that this stays a plain function: under hidden
// default visibility, an instantiation cannot be exported from the shared library when one of its
// template arguments (`half`, or any mdspan type) is itself hidden, because the visibility of an
// instantiation is capped by that of its template arguments.
[[nodiscard]] CUVS_EXPORT cuvs::neighbors::device_vpq_dataset<half, int64_t> vpq_train_from_rows(
  raft::resources const& res,
  cuvs::neighbors::vpq_params const& params,
  void const* src_ptr,
  cudaDataType_t dtype,
  int64_t n_rows,
  int64_t dim,
  int64_t stride);

}  // namespace detail

/**
 * @brief Train VPQ storage (codebooks + encoded rows) from a row-major mdspan/mdarray/dataset.
 *
 * Accepts either a row-major mdspan with `value_type`, `extent`, `stride`, and `data_handle` (same
 * pattern as `cuvs::neighbors::make_device_padded_dataset`), or any cuVS dense dataset / dataset
 * view exposing `view`, `dim` and `stride`, in which case the logical `dim()` is quantized and the
 * row padding is skipped. The rows may be device-accessible or host-resident. Device-accessible
 * rows (device, managed or pinned) with tight row-major storage (logical stride equals dimension)
 * are passed through to training as they are; a wider row pitch triggers a contiguous dense copy
 * first. Host-resident rows are subsampled for training and encoded in bounded batches, so the
 * dense dataset is never staged on the device in full; they must be tightly packed. Empty sources
 * are rejected. The element type must be `float`, `half`, `int8_t` or `uint8_t`.
 *
 * Only the input streams. The result is a single device allocation of `n_rows` encoded rows, so the
 * compressed dataset has to fit in whatever the current device memory resource can serve, and there
 * is no host-resident output to fall back on: nothing produces, searches or serializes the
 * `host_vpq_dataset` type today. A row is `sizeof(uint32_t) + pq_dim * pq_bits / 8` bytes rounded
 * up to a multiple of 4, so at `pq_bits = 8` and `pq_dim = 384` a hundred million rows come to
 * about 39 GB, and a billion rows exceed any single device. Past that point the options are an
 * oversubscribed (managed) memory resource, which is enough to encode and serialize but not to
 * search, or sharding the rows and merging the search results.
 *
 * Typical **CAGRA** usage: build the graph on dense vectors, then attach VPQ for search (metric
 * must remain `L2Expanded` for this path). Train VPQ from the same CAGRA-padded device layout you
 * used for graph build, keep the `device_vpq_dataset` alive, and call
 * `index::update_device_dataset_same_layout` with a non-owning view.
 *
 * @code{.cpp}
 * #include <cuvs/neighbors/cagra.hpp>
 * #include <cuvs/preprocessing/quantize/pq.hpp>
 *
 * // `idx` is a `cagra::index<float, uint32_t>` with graph built on dense rows.
 * // `padded` is a `device_padded_dataset_view<float, int64_t>` view of those same rows.
 * cuvs::neighbors::vpq_params vpq_params{};
 * auto vpq = cuvs::preprocessing::quantize::pq::make_vpq_dataset(res, vpq_params, padded);
 * idx.update_device_dataset_same_layout(res, vpq.as_dataset_view());
 * @endcode
 */
template <typename SrcT>
[[nodiscard]] auto make_vpq_dataset(raft::resources const& res,
                                    cuvs::neighbors::vpq_params const& params,
                                    SrcT const& src)
  -> cuvs::neighbors::device_vpq_dataset<half, int64_t>
{
  // A cuVS dataset keeps its logical width in `dim()` while `view()` spans the full row pitch.
  if constexpr (requires {
                  src.view();
                  src.dim();
                  src.stride();
                }) {
    auto const rows    = src.view();
    using value_type   = typename decltype(rows)::value_type;
    using extents_type = raft::matrix_extent<int64_t>;
    return make_vpq_dataset(
      res,
      params,
      raft::mdspan<const value_type, extents_type, raft::layout_stride>{
        rows.data_handle(),
        raft::make_strided_layout(extents_type{rows.extent(0), int64_t{src.dim()}},
                                  cuda::std::array<int64_t, 2>{int64_t{src.stride()}, 1})});
  } else {
    using value_type = typename SrcT::value_type;
    static_assert(std::is_same_v<value_type, float> || std::is_same_v<value_type, half> ||
                    std::is_same_v<value_type, int8_t> || std::is_same_v<value_type, uint8_t>,
                  "make_vpq_dataset: element type must be float, half, int8_t or uint8_t");
    const int64_t n_rows = src.extent(0);
    const int64_t dim    = src.extent(1);
    const int64_t stride = src.stride(0) > 0 ? src.stride(0) : dim;
    RAFT_EXPECTS(n_rows > 0, "make_vpq_dataset: dataset is empty");
    return detail::vpq_train_from_rows(
      res, params, src.data_handle(), raft::get_cuda_data_type<value_type>(), n_rows, dim, stride);
  }
}

/** Current VPQ dataset serialization format version. */
inline constexpr int pq_serialization_version = 1;

/**
 * @brief Write a VPQ dataset (both codebooks plus the encoded rows) to a stream.
 *
 * Lets compression be done once, offline, and reused: a CAGRA graph over a compressed dataset
 * builds and searches on the encoded rows, so storing them removes the need to keep the dense
 * vectors around or to re-quantize them on every run.
 *
 * The file opens with the same preamble as `cagra::serialize` — a 4-byte NumPy dtype prefix then
 * `pq_serialization_version` — followed by a dataset kind tag and the codebook element type. A
 * file of the wrong kind, or one written by an older format, is rejected rather than misread. Bump
 * the version whenever the encoded row layout changes, since that layout is a library convention
 * and is not otherwise described by the file.
 *
 * Writing copies the encoded rows to the host in one piece, as `raft::serialize_mdspan` does for
 * any device matrix: it allocates a host buffer the size of those rows alongside the device copy
 * it reads from, and frees it afterwards. The two codebooks go the same way and are small. Reading
 * is the mirror image, host buffer first and then a copy to the device. So a file costs the encoded
 * rows twice while it is being written or read, once on each side, and neither direction streams.
 *
 * @code{.cpp}
 * #include <cuvs/neighbors/cagra.hpp>
 * #include <cuvs/preprocessing/quantize/pq.hpp>
 *
 * // Offline, once.
 * auto vpq = cuvs::preprocessing::quantize::pq::make_vpq_dataset(res, vpq_params, rows);
 * cuvs::preprocessing::quantize::pq::serialize(res, vpq, "base.vpq");
 *
 * // Later, per run: load the compressed rows and build a CAGRA graph over them.
 * std::unique_ptr<cuvs::neighbors::device_vpq_dataset<half, int64_t>> loaded;
 * cuvs::preprocessing::quantize::pq::deserialize(res, "base.vpq", &loaded);
 * auto index = cuvs::neighbors::cagra::build(res, index_params, loaded->as_dataset_view());
 * // `loaded` must outlive `index`, which only holds a view of it.
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] dataset the VPQ dataset to write
 * @param[out] os output stream, opened in binary mode
 */
void serialize(raft::resources const& res,
               const cuvs::neighbors::device_vpq_dataset<half, int64_t>& dataset,
               std::ostream& os);

/**
 * @copydoc serialize
 *
 * @param[in] res raft resource
 * @param[in] dataset the VPQ dataset to write
 * @param[out] filename path to write, truncated if it exists
 */
void serialize(raft::resources const& res,
               const cuvs::neighbors::device_vpq_dataset<half, int64_t>& dataset,
               const std::string& filename);

/**
 * @brief Read a VPQ dataset written by `serialize`.
 *
 * Returned through an out-parameter because the dataset owns device allocations and has no default
 * constructor, matching how `cagra::deserialize` hands back its dataset. Throws if the blob was not
 * written by `serialize` or holds codebooks of a different element type.
 *
 * @param[in] res raft resource
 * @param[in] is input stream, opened in binary mode
 * @param[out] out_dataset receives the loaded dataset; must not be null
 */
void deserialize(raft::resources const& res,
                 std::istream& is,
                 std::unique_ptr<cuvs::neighbors::device_vpq_dataset<half, int64_t>>* out_dataset);

/**
 * @copydoc deserialize
 *
 * @param[in] res raft resource
 * @param[in] filename path to read
 * @param[out] out_dataset receives the loaded dataset; must not be null
 */
void deserialize(raft::resources const& res,
                 const std::string& filename,
                 std::unique_ptr<cuvs::neighbors::device_vpq_dataset<half, int64_t>>* out_dataset);

/** @} */  // end of group product

}  // namespace pq
}  // namespace quantize
}  // namespace preprocessing
}  // namespace CUVS_EXPORT cuvs
