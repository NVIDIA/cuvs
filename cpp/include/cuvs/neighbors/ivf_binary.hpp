/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/binary.hpp>
#include <cuvs/core/export.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/preprocessing/quantize/binary.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace ivf_binary {

namespace detail {
struct index_access;
}  // namespace detail

/**
 * @defgroup ivf_binary_cpp_index_params IVF-Binary index build parameters
 * @{
 */
struct index_params {
  /** The number of inverted lists (clusters). */
  uint32_t n_lists = 1024;
  /** The number of iterations searching for k-means centers. */
  uint32_t kmeans_n_iters = 20;
  /** The fraction of data to use during iterative k-means building. */
  double kmeans_trainset_fraction = 0.5;
  /**
   * By default (adaptive_centers = false), the cluster centers are trained in `build` and never
   * modified in `extend`.
   *
   * With adaptive_centers = true, `extend` updates the centers to the majority bit of the vectors
   * added to each cluster, with ties resolved to 0. The index keeps exact per-bit counts of ones to
   * make the update independent of how the data are split between `extend` calls.
   */
  bool adaptive_centers = false;
  /**
   * Allocate the minimum amount of memory required for the inverted lists, instead of reserving
   * extra space to amortize repeated calls of `extend`.
   */
  bool conservative_memory_allocation = false;
  /**
   * Whether to add the dataset content to the index in `build`. If false, `build` only trains the
   * clustering and the index is populated later with `extend`.
   */
  bool add_data_on_build = true;
};
/** @} */

/**
 * @defgroup ivf_binary_cpp_search_params IVF-Binary index search parameters
 * @{
 */
struct search_params : cuvs::neighbors::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
};
/** @} */

/**
 * @defgroup ivf_binary_cpp_index IVF-Binary index
 * @{
 */
/**
 * @brief IVF index of bit-packed binary vectors searched with the Hamming distance.
 *
 * The index stores the codes of a binary-quantized dataset (see `device_binary_dataset_view`),
 * together with the quantizer that produced them. Queries must be supplied as bit-packed bytes.
 * Neighbors are returned with their Hamming distances, i.e. the number of differing bits.
 *
 * @tparam IdxT type of the indices in the source dataset
 */
template <typename IdxT = int64_t>
struct index : cuvs::neighbors::index {
  using index_params_type  = ivf_binary::index_params;
  using search_params_type = ivf_binary::search_params;
  using index_type         = IdxT;
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  index(const index&)            = delete;
  index(index&&)                 = default;
  index& operator=(const index&) = delete;
  index& operator=(index&&)      = default;
  ~index()                       = default;

  /** Construct an empty index. It needs to be trained with `build` or loaded with `deserialize`. */
  explicit index(raft::resources const& res);

  /** Construct an empty index of `dim`-bit vectors. It needs to be trained and then populated. */
  index(raft::resources const& res, const index_params& params, uint32_t dim);

  /** The number of bits in each vector. */
  uint32_t dim() const noexcept;
  /** The number of bytes used to store each vector. */
  uint32_t storage_dim() const noexcept;
  /** The number of clusters/inverted lists. */
  uint32_t n_lists() const noexcept;
  /** The total number of vectors in the index. */
  IdxT size() const noexcept;
  /** Whether `centers()` change upon extending the index. */
  bool adaptive_centers() const noexcept;
  /** Whether the lists are allocated without extra space (see index_params). */
  bool conservative_memory_allocation() const noexcept;

  /** Sizes of the lists (clusters) [n_lists]. */
  raft::device_vector_view<const uint32_t, uint32_t> list_sizes() const noexcept;
  /** Bit-packed cluster centers [n_lists, storage_dim]. */
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> centers() const noexcept;
  /**
   * Exact number of ones at each bit of the vectors in each cluster [n_lists, 8 * storage_dim];
   * empty unless the centers are adaptive.
   */
  raft::device_matrix_view<const uint32_t, int64_t, raft::row_major> center_counts() const noexcept;
  /** The build dataset's quantizer, available for external encoding of queries or new vectors. */
  const preprocessing::quantize::binary::quantizer<float>& quantizer() const noexcept;

 private:
  friend struct detail::index_access;

  bool adaptive_centers_;
  preprocessing::quantize::binary::quantizer<float> quantizer_;
  raft::device_matrix<uint8_t, int64_t, raft::row_major> centers_;
  raft::device_matrix<uint32_t, int64_t, raft::row_major> center_counts_;
  // The inverted lists of codes, scanned by the interleaved Hamming kernel of IVF-Flat.
  ivf_flat::index<uint8_t, IdxT> lists_;
};
/** @} */

/**
 * @defgroup ivf_binary_cpp_index_build IVF-Binary index build
 * @{
 */
/**
 * @brief Build the index from a binary-quantized dataset.
 *
 * The quantizer of the dataset manifest (the thresholds and the dimension) is copied into the
 * index.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   namespace binary = cuvs::preprocessing::quantize::binary;
 *   auto quantizer = binary::train(handle, binary::params{}, raw_dataset);
 *   auto codes = raft::make_device_matrix<uint8_t, int64_t>(
 *     handle, n_rows, raft::div_rounding_up_safe<int64_t>(quantizer.dim(), 8));
 *   binary::transform(handle, quantizer, raw_dataset, codes.view());
 *   cuvs::device_binary_dataset_view<float, int64_t> dataset{
 *     raft::make_const_mdspan(codes.view()), quantizer.view()};
 *   auto index = ivf_binary::build(handle, ivf_binary::index_params{}, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset the binary-quantized vectors
 *
 * @return the constructed index
 */
auto build(raft::resources const& handle,
           const index_params& index_params,
           device_binary_dataset_view<float, int64_t> dataset) -> index<int64_t>;

/**
 * @brief Add bit-packed vectors to the index.
 *
 * The supplied codes are used as-is. The dictionary of the new vectors is not compared with or
 * copied into the index; choosing the quantizer is the caller's responsibility.
 *
 * @param[in] handle
 * @param[in] new_vectors the binary-quantized vectors to add
 * @param[in] new_indices ids of the new vectors; required if the index is not empty, otherwise the
 *   ids start from zero
 * @param[inout] index the index to extend
 */
void extend(raft::resources const& handle,
            device_binary_dataset_view<float, int64_t> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            index<int64_t>* index);
/** @} */

/**
 * @defgroup ivf_binary_cpp_index_search IVF-Binary index search
 * @{
 */
/**
 * @brief Search the index with bit-packed byte queries using Hamming distance.
 *
 * Each query has `index.storage_dim()` bytes, with bit `i` stored in bit `i % 8` of byte `i / 8`.
 * Unused bits in the final byte must be zero. The supplied codes are searched as-is, without
 * quantization or threshold comparisons; choosing the quantizer is the caller's responsibility.
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] index the index
 * @param[in] queries a device matrix [n_queries, index.storage_dim()] of bit-packed queries
 * @param[out] neighbors ids of the k nearest neighbors [n_queries, k]
 * @param[out] distances Hamming distances to the neighbors [n_queries, k]
 * @param[in] sample_filter an optional device filter on the dataset ids
 */
void search(raft::resources const& handle,
            const search_params& params,
            const index<int64_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});
/** @} */

/**
 * @defgroup ivf_binary_cpp_serialize IVF-Binary index serialize
 * @{
 */
/**
 * Save the index to an output stream.
 *
 * Experimental, both the API and the serialization format are subject to change.
 */
void serialize(raft::resources const& handle, std::ostream& os, const index<int64_t>& index);

/** Save the index to a file. */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const index<int64_t>& index);

/**
 * Load the index from an input stream.
 *
 * Experimental, both the API and the serialization format are subject to change.
 */
void deserialize(raft::resources const& handle, std::istream& is, index<int64_t>* index);

/** Load the index from a file. */
void deserialize(raft::resources const& handle, const std::string& filename, index<int64_t>* index);
/** @} */

}  // namespace ivf_binary
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
