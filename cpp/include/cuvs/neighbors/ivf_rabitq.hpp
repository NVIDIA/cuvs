/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_fp16.h>

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>

#include <optional>
#include <tuple>
#include <variant>
#include <vector>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace ivf_rabitq {

// forward declaration
namespace detail {
class IVFGPU;
}

/**
 * @defgroup ivf_rabitq_cpp_index_params IVF-RaBitQ index build parameters
 * @{
 */
struct index_params : cuvs::neighbors::index_params {
  /**
   * The number of inverted lists (clusters)
   *
   * Hint: Increasing this parameter may alleviate shared memory pressure.
   */
  uint32_t n_lists = 1024;
  /**
   * The total number of bits per dimension (single bit required for the binary RaBitQ algorithm +
   * additional bits for extended RaBitQ).
   *
   * Supported values: [1, 2, 3, 4, 5, 6, 7, 8, 9].
   *
   * Hint: the smaller the 'bits_per_dim', the smaller the index size and the better the search
   * performance, but the lower the recall.
   */
  uint32_t bits_per_dim = 3;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters = 20;
  /** The number of data vectors (per cluster) to use during iterative kmeans building. */
  uint32_t max_train_points_per_cluster = 256;
  /** Flag for using the fast quantize method */
  bool fast_quantize_flag = true;
  /**
   * Maximum number of vectors per batch when using streaming construction.
   *
   * This parameter controls the batch size during streaming construction from host memory.
   * Batches contain complete clusters only (no partial clusters across batch boundaries).
   *
   * Note: Streaming construction is automatically used when the dataset doesn't fit
   * comfortably in GPU memory (determined by available workspace and kTolerableRatio).
   */
  size_t streaming_batch_size = 100000;
  /**
   * Force streaming construction regardless of dataset size.
   *
   * When set to true, streaming construction will be used even if the dataset would fit
   * in GPU memory. This is useful for testing or when you want explicit control over
   * the construction method.
   *
   * Note: This parameter only applies when the input dataset is in host memory. If the
   * dataset is already in device memory, streaming construction is not applicable and
   * this parameter has no effect.
   *
   * Default: false (auto-detect based on available memory)
   */
  bool force_streaming = false;
};
/**
 * @}
 */

/**
 * @defgroup ivf_rabitq_cpp_search_params IVF-RaBitQ index search parameters
 * @{
 */
/** A type for specifying the mode for searching the RaBitQ index. */
enum class search_mode {
  LUT16  = 0,
  LUT32  = 1,
  QUANT4 = 2,
  QUANT8 = 3,
};

struct search_params : cuvs::neighbors::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
  /** The search mode to be used. */
  search_mode mode = search_mode::QUANT4;
};
/**
 * @}
 */

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

/**
 * @defgroup ivf_rabitq_cpp_index IVF-RaBitQ index
 * @{
 */
/**
 * @brief IVF-RaBitQ index.
 * @tparam IdxT type of the indices in the source dataset
 *
 */
template <typename IdxT>
struct index : cuvs::neighbors::index {
  using index_params_type  = ivf_rabitq::index_params;
  using search_params_type = ivf_rabitq::search_params;
  using index_type         = IdxT;
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  index(const index&) = delete;
  index(
    index&&);  // declaration only; definition in impl to allow member unique_ptr of incomplete type
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&;  // declaration only; definition in impl to allow member
                                      // unique_ptr of incomplete type
  ~index();  // declaration only; definition in impl to allow member unique_ptr of incomplete type

  /**
   * @brief Construct an empty index yet to be populated.
   *
   */
  index(raft::resources const& handle);

  /** Construct an empty index yet to be populated. */
  index(raft::resources const& handle,
        size_t n_rows,
        uint32_t dim,
        uint32_t n_lists,
        uint32_t bits_per_dim,
        cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);

  /** Dimensionality of the input data. */
  uint32_t dim() const noexcept;

  /** Total length of the index. */
  IdxT size() const noexcept;

  /** Distance metric used to build the index. */
  cuvs::distance::DistanceType metric() const noexcept;

  /** Accessor for underlying RaBitQ index */
  detail::IVFGPU& rabitq_index() noexcept;

 private:
  std::unique_ptr<detail::IVFGPU> rabitq_index_;
};
/**
 * @}
 */

/**
 * @defgroup ivf_rabitq_cpp_index_build IVF-RaBitQ index build
 * @{
 */
/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_rabitq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_rabitq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device_matrix_view to a row-major matrix [n_rows, dim]
 * @return the constructed ivf-rabitq index
 *
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_rabitq::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_rabitq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_rabitq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_rabitq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host_matrix_view to a row-major matrix [n_rows, dim]
 * @return the constructed ivf-rabitq index
 *
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_rabitq::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_rabitq::index<int64_t>;
/**
 * @}
 */

/**
 * @defgroup ivf_rabitq_cpp_index_search IVF-RaBitQ index search
 * @{
 */
/**
 * @brief Search ANN using the constructed index.
 *
 * @param[in] handle
 * @param[in] search_params configure the search
 * @param[in] index ivf-rabitq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_rabitq::search_params& search_params,
            cuvs::neighbors::ivf_rabitq::index<int64_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @}
 */

/**
 * @defgroup ivf_rabitq_cpp_serialize IVF-RaBitQ index serialize
 * @{
 */
/**
 * Save the index to file.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = ivf_rabitq::build(...);`
 * cuvs::neighbors::ivf_rabitq::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-RaBitQ index
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               cuvs::neighbors::ivf_rabitq::index<int64_t>& index);

/**
 * Load index from file.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using IdxT = int64_t; // type of the index
 * // create an empty index
 * ivf_rabitq::index<IdxT> index(handle);
 *
 * cuvs::neighbors::ivf_rabitq::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index IVF-PQ index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_rabitq::index<int64_t>* index);
/**
 * @}
 */

}  // namespace ivf_rabitq
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
