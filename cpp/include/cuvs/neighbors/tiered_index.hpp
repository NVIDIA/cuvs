/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <mutex>
#include <shared_mutex>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace ivf_pq {
// The default ivf-pq index doesn't have a 'value_type', since it
// can accept multiple different types at search time.
// However, the tiered index code needs a value_type (for the bfknn tier),
// defined in the ann index - so this class adds this for compatibility
template <typename T, typename IdxT>
struct typed_index : index<IdxT> {
  using value_type = T;
};
}  // namespace ivf_pq
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace tiered_index {

// forward reference to tiered_index implementation.
namespace detail {
template <typename UpstreamT>
struct index_state;
}  // namespace detail

/**
 * @brief Tiered Index class
 */
template <typename UpstreamT>
struct index : cuvs::neighbors::index {
  std::shared_ptr<detail::index_state<UpstreamT>> state;
  std::mutex write_mutex;
  mutable std::shared_mutex ann_mutex;

  explicit index(std::shared_ptr<detail::index_state<UpstreamT>> state) : state(state) {}
  explicit index(const index<UpstreamT>& other) : state(other.state) {}

  /** Total length of the index. */
  int64_t size() const noexcept;

  /** Dimensionality of the data. */
  int64_t dim() const noexcept;
};

template <typename upstream_index_params_type>
struct index_params : upstream_index_params_type {
  /** The minimum number of rows necessary in the index to create an
  ann index */
  int64_t min_ann_rows = 100000;

  /** Whether or not to create a new ann index on extend, if the number
  of rows in the incremental (bfknn) portion is above min_ann_rows */
  bool create_ann_index_on_extend = false;
};

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   tiered_index::index_params<cagra::index_params> build_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = tiered_index::build(handle, build_params, dataset);
 * @endcode
 *
 * @param[in] res
 * @param[in] index_params configure the index building
 * @param[in] dataset a device matrix view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed tiered index
 */
auto build(raft::resources const& res,
           const index_params<cagra::index_params>& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> tiered_index::index<cagra::index<float, uint32_t>>;

/** @copydoc build */
auto build(raft::resources const& res,
           const index_params<ivf_flat::index_params>& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> tiered_index::index<ivf_flat::index<float, int64_t>>;

/** @copydoc build */
auto build(raft::resources const& res,
           const index_params<ivf_pq::index_params>& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> tiered_index::index<ivf_pq::typed_index<float, int64_t>>;

/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   tiered_index::index_params<cagra::index_params> index_params;
 *   // train the index from a [N, D] dataset
 *   auto index_empty = tiered_index::build(res, index_params, dataset);
 *
 *   // add new data to the index
 *   tiered_index::extend(res, new_vectors, &index_empty);
 * @endcode
 *
 * @param[in] res
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[inout] idx
 */
void extend(raft::resources const& res,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            tiered_index::index<cagra::index<float, uint32_t>>* idx);

/** @copydoc extend */
void extend(raft::resources const& res,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            tiered_index::index<ivf_flat::index<float, int64_t>>* idx);

/** @copydoc extend */
void extend(raft::resources const& res,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            tiered_index::index<ivf_pq::typed_index<float, int64_t>>* idx);
/**
 * @brief Compact the index
 *
 * This function takes any data that has been added incrementally, and ensures that it
 * been added to the ANN index.
 *
 * @param[in] res
 * @param[inout] idx
 */
void compact(raft::resources const& res, tiered_index::index<cagra::index<float, uint32_t>>* idx);

/** @copydoc compact */
void compact(raft::resources const& res, tiered_index::index<ivf_flat::index<float, int64_t>>* idx);

/** @copydoc compact */
void compact(raft::resources const& res,
             tiered_index::index<ivf_pq::typed_index<float, int64_t>>* idx);

/**
 * @brief Search the tiered_index
 *
 * @param[in] res
 * @param[in] search_params configure the search
 * @param[in] index tiered-index constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& res,
            const cagra::search_params& search_params,
            const tiered_index::index<cagra::index<float, uint32_t>>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/** @copydoc search */
void search(raft::resources const& res,
            const ivf_flat::search_params& search_params,
            const tiered_index::index<ivf_flat::index<float, int64_t>>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/** @copydoc search */
void search(raft::resources const& res,
            const ivf_pq::search_params& search_params,
            const tiered_index::index<ivf_pq::typed_index<float, int64_t>>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/** @brief Merge multiple tiered indices into a single index.
 *
 * This function merges multiple tiered indices into one, combining both the datasets and graph
 * structures.
 *
 * @param[in] res
 * @param[in] index_params configure the index building
 * @param[in] indices A vector of pointers to the indices to merge. All indices should
 *                    be of the same type, and have datasets with the same dimensionality
 *
 * @return A new tiered index containing the merged indices
 */
auto merge(raft::resources const& res,
           const index_params<cagra::index_params>& index_params,
           const std::vector<tiered_index::index<cagra::index<float, uint32_t>>*>& indices)
  -> tiered_index::index<cagra::index<float, uint32_t>>;

/** @copydoc merge */
auto merge(raft::resources const& res,
           const index_params<ivf_flat::index_params>& index_params,
           const std::vector<tiered_index::index<ivf_flat::index<float, int64_t>>*>& indices)
  -> tiered_index::index<ivf_flat::index<float, int64_t>>;

/** @copydoc merge */
auto merge(raft::resources const& res,
           const index_params<ivf_pq::index_params>& index_params,
           const std::vector<tiered_index::index<ivf_pq::typed_index<float, int64_t>>*>& indices)
  -> tiered_index::index<ivf_pq::typed_index<float, int64_t>>;

}  // namespace tiered_index
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
