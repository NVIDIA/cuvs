/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <type_traits>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/types.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/knn_merge_parts.hpp>
#include <cuvs/neighbors/tiered_index.hpp>

#include <raft/util/integer_utils.hpp>

namespace cuvs::neighbors::tiered_index::detail {

/**
  Storage for brute force based incremental indices

  This lets you incrementally add vectors to the storage , without having to reallocate
  and copy on every insert.
  */
template <typename T>
struct brute_force_storage {
  rmm::device_uvector<T> dataset;
  std::optional<rmm::device_uvector<T>> norms;

  size_t num_rows_used;
  size_t num_rows_allocated;
  size_t dim;
  bool include_norms;

  brute_force_storage(const raft::resources& res,
                      size_t initial_rows,
                      size_t dim,
                      bool include_norms)
    : dataset(initial_rows * dim * sizeof(T), raft::resource::get_cuda_stream(res)),
      num_rows_used(0),
      num_rows_allocated(initial_rows),
      dim(dim),
      include_norms(include_norms)
  {
    if (include_norms) {
      norms =
        rmm::device_uvector<T>(initial_rows * sizeof(T), raft::resource::get_cuda_stream(res));
    }
  }

  size_t num_rows_available() const { return num_rows_allocated - num_rows_used; }

  void append_vectors(const raft::resources& res,
                      raft::device_matrix_view<const T, int64_t, raft::row_major> new_vectors,
                      cuvs::distance::DistanceType metric)
  {
    RAFT_EXPECTS(num_rows_available() >= static_cast<size_t>(new_vectors.extent(0)),
                 "Insufficient storage to append new vectors");
    RAFT_EXPECTS(dim == static_cast<size_t>(new_vectors.extent(1)),
                 "Dimension mismatch on appending new vectors");

    // append the vectors to the end of the allocated storage
    auto dst_ptr  = dataset.data() + num_rows_used * dim;
    auto dst_view = raft::make_device_matrix_view<T, int64_t, raft::row_major>(
      dst_ptr, new_vectors.extent(0), new_vectors.extent(1));
    raft::copy(res, dst_view, new_vectors);

    if (include_norms) {
      auto norms_view =
        raft::make_device_vector_view<T>(norms->data() + num_rows_used, new_vectors.extent(0));
      if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        raft::linalg::norm<raft::linalg::NormType::L2Norm, raft::Apply::ALONG_ROWS>(
          res, new_vectors, norms_view, raft::sqrt_op{});
      } else {
        raft::linalg::norm<raft::linalg::NormType::L2Norm, raft::Apply::ALONG_ROWS>(
          res, new_vectors, norms_view);
      }
    }
    num_rows_used += new_vectors.extent(0);
  }
};

template <typename UpstreamT>
using upstream_build_function_type = UpstreamT(
  raft::resources const&,
  typename UpstreamT::index_params_type const&,
  raft::device_matrix_view<const typename UpstreamT::value_type, int64_t, raft::row_major>);
template <typename UpstreamT>
using upstream_search_function_type =
  void(raft::resources const&,
       typename UpstreamT::search_params_type const&,
       UpstreamT const&,
       raft::device_matrix_view<const typename UpstreamT::value_type, int64_t, raft::row_major>,
       raft::device_matrix_view<int64_t, int64_t, raft::row_major>,
       raft::device_matrix_view<typename UpstreamT::value_type, int64_t, raft::row_major>,
       const cuvs::neighbors::filtering::base_filter&);

/**
  Holds a view of the current tiered index state
*/
template <typename UpstreamT>
struct index_state {
  using value_type = typename UpstreamT::value_type;

  /**
   * Build upstream ANN, preserving row stride for standard CAGRA when needed.
   */
  template <typename BuildFn, typename DatasetView>
  [[nodiscard]] static auto build_upstream_ann(
    raft::resources const& res,
    index_params<typename UpstreamT::index_params_type> const& tiered_params,
    BuildFn&& build_fn,
    DatasetView dataset) -> std::shared_ptr<UpstreamT>
  {
    if (!cuvs::neighbors::matrix_row_width_matches_cagra_required(dataset)) {
      if constexpr (std::is_same_v<UpstreamT,
                                   cuvs::neighbors::cagra::device_standard_index<float>>) {
        auto own =
          cuvs::neighbors::make_device_standard_dataset(res,
                                                        dataset,
                                                        static_cast<uint32_t>(dataset.extent(1)),
                                                        static_cast<uint32_t>(dataset.stride(0)));
        auto index = cuvs::neighbors::cagra::build(res, tiered_params, own->as_dataset_view());
        index.update_dataset(res, own->as_dataset_view());
        return std::make_shared<UpstreamT>(std::move(index));
      }
    }

    auto index = std::forward<BuildFn>(build_fn)(res, tiered_params, dataset);
    if constexpr (std::is_same_v<UpstreamT, cuvs::neighbors::cagra::device_standard_index<float>>) {
      index.update_dataset(res, cuvs::neighbors::make_device_standard_dataset_view(dataset));
    }
    return std::make_shared<UpstreamT>(std::move(index));
  }

  index_state(const index_state<UpstreamT>& other)
    : storage(other.storage),
      ann_index(other.ann_index),
      build_params(other.build_params),
      build_fn(other.build_fn)
  {
  }

  index_state(raft::resources const& res,
              const index_params<typename UpstreamT::index_params_type>& index_params,
              upstream_build_function_type<UpstreamT> build_fn,
              raft::device_matrix_view<const value_type, int64_t, raft::row_major> dataset)
    : build_params(index_params), build_fn(build_fn)
  {
    // allocate new storage for growing the index, keeping only a small buffer over the initial
    // dataset size for
    auto initial_size = dataset.extent(0) + dataset.extent(0) / 16;

    // Create an ANN index if we have sufficient rows in initial dataset
    if (dataset.extent(0) > index_params.min_ann_rows) {
      ann_index = build_upstream_ann(res, index_params, build_fn, dataset);
    }

    // allocate bfknn storage for growing the index incrementally
    auto dim       = dataset.extent(1);
    auto metric    = build_params.metric;
    bool use_norms = (metric == cuvs::distance::DistanceType::L2Expanded ||
                      metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                      metric == cuvs::distance::DistanceType::CosineExpanded);
    storage        = std::make_shared<brute_force_storage<typename UpstreamT::value_type>>(
      res, initial_size, dim, use_norms);
    storage->append_vectors(res, dataset, metric);
  }

  size_t dim() const { return storage->dim; }
  size_t size() const { return storage->num_rows_used; }

  // Number of rows inside the ann index
  size_t ann_rows() const { return ann_index ? ann_index->size() : 0; }

  // Number of rows for bkfnn
  size_t bfknn_rows() const { return storage->num_rows_used - ann_rows(); }

  void search(raft::resources const& res,
              const typename UpstreamT::search_params_type& search_params,
              upstream_search_function_type<UpstreamT> search_fn,
              raft::device_matrix_view<const value_type, int64_t, raft::row_major> queries,
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
              raft::device_matrix_view<value_type, int64_t, raft::row_major> distances,
              const cuvs::neighbors::filtering::base_filter& sample_filter)
  {
    // if we only have ANN vectors, search those and return immendiately
    if (bfknn_rows() == 0) {
      search_fn(res, search_params, *ann_index, queries, neighbors, distances, sample_filter);
      return;
    }

    // Create a BFKNN index - since we already have precalculated norms this is basically a no-op
    std::optional<raft::device_vector_view<const value_type, int64_t>> norms_view;
    if (storage->include_norms) {
      norms_view = raft::make_device_vector_view<const value_type, int64_t>(
        storage->norms->data() + ann_rows(), bfknn_rows());
    }

    auto bfknn_dataset_view = raft::make_device_matrix_view<const value_type, int64_t>(
      storage->dataset.data() + ann_rows() * storage->dim, bfknn_rows(), storage->dim);

    brute_force::index<value_type> bfknn_index(
      res, bfknn_dataset_view, norms_view, build_params.metric);

    // if we don't have an ANN index, just return the bfknn results right away
    if (!ann_index) {
      brute_force::search(res,
                          brute_force::search_params(),
                          bfknn_index,
                          queries,
                          neighbors,
                          distances,
                          sample_filter);
      return;
    }

    // query the ann/bfknn results into temporary memory. Since the knn_merge_parts
    // api expects inputs to be in a contiguous layout - create a single temporary
    // and then slice from it for the ann/bfknn search
    auto n_queries      = distances.extent(0);
    auto k              = distances.extent(1);
    auto temp_distances = raft::make_device_matrix<value_type, int64_t>(res, 2 * n_queries, k);
    auto temp_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, 2 * n_queries, k);

    // search the ann index
    search_fn(
      res,
      search_params,
      *ann_index,
      queries,
      raft::make_device_matrix_view<int64_t, int64_t>(temp_neighbors.data_handle(), n_queries, k),
      raft::make_device_matrix_view<value_type, int64_t>(
        temp_distances.data_handle(), n_queries, k),
      sample_filter);

    // search the bfknn index
    auto offset          = n_queries * k;
    auto bfknn_neighbors = raft::make_device_matrix_view<int64_t, int64_t>(
      temp_neighbors.data_handle() + offset, n_queries, k);
    auto bfknn_distances = raft::make_device_matrix_view<value_type, int64_t>(
      temp_distances.data_handle() + offset, n_queries, k);
    brute_force::search(res,
                        brute_force::search_params(),
                        bfknn_index,
                        queries,
                        bfknn_neighbors,
                        bfknn_distances,
                        sample_filter);

    if (!distance::is_min_close(build_params.metric)) {
      // knn_merge_parts doesn't currently support InnerProduct distances etc
      // instead negate here and then undo after
      raft::linalg::map(res,
                        temp_distances.view(),
                        raft::mul_const_op<value_type>(-1),
                        raft::make_const_mdspan(temp_distances.view()));
    }

    // merge results from ann_index/bfknn together, translating the bfknn ids
    auto stream                  = raft::resource::get_cuda_stream(res);
    int64_t host_translations[2] = {0, static_cast<int64_t>(ann_rows())};
    auto device_translations     = raft::make_device_vector<int64_t>(res, 2);
    raft::copy(
      res,
      device_translations.view(),
      raft::make_host_vector_view<const int64_t>(host_translations, device_translations.extent(0)));

    knn_merge_parts(res,
                    temp_distances.view(),
                    temp_neighbors.view(),
                    distances,
                    neighbors,
                    device_translations.view());

    if (!distance::is_min_close(build_params.metric)) {
      raft::linalg::map(
        res, distances, raft::mul_const_op<value_type>(-1), raft::make_const_mdspan(distances));
    }
  }

  // Stores the raw vectors that we use to do bfknn on
  std::shared_ptr<brute_force_storage<value_type>> storage;

  // ANN index data
  std::shared_ptr<UpstreamT> ann_index;

  // stores a copy of the build params - used during compact
  index_params<typename UpstreamT::index_params_type> build_params;

  // stores a pointer to the build
  std::function<upstream_build_function_type<UpstreamT>> build_fn;
};

/**
 * After BF storage grows, repoint CAGRA at the first \p ann_rows rows.
 */
inline void update_cagra_ann_dataset_for_stride(
  raft::resources const& res,
  cuvs::neighbors::cagra::device_standard_index<float>& ann_index,
  raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
{
  ann_index.update_dataset(res, cuvs::neighbors::make_device_standard_dataset_view(dataset));
}

/**
 * @brief Build the tiered index from the dataset for efficient search.
 *
 * @param[in] res
 * @param[in] index_params upstream index parameters such as the distance metric to use
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed tiered_index
 */
template <typename UpstreamT>
auto build(
  raft::resources const& res,
  const index_params<typename UpstreamT::index_params_type>& index_params,
  upstream_build_function_type<UpstreamT> build_fn,
  raft::device_matrix_view<const typename UpstreamT::value_type, int64_t, raft::row_major> dataset)
  -> std::shared_ptr<index_state<UpstreamT>>
{
  auto ret = new index_state<UpstreamT>(res, index_params, build_fn, dataset);
  return std::shared_ptr<index_state<UpstreamT>>(ret);
}

/**
 * @brief merge multiple indices together
 */
template <typename UpstreamT>
auto merge(raft::resources const& res,
           const index_params<typename UpstreamT::index_params_type>& index_params,
           const std::vector<tiered_index::index<UpstreamT>*>& indices)
  -> std::shared_ptr<index_state<UpstreamT>>
{
  using value_type = typename UpstreamT::value_type;

  RAFT_EXPECTS(indices.size() > 0, "Must pass at least one index to merge");

  for (auto index : indices) {
    RAFT_EXPECTS(index != nullptr,
                 "Null pointer detected in 'indices'. Ensure all elements are valid before usage.");
  }

  // handle simple case of only one index being merged
  if (indices.size() == 1) { return indices[0]->state; }

  auto dim           = indices[0]->state->dim();
  auto include_norms = indices[0]->state->storage->include_norms;

  // validate data and check what needs to be merged
  size_t bfknn_rows = 0, ann_rows = 0;
  for (auto index : indices) {
    RAFT_EXPECTS(dim == index->state->dim(), "Each index must have the same dimensionality");
    bfknn_rows += index->state->bfknn_rows();
    ann_rows += index->state->ann_rows();
  }

  // degenerate case - all indices are empty, just re-use the state from the first index
  if (!bfknn_rows && !ann_rows) { return indices[0]->state; }

  // concatenate all the storages together
  auto to_allocate = bfknn_rows + ann_rows;
  auto new_storage =
    std::make_shared<brute_force_storage<value_type>>(res, to_allocate, dim, include_norms);

  for (auto index : indices) {
    auto storage = index->state->storage;

    // copy over dataset to new storage
    raft::copy(res,
               raft::make_device_matrix_view<value_type, int64_t, raft::row_major>(
                 new_storage->dataset.data() + new_storage->num_rows_used * dim,
                 storage->num_rows_used,
                 dim),
               raft::make_device_matrix_view<const value_type, int64_t, raft::row_major>(
                 storage->dataset.data(), storage->num_rows_used, dim));

    // copy over precalculated norms
    if (include_norms) {
      raft::copy(res,
                 raft::make_device_vector_view<value_type, int64_t, raft::row_major>(
                   new_storage->norms->data() + new_storage->num_rows_used, storage->num_rows_used),
                 raft::make_device_vector_view<const value_type, int64_t, raft::row_major>(
                   storage->norms->data(), storage->num_rows_used));
    }
    new_storage->num_rows_used += storage->num_rows_used;
  }

  auto next_state          = std::make_shared<index_state<UpstreamT>>(*indices[0]->state);
  next_state->storage      = new_storage;
  next_state->build_params = index_params;

  if (next_state->bfknn_rows() > static_cast<size_t>(next_state->build_params.min_ann_rows)) {
    next_state = compact(res, *next_state);
  }
  return next_state;
}

template <typename UpstreamT>
auto extend(raft::resources const& res,
            const index_state<UpstreamT>& current,
            raft::device_matrix_view<const typename UpstreamT::value_type, int64_t, raft::row_major>
              new_vectors) -> std::shared_ptr<index_state<UpstreamT>>
{
  using value_type = typename UpstreamT::value_type;

  auto next_state = std::make_shared<index_state<UpstreamT>>(current);
  auto storage    = next_state->storage;

  RAFT_EXPECTS(static_cast<size_t>(new_vectors.extent(1)) == storage->dim,
               "Dimension of new vectors must match existing data");

  size_t new_rows    = new_vectors.extent(0);
  auto include_norms = storage->include_norms;
  auto dim           = storage->dim;

  // check to see if we have enough storage to append the new_rows without allocating
  if (storage->num_rows_available() < new_rows) {
    // allocate new storage, growing exponentially by doubling the allocated space
    // (or potentially larger as needed)
    size_t min_needed  = new_vectors.extent(0) + storage->num_rows_used;
    size_t to_allocate = std::max(min_needed, 2 * storage->num_rows_allocated);

    auto new_storage =
      std::make_shared<brute_force_storage<value_type>>(res, to_allocate, dim, include_norms);

    // copy over dataset to new storage
    raft::copy(res,
               raft::make_device_matrix_view<value_type, int64_t, raft::row_major>(
                 new_storage->dataset.data(), storage->num_rows_used, dim),
               raft::make_device_matrix_view<const value_type, int64_t, raft::row_major>(
                 storage->dataset.data(), storage->num_rows_used, dim));

    // copy over precalculated norms
    if (include_norms) {
      raft::copy(res,
                 raft::make_device_vector_view<value_type, int64_t, raft::row_major>(
                   new_storage->norms->data(), storage->num_rows_used),
                 raft::make_device_vector_view<const value_type, int64_t, raft::row_major>(
                   storage->norms->data(), storage->num_rows_used));
    }

    new_storage->num_rows_used = storage->num_rows_used;
    next_state->storage = storage = new_storage;
  }
  storage->append_vectors(res, new_vectors, next_state->build_params.metric);

  // Create a new ann index if we are over the threshold
  if (next_state->build_params.create_ann_index_on_extend) {
    if (next_state->bfknn_rows() > static_cast<size_t>(next_state->build_params.min_ann_rows)) {
      next_state = compact(res, *next_state);
    }
  }
  return next_state;
}

template <typename UpstreamT>
auto compact(raft::resources const& res, const index_state<UpstreamT>& current)
  -> std::shared_ptr<index_state<UpstreamT>>
{
  auto next_state = std::make_shared<index_state<UpstreamT>>(current);

  // if we don't have any new data to compact, then we're done
  if (next_state->storage->num_rows_used == next_state->ann_rows()) { return next_state; }

  // Create the new ann index based off all available data
  using value_type = typename UpstreamT::value_type;
  auto storage     = next_state->storage;
  auto dataset     = raft::make_device_matrix_view<const value_type, int64_t>(
    storage->dataset.data(), storage->num_rows_used, storage->dim);

  next_state->ann_index = index_state<UpstreamT>::build_upstream_ann(
    res, next_state->build_params, next_state->build_fn, dataset);
  return next_state;
}
}  // namespace cuvs::neighbors::tiered_index::detail
