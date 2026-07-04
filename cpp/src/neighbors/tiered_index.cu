/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/tiered_index.cuh"

#include <cuvs/neighbors/tiered_index.hpp>

namespace cuvs::neighbors::ivf_pq {
auto typed_build(raft::resources const& res,
                 const cuvs::neighbors::ivf_pq::index_params& index_params,
                 raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::typed_index<float, int64_t>
{
  return static_cast<typed_index<float, int64_t>&&>(ivf_pq::build(res, index_params, dataset));
}

void typed_search(raft::resources const& res,
                  const ivf_pq::search_params& search_params,
                  const ivf_pq::typed_index<float, int64_t>& index,
                  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
                  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                  const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  ivf_pq::search(res, search_params, index, queries, neighbors, distances, sample_filter);
}
}  // namespace cuvs::neighbors::ivf_pq

namespace {
// Wrapper with the exact signature expected by upstream_build_function_type<device_padded_index>.
// cagra::build is now a template (no concrete device_matrix_view overload), so it cannot be
// passed as a plain function pointer; this wrapper bridges the gap.
cuvs::neighbors::cagra::device_padded_index<float, uint32_t> cagra_build_for_tiered(
  raft::resources const& res,
  cuvs::neighbors::cagra::index_params const& params,
  raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
{
  cuvs::neighbors::device_padded_dataset_view<float, int64_t> view(
    dataset, static_cast<uint32_t>(dataset.extent(1)));
  return cuvs::neighbors::cagra::build(res, params, view);
}
}  // namespace

namespace cuvs::neighbors::tiered_index {
auto build(raft::resources const& res,
           const index_params<cagra::index_params>& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> tiered_index::index<cagra::device_padded_index<float, uint32_t>>
{
  auto state = detail::build<cagra::device_padded_index<float, uint32_t>>(
    res, params, cagra_build_for_tiered, dataset);
  return cuvs::neighbors::tiered_index::index<cagra::device_padded_index<float, uint32_t>>(state);
}

auto build(raft::resources const& res,
           const index_params<ivf_flat::index_params>& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> tiered_index::index<ivf_flat::index<float, int64_t>>
{
  auto state =
    detail::build<ivf_flat::index<float, int64_t>>(res, params, ivf_flat::build, dataset);
  return cuvs::neighbors::tiered_index::index<ivf_flat::index<float, int64_t>>(state);
}

auto build(raft::resources const& res,
           const index_params<ivf_pq::index_params>& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> tiered_index::index<ivf_pq::typed_index<float, int64_t>>
{
  auto state =
    detail::build<ivf_pq::typed_index<float, int64_t>>(res, params, ivf_pq::typed_build, dataset);
  return cuvs::neighbors::tiered_index::index<ivf_pq::typed_index<float, int64_t>>(state);
}

void extend(raft::resources const& res,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            tiered_index::index<cagra::device_padded_index<float, uint32_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::extend(res, *idx->state, new_vectors);

  auto storage = next_state->storage;
  if (storage->num_rows_allocated != idx->state->storage->num_rows_allocated) {
    // CAGRA could be holding on to a non-owning view of the previous dataset in the ann_index,
    // which is problematic since the underlying ownership of the dataset could be freed here
    // call cagra::index::update_dataset on it to update the ann_index to point to the
    // new dataset
    if (next_state->ann_index) {
      auto dataset = raft::make_device_matrix_view<const float, int64_t>(
        storage->dataset.data(), next_state->ann_rows(), storage->dim);

      // Block 'search' calls during the update_dataset call to ensure that this
      // doesn't cause issues in a multithreaded environment
      std::unique_lock<std::shared_mutex> lock(idx->ann_mutex);
      detail::update_cagra_ann_dataset_for_stride(
        res, *next_state->ann_index, dataset, next_state->ann_build_pad_);
    }
  }

  idx->state = next_state;
}

void extend(raft::resources const& res,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            tiered_index::index<ivf_flat::index<float, int64_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::extend(res, *idx->state, new_vectors);
  idx->state      = next_state;
}

void extend(raft::resources const& res,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            tiered_index::index<ivf_pq::typed_index<float, int64_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::extend(res, *idx->state, new_vectors);
  idx->state      = next_state;
}

void compact(raft::resources const& res,
             tiered_index::index<cagra::device_padded_index<float, uint32_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::compact(res, *idx->state);
  idx->state      = next_state;
}

void compact(raft::resources const& res, tiered_index::index<ivf_flat::index<float, int64_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::compact(res, *idx->state);
  idx->state      = next_state;
}

void compact(raft::resources const& res,
             tiered_index::index<ivf_pq::typed_index<float, int64_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::compact(res, *idx->state);
  idx->state      = next_state;
}

void search(raft::resources const& res,
            const cagra::search_params& search_params,
            const tiered_index::index<cagra::device_padded_index<float, uint32_t>>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  // use a read-write lock to handle calls to update_dataset for cagra
  // This allows multiple readers concurrently, but only one writer
  std::shared_lock<std::shared_mutex> lock(index.ann_mutex);
  index.state->search(
    res, search_params, cagra::search, queries, neighbors, distances, sample_filter);
}

void search(raft::resources const& res,
            const ivf_flat::search_params& search_params,
            const tiered_index::index<ivf_flat::index<float, int64_t>>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  index.state->search(
    res, search_params, ivf_flat::search, queries, neighbors, distances, sample_filter);
}

void search(raft::resources const& res,
            const ivf_pq::search_params& search_params,
            const tiered_index::index<ivf_pq::typed_index<float, int64_t>>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  index.state->search(
    res, search_params, ivf_pq::typed_search, queries, neighbors, distances, sample_filter);
}

auto merge(
  raft::resources const& res,
  const index_params<cagra::index_params>& index_params,
  const std::vector<tiered_index::index<cagra::device_padded_index<float, uint32_t>>*>& indices)
  -> tiered_index::index<cagra::device_padded_index<float, uint32_t>>
{
  auto state = detail::merge(res, index_params, indices);
  return cuvs::neighbors::tiered_index::index<cagra::device_padded_index<float, uint32_t>>(state);
}

auto merge(raft::resources const& res,
           const index_params<ivf_flat::index_params>& index_params,
           const std::vector<tiered_index::index<ivf_flat::index<float, int64_t>>*>& indices)
  -> tiered_index::index<ivf_flat::index<float, int64_t>>
{
  auto state = detail::merge(res, index_params, indices);
  return cuvs::neighbors::tiered_index::index<ivf_flat::index<float, int64_t>>(state);
}

auto merge(raft::resources const& res,
           const index_params<ivf_pq::index_params>& index_params,
           const std::vector<tiered_index::index<ivf_pq::typed_index<float, int64_t>>*>& indices)
  -> tiered_index::index<ivf_pq::typed_index<float, int64_t>>
{
  auto state = detail::merge(res, index_params, indices);
  return cuvs::neighbors::tiered_index::index<ivf_pq::typed_index<float, int64_t>>(state);
}

template <typename UpstreamT>
int64_t index<UpstreamT>::size() const noexcept
{
  return state->size();
}

template <typename UpstreamT>
int64_t index<UpstreamT>::dim() const noexcept
{
  return state->dim();
}

template CUVS_EXPORT int64_t
index<cagra::device_padded_index<float, uint32_t>>::size() const noexcept;
template CUVS_EXPORT int64_t
index<cagra::device_padded_index<float, uint32_t>>::dim() const noexcept;

template struct index<ivf_flat::index<float, int64_t>>;
template struct index<ivf_pq::typed_index<float, int64_t>>;

}  // namespace cuvs::neighbors::tiered_index
