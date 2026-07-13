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
// Wrapper with the exact signature expected by upstream_build_function_type<device_standard_index>.
// cagra::build is now a template (no concrete device_matrix_view overload), so it cannot be
// passed as a plain function pointer; this wrapper bridges the gap.
cuvs::neighbors::cagra::device_standard_index<float, uint32_t> cagra_build_for_tiered_standard(
  raft::resources const& res,
  cuvs::neighbors::cagra::index_params const& params,
  raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
{
  auto view = cuvs::neighbors::make_device_standard_dataset_view(dataset);
  return cuvs::neighbors::cagra::build(res, params, view);
}

cuvs::neighbors::cagra::device_padded_index<float, uint32_t> cagra_build_for_tiered_padded(
  raft::resources const& res,
  cuvs::neighbors::cagra::index_params const& params,
  raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
{
  auto view = cuvs::neighbors::make_device_padded_dataset_view(res, dataset);
  return cuvs::neighbors::cagra::build(res, params, view);
}

}  // namespace

namespace cuvs::neighbors::tiered_index {
auto build(raft::resources const& res,
           const index_params<cagra::index_params>& params,
           cuvs::neighbors::device_padded_dataset_view<float, int64_t> dataset)
  -> tiered_index::index<cagra::device_padded_index<float, uint32_t>>
{
  auto state = detail::build<cagra::device_padded_index<float, uint32_t>>(
    res, params, cagra_build_for_tiered_padded, dataset.view());
  return cuvs::neighbors::tiered_index::index<cagra::device_padded_index<float, uint32_t>>(state);
}

auto build(raft::resources const& res,
           const index_params<cagra::index_params>& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> tiered_index::index<cagra::device_standard_index<float, uint32_t>>
{
  auto state = detail::build<cagra::device_standard_index<float, uint32_t>>(
    res, params, cagra_build_for_tiered_standard, dataset);
  return cuvs::neighbors::tiered_index::index<cagra::device_standard_index<float, uint32_t>>(state);
}

auto attach_padded_dataset_for_search(
  raft::resources const& res,
  const tiered_index::index<cagra::device_standard_index<float, uint32_t>>& idx,
  cuvs::neighbors::device_padded_dataset_view<float, int64_t> padded_dataset)
  -> tiered_index::index<cagra::device_padded_index<float, uint32_t>>
{
  RAFT_EXPECTS(padded_dataset.n_rows() == idx.size(),
               "padded_dataset row count must match tiered index size");
  RAFT_EXPECTS(padded_dataset.dim() == idx.dim(),
               "padded_dataset dimension must match tiered index dimension");

  auto next_state =
    std::make_shared<detail::index_state<cagra::device_padded_index<float, uint32_t>>>();
  next_state->storage      = idx.state->storage;
  next_state->build_params = idx.state->build_params;
  next_state->build_fn     = cagra_build_for_tiered_padded;
  next_state->ann_index.reset();

  if (idx.state->ann_index) {
    auto padded_mds = padded_dataset.view();
    auto ann_rows   = static_cast<int64_t>(idx.state->ann_rows());
    auto ann_mds    = raft::make_device_matrix_view<const float, int64_t>(
      padded_mds.data_handle(), ann_rows, static_cast<int64_t>(padded_mds.extent(1)));
    auto ann_padded_view =
      cuvs::neighbors::device_padded_dataset_view<float, int64_t>(ann_mds, padded_dataset.dim());
    auto ann_padded_idx = cuvs::neighbors::cagra::attach_padded_dataset_for_search(
      res, *idx.state->ann_index, ann_padded_view);
    next_state->ann_index =
      std::make_shared<cuvs::neighbors::cagra::device_padded_index<float, uint32_t>>(
        std::move(ann_padded_idx));
  }

  return cuvs::neighbors::tiered_index::index<cagra::device_padded_index<float, uint32_t>>(
    next_state);
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
  idx->state      = next_state;
}

void extend(raft::resources const& res,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            tiered_index::index<cagra::device_standard_index<float, uint32_t>>* idx)
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
      detail::update_cagra_ann_dataset_for_stride(res, *next_state->ann_index, dataset);
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

void compact(raft::resources const& res,
             tiered_index::index<cagra::device_standard_index<float, uint32_t>>* idx)
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
  std::shared_lock<std::shared_mutex> lock(index.ann_mutex);
  index.state->search(
    res, search_params, cagra::search, queries, neighbors, distances, sample_filter);
}

void search(raft::resources const& res,
            const cagra::search_params& search_params,
            const tiered_index::index<cagra::device_standard_index<float, uint32_t>>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  RAFT_FAIL(
    "tiered_index::search(standard CAGRA) requires explicit attach first. "
    "Call tiered_index::attach_padded_dataset_for_search(...) and then search the returned padded "
    "tiered index.");
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

auto merge(
  raft::resources const& res,
  const index_params<cagra::index_params>& index_params,
  const std::vector<tiered_index::index<cagra::device_standard_index<float, uint32_t>>*>& indices)
  -> tiered_index::index<cagra::device_standard_index<float, uint32_t>>
{
  auto state = detail::merge(res, index_params, indices);
  return cuvs::neighbors::tiered_index::index<cagra::device_standard_index<float, uint32_t>>(state);
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
template CUVS_EXPORT int64_t
index<cagra::device_standard_index<float, uint32_t>>::size() const noexcept;
template CUVS_EXPORT int64_t
index<cagra::device_standard_index<float, uint32_t>>::dim() const noexcept;

template struct index<ivf_flat::index<float, int64_t>>;
template struct index<ivf_pq::typed_index<float, int64_t>>;

}  // namespace cuvs::neighbors::tiered_index
