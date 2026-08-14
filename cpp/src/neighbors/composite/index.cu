/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/composite/index.hpp>
#include <cuvs/selection/select_k.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/linalg/add.cuh>
#include <rmm/device_uvector.hpp>

namespace cuvs::neighbors::composite {

template <typename T, typename IdxT, typename OutputIdxT>
void composite_index<T, IdxT, OutputIdxT>::search(
  const raft::resources& handle,
  const cuvs::neighbors::cagra::search_params& params,
  raft::device_matrix_view<const value_type, matrix_index_type, raft::row_major> queries,
  raft::device_matrix_view<out_index_type, matrix_index_type, raft::row_major> neighbors,
  raft::device_matrix_view<float, matrix_index_type, raft::row_major> distances,
  const cuvs::neighbors::filtering::base_filter& filter) const
{
  if (children_.empty()) {
    RAFT_FAIL("The composite index is empty!");
    return;
  }

  if (children_.size() == 1) {
    cuvs::neighbors::cagra::search(
      handle, params, *children_.front(), queries, neighbors, distances, filter);
    return;
  }

  size_t num_queries = queries.extent(0);
  size_t K           = neighbors.extent(1);
  size_t num_indices = children_.size();
  size_t buffer_size = num_queries * K * num_indices;

  auto main_stream = raft::resource::get_cuda_stream(handle);
  auto tmp_res     = raft::resource::get_workspace_resource_ref(handle);

  rmm::device_uvector<out_index_type> neighbors_buffer(buffer_size, main_stream, tmp_res);
  rmm::device_uvector<float> distances_buffer(buffer_size, main_stream, tmp_res);

  std::vector<rmm::device_uvector<out_index_type>> temp_neighbors;
  std::vector<rmm::device_uvector<float>> temp_distances;

  for (size_t i = 0; i < num_indices; i++) {
    temp_neighbors.emplace_back(num_queries * K, main_stream, tmp_res);
    temp_distances.emplace_back(num_queries * K, main_stream, tmp_res);
  }

  raft::resource::wait_stream_pool_on_stream(handle);

  out_index_type offset = 0;
  out_index_type stride = K * num_indices;

  for (size_t i = 0; i < num_indices; i++) {
    auto stream = raft::resource::get_next_usable_stream(handle, i);

    raft::resources stream_pool_handle(handle);
    raft::resource::set_cuda_stream(stream_pool_handle, stream);

    auto temp_neighbors_view =
      raft::make_device_matrix_view<out_index_type, matrix_index_type, raft::row_major>(
        temp_neighbors[i].data(), num_queries, K);
    auto temp_distances_view =
      raft::make_device_matrix_view<float, matrix_index_type, raft::row_major>(
        temp_distances[i].data(), num_queries, K);

    cuvs::neighbors::cagra::search(stream_pool_handle,
                                   params,
                                   *children_[i],
                                   queries,
                                   temp_neighbors_view,
                                   temp_distances_view,
                                   filter);

    if (offset != 0) {
      raft::linalg::addScalar(temp_neighbors[i].data(),
                              temp_neighbors[i].data(),
                              offset,
                              temp_neighbors[i].size(),
                              stream);
    }

    raft::copy_matrix(
      neighbors_buffer.data() + i * K, stride, temp_neighbors[i].data(), K, K, num_queries, stream);
    raft::copy_matrix(
      distances_buffer.data() + i * K, stride, temp_distances[i].data(), K, K, num_queries, stream);

    offset += children_[i]->size();
  }
  raft::resource::sync_stream_pool(handle);

  auto distances_view = raft::make_device_matrix_view<const float, matrix_index_type>(
    distances_buffer.data(), num_queries, K * num_indices);
  auto neighbors_view = raft::make_device_matrix_view<const out_index_type, matrix_index_type>(
    neighbors_buffer.data(), num_queries, K * num_indices);

  cuvs::selection::select_k(handle,
                            distances_view,
                            neighbors_view,
                            distances,
                            neighbors,
                            cuvs::distance::is_min_close(metric()),
                            true,
                            cuvs::selection::SelectAlgo::kAuto);
}

template class CUVS_EXPORT composite_index<float, uint32_t, uint32_t>;
template class CUVS_EXPORT composite_index<float, uint32_t, int64_t>;
template class CUVS_EXPORT composite_index<half, uint32_t, uint32_t>;
template class CUVS_EXPORT composite_index<half, uint32_t, int64_t>;
template class CUVS_EXPORT composite_index<int8_t, uint32_t, uint32_t>;
template class CUVS_EXPORT composite_index<int8_t, uint32_t, int64_t>;
template class CUVS_EXPORT composite_index<uint8_t, uint32_t, uint32_t>;
template class CUVS_EXPORT composite_index<uint8_t, uint32_t, int64_t>;

}  // namespace cuvs::neighbors::composite
