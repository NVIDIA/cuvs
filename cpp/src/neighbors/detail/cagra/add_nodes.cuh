/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../../../core/omp_wrapper.hpp"
#include "../ann_utils.cuh"
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/device_resources.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>
#include <raft/stats/histogram.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_buffer.hpp>

#include <cstdint>

namespace cuvs::neighbors::cagra {

template <class T, class IdxT, class Accessor>
void add_node_core(
  raft::resources const& handle,
  const cuvs::neighbors::cagra::index<T, IdxT>& idx,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::layout_stride, Accessor>
    additional_dataset_view,
  raft::host_matrix_view<IdxT, std::int64_t> updated_graph,
  const cuvs::neighbors::cagra::extend_params& extend_params)
{
  using DistanceT                 = float;
  const std::size_t degree        = idx.graph_degree();
  const std::size_t dim           = idx.dim();
  const std::size_t old_size      = idx.dataset().extent(0);
  const std::size_t num_add       = additional_dataset_view.extent(0);
  const std::size_t new_size      = old_size + num_add;
  const std::uint32_t base_degree = degree * 2;

  // Step 0: Calculate the number of incoming edges for each node
  auto dev_num_incoming_edges = raft::make_device_vector<int, std::uint64_t>(handle, new_size);

  raft::matrix::fill(handle, dev_num_incoming_edges.view(), int(0));
  raft::stats::histogram<IdxT, std::int64_t>(raft::stats::HistTypeAuto,
                                             dev_num_incoming_edges.data_handle(),
                                             old_size,
                                             idx.graph().data_handle(),
                                             old_size * degree,
                                             1,
                                             raft::resource::get_cuda_stream(handle));

  auto host_num_incoming_edges = raft::make_host_vector<int, std::uint64_t>(new_size);
  raft::copy(handle, host_num_incoming_edges.view(), dev_num_incoming_edges.view());

  std::size_t data_size_per_vector =
    sizeof(IdxT) * base_degree + sizeof(DistanceT) * base_degree + sizeof(T) * dim;
  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, additional_dataset_view.data_handle()));
  if (attr.devicePointer == nullptr) {
    // for batch_load_iterator
    data_size_per_vector += sizeof(T) * dim;
  }

  const std::size_t max_search_batch_size =
    std::min(std::max(1lu, raft::resource::get_workspace_free_bytes(handle) / data_size_per_vector),
             num_add);
  RAFT_EXPECTS(max_search_batch_size > 0, "No enough working memory space is left.");

  cuvs::neighbors::cagra::search_params params;
  params.itopk_size = std::max(base_degree * 2lu, 256lu);

  // Memory space for rank-based neighbor list
  auto mr = raft::resource::get_workspace_resource_ref(handle);

  auto neighbor_indices = raft::make_device_mdarray<IdxT, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_search_batch_size, base_degree));

  auto neighbor_distances = raft::make_device_mdarray<DistanceT, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_search_batch_size, base_degree));

  auto queries = raft::make_device_mdarray<T, std::int64_t>(
    handle, mr, raft::make_extents<std::int64_t>(max_search_batch_size, dim));

  auto host_neighbor_indices =
    raft::make_host_matrix<IdxT, std::int64_t>(max_search_batch_size, base_degree);

  auto additional_dataset_batch = cuvs::spatial::knn::detail::utils::make_batch_load_iterator<T>(
    handle,
    additional_dataset_view.data_handle(),
    static_cast<std::int64_t>(num_add),
    static_cast<std::int64_t>(additional_dataset_view.stride(0)),
    max_search_batch_size,
    raft::resource::get_cuda_stream(handle),
    mr);
  for (const auto& batch : additional_dataset_batch) {
    // Step 1: Obtain K (=base_degree) nearest neighbors of the new vectors by CAGRA search
    // Create queries
    raft::copy_matrix(queries.data_handle(),
                      dim,
                      batch.data(),
                      additional_dataset_view.stride(0),
                      dim,
                      batch.size(),
                      raft::resource::get_cuda_stream(handle));

    const auto queries_view = raft::make_device_matrix_view<const T, std::int64_t>(
      queries.data_handle(), batch.size(), dim);

    auto neighbor_indices_view = raft::make_device_matrix_view<IdxT, std::int64_t>(
      neighbor_indices.data_handle(), batch.size(), base_degree);
    auto neighbor_distances_view = raft::make_device_matrix_view<float, std::int64_t>(
      neighbor_distances.data_handle(), batch.size(), base_degree);

    neighbors::cagra::search(
      handle, params, idx, queries_view, neighbor_indices_view, neighbor_distances_view);

    raft::copy(
      handle,
      raft::make_host_vector_view(host_neighbor_indices.data_handle(), batch.size() * base_degree),
      raft::make_device_vector_view(neighbor_indices.data_handle(), batch.size() * base_degree));
    raft::resource::sync_stream(handle);

    // Check search results
    constexpr int max_warnings = 3;
    int num_warnings           = 0;
    for (std::size_t vec_i = 0; vec_i < batch.size(); vec_i++) {
      std::uint32_t invalid_edges = 0;
      for (std::uint32_t i = 0; i < base_degree; i++) {
        if (host_neighbor_indices(vec_i, i) >= old_size) { invalid_edges++; }
      }
      if (invalid_edges > 0) {
        if (num_warnings < max_warnings) {
          RAFT_LOG_WARN(
            "Invalid edges found in search results "
            "(vec_i:%lu, invalid_edges:%lu, degree:%lu, base_degree:%lu)",
            (uint64_t)vec_i,
            (uint64_t)invalid_edges,
            (uint64_t)degree,
            (uint64_t)base_degree);
        }
        num_warnings += 1;
      }
    }
    if (num_warnings > max_warnings) {
      RAFT_LOG_WARN("The number of queries that contain invalid search results: %d", num_warnings);
    }

    // Step 2: rank-based reordering
#pragma omp parallel
    {
      std::vector<std::pair<IdxT, std::size_t>> detourable_node_count_list(base_degree);
      for (std::size_t vec_i = cuvs::core::omp::get_thread_num(); vec_i < batch.size();
           vec_i += cuvs::core::omp::get_num_threads()) {
        // Count detourable edges
        for (std::uint32_t i = 0; i < base_degree; i++) {
          std::uint32_t detourable_node_count = 0;
          const auto a_id                     = host_neighbor_indices(vec_i, i);
          if (a_id >= idx.size()) {
            // If the node ID is not valid, the number of detours is increased
            // to a value greater than the maximum, so that the edge to that
            // node is not selected as much as possible.
            detourable_node_count_list[i] = std::make_pair(a_id, base_degree + 1);
            continue;
          }
          for (std::uint32_t j = 0; j < i; j++) {
            const auto b0_id = host_neighbor_indices(vec_i, j);
            if (b0_id >= idx.size()) { continue; }
            for (std::uint32_t k = 0; k < degree; k++) {
              const auto b1_id = updated_graph(b0_id, k);
              if (a_id == b1_id) {
                detourable_node_count++;
                break;
              }
            }
          }
          detourable_node_count_list[i] = std::make_pair(a_id, detourable_node_count);
        }

        std::sort(detourable_node_count_list.begin(),
                  detourable_node_count_list.end(),
                  [&](const std::pair<IdxT, std::size_t> a, const std::pair<IdxT, std::size_t> b) {
                    return a.second < b.second;
                  });

        for (std::size_t i = 0; i < degree; i++) {
          updated_graph(old_size + batch.offset() + vec_i, i) = detourable_node_count_list[i].first;
        }
      }
    }

    // Step 3: Add reverse edges
    const std::uint32_t rev_edge_search_range = degree / 2;
    const std::uint32_t num_rev_edges         = degree / 2;
    std::vector<IdxT> rev_edges(num_rev_edges), temp(degree);
    for (std::size_t vec_i = 0; vec_i < batch.size(); vec_i++) {
      // Create a reverse edge list
      const auto target_new_node_id = old_size + batch.offset() + vec_i;
      for (std::size_t i = 0; i < num_rev_edges; i++) {
        const auto target_node_id = updated_graph(old_size + batch.offset() + vec_i, i);
        if (target_node_id >= new_size) {
          RAFT_FAIL("Invalid node ID found in updated_graph (%u)\n", target_node_id);
        }
        IdxT replace_id                        = new_size;
        IdxT replace_id_j                      = 0;
        std::size_t replace_num_incoming_edges = 0;
        for (std::int32_t j = degree - 1; j >= static_cast<std::int32_t>(rev_edge_search_range);
             j--) {
          const auto neighbor_id = updated_graph(target_node_id, j);
          if (neighbor_id >= new_size) {
            RAFT_FAIL("Invalid node ID found in updated_graph (%u)\n", neighbor_id);
          }
          const std::size_t num_incoming_edges = host_num_incoming_edges(neighbor_id);
          if (num_incoming_edges > replace_num_incoming_edges) {
            // Check duplication
            bool dup = false;
            for (std::uint32_t k = 0; k < i; k++) {
              if (rev_edges[k] == neighbor_id) {
                dup = true;
                break;
              }
            }
            if (dup) { continue; }

            // Update rev edge candidate
            replace_num_incoming_edges = num_incoming_edges;
            replace_id                 = neighbor_id;
            replace_id_j               = j;
          }
        }
        updated_graph(target_node_id, replace_id_j) = target_new_node_id;
        rev_edges[i]                                = replace_id;
      }
      host_num_incoming_edges(target_new_node_id) = num_rev_edges;

      // Create a neighbor list of a new node by interleaving the kNN neighbor list and reverse edge
      // list
      std::uint32_t interleave_switch = 0, rank_base_i = 0, rev_edges_return_i = 0, num_add = 0;
      const auto rank_based_list_ptr =
        updated_graph.data_handle() + (old_size + batch.offset() + vec_i) * degree;
      const auto rev_edges_return_list_ptr = rev_edges.data();
      while ((num_add < degree) &&
             ((rank_base_i < degree) || (rev_edges_return_i < num_rev_edges))) {
        const auto node_list_ptr =
          interleave_switch == 0 ? rank_based_list_ptr : rev_edges_return_list_ptr;
        auto& node_list_index          = interleave_switch == 0 ? rank_base_i : rev_edges_return_i;
        const auto max_node_list_index = interleave_switch == 0 ? degree : num_rev_edges;
        for (; node_list_index < max_node_list_index; node_list_index++) {
          const auto candidate = node_list_ptr[node_list_index];
          if (candidate >= new_size) { continue; }
          // Check duplication
          bool dup = false;
          for (std::uint32_t j = 0; j < num_add; j++) {
            if (temp[j] == candidate) {
              dup = true;
              break;
            }
          }
          if (!dup) {
            temp[num_add] = candidate;
            num_add++;
            break;
          }
        }
        interleave_switch = 1 - interleave_switch;
      }
      if (num_add < degree) {
        RAFT_FAIL("Number of edges is not enough (target_new_node_id:%lu, num_add:%lu, degree:%lu)",
                  (uint64_t)target_new_node_id,
                  (uint64_t)num_add,
                  (uint64_t)degree);
      }
      for (std::uint32_t i = 0; i < degree; i++) {
        updated_graph(target_new_node_id, i) = temp[i];
      }
    }
  }
}

template <class T, class IdxT>
void add_graph_nodes(
  raft::resources const& handle,
  raft::device_matrix_view<const T, int64_t, raft::layout_stride> input_updated_dataset_view,
  const neighbors::cagra::index<T, IdxT>& index,
  raft::host_matrix_view<IdxT, std::int64_t> updated_graph_view,
  const cagra::extend_params& params)
{
  if (input_updated_dataset_view.extent(0) < index.size()) {
    RAFT_FAIL("Updated dataset must be not smaller than the previous index state.");
  }

  const std::size_t initial_dataset_size = index.size();
  const std::size_t new_dataset_size     = input_updated_dataset_view.extent(0);
  const std::size_t num_new_nodes        = new_dataset_size - initial_dataset_size;
  const std::size_t degree               = index.graph_degree();
  const std::size_t dim                  = index.dim();
  const std::size_t stride               = input_updated_dataset_view.stride(0);
  const std::size_t max_chunk_size_ =
    params.max_chunk_size == 0 ? new_dataset_size : params.max_chunk_size;

  raft::copy(handle,
             raft::make_device_vector_view(updated_graph_view.data_handle(), index.graph().size()),
             raft::make_device_vector_view(index.graph().data_handle(), index.graph().size()));

  neighbors::cagra::index<T, IdxT> internal_index(
    handle,
    index.metric(),
    raft::make_device_matrix_view<const T, int64_t>(nullptr, 0, dim),
    raft::make_device_matrix_view<const IdxT, int64_t>(nullptr, 0, degree));

  for (std::size_t additional_dataset_offset = 0; additional_dataset_offset < num_new_nodes;
       additional_dataset_offset += max_chunk_size_) {
    const auto actual_chunk_size =
      std::min(num_new_nodes - additional_dataset_offset, max_chunk_size_);

    auto dataset_view = raft::make_device_strided_matrix_view<const T, std::int64_t>(
      input_updated_dataset_view.data_handle(),
      initial_dataset_size + additional_dataset_offset,
      dim,
      stride);
    auto graph_view = raft::make_host_matrix_view<const IdxT, std::int64_t>(
      updated_graph_view.data_handle(), initial_dataset_size + additional_dataset_offset, degree);

    internal_index.update_dataset(handle, dataset_view);

    // Note: The graph is copied to the device memory.
    internal_index.update_graph(handle, graph_view);
    raft::resource::sync_stream(handle);

    auto updated_graph = raft::make_host_matrix_view<IdxT, std::int64_t>(
      updated_graph_view.data_handle(),
      initial_dataset_size + additional_dataset_offset + actual_chunk_size,
      degree);
    auto additional_dataset_view = raft::make_device_strided_matrix_view<const T, std::int64_t>(
      input_updated_dataset_view.data_handle() +
        (initial_dataset_size + additional_dataset_offset) * stride,
      actual_chunk_size,
      dim,
      stride);

    neighbors::cagra::add_node_core<T, IdxT>(
      handle, internal_index, additional_dataset_view, updated_graph, params);
    raft::resource::sync_stream(handle);
  }
}

template <class T, class IdxT, class Accessor>
void extend_core(
  raft::resources const& handle,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> additional_dataset,
  cuvs::neighbors::cagra::index<T, IdxT>& index,
  const cagra::extend_params& params,
  std::optional<raft::device_matrix_view<T, int64_t, raft::layout_stride>> new_dataset_buffer_view,
  std::optional<raft::device_matrix_view<IdxT, int64_t>> new_graph_buffer_view)
{
  RAFT_EXPECTS(!index.dataset_fd().has_value(),
               "Cannot extend a disk-backed CAGRA index. Convert it with "
               "cuvs::neighbors::hnsw::from_cagra() and load it into memory via "
               "cuvs::neighbors::hnsw::deserialize() before calling extend().");

  if (dynamic_cast<const non_owning_dataset<T, IdxT>*>(&index.data()) != nullptr &&
      !new_dataset_buffer_view.has_value()) {
    RAFT_LOG_WARN(
      "New memory space for extended dataset will be allocated while the memory space for the old "
      "dataset is allocated by user.");
  }
  const std::size_t num_new_nodes        = additional_dataset.extent(0);
  const std::size_t initial_dataset_size = index.size();
  const std::size_t new_dataset_size     = initial_dataset_size + num_new_nodes;
  const std::size_t degree               = index.graph_degree();
  const std::size_t dim                  = index.dim();

  if (new_dataset_buffer_view.has_value() &&
      static_cast<std::size_t>(new_dataset_buffer_view.value().extent(0)) != new_dataset_size) {
    RAFT_LOG_ERROR(
      "The extended dataset size (%lu) must be the initial dataset size (%lu) + additional dataset "
      "size (%lu). "
      "Please fix the memory buffer size for the extended dataset.",
      new_dataset_buffer_view.value().extent(0),
      initial_dataset_size,
      num_new_nodes);
  }

  if (new_graph_buffer_view.has_value() &&
      static_cast<std::size_t>(new_graph_buffer_view.value().extent(0)) != new_dataset_size) {
    RAFT_LOG_ERROR(
      "The extended graph size (%lu) must be the initial dataset size (%lu) + additional dataset "
      "size (%lu). "
      "Please fix the memory buffer size for the extended graph.",
      new_graph_buffer_view.value().extent(0),
      initial_dataset_size,
      num_new_nodes);
  }

  using ds_idx_type = decltype(index.data().n_rows());
  if (auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index.data());
      strided_dset != nullptr) {
    // Allocate memory space for updated graph on host
    auto updated_graph = raft::make_host_matrix<IdxT, std::int64_t>(new_dataset_size, degree);

    const auto stride    = strided_dset->stride();
    auto updated_dataset = raft::make_device_matrix<T, std::int64_t>(handle, 0, stride);
    auto updated_dataset_view =
      raft::make_device_strided_matrix_view<T, std::int64_t>(nullptr, 0, dim, stride);

    // Update dataset
    auto host_updated_dataset = raft::make_host_matrix<T, std::int64_t>(new_dataset_size, stride);

    // The padding area must be filled with zeros.!!!!!!!!!!!!!!!!!!!
    memset(host_updated_dataset.data_handle(), 0, sizeof(T) * host_updated_dataset.size());

    raft::copy_matrix(host_updated_dataset.data_handle(),
                      stride,
                      strided_dset->view().data_handle(),
                      stride,
                      dim,
                      initial_dataset_size,
                      raft::resource::get_cuda_stream(handle));
    raft::copy_matrix(host_updated_dataset.data_handle() + initial_dataset_size * stride,
                      stride,
                      additional_dataset.data_handle(),
                      additional_dataset.stride(0),
                      dim,
                      num_new_nodes,
                      raft::resource::get_cuda_stream(handle));

    if (new_dataset_buffer_view.has_value()) {
      updated_dataset_view = new_dataset_buffer_view.value();
    } else {
      // Deallocate the current dataset memory space if the dataset is `owning'.
      index.update_dataset(
        handle, raft::make_device_strided_matrix_view<const T, int64_t>(nullptr, 0, dim, stride));

      // Allocate the new dataset
      updated_dataset = raft::make_device_matrix<T, std::int64_t>(handle, new_dataset_size, stride);
      updated_dataset_view = raft::make_device_strided_matrix_view<T, std::int64_t>(
        updated_dataset.data_handle(), new_dataset_size, dim, stride);
    }

    // Copy updated dataset on host memory to device memory
    raft::copy(
      handle,
      raft::make_device_vector_view(updated_dataset_view.data_handle(), new_dataset_size * stride),
      raft::make_host_vector_view(host_updated_dataset.data_handle(), new_dataset_size * stride));

    // Add graph nodes
    cuvs::neighbors::cagra::add_graph_nodes<T, IdxT>(
      handle, raft::make_const_mdspan(updated_dataset_view), index, updated_graph.view(), params);

    // Update index dataset
    if (new_dataset_buffer_view.has_value()) {
      index.update_dataset(handle, raft::make_const_mdspan(updated_dataset_view));
    } else {
      using out_mdarray_type          = decltype(updated_dataset);
      using out_layout_type           = typename out_mdarray_type::layout_type;
      using out_container_policy_type = typename out_mdarray_type::container_policy_type;
      using out_owning_type =
        owning_dataset<T, int64_t, out_layout_type, out_container_policy_type>;
      auto out_layout = raft::make_strided_layout(updated_dataset_view.extents(),
                                                  cuda::std::array<int64_t, 2>{stride, 1});

      index.update_dataset(handle, out_owning_type{std::move(updated_dataset), out_layout});
    }

    // Update index graph
    if (new_graph_buffer_view.has_value()) {
      auto device_graph_view = new_graph_buffer_view.value();
      raft::copy(
        handle,
        raft::make_device_vector_view(device_graph_view.data_handle(), updated_graph.size()),
        raft::make_host_vector_view(updated_graph.data_handle(), updated_graph.size()));
      index.update_graph(handle, raft::make_const_mdspan(device_graph_view));
    } else {
      index.update_graph(handle, raft::make_const_mdspan(updated_graph.view()));
    }
  } else if (dynamic_cast<const cuvs::neighbors::empty_dataset<int64_t>*>(&index.data()) !=
             nullptr) {
    RAFT_FAIL(
      "cagra::extend only supports an index to which the dataset is attached. Please check if the "
      "index was built with index_param.attach_dataset_on_build = true, or if a dataset was "
      "attached after the build.");
  } else {
    RAFT_FAIL("cagra::extend only supports an uncompressed dataset index");
  }
}
}  // namespace cuvs::neighbors::cagra
