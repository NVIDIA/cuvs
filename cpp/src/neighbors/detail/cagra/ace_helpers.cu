/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra_ace.cuh"
#include <cuvs/core/cuda_fp16.hpp>

namespace cuvs::neighbors::cagra::detail {

template <typename GraphIdxT, typename IdxT>
__global__ void ace_adjust_sub_graph_ids_disk_kernel(const GraphIdxT* sub_search_graph,
                                                     IdxT* adjusted_search_graph,
                                                     size_t graph_edges,
                                                     size_t core_sub_dataset_size,
                                                     IdxT core_partition_offset,
                                                     const IdxT* augmented_reordered_ids)
{
  const size_t edge_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (edge_id >= graph_edges) { return; }

  const GraphIdxT neighbor = sub_search_graph[edge_id];
  if (static_cast<size_t>(neighbor) < core_sub_dataset_size) {
    adjusted_search_graph[edge_id] = static_cast<IdxT>(neighbor) + core_partition_offset;
  } else {
    adjusted_search_graph[edge_id] =
      augmented_reordered_ids[static_cast<size_t>(neighbor) - core_sub_dataset_size];
  }
}

template <typename IdxT, typename GraphIdxT>
void ace_adjust_sub_graph_ids_disk(
  raft::resources const& res,
  size_t core_sub_dataset_size,
  size_t augmented_sub_dataset_size,
  size_t graph_degree,
  size_t partition_id,
  raft::device_matrix_view<const GraphIdxT, int64_t, raft::row_major> sub_search_graph,
  raft::device_matrix_view<IdxT, int64_t, raft::row_major> adjusted_search_graph,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_partition_offsets,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> augmented_partition_offsets,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> augmented_backward_mapping,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_forward_mapping)
{
  RAFT_EXPECTS(static_cast<size_t>(sub_search_graph.extent(0)) >= core_sub_dataset_size,
               "ACE: source graph has fewer rows than the core partition");
  RAFT_EXPECTS(static_cast<size_t>(sub_search_graph.extent(1)) == graph_degree,
               "ACE: source graph degree does not match the requested graph degree");
  RAFT_EXPECTS(static_cast<size_t>(adjusted_search_graph.extent(0)) == core_sub_dataset_size &&
                 static_cast<size_t>(adjusted_search_graph.extent(1)) == graph_degree,
               "ACE: adjusted graph shape does not match the core partition");

  const size_t augmented_offset = augmented_partition_offsets(partition_id);
  RAFT_EXPECTS(augmented_offset + augmented_sub_dataset_size <=
                 static_cast<size_t>(augmented_backward_mapping.extent(0)),
               "ACE: augmented partition exceeds the backward mapping");

  auto augmented_reordered_ids = raft::make_host_vector<IdxT, int64_t>(augmented_sub_dataset_size);
#pragma omp parallel for
  for (size_t i = 0; i < augmented_sub_dataset_size; i++) {
    const size_t original_id   = augmented_backward_mapping(augmented_offset + i);
    augmented_reordered_ids(i) = core_forward_mapping(original_id);
  }

  auto augmented_reordered_ids_dev =
    raft::make_device_vector<IdxT, int64_t>(res, augmented_sub_dataset_size);
  raft::copy(res, augmented_reordered_ids_dev.view(), augmented_reordered_ids.view());

  const size_t graph_edges = core_sub_dataset_size * graph_degree;
  if (graph_edges == 0) { return; }

  constexpr uint32_t block_size = 256;
  const dim3 grid_size(raft::ceildiv(graph_edges, static_cast<size_t>(block_size)));
  ace_adjust_sub_graph_ids_disk_kernel<<<grid_size,
                                         block_size,
                                         0,
                                         raft::resource::get_cuda_stream(res).get()>>>(
    sub_search_graph.data_handle(),
    adjusted_search_graph.data_handle(),
    graph_edges,
    core_sub_dataset_size,
    core_partition_offsets(partition_id),
    augmented_reordered_ids_dev.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::resource::sync_stream(res);
}

template void
cuvs::neighbors::cagra::detail::ace_adjust_sub_graph_ids_disk<unsigned int, unsigned int>(
  raft::resources const&,
  unsigned long,
  unsigned long,
  unsigned long,
  unsigned long,
  ::cuda::std::mdspan<unsigned int const,
                      ::cuda::std::extents<long, 18446744073709551615ul, 18446744073709551615ul>,
                      ::cuda::std::layout_right,
                      raft::host_device_accessor<::cuda::std::default_accessor<unsigned int const>,
                                                 (raft::memory_type)2>>,
  ::cuda::std::mdspan<
    unsigned int,
    ::cuda::std::extents<long, 18446744073709551615ul, 18446744073709551615ul>,
    ::cuda::std::layout_right,
    raft::host_device_accessor<::cuda::std::default_accessor<unsigned int>, (raft::memory_type)2>>,
  ::cuda::std::mdspan<
    unsigned int,
    ::cuda::std::extents<long, 18446744073709551615ul>,
    ::cuda::std::layout_right,
    raft::host_device_accessor<::cuda::std::default_accessor<unsigned int>, (raft::memory_type)0>>,
  ::cuda::std::mdspan<
    unsigned int,
    ::cuda::std::extents<long, 18446744073709551615ul>,
    ::cuda::std::layout_right,
    raft::host_device_accessor<::cuda::std::default_accessor<unsigned int>, (raft::memory_type)0>>,
  ::cuda::std::mdspan<
    unsigned int,
    ::cuda::std::extents<long, 18446744073709551615ul>,
    ::cuda::std::layout_right,
    raft::host_device_accessor<::cuda::std::default_accessor<unsigned int>, (raft::memory_type)0>>,
  ::cuda::std::mdspan<
    unsigned int,
    ::cuda::std::extents<long, 18446744073709551615ul>,
    ::cuda::std::layout_right,
    raft::host_device_accessor<::cuda::std::default_accessor<unsigned int>, (raft::memory_type)0>>);

}  // namespace cuvs::neighbors::cagra::detail
