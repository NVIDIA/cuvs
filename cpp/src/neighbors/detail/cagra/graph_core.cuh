/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cagra_helpers.hpp"
#include "graph_shared.cuh"
#include "utils.hpp"

#include <raft/core/copy.cuh>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>

// TODO: This shouldn't be invoking anything from spatial/knn
#include "../../../core/nvtx.hpp"
#include "../../../core/omp_wrapper.hpp"
#include "../ann_utils.cuh"

#include <cuvs/neighbors/cagra.hpp>

#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/integer_utils.hpp>

#include <cuda_fp16.h>

#include <float.h>
#include <sys/time.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <climits>
#include <iostream>
#include <memory>
#include <random>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail::graph {
namespace {
inline double cur_time(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return ((double)tv.tv_sec + (double)tv.tv_usec * 1e-6);
}

template <typename DataT,
          typename IdxT       = uint32_t,
          typename d_accessor = raft::host_device_accessor<cuda::std::default_accessor<DataT>,
                                                           raft::memory_type::device>,
          typename g_accessor =
            raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::host>>
void sort_knn_graph(
  raft::resources const& res,
  const cuvs::distance::DistanceType metric,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, d_accessor> dataset,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor> knn_graph)
{
  static_assert(std::is_same_v<IdxT, uint32_t>, "CAGRA graph indices must be uint32_t");
  RAFT_EXPECTS(dataset.extent(0) == knn_graph.extent(0),
               "dataset size is expected to have the same number of graph index size");
  RAFT_EXPECTS(
    metric == cuvs::distance::DistanceType::InnerProduct ||
      metric == cuvs::distance::DistanceType::CosineExpanded ||
      metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::BitwiseHamming ||
      metric == cuvs::distance::DistanceType::L1,
    "Unsupported metric. Only InnerProduct, CosineExpanded, L2Expanded, BitwiseHamming and L1 are "
    "supported");
  const uint64_t dataset_size = dataset.extent(0);
  const uint64_t dataset_dim  = dataset.extent(1);
  const DataT* dataset_ptr    = dataset.data_handle();

  const IdxT graph_size             = dataset_size;
  const uint64_t input_graph_degree = knn_graph.extent(1);
  IdxT* const input_graph_ptr       = knn_graph.data_handle();

  auto large_tmp_mr = raft::resource::get_large_workspace_resource_ref(res);

  auto d_input_graph = raft::make_device_mdarray<IdxT>(
    res, large_tmp_mr, raft::make_extents<int64_t>(graph_size, input_graph_degree));

  //
  // Sorting kNN graph
  //
  const double time_sort_start = cur_time();
  RAFT_LOG_DEBUG("# Sorting kNN Graph on GPUs ");

  auto d_dataset = raft::make_device_mdarray<DataT>(
    res, large_tmp_mr, raft::make_extents<int64_t>(dataset_size, dataset_dim));
  raft::copy(res, d_dataset.view(), dataset);

  raft::copy(res, d_input_graph.view(), knn_graph);

  RAFT_LOG_DEBUG(".");
  launch_sort_knn_graph(res,
                        metric,
                        d_dataset.data_handle(),
                        static_cast<uint32_t>(dataset_size),
                        static_cast<uint32_t>(dataset_dim),
                        d_input_graph.data_handle(),
                        static_cast<uint32_t>(input_graph_degree));
  raft::resource::sync_stream(res);
  RAFT_LOG_DEBUG(".");
  raft::copy(res, knn_graph, raft::make_const_mdspan(d_input_graph.view()));
  RAFT_LOG_DEBUG("\n");

  const double time_sort_end = cur_time();
  RAFT_LOG_DEBUG("# Sorting kNN graph time: %.1lf sec\n", time_sort_end - time_sort_start);
}

}  // namespace

template <typename IdxT = uint32_t,
          typename AccessorKnnGraph =
            raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::host>,
          typename AccessorOutputGraph =
            raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::host>>
void optimize(
  raft::resources const& res_const,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorKnnGraph> knn_graph,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph> new_graph,
  const bool guarantee_connectivity       = true,
  const bool use_gpu_for_mst_optimization = true);

}  // namespace cuvs::neighbors::cagra::detail::graph
