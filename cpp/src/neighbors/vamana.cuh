/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/vamana/vamana_build.cuh"
#include "detail/vamana/vamana_codebooks.cuh"
#include "detail/vamana/vamana_serialize.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/vamana.hpp>

#include <cuvs/neighbors/common.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cuvs::neighbors::vamana {

/**
 * @defgroup VAMANA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Build the VAMANA / DiskANN index from the dataset.
 *
 * The Vamana index construction algorithm is an iterative insertion based algorithm.
 * We start with an empty graph and insert batches of new nodes into the graph until
 * all nodes have been inserted. Each insertion involves:
 * - Perform GreedySearch into the current graph and collect all visited nodes.
 * - Perform RobustPrune on the list of visited nodes to get the edge list for new node.
 * - Compute the reverse edges for all edges from the newly inserted node.
 * - Combine reverse edges with existing edge lists and perform RobustPrune as needed.
 *
 * Currently only build and serialize (write to graph) is supported in cuVS. The format
 * of the serialized graph matches the DiskANN format, so search can be done with other
 * CPU DiskANN libraries as needed.
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - L2SqrtExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   vamana::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = vamana::build(res, index_params, dataset);
 *   // write graph to file for later use.
     vamana.serialize(res, filename, index);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (host or device) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed vamana index
 */
template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
index<T, IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  return cuvs::neighbors::vamana::detail::build<T, IdxT, Accessor>(res, params, dataset);
}

template <typename T, typename IdxT>
void serialize(raft::resources const& res,
               const std::string& file_prefix,
               const index<T, IdxT>& index_)
{
  cuvs::neighbors::vamana::detail::serialize<T, IdxT>(res, file_prefix, index_);
}

template <typename T>
codebook_params<T> deserialize_codebooks(const std::string& codebook_prefix, const int dim)
{
  return cuvs::neighbors::vamana::detail::deserialize_codebooks<T>(codebook_prefix, dim);
}

/** @} */  // end group vamana

}  // namespace cuvs::neighbors::vamana
