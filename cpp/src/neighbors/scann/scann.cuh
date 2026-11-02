/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/scann_build.cuh"
#include "detail/scann_serialize.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/scann.hpp>

#include <cuvs/neighbors/common.hpp>

namespace cuvs::neighbors::experimental::scann {

/**
 * @defgroup ScaNN ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Build the ScaNN / DiskANN index from the dataset.
 *
 * TODO - add brief explanation of the algo
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   scann::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = scann::build(res, index_params, dataset);
 *   // write graph to file for later use.
     scann.serialize(res, filename, index);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (host or device) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed scann index
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
  return cuvs::neighbors::experimental::scann::detail::build<T, IdxT, Accessor>(
    res, params, dataset);
}

template <typename T, typename IdxT>
void serialize(raft::resources const& res,
               const std::string& file_prefix,
               const index<T, IdxT>& index_)
{
  cuvs::neighbors::experimental::scann::detail::serialize<T, IdxT>(res, file_prefix, index_);
}

/** @} */  // end group scann

}  // namespace cuvs::neighbors::experimental::scann
