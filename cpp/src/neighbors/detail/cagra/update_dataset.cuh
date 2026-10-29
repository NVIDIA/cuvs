/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace cuvs::neighbors::cagra::detail {

template <typename T, typename IdxT>
CUVS_HIDDEN auto convert_standard_to_padded_index(
  raft::resources const& res,
  index<T, IdxT, cuvs::neighbors::device_standard_dataset_view<T, int64_t>> const& standard_idx,
  cuvs::neighbors::device_padded_dataset_view<T, int64_t> const& padded_dataset)
  -> device_padded_index<T, IdxT>
{
  RAFT_EXPECTS(padded_dataset.n_rows() == standard_idx.size(),
               "Padded dataset row count must match the index size");

  device_padded_index<T, IdxT> out(res, standard_idx.metric());
  if (standard_idx.graph().extent(0) > 0) {
    using graph_index_type =
      typename index<T, IdxT, cuvs::neighbors::device_standard_dataset_view<T, int64_t>>::
        graph_index_type;
    auto graph_host = raft::make_host_matrix<graph_index_type, int64_t>(
      standard_idx.graph().extent(0), standard_idx.graph().extent(1));
    raft::copy(graph_host.data_handle(),
               standard_idx.graph().data_handle(),
               standard_idx.graph().size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    out.update_graph(res, raft::make_const_mdspan(graph_host.view()));
  }
  if (standard_idx.source_indices().has_value()) {
    out.update_source_indices(res, standard_idx.source_indices().value());
  }
  update_dataset(res, out, padded_dataset);
  return out;
}

}  // namespace cuvs::neighbors::cagra::detail
