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

template <typename T, typename IdxT, typename HostViewT>
  requires cuvs::neighbors::is_host_dataset_view_v<HostViewT>
CUVS_HIDDEN auto convert_host_to_device_index(raft::resources const& res,
                                              index<T, IdxT, HostViewT> const& src)
  -> index<T, IdxT, cuvs::neighbors::device_counterpart_t<HostViewT>>
{
  using device_view_type = cuvs::neighbors::device_counterpart_t<HostViewT>;
  using graph_index_type = typename index<T, IdxT, HostViewT>::graph_index_type;

  index<T, IdxT, device_view_type> out(res, src.metric());
  if (src.graph().size() > 0) {
    auto graph_host = raft::make_host_matrix<graph_index_type, int64_t>(src.graph().extent(0),
                                                                        src.graph().extent(1));
    raft::copy(graph_host.data_handle(),
               src.graph().data_handle(),
               src.graph().size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    out.update_graph(res, raft::make_const_mdspan(graph_host.view()));
  }
  return out;
}

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
  out.update_device_dataset_same_layout(res, padded_dataset);
  return out;
}

template <typename T, typename IdxT, typename IndexViewT>
  requires cuvs::neighbors::ann_dataset_view<IndexViewT>
CUVS_HIDDEN auto convert_dense_to_vpq_f16_index(
  raft::resources const& res,
  index<T, IdxT, IndexViewT> const& src,
  cuvs::neighbors::device_vpq_dataset_view<half, int64_t> const& vpq_dataset)
  -> vpq_f16_index<T, IdxT>
{
  RAFT_EXPECTS(vpq_dataset.n_rows() == src.size(),
               "VPQ dataset row count must match the index size");

  vpq_f16_index<T, IdxT> out(res, src.metric());
  if (src.graph().extent(0) > 0) {
    using graph_index_type = typename index<T, IdxT, IndexViewT>::graph_index_type;
    auto graph_host = raft::make_host_matrix<graph_index_type, int64_t>(src.graph().extent(0),
                                                                        src.graph().extent(1));
    raft::copy(graph_host.data_handle(),
               src.graph().data_handle(),
               src.graph().size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    out.update_graph(res, raft::make_const_mdspan(graph_host.view()));
  }
  if (src.source_indices().has_value()) {
    out.update_source_indices(res, src.source_indices().value());
  }
  out.update_device_dataset_same_layout(res, vpq_dataset);
  return out;
}

template <typename T, typename IdxT, typename IndexViewT>
  requires cuvs::neighbors::ann_dataset_view<IndexViewT>
CUVS_HIDDEN auto attach_dataset(
  raft::resources const& res,
  index<T, IdxT, IndexViewT> const& idx,
  cuvs::neighbors::device_padded_dataset_view<T, int64_t> const& device_padded_dataset)
  -> device_padded_index<T, IdxT>
{
  RAFT_EXPECTS(device_padded_dataset.n_rows() == idx.size(),
               "Padded dataset row count must match the index size");

  if constexpr (cuvs::neighbors::is_host_standard_dataset_view_v<IndexViewT>) {
    auto dev_std = convert_host_to_device_index(res, idx);
    return convert_standard_to_padded_index(res, dev_std, device_padded_dataset);
  } else if constexpr (cuvs::neighbors::is_host_padded_dataset_view_v<IndexViewT>) {
    auto dev_pad = convert_host_to_device_index(res, idx);
    dev_pad.update_device_dataset_same_layout(res, device_padded_dataset);
    return dev_pad;
  } else if constexpr (cuvs::neighbors::is_device_standard_dataset_view_v<IndexViewT>) {
    return convert_standard_to_padded_index(res, idx, device_padded_dataset);
  } else if constexpr (cuvs::neighbors::is_device_padded_dataset_view_v<IndexViewT>) {
    RAFT_LOG_WARN(
      "cagra::attach_dataset called with an already device-padded index. "
      "To avoid an unnecessary index copy, call "
      "index.update_device_dataset_same_layout(res, device_padded_dataset) "
      "directly on the original index.");
    RAFT_FAIL(
      "cagra::attach_dataset: device_padded_index input is not supported in this overload. "
      "Call index.update_device_dataset_same_layout(res, device_padded_dataset) directly.");
  } else {
    static_assert(!sizeof(IndexViewT), "Unsupported CAGRA index dataset view type");
  }
}

template <typename T, typename IdxT, typename IndexViewT>
  requires cuvs::neighbors::ann_dataset_view<IndexViewT>
CUVS_HIDDEN auto attach_dataset(
  raft::resources const& res,
  index<T, IdxT, IndexViewT> const& idx,
  cuvs::neighbors::device_vpq_dataset_view<half, int64_t> const& vpq_dataset)
  -> vpq_f16_index<T, IdxT>
{
  if constexpr (cuvs::neighbors::is_device_vpq_f16_dataset_view_v<IndexViewT>) {
    RAFT_LOG_WARN(
      "cagra::attach_dataset called with an already vpq_f16 index. "
      "To avoid an unnecessary index copy, call "
      "index.update_device_dataset_same_layout(res, vpq_dataset) "
      "directly on the original index.");
    RAFT_FAIL(
      "cagra::attach_dataset: vpq_f16_index input is not supported in this overload. "
      "Call index.update_device_dataset_same_layout(res, vpq_dataset) directly.");
  } else {
    return convert_dense_to_vpq_f16_index(res, idx, vpq_dataset);
  }
}

}  // namespace cuvs::neighbors::cagra::detail
