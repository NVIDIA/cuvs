/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"
#include "../detail/ann_utils.cuh"
#include "../ivf_common.cuh"
#include "ivf_pq_build.cuh"
#include "ivf_pq_encode.hpp"
#include "ivf_pq_process_and_fill_codes.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

template <typename T, typename IdxT>
void transform_batch(raft::resources const& res,
                     const index<IdxT>& index,
                     raft::device_matrix_view<const float, IdxT, raft::row_major> cluster_centers,
                     raft::device_matrix_view<const T, IdxT, raft::row_major> dataset,
                     raft::device_vector_view<uint32_t, IdxT> output_labels,
                     raft::device_matrix_view<uint8_t, IdxT, raft::row_major> output_dataset)
{
  IdxT n_rows                       = dataset.extent(0);
  rmm::device_async_resource_ref mr = raft::resource::get_workspace_resource_ref(res);

  // Compute the labels for each vector
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.metric = coarse_clustering_metric(index.metric());

  cuvs::cluster::kmeans_balanced::predict(
    res, kmeans_params, dataset, cluster_centers, output_labels, utils::mapping<float>{});

  // Compute the residuals for each vector in the dataset
  auto dataset_residuals =
    raft::make_device_mdarray<float>(res, mr, raft::make_extents<IdxT>(n_rows, index.rot_dim()));

  flat_compute_residuals<T, IdxT>(res,
                                  dataset_residuals.data_handle(),
                                  n_rows,
                                  index.rotation_matrix(),
                                  index.centers(),
                                  dataset.data_handle(),
                                  output_labels.data_handle(),
                                  mr,
                                  index.metric());

  encode_residuals(dataset_residuals.view(),
                   output_labels,
                   output_dataset,
                   index.pq_centers(),
                   index.codebook_kind(),
                   index.pq_bits(),
                   index.pq_book_size(),
                   raft::resource::get_cuda_stream(res).get());
}

template <typename T, typename IdxT>
void transform(raft::resources const& res,
               const index<IdxT>& index,
               raft::device_matrix_view<const T, IdxT, raft::row_major> dataset,
               raft::device_vector_view<uint32_t, IdxT> output_labels,
               raft::device_matrix_view<uint8_t, IdxT, raft::row_major> output_dataset)
{
  IdxT n_rows = dataset.extent(0);
  RAFT_EXPECTS(output_labels.extent(0) == n_rows, "incorrect number of rows in output_labels");
  RAFT_EXPECTS(output_dataset.extent(0) == n_rows, "incorrect number of rows in output_dataset");

  RAFT_EXPECTS(
    output_dataset.extent(1) == raft::ceildiv<uint32_t>(index.pq_dim() * index.pq_bits(), 8),
    "incorrect number of cols in output_dataset");

  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_pq::transform(n_rows = %u, dim = %u)", n_rows, dataset.extent(1));

  rmm::device_async_resource_ref mr = raft::resource::get_workspace_resource_ref(res);

  // The cluster centers in the index are stored padded, which is not acceptable by
  // the kmeans_balanced::predict. Thus, we need the restructuring raft::copy.
  const auto n_clusters = index.n_lists();

  auto cluster_centers =
    raft::make_device_mdarray<float>(res, mr, raft::make_extents<IdxT>(n_clusters, index.dim()));
  extract_centers(res, index, cluster_centers.view());

  // Determine if a stream pool exist and make sure there is at least one stream in it so we
  // could use the stream for kernel/copy overlapping by enabling prefetch.
  auto copy_stream     = raft::resource::get_cuda_stream(res);  // Using the main stream by default
  bool enable_prefetch = false;
  if (res.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL)) {
    if (raft::resource::get_stream_pool_size(res) >= 1) {
      enable_prefetch = true;
      copy_stream     = raft::resource::get_stream_from_stream_pool(res);
    }
  }

  constexpr size_t max_batch_size              = 65536;
  rmm::device_async_resource_ref device_memory = raft::resource::get_workspace_resource_ref(res);

  auto vec_batches = utils::make_batch_load_iterator<T>(res,
                                                        dataset.data_handle(),
                                                        n_rows,
                                                        IdxT{index.dim()},
                                                        max_batch_size,
                                                        copy_stream,
                                                        device_memory,
                                                        enable_prefetch);

  vec_batches.prefetch_next_batch();
  for (const auto& batch : vec_batches) {
    auto batch_dataset =
      raft::make_device_matrix_view<const T, IdxT>(batch.data(), batch.size(), index.dim());
    auto batch_labels = raft::make_device_vector_view<uint32_t, IdxT>(
      output_labels.data_handle() + batch.offset(), batch.size());
    auto batch_output_dataset = raft::make_device_matrix_view<uint8_t, IdxT>(
      output_dataset.data_handle() + batch.offset() * output_dataset.extent(1),
      batch.size(),
      output_dataset.extent(1));

    transform_batch(res,
                    index,
                    raft::make_const_mdspan(cluster_centers.view()),
                    batch_dataset,
                    batch_labels,
                    batch_output_dataset);
    vec_batches.prefetch_next_batch();
    raft::resource::sync_stream(res);
  }
}
}  // namespace cuvs::neighbors::ivf_pq::detail
