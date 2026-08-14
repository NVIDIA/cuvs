/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"
#include "../detail/ann_utils.cuh"
#include "../ivf_common.cuh"
#include "ivf_pq_build.cuh"
#include "ivf_pq_process_and_fill_codes.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

template <uint32_t BlockSize, uint32_t PqBits, typename IdxT>
__launch_bounds__(BlockSize) static __global__ void transform_codes_kernel(
  raft::device_matrix_view<float, IdxT> dataset_residual,
  raft::device_vector_view<uint32_t, IdxT> output_labels,
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> output_dataset,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
  codebook_gen codebook_kind)
{
  constexpr uint32_t kSubWarpSize = std::min<uint32_t>(raft::WarpSize, 1u << PqBits);
  using subwarp_align             = raft::Pow2<kSubWarpSize>;
  const uint32_t lane_id          = subwarp_align::mod(threadIdx.x);
  const IdxT row_ix = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= dataset_residual.extent(0)) { return; }

  const uint32_t cluster_ix = output_labels[row_ix];
  const uint32_t pq_dim     = dataset_residual.extent(1) / pq_centers.extent(1);

  auto encoder =
    encode_vectors<kSubWarpSize, IdxT>{pq_centers, dataset_residual, codebook_kind, cluster_ix};

  write_vector_flat<PqBits, kSubWarpSize, IdxT>(
    output_dataset.data_handle(), row_ix, row_ix, pq_dim, output_dataset.extent(1), encoder);
}

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

  // Launch kernel to transform the code output
  constexpr uint32_t kBlockSize  = 256;
  const uint32_t threads_per_vec = std::min<uint32_t>(raft::WarpSize, index.pq_book_size());
  dim3 blocks(raft::div_rounding_up_safe<IdxT>(n_rows, kBlockSize / threads_per_vec), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4: return transform_codes_kernel<kBlockSize, 4, IdxT>;
      case 5: return transform_codes_kernel<kBlockSize, 5, IdxT>;
      case 6: return transform_codes_kernel<kBlockSize, 6, IdxT>;
      case 7: return transform_codes_kernel<kBlockSize, 7, IdxT>;
      case 8: return transform_codes_kernel<kBlockSize, 8, IdxT>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }(index.pq_bits());

  kernel<<<blocks, threads, 0, raft::resource::get_cuda_stream(res)>>>(dataset_residuals.view(),
                                                                       output_labels,
                                                                       output_dataset,
                                                                       index.pq_centers(),
                                                                       index.codebook_kind());
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
  auto stream           = raft::resource::get_cuda_stream(res);
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
