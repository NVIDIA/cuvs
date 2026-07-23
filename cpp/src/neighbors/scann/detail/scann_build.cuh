/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../detail/ann_utils.cuh"
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/scann.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>

#include <cuda_bf16.h>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/matrix/slice.cuh>
#include <raft/random/rng.cuh>

#include "scann_avq.cuh"
#include "scann_common.cuh"
#include "scann_quantize.cuh"
#include "scann_soar.cuh"

namespace cuvs::neighbors::experimental::scann::detail {
using namespace cuvs::spatial::knn::detail;  // NOLINT

/* @defgroup scann_build_detail scann build
 * @{
 */

/**
 * Helper for training kmeans centers in a balaned or unbalaned manner
 *
 * @tparam T
 * @tparam IxT
 * @tparms Accessor
 * @param res raft resources
 * @param dataset the dataset (host or device), size [n_rows, dim]
 * @param centers cluster centers, size [n_clusters, dim]
 * @param n_iters max kmeans training iterations
 * @param n_rows_train number of dataset rows for training
 * @param random_state random generator state
 * @param kmeans_type whether to use balanced or unbalanced training
 */
template <typename T,
          typename IdxT = int64_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
void train_kmeans(
  raft::resources const& res,
  raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, Accessor> dataset,
  raft::device_matrix_view<float, IdxT> centers,
  uint32_t n_iters,
  uint32_t n_rows_train,
  raft::random::RngState random_state,
  cuvs::cluster::kmeans::kmeans_type type)
{
  auto trainset = raft::make_device_matrix<T, int64_t>(res, 0, 0);

  // sample/project
  try {
    trainset = raft::make_device_matrix<T, int64_t>(res, n_rows_train, dataset.extent(1));
  } catch (raft::logic_error& e) {
    RAFT_LOG_ERROR(
      "Insufficient device memory for kmeans training set allocation. Please "
      "reduce kmeans_n_rows_train.");
    throw;
  }

  raft::matrix::sample_rows(res, random_state, dataset, trainset.view());

  if (type == cuvs::cluster::kmeans::kmeans_type::KMeansBalanced) {
    cuvs::cluster::kmeans::balanced_params kmeans_params;
    kmeans_params.n_iters = n_iters;

    cuvs::cluster::kmeans::fit(
      res, kmeans_params, raft::make_const_mdspan(trainset.view()), centers);
  } else {
    cuvs::cluster::kmeans::params kmeans_params;
    kmeans_params.max_iter   = n_iters;
    kmeans_params.n_clusters = centers.extent(0);
    kmeans_params.init       = cuvs::cluster::kmeans::params::InitMethod::Random;
    kmeans_params.tol        = 1e-5;

    float inertia  = 0.0;
    int64_t n_iter = 0;

    cuvs::cluster::kmeans::fit(res,
                               kmeans_params,
                               raft::make_const_mdspan(trainset.view()),
                               std::nullopt,
                               centers,
                               raft::make_host_scalar_view<float>(&inertia),
                               raft::make_host_scalar_view<int64_t>(&n_iter));
  }
  raft::resource::sync_stream(res);
}

/**
 * Helper for kmeans prediction
 *
 * @tparam T
 * @tparam IxT
 * @tparms Accessor
 * @param res raft resources
 * @param dataset the dataset (host or device), size [n_rows, dim]
 * @param centers cluster centers, size [n_clusters, dim]
 * @param labels cluster labels for dataset rows, size [n_rows]
 * @param batch_size number of rows per batch
 * @param enable_prefetch toggle prefetching for HtoD when dataset in host memory
 * @param copy_stream cuda stream for prefetching
 */
template <typename T,
          typename IdxT = int64_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
void predict_kmeans(
  raft::resources const& res,
  raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, Accessor> dataset,
  raft::device_matrix_view<const float, IdxT> centers,
  raft::device_vector_view<uint32_t, IdxT> labels,
  size_t batch_size,
  bool enable_prefetch,
  cudaStream_t copy_stream)
{
  cuvs::cluster::kmeans::balanced_params kmeans_params;

  auto device_memory = raft::resource::get_workspace_resource_ref(res);

  auto dataset_vec_batches = utils::make_batch_load_iterator<T>(res,
                                                                dataset.data_handle(),
                                                                dataset.extent(0),
                                                                dataset.extent(1),
                                                                batch_size,
                                                                copy_stream,
                                                                device_memory,
                                                                enable_prefetch);

  dataset_vec_batches.reset();
  dataset_vec_batches.prefetch_next_batch();

  RAFT_LOG_DEBUG("Batched kmeans prediction.\n");
  for (const auto& batch : dataset_vec_batches) {
    auto batch_view = raft::make_device_matrix_view<const T, int64_t>(
      batch.data(), batch.size(), dataset.extent(1));

    auto batch_labels_view = raft::make_device_vector_view<uint32_t, int64_t>(
      labels.data_handle() + batch.offset(), batch.size());

    cuvs::cluster::kmeans::predict(res, kmeans_params, batch_view, centers, batch_labels_view);

    dataset_vec_batches.prefetch_next_batch();

    // Make sure work on device is finished before swapping buffers
    raft::resource::sync_stream(res);
  }
}

template <typename T,
          typename IdxT = int64_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
index<T, IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, Accessor> dataset)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res);
  IdxT dim            = dataset.extent(1);

  RAFT_LOG_DEBUG("Creating empty index");

  index<T, IdxT> idx(res, params, dataset.extent(0), dataset.extent(1));
  raft::device_matrix_view<float, int64_t> centers_view   = idx.centers();
  raft::device_vector_view<uint32_t, int64_t> labels_view = idx.labels();
  raft::random::RngState random_state{137};

  // setup params for kmeans prediction + quantization
  auto device_memory = raft::resource::get_workspace_resource_ref(res);

  constexpr size_t kReasonableMaxBatchSize = 65536;
  size_t max_batch_size = std::min<size_t>(dataset.extent(0), kReasonableMaxBatchSize);
  auto copy_stream      = raft::resource::get_cuda_stream(res);
  bool enable_prefetch  = false;

  if (res.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL)) {
    if (raft::resource::get_stream_pool_size(res) >= 1) {
      enable_prefetch = true;

      copy_stream = raft::resource::get_stream_from_stream_pool(res);
    }
  }

  train_kmeans(res,
               dataset,
               centers_view,
               params.kmeans_n_iters,
               params.kmeans_n_rows_train,
               random_state,
               cuvs::cluster::kmeans::kmeans_type::KMeansBalanced);

  predict_kmeans(res,
                 dataset,
                 raft::make_const_mdspan(centers_view),
                 labels_view,
                 max_batch_size,
                 enable_prefetch,
                 copy_stream);

  // AVQ update of KMeans centroids
  apply_avq(res,
            dataset,
            centers_view,
            raft::make_const_mdspan(labels_view),
            params.partitioning_eta,
            copy_stream);

  if (params.n_coarse_clusters > 0) {
    raft::device_matrix_view<float, int64_t> coarse_centers   = idx.coarse_centers();
    raft::device_vector_view<uint32_t, int64_t> coarse_labels = idx.coarse_labels();

    // Coarse clusters are trained in an unbalanced manner, which gives better recall
    train_kmeans(res,
                 raft::make_const_mdspan(centers_view),
                 coarse_centers,
                 params.kmeans_n_iters,
                 centers_view.extent(0),
                 random_state,
                 cuvs::cluster::kmeans::kmeans_type::KMeans);

    size_t top_level_batch_size = std::min<size_t>(centers_view.extent(0), kReasonableMaxBatchSize);

    predict_kmeans(res,
                   raft::make_const_mdspan(centers_view),
                   raft::make_const_mdspan(coarse_centers),
                   coarse_labels,
                   top_level_batch_size,
                   enable_prefetch,
                   copy_stream);

    // AVQ coarse centers are not rescaled
    apply_avq(res,
              raft::make_const_mdspan(centers_view),
              coarse_centers,
              raft::make_const_mdspan(coarse_labels),
              params.partitioning_eta,
              copy_stream);

    auto residuals = compute_residuals<T, uint32_t>(res,
                                                    raft::make_const_mdspan(centers_view),
                                                    raft::make_const_mdspan(coarse_centers),
                                                    coarse_labels);

    compute_soar_labels<T, uint32_t>(res,
                                     raft::make_const_mdspan(centers_view),
                                     raft::make_const_mdspan(residuals.view()),
                                     coarse_centers,
                                     coarse_labels,
                                     idx.coarse_soar_labels(),
                                     params.soar_lambda);
  }

  raft::device_vector_view<uint32_t, int64_t> soar_labels_view = idx.soar_labels();

  // Train PQ codebooks
  RAFT_LOG_DEBUG("Train PQ Codebooks\n");

  // Limit PQ training rows to 100k. IVF-PQ/VPQ limit to 65k as recall
  // doesn't increase significantly. ScaNN configs in common benchmarks use 100k,
  // so staying consistent with that.
  constexpr int64_t kMaxPQTrainRows = 100000;
  int64_t pq_n_rows_train           = std::min(params.pq_n_rows_train, kMaxPQTrainRows);

  auto trainset_residuals = sample_training_residuals(res,
                                                      random_state,
                                                      dataset,
                                                      raft::make_const_mdspan(centers_view),
                                                      raft::make_const_mdspan(labels_view),
                                                      pq_n_rows_train);

  int num_subspaces    = dataset.extent(1) / params.pq_dim;
  int dim_per_subspace = params.pq_dim;
  int num_clusters     = 1 << params.pq_bits;

  cuvs::preprocessing::quantize::pq::params pq_build_params(
    params.pq_bits,
    num_subspaces,
    true,
    false,
    0,
    params.pq_train_iters,
    cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
    pq_n_rows_train / num_clusters,
    1024);

  auto pq_quantizer = cuvs::preprocessing::quantize::pq::build(
    res, pq_build_params, raft::make_const_mdspan(trainset_residuals.view()));

  auto dataset_vec_batches = utils::make_batch_load_iterator<T>(res,
                                                                dataset.data_handle(),
                                                                dataset.extent(0),
                                                                dataset.extent(1),
                                                                max_batch_size,
                                                                copy_stream,
                                                                device_memory,
                                                                enable_prefetch);

  dataset_vec_batches.reset();
  dataset_vec_batches.prefetch_next_batch();

  // Quantize residuals for both normal and SOAR assignments
  RAFT_LOG_DEBUG("Quantize residuals");
  for (const auto& batch : dataset_vec_batches) {
    auto batch_view = raft::make_device_matrix_view<const T, int64_t>(
      batch.data(), batch.size(), dataset.extent(1));

    auto batch_labels_view = raft::make_device_vector_view<const uint32_t, int64_t>(
      labels_view.data_handle() + batch.offset(), batch.size());

    auto batch_soar_labels_view = raft::make_device_vector_view<uint32_t, int64_t>(
      soar_labels_view.data_handle() + batch.offset(), batch.size());

    // Compute residuals for use in SOAR + quantization
    auto avq_residuals = compute_residuals<T, uint32_t>(
      res, batch_view, raft::make_const_mdspan(centers_view), batch_labels_view);

    // Compute SOAR labels.
    // We compute SOAR labels in this loop to eliminate one HtoD copy of the full dataset.
    compute_soar_labels<T, uint32_t>(res,
                                     batch_view,
                                     raft::make_const_mdspan(avq_residuals.view()),
                                     centers_view,
                                     batch_labels_view,
                                     batch_soar_labels_view,
                                     params.soar_lambda);

    // Compute and quantize residuals using the public PQ API
    int64_t codes_dim = cuvs::preprocessing::quantize::pq::get_quantized_dim(pq_build_params);
    auto avq_quant    = raft::make_device_matrix<uint8_t, IdxT>(res, batch.size(), codes_dim);
    cuvs::preprocessing::quantize::pq::transform(
      res, pq_quantizer, raft::make_const_mdspan(avq_residuals.view()), avq_quant.view());

    // Compute and quantize SOAR residuals
    auto soar_residuals =
      compute_residuals<T, uint32_t>(res,
                                     batch_view,
                                     raft::make_const_mdspan(centers_view),
                                     raft::make_const_mdspan(batch_soar_labels_view));

    auto soar_quant = raft::make_device_matrix<uint8_t, IdxT>(res, batch.size(), codes_dim);
    cuvs::preprocessing::quantize::pq::transform(
      res, pq_quantizer, raft::make_const_mdspan(soar_residuals.view()), soar_quant.view());

    // Prefetch next batch
    dataset_vec_batches.prefetch_next_batch();
    // unpack codes
    if (pq_quantizer.params_quantizer.pq_bits == 8) {
      // Copy unpacked codes to host
      // TODO (rmaschal): these copies are blocking and not overlapped
      raft::copy(idx.quantized_residuals().data_handle() + batch.offset() * num_subspaces,
                 avq_quant.data_handle(),
                 avq_quant.size(),
                 stream);

      raft::copy(idx.quantized_soar_residuals().data_handle() + batch.offset() * num_subspaces,
                 soar_quant.data_handle(),
                 soar_quant.size(),
                 stream);
    } else {
      auto quantized_residuals =
        raft::make_device_matrix<uint8_t, IdxT>(res, batch.size(), num_subspaces);
      auto quantized_soar_residuals =
        raft::make_device_matrix<uint8_t, IdxT>(res, batch.size(), num_subspaces);

      unpack_codes(res,
                   quantized_residuals.view(),
                   raft::make_const_mdspan(avq_quant.view()),
                   params.pq_bits,
                   num_subspaces);
      unpack_codes(res,
                   quantized_soar_residuals.view(),
                   raft::make_const_mdspan(soar_quant.view()),
                   params.pq_bits,
                   num_subspaces);
      raft::copy(res,
                 raft::make_host_vector_view(
                   idx.quantized_residuals().data_handle() + batch.offset() * num_subspaces,
                   quantized_residuals.size()),
                 raft::make_device_vector_view<const uint8_t>(quantized_residuals.data_handle(),
                                                              quantized_residuals.size()));

      raft::copy(res,
                 raft::make_host_vector_view(
                   idx.quantized_soar_residuals().data_handle() + batch.offset() * num_subspaces,
                   quantized_soar_residuals.size()),
                 raft::make_device_vector_view<const uint8_t>(
                   quantized_soar_residuals.data_handle(), quantized_soar_residuals.size()));
    }

    // quantize dataset to bfloat16, if enabled. Similar to SOAR, quantization
    // is performed in this loop to improve locality
    // TODO (rmaschal): Might be more efficient to do on CPU, to avoid DtoH copy
    if (params.reordering_bf16) {
      auto bf16_dataset =
        raft::make_device_matrix<int16_t, int64_t>(res, batch_view.extent(0), dim);
      quantize_bfloat16(
        res, batch_view, bf16_dataset.view(), params.reordering_noise_shaping_threshold);
      raft::copy(res,
                 raft::make_host_vector_view(
                   idx.bf16_dataset().data_handle() + batch.offset() * dim, bf16_dataset.size()),
                 raft::make_device_vector_view<const int16_t>(bf16_dataset.data_handle(),
                                                              bf16_dataset.size()));
    }

    // Make sure work on device is finished before swapping buffers
    raft::resource::sync_stream(res);
  }

  // Codebooks from VPQ have the shape [subspace idx, subspace dim, code]
  // This converts the codebook into matrix format for easy interoperability
  // with open-source ScaNN search
  auto full_codebook_view = pq_quantizer.vpq_codebooks.pq_code_book.view();

  raft::linalg::map_offset(
    res,
    idx.pq_codebook(),
    [full_codebook_view, num_subspaces, dim_per_subspace, num_clusters] __device__(size_t i) {
      int codebook_dim   = num_subspaces * dim_per_subspace;
      int row_idx        = i / codebook_dim;
      int el_idx         = i % codebook_dim;
      int subspace_id    = el_idx / dim_per_subspace;
      int id_in_subspace = el_idx % dim_per_subspace;

      return full_codebook_view(row_idx + subspace_id * num_clusters, id_in_subspace);
    });

  raft::resource::sync_stream(res);

  return idx;
}

/**
 * @}
 */

}  // namespace cuvs::neighbors::experimental::scann::detail
