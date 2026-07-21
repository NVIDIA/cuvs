/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../core/nvtx.hpp"
#include "../../../preprocessing/quantize/vpq_build-ext.cuh"
#include "../../ivf_pq/ivf_pq_fp16_overflow.cuh"
#include "graph_core.cuh"

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/integer_utils.hpp>

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <cuvs/neighbors/refine.hpp>
#include <cuvs/util/file_io.hpp>
#include <cuvs/util/host_memory.hpp>

// TODO: This shouldn't be calling spatial/knn APIs
#include "../ann_utils.cuh"

#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <omp.h>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>

namespace cuvs::neighbors::cagra::detail {

// Helpers to convert bytes to MiB and GiB
constexpr double to_mib(size_t bytes) { return static_cast<double>(bytes) / (1 << 20); }
constexpr double to_gib(size_t bytes) { return static_cast<double>(bytes) / (1 << 30); }

template <typename T, typename IdxT>
void check_graph_degree(size_t& intermediate_degree, size_t& graph_degree, size_t dataset_size)
{
  if (intermediate_degree >= static_cast<size_t>(dataset_size)) {
    RAFT_LOG_WARN(
      "Intermediate graph degree cannot be larger than dataset size, reducing it to %lu",
      dataset_size);
    intermediate_degree = dataset_size - 1;
  }
  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }
}

// ACE: Get partition labels for partitioned approach
// TODO(julianmi): Use all neighbors APIs.
template <typename T, typename IdxT>
void ace_get_partition_labels(
  raft::resources const& res,
  raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> partition_labels,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> partition_histogram,
  size_t min_partition_size,
  double sampling_rate = 0.01)
{
  size_t dataset_size = dataset.extent(0);
  size_t dataset_dim  = dataset.extent(1);
  size_t labels_size  = partition_labels.extent(0);
  size_t labels_dim   = partition_labels.extent(1);
  RAFT_EXPECTS(dataset_size == labels_size, "Dataset size must match partition labels extent");
  size_t n_partitions = partition_histogram.extent(0);
  RAFT_EXPECTS(labels_dim == 2, "Labels must have 2 columns");
  RAFT_EXPECTS(partition_histogram.extent(1) == 2, "Partition histogram must have 2 columns");
  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  // Sampling vectors from dataset. Uses float conversion on host instead of
  // raft::matrix::sample_rows to minimize GPU memory usage.
  // TODO(julianmi): Switch to sample_rows when https://github.com/rapidsai/cuvs/issues/1461 is
  // addressed.
  size_t n_samples         = dataset_size * sampling_rate;
  const size_t min_samples = 100 * n_partitions;
  n_samples                = std::max(n_samples, min_samples);
  n_samples                = std::min(n_samples, dataset_size);
  RAFT_LOG_DEBUG("ACE: n_samples: %lu", n_samples);

  auto sample_db = raft::make_host_matrix<float, int64_t>(n_samples, dataset_dim);
#pragma omp parallel for
  for (size_t i = 0; i < n_samples; i++) {
    size_t j = i * dataset_size / n_samples;
    for (size_t k = 0; k < dataset_dim; k++) {
      sample_db(i, k) = static_cast<float>(dataset(j, k));
    }
  }
  auto sample_db_dev = raft::make_device_matrix<float, int64_t>(res, n_samples, dataset_dim);
  raft::copy(res, sample_db_dev.view(), sample_db.view());

  cuvs::cluster::kmeans::balanced_params kmeans_params;
  auto centroids_dev = raft::make_device_matrix<float, int64_t>(res, n_partitions, dataset_dim);
  cuvs::cluster::kmeans::fit(res, kmeans_params, sample_db_dev.view(), centroids_dev.view());

  // Compute distances between dataset and centroid vectors
  // Uses float conversion on host instead of batch_load_iterator to minimize GPU memory usage.
  const size_t chunk_size = 32 * 1024;
  auto _sub_dataset       = raft::make_host_matrix<float, int64_t>(chunk_size, dataset_dim);
  auto _sub_distances     = raft::make_host_matrix<float, int64_t>(chunk_size, n_partitions);
  auto _sub_dataset_dev   = raft::make_device_matrix<float, int64_t>(res, chunk_size, dataset_dim);
  auto _sub_distances_dev = raft::make_device_matrix<float, int64_t>(res, chunk_size, n_partitions);
  size_t report_interval  = dataset_size / 10;
  report_interval         = (report_interval / chunk_size) * chunk_size;
  report_interval         = std::max(report_interval, chunk_size);

  for (size_t i_base = 0; i_base < dataset_size; i_base += chunk_size) {
    const size_t sub_dataset_size = std::min(chunk_size, dataset_size - i_base);
    if (i_base % report_interval == 0) {
      RAFT_LOG_INFO("ACE: Processing chunk %lu / %lu (%.1f%%)",
                    i_base,
                    dataset_size,
                    static_cast<double>(100 * i_base) / dataset_size);
    }

    auto sub_dataset = raft::make_host_matrix_view<float, int64_t>(
      _sub_dataset.data_handle(), sub_dataset_size, dataset_dim);
#pragma omp parallel for
    for (size_t i_sub = 0; i_sub < sub_dataset_size; i_sub++) {
      size_t i = i_base + i_sub;
      for (size_t k = 0; k < dataset_dim; k++) {
        sub_dataset(i_sub, k) = static_cast<float>(dataset(i, k));
      }
    }
    auto sub_dataset_dev_view = raft::make_device_matrix_view<float, int64_t>(
      _sub_dataset_dev.data_handle(), sub_dataset_size, dataset_dim);
    raft::copy(res, sub_dataset_dev_view, sub_dataset);
    auto sub_dataset_dev = raft::make_device_matrix_view<const float, int64_t>(
      _sub_dataset_dev.data_handle(), sub_dataset_size, dataset_dim);

    auto sub_distances = raft::make_host_matrix_view<float, int64_t>(
      _sub_distances.data_handle(), sub_dataset_size, n_partitions);
    auto sub_distances_dev = raft::make_device_matrix_view<float, int64_t>(
      _sub_distances_dev.data_handle(), sub_dataset_size, n_partitions);

    cuvs::distance::pairwise_distance(res,
                                      sub_dataset_dev,
                                      centroids_dev.view(),
                                      sub_distances_dev,
                                      cuvs::distance::DistanceType::L2Expanded);

    raft::copy(res, sub_distances, sub_distances_dev);
    raft::resource::sync_stream(res, stream);

    // Find two closest partitions to each dataset vector
#pragma omp parallel for
    for (size_t i_sub = 0; i_sub < sub_dataset_size; i_sub++) {
      size_t core_label      = 0;
      size_t augmented_label = 1;
      if (sub_distances(i_sub, 0) > sub_distances(i_sub, 1)) {
        core_label      = 1;
        augmented_label = 0;
      }
      for (size_t c = 2; c < n_partitions; c++) {
        if (sub_distances(i_sub, c) < sub_distances(i_sub, core_label)) {
          augmented_label = core_label;
          core_label      = c;
        } else if (sub_distances(i_sub, c) < sub_distances(i_sub, augmented_label)) {
          augmented_label = c;
        }
      }
      size_t i               = i_base + i_sub;
      partition_labels(i, 0) = core_label;
      partition_labels(i, 1) = augmented_label;

#pragma omp atomic update
      partition_histogram(core_label, 0) += 1;
#pragma omp atomic update
      partition_histogram(augmented_label, 1) += 1;
    }
  }
}

// ACE: Check partition sizes for stable KNN graph construction
template <typename IdxT>
void ace_check_partition_sizes(
  size_t dataset_size,
  size_t n_partitions,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> partition_labels,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> partition_histogram,
  size_t min_partition_size)
{
  // Collect partition histogram statistics
  size_t total_core_vectors      = 0;
  size_t total_augmented_vectors = 0;
  size_t min_core_vectors        = dataset_size;
  size_t max_core_vectors        = 0;
  size_t min_augmented_vectors   = dataset_size;
  size_t max_augmented_vectors   = 0;
  size_t min_total_vectors       = dataset_size;
  size_t max_total_vectors       = 0;

  for (size_t c = 0; c < n_partitions; c++) {
    size_t core_count      = partition_histogram(c, 0);
    size_t augmented_count = partition_histogram(c, 1);
    size_t total_count     = core_count + augmented_count;

    if (total_count > 0) {
      total_core_vectors += core_count;
      total_augmented_vectors += augmented_count;

      min_core_vectors      = std::min(min_core_vectors, core_count);
      max_core_vectors      = std::max(max_core_vectors, core_count);
      min_augmented_vectors = std::min(min_augmented_vectors, augmented_count);
      max_augmented_vectors = std::max(max_augmented_vectors, augmented_count);
      min_total_vectors     = std::min(min_total_vectors, total_count);
      max_total_vectors     = std::max(max_total_vectors, total_count);
    }
  }

  double avg_core_vectors      = static_cast<double>(total_core_vectors) / n_partitions;
  double avg_augmented_vectors = static_cast<double>(total_augmented_vectors) / n_partitions;
  double avg_total_vectors     = 2.0 * static_cast<double>(dataset_size) / n_partitions;
  double expected_avg_vectors  = 2.0 * static_cast<double>(dataset_size) / n_partitions;

  RAFT_LOG_INFO("ACE: Core vectors        - Total: %lu, Avg: %.1f, Min: %lu, Max: %lu",
                total_core_vectors,
                avg_core_vectors,
                min_core_vectors,
                max_core_vectors);
  RAFT_LOG_INFO("ACE: Augmented vectors   - Total: %lu, Avg: %.1f, Min: %lu, Max: %lu",
                total_augmented_vectors,
                avg_augmented_vectors,
                min_augmented_vectors,
                max_augmented_vectors);
  RAFT_LOG_INFO("ACE: Total per partition - Total: %lu, Avg: %.1f, Min: %lu, Max: %lu",
                total_core_vectors + total_augmented_vectors,
                avg_total_vectors,
                min_total_vectors,
                max_total_vectors);

  // Check for partition imbalance and issue warnings
  size_t very_small_threshold = min_partition_size;
  size_t very_large_threshold = static_cast<size_t>(5.0 * expected_avg_vectors);

  for (size_t c = 0; c < n_partitions; c++) {
    size_t total_count = partition_histogram(c, 0) + partition_histogram(c, 1);

    if (total_count > 0 && total_count < very_small_threshold) {
      RAFT_LOG_WARN(
        "ACE: Partition %lu is very small (%lu vectors, expected ~%.1f). This may affect graph "
        "quality.",
        c,
        total_count,
        expected_avg_vectors);
    } else if (total_count > very_large_threshold) {
      RAFT_LOG_WARN(
        "ACE: Partition %lu is very large (%lu vectors, expected ~%.1f, threshold: %lu). This may "
        "indicate imbalance and can lead to memory issues in restricted environments.",
        c,
        total_count,
        expected_avg_vectors,
        very_large_threshold);
    }
  }
}

// ACE: Create forward/backward mappings between original and reordered vector IDs
// The in-memory path can be parallelized but the disk path requires ordering.
template <typename IdxT>
void ace_create_forward_and_backward_lists(
  size_t dataset_size,
  size_t n_partitions,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> partition_labels,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> partition_histogram,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_forward_mapping,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_backward_mapping,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> augmented_backward_mapping,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_partition_offsets,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> augmented_partition_offsets)
{
  core_partition_offsets(0)      = 0;
  augmented_partition_offsets(0) = 0;
  for (size_t c = 1; c < n_partitions; c++) {
    core_partition_offsets(c) = core_partition_offsets(c - 1) + partition_histogram(c - 1, 0);
    augmented_partition_offsets(c) =
      augmented_partition_offsets(c - 1) + partition_histogram(c - 1, 1);
  }

  if (static_cast<size_t>(core_forward_mapping.extent(0)) == 0) {
    // Memory path: both backward mappings
    RAFT_EXPECTS(static_cast<size_t>(core_backward_mapping.extent(0)) == dataset_size,
                 "core_backward_mapping must be of size dataset_size");
    RAFT_EXPECTS(static_cast<size_t>(augmented_backward_mapping.extent(0)) == dataset_size,
                 "augmented_backward_mapping must be of size dataset_size");
#pragma omp parallel for
    for (size_t i = 0; i < dataset_size; i++) {
      size_t core_partition_id = partition_labels(i, 0);
      size_t core_id;
#pragma omp atomic capture
      core_id = core_partition_offsets(core_partition_id)++;
      RAFT_EXPECTS(core_id < dataset_size, "Vector ID must be smaller than dataset_size");
      core_backward_mapping(core_id) = i;

      size_t augmented_partition_id = partition_labels(i, 1);
      size_t augmented_id;
#pragma omp atomic capture
      augmented_id = augmented_partition_offsets(augmented_partition_id)++;
      RAFT_EXPECTS(augmented_id < dataset_size, "Vector ID must be smaller than dataset_size");
      augmented_backward_mapping(augmented_id) = i;
    }
  } else {
    // Disk path: all three mappings
    RAFT_EXPECTS(static_cast<size_t>(core_forward_mapping.extent(0)) == dataset_size,
                 "core_forward_mapping must be of size dataset_size");
    RAFT_EXPECTS(static_cast<size_t>(core_backward_mapping.extent(0)) == dataset_size,
                 "core_backward_mapping must be of size dataset_size");
    RAFT_EXPECTS(static_cast<size_t>(augmented_backward_mapping.extent(0)) == dataset_size,
                 "augmented_backward_mapping must be of size dataset_size");
    for (size_t i = 0; i < dataset_size; i++) {
      size_t core_partition_id = partition_labels(i, 0);
      size_t core_id;
      core_id = core_partition_offsets(core_partition_id)++;
      RAFT_EXPECTS(core_id < dataset_size, "Vector ID must be smaller than dataset_size");
      core_backward_mapping(core_id) = i;
      core_forward_mapping(i)        = core_id;

      size_t augmented_partition_id = partition_labels(i, 1);
      size_t augmented_id;
      augmented_id = augmented_partition_offsets(augmented_partition_id)++;
      RAFT_EXPECTS(augmented_id < dataset_size, "Vector ID must be smaller than dataset_size");
      augmented_backward_mapping(augmented_id) = i;
    }
  }

  // Restore idxptr arrays
  for (size_t c = n_partitions; c > 0; c--) {
    core_partition_offsets(c)      = core_partition_offsets(c - 1);
    augmented_partition_offsets(c) = augmented_partition_offsets(c - 1);
  }
  core_partition_offsets(0)      = 0;
  augmented_partition_offsets(0) = 0;
}

// ACE: Gather partition dataset
template <typename T, typename IdxT>
void ace_gather_partition_dataset(
  size_t core_sub_dataset_size,
  size_t augmented_sub_dataset_size,
  size_t dataset_dim,
  size_t partition_id,
  raft::host_matrix_view<const T, int64_t, row_major> dataset,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_backward_mapping,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> augmented_backward_mapping,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_partition_offsets,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> augmented_partition_offsets,
  raft::host_matrix_view<T, int64_t, raft::row_major> sub_dataset)
{
  const size_t vector_size_bytes = dataset_dim * sizeof(T);

  // Copy core partition vectors
#pragma omp parallel for
  for (size_t j = 0; j < core_sub_dataset_size; j++) {
    size_t i = core_backward_mapping(j + core_partition_offsets(partition_id));
    memcpy(&sub_dataset(j, 0), &dataset(i, 0), vector_size_bytes);
  }

  // Copy augmented partition vectors (2nd closest partition)
#pragma omp parallel for
  for (size_t j = 0; j < augmented_sub_dataset_size; j++) {
    size_t i = augmented_backward_mapping(j + augmented_partition_offsets(partition_id));
    memcpy(&sub_dataset(j + core_sub_dataset_size, 0), &dataset(i, 0), vector_size_bytes);
  }
}

// ACE: Adjust IDs from core and augmented partitions to global reordered IDs
template <typename IdxT>
void ace_adjust_sub_graph_ids(
  size_t core_sub_dataset_size,
  size_t augmented_sub_dataset_size,
  size_t graph_degree,
  size_t partition_id,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> sub_search_graph,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> search_graph,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_partition_offsets,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> augmented_partition_offsets,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_backward_mapping,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> augmented_backward_mapping)
{
#pragma omp parallel for
  for (size_t i = 0; i < core_sub_dataset_size; i++) {
    // Map row index from local → reordered → original
    size_t i_reordered = i + core_partition_offsets(partition_id);
    size_t i_original  = core_backward_mapping(i_reordered);

    for (size_t k = 0; k < graph_degree; k++) {
      size_t j = sub_search_graph(i, k);
      size_t j_original;

      if (j == kInvalidNeighbor<IdxT>) {
        // Variable-degree padding / invalid-neighbor sentinel: propagate as-is.
        search_graph(i_original, k) = kInvalidNeighbor<IdxT>;
        continue;
      }

      if (j < core_sub_dataset_size) {
        // core partition neighbor: local → core reordered → original
        size_t j_reordered = j + core_partition_offsets(partition_id);
        j_original         = core_backward_mapping(j_reordered);
      } else {
        // Augmented partition neighbor: local → augmented reordered → original
        size_t j_augmented = j - core_sub_dataset_size;
        j_original =
          augmented_backward_mapping(j_augmented + augmented_partition_offsets(partition_id));
      }
      search_graph(i_original, k) = j_original;
    }
  }
}

// ACE: Adjust ids in sub search graph in place for disk version
template <typename IdxT>
void ace_adjust_sub_graph_ids_disk(
  size_t core_sub_dataset_size,
  size_t augmented_sub_dataset_size,
  size_t graph_degree,
  size_t partition_id,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> sub_search_graph,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_partition_offsets,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> augmented_partition_offsets,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> augmented_backward_mapping,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_forward_mapping)
{
#pragma omp parallel for
  for (size_t i = 0; i < core_sub_dataset_size; i++) {
    for (size_t k = 0; k < graph_degree; k++) {
      size_t j = sub_search_graph(i, k);
      if (j == kInvalidNeighbor<IdxT>) {
        // Variable-degree padding / invalid-neighbor sentinel: propagate as-is.
        sub_search_graph(i, k) = kInvalidNeighbor<IdxT>;
        continue;
      }
      if (j < core_sub_dataset_size) {
        // core partition neighbor: local → core reordered
        sub_search_graph(i, k) = j + core_partition_offsets(partition_id);
      } else {
        // Augmented partition neighbor: local → augmented reordered→ original → core reordered
        size_t j_augmented = j - core_sub_dataset_size;
        size_t j_original =
          augmented_backward_mapping(j_augmented + augmented_partition_offsets(partition_id));
        sub_search_graph(i, k) = core_forward_mapping(j_original);
      }
    }
  }
}

// ACE: Reorder dataset based on partition assignments and store to disk
// Writes two files: reordered_dataset.npy (core partitions) and augmented_dataset.npy (secondary
// partitions). Uses buffered writes optimized for NVMe storage.
template <typename T, typename IdxT>
void ace_reorder_and_store_dataset(
  raft::resources const& res,
  const std::string& build_dir,
  raft::host_matrix_view<const T, int64_t, row_major> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> partition_labels,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> partition_histogram,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_backward_mapping,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_partition_offsets,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> augmented_partition_offsets,
  cuvs::util::file_descriptor& reordered_fd,
  cuvs::util::file_descriptor& augmented_fd,
  cuvs::util::file_descriptor& mapping_fd,
  size_t reordered_header_size,
  size_t augmented_header_size,
  size_t mapping_header_size)
{
  auto start = std::chrono::high_resolution_clock::now();

  size_t dataset_size = dataset.extent(0);
  size_t dataset_dim  = dataset.extent(1);
  size_t n_partitions = partition_histogram.extent(0);

  RAFT_LOG_DEBUG(
    "ACE: Reordering and storing dataset to disk (%lu vectors, %lu dimensions, %lu partitions)",
    dataset_size,
    dataset_dim,
    n_partitions);

  // Calculate total sizes for pre-allocation
  size_t total_core_vectors      = 0;
  size_t total_augmented_vectors = 0;
  size_t max_core_vectors        = 0;
  size_t max_augmented_vectors   = 0;
  for (size_t p = 0; p < n_partitions; p++) {
    total_core_vectors += partition_histogram(p, 0);
    total_augmented_vectors += partition_histogram(p, 1);
    max_core_vectors      = std::max<size_t>(max_core_vectors, partition_histogram(p, 0));
    max_augmented_vectors = std::max<size_t>(max_augmented_vectors, partition_histogram(p, 1));
  }
  RAFT_EXPECTS(total_core_vectors == dataset_size,
               "Total core vectors must be equal to dataset size");
  RAFT_EXPECTS(total_augmented_vectors == dataset_size,
               "Total augmented vectors must be equal to dataset size");

  // Pre-allocate file space for better performance
  const size_t vector_size   = dataset_dim * sizeof(T);
  size_t reordered_file_size = total_core_vectors * vector_size;
  size_t augmented_file_size = total_augmented_vectors * vector_size;

  RAFT_LOG_DEBUG("ACE: Reordered dataset: %lu core vectors (%.2f GiB)",
                 total_core_vectors,
                 reordered_file_size / (1024.0 * 1024.0 * 1024.0));
  RAFT_LOG_DEBUG("ACE: Augmented dataset: %lu secondary vectors (%.2f GiB)",
                 total_augmented_vectors,
                 augmented_file_size / (1024.0 * 1024.0 * 1024.0));

  // Calculate partition start offsets for reordered and augmented datasets
  auto core_partition_starts = raft::make_host_vector<size_t, int64_t>(n_partitions + 1);
  memset(core_partition_starts.data_handle(), 0, (n_partitions + 1) * sizeof(size_t));
  auto augmented_partition_starts = raft::make_host_vector<size_t, int64_t>(n_partitions + 1);
  memset(augmented_partition_starts.data_handle(), 0, (n_partitions + 1) * sizeof(size_t));
  auto core_partition_current = raft::make_host_vector<size_t, int64_t>(n_partitions);
  memset(core_partition_current.data_handle(), 0, n_partitions * sizeof(size_t));
  auto augmented_partition_current = raft::make_host_vector<size_t, int64_t>(n_partitions);
  memset(augmented_partition_current.data_handle(), 0, n_partitions * sizeof(size_t));

  for (size_t p = 0; p < n_partitions; p++) {
    core_partition_starts(p + 1)      = core_partition_starts(p) + partition_histogram(p, 0);
    augmented_partition_starts(p + 1) = augmented_partition_starts(p) + partition_histogram(p, 1);
  }

  const size_t free_memory = cuvs::util::get_free_host_memory();
  // Conservatively allocate 50% of free memory per partition. Accounts for internal buffers and
  // overhead.
  // TODO: Adjust overhead if needed.
  const size_t memory_per_partition = 0.5 * free_memory / (n_partitions * 2);
  size_t disk_write_size            = raft::bound_by_power_of_two<size_t>(memory_per_partition);
  // 64MB should be enough to saturate typical NVMe SSDs.
  disk_write_size           = std::min<size_t>(disk_write_size, 64 * 1024 * 1024);
  size_t vectors_per_buffer = std::max<size_t>(64, disk_write_size / vector_size);

  RAFT_LOG_DEBUG("ACE: Reorder buffers: %lu vectors per buffer (%.2f MiB)",
                 vectors_per_buffer,
                 to_mib(vectors_per_buffer * vector_size));

  std::vector<raft::host_matrix<T, int64_t>> core_buffers;
  std::vector<raft::host_matrix<T, int64_t>> augmented_buffers;
  auto core_buffer_counts      = raft::make_host_vector<size_t, int64_t>(n_partitions);
  auto augmented_buffer_counts = raft::make_host_vector<size_t, int64_t>(n_partitions);

  core_buffers.reserve(n_partitions);
  augmented_buffers.reserve(n_partitions);

  for (size_t p = 0; p < n_partitions; p++) {
    core_buffers.emplace_back(raft::make_host_matrix<T, int64_t>(vectors_per_buffer, dataset_dim));
    augmented_buffers.emplace_back(
      raft::make_host_matrix<T, int64_t>(vectors_per_buffer, dataset_dim));
    core_buffer_counts(p)      = 0;
    augmented_buffer_counts(p) = 0;
  }
  auto flush_core_buffer = [&](size_t partition_id) {
    const size_t count = core_buffer_counts(partition_id);
    if (count > 0) {
      const size_t bytes_to_write = count * vector_size;
      const size_t file_offset =
        (core_partition_starts(partition_id) + core_partition_current(partition_id)) * vector_size +
        reordered_header_size;

      cuvs::util::write_large_file(
        reordered_fd, core_buffers[partition_id].data_handle(), bytes_to_write, file_offset);

      core_partition_current(partition_id) += count;
      core_buffer_counts(partition_id) = 0;
    }
  };

  auto flush_augmented_buffer = [&](size_t partition_id) {
    const size_t count = augmented_buffer_counts(partition_id);
    if (count > 0) {
      const size_t bytes_to_write = count * vector_size;
      const size_t file_offset =
        (augmented_partition_starts(partition_id) + augmented_partition_current(partition_id)) *
          vector_size +
        augmented_header_size;

      cuvs::util::write_large_file(
        augmented_fd, augmented_buffers[partition_id].data_handle(), bytes_to_write, file_offset);

      augmented_partition_current(partition_id) += count;
      augmented_buffer_counts(partition_id) = 0;
    }
  };

  size_t vectors_processed  = 0;
  const size_t log_interval = std::max(dataset_size / 10, size_t(1));
  for (size_t i = 0; i < dataset_size; i++) {
    size_t core_partition      = partition_labels(i, 0);
    size_t secondary_partition = partition_labels(i, 1);

    // Add vector to core partition buffer
    size_t core_buffer_row = core_buffer_counts(core_partition);
    memcpy(
      &core_buffers[core_partition](core_buffer_row, 0), &dataset(i, 0), dataset_dim * sizeof(T));
    core_buffer_counts(core_partition)++;

    // Flush core buffer if full
    if (core_buffer_counts(core_partition) >= vectors_per_buffer) {
      flush_core_buffer(core_partition);
    }

    // Add vector to augmented partition buffer
    size_t augmented_buffer_row = augmented_buffer_counts(secondary_partition);
    memcpy(&augmented_buffers[secondary_partition](augmented_buffer_row, 0),
           &dataset(i, 0),
           dataset_dim * sizeof(T));
    augmented_buffer_counts(secondary_partition)++;

    // Flush augmented buffer if full
    if (augmented_buffer_counts(secondary_partition) >= vectors_per_buffer) {
      flush_augmented_buffer(secondary_partition);
    }

    vectors_processed++;
    if (vectors_processed % log_interval == 0) {
      RAFT_LOG_INFO("ACE: Processed %lu/%lu vectors (%.1f%%)",
                    vectors_processed,
                    dataset_size,
                    100.0 * vectors_processed / dataset_size);
    }
  }

  // Flush all remaining buffers
  RAFT_LOG_DEBUG("ACE: Flushing remaining buffers...");
#pragma omp parallel sections
  {
#pragma omp section
    {
      for (size_t p = 0; p < n_partitions; p++) {
        flush_core_buffer(p);
      }
    }
#pragma omp section
    {
      for (size_t p = 0; p < n_partitions; p++) {
        flush_augmented_buffer(p);
      }
    }
  }

  const size_t mapping_file_size = dataset_size * sizeof(IdxT);
  cuvs::util::write_large_file(
    mapping_fd, core_backward_mapping.data_handle(), mapping_file_size, mapping_header_size);

  auto end        = std::chrono::high_resolution_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  // Calculate total bytes written
  size_t total_bytes_written = reordered_file_size + augmented_file_size + mapping_file_size;
  double throughput_mb_s =
    elapsed_ms > 0 ? to_mib(total_bytes_written) / (elapsed_ms / 1000.0) : 0.0;

  RAFT_LOG_INFO(
    "ACE: Dataset (%.2f GiB reordered, %.2f GiB augmented, %.2f GiB mapping) reordering completed "
    "in %ld ms (%.1f MiB/s)",
    reordered_file_size / (1024.0 * 1024.0 * 1024.0),
    augmented_file_size / (1024.0 * 1024.0 * 1024.0),
    mapping_file_size / (1024.0 * 1024.0 * 1024.0),
    elapsed_ms,
    throughput_mb_s);
}

// ACE: Load partition dataset and augmented dataset from disk
template <typename T, typename IdxT>
void ace_load_partition_dataset_from_disk(
  raft::resources const& res,
  const std::string& build_dir,
  size_t partition_id,
  size_t dataset_dim,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> partition_histogram,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_partition_offsets,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> augmented_partition_offsets,
  raft::host_matrix_view<T, int64_t, raft::row_major> sub_dataset)
{
  size_t n_partitions = partition_histogram.extent(0);

  RAFT_LOG_DEBUG("ACE: Loading partition %lu dataset from disk", partition_id);

  size_t core_size            = partition_histogram(partition_id, 0);
  size_t augmented_size       = partition_histogram(partition_id, 1);
  size_t total_partition_size = core_size + augmented_size;

  RAFT_LOG_DEBUG("ACE: Partition %lu: %lu core + %lu augmented = %lu total vectors",
                 partition_id,
                 core_size,
                 augmented_size,
                 total_partition_size);

  RAFT_EXPECTS(static_cast<size_t>(sub_dataset.extent(0)) == total_partition_size,
               "sub_dataset rows (%lu) must match total partition size (%lu)",
               sub_dataset.extent(0),
               total_partition_size);
  RAFT_EXPECTS(static_cast<size_t>(sub_dataset.extent(1)) == dataset_dim,
               "sub_dataset columns (%lu) must match dataset dimensions (%lu)",
               sub_dataset.extent(1),
               dataset_dim);

  const size_t vector_size = dataset_dim * sizeof(T);

  const std::string reordered_dataset_path = build_dir + "/reordered_dataset.npy";
  const std::string augmented_dataset_path = build_dir + "/augmented_dataset.npy";

  if (!std::filesystem::exists(reordered_dataset_path)) {
    RAFT_FAIL("ACE: Required file does not exist: %s", reordered_dataset_path.c_str());
  }
  if (!std::filesystem::exists(augmented_dataset_path)) {
    RAFT_FAIL("ACE: Required file does not exist: %s", augmented_dataset_path.c_str());
  }

  size_t core_header_size      = 0;
  size_t augmented_header_size = 0;
  size_t core_file_offset      = 0;
  size_t augmented_file_offset = 0;
  {
    std::ifstream is(reordered_dataset_path, std::ios::in | std::ios::binary);
    if (!is) { RAFT_FAIL("Cannot open file %s", reordered_dataset_path.c_str()); }
    auto start_pos = is.tellg();
    raft::numpy_serializer::read_header(is);
    core_header_size = static_cast<size_t>(is.tellg() - start_pos);
  }
  {
    std::ifstream is(augmented_dataset_path, std::ios::in | std::ios::binary);
    if (!is) { RAFT_FAIL("Cannot open file %s", augmented_dataset_path.c_str()); }
    auto start_pos = is.tellg();
    raft::numpy_serializer::read_header(is);
    augmented_header_size = static_cast<size_t>(is.tellg() - start_pos);
  }

  for (size_t p = 0; p < partition_id; p++) {
    core_file_offset += partition_histogram(p, 0);
    augmented_file_offset += partition_histogram(p, 1);
  }

  core_file_offset *= vector_size;
  augmented_file_offset *= vector_size;

  core_file_offset += core_header_size;
  augmented_file_offset += augmented_header_size;

  RAFT_LOG_DEBUG("ACE: Core file offset: %lu bytes, Augmented file offset: %lu bytes",
                 core_file_offset,
                 augmented_file_offset);

  // Read core and augmented data in parallel
  std::exception_ptr core_exception      = nullptr;
  std::exception_ptr augmented_exception = nullptr;

#pragma omp parallel sections
  {
#pragma omp section
    {
      try {
        if (core_size > 0) {
          RAFT_LOG_DEBUG(
            "ACE: Reading %lu core vectors from offset %lu", core_size, core_file_offset);
          cuvs::util::file_descriptor reordered_fd(reordered_dataset_path, O_RDONLY);
          const size_t core_bytes = core_size * vector_size;
          cuvs::util::read_large_file(
            reordered_fd, sub_dataset.data_handle(), core_bytes, core_file_offset);
        }
      } catch (...) {
        core_exception = std::current_exception();
      }
    }
#pragma omp section
    {
      try {
        if (augmented_size > 0) {
          RAFT_LOG_DEBUG("ACE: Reading %lu augmented vectors from offset %lu",
                         augmented_size,
                         augmented_file_offset);
          cuvs::util::file_descriptor augmented_fd(augmented_dataset_path, O_RDONLY);
          const size_t augmented_bytes = augmented_size * vector_size;
          T* augmented_dest            = sub_dataset.data_handle() + (core_size * dataset_dim);
          cuvs::util::read_large_file(
            augmented_fd, augmented_dest, augmented_bytes, augmented_file_offset);
        }
      } catch (...) {
        augmented_exception = std::current_exception();
      }
    }
  }

  // Check for exceptions from parallel sections
  if (core_exception) { std::rethrow_exception(core_exception); }
  if (augmented_exception) { std::rethrow_exception(augmented_exception); }
}

// Memory requirements for ACE operation
struct ace_memory_requirements {
  size_t partition_labels_size;
  size_t id_mapping_size;
  size_t sub_dataset_size;
  size_t sub_graph_size;
  size_t cagra_graph_size;
  size_t total_size;
  size_t available_host_memory;
  size_t available_gpu_memory;
};

// Amount of host memory that can be used for the build
constexpr double usable_cpu_memory_fraction = 0.8;

// Factor to account for imbalances in the partitions (maximum allowed is 3x the average)
constexpr double imbalance_factor = 3.0;

// Current partitioning adds each vector into 2 partitions (core and augmented)
constexpr double vector_expansion_factor = 2.0;

// Check if disk mode should be used for ACE based on memory constraints
template <typename T, typename IdxT>
bool ace_check_use_disk_mode(raft::resources const& res,
                             bool use_disk,
                             std::string& build_dir,
                             size_t dataset_size,
                             size_t dataset_dim,
                             size_t n_partitions,
                             size_t intermediate_degree,
                             size_t graph_degree,
                             std::optional<double> max_host_memory_gb,
                             std::optional<double> max_gpu_memory_gb,
                             bool guarantee_connectivity,
                             ace_memory_requirements& mem)
{
  // Use overridden memory limits if provided (> 0), otherwise query actual system memory
  if (max_host_memory_gb.has_value() && max_host_memory_gb.value() > 0) {
    auto actual_available_host_memory = cuvs::util::get_free_host_memory();
    auto configured_host_memory = static_cast<size_t>(max_host_memory_gb.value() * (1ULL << 30));
    if (actual_available_host_memory < configured_host_memory) {
      RAFT_LOG_WARN(
        "ACE: Actual host memory (%zu GiB) is less than configured limit (%zu GiB). "
        "Using actual host memory.",
        to_gib(actual_available_host_memory),
        to_gib(configured_host_memory));
      mem.available_host_memory = actual_available_host_memory;
    } else {
      RAFT_LOG_INFO("ACE: Using overridden host memory limit: %.2f GiB",
                    max_host_memory_gb.value());
      mem.available_host_memory = configured_host_memory;
    }
  } else {
    mem.available_host_memory = cuvs::util::get_free_host_memory();
  }
  size_t sub_partition_size =
    static_cast<size_t>(imbalance_factor * vector_expansion_factor *
                        raft::div_rounding_up_safe(dataset_size, n_partitions));
  auto [opt_host_ws_total, opt_dev_ws_total, opt_host_ws_fixed, opt_dev_ws_fixed] =
    helpers::optimize_workspace_size(
      sub_partition_size, graph_degree, intermediate_degree, sizeof(IdxT), guarantee_connectivity);

  // Optimistic memory model: focus on largest arrays, assumes all partitions are of equal size
  // For memory path:
  //   - Partition labels (core + augmented): vector_expansion_factor * dataset_size * sizeof(IdxT)
  //   - Backward ID mapping arrays (core + augmented): vector_expansion_factor * dataset_size *
  //   sizeof(IdxT)
  //   - Avg. per-partition dataset: vector_expansion_factor * (dataset_size / n_partitions) *
  //   dataset_dim * sizeof(T)
  //   - Avg. per-partition graph during build: vector_expansion_factor * (dataset_size /
  //   n_partitions) * (intermediate + final) * sizeof(IdxT)
  //   - Final assembled graph: dataset_size * graph_degree * sizeof(IdxT)
  mem.partition_labels_size = vector_expansion_factor * dataset_size * sizeof(IdxT);
  mem.id_mapping_size       = vector_expansion_factor * dataset_size * sizeof(IdxT);
  mem.sub_dataset_size      = sub_partition_size * dataset_dim * sizeof(T);
  mem.sub_graph_size   = sub_partition_size * (intermediate_degree + graph_degree) * sizeof(IdxT);
  mem.cagra_graph_size = dataset_size * graph_degree * sizeof(IdxT);
  mem.total_size       = mem.partition_labels_size + mem.id_mapping_size + mem.sub_dataset_size +
                   mem.sub_graph_size + mem.cagra_graph_size + opt_host_ws_total;

  RAFT_LOG_INFO("ACE: Estimated host memory required: %.2f GiB, available: %.2f GiB",
                to_gib(mem.total_size),
                to_gib(mem.available_host_memory));

  bool host_memory_limited =
    static_cast<size_t>(usable_cpu_memory_fraction * mem.available_host_memory) < mem.total_size;

  // GPU is mostly limited by the index size (update_graph() in the end of this routine).
  // Check if GPU has enough memory for the final graph or use disk mode instead.
  // TODO: Extend model or use managed memory if running out of GPU memory.
  if (max_gpu_memory_gb.has_value() && max_gpu_memory_gb.value() > 0) {
    auto actual_available_gpu_memory = rmm::available_device_memory().second;
    auto configured_gpu_memory = static_cast<size_t>(max_gpu_memory_gb.value() * (1ULL << 30));
    if (actual_available_gpu_memory < configured_gpu_memory) {
      RAFT_LOG_WARN(
        "ACE: Actual GPU memory (%zu GiB) is less than configured limit (%zu GiB). "
        "Using actual GPU memory.",
        to_gib(actual_available_gpu_memory),
        to_gib(configured_gpu_memory));
      mem.available_gpu_memory = actual_available_gpu_memory;
    } else {
      RAFT_LOG_INFO("ACE: Using overridden GPU memory limit: %.2f GiB", max_gpu_memory_gb.value());
      mem.available_gpu_memory = configured_gpu_memory;
    }
  } else {
    mem.available_gpu_memory = rmm::available_device_memory().second;
  }

  // what we need is maximum of:
  // * IVF-PQ on partition  (sub_dataset_size, uncompressed upper bound)
  // * optimize workspace (opt_dev_ws_total)
  // + some extra workspace (IVF-PQ search, ...)
  size_t extra_gpu_workspace_size = raft::resource::get_workspace_total_bytes(res);
  size_t gpu_memory_required =
    std::max(mem.sub_dataset_size, opt_dev_ws_total) + extra_gpu_workspace_size;

  bool gpu_memory_limited = mem.available_gpu_memory < gpu_memory_required;

  RAFT_LOG_INFO("ACE: Estimated GPU memory required: %.2f GiB, available: %.2f GiB",
                to_gib(gpu_memory_required),
                to_gib(mem.available_gpu_memory));

  bool use_disk_mode = use_disk || host_memory_limited || gpu_memory_limited;
  if (use_disk_mode) {
    bool valid_build_dir = !build_dir.empty();
    valid_build_dir &= build_dir.length() <= 255;
    valid_build_dir &= build_dir.find('\0') == std::string::npos;
    valid_build_dir &= build_dir.find("//") == std::string::npos;
    if (!valid_build_dir) {
      RAFT_LOG_WARN("ACE: Invalid build_dir path, resetting to default: /tmp/ace_build");
      build_dir = "/tmp/ace_build";
    }
    if (mkdir(build_dir.c_str(), 0755) != 0 && errno != EEXIST) {
      RAFT_EXPECTS(false, "Failed to create ACE build directory: %s", build_dir.c_str());
    }
  }

  if (host_memory_limited && gpu_memory_limited) {
    RAFT_LOG_INFO(
      "ACE: Graph does not fit in host and GPU memory. Using disk-mode with temporary storage %s",
      build_dir.c_str());
  } else if (host_memory_limited) {
    RAFT_LOG_INFO(
      "ACE: Graph does not fit in host memory. Using disk-mode with temporary storage %s",
      build_dir.c_str());
  } else if (gpu_memory_limited) {
    RAFT_LOG_INFO(
      "ACE: Graph does not fit in GPU memory. Using disk-mode with temporary storage %s",
      build_dir.c_str());
  } else if (use_disk) {
    RAFT_LOG_INFO(
      "ACE: Graph fits in host and GPU memory but disk mode is forced. Using disk-mode with "
      "temporary storage %s",
      build_dir.c_str());
  } else {
    RAFT_LOG_INFO("ACE: Graph fits in host and GPU memory. Using in-memory mode.");
  }

  return use_disk_mode;
}

// Validate and adjust partitions for disk mode memory requirements
template <typename T, typename IdxT>
void ace_validate_disk_mode_partitions(raft::resources const& res,
                                       size_t& n_partitions,
                                       size_t dataset_size,
                                       size_t dataset_dim,
                                       size_t intermediate_degree,
                                       size_t graph_degree,
                                       bool guarantee_connectivity,
                                       ace_memory_requirements& mem)
{
  // In disk mode, we don't need the full dataset or final graph in memory.
  // Host memory model for disk mode:
  //   - Partition labels (core + augmented): vector_expansion_factor * dataset_size * sizeof(IdxT)
  //   - ID mapping arrays (core + augmented): vector_expansion_factor * dataset_size * sizeof(IdxT)
  //   - Avg. per-partition dataset during processing: vector_expansion_factor * (dataset_size /
  //   n_partitions) * dataset_dim * sizeof(T)
  //   - Avg. per-partition graph during build: vector_expansion_factor * (dataset_size /
  //   n_partitions) * (intermediate + final) * sizeof(IdxT)

  size_t original_n_partitions     = n_partitions;
  size_t host_suggested_partitions = n_partitions;
  size_t gpu_suggested_partitions  = n_partitions;
  bool host_memory_insufficient    = false;
  bool gpu_memory_insufficient     = false;

  // Compute optimize workspace requirements
  size_t sub_partition_size =
    static_cast<size_t>(imbalance_factor * vector_expansion_factor *
                        raft::div_rounding_up_safe(dataset_size, n_partitions));
  auto [host_workspace_size_total,
        gpu_workspace_size_total,
        host_workspace_size_fixed,
        gpu_workspace_size_fixed] =
    helpers::optimize_workspace_size(
      sub_partition_size, graph_degree, intermediate_degree, sizeof(IdxT), guarantee_connectivity);

  // Check host memory requirements
  size_t disk_mode_host_required = mem.partition_labels_size + mem.id_mapping_size +
                                   mem.sub_dataset_size + mem.sub_graph_size +
                                   host_workspace_size_total;

  if (static_cast<size_t>(usable_cpu_memory_fraction * mem.available_host_memory) <
      disk_mode_host_required) {
    host_memory_insufficient = true;
    RAFT_LOG_WARN(
      "ACE: Host memory insufficient for disk mode. Required: %.2f GiB, available: %.2f GiB. "
      "Per-partition breakdown: dataset %.2f GiB, graph %.2f GiB, workspace %.2f GiB",
      to_gib(disk_mode_host_required),
      to_gib(mem.available_host_memory),
      to_gib(mem.sub_dataset_size),
      to_gib(mem.sub_graph_size),
      to_gib(host_workspace_size_total));

    // Calculate suggested number of partitions for host memory
    size_t disk_mode_host_static =
      mem.partition_labels_size + mem.id_mapping_size + host_workspace_size_fixed;
    size_t disk_mode_host_dynamic = disk_mode_host_required - disk_mode_host_static;
    double available_for_scaling =
      usable_cpu_memory_fraction * mem.available_host_memory - disk_mode_host_static;

    RAFT_EXPECTS(available_for_scaling > 0,
                 "ACE: Host memory insufficient even for constant overhead (labels + id_mapping + "
                 "static workspace). "
                 "Required: %.2f GiB, available: %.2f GiB",
                 to_gib(disk_mode_host_static),
                 to_gib(usable_cpu_memory_fraction * mem.available_host_memory));
    host_suggested_partitions =
      static_cast<size_t>(std::ceil(disk_mode_host_dynamic * n_partitions / available_for_scaling));
    // Ensure we always increase partitions (current count is insufficient by definition)
    host_suggested_partitions = std::max(host_suggested_partitions, n_partitions + 1);
  }

  // Check GPU memory requirements in disk mode
  // GPU memory model for disk mode (per-partition processing):
  // * IVF-PQ on partition  (mem.sub_dataset_size) (compressed?)
  // * optimize workspace (gpu_workspace_size_total)
  // + some extra workspace (IVF-PQ search, ...)
  size_t extra_gpu_workspace_size = raft::resource::get_workspace_total_bytes(res);
  size_t disk_mode_gpu_required =
    std::max(mem.sub_dataset_size, gpu_workspace_size_total) + extra_gpu_workspace_size;

  if (mem.available_gpu_memory < disk_mode_gpu_required) {
    gpu_memory_insufficient = true;
    RAFT_LOG_WARN(
      "ACE: GPU memory insufficient for per-partition processing. Required: %.2f GiB, "
      "available: %.2f GiB. Per-partition breakdown: dataset %.2f GiB, workspace %.2f GiB",
      to_gib(disk_mode_gpu_required),
      to_gib(mem.available_gpu_memory),
      to_gib(mem.sub_dataset_size),
      to_gib(gpu_workspace_size_total + extra_gpu_workspace_size));

    size_t disk_mode_gpu_static  = gpu_workspace_size_fixed + extra_gpu_workspace_size;
    size_t disk_mode_gpu_dynamic = disk_mode_gpu_required - disk_mode_gpu_static;
    double available_for_scaling = mem.available_gpu_memory - disk_mode_gpu_static;

    RAFT_EXPECTS(available_for_scaling > 0,
                 "ACE: GPU memory insufficient even for constant overhead. Required: %.2f GiB, "
                 "available: %.2f GiB",
                 to_gib(disk_mode_gpu_static),
                 to_gib(mem.available_gpu_memory));

    gpu_suggested_partitions =
      static_cast<size_t>(std::ceil(disk_mode_gpu_dynamic * n_partitions / available_for_scaling));
    gpu_suggested_partitions = std::max(gpu_suggested_partitions, n_partitions + 1);
  }

  // Auto-adjust to the maximum of host and GPU requirements
  if (host_memory_insufficient || gpu_memory_insufficient) {
    size_t new_n_partitions = std::max(host_suggested_partitions, gpu_suggested_partitions);

    RAFT_LOG_WARN(
      "ACE: Automatically increasing number of partitions from %zu to %zu to satisfy memory "
      "constraints.%s%s",
      original_n_partitions,
      new_n_partitions,
      host_memory_insufficient
        ? " Host memory requires >= " + std::to_string(host_suggested_partitions) + " partitions."
        : "",
      gpu_memory_insufficient
        ? " GPU memory requires >= " + std::to_string(gpu_suggested_partitions) + " partitions."
        : "");

    n_partitions = new_n_partitions;

    size_t new_sub_partition_size =
      static_cast<size_t>(imbalance_factor * vector_expansion_factor *
                          raft::div_rounding_up_safe(dataset_size, n_partitions));
    auto [new_opt_host_ws, new_opt_dev_ws, new_opt_host_ws_fixed, new_opt_dev_ws_fixed] =
      helpers::optimize_workspace_size(new_sub_partition_size,
                                       graph_degree,
                                       intermediate_degree,
                                       sizeof(IdxT),
                                       guarantee_connectivity);

    mem.sub_dataset_size = new_sub_partition_size * dataset_dim * sizeof(T);
    mem.sub_graph_size =
      new_sub_partition_size * (intermediate_degree + graph_degree) * sizeof(IdxT);
    mem.total_size = mem.partition_labels_size + mem.id_mapping_size + mem.sub_dataset_size +
                     mem.sub_graph_size + mem.cagra_graph_size + new_opt_host_ws;

    RAFT_LOG_INFO(
      "ACE: Updated per-partition memory estimates: dataset %.2f GiB, graph %.2f GiB, "
      "host workspace %.2f GiB, GPU workspace %.2f GiB",
      to_gib(mem.sub_dataset_size),
      to_gib(mem.sub_graph_size),
      to_gib(new_opt_host_ws),
      to_gib(new_opt_dev_ws + extra_gpu_workspace_size));
  }
}

// Build CAGRA index using ACE (Augmented Core Extraction) partitioning
// ACE enables building indexes for datasets too large to fit in GPU memory by:
// 1. Partitioning the dataset using balanced k-means in core (non-overlapping) and augmented
// (second-closest) partitions
// 2. Building sub-indexes for each partition independently
// 3. Concatenating sub-graphs (of core partitions) into a final unified index
// Supports both in-memory and disk-based modes depending on available host memory.
// In disk mode, the graph is stored in build_dir and dataset is reordered on disk.
// The returned index is not usable for search. Use the created files for search instead.
template <typename T, typename IdxT>
index<T, IdxT> build_ace(raft::resources const& res,
                         const index_params& params,
                         raft::host_matrix_view<const T, int64_t, row_major> dataset)
{
  // Extract ACE parameters from graph_build_params
  RAFT_EXPECTS(
    std::holds_alternative<cagra::graph_build_params::ace_params>(params.graph_build_params),
    "ACE build requires graph_build_params to be set to ace_params");

  auto ace_params    = std::get<cagra::graph_build_params::ace_params>(params.graph_build_params);
  size_t npartitions = ace_params.npartitions;
  size_t ef_construction = ace_params.ef_construction;
  std::string build_dir  = ace_params.build_dir;
  bool use_disk          = ace_params.use_disk;

  common::nvtx::range<common::nvtx::domain::cuvs> function_scope(
    "cagra::build_ace<host>(%zu, %zu, %zu)",
    params.intermediate_graph_degree,
    params.graph_degree,
    npartitions);

  size_t dataset_size = dataset.extent(0);
  size_t dataset_dim  = dataset.extent(1);

  RAFT_EXPECTS(dataset_size > 0, "ACE: Dataset must not be empty");
  if (dataset_size < 1000) {
    RAFT_LOG_WARN("ACE: Very small dataset size (%zu), consider using regular CAGRA build instead.",
                  dataset_size);
  }
  RAFT_EXPECTS(dataset_dim > 0, "ACE: Dataset dimension must be greater than 0");
  RAFT_EXPECTS(params.intermediate_graph_degree > 0,
               "ACE: Intermediate graph degree must be greater than 0");
  RAFT_EXPECTS(params.graph_degree > 0, "ACE: Graph degree must be greater than 0");

  size_t n_partitions = npartitions;
  if (n_partitions == 0) {
    // Default: start with 2 partitions and increase if needed (minimum for ACE to make sense).
    n_partitions = 2;
  }

  size_t min_required_per_partition = 1000;
  if (n_partitions > dataset_size / min_required_per_partition) {
    n_partitions = dataset_size / min_required_per_partition;
    if (n_partitions < 2) {
      RAFT_LOG_WARN(
        "ACE: Reduced number of partitions to the minimum of 2 to avoid tiny partitions. Consider "
        "using regular CAGRA build instead.");
      n_partitions = 2;
    } else {
      RAFT_LOG_WARN("ACE: Reduced number of partitions to %zu to avoid tiny partitions",
                    n_partitions);
    }
  }

  auto total_start = std::chrono::high_resolution_clock::now();
  RAFT_LOG_INFO("ACE: Starting partitioned CAGRA build with %zu partitions", n_partitions);

  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;

  // Track whether to clean up build directory on failure
  bool cleanup_on_failure = false;

  try {
    check_graph_degree<T, IdxT>(intermediate_degree, graph_degree, dataset_size);

    // Check if disk mode should be used based on memory constraints
    ace_memory_requirements mem;
    bool use_disk_mode = ace_check_use_disk_mode<T, IdxT>(res,
                                                          use_disk,
                                                          build_dir,
                                                          dataset_size,
                                                          dataset_dim,
                                                          n_partitions,
                                                          intermediate_degree,
                                                          graph_degree,
                                                          ace_params.max_host_memory_gb,
                                                          ace_params.max_gpu_memory_gb,
                                                          params.guarantee_connectivity,
                                                          mem);

    // Validate and adjust partitions if disk mode is enabled
    if (use_disk_mode) {
      ace_validate_disk_mode_partitions<T, IdxT>(res,
                                                 n_partitions,
                                                 dataset_size,
                                                 dataset_dim,
                                                 intermediate_degree,
                                                 graph_degree,
                                                 params.guarantee_connectivity,
                                                 mem);
    }

    // Preallocate space for files for better performance and fail early if not enough space.
    cuvs::util::file_descriptor reordered_fd;
    cuvs::util::file_descriptor augmented_fd;
    cuvs::util::file_descriptor mapping_fd;
    cuvs::util::file_descriptor graph_fd;
    size_t reordered_header_size = 0;
    size_t augmented_header_size = 0;
    size_t mapping_header_size   = 0;
    size_t graph_header_size     = 0;

    if (use_disk_mode) {
      if (mkdir(build_dir.c_str(), 0755) != 0 && errno != EEXIST) {
        RAFT_EXPECTS(false, "Failed to create ACE build directory: %s", build_dir.c_str());
      }
      // Mark for cleanup if we fail after creating the directory
      cleanup_on_failure = true;

      // Create numpy files with pre-allocated space
      std::tie(reordered_fd, reordered_header_size) = cuvs::util::create_numpy_file<T>(
        build_dir + "/reordered_dataset.npy", {dataset_size, dataset_dim});

      std::tie(augmented_fd, augmented_header_size) = cuvs::util::create_numpy_file<T>(
        build_dir + "/augmented_dataset.npy", {dataset_size, dataset_dim});

      std::tie(mapping_fd, mapping_header_size) =
        cuvs::util::create_numpy_file<IdxT>(build_dir + "/dataset_mapping.npy", {dataset_size});

      std::tie(graph_fd, graph_header_size) = cuvs::util::create_numpy_file<IdxT>(
        build_dir + "/cagra_graph.npy", {dataset_size, graph_degree});

      RAFT_LOG_DEBUG(
        "ACE: Wrote numpy headers (reordered: %zu, augmented: %zu, mapping: %zu, graph: %zu bytes)",
        reordered_header_size,
        augmented_header_size,
        mapping_header_size,
        graph_header_size);
    }

    auto partition_start     = std::chrono::high_resolution_clock::now();
    auto partition_labels    = raft::make_host_matrix<IdxT, int64_t>(dataset_size, 2);
    auto partition_histogram = raft::make_host_matrix<IdxT, int64_t>(n_partitions, 2);
    for (size_t c = 0; c < n_partitions; c++) {
      partition_histogram(c, 0) = 0;
      partition_histogram(c, 1) = 0;
    }

    // Determine minimum partition size for stable KNN graph construction
    size_t min_partition_size = std::max<size_t>(1000ULL, dataset_size / n_partitions * 0.1);

    ace_get_partition_labels<T, IdxT>(
      res, dataset, partition_labels.view(), partition_histogram.view(), min_partition_size);

    ace_check_partition_sizes<IdxT>(dataset_size,
                                    n_partitions,
                                    partition_labels.view(),
                                    partition_histogram.view(),
                                    min_partition_size);

    auto partition_end = std::chrono::high_resolution_clock::now();
    auto partition_elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(partition_end - partition_start)
        .count();
    RAFT_LOG_INFO(
      "ACE: Partition labeling completed in %ld ms (min_partition_size: "
      "%lu)",
      partition_elapsed,
      min_partition_size);

    // Create vector lists for each partition
    auto vectorlist_start      = std::chrono::high_resolution_clock::now();
    auto core_forward_mapping  = use_disk_mode ? raft::make_host_vector<IdxT, int64_t>(dataset_size)
                                               : raft::make_host_vector<IdxT, int64_t>(0);
    auto core_backward_mapping = raft::make_host_vector<IdxT, int64_t>(dataset_size);
    auto augmented_backward_mapping  = raft::make_host_vector<IdxT, int64_t>(dataset_size);
    auto core_partition_offsets      = raft::make_host_vector<IdxT, int64_t>(n_partitions + 1);
    auto augmented_partition_offsets = raft::make_host_vector<IdxT, int64_t>(n_partitions + 1);

    ace_create_forward_and_backward_lists<IdxT>(dataset_size,
                                                n_partitions,
                                                partition_labels.view(),
                                                partition_histogram.view(),
                                                core_forward_mapping.view(),
                                                core_backward_mapping.view(),
                                                augmented_backward_mapping.view(),
                                                core_partition_offsets.view(),
                                                augmented_partition_offsets.view());

    auto vectorlist_end = std::chrono::high_resolution_clock::now();
    auto vectorlist_elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(vectorlist_end - vectorlist_start)
        .count();
    RAFT_LOG_INFO("ACE: Vector list creation completed in %ld ms", vectorlist_elapsed);

    // Reorder the dataset based on partitions and store to disk. Uses write buffers to improve
    // performance.
    if (use_disk_mode) {
      ace_reorder_and_store_dataset<T, IdxT>(res,
                                             build_dir,
                                             dataset,
                                             partition_labels.view(),
                                             partition_histogram.view(),
                                             core_backward_mapping.view(),
                                             core_partition_offsets.view(),
                                             augmented_partition_offsets.view(),
                                             reordered_fd,
                                             augmented_fd,
                                             mapping_fd,
                                             reordered_header_size,
                                             augmented_header_size,
                                             mapping_header_size);
      // core_backward_mapping is not needed anymore.
      core_backward_mapping = raft::make_host_vector<IdxT, int64_t>(0);
    }

    // Placeholder search graph for in-memory version
    auto search_graph = use_disk_mode
                          ? raft::make_host_matrix<IdxT, int64_t>(0, 0)
                          : raft::make_host_matrix<IdxT, int64_t>(dataset_size, graph_degree);

    // Process each partition
    auto partition_processing_start = std::chrono::high_resolution_clock::now();
    for (size_t partition_id = 0; partition_id < n_partitions; partition_id++) {
      RAFT_LOG_DEBUG("ACE: Processing partition %lu/%lu", partition_id + 1, n_partitions);
      auto start = std::chrono::high_resolution_clock::now();

      // Extract vectors for this partition
      size_t core_sub_dataset_size      = partition_histogram(partition_id, 0);
      size_t augmented_sub_dataset_size = partition_histogram(partition_id, 1);
      size_t sub_dataset_size           = core_sub_dataset_size + augmented_sub_dataset_size;

      if (sub_dataset_size == 0) {
        RAFT_LOG_WARN("ACE: Skipping empty partition %lu", partition_id);
        continue;
      }
      RAFT_LOG_DEBUG("ACE: Sub-dataset size: %lu (%lu + %lu)",
                     sub_dataset_size,
                     core_sub_dataset_size,
                     augmented_sub_dataset_size);

      auto sub_dataset = raft::make_host_matrix<T, int64_t>(sub_dataset_size, dataset_dim);

      if (use_disk_mode) {
        // Load partition dataset from disk files
        ace_load_partition_dataset_from_disk<T, IdxT>(res,
                                                      build_dir,
                                                      partition_id,
                                                      dataset_dim,
                                                      partition_histogram.view(),
                                                      core_partition_offsets.view(),
                                                      augmented_partition_offsets.view(),
                                                      sub_dataset.view());
      } else {
        // Gather partition dataset from memory
        ace_gather_partition_dataset<T, IdxT>(core_sub_dataset_size,
                                              augmented_sub_dataset_size,
                                              dataset_dim,
                                              partition_id,
                                              dataset,
                                              core_backward_mapping.view(),
                                              augmented_backward_mapping.view(),
                                              core_partition_offsets.view(),
                                              augmented_partition_offsets.view(),
                                              sub_dataset.view());
      }
      auto read_end = std::chrono::high_resolution_clock::now();
      auto read_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(read_end - start).count();

      // Create index for this partition
      cuvs::neighbors::cagra::index_params sub_index_params;
      /* TODO(achirkin): split out the core of the heuristics from HNSW params

         The parameter selection below has two problems:
           1. General:
               HNSW-specific heuristics depend on parameters that are not native for CAGRA,
               which makes the code confusing; they are also subject to changes and
               they are design to fit better the Recall-QPS of HNSW not the generally
               best parameters.
           2. Specific:
               from_hnsw_params assigns params.graph_degree = M * 2 = (graph_degree / 2) * 2,
               so it rounds down the degree, which is a bug.
               The default variable degree fraction only makes sense for HNSW, not for CAGRA.

          What we need here instead is just a function that produces optimal graph build
          parameters and selects the build method based on the dataset size.
       */
      sub_index_params = cuvs::neighbors::cagra::index_params::from_hnsw_params(
        raft::make_extents<int64_t>(sub_dataset_size, dataset_dim),
        graph_degree / 2,
        ef_construction,
        cuvs::neighbors::cagra::hnsw_heuristic_type::SAME_GRAPH_FOOTPRINT,
        params.metric);
      // avoid rounding down graph_degree via heuristics doing `(graph_degree / 2) * 2`
      sub_index_params.graph_degree                   = graph_degree;
      sub_index_params.variable_graph_degree_fraction = 1.0;
      sub_index_params.attach_dataset_on_build        = false;
      sub_index_params.guarantee_connectivity         = params.guarantee_connectivity;

      auto sub_index = cuvs::neighbors::cagra::build(
        res, sub_index_params, raft::make_const_mdspan(sub_dataset.view()));

      auto optimize_end = std::chrono::high_resolution_clock::now();
      auto optimize_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(optimize_end - read_end).count();

      // Copy graph edges for core members of this partition
      auto sub_search_graph =
        raft::make_host_matrix<IdxT, int64_t>(core_sub_dataset_size, graph_degree);
      cudaStream_t stream = raft::resource::get_cuda_stream(res);
      raft::copy(
        res,
        raft::make_host_vector_view(sub_search_graph.data_handle(), sub_search_graph.size()),
        raft::make_device_vector_view(sub_index.graph().data_handle(), sub_search_graph.size()));
      raft::resource::sync_stream(res, stream);

      if (use_disk_mode) {
        // Adjust IDs in sub_search_graph in place for disk storage
        ace_adjust_sub_graph_ids_disk<IdxT>(core_sub_dataset_size,
                                            augmented_sub_dataset_size,
                                            graph_degree,
                                            partition_id,
                                            sub_search_graph.view(),
                                            core_partition_offsets.view(),
                                            augmented_partition_offsets.view(),
                                            augmented_backward_mapping.view(),
                                            core_forward_mapping.view());
      } else {
        // Adjust IDs in sub_search_graph and save to search_graph
        ace_adjust_sub_graph_ids<IdxT>(core_sub_dataset_size,
                                       augmented_sub_dataset_size,
                                       graph_degree,
                                       partition_id,
                                       sub_search_graph.view(),
                                       search_graph.view(),
                                       core_partition_offsets.view(),
                                       augmented_partition_offsets.view(),
                                       core_backward_mapping.view(),
                                       augmented_backward_mapping.view());
      }

      auto adjust_end = std::chrono::high_resolution_clock::now();
      auto adjust_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(adjust_end - optimize_end).count();

      if (use_disk_mode) {
        const size_t graph_offset =
          static_cast<size_t>(core_partition_offsets(partition_id)) * graph_degree * sizeof(IdxT) +
          graph_header_size;
        const size_t graph_bytes = core_sub_dataset_size * graph_degree * sizeof(IdxT);
        cuvs::util::write_large_file(
          graph_fd, sub_search_graph.data_handle(), graph_bytes, graph_offset);
      }

      auto end = std::chrono::high_resolution_clock::now();
      auto write_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - adjust_end).count();
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      double read_throughput =
        read_elapsed > 0
          ? to_mib(sub_dataset_size * dataset_dim * sizeof(T)) / (read_elapsed / 1000.0)
          : 0.0;
      double write_throughput =
        write_elapsed > 0
          ? to_mib(core_sub_dataset_size * dataset_dim * sizeof(T)) / (write_elapsed / 1000.0)
          : 0.0;
      RAFT_LOG_INFO(
        "ACE: Partition %4lu (%8lu + %8lu) completed in %6ld ms: read %6ld ms (%7.1f MiB/s), "
        "optimize %6ld ms, adjust %6ld ms, write %6ld ms (%7.1f MiB/s)",
        partition_id,
        core_sub_dataset_size,
        augmented_sub_dataset_size,
        elapsed_ms,
        read_elapsed,
        read_throughput,
        optimize_elapsed,
        adjust_elapsed,
        write_elapsed,
        write_throughput);
    }

    auto partition_processing_end     = std::chrono::high_resolution_clock::now();
    auto partition_processing_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          partition_processing_end - partition_processing_start)
                                          .count();
    RAFT_LOG_INFO("ACE: All partition processing completed in %ld ms (%zu partitions)",
                  partition_processing_elapsed,
                  n_partitions);

    // Clean up augmented dataset file to save disk space (no longer needed after partitions
    // processed)
    if (use_disk_mode) {
      const std::string augmented_dataset_path = build_dir + "/augmented_dataset.npy";
      if (std::filesystem::exists(augmented_dataset_path)) {
        std::filesystem::remove(augmented_dataset_path);
        RAFT_LOG_INFO("ACE: Removed augmented dataset file to save disk space");
      }
    }

    auto index_creation_start = std::chrono::high_resolution_clock::now();
    index<T, IdxT> idx(res, params.metric);
    // Only add graph and dataset if not using disk storage. The returned index is empty if using
    // disk storage. Use the files written to disk for search.
    if (!use_disk_mode) {
      idx.update_graph(res, raft::make_const_mdspan(search_graph.view()));

      if (params.attach_dataset_on_build) {
        try {
          idx.update_dataset(res, dataset);
        } catch (std::bad_alloc& e) {
          RAFT_LOG_WARN(
            "Insufficient GPU memory to attach dataset to ACE index. Only the graph will be "
            "stored.");
        } catch (raft::logic_error& e) {
          RAFT_LOG_WARN(
            "Insufficient GPU memory to attach dataset to ACE index. Only the graph will be "
            "stored.");
        }
      }
    } else {
      idx.update_dataset(res, std::move(reordered_fd));
      idx.update_graph(res, std::move(graph_fd));
      idx.update_mapping(res, std::move(mapping_fd));

      RAFT_LOG_INFO(
        "ACE: Set disk storage at %s (dataset shape [%zu, %zu], graph shape [%zu, %zu])",
        build_dir.c_str(),
        idx.size(),
        idx.dim(),
        idx.size(),
        idx.graph_degree());
    }

    auto index_creation_end     = std::chrono::high_resolution_clock::now();
    auto index_creation_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    index_creation_end - index_creation_start)
                                    .count();
    RAFT_LOG_INFO("ACE: Final index creation completed in %ld ms", index_creation_elapsed);

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    RAFT_LOG_INFO("ACE: Partitioned CAGRA build completed in %ld ms total", total_elapsed);

    return idx;
  } catch (const std::exception& e) {
    // Clean up build directory on failure if we created it
    RAFT_LOG_ERROR("ACE: Build failed with exception: %s", e.what());
    if (cleanup_on_failure && !build_dir.empty()) {
      RAFT_LOG_INFO("ACE: Cleaning up build directory: %s", build_dir.c_str());
      try {
        std::filesystem::remove_all(build_dir);
        RAFT_LOG_INFO("ACE: Successfully removed build directory");
      } catch (const std::exception& cleanup_error) {
        RAFT_LOG_WARN("ACE: Failed to clean up build directory: %s", cleanup_error.what());
      }
    }
    // Re-throw the original exception
    throw;
  }
}

template <typename IdxT>
void write_to_graph(raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
                    raft::host_matrix_view<int64_t, int64_t, raft::row_major> neighbors_host_view,
                    size_t& num_self_included,
                    size_t batch_size,
                    size_t batch_offset)
{
  uint32_t node_degree = knn_graph.extent(1);
  size_t top_k         = neighbors_host_view.extent(1);
  // omit itself & write out
  for (std::size_t i = 0; i < batch_size; i++) {
    size_t vec_idx = i + batch_offset;
    for (std::size_t j = 0, num_added = 0; j < top_k && num_added < node_degree; j++) {
      const auto v = neighbors_host_view(i, j);
      if (static_cast<size_t>(v) == vec_idx) {
        num_self_included++;
        continue;
      }
      knn_graph(vec_idx, num_added) = v;
      num_added++;
    }
  }
}

template <typename DataT, typename IdxT, typename accessor>
void refine_host_and_write_graph(
  raft::resources const& res,
  raft::host_matrix<DataT, int64_t>& queries_host,
  raft::host_matrix<int64_t, int64_t>& neighbors_host,
  raft::host_matrix<int64_t, int64_t>& refined_neighbors_host,
  raft::host_matrix<float, int64_t>& refined_distances_host,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cuvs::distance::DistanceType metric,
  size_t& num_self_included,
  size_t batch_size,
  size_t batch_offset,
  int top_k,
  int gpu_top_k)
{
  bool do_refine = top_k != gpu_top_k;

  auto refined_neighbors_host_view = raft::make_host_matrix_view<int64_t, int64_t>(
    do_refine ? refined_neighbors_host.data_handle() : neighbors_host.data_handle(),
    batch_size,
    top_k);

  if (do_refine) {
    // needed for compilation as this routine will also be run for device data with !do_refine
    if constexpr (raft::is_host_mdspan_v<decltype(dataset)>) {
      auto queries_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
        queries_host.data_handle(), batch_size, dataset.extent(1));
      auto neighbors_host_view = raft::make_host_matrix_view<const int64_t, int64_t>(
        neighbors_host.data_handle(), batch_size, neighbors_host.extent(1));
      auto refined_distances_host_view = raft::make_host_matrix_view<float, int64_t>(
        refined_distances_host.data_handle(), batch_size, top_k);
      cuvs::neighbors::refine(res,
                              dataset,
                              queries_host_view,
                              neighbors_host_view,
                              refined_neighbors_host_view,
                              refined_distances_host_view,
                              metric);
    }
  }

  write_to_graph(
    knn_graph, refined_neighbors_host_view, num_self_included, batch_size, batch_offset);
}

template <typename DataT, typename IdxT, typename accessor>
void build_knn_graph(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cuvs::neighbors::cagra::graph_build_params::ivf_pq_params pq)
{
  RAFT_EXPECTS(pq.build_params.metric == cuvs::distance::DistanceType::L2Expanded ||
                 pq.build_params.metric == cuvs::distance::DistanceType::InnerProduct ||
                 pq.build_params.metric == cuvs::distance::DistanceType::CosineExpanded,
               "Currently only L2Expanded, InnerProduct and CosineExpanded metrics are supported");

  uint32_t node_degree = knn_graph.extent(1);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "cagra::build_knn_graph<IVF-PQ>(%zu, %zu, %u)",
    size_t(dataset.extent(0)),
    size_t(dataset.extent(1)),
    node_degree);

  // Make model name
  const std::string model_name = [&]() {
    char model_name[1024];
    sprintf(model_name,
            "%s-%lux%lu.cluster_%u.pq_%u.%ubit.itr_%u.metric_%d.pqcenter_%u",
            "IVF-PQ",
            static_cast<size_t>(dataset.extent(0)),
            static_cast<size_t>(dataset.extent(1)),
            pq.build_params.n_lists,
            pq.build_params.pq_dim,
            pq.build_params.pq_bits,
            pq.build_params.kmeans_n_iters,
            static_cast<int>(pq.build_params.metric),
            static_cast<uint32_t>(pq.build_params.codebook_kind));
    return std::string(model_name);
  }();

  RAFT_LOG_DEBUG("# Building IVF-PQ index %s", model_name.c_str());
  auto index = cuvs::neighbors::ivf_pq::build(res, pq.build_params, dataset);

  // Empirically detect FP16 distance overflow on the just-built index: run a small FP16 probe
  // search and downgrade the internal/coarse dtypes to FP32 if any distance comes back non-finite.
  // This observes the actual computation, so it is agnostic of the selected distance type.
  if (pq.search_params.internal_distance_dtype == CUDA_R_16F ||
      pq.search_params.coarse_search_dtype == CUDA_R_16F) {
    {
      const bool fp16_overflow = cuvs::neighbors::ivf_pq::helpers::detect_fp16_overflow(
        res, index, pq.search_params, dataset, node_degree + 1);
      if (fp16_overflow) {
        RAFT_LOG_WARN(
          "IVF-PQ FP16 distance produced non-finite results on a probe search for this dataset -> "
          "switching 'internal_distance_dtype' and 'coarse_search_dtype' to FP32");
        pq.search_params.internal_distance_dtype = CUDA_R_32F;
        pq.search_params.coarse_search_dtype     = CUDA_R_32F;
      }
    }
  }

  //
  // search top (k + 1) neighbors
  //

  const auto top_k       = node_degree + 1;
  uint32_t gpu_top_k     = node_degree * pq.refinement_rate;
  gpu_top_k              = std::min<IdxT>(std::max(gpu_top_k, top_k), dataset.extent(0));
  const auto num_queries = dataset.extent(0);

  // Use the same maximum batch size as the ivf_pq::search to avoid allocating more than needed.
  uint32_t max_queries = pq.search_params.max_internal_batch_size;

  // Heuristic: the build_knn_graph code should use only a fraction of the workspace memory; the
  // rest should be used by the ivf_pq::search. Here we say that the workspace size should be a good
  // multiple of what is required for the I/O batching below.
  constexpr size_t kMinWorkspaceRatio = 5;
  constexpr size_t kMinLargeBatchSize = 512;
  auto desired_workspace_size =
    max_queries * (sizeof(DataT) * dataset.extent(1)  // queries (dataset batch)
                   + sizeof(float) * gpu_top_k        // distances
                   + sizeof(int64_t) * gpu_top_k      // neighbors
                   + sizeof(float) * top_k            // refined_distances
                   + sizeof(int64_t) * top_k          // refined_neighbors
                  );
  auto free_space_ratio    = raft::resource::get_workspace_free_bytes(res) / desired_workspace_size;
  bool use_large_workspace = false;
  if (free_space_ratio < kMinWorkspaceRatio) {
    auto adjusted_max_queries =
      static_cast<uint32_t>(max_queries * free_space_ratio / kMinWorkspaceRatio);
    if (adjusted_max_queries >= kMinLargeBatchSize) {
      // adjust max_queries, so that the ratio free_space_ratio gets not larger than
      // kMinWorkspaceRatio.
      RAFT_LOG_INFO(
        "CAGRA graph build: reducing IVF-PQ search max_internal_batch_size from %u -> %u to fit "
        "the workspace",
        max_queries,
        adjusted_max_queries);
      max_queries                              = adjusted_max_queries;
      pq.search_params.max_internal_batch_size = adjusted_max_queries;
    } else {
      // adjusting max_queries to a very small value isn't practical, so we use the large workspace
      // instead.
      use_large_workspace = true;
      RAFT_LOG_WARN(
        "Using large workspace memory for IVF-PQ search during CAGRA graph build. Desired "
        "workspace size: %zu, free workspace size: %zu",
        desired_workspace_size * kMinWorkspaceRatio,
        raft::resource::get_workspace_free_bytes(res));
    }
  }

  // If the workspace is smaller than desired, put the I/O buffers into the large workspace.
  rmm::device_async_resource_ref workspace_mr =
    use_large_workspace ? raft::resource::get_large_workspace_resource_ref(res)
                        : raft::resource::get_workspace_resource_ref(res);

  RAFT_LOG_DEBUG(
    "IVF-PQ search node_degree: %d, top_k: %d,  gpu_top_k: %d,  max_batch_size:: %d, n_probes: %u",
    node_degree,
    top_k,
    gpu_top_k,
    max_queries,
    pq.search_params.n_probes);

  auto distances = raft::make_device_mdarray<float>(
    res, workspace_mr, raft::make_extents<int64_t>(max_queries, gpu_top_k));
  auto neighbors = raft::make_device_mdarray<int64_t>(
    res, workspace_mr, raft::make_extents<int64_t>(max_queries, gpu_top_k));
  auto refined_distances = raft::make_device_mdarray<float>(
    res, workspace_mr, raft::make_extents<int64_t>(max_queries, top_k));
  auto refined_neighbors = raft::make_device_mdarray<int64_t>(
    res, workspace_mr, raft::make_extents<int64_t>(max_queries, top_k));
  auto neighbors_host = raft::make_host_matrix<int64_t, int64_t>(max_queries, gpu_top_k);
  auto queries_host   = raft::make_host_matrix<DataT, int64_t>(max_queries, dataset.extent(1));
  auto refined_neighbors_host = raft::make_host_matrix<int64_t, int64_t>(max_queries, top_k);
  auto refined_distances_host = raft::make_host_matrix<float, int64_t>(max_queries, top_k);

  // TODO(tfeher): batched search with multiple GPUs
  std::size_t num_self_included = 0;
  bool first                    = true;
  const auto start_clock        = std::chrono::system_clock::now();
  auto last_tick                = start_clock;

  auto vec_batches = cuvs::spatial::knn::detail::utils::make_batch_load_iterator<DataT>(
    res,
    dataset.data_handle(),
    static_cast<int64_t>(dataset.extent(0)),
    static_cast<int64_t>(dataset.extent(1)),
    static_cast<size_t>(max_queries),
    raft::resource::get_cuda_stream(res),
    workspace_mr);

  size_t next_report_offset = 0;
  size_t d_report_offset    = dataset.extent(0) / 100;  // Report progress in 1% steps.

  bool async_host_processing   = raft::is_host_mdspan_v<decltype(dataset)> || top_k == gpu_top_k;
  size_t previous_batch_size   = 0;
  size_t previous_batch_offset = 0;

  for (const auto& batch : vec_batches) {
    auto queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
      batch.data(), batch.size(), batch.row_width());
    auto neighbors_view = raft::make_device_matrix_view<int64_t, int64_t>(
      neighbors.data_handle(), batch.size(), neighbors.extent(1));
    auto distances_view = raft::make_device_matrix_view<float, int64_t>(
      distances.data_handle(), batch.size(), distances.extent(1));

    cuvs::neighbors::ivf_pq::search(
      res, pq.search_params, index, queries_view, neighbors_view, distances_view);

    if (async_host_processing) {
      // process previous batch async on host
      // NOTE: the async path also covers disabled refinement (top_k == gpu_top_k)
      if (previous_batch_size > 0) {
        refine_host_and_write_graph(res,
                                    queries_host,
                                    neighbors_host,
                                    refined_neighbors_host,
                                    refined_distances_host,
                                    dataset,
                                    knn_graph,
                                    pq.build_params.metric,
                                    num_self_included,
                                    previous_batch_size,
                                    previous_batch_offset,
                                    top_k,
                                    gpu_top_k);
      }

      // copy next batch to host
      raft::copy(res,
                 raft::make_host_vector_view(neighbors_host.data_handle(), neighbors_view.size()),
                 raft::make_device_vector_view(neighbors.data_handle(), neighbors_view.size()));
      if (top_k != gpu_top_k) {
        // can be skipped for disabled refinement
        raft::copy(res,
                   raft::make_host_vector_view(queries_host.data_handle(), queries_view.size()),
                   raft::make_device_vector_view(batch.data(), queries_view.size()));
      }

      previous_batch_size   = batch.size();
      previous_batch_offset = batch.offset();

      // we need to ensure the copy operations are done prior using the host data
      raft::resource::sync_stream(res);

      // process last batch
      if (previous_batch_offset + previous_batch_size == (size_t)num_queries) {
        refine_host_and_write_graph(res,
                                    queries_host,
                                    neighbors_host,
                                    refined_neighbors_host,
                                    refined_distances_host,
                                    dataset,
                                    knn_graph,
                                    pq.build_params.metric,
                                    num_self_included,
                                    previous_batch_size,
                                    previous_batch_offset,
                                    top_k,
                                    gpu_top_k);
      }
    } else {
      auto neighbor_candidates_view = raft::make_device_matrix_view<const int64_t, int64_t>(
        neighbors.data_handle(), batch.size(), gpu_top_k);
      auto refined_neighbors_view = raft::make_device_matrix_view<int64_t, int64_t>(
        refined_neighbors.data_handle(), batch.size(), top_k);
      auto refined_distances_view = raft::make_device_matrix_view<float, int64_t>(
        refined_distances.data_handle(), batch.size(), top_k);

      auto dataset_view = raft::make_device_matrix_view<const DataT, int64_t>(
        dataset.data_handle(), dataset.extent(0), dataset.extent(1));
      cuvs::neighbors::refine(res,
                              dataset_view,
                              queries_view,
                              neighbor_candidates_view,
                              refined_neighbors_view,
                              refined_distances_view,
                              pq.build_params.metric);
      raft::copy(res,
                 raft::make_host_vector_view(refined_neighbors_host.data_handle(),
                                             refined_neighbors_view.size()),
                 raft::make_device_vector_view(refined_neighbors_view.data_handle(),
                                               refined_neighbors_view.size()));
      raft::resource::sync_stream(res);

      auto refined_neighbors_host_view = raft::make_host_matrix_view<int64_t, int64_t>(
        refined_neighbors_host.data_handle(), batch.size(), top_k);
      write_to_graph(
        knn_graph, refined_neighbors_host_view, num_self_included, batch.size(), batch.offset());
    }

    size_t num_queries_done = batch.offset() + batch.size();
    const auto end_clock    = std::chrono::system_clock::now();
    if (batch.offset() > next_report_offset &&
        std::chrono::duration_cast<std::chrono::seconds>(end_clock - last_tick) >
          std::chrono::seconds(10)) {
      next_report_offset += d_report_offset;
      const auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() *
        1e-6;
      const auto throughput = num_queries_done / time;
      last_tick             = end_clock;

      RAFT_LOG_DEBUG(
        "# Search %12lu / %12lu (%3.2f %%), %e queries/sec, %.2f minutes ETA, self included = "
        "%3.2f %%    \r",
        num_queries_done,
        dataset.extent(0),
        num_queries_done / static_cast<double>(dataset.extent(0)) * 100,
        throughput,
        (num_queries - num_queries_done) / throughput / 60,
        static_cast<double>(num_self_included) / num_queries_done * 100.);
    }
    first = false;
  }

  if (!first) RAFT_LOG_DEBUG("# Finished building kNN graph");
  if (static_cast<double>(num_self_included) / dataset.extent(0) * 100. < 5) {
    RAFT_LOG_WARN(
      "Self-included ratio is low: %2.2f %%. This can lead to poor recall. "
      "Consider using a different configuration for the IVF-PQ index, "
      "increasing the refinement rate, or using higher-precision data type for "
      "LUT/Internal Distance.",
      static_cast<double>(num_self_included) / dataset.extent(0) * 100.);
  }
}

template <typename DataT, typename IdxT, typename accessor>
void build_knn_graph(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cuvs::neighbors::nn_descent::index_params build_params)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "cagra::build_knn_graph<NN-DESCENT>(%zu, %zu, %u)",
    size_t(dataset.extent(0)),
    size_t(dataset.extent(1)),
    size_t(knn_graph.extent(1)));

  std::optional<raft::host_matrix_view<IdxT, int64_t, row_major>> graph_view = knn_graph;
  auto nn_descent_idx = cuvs::neighbors::nn_descent::build(res, build_params, dataset, graph_view);

  using internal_IdxT = typename std::make_unsigned<IdxT>::type;
  using g_accessor    = typename decltype(nn_descent_idx.graph())::accessor_type;
  using g_accessor_internal =
    raft::host_device_accessor<cuda::std::default_accessor<internal_IdxT>, g_accessor::mem_type>;

  auto knn_graph_internal =
    raft::mdspan<internal_IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor_internal>(
      reinterpret_cast<internal_IdxT*>(nn_descent_idx.graph().data_handle()),
      nn_descent_idx.graph().extent(0),
      nn_descent_idx.graph().extent(1));

  cuvs::neighbors::cagra::detail::graph::sort_knn_graph(
    res, build_params.metric, dataset, knn_graph_internal);
}

template <typename IdxT = uint32_t,
          typename g_accessor =
            raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::host>>
void optimize(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor> knn_graph,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> new_graph,
  const bool guarantee_connectivity           = false,
  const double variable_graph_degree_fraction = 1.0)
{
  using internal_IdxT = typename std::make_unsigned<IdxT>::type;

  auto new_graph_internal = raft::make_host_matrix_view<internal_IdxT, int64_t>(
    reinterpret_cast<internal_IdxT*>(new_graph.data_handle()),
    new_graph.extent(0),
    new_graph.extent(1));

  using g_accessor_internal =
    raft::host_device_accessor<cuda::std::default_accessor<internal_IdxT>, raft::memory_type::host>;
  auto knn_graph_internal =
    raft::mdspan<internal_IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor_internal>(
      reinterpret_cast<internal_IdxT*>(knn_graph.data_handle()),
      knn_graph.extent(0),
      knn_graph.extent(1));

  cagra::detail::graph::optimize(res,
                                 knn_graph_internal,
                                 new_graph_internal,
                                 guarantee_connectivity,
                                 true,
                                 variable_graph_degree_fraction);
}

// RAII wrapper for allocating memory with Transparent HugePage
struct mmap_owner {
  // Allocate a new memory (not backed by a file)
  mmap_owner(size_t size) : size_{size}
  {
    int flags = MAP_ANONYMOUS | MAP_PRIVATE;
    ptr_      = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (ptr_ == MAP_FAILED) {
      ptr_ = nullptr;
      throw std::runtime_error("cuvs::mmap_owner error");
    }
    if (madvise(ptr_, size, MADV_HUGEPAGE) != 0) {
      munmap(ptr_, size);
      ptr_ = nullptr;
      throw std::runtime_error("cuvs::mmap_owner error");
    }
  }

  ~mmap_owner() noexcept
  {
    if (ptr_ != nullptr) { munmap(ptr_, size_); }
  }

  // No copies for owning struct
  mmap_owner(const mmap_owner& res)                      = delete;
  auto operator=(const mmap_owner& other) -> mmap_owner& = delete;
  // Moving is fine
  mmap_owner(mmap_owner&& other)
    : ptr_{std::exchange(other.ptr_, nullptr)}, size_{std::exchange(other.size_, 0)}
  {
  }
  auto operator=(mmap_owner&& other) -> mmap_owner&
  {
    std::swap(this->ptr_, other.ptr_);
    std::swap(this->size_, other.size_);
    return *this;
  }

  [[nodiscard]] auto data() const -> void* { return ptr_; }
  [[nodiscard]] auto size() const -> size_t { return size_; }

 private:
  void* ptr_;
  size_t size_;
};

template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
auto iterative_build_graph(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;

  auto cagra_graph = raft::make_host_matrix<IdxT, int64_t>(0, 0);

  // Iteratively improve the accuracy of the graph by repeatedly running
  // CAGRA's search() and optimize(). As for the size of the graph, instead
  // of targeting all nodes from the beginning, the number of nodes is
  // initially small, and the number of nodes is doubled with each iteration.
  RAFT_LOG_INFO("Iteratively creating/improving graph index using CAGRA's search() and optimize()");

  // If dataset is a host matrix, change it to a device matrix. Also, if the
  // dimensionality of the dataset does not meet the alighnemt restriction,
  // add extra dimensions and change it to a strided matrix.
  std::unique_ptr<strided_dataset<T, int64_t>> dev_aligned_dataset;
  try {
    dev_aligned_dataset = make_aligned_dataset(res, dataset);
  } catch (raft::logic_error& e) {
    RAFT_LOG_ERROR("Iterative CAGRA graph build requires the dataset to fit GPU memory");
    throw e;
  }
  auto dev_aligned_dataset_view = dev_aligned_dataset.get()->view();

  // If the matrix stride and extent do no match, the extra dimensions are
  // also as extent since it cannot be used as query matrix.
  auto dev_dataset =
    raft::make_device_matrix_view<const T, int64_t>(dev_aligned_dataset_view.data_handle(),
                                                    dev_aligned_dataset_view.extent(0),
                                                    dev_aligned_dataset_view.stride(0));

  // Determine initial graph size.
  uint64_t final_graph_size   = (uint64_t)dataset.extent(0);
  uint64_t initial_graph_size = (final_graph_size + 1) / 2;
  while (initial_graph_size > graph_degree * 64) {
    initial_graph_size = (initial_graph_size + 1) / 2;
  }
  RAFT_LOG_DEBUG("# initial graph size = %lu", (uint64_t)initial_graph_size);

  // Allocate memory for search results.
  constexpr uint64_t max_chunk_size = 8192;
  // +1 because the search may return the query node itself as a neighbor;
  // this is consistent with the per-iteration curr_topk = next_graph_degree + 1
  auto topk          = intermediate_degree + 1;
  auto dev_neighbors = raft::make_device_matrix<IdxT, int64_t>(res, max_chunk_size, topk);
  auto dev_distances = raft::make_device_matrix<float, int64_t>(res, max_chunk_size, topk);

  // Determine graph degree and number of search results while increasing
  // graph size.
  auto small_graph_degree = std::max(graph_degree / 2, std::min(graph_degree, (uint64_t)24));
  RAFT_LOG_DEBUG("# small_graph_degree = %lu", (uint64_t)small_graph_degree);
  RAFT_LOG_DEBUG("# graph_degree = %lu", (uint64_t)graph_degree);
  RAFT_LOG_DEBUG("# topk = %lu", (uint64_t)topk);

  // Create an initial graph. The initial graph created here is not suitable for
  // searching, but connectivity is guaranteed.
  auto offset = raft::make_host_vector<IdxT, int64_t>(small_graph_degree);
  for (uint64_t j = 0; j < small_graph_degree; j++) {
    if (j == 0) {
      offset(j) = 1;
    } else {
      offset(j) = offset(j - 1) + 1;
    }
    IdxT ofst = pow((double)(initial_graph_size - 1) / 2, (double)(j + 1) / small_graph_degree);
    if (offset(j) < ofst) { offset(j) = ofst; }
    RAFT_LOG_DEBUG("# offset(%lu) = %lu", (uint64_t)j, (uint64_t)offset(j));
  }
  cagra_graph = raft::make_host_matrix<IdxT, int64_t>(initial_graph_size, small_graph_degree);
  for (uint64_t i = 0; i < initial_graph_size; i++) {
    for (uint64_t j = 0; j < small_graph_degree; j++) {
      cagra_graph(i, j) = (i + offset(j)) % initial_graph_size;
    }
  }

  // Allocate memory for neighbors list using Transparent HugePage
  constexpr size_t thp_size = 2 * 1024 * 1024;
  size_t byte_size          = sizeof(IdxT) * final_graph_size * topk;
  if (byte_size % thp_size) { byte_size += thp_size - (byte_size % thp_size); }
  mmap_owner neighbors_list(byte_size);
  IdxT* neighbors_ptr = (IdxT*)neighbors_list.data();
  memset(neighbors_ptr, 0, byte_size);

  bool flag_last       = false;
  auto curr_graph_size = initial_graph_size;
  while (true) {
    auto start           = std::chrono::high_resolution_clock::now();
    auto curr_query_size = std::min(2 * curr_graph_size, final_graph_size);

    auto next_graph_degree = small_graph_degree;
    if (curr_graph_size == final_graph_size) { next_graph_degree = graph_degree; }

    // The search count (topk) is set to the next graph degree + 1, because
    // pruning is not used except in the last iteration.
    // (*) The appropriate setting for itopk_size requires careful consideration.
    auto curr_topk       = next_graph_degree + 1;
    auto curr_itopk_size = next_graph_degree + 32;
    if (flag_last) {
      curr_topk       = topk;
      curr_itopk_size = curr_topk + 32;
    }

    RAFT_LOG_DEBUG(
      "# graph_size = %lu (%.3lf), graph_degree = %lu, query_size = %lu, itopk = %lu, topk = %lu",
      (uint64_t)cagra_graph.extent(0),
      (double)cagra_graph.extent(0) / final_graph_size,
      (uint64_t)cagra_graph.extent(1),
      (uint64_t)curr_query_size,
      (uint64_t)curr_itopk_size,
      (uint64_t)curr_topk);

    cuvs::neighbors::cagra::search_params search_params;
    search_params.algo        = cuvs::neighbors::cagra::search_algo::AUTO;
    search_params.max_queries = max_chunk_size;
    search_params.itopk_size  = curr_itopk_size;

    // Create an index (idx), a query view (dev_query_view), and a mdarray for
    // search results (neighbors).
    auto dev_dataset_view = raft::make_device_matrix_view<const T, int64_t>(
      dev_dataset.data_handle(), (int64_t)curr_graph_size, dev_dataset.extent(1));

    auto idx = index<T, IdxT>(
      res, params.metric, dev_dataset_view, raft::make_const_mdspan(cagra_graph.view()));

    auto dev_query_view = raft::make_device_matrix_view<const T, int64_t>(
      dev_dataset.data_handle(), (int64_t)curr_query_size, dev_dataset.extent(1));

    auto neighbors_view =
      raft::make_host_matrix_view<IdxT, int64_t>(neighbors_ptr, curr_query_size, curr_topk);

    // Search.
    // Since there are many queries, divide them into batches and search them.
    auto query_batch = cuvs::spatial::knn::detail::utils::make_batch_load_iterator<T>(
      res,
      dev_query_view.data_handle(),
      static_cast<int64_t>(curr_query_size),
      static_cast<int64_t>(dev_query_view.extent(1)),
      max_chunk_size,
      raft::resource::get_cuda_stream(res),
      raft::resource::get_workspace_resource_ref(res));
    for (const auto& batch : query_batch) {
      auto batch_dev_query_view = raft::make_device_matrix_view<const T, int64_t>(
        batch.data(), batch.size(), dev_query_view.extent(1));
      auto batch_dev_neighbors_view = raft::make_device_matrix_view<IdxT, int64_t>(
        dev_neighbors.data_handle(), batch.size(), curr_topk);
      auto batch_dev_distances_view = raft::make_device_matrix_view<float, int64_t>(
        dev_distances.data_handle(), batch.size(), curr_topk);

      cuvs::neighbors::cagra::search(res,
                                     search_params,
                                     idx,
                                     batch_dev_query_view,
                                     batch_dev_neighbors_view,
                                     batch_dev_distances_view);

      auto batch_neighbors_view = raft::make_host_matrix_view<IdxT, int64_t>(
        neighbors_view.data_handle() + batch.offset() * curr_topk, batch.size(), curr_topk);
      raft::copy(res, batch_neighbors_view, batch_dev_neighbors_view);
    }

    // Optimize graph
    auto next_graph_size = curr_query_size;
    cagra_graph          = raft::make_host_matrix<IdxT, int64_t>(0, 0);  // delete existing grahp
    cagra_graph = raft::make_host_matrix<IdxT, int64_t>(next_graph_size, next_graph_degree);
    optimize<IdxT>(res,
                   neighbors_view,
                   cagra_graph.view(),
                   flag_last ? params.guarantee_connectivity : false,
                   flag_last ? params.variable_graph_degree_fraction : 1.0);

    auto end        = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    RAFT_LOG_DEBUG("# elapsed time: %.3lf sec", (double)elapsed_ms / 1000);

    if (flag_last) { break; }
    flag_last       = (curr_graph_size == final_graph_size);
    curr_graph_size = next_graph_size;
  }

  return cagra_graph;
}

template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
index<T, IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;
  common::nvtx::range<common::nvtx::domain::cuvs> function_scope(
    "cagra::build<%s>(%zu, %zu)",
    Accessor::is_managed_type::value ? "managed"
    : Accessor::is_host_type::value  ? "host"
                                     : "device",
    intermediate_degree,
    graph_degree);
  check_graph_degree<T, IdxT>(intermediate_degree, graph_degree, dataset.extent(0));

  // Set default value in case knn_build_params is not defined.
  auto knn_build_params = params.graph_build_params;
  if (std::holds_alternative<std::monostate>(params.graph_build_params)) {
    // Heuristic to decide default build algo and its params.
    if (cuvs::neighbors::nn_descent::has_enough_device_memory(
          res, dataset.extents(), sizeof(IdxT))) {
      RAFT_LOG_DEBUG("NN descent solver");
      knn_build_params =
        cagra::graph_build_params::nn_descent_params(intermediate_degree, params.metric);
    } else {
      RAFT_LOG_DEBUG("Selecting IVF-PQ solver");
      knn_build_params = cagra::graph_build_params::ivf_pq_params(dataset.extents(), params.metric);
    }
  }

  RAFT_EXPECTS(
    params.metric != cuvs::distance::DistanceType::BitwiseHamming ||
      std::holds_alternative<cagra::graph_build_params::iterative_search_params>(
        knn_build_params) ||
      std::holds_alternative<cagra::graph_build_params::nn_descent_params>(knn_build_params),
    "IVF_PQ for CAGRA graph build does not support BitwiseHamming as a metric. Please "
    "use nn-descent or the iterative CAGRA search build.");
  RAFT_EXPECTS(
    params.metric != cuvs::distance::DistanceType::CosineExpanded ||
      std::holds_alternative<cagra::graph_build_params::ivf_pq_params>(knn_build_params) ||
      std::holds_alternative<cagra::graph_build_params::nn_descent_params>(knn_build_params),
    "CosineExpanded distance is not supported for iterative CAGRA graph build.");

  // Validate data type for BitwiseHamming metric
  RAFT_EXPECTS(params.metric != cuvs::distance::DistanceType::BitwiseHamming ||
                 (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>),
               "BitwiseHamming distance is only supported for int8_t and uint8_t data types. "
               "Current data type is not supported.");

  auto cagra_graph = raft::make_host_matrix<IdxT, int64_t>(0, 0);

  // Dispatch based on graph_build_params
  if (std::holds_alternative<cagra::graph_build_params::iterative_search_params>(
        knn_build_params)) {
    cagra_graph = iterative_build_graph<T, IdxT, Accessor>(res, params, dataset);
  } else {
    std::optional<raft::host_matrix<IdxT, int64_t>> knn_graph(
      raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), intermediate_degree));

    if (std::holds_alternative<cagra::graph_build_params::ivf_pq_params>(knn_build_params)) {
      auto ivf_pq_params =
        std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(knn_build_params);
      if (ivf_pq_params.build_params.metric != params.metric) {
        RAFT_LOG_WARN(
          "Metric (%lu) for IVF-PQ needs to match cagra metric (%lu), "
          "aligning IVF-PQ metric.",
          ivf_pq_params.build_params.metric,
          params.metric);
        ivf_pq_params.build_params.metric = params.metric;
      }
      build_knn_graph(res, dataset, knn_graph->view(), ivf_pq_params);
    } else {
      auto nn_descent_params =
        std::get<cagra::graph_build_params::nn_descent_params>(knn_build_params);

      if (nn_descent_params.metric != params.metric) {
        RAFT_LOG_WARN(
          "Metric (%lu) for nn-descent needs to match cagra metric (%lu), "
          "aligning nn-descent metric.",
          nn_descent_params.metric,
          params.metric);
        nn_descent_params.metric = params.metric;
      }
      if (nn_descent_params.graph_degree != intermediate_degree) {
        RAFT_LOG_WARN(
          "Graph degree (%lu) for nn-descent needs to match cagra intermediate graph degree (%lu), "
          "aligning "
          "nn-descent graph_degree.",
          nn_descent_params.graph_degree,
          intermediate_degree);
        nn_descent_params =
          cagra::graph_build_params::nn_descent_params(intermediate_degree, params.metric);
      }

      // Use nn-descent to build CAGRA knn graph
      nn_descent_params.return_distances = false;
      build_knn_graph<T, IdxT>(res, dataset, knn_graph->view(), nn_descent_params);
    }

    cagra_graph = raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), graph_degree);

    RAFT_LOG_TRACE("optimizing graph");
    optimize<IdxT>(res,
                   knn_graph->view(),
                   cagra_graph.view(),
                   params.guarantee_connectivity,
                   params.variable_graph_degree_fraction);

    // free intermediate graph before trying to create the index
    knn_graph.reset();
  }

  RAFT_LOG_TRACE("Graph optimized, creating index");

  // Construct an index from dataset and optimized knn graph.
  if (params.compression.has_value()) {
    RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded,
                 "VPQ compression is only supported with L2Expanded distance mertric");
    index<T, IdxT> idx(res, params.metric);
    idx.update_graph(res, raft::make_const_mdspan(cagra_graph.view()));
    idx.update_dataset(
      res,
      // TODO: hardcoding codebook math to `half`, we can do runtime dispatching later
      cuvs::preprocessing::quantize::pq::vpq_build(res, *params.compression, dataset));

    return idx;
  }
  if (params.attach_dataset_on_build) {
    try {
      return index<T, IdxT>(
        res, params.metric, dataset, raft::make_const_mdspan(cagra_graph.view()));
    } catch (std::bad_alloc& e) {
      RAFT_LOG_WARN(
        "Insufficient GPU memory to construct CAGRA index with dataset on GPU. Only the graph will "
        "be added to the index");
      // We just add the graph. User is expected to update dataset separately (e.g allocating in
      // managed memory).
    } catch (raft::logic_error& e) {
      // The memory error can also manifest as logic_error.
      RAFT_LOG_WARN(
        "Insufficient GPU memory to construct CAGRA index with dataset on GPU. Only the graph will "
        "be added to the index");
    }
  }
  index<T, IdxT> idx(res, params.metric);
  idx.update_graph(res, raft::make_const_mdspan(cagra_graph.view()));
  return idx;
}
}  // namespace cuvs::neighbors::cagra::detail
