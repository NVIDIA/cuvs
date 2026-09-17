/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../util/kvikio_io.hpp"
#include "cagra_build.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/util/host_memory.hpp>
#include <kvikio/file_handle.hpp>

#include <array>
#include <cerrno>
#include <exception>
#include <filesystem>
#include <sys/stat.h>
#include <unordered_set>

namespace cuvs::neighbors::cagra::detail {

// Helpers to convert bytes to MiB and GiB
constexpr double to_mib(size_t bytes) { return static_cast<double>(bytes) / (1 << 20); }
constexpr double to_gib(size_t bytes) { return static_cast<double>(bytes) / (1 << 30); }

class ace_disk_workspace {
 public:
  enum class artifact : size_t {
    reordered_dataset,
    augmented_dataset,
    dataset_mapping,
    cagra_graph,
  };

  explicit ace_disk_workspace(std::string build_dir)
    : build_dir_(std::move(build_dir)),
      artifacts_{build_dir_ / "reordered_dataset.npy",
                 build_dir_ / "augmented_dataset.npy",
                 build_dir_ / "dataset_mapping.npy",
                 build_dir_ / "cagra_graph.npy"}
  {
  }

  void initialize()
  {
    if (mkdir(build_dir_.c_str(), 0755) == 0) {
      directory_created_by_this_build_ = true;
      return;
    }

    if (errno != EEXIST) {
      RAFT_FAIL("Failed to create ACE build directory: %s (errno: %d, %s)",
                build_dir_.c_str(),
                errno,
                strerror(errno));
    }

    std::error_code error;
    const bool is_directory = std::filesystem::is_directory(build_dir_, error);
    RAFT_EXPECTS(!error,
                 "Failed to inspect ACE build directory: %s (%s)",
                 build_dir_.c_str(),
                 error.message().c_str());
    RAFT_EXPECTS(is_directory, "ACE build path is not a directory: %s", build_dir_.c_str());
  }

  [[nodiscard]] std::string artifact_path(artifact which) const
  {
    return artifacts_[static_cast<size_t>(which)].string();
  }

  void mark_artifact_created(artifact which) noexcept
  {
    artifacts_created_[static_cast<size_t>(which)] = true;
  }

  void commit() noexcept { committed_ = true; }

  void cleanup() noexcept
  {
    if (committed_) { return; }

    for (size_t i = artifacts_.size(); i > 0; --i) {
      if (!artifacts_created_[i - 1]) { continue; }

      std::error_code error;
      std::filesystem::remove(artifacts_[i - 1], error);
      if (error) {
        RAFT_LOG_WARN("ACE: Failed to remove build artifact %s: %s",
                      artifacts_[i - 1].c_str(),
                      error.message().c_str());
      }
    }

    if (directory_created_by_this_build_) {
      std::error_code error;
      std::filesystem::remove(build_dir_, error);
      if (error) {
        RAFT_LOG_WARN("ACE: Failed to remove empty build directory %s: %s",
                      build_dir_.c_str(),
                      error.message().c_str());
      }
    }
  }

 private:
  std::filesystem::path build_dir_;
  std::array<std::filesystem::path, 4> artifacts_;
  std::array<bool, 4> artifacts_created_{};
  bool directory_created_by_this_build_ = false;
  bool committed_                       = false;
};

// ACE: Get partition labels for partitioned approach
// TODO(julianmi): Use all neighbors APIs.
template <typename T, typename IdxT>
void ace_get_partition_labels(
  raft::resources const& res,
  raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
  size_t dataset_dim,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> partition_labels,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> partition_histogram,
  size_t min_partition_size,
  double sampling_rate = 0.01)
{
  size_t dataset_size = dataset.extent(0);
  size_t labels_size  = partition_labels.extent(0);
  size_t labels_dim   = partition_labels.extent(1);
  RAFT_EXPECTS(dataset_size == labels_size, "Dataset size must match partition labels extent");
  RAFT_EXPECTS(dataset_dim > 0, "Dataset dimension must be greater than 0");
  RAFT_EXPECTS(static_cast<size_t>(dataset.extent(1)) >= dataset_dim,
               "Dataset row extent (%zu) must be >= logical dimension (%zu)",
               static_cast<size_t>(dataset.extent(1)),
               dataset_dim);
  size_t n_partitions = partition_histogram.extent(0);
  RAFT_EXPECTS(labels_dim == 2, "Labels must have 2 columns");
  RAFT_EXPECTS(partition_histogram.extent(1) == 2, "Partition histogram must have 2 columns");
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();

  // Sampling vectors from dataset. Uses float conversion on host instead of
  // raft::matrix::sample_rows to minimize GPU memory usage.
  // TODO(julianmi): Switch to sample_rows when https://github.com/nvidia/cuvs/issues/1461 is
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

// ACE: Materialize the core partition followed by its argument partition.
template <typename T, typename IdxT>
void ace_make_core_argument_partition(
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

// ACE: Build the intermediate kNN graph for a core-plus-argument partition.
template <typename T, typename IdxT>
auto ace_build_partial_knn_graph(
  raft::resources const& res,
  raft::host_matrix_view<const T, int64_t, row_major> core_argument_partition,
  size_t intermediate_graph_degree,
  cuvs::distance::DistanceType metric,
  graph_build_params::ace_params::knn_graph_build_algo knn_build_algo)
  -> raft::host_matrix<IdxT, int64_t>
{
  auto const partition_size = static_cast<size_t>(core_argument_partition.extent(0));
  RAFT_EXPECTS(partition_size > intermediate_graph_degree,
               "ACE: partition size (%lu) must exceed the intermediate graph degree (%lu)",
               partition_size,
               intermediate_graph_degree);

  auto knn_graph = raft::make_host_matrix<IdxT, int64_t>(partition_size, intermediate_graph_degree);
  if (knn_build_algo == graph_build_params::ace_params::knn_graph_build_algo::IVF_PQ) {
    auto knn_params = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
      raft::make_extents<int64_t>(partition_size, core_argument_partition.extent(1)), metric);
    knn_params.search_params.n_probes =
      std::min(knn_params.search_params.n_probes * 2u, knn_params.build_params.n_lists);
    build_knn_graph(res, core_argument_partition, knn_graph.view(), knn_params);
  } else {
    auto knn_params                = graph_build_params::brute_force_params{};
    knn_params.build_params.metric = metric;
    build_knn_graph(res, core_argument_partition, knn_graph.view(), knn_params);
  }

  return knn_graph;
}

// ACE global reverse-edge mode: prune without merging local reverse edges.
template <typename T, typename IdxT>
auto ace_build_pruned_partial_graph(
  raft::resources const& res,
  raft::host_matrix_view<const T, int64_t, row_major> core_argument_partition,
  size_t intermediate_graph_degree,
  size_t graph_degree,
  cuvs::distance::DistanceType metric,
  graph_build_params::ace_params::knn_graph_build_algo knn_build_algo)
  -> raft::host_matrix<IdxT, int64_t>
{
  auto knn_graph = ace_build_partial_knn_graph<T, IdxT>(
    res, core_argument_partition, intermediate_graph_degree, metric, knn_build_algo);
  auto partial_graph =
    raft::make_host_matrix<IdxT, int64_t>(core_argument_partition.extent(0), graph_degree);
  graph::prune_graph_gpu<IdxT>(res, knn_graph.view(), partial_graph.view());
  return partial_graph;
}

// Original ACE mode: optimize each partial graph, including its local reverse-edge merge.
template <typename T, typename IdxT>
auto ace_build_optimized_partial_graph(
  raft::resources const& res,
  raft::host_matrix_view<const T, int64_t, row_major> core_argument_partition,
  size_t intermediate_graph_degree,
  size_t graph_degree,
  cuvs::distance::DistanceType metric,
  graph_build_params::ace_params::knn_graph_build_algo knn_build_algo,
  bool guarantee_connectivity) -> raft::host_matrix<IdxT, int64_t>
{
  auto knn_graph = ace_build_partial_knn_graph<T, IdxT>(
    res, core_argument_partition, intermediate_graph_degree, metric, knn_build_algo);
  auto partial_graph =
    raft::make_host_matrix<IdxT, int64_t>(core_argument_partition.extent(0), graph_degree);
  graph::optimize<IdxT>(res, knn_graph.view(), partial_graph.view(), guarantee_connectivity);
  return partial_graph;
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

// ACE: Adjust IDs into global reordered IDs on the device for the disk version.
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

// ACE: Reorder dataset based on partition assignments and store to disk
// Writes two files: reordered_dataset.npy (core partitions) and augmented_dataset.npy (secondary
// partitions). Uses buffered writes optimized for NVMe storage.
template <typename T, typename IdxT>
void ace_reorder_and_store_dataset(
  raft::resources const& res,
  const std::string& build_dir,
  raft::host_matrix_view<const T, int64_t, row_major> dataset,
  size_t dataset_dim,
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
  size_t n_partitions = partition_histogram.extent(0);
  RAFT_EXPECTS(dataset_dim > 0, "Dataset dimension must be greater than 0");
  RAFT_EXPECTS(static_cast<size_t>(dataset.extent(1)) >= dataset_dim,
               "Dataset row extent (%zu) must be >= logical dimension (%zu)",
               static_cast<size_t>(dataset.extent(1)),
               dataset_dim);

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
template <typename T, typename IdxT, typename Accessor>
void ace_load_partition_dataset_from_disk(
  size_t partition_id,
  size_t dataset_dim,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> partition_histogram,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> core_partition_offsets,
  raft::host_vector_view<IdxT, int64_t, raft::row_major> augmented_partition_offsets,
  const cuvs::util::file_descriptor& reordered_fd,
  const cuvs::util::file_descriptor& augmented_fd,
  size_t reordered_header_size,
  size_t augmented_header_size,
  raft::mdspan<T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> sub_dataset)
{
  RAFT_LOG_DEBUG("ACE: Loading partition %lu dataset from disk", partition_id);

  const std::string reordered_path = reordered_fd.get_path();
  const std::string augmented_path = augmented_fd.get_path();
  T* sub_dataset_ptr               = sub_dataset.data_handle();
  RAFT_EXPECTS(sub_dataset_ptr != nullptr, "ACE: sub-dataset destination must not be null");
  RAFT_EXPECTS(reordered_fd.is_valid() && !reordered_path.empty(),
               "ACE: reordered dataset file descriptor is not valid");
  RAFT_EXPECTS(augmented_fd.is_valid() && !augmented_path.empty(),
               "ACE: augmented dataset file descriptor is not valid");
  RAFT_EXPECTS(partition_id < static_cast<size_t>(partition_histogram.extent(0)),
               "ACE: partition id is out of range");
  RAFT_EXPECTS(partition_id < static_cast<size_t>(core_partition_offsets.extent(0)),
               "ACE: core partition offset is out of range");
  RAFT_EXPECTS(partition_id < static_cast<size_t>(augmented_partition_offsets.extent(0)),
               "ACE: augmented partition offset is out of range");

  size_t core_size      = partition_histogram(partition_id, 0);
  size_t augmented_size = partition_histogram(partition_id, 1);
  size_t total_size     = core_size + augmented_size;

  RAFT_EXPECTS(static_cast<size_t>(sub_dataset.extent(0)) == total_size,
               "ACE: sub-dataset rows (%zu) must match partition size (%zu)",
               static_cast<size_t>(sub_dataset.extent(0)),
               total_size);
  RAFT_EXPECTS(static_cast<size_t>(sub_dataset.extent(1)) == dataset_dim,
               "ACE: sub-dataset columns (%zu) must match dataset dimensions (%zu)",
               static_cast<size_t>(sub_dataset.extent(1)),
               dataset_dim);

  RAFT_LOG_DEBUG("ACE: Partition %lu: %lu core + %lu augmented = %lu total vectors",
                 partition_id,
                 core_size,
                 augmented_size,
                 core_size + augmented_size);

  const size_t vector_size = dataset_dim * sizeof(T);
  const size_t core_file_offset =
    reordered_header_size + static_cast<size_t>(core_partition_offsets(partition_id)) * vector_size;
  const size_t augmented_file_offset =
    augmented_header_size +
    static_cast<size_t>(augmented_partition_offsets(partition_id)) * vector_size;

  RAFT_LOG_DEBUG("ACE: Core file offset: %lu bytes, Augmented file offset: %lu bytes",
                 core_file_offset,
                 augmented_file_offset);

  const size_t core_bytes      = core_size * vector_size;
  const size_t augmented_bytes = augmented_size * vector_size;
  T* augmented_dest            = sub_dataset_ptr + (core_size * dataset_dim);

  auto expect_complete_read = [partition_id](const char* name, size_t expected, size_t actual) {
    RAFT_EXPECTS(actual == expected,
                 "ACE: Short %s read for partition %lu: expected %zu bytes, got %zu",
                 name,
                 partition_id,
                 expected,
                 actual);
  };

  if (core_bytes > 0 && augmented_bytes > 0) {
    RAFT_LOG_DEBUG("ACE: Reading %lu core vectors from offset %lu", core_size, core_file_offset);
    RAFT_LOG_DEBUG(
      "ACE: Reading %lu augmented vectors from offset %lu", augmented_size, augmented_file_offset);
    auto reordered_handle =
      cuvs::util::detail::open_kvikio_file_for_ace_io(reordered_path, "r", sub_dataset_ptr);
    auto augmented_handle =
      cuvs::util::detail::open_kvikio_file_for_ace_io(augmented_path, "r", augmented_dest);
    auto core_future = reordered_handle.pread(sub_dataset_ptr, core_bytes, core_file_offset);
    auto augmented_future =
      augmented_handle.pread(augmented_dest, augmented_bytes, augmented_file_offset);
    std::exception_ptr read_exception = nullptr;
    size_t core_read                  = 0;
    size_t augmented_read             = 0;
    try {
      core_read = core_future.get();
    } catch (...) {
      read_exception = std::current_exception();
    }
    try {
      augmented_read = augmented_future.get();
    } catch (...) {
      if (!read_exception) { read_exception = std::current_exception(); }
    }
    if (read_exception) { std::rethrow_exception(read_exception); }
    expect_complete_read("core", core_bytes, core_read);
    expect_complete_read("augmented", augmented_bytes, augmented_read);
  } else if (core_bytes > 0) {
    RAFT_LOG_DEBUG("ACE: Reading %lu core vectors from offset %lu", core_size, core_file_offset);
    auto reordered_handle =
      cuvs::util::detail::open_kvikio_file_for_ace_io(reordered_path, "r", sub_dataset_ptr);
    expect_complete_read(
      "core",
      core_bytes,
      reordered_handle.pread(sub_dataset_ptr, core_bytes, core_file_offset).get());
  } else if (augmented_bytes > 0) {
    RAFT_LOG_DEBUG(
      "ACE: Reading %lu augmented vectors from offset %lu", augmented_size, augmented_file_offset);
    auto augmented_handle =
      cuvs::util::detail::open_kvikio_file_for_ace_io(augmented_path, "r", augmented_dest);
    expect_complete_read(
      "augmented",
      augmented_bytes,
      augmented_handle.pread(augmented_dest, augmented_bytes, augmented_file_offset).get());
  }
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
                             const std::string& build_dir,
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
  const auto host_memory = cuvs::util::get_host_memory_info();
  RAFT_EXPECTS(host_memory.available > 0,
               "ACE: No host memory is available within the current system or cgroup limit");
  if (host_memory.cgroup_limit.has_value() && host_memory.cgroup_current.has_value() &&
      host_memory.cgroup_reclaimable_file.has_value() &&
      host_memory.cgroup_working_set.has_value()) {
    const size_t cgroup_headroom = *host_memory.cgroup_working_set < *host_memory.cgroup_limit
                                     ? *host_memory.cgroup_limit - *host_memory.cgroup_working_set
                                     : 0;
    RAFT_LOG_INFO(
      "ACE: Cgroup host memory limit: %.2f GiB, current: %.2f GiB, reclaimable file cache: %.2f "
      "GiB, working set: %.2f GiB, headroom: %.2f GiB; system available: %.2f GiB, effective "
      "available: %.2f GiB",
      to_gib(*host_memory.cgroup_limit),
      to_gib(*host_memory.cgroup_current),
      to_gib(*host_memory.cgroup_reclaimable_file),
      to_gib(*host_memory.cgroup_working_set),
      to_gib(cgroup_headroom),
      to_gib(host_memory.system_available),
      to_gib(host_memory.available));
  }

  // Use overridden memory limits if provided (> 0), otherwise query actual system memory
  if (max_host_memory_gb.has_value() && max_host_memory_gb.value() > 0) {
    auto actual_available_host_memory = host_memory.available;
    auto configured_host_memory = static_cast<size_t>(max_host_memory_gb.value() * (1ULL << 30));
    if (actual_available_host_memory < configured_host_memory) {
      RAFT_LOG_WARN(
        "ACE: Actual host memory (%.2f GiB) is less than configured limit (%.2f GiB). "
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
    mem.available_host_memory = host_memory.available;
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
  if (use_disk_mode) { RAFT_EXPECTS(!build_dir.empty(), "ACE build directory must not be empty"); }

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

// Resolve the ACE partition count while preserving 0 as the auto-selection sentinel.
inline size_t ace_resolve_partition_count(size_t n_partitions)
{
  if (n_partitions == 0) { return 2; }
  if (n_partitions == 1) {
    RAFT_LOG_WARN(
      "ACE: Requested 1 partition; adjusted to 2 before applying partitioning heuristics");
    return 2;
  }
  return n_partitions;
}

// Validate the structural ACE partition-count invariants required by the labeler.
inline void ace_validate_partition_count(size_t n_partitions,
                                         size_t dataset_size,
                                         bool adjusted_for_memory = false)
{
  RAFT_EXPECTS(n_partitions <= dataset_size,
               adjusted_for_memory
                 ? "ACE: configured memory limit is unsatisfiable because the requested partition "
                   "count cannot exceed dataset size"
                 : "ACE: number of partitions cannot exceed dataset size");
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

template <typename IdxT>
__global__ void ace_merge_partition_reverse_edges(IdxT* partition_graph,
                                                  IdxT const* reverse_graph,
                                                  uint32_t const* reverse_graph_count,
                                                  uint32_t partition_size,
                                                  uint32_t graph_degree)
{
  auto const num_protected_edges = graph_degree / 2;
  for (uint32_t local_node = blockIdx.x * blockDim.x + threadIdx.x; local_node < partition_size;
       local_node += blockDim.x * gridDim.x) {
    auto* row              = partition_graph + static_cast<size_t>(local_node) * graph_degree;
    auto num_reverse_edges = min(reverse_graph_count[local_node], graph_degree);
    while (num_reverse_edges > 0) {
      auto const reverse_node =
        reverse_graph[static_cast<size_t>(local_node) * graph_degree + --num_reverse_edges];
      uint32_t existing_rank = 0;
      for (; existing_rank < graph_degree; ++existing_rank) {
        if (row[existing_rank] == reverse_node) { break; }
      }
      if (existing_rank < num_protected_edges) { continue; }

      auto const shift_end = existing_rank == graph_degree ? graph_degree - 1 : existing_rank;
      for (uint32_t rank = shift_end; rank > num_protected_edges; --rank) {
        row[rank] = row[rank - 1];
      }
      row[num_protected_edges] = reverse_node;
    }
  }
}

template <typename IdxT>
void ace_merge_partition_reverse_edges(raft::resources const& res,
                                       raft::host_matrix_view<IdxT, int64_t> source_graph,
                                       raft::host_matrix_view<IdxT, int64_t> graph,
                                       std::vector<IdxT> const& partition_nodes,
                                       std::vector<IdxT> candidate_sources)
{
  if (partition_nodes.empty() || candidate_sources.empty()) { return; }
  RAFT_EXPECTS(
    source_graph.extent(0) == graph.extent(0) && source_graph.extent(1) == graph.extent(1),
    "ACE: source and output graphs have different sizes");
  RAFT_EXPECTS(partition_nodes.size() <= std::numeric_limits<uint32_t>::max(),
               "ACE: partition is too large for reverse-edge processing");

  auto const graph_degree   = static_cast<uint32_t>(graph.extent(1));
  auto const partition_size = static_cast<uint32_t>(partition_nodes.size());
  auto const num_candidates = candidate_sources.size();
  std::sort(candidate_sources.begin(), candidate_sources.end());

  std::vector<IdxT> host_partition_graph(static_cast<size_t>(partition_size) * graph_degree);
  std::vector<IdxT> host_candidate_destinations(num_candidates);
  for (uint32_t local_node = 0; local_node < partition_size; ++local_node) {
    std::copy_n(
      graph.data_handle() + static_cast<size_t>(partition_nodes[local_node]) * graph_degree,
      graph_degree,
      host_partition_graph.data() + static_cast<size_t>(local_node) * graph_degree);
  }

  std::vector<std::pair<IdxT, uint32_t>> sorted_partition_nodes;
  sorted_partition_nodes.reserve(partition_size);
  for (uint32_t local_node = 0; local_node < partition_size; ++local_node) {
    sorted_partition_nodes.emplace_back(partition_nodes[local_node], local_node);
  }
  std::sort(sorted_partition_nodes.begin(), sorted_partition_nodes.end());
  std::vector<IdxT> host_sorted_partition_nodes(partition_size);
  std::vector<uint32_t> host_sorted_partition_local_indices(partition_size);
  for (uint32_t i = 0; i < partition_size; ++i) {
    host_sorted_partition_nodes[i]         = sorted_partition_nodes[i].first;
    host_sorted_partition_local_indices[i] = sorted_partition_nodes[i].second;
  }

  auto large_workspace   = raft::resource::get_large_workspace_resource_ref(res);
  auto workspace         = raft::resource::get_workspace_resource_ref(res);
  auto d_partition_graph = raft::make_device_mdarray<IdxT, int64_t>(
    res, large_workspace, raft::make_extents<int64_t>(partition_size, graph_degree));
  auto d_reverse_graph = raft::make_device_mdarray<IdxT, int64_t>(
    res, large_workspace, raft::make_extents<int64_t>(partition_size, graph_degree));
  auto d_reverse_graph_count = raft::make_device_mdarray<uint32_t, int64_t>(
    res, workspace, raft::make_extents<int64_t>(partition_size));
  auto d_candidate_sources = raft::make_device_mdarray<IdxT, int64_t>(
    res, workspace, raft::make_extents<int64_t>(num_candidates));
  auto d_candidate_destinations = raft::make_device_mdarray<IdxT, int64_t>(
    res, workspace, raft::make_extents<int64_t>(num_candidates, 1));
  auto d_sorted_partition_nodes = raft::make_device_mdarray<IdxT, int64_t>(
    res, workspace, raft::make_extents<int64_t>(partition_size));
  auto d_sorted_partition_local_indices = raft::make_device_mdarray<uint32_t, int64_t>(
    res, workspace, raft::make_extents<int64_t>(partition_size));

  auto const stream = raft::resource::get_cuda_stream(res);
  raft::copy(d_partition_graph.data_handle(),
             host_partition_graph.data(),
             host_partition_graph.size(),
             stream);
  raft::copy(d_candidate_sources.data_handle(), candidate_sources.data(), num_candidates, stream);
  raft::copy(d_sorted_partition_nodes.data_handle(),
             host_sorted_partition_nodes.data(),
             partition_size,
             stream);
  raft::copy(d_sorted_partition_local_indices.data_handle(),
             host_sorted_partition_local_indices.data(),
             partition_size,
             stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(
    d_reverse_graph_count.data_handle(), 0, partition_size * sizeof(uint32_t), stream));

  constexpr uint32_t threads  = 256;
  auto const candidate_blocks = static_cast<uint32_t>(
    std::min<size_t>(1024, raft::div_rounding_up_safe(num_candidates, size_t{threads})));
  auto const partition_blocks = static_cast<uint32_t>(
    std::min<uint32_t>(1024, raft::div_rounding_up_safe(partition_size, threads)));
  for (uint32_t rank = 0; rank < graph_degree; ++rank) {
    for (size_t candidate_index = 0; candidate_index < num_candidates; ++candidate_index) {
      host_candidate_destinations[candidate_index] =
        source_graph(static_cast<size_t>(candidate_sources[candidate_index]), rank);
    }
    raft::copy(d_candidate_destinations.data_handle(),
               host_candidate_destinations.data(),
               num_candidates,
               stream);
    graph::kern_make_rev_graph_k<IdxT>
      <<<candidate_blocks, threads, 0, stream>>>(d_candidate_destinations.view(),
                                                 d_reverse_graph.view(),
                                                 d_reverse_graph_count.view(),
                                                 0,
                                                 d_candidate_sources.data_handle(),
                                                 d_sorted_partition_nodes.data_handle(),
                                                 d_sorted_partition_local_indices.data_handle());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  ace_merge_partition_reverse_edges<<<partition_blocks, threads, 0, stream>>>(
    d_partition_graph.data_handle(),
    d_reverse_graph.data_handle(),
    d_reverse_graph_count.data_handle(),
    partition_size,
    graph_degree);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::copy(host_partition_graph.data(),
             d_partition_graph.data_handle(),
             host_partition_graph.size(),
             stream);
  raft::resource::sync_stream(res);

  for (uint32_t local_node = 0; local_node < partition_size; ++local_node) {
    std::copy_n(
      host_partition_graph.data() + static_cast<size_t>(local_node) * graph_degree,
      graph_degree,
      graph.data_handle() + static_cast<size_t>(partition_nodes[local_node]) * graph_degree);
  }
}

template <typename IdxT>
void ace_add_global_reverse_edges(raft::resources const& res,
                                  raft::host_matrix_view<IdxT, int64_t> graph,
                                  raft::host_matrix_view<IdxT, int64_t> partition_labels,
                                  raft::host_vector_view<IdxT, int64_t> core_backward_mapping,
                                  raft::host_vector_view<IdxT, int64_t> core_partition_offsets)
{
  auto const dataset_size = static_cast<size_t>(graph.extent(0));
  auto const graph_degree = static_cast<uint32_t>(graph.extent(1));
  auto const n_partitions = static_cast<size_t>(core_partition_offsets.extent(0) - 1);
  RAFT_EXPECTS(partition_labels.extent(0) == graph.extent(0),
               "ACE: partition labels must match the graph size");
  RAFT_EXPECTS(core_backward_mapping.extent(0) == graph.extent(0),
               "ACE: core mapping must match the graph size");

  auto source_graph = raft::make_host_matrix<IdxT, int64_t>(graph.extent(0), graph.extent(1));
  std::copy_n(graph.data_handle(), graph.size(), source_graph.data_handle());
  std::vector<std::unordered_set<IdxT>> candidate_sets(n_partitions);
  for (size_t source = 0; source < dataset_size; ++source) {
    auto const source_partition = static_cast<size_t>(partition_labels(source, 0));
    for (uint32_t rank = 0; rank < graph_degree; ++rank) {
      auto const destination = source_graph(source, rank);
      RAFT_EXPECTS(destination < dataset_size, "ACE: graph contains an invalid neighbor");
      auto const destination_partition = static_cast<size_t>(partition_labels(destination, 0));
      if (source_partition != destination_partition) {
        candidate_sets[destination_partition].insert(static_cast<IdxT>(source));
      }
    }
  }

  for (size_t partition_id = 0; partition_id < n_partitions; ++partition_id) {
    if (candidate_sets[partition_id].empty()) { continue; }
    auto const first = static_cast<size_t>(core_partition_offsets(partition_id));
    auto const last  = static_cast<size_t>(core_partition_offsets(partition_id + 1));
    std::vector<IdxT> partition_nodes(core_backward_mapping.data_handle() + first,
                                      core_backward_mapping.data_handle() + last);
    std::vector<IdxT> candidate_sources(candidate_sets[partition_id].begin(),
                                        candidate_sets[partition_id].end());
    ace_merge_partition_reverse_edges(
      res, source_graph.view(), graph, partition_nodes, std::move(candidate_sources));
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
template <typename T, typename IdxT, typename DatasetViewT>
  requires cuvs::neighbors::is_host_dataset_view_v<DatasetViewT>
auto build_ace(raft::resources const& res, const index_params& params, DatasetViewT const& dataset)
  -> cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>
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
    "cagra::detail::build_ace<host>(%zu, %zu, %zu)",
    params.intermediate_graph_degree,
    params.graph_degree,
    npartitions);

  auto dataset_view   = dataset.view();
  size_t dataset_size = dataset.n_rows();
  size_t dataset_dim  = dataset.dim();

  RAFT_EXPECTS(dataset_size > 0, "ACE: Dataset must not be empty");
  if (dataset_size < 1000) {
    RAFT_LOG_WARN("ACE: Very small dataset size (%zu), consider using regular CAGRA build instead.",
                  dataset_size);
  }
  RAFT_EXPECTS(dataset_dim > 0, "ACE: Dataset dimension must be greater than 0");
  RAFT_EXPECTS(params.intermediate_graph_degree > 0,
               "ACE: Intermediate graph degree must be greater than 0");
  RAFT_EXPECTS(params.graph_degree > 0, "ACE: Graph degree must be greater than 0");

  size_t n_partitions = ace_resolve_partition_count(npartitions);

  ace_validate_partition_count(n_partitions, dataset_size);

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

  ace_disk_workspace workspace(build_dir);

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
    RAFT_EXPECTS(!ace_params.add_global_reverse_edges || !use_disk_mode,
                 "ACE: partitioned global reverse-edge addition does not support disk mode");

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
      ace_validate_partition_count(n_partitions, dataset_size, true);
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
      workspace.initialize();

      // Create numpy files with pre-allocated space
      std::tie(reordered_fd, reordered_header_size) = cuvs::util::create_numpy_file<T>(
        workspace.artifact_path(ace_disk_workspace::artifact::reordered_dataset),
        {dataset_size, dataset_dim},
        true);
      workspace.mark_artifact_created(ace_disk_workspace::artifact::reordered_dataset);

      std::tie(augmented_fd, augmented_header_size) = cuvs::util::create_numpy_file<T>(
        workspace.artifact_path(ace_disk_workspace::artifact::augmented_dataset),
        {dataset_size, dataset_dim},
        true);
      workspace.mark_artifact_created(ace_disk_workspace::artifact::augmented_dataset);

      std::tie(mapping_fd, mapping_header_size) = cuvs::util::create_numpy_file<IdxT>(
        workspace.artifact_path(ace_disk_workspace::artifact::dataset_mapping),
        {dataset_size},
        true);
      workspace.mark_artifact_created(ace_disk_workspace::artifact::dataset_mapping);

      std::tie(graph_fd, graph_header_size) = cuvs::util::create_numpy_file<IdxT>(
        workspace.artifact_path(ace_disk_workspace::artifact::cagra_graph),
        {dataset_size, graph_degree},
        true);
      workspace.mark_artifact_created(ace_disk_workspace::artifact::cagra_graph);

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

    ace_get_partition_labels<T, IdxT>(res,
                                      dataset_view,
                                      dataset_dim,
                                      partition_labels.view(),
                                      partition_histogram.view(),
                                      min_partition_size);

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
                                             dataset_view,
                                             dataset_dim,
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

      // Create index for this partition
      auto sub_index_params = cuvs::neighbors::cagra::index_params::from_dataset(
        raft::make_extents<int64_t>(sub_dataset_size, dataset_dim),
        graph_degree,
        params.metric,
        ef_construction / 16);
      sub_index_params.attach_dataset_on_build = false;
      sub_index_params.guarantee_connectivity  = params.guarantee_connectivity;

      auto read_end             = start;
      bool used_device_read     = false;
      const size_t sub_ds_bytes = sub_dataset_size * dataset_dim * sizeof(T);
      using device_sub_index_t  = cuvs::neighbors::cagra::device_padded_index<T, IdxT>;
      using host_sub_index_t    = cuvs::neighbors::cagra::host_standard_index<T, IdxT>;
      using sub_index_t         = std::variant<device_sub_index_t, host_sub_index_t>;
      std::optional<raft::host_matrix<IdxT, int64_t>> partial_sub_graph;
      std::optional<sub_index_t> sub_index;
      const bool build_partial_graph_on_host =
        ace_params.add_global_reverse_edges ||
        ace_params.partial_graph_knn_build_algo ==
          graph_build_params::ace_params::knn_graph_build_algo::BRUTE_FORCE;
      if (build_partial_graph_on_host) {
        auto core_argument_partition =
          raft::make_host_matrix<T, int64_t>(sub_dataset_size, dataset_dim);
        if (use_disk_mode) {
          ace_load_partition_dataset_from_disk<T, IdxT>(partition_id,
                                                        dataset_dim,
                                                        partition_histogram.view(),
                                                        core_partition_offsets.view(),
                                                        augmented_partition_offsets.view(),
                                                        reordered_fd,
                                                        augmented_fd,
                                                        reordered_header_size,
                                                        augmented_header_size,
                                                        core_argument_partition.view());
        } else {
          ace_make_core_argument_partition<T, IdxT>(core_sub_dataset_size,
                                                    augmented_sub_dataset_size,
                                                    dataset_dim,
                                                    partition_id,
                                                    dataset_view,
                                                    core_backward_mapping.view(),
                                                    augmented_backward_mapping.view(),
                                                    core_partition_offsets.view(),
                                                    augmented_partition_offsets.view(),
                                                    core_argument_partition.view());
        }
        read_end                          = std::chrono::high_resolution_clock::now();
        auto core_argument_partition_view = raft::make_host_matrix_view<const T, int64_t>(
          core_argument_partition.data_handle(), sub_dataset_size, dataset_dim);
        if (ace_params.add_global_reverse_edges) {
          partial_sub_graph.emplace(
            ace_build_pruned_partial_graph<T, IdxT>(res,
                                                    core_argument_partition_view,
                                                    intermediate_degree,
                                                    graph_degree,
                                                    params.metric,
                                                    ace_params.partial_graph_knn_build_algo));
        } else {
          partial_sub_graph.emplace(
            ace_build_optimized_partial_graph<T, IdxT>(res,
                                                       core_argument_partition_view,
                                                       intermediate_degree,
                                                       graph_degree,
                                                       params.metric,
                                                       ace_params.partial_graph_knn_build_algo,
                                                       params.guarantee_connectivity));
        }
      } else {
        sub_index.emplace([&]() -> sub_index_t {
          if (use_disk_mode) {
            const size_t current_free_gpu = rmm::available_device_memory().first;
            const size_t configured_gpu =
              ace_params.max_gpu_memory_gb > 0
                ? static_cast<size_t>(ace_params.max_gpu_memory_gb * (1ULL << 30))
                : current_free_gpu;
            const size_t free_gpu_bytes = std::min(current_free_gpu, configured_gpu);
            if (sub_ds_bytes < static_cast<size_t>(0.4 * free_gpu_bytes)) {
              try {
                auto sub_dataset_tight =
                  raft::make_device_matrix<T, int64_t>(res, sub_dataset_size, dataset_dim);
                raft::resource::sync_stream(res);
                ace_load_partition_dataset_from_disk<T, IdxT>(partition_id,
                                                              dataset_dim,
                                                              partition_histogram.view(),
                                                              core_partition_offsets.view(),
                                                              augmented_partition_offsets.view(),
                                                              reordered_fd,
                                                              augmented_fd,
                                                              reordered_header_size,
                                                              augmented_header_size,
                                                              sub_dataset_tight.view());
                read_end              = std::chrono::high_resolution_clock::now();
                auto sub_dataset_view = raft::make_const_mdspan(sub_dataset_tight.view());
                std::unique_ptr<cuvs::neighbors::device_padded_dataset<T, int64_t>>
                  sub_dataset_padded;
                auto sub_dataset_dev = [&]() {
                  if (cuvs::neighbors::matrix_row_width_matches_cagra_required(sub_dataset_view)) {
                    return cuvs::neighbors::make_device_padded_dataset_view(res, sub_dataset_view);
                  }
                  sub_dataset_padded =
                    cuvs::neighbors::make_device_padded_dataset(res, sub_dataset_view);
                  return sub_dataset_padded->as_dataset_view();
                }();
                auto direct_index =
                  ::cuvs::neighbors::cagra::detail::build_from_device_matrix<T, IdxT>(
                    res, sub_index_params, sub_dataset_dev);
                used_device_read = true;
                return sub_index_t{std::in_place_type<device_sub_index_t>, std::move(direct_index)};
              } catch (const std::bad_alloc& e) {
                RAFT_LOG_WARN(
                  "ACE: partition %lu did not fit in device memory for a direct (GDS) read: %s; "
                  "falling back to a host read",
                  partition_id,
                  e.what());
              }
            }

            auto sub_dataset = raft::make_host_matrix<T, int64_t>(sub_dataset_size, dataset_dim);
            ace_load_partition_dataset_from_disk<T, IdxT>(partition_id,
                                                          dataset_dim,
                                                          partition_histogram.view(),
                                                          core_partition_offsets.view(),
                                                          augmented_partition_offsets.view(),
                                                          reordered_fd,
                                                          augmented_fd,
                                                          reordered_header_size,
                                                          augmented_header_size,
                                                          sub_dataset.view());
            read_end              = std::chrono::high_resolution_clock::now();
            auto sub_dataset_view = cuvs::neighbors::make_host_standard_dataset_view(
              raft::make_const_mdspan(sub_dataset.view()));
            auto host_index =
              ::cuvs::neighbors::cagra::build(res, sub_index_params, sub_dataset_view);
            static_assert(std::is_same_v<decltype(host_index), host_sub_index_t>);
            return sub_index_t{std::in_place_type<host_sub_index_t>, std::move(host_index)};
          }

          auto sub_dataset = raft::make_host_matrix<T, int64_t>(sub_dataset_size, dataset_dim);
          ace_make_core_argument_partition<T, IdxT>(core_sub_dataset_size,
                                                    augmented_sub_dataset_size,
                                                    dataset_dim,
                                                    partition_id,
                                                    dataset_view,
                                                    core_backward_mapping.view(),
                                                    augmented_backward_mapping.view(),
                                                    core_partition_offsets.view(),
                                                    augmented_partition_offsets.view(),
                                                    sub_dataset.view());
          read_end              = std::chrono::high_resolution_clock::now();
          auto sub_dataset_view = cuvs::neighbors::make_host_standard_dataset_view(
            raft::make_const_mdspan(sub_dataset.view()));
          auto host_index =
            ::cuvs::neighbors::cagra::build(res, sub_index_params, sub_dataset_view);
          static_assert(std::is_same_v<decltype(host_index), host_sub_index_t>);
          return sub_index_t{std::in_place_type<host_sub_index_t>, std::move(host_index)};
        }());
      }
      if (used_device_read) {
        RAFT_LOG_DEBUG("ACE: partition %lu read directly into device memory (GDS path)",
                       partition_id);
      }
      auto read_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(read_end - start).count();
      auto optimize_end = std::chrono::high_resolution_clock::now();
      auto optimize_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(optimize_end - read_end).count();

      auto adjust_end          = optimize_end;
      auto write_elapsed       = 0L;
      const size_t graph_bytes = core_sub_dataset_size * graph_degree * sizeof(IdxT);
      if (use_disk_mode) {
        auto adjusted_search_graph =
          raft::make_device_matrix<IdxT, int64_t>(res, core_sub_dataset_size, graph_degree);
        auto adjust_sub_graph = [&](auto sub_graph) {
          ace_adjust_sub_graph_ids_disk<IdxT>(res,
                                              core_sub_dataset_size,
                                              augmented_sub_dataset_size,
                                              graph_degree,
                                              partition_id,
                                              sub_graph,
                                              adjusted_search_graph.view(),
                                              core_partition_offsets.view(),
                                              augmented_partition_offsets.view(),
                                              augmented_backward_mapping.view(),
                                              core_forward_mapping.view());
        };
        if (build_partial_graph_on_host) {
          auto partial_sub_graph_device =
            raft::make_device_matrix<IdxT, int64_t>(res, core_sub_dataset_size, graph_degree);
          raft::copy(res,
                     partial_sub_graph_device.view(),
                     raft::make_host_matrix_view<const IdxT, int64_t>(
                       partial_sub_graph->data_handle(), core_sub_dataset_size, graph_degree));
          adjust_sub_graph(raft::make_const_mdspan(partial_sub_graph_device.view()));
        } else {
          adjust_sub_graph(std::visit([](auto const& index) { return index.graph(); }, *sub_index));
        }
        adjust_end = std::chrono::high_resolution_clock::now();

        const size_t graph_offset =
          static_cast<size_t>(core_partition_offsets(partition_id)) * graph_degree * sizeof(IdxT) +
          graph_header_size;
        cuvs::util::write_large_file(
          graph_fd, adjusted_search_graph.data_handle(), graph_bytes, graph_offset);
        auto write_end = std::chrono::high_resolution_clock::now();
        write_elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(write_end - adjust_end).count();
      } else {
        auto sub_search_graph =
          raft::make_host_matrix<IdxT, int64_t>(core_sub_dataset_size, graph_degree);
        if (build_partial_graph_on_host) {
          std::copy_n(partial_sub_graph->data_handle(),
                      sub_search_graph.size(),
                      sub_search_graph.data_handle());
        } else {
          auto sub_graph = std::visit([](auto const& index) { return index.graph(); }, *sub_index);
          raft::copy(
            res,
            raft::make_host_vector_view(sub_search_graph.data_handle(), sub_search_graph.size()),
            raft::make_device_vector_view(sub_graph.data_handle(), sub_search_graph.size()));
          raft::resource::sync_stream(res);
        }

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
        adjust_end = std::chrono::high_resolution_clock::now();
      }

      auto adjust_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(adjust_end - optimize_end).count();

      auto end        = std::chrono::high_resolution_clock::now();
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      double read_throughput =
        read_elapsed > 0
          ? to_mib(sub_dataset_size * dataset_dim * sizeof(T)) / (read_elapsed / 1000.0)
          : 0.0;
      double write_throughput =
        write_elapsed > 0 ? to_mib(graph_bytes) / (write_elapsed / 1000.0) : 0.0;
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

    if (ace_params.add_global_reverse_edges) {
      RAFT_LOG_INFO("ACE: Adding global cross-partition reverse edges partition by partition");
      ace_add_global_reverse_edges(res,
                                   search_graph.view(),
                                   partition_labels.view(),
                                   core_backward_mapping.view(),
                                   core_partition_offsets.view());
    }

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
    cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT> idx(res, params.metric);
    if (!use_disk_mode) {
      if (params.attach_dataset_on_build) {
        idx = cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>(
          res, params.metric, dataset, raft::make_const_mdspan(search_graph.view()));
      } else {
        idx.update_graph(res, raft::make_const_mdspan(search_graph.view()));
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

    workspace.commit();
    return std::move(idx);
  } catch (const std::exception& e) {
    RAFT_LOG_ERROR("ACE: Build failed with exception: %s", e.what());
    workspace.cleanup();
    throw;
  } catch (...) {
    workspace.cleanup();
    throw;
  }
}

}  // namespace cuvs::neighbors::cagra::detail
