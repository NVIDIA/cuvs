/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Large graphs exceeding the memory capacity can be built using the Augmented Core Extraction (ACE)
// algorithm, which partitions the dataset. The resulting HNSW index is too large to fit in memory
// as well. Thus, the index needs to be transferred to a search server with enough memory.
//
// HNSWHierarchy::GPU_LAYERED_ON_DISK is a special hierarchy that builds a layered HNSW index on
// disk. It emits one topology-only artifact, hnsw_index.cuvs. The dataset remains separate and does
// not need to be transferred to the search server, which typically has the dataset locally.
//
// This example demonstrates how to build a layered HNSW index with ACE:
//
// 1. Optionally quantize the dataset and queries to int8.
// 2. Build a single-file layered HNSW artifact with ACE using hnsw::build.
// 3. Deserialize the layered HNSW artifact using hnsw::deserialize.
// 4. Search the in-memory HNSW index.
//
// Layered-on-disk layout:
//
//   index_dir/hnsw_index.cuvs
//     fixed header + metadata JSON
//     levels: uint8 [N], max HNSW level for each original row id
//     base nodes + base links: uint32 node ids with hnswlib-ready link rows
//     upper nodes + upper links: hnswlib-ready upper-layer topology
//
// The transferred index artifact is topology-only. The dataset is loaded locally during
// deserialization from hnsw::index_params::dataset_path. The loader supports .npy and ANN benchmark
// *.bin datasets; this example writes a local dataset .npy only to make the demo self-contained.
//
// Layer 0 node IDs and neighbor IDs are original dataset row IDs. Upper layers are generated with
// the same level/order/KNN logic as serialize_to_hnswlib_from_disk, then stored as hnswlib-ready
// link rows so deserialization does no graph remapping or link padding on the search node.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/random/make_blobs.cuh>

#include <cuvs/neighbors/hnsw.hpp>
#include <cuvs/preprocessing/quantize/scalar.hpp>
#include <cuvs/util/file_io.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include "common.cuh"

// When 1, scalar-quantize the float dataset to int8.
#define HNSW_ACE_LAYERED_USE_QUANTIZATION 1

namespace {

constexpr const char* kBuildDir = "/tmp/hnsw_ace_layered";

template <typename T>
std::string write_local_dataset(raft::host_matrix_view<const T, int64_t> dataset,
                                const std::string& path)
{
  auto [fd, header_size] = cuvs::util::create_numpy_file<T>(
    path, {static_cast<size_t>(dataset.extent(0)), static_cast<size_t>(dataset.extent(1))});
  cuvs::util::write_large_file(
    fd, dataset.data_handle(), dataset.extent(0) * dataset.extent(1) * sizeof(T), header_size);
  return path;
}

template <typename T>
struct quantized_pair {
  raft::host_matrix<T, int64_t> dataset;
  raft::host_matrix<T, int64_t> queries;
};

quantized_pair<int8_t> quantize_dataset(raft::device_resources const& dev_resources,
                                        raft::host_matrix_view<const float, int64_t> dataset_float,
                                        raft::host_matrix_view<const float, int64_t> queries_float)
{
  std::cout << "  quantize_dataset: training scalar quantizer (float -> int8)" << std::endl;
  cuvs::preprocessing::quantize::scalar::params qp;
  auto quantizer = cuvs::preprocessing::quantize::scalar::train(dev_resources, qp, dataset_float);

  auto dataset_i8 =
    raft::make_host_matrix<int8_t, int64_t>(dataset_float.extent(0), dataset_float.extent(1));
  cuvs::preprocessing::quantize::scalar::transform(
    dev_resources, quantizer, dataset_float, dataset_i8.view());

  auto queries_i8 =
    raft::make_host_matrix<int8_t, int64_t>(queries_float.extent(0), queries_float.extent(1));
  cuvs::preprocessing::quantize::scalar::transform(
    dev_resources, quantizer, queries_float, queries_i8.view());

  return {std::move(dataset_i8), std::move(queries_i8)};
}

auto make_hnsw_ace_params(const std::string& build_dir, const std::string& dataset_path)
  -> cuvs::neighbors::hnsw::index_params
{
  using namespace cuvs::neighbors;

  hnsw::index_params hnsw_params;
  hnsw_params.metric          = cuvs::distance::DistanceType::L2Expanded;
  hnsw_params.hierarchy       = hnsw::HnswHierarchy::GPU_LAYERED_ON_DISK;
  hnsw_params.M               = 32;
  hnsw_params.dataset_path    = dataset_path;  // Override this path on the search server.
  hnsw_params.ef_construction = 120;

  auto ace_params                = hnsw::graph_build_params::ace_params();
  ace_params.npartitions         = 4;
  ace_params.use_disk            = true;
  ace_params.build_dir           = build_dir;
  hnsw_params.graph_build_params = ace_params;

  return hnsw_params;
}

template <typename T>
auto hnsw_build(raft::device_resources const& dev_resources,
                const cuvs::neighbors::hnsw::index_params& hnsw_params,
                raft::host_matrix_view<const T, int64_t> dataset) -> std::string
{
  using namespace cuvs::neighbors;

  auto hnsw_index          = hnsw::build(dev_resources, hnsw_params, dataset);
  const auto artifact_path = hnsw_index->file_path();
  if (artifact_path.empty()) {
    throw std::runtime_error("Expected layered HNSW build to return an artifact path.");
  }
  std::cout << "  hnsw_build: layered artifact written to " << artifact_path << std::endl;
  return artifact_path;
}

template <typename T>
auto hnsw_deserialize(raft::device_resources const& dev_resources,
                      const cuvs::neighbors::hnsw::index_params& hnsw_params,
                      const std::string& artifact_path,
                      int64_t dim) -> std::unique_ptr<cuvs::neighbors::hnsw::index<T>>
{
  using namespace cuvs::neighbors;

  hnsw::index<T>* deserialized_index = nullptr;
  // Set params.dataset_path to the local dataset path to load the dataset from disk.
  // hnsw_params.dataset_path = "/tmp/dataset.npy";
  hnsw::deserialize(
    dev_resources, hnsw_params, artifact_path, dim, hnsw_params.metric, &deserialized_index);
  return std::unique_ptr<hnsw::index<T>>(deserialized_index);
}

template <typename T>
void hnsw_search(raft::device_resources const& dev_resources,
                 const cuvs::neighbors::hnsw::index<T>& hnsw_index,
                 raft::host_matrix_view<const T, int64_t> queries,
                 int64_t topk = 12)
{
  using namespace cuvs::neighbors;

  const int64_t n_queries  = queries.extent(0);
  auto indices_hnsw_host   = raft::make_host_matrix<uint64_t, int64_t>(n_queries, topk);
  auto distances_hnsw_host = raft::make_host_matrix<float, int64_t>(n_queries, topk);

  hnsw::search_params search_params;
  search_params.ef          = std::max(200, static_cast<int>(topk) * 2);
  search_params.num_threads = 1;

  hnsw::search(dev_resources,
               search_params,
               hnsw_index,
               queries,
               indices_hnsw_host.view(),
               distances_hnsw_host.view());

  auto neighbors      = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances      = raft::make_device_matrix<float>(dev_resources, n_queries, topk);
  auto neighbors_host = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
  for (int64_t i = 0; i < n_queries; ++i) {
    for (int64_t j = 0; j < topk; ++j) {
      neighbors_host(i, j) = static_cast<uint32_t>(indices_hnsw_host(i, j));
    }
  }

  raft::copy(neighbors.data_handle(),
             neighbors_host.data_handle(),
             n_queries * topk,
             raft::resource::get_cuda_stream(dev_resources));
  raft::copy(distances.data_handle(),
             distances_hnsw_host.data_handle(),
             n_queries * topk,
             raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  print_results(dev_resources, neighbors.view(), distances.view());
}

}  // namespace

int main()
{
  raft::device_resources dev_resources;

  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

#if HNSW_ACE_LAYERED_USE_QUANTIZATION
  std::cout << "[stage 1] Generate and quantize dataset (float -> int8)" << std::endl;
#else
  std::cout << "[stage 1] Generate dataset (float)" << std::endl;
#endif

  int64_t n_samples = 10000;
  int64_t n_dim     = 90;
  int64_t n_queries = 10;
  auto dataset      = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries      = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);
  generate_dataset(dev_resources, dataset.view(), queries.view());

  auto dataset_host = raft::make_host_matrix<float, int64_t>(n_samples, n_dim);
  auto queries_host = raft::make_host_matrix<float, int64_t>(n_queries, n_dim);
  raft::copy(dataset_host.data_handle(),
             dataset.data_handle(),
             dataset.extent(0) * dataset.extent(1),
             raft::resource::get_cuda_stream(dev_resources));
  raft::copy(queries_host.data_handle(),
             queries.data_handle(),
             queries.extent(0) * queries.extent(1),
             raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  auto dataset_host_view = raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
    dataset_host.data_handle(), n_samples, n_dim);
  auto queries_host_view = raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
    queries_host.data_handle(), n_queries, n_dim);

  std::filesystem::create_directories(kBuildDir);

#if HNSW_ACE_LAYERED_USE_QUANTIZATION
  auto q               = quantize_dataset(dev_resources, dataset_host_view, queries_host_view);
  auto dataset_i8_view = raft::make_host_matrix_view<const int8_t, int64_t, raft::row_major>(
    q.dataset.data_handle(), n_samples, n_dim);
  auto queries_i8_view = raft::make_host_matrix_view<const int8_t, int64_t, raft::row_major>(
    q.queries.data_handle(), n_queries, n_dim);
  auto dataset_path = write_local_dataset(dataset_i8_view, std::string{kBuildDir} + "/dataset.npy");
  auto hnsw_params  = make_hnsw_ace_params(kBuildDir, dataset_path);

  std::cout << "[stage 2] Build layered HNSW index with ACE" << std::endl;
  auto artifact_path = hnsw_build<int8_t>(dev_resources, hnsw_params, dataset_i8_view);

  std::cout << "[stage 3] Deserialize layered HNSW index" << std::endl;
  auto hnsw_index = hnsw_deserialize<int8_t>(dev_resources, hnsw_params, artifact_path, n_dim);

  std::cout << "[stage 4] Search HNSW index" << std::endl;
  hnsw_search<int8_t>(dev_resources, *hnsw_index, queries_i8_view);
#else
  auto dataset_path =
    write_local_dataset(dataset_host_view, std::string{kBuildDir} + "/dataset.npy");
  auto hnsw_params = make_hnsw_ace_params(kBuildDir, dataset_path);

  std::cout << "[stage 2] Build layered HNSW index with ACE" << std::endl;
  auto artifact_path = hnsw_build<float>(dev_resources, hnsw_params, dataset_host_view);

  std::cout << "[stage 3] Deserialize layered HNSW index" << std::endl;
  auto hnsw_index = hnsw_deserialize<float>(dev_resources, hnsw_params, artifact_path, n_dim);

  std::cout << "[stage 4] Search HNSW index" << std::endl;
  hnsw_search<float>(dev_resources, *hnsw_index, queries_host_view);
#endif

  return 0;
}
