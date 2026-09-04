/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"
#include "../../core/omp_wrapper.hpp"

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/hnsw.hpp>
#include <cuvs/util/file_io.hpp>
#include <cuvs/util/host_memory.hpp>

#include <kvikio/file_handle.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/util/cudart_utils.hpp>

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <library_types.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <random>
#include <sys/mman.h>
#include <thread>
#include <type_traits>
#include <unistd.h>

namespace cuvs::neighbors::hnsw::detail {

template <typename T>
void all_neighbors_graph(raft::resources const& res,
                         raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
                         raft::host_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
                         cuvs::distance::DistanceType metric);

}  // namespace cuvs::neighbors::hnsw::detail

#include "hnsw/external_build.cuh"
#include "hnsw/index_impl.hpp"
#include "hnsw/serialize_layout.hpp"

namespace cuvs::neighbors::hnsw::detail {

class exclusive_hnsw_output_file {
 public:
  explicit exclusive_hnsw_output_file(std::filesystem::path output_path)
    : output_path_{std::move(output_path)}
  {
    std::string temporary_path = output_path_.string() + ".tmp.XXXXXX";
    int fd                     = ::mkstemp(temporary_path.data());
    RAFT_EXPECTS(fd != -1,
                 "Cannot create temporary file for %s (errno: %d, %s)",
                 output_path_.c_str(),
                 errno,
                 strerror(errno));
    temporary_path_ = std::move(temporary_path);

    if (::close(fd) != 0) {
      const int error = errno;
      cleanup();
      RAFT_FAIL("Cannot close temporary file for %s (errno: %d, %s)",
                output_path_.c_str(),
                error,
                strerror(error));
    }

    try {
      stream_ = std::make_unique<cuvs::util::kvikio_ofstream>(temporary_path_);
    } catch (...) {
      cleanup();
      throw;
    }
  }

  exclusive_hnsw_output_file(const exclusive_hnsw_output_file&)            = delete;
  exclusive_hnsw_output_file& operator=(const exclusive_hnsw_output_file&) = delete;

  ~exclusive_hnsw_output_file()
  {
    stream_.reset();
    cleanup();
  }

  std::ostream& stream() { return *stream_; }

  void publish()
  {
    stream_->close();
    RAFT_EXPECTS(*stream_, "Error writing output %s", output_path_.c_str());
    stream_.reset();

    if (::link(temporary_path_.c_str(), output_path_.c_str()) != 0) {
      const int error = errno;
      RAFT_FAIL("Cannot publish HNSW index %s (errno: %d, %s)",
                output_path_.c_str(),
                error,
                strerror(error));
    }
  }

 private:
  void cleanup() noexcept
  {
    if (!temporary_path_.empty()) { (void)::unlink(temporary_path_.c_str()); }
  }

  std::filesystem::path output_path_;
  std::string temporary_path_;
  std::unique_ptr<cuvs::util::kvikio_ofstream> stream_;
};

template <typename T, typename CagraIndexT>
inline constexpr bool is_cagra_hnsw_export_index_v =
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::device_padded_index<T, uint32_t>> ||
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::device_standard_index<T, uint32_t>> ||
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::host_padded_index<T, uint32_t>> ||
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::host_standard_index<T, uint32_t>>;

template <typename T, typename CagraIndexT>
inline constexpr bool is_host_cagra_hnsw_export_index_v =
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::host_padded_index<T, uint32_t>> ||
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::host_standard_index<T, uint32_t>>;

template <typename T, typename CagraIndexT>
inline constexpr bool is_device_cagra_hnsw_export_index_v =
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::device_padded_index<T, uint32_t>> ||
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::device_standard_index<T, uint32_t>>;

// Map the dataset element type to a cudaDataType_t. This is a host-only helper that
// intentionally avoids pulling CUDA/device dependencies.
template <typename T>
cudaDataType_t to_cuda_data_type();
template <>
inline cudaDataType_t to_cuda_data_type<float>()
{
  return CUDA_R_32F;
}
template <>
inline cudaDataType_t to_cuda_data_type<half>()
{
  return CUDA_R_16F;
}
template <>
inline cudaDataType_t to_cuda_data_type<int8_t>()
{
  return CUDA_R_8I;
}
template <>
inline cudaDataType_t to_cuda_data_type<uint8_t>()
{
  return CUDA_R_8U;
}

template <typename T, HnswHierarchy hierarchy, typename CagraIndexT>
std::enable_if_t<hierarchy == HnswHierarchy::NONE && is_cagra_hnsw_export_index_v<T, CagraIndexT>,
                 std::unique_ptr<index<T>>>
from_cagra(raft::resources const& res,
           const index_params& params,
           CagraIndexT const& cagra_index,
           std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  common::nvtx::range<common::nvtx::domain::cuvs> fun_scope("hnsw::from_cagra<NONE>");
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(0);
  auto uuid            = std::to_string(dist(rng));
  std::string filepath = "/tmp/" + uuid + ".bin";
  cuvs::neighbors::cagra::serialize_to_hnswlib(res, filepath, cagra_index, dataset);

  index<T>* hnsw_index = nullptr;
  int dim;
  if (dataset.has_value()) {
    dim = dataset.value().extent(1);
  } else {
    dim = cagra_index.dim();
  }

  cuvs::neighbors::hnsw::deserialize(res, params, filepath, dim, cagra_index.metric(), &hnsw_index);
  std::filesystem::remove(filepath);
  return std::unique_ptr<index<T>>(hnsw_index);
}

template <typename T, HnswHierarchy hierarchy, typename CagraIndexT>
std::enable_if_t<hierarchy == HnswHierarchy::CPU && is_cagra_hnsw_export_index_v<T, CagraIndexT>,
                 std::unique_ptr<index<T>>>
from_cagra(raft::resources const& res,
           const index_params& params,
           CagraIndexT const& cagra_index,
           std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  common::nvtx::range<common::nvtx::domain::cuvs> fun_scope("hnsw::from_cagra<CPU>");
  auto host_dataset = raft::make_host_matrix<T, int64_t>(0, 0);
  raft::host_matrix_view<const T, int64_t, raft::row_major> host_dataset_view(
    host_dataset.data_handle(), host_dataset.extent(0), host_dataset.extent(1));
  if (dataset.has_value()) {
    host_dataset_view = dataset.value();
  } else if constexpr (is_host_cagra_hnsw_export_index_v<T, CagraIndexT>) {
    RAFT_FAIL("hnsw::from_cagra<CPU> requires dataset for host CAGRA index");
  } else {
    // move dataset to host, remove padding
    auto dataset_view = cagra_index.dataset();
    RAFT_EXPECTS(dataset_view.n_rows() > 0,
                 "Invalid CAGRA dataset of size 0, shape %zux%zu",
                 static_cast<size_t>(dataset_view.n_rows()),
                 static_cast<size_t>(dataset_view.dim()));
    host_dataset = raft::make_host_matrix<T, int64_t>(dataset_view.n_rows(), dataset_view.dim());
    raft::copy_matrix(host_dataset.data_handle(),
                      host_dataset.extent(1),
                      dataset_view.view().data_handle(),
                      dataset_view.stride(),
                      host_dataset.extent(1),
                      dataset_view.n_rows(),
                      raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    host_dataset_view = host_dataset.view();
  }
  // build upper layers of hnsw index
  int dim         = host_dataset_view.extent(1);
  auto hnsw_index = std::make_unique<index_impl<T>>(dim, cagra_index.metric(), hierarchy);
  auto appr_algo  = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    hnsw_index->get_space(),
    host_dataset_view.extent(0),
    (cagra_index.graph().extent(1) + 1) / 2,
    params.ef_construction);
  appr_algo->base_layer_init = false;  // tell hnswlib to build upper layers only
  [[maybe_unused]] auto num_threads =
    params.num_threads == 0 ? cuvs::core::omp::get_max_threads() : params.num_threads;
#pragma omp parallel for num_threads(num_threads)
  for (int64_t i = 0; i < host_dataset_view.extent(0); i++) {
    appr_algo->addPoint((void*)(host_dataset_view.data_handle() + i * host_dataset_view.extent(1)),
                        i);
  }
  appr_algo->base_layer_init = true;  // reset to true to allow addition of new points

  // move cagra graph to host or access it from host if available
  auto host_graph_view = cagra_index.graph();
  auto host_graph      = raft::make_host_matrix<uint32_t, int64_t>(0, 0);
  if (!raft::is_host_accessible(raft::memory_type_from_pointer(host_graph_view.data_handle()))) {
    // copy cagra graph to host
    host_graph = raft::make_host_matrix<uint32_t, int64_t>(host_graph_view.extent(0),
                                                           host_graph_view.extent(1));
    raft::copy(res, host_graph.view(), host_graph_view);
    raft::resource::sync_stream(res);
    host_graph_view = host_graph.view();
  }

// copy cagra graph to hnswlib base layer
#pragma omp parallel for num_threads(num_threads)
  for (size_t i = 0; i < static_cast<size_t>(host_graph_view.extent(0)); ++i) {
    auto hnsw_internal_id = appr_algo->label_lookup_.find(i)->second;
    auto ll_i             = appr_algo->get_linklist0(hnsw_internal_id);
    appr_algo->setListCount(ll_i, host_graph_view.extent(1));
    auto* data = (uint32_t*)(ll_i + 1);
    for (size_t j = 0; j < static_cast<size_t>(host_graph_view.extent(1)); ++j) {
      auto neighbor_internal_id = appr_algo->label_lookup_.find(host_graph(i, j))->second;
      data[j]                   = neighbor_internal_id;
    }
  }

  hnsw_index->set_index(std::move(appr_algo));
  return hnsw_index;
}

template <typename T, typename DistT>
int initialize_point_in_hnsw(hnswlib::HierarchicalNSW<DistT>* appr_algo,
                             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
                             int64_t real_index,
                             int32_t curlevel)
{
  auto cur_c                        = appr_algo->cur_element_count++;
  appr_algo->element_levels_[cur_c] = curlevel;
  memset(appr_algo->data_level0_memory_ + cur_c * appr_algo->size_data_per_element_ +
           appr_algo->offsetLevel0_,
         0,
         appr_algo->size_data_per_element_);

  // Initialisation of the data and label
  memcpy(appr_algo->getExternalLabeLp(cur_c), &real_index, sizeof(hnswlib::labeltype));
  memcpy(appr_algo->getDataByInternalId(cur_c),
         dataset.data_handle() + real_index * dataset.extent(1),
         appr_algo->data_size_);

  if (curlevel) {
    appr_algo->linkLists_[cur_c] = (char*)malloc(appr_algo->size_links_per_element_ * curlevel + 1);
    if (appr_algo->linkLists_[cur_c] == nullptr)
      throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
    memset(appr_algo->linkLists_[cur_c], 0, appr_algo->size_links_per_element_ * curlevel + 1);
  }
  return cur_c;
}

template <typename T>
void all_neighbors_graph(raft::resources const& res,
                         raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
                         raft::host_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
                         cuvs::distance::DistanceType metric)
{
  // FIXME: choose better heuristic
  bool use_nn_decent = neighbors.size() < 1e7;
  if (use_nn_decent) {
    nn_descent::index_params nn_params;
    nn_params.graph_degree              = neighbors.extent(1);
    nn_params.intermediate_graph_degree = neighbors.extent(1) * 2;
    nn_params.metric                    = metric;
    nn_params.return_distances          = false;
    auto nn_index                       = nn_descent::build(res, nn_params, dataset, neighbors);
  } else {
    // TODO: choose parameters to minimize memory consumption
    cagra::graph_build_params::ivf_pq_params ivfpq_params(dataset.extents(), metric);
    cagra::build_knn_graph(res, dataset, neighbors, ivfpq_params);
  }
}

// Source-agnostic core that streams a CAGRA index into hnswlib format on disk.
// The disk-backed and in-memory variants differ only in the `read_batch` callable,
// which fills the provided host buffers (graph, dataset, labels) for a given row range.
template <typename T, typename IdxT, typename ReadBatchFn>
void serialize_to_hnswlib_batched(raft::resources const& res,
                                  std::ostream& os_raw,
                                  const cuvs::neighbors::hnsw::index_params& params,
                                  int64_t n_rows,
                                  int64_t dim,
                                  int graph_degree_int,
                                  cuvs::distance::DistanceType metric,
                                  ReadBatchFn read_batch)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("cagra::serialize");

  auto start_time = std::chrono::system_clock::now();

  cuvs::util::buffered_ofstream os(&os_raw, 1 << 20 /*1MB*/);

  RAFT_EXPECTS(params.hierarchy != HnswHierarchy::CPU,
               "Disk serialization not supported for CPU hierarchy.");

  RAFT_LOG_INFO("Saving CAGRA index to hnswlib format, size %zu, dim %zu, graph_degree %zu",
                static_cast<size_t>(n_rows),
                static_cast<size_t>(dim),
                static_cast<size_t>(graph_degree_int));

  const size_t row_size_bytes =
    graph_degree_int * sizeof(IdxT) + dim * sizeof(T) + sizeof(uint32_t);
  const size_t target_batch_bytes = 64 * 1024 * 1024;
  const size_t batch_size         = std::max<size_t>(1, target_batch_bytes / row_size_bytes);

  RAFT_LOG_DEBUG("Using batch size %zu rows (~%.2f MiB/batch)",
                 batch_size,
                 (batch_size * row_size_bytes) / (1024.0 * 1024.0));

  // Allocate buffers for batched reading
  auto graph_buffer   = raft::make_host_matrix<IdxT, int64_t>(batch_size, graph_degree_int);
  auto dataset_buffer = raft::make_host_matrix<T, int64_t>(batch_size, dim);
  auto label_buffer   = raft::make_host_vector<uint32_t, int64_t>(batch_size);

  RAFT_LOG_DEBUG("Allocated buffers: graph[%ld,%d], dataset[%ld,%ld], labels[%ld]",
                 graph_buffer.extent(0),
                 graph_degree_int,
                 dataset_buffer.extent(0),
                 dataset_buffer.extent(1),
                 label_buffer.extent(0));

  // initialize dummy HNSW index to retrieve constants
  auto hnsw_index = std::make_unique<index_impl<T>>(dim, metric, params.hierarchy);

  auto appr_algo = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    hnsw_index->get_space(), 1, (graph_degree_int + 1) / 2, params.ef_construction);

  bool create_hierarchy = params.hierarchy != HnswHierarchy::NONE;

  // create hierarchy order
  // sort the points by levels
  // roll dice & build histogram
  std::vector<size_t> hist;
  std::vector<size_t> order(n_rows);
  std::vector<size_t> order_bw(n_rows);
  std::vector<int> levels(n_rows);
  std::vector<size_t> offsets;

  if (create_hierarchy) {
    RAFT_LOG_INFO("Sort points by levels");
    for (int64_t i = 0; i < n_rows; i++) {
      auto pt_level = appr_algo->getRandomLevel(appr_algo->mult_);
      while (pt_level >= static_cast<int32_t>(hist.size()))
        hist.push_back(0);
      hist[pt_level]++;
      levels[i] = pt_level;
    }

    // accumulate
    offsets.resize(hist.size() + 1, 0);
    for (size_t i = 0; i < hist.size() - 1; i++) {
      offsets[i + 1] = offsets[i] + hist[i];
      RAFT_LOG_INFO("Level %zu : %zu", i + 1, size_t(n_rows) - offsets[i + 1]);
    }

    // fw/bw indices
    for (int64_t i = 0; i < n_rows; i++) {
      auto pt_level              = levels[i];
      order_bw[i]                = offsets[pt_level];
      order[offsets[pt_level]++] = i;
    }
  }

  // set last point of the highest level as the entry point
  appr_algo->enterpoint_node_ = create_hierarchy ? order.back() : n_rows / 2;
  appr_algo->maxlevel_        = create_hierarchy ? hist.size() - 1 : 1;
  auto serialize_layout =
    external::hnsw_serialize_layout_from_algorithm<T>(*appr_algo, n_rows, dim, graph_degree_int);

  // write header information
  RAFT_LOG_DEBUG("Writing HNSW header: offsetLevel0=%zu, n_rows=%zu, size_data_per_element=%zu",
                 appr_algo->offsetLevel0_,
                 static_cast<size_t>(n_rows),
                 appr_algo->size_data_per_element_);
  RAFT_LOG_DEBUG("  maxlevel=%d, enterpoint=%d, maxM=%zu, maxM0=%zu, M=%zu",
                 appr_algo->maxlevel_,
                 appr_algo->enterpoint_node_,
                 appr_algo->maxM_,
                 appr_algo->maxM0_,
                 appr_algo->M_);

  external::write_hnsw_header_fields(os, serialize_layout);

  // host queries
  auto host_query_set =
    raft::make_host_matrix<T, int64_t>(create_hierarchy ? n_rows - hist[0] : 0, dim);

  int64_t d_report_offset    = n_rows / 10;  // Report progress in 10% steps.
  int64_t next_report_offset = d_report_offset;
  auto start_clock           = std::chrono::system_clock::now();

  RAFT_EXPECTS(appr_algo->size_data_per_element_ ==
                 dim * sizeof(T) + appr_algo->maxM0_ * sizeof(IdxT) + sizeof(int) + sizeof(size_t),
               "Size data per element mismatch");

  RAFT_LOG_INFO("Writing base level");
  size_t bytes_written = 0;
  float GiB            = 1 << 30;
  RAFT_EXPECTS(appr_algo->size_data_per_element_ ==
                 dim * sizeof(T) + appr_algo->maxM0_ * sizeof(IdxT) + sizeof(int) + sizeof(size_t),
               "Size data per element mismatch");

  for (int64_t batch_start = 0; batch_start < n_rows; batch_start += batch_size) {
    const int64_t current_batch_size = std::min<int64_t>(batch_size, n_rows - batch_start);

    RAFT_LOG_DEBUG("Reading batch: start=%ld, size=%ld (batch_size=%zu)",
                   batch_start,
                   current_batch_size,
                   batch_size);
    read_batch(batch_start,
               current_batch_size,
               graph_buffer.view(),
               dataset_buffer.view(),
               label_buffer.view());

    for (int64_t batch_idx = 0; batch_idx < current_batch_size; batch_idx++) {
      const int64_t i = batch_start + batch_idx;

      const IdxT* graph_row = &graph_buffer(batch_idx, 0);
      const T* data_row     = &dataset_buffer(batch_idx, 0);
      static_assert(std::is_same_v<IdxT, uint32_t>,
                    "hnswlib serialization requires uint32_t neighbor IDs");
      external::write_hnsw_base_row<T>(
        os, serialize_layout, graph_row, data_row, label_buffer(batch_idx));

      if (create_hierarchy && levels[i] > 0) {
        // position in query: order_bw[i]-hist[0]
        std::memcpy(&host_query_set(order_bw[i] - hist[0], 0), data_row, dim * sizeof(T));
      }
      bytes_written += appr_algo->size_data_per_element_;

      const auto end_clock = std::chrono::system_clock::now();
      // if (!os.good()) { RAFT_FAIL("Error writing HNSW file, row %zu", i); }
      if (i > next_report_offset) {
        next_report_offset += d_report_offset;
        const auto time =
          std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() *
          1e-6;
        float throughput      = bytes_written / GiB / time;
        float rows_throughput = i / time;
        float ETA             = (n_rows - i) / rows_throughput;
        RAFT_LOG_INFO(
          "# Writing rows %12lu / %12lu (%3.2f %%), %3.2f GiB/sec, ETA %d:%3.1f, written %3.2f "
          "GiB\r",
          i,
          n_rows,
          i / static_cast<double>(n_rows) * 100,
          throughput,
          int(ETA / 60),
          std::fmod(ETA, 60.0f),
          bytes_written / GiB);
      }
    }
  }

  RAFT_LOG_DEBUG("Completed writing %ld base level rows", n_rows);

  // trigger knn builds for all levels
  std::vector<raft::host_matrix<IdxT, int64_t>> host_neighbors;
  if (create_hierarchy) {
    for (size_t pt_level = 1; pt_level < hist.size(); pt_level++) {
      auto num_pts       = n_rows - offsets[pt_level - 1];
      auto neighbor_size = num_pts > appr_algo->M_ ? appr_algo->M_ : num_pts - 1;
      host_neighbors.emplace_back(raft::make_host_matrix<IdxT, int64_t>(num_pts, neighbor_size));
    }
    for (size_t pt_level = 1; pt_level < hist.size(); pt_level++) {
      RAFT_LOG_INFO("Compute hierarchy neighbors level %zu", pt_level);
      auto removed_rows = offsets[pt_level - 1] - offsets[0];
      raft::host_matrix_view<T, int64_t, raft::row_major> sub_query_view(
        host_query_set.data_handle() + removed_rows * dim,
        host_query_set.extent(0) - removed_rows,
        dim);
      auto neighbor_view = host_neighbors[pt_level - 1].view();
      all_neighbors_graph(res, raft::make_const_mdspan(sub_query_view), neighbor_view, metric);
    }
  }

  if (create_hierarchy) {
    RAFT_LOG_INFO("Assemble hierarchy linklists");
    next_report_offset = d_report_offset;
  }
  bytes_written = 0;
  start_clock   = std::chrono::system_clock::now();

  std::vector<uint32_t> converted_neighbors(appr_algo->M_);
  for (int64_t i = 0; i < n_rows; i++) {
    size_t cur_level = create_hierarchy ? levels[i] : 0;
    external::write_hnsw_upper_node_header(os, serialize_layout, cur_level);
    unsigned int linkListSize =
      create_hierarchy && cur_level > 0 ? appr_algo->size_links_per_element_ * cur_level : 0;
    bytes_written += sizeof(int);
    if (linkListSize) {
      for (size_t pt_level = 1; pt_level <= cur_level; pt_level++) {
        auto neighbor_view = host_neighbors[pt_level - 1].view();
        auto my_row        = order_bw[i] - offsets[pt_level - 1];

        IdxT* neighbors     = &neighbor_view(my_row, 0);
        unsigned int extent = neighbor_view.extent(1);
        for (unsigned int j = 0; j < extent; j++) {
          converted_neighbors[j] = order[neighbors[j] + offsets[pt_level - 1]];
        }
        external::write_hnsw_upper_block(os, serialize_layout, converted_neighbors.data(), extent);
        auto remainder = appr_algo->M_ - neighbor_view.extent(1);
        bytes_written += (neighbor_view.extent(1) + remainder) * sizeof(IdxT) + sizeof(int);
        RAFT_EXPECTS(appr_algo->size_links_per_element_ ==
                       (neighbor_view.extent(1) + remainder) * sizeof(IdxT) + sizeof(int),
                     "Size links per element mismatch");
      }
    }

    const auto end_clock = std::chrono::system_clock::now();
    if (i > next_report_offset) {
      next_report_offset += d_report_offset;
      const auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() *
        1e-6;
      float throughput      = bytes_written / GiB / time;
      float rows_throughput = i / time;
      float ETA             = (n_rows - i) / rows_throughput;
      RAFT_LOG_INFO(
        "# Writing rows %12lu / %12lu (%3.2f %%), %3.2f GiB/sec, ETA %d:%3.1f, written %3.2f GiB\r",
        i,
        n_rows,
        i / static_cast<double>(n_rows) * 100,
        throughput,
        int(ETA / 60),
        std::fmod(ETA, 60.0f),
        bytes_written / GiB);
    }
  }

  // Flush buffered output and check data was written
  os.flush();
  os_raw.flush();
  auto final_pos = os_raw.tellp();
  RAFT_LOG_DEBUG("HNSW file size: %ld bytes", static_cast<int64_t>(final_pos));
  if (!os_raw.good()) { RAFT_LOG_WARN("Output stream is not in good state after serialization"); }

  auto end_time = std::chrono::system_clock::now();
  auto elapsed_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  RAFT_LOG_INFO("HNSW serialization complete in %ld ms", elapsed_time);
}

// Serialize a disk-backed CAGRA index into hnswlib format by reading graph/dataset/label
// rows directly from the backing files via pread.
template <typename CagraIndexT>
  requires is_cagra_hnsw_export_index_v<typename CagraIndexT::value_type, CagraIndexT>
void serialize_to_hnswlib_from_disk(raft::resources const& res,
                                    std::ostream& os_raw,
                                    const cuvs::neighbors::hnsw::index_params& params,
                                    CagraIndexT const& index_)
{
  using T    = typename CagraIndexT::value_type;
  using IdxT = typename CagraIndexT::index_type;
  RAFT_EXPECTS(index_.dataset_fd().has_value() && index_.graph_fd().has_value(),
               "Function only implements serialization from disk.");

  auto n_rows           = static_cast<int64_t>(index_.size());
  auto dim              = static_cast<int64_t>(index_.dim());
  auto graph_degree_int = static_cast<int>(index_.graph_degree());

  // Get file descriptors from index
  const auto& graph_fd_opt   = index_.graph_fd();
  const auto& dataset_fd_opt = index_.dataset_fd();
  const auto& mapping_fd_opt = index_.mapping_fd();

  RAFT_EXPECTS(graph_fd_opt.has_value() && graph_fd_opt->is_valid(),
               "Graph file descriptor is not available");
  RAFT_EXPECTS(dataset_fd_opt.has_value() && dataset_fd_opt->is_valid(),
               "Dataset file descriptor is not available");
  RAFT_EXPECTS(mapping_fd_opt.has_value() && mapping_fd_opt->is_valid(),
               "Mapping file descriptor is not available");

  // Get file paths from file descriptors
  std::string graph_path   = graph_fd_opt->get_path();
  std::string dataset_path = dataset_fd_opt->get_path();
  std::string mapping_path = mapping_fd_opt->get_path();

  RAFT_EXPECTS(!graph_path.empty(), "Unable to get path from graph file descriptor");
  RAFT_EXPECTS(!dataset_path.empty(), "Unable to get path from dataset file descriptor");
  RAFT_EXPECTS(!mapping_path.empty(), "Unable to get path from mapping file descriptor");

  // Open kvikio handles for the disk-backed artifacts. Reads here target host buffers (the hnswlib
  // layout is assembled on the CPU), so kvikio uses its POSIX + threadpool backend, with O_DIRECT
  // when available; it handles any alignment internally.
  kvikio::FileHandle graph_kv(graph_path, "r");
  kvikio::FileHandle dataset_kv(dataset_path, "r");
  kvikio::FileHandle label_kv(mapping_path, "r");

  // Read headers from files to get dimensions
  size_t graph_header_size = 0;
  size_t graph_n_rows      = 0;
  size_t graph_n_cols      = 0;
  {
    std::ifstream graph_stream(graph_path, std::ios::binary);
    RAFT_EXPECTS(graph_stream.good(), "Failed to open graph file: %s", graph_path.c_str());

    auto header       = raft::numpy_serializer::read_header(graph_stream);
    graph_header_size = static_cast<size_t>(graph_stream.tellg());
    RAFT_EXPECTS(
      header.shape.size() == 2, "Graph file should be 2D, got %zu dimensions", header.shape.size());

    graph_n_rows = header.shape[0];
    graph_n_cols = header.shape[1];
    RAFT_LOG_DEBUG("Graph file: %zu x %zu, header size: %zu bytes",
                   graph_n_rows,
                   graph_n_cols,
                   graph_header_size);
  }

  size_t dataset_header_size = 0;
  size_t dataset_n_rows      = 0;
  size_t dataset_n_cols      = 0;
  {
    std::ifstream dataset_stream(dataset_path, std::ios::binary);
    RAFT_EXPECTS(dataset_stream.good(), "Failed to open dataset file: %s", dataset_path.c_str());

    auto header         = raft::numpy_serializer::read_header(dataset_stream);
    dataset_header_size = static_cast<size_t>(dataset_stream.tellg());
    RAFT_EXPECTS(header.shape.size() == 2,
                 "Dataset file should be 2D, got %zu dimensions",
                 header.shape.size());

    dataset_n_rows = header.shape[0];
    dataset_n_cols = header.shape[1];
    RAFT_LOG_DEBUG("Dataset file: %zu x %zu, header size: %zu bytes",
                   dataset_n_rows,
                   dataset_n_cols,
                   dataset_header_size);
  }

  size_t label_header_size = 0;
  size_t label_n_elements  = 0;
  {
    std::ifstream mapping_stream(mapping_path, std::ios::binary);
    RAFT_EXPECTS(mapping_stream.good(), "Failed to open mapping file: %s", mapping_path.c_str());

    auto header       = raft::numpy_serializer::read_header(mapping_stream);
    label_header_size = static_cast<size_t>(mapping_stream.tellg());
    RAFT_EXPECTS(header.shape.size() == 1,
                 "Mapping file should be 1D, got %zu dimensions",
                 header.shape.size());

    label_n_elements = header.shape[0];
    RAFT_LOG_DEBUG(
      "Mapping file: %zu elements, header size: %zu bytes", label_n_elements, label_header_size);
  }

  // Verify consistency
  RAFT_EXPECTS(graph_n_rows == static_cast<size_t>(n_rows),
               "Graph rows (%zu) != index size (%zu)",
               graph_n_rows,
               static_cast<size_t>(n_rows));
  RAFT_EXPECTS(dataset_n_rows == static_cast<size_t>(n_rows),
               "Dataset rows (%zu) != index size (%zu)",
               dataset_n_rows,
               static_cast<size_t>(n_rows));
  RAFT_EXPECTS(label_n_elements == static_cast<size_t>(n_rows),
               "Label elements (%zu) != index size (%zu)",
               label_n_elements,
               static_cast<size_t>(n_rows));
  RAFT_EXPECTS(graph_n_cols == static_cast<size_t>(graph_degree_int),
               "Graph cols (%zu) != graph degree (%d)",
               graph_n_cols,
               graph_degree_int);
  RAFT_EXPECTS(dataset_n_cols == static_cast<size_t>(dim),
               "Dataset cols (%zu) != dimensions (%zu)",
               dataset_n_cols,
               static_cast<size_t>(dim));

  // Disk-specific batch reader: pread graph/dataset/label rows into the host buffers.
  auto read_batch = [&](int64_t start_row,
                        int64_t rows_to_read,
                        raft::host_matrix_view<IdxT, int64_t, raft::row_major> graph_buf,
                        raft::host_matrix_view<T, int64_t, raft::row_major> dataset_buf,
                        raft::host_vector_view<uint32_t, int64_t> label_buf) {
    RAFT_EXPECTS(start_row >= 0 && rows_to_read >= 0,
                 "Batch start row and row count must be non-negative");
    const size_t row          = static_cast<size_t>(start_row);
    const size_t rows         = static_cast<size_t>(rows_to_read);
    const size_t graph_degree = static_cast<size_t>(graph_degree_int);
    const size_t dim_size     = static_cast<size_t>(dim);

    const size_t graph_bytes   = rows * graph_degree * sizeof(IdxT);
    const size_t dataset_bytes = rows * dim_size * sizeof(T);
    const size_t label_bytes   = rows * sizeof(uint32_t);

    const size_t graph_offset   = graph_header_size + row * graph_degree * sizeof(IdxT);
    const size_t dataset_offset = dataset_header_size + row * dim_size * sizeof(T);
    const size_t label_offset   = label_header_size + row * sizeof(uint32_t);

    RAFT_LOG_DEBUG("Reading batch: row=%ld, rows=%ld", start_row, rows_to_read);

    // Issue the three reads concurrently through kvikio (its threadpool parallelizes each), then
    // wait for all to complete.
    auto graph_future = graph_kv.pread(graph_buf.data_handle(), graph_bytes, graph_offset);
    auto dataset_future =
      dataset_kv.pread(dataset_buf.data_handle(), dataset_bytes, dataset_offset);
    auto label_future = label_kv.pread(label_buf.data_handle(), label_bytes, label_offset);

    // Drain all three futures before propagating any failure.
    std::exception_ptr read_error;
    auto drain = [&read_error](auto& fut) -> size_t {
      try {
        return fut.get();
      } catch (...) {
        if (!read_error) { read_error = std::current_exception(); }
        return 0;
      }
    };
    const size_t graph_read   = drain(graph_future);
    const size_t dataset_read = drain(dataset_future);
    const size_t label_read   = drain(label_future);
    if (read_error) { std::rethrow_exception(read_error); }
    RAFT_EXPECTS(graph_read == graph_bytes,
                 "Short graph read at row %ld: expected %zu, got %zu",
                 start_row,
                 graph_bytes,
                 graph_read);
    RAFT_EXPECTS(dataset_read == dataset_bytes,
                 "Short dataset read at row %ld: expected %zu, got %zu",
                 start_row,
                 dataset_bytes,
                 dataset_read);
    RAFT_EXPECTS(label_read == label_bytes,
                 "Short label read at row %ld: expected %zu, got %zu",
                 start_row,
                 label_bytes,
                 label_read);
  };

  serialize_to_hnswlib_batched<T, IdxT>(
    res, os_raw, params, n_rows, dim, graph_degree_int, index_.metric(), read_batch);
}

// Serialize an in-memory CAGRA index into hnswlib format on disk, copying graph/dataset
// rows from the in-memory (device or host) structures batch by batch. This avoids
// materializing the full HNSW index in host memory.
template <typename CagraIndexT>
  requires is_cagra_hnsw_export_index_v<typename CagraIndexT::value_type, CagraIndexT>
void serialize_to_hnswlib_from_inmem(
  raft::resources const& res,
  std::ostream& os_raw,
  const cuvs::neighbors::hnsw::index_params& params,
  CagraIndexT const& index_,
  std::optional<
    raft::host_matrix_view<const typename CagraIndexT::value_type, int64_t, raft::row_major>>
    dataset)
{
  using T     = typename CagraIndexT::value_type;
  using IdxT  = typename CagraIndexT::index_type;
  auto stream = raft::resource::get_cuda_stream(res);
  [[maybe_unused]] auto num_threads =
    params.num_threads == 0 ? cuvs::core::omp::get_max_threads() : params.num_threads;

  // Resolve dataset source (host view if provided, else the CAGRA device dataset).
  const T* source_dataset = nullptr;
  int64_t n_rows, dim, source_stride;
  bool device_dataset;
  if (dataset.has_value()) {
    n_rows         = dataset->extent(0);
    dim            = dataset->extent(1);
    device_dataset = false;
    source_dataset = dataset->data_handle();
    source_stride  = dim;
  } else if constexpr (is_host_cagra_hnsw_export_index_v<T, CagraIndexT>) {
    RAFT_FAIL("serialize_to_hnswlib_from_inmem requires dataset for host CAGRA index");
  } else if (auto dataset_view = index_.dataset(); dataset_view.view().data_handle() != nullptr) {
    n_rows         = dataset_view.n_rows();
    dim            = dataset_view.dim();
    device_dataset = true;
    source_dataset = dataset_view.view().data_handle();
    source_stride  = dataset_view.stride();
  } else {
    RAFT_FAIL("serialize_to_hnswlib_from_inmem: No dataset provided");
  }

  // Resolve graph source and determine whether it is host-accessible.
  auto graph_view            = index_.graph();
  const int64_t degree       = graph_view.extent(1);
  const int graph_degree_int = static_cast<int>(degree);
  RAFT_EXPECTS(graph_view.extent(0) == n_rows,
               "Graph rows (%zu) != dataset rows (%zu)",
               static_cast<size_t>(graph_view.extent(0)),
               static_cast<size_t>(n_rows));

  const IdxT* graph_ptr = graph_view.data_handle();
  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, graph_ptr));
  bool graph_host_accessible = false;
  if (attr.type == cudaMemoryTypeUnregistered) {
    graph_host_accessible = true;
  } else if (attr.hostPointer != nullptr) {
    graph_ptr             = static_cast<const IdxT*>(attr.hostPointer);
    graph_host_accessible = true;
  }

  // In-memory batch reader: copy graph/dataset rows into the host buffers and assign
  // identity labels (in-memory CAGRA uses a 1:1 labeling, there is no mapping file).
  auto read_batch = [&](int64_t start_row,
                        int64_t rows_to_read,
                        raft::host_matrix_view<IdxT, int64_t, raft::row_major> graph_buf,
                        raft::host_matrix_view<T, int64_t, raft::row_major> dataset_buf,
                        raft::host_vector_view<uint32_t, int64_t> label_buf) {
    // graph rows
    if (graph_host_accessible) {
#pragma omp parallel for num_threads(num_threads)
      for (int64_t r = 0; r < rows_to_read; r++) {
        std::copy(graph_ptr + (start_row + r) * degree,
                  graph_ptr + (start_row + r + 1) * degree,
                  &graph_buf(r, 0));
      }
    } else {
      raft::copy_matrix(graph_buf.data_handle(),
                        degree,
                        graph_ptr + start_row * degree,
                        degree,
                        degree,
                        rows_to_read,
                        stream);
      raft::resource::sync_stream(res);
    }

    // dataset rows (drop any device-side row padding via the source stride)
    if (!device_dataset) {
#pragma omp parallel for num_threads(num_threads)
      for (int64_t r = 0; r < rows_to_read; r++) {
        std::copy(source_dataset + (start_row + r) * source_stride,
                  source_dataset + (start_row + r) * source_stride + dim,
                  &dataset_buf(r, 0));
      }
    } else {
      raft::copy_matrix(dataset_buf.data_handle(),
                        dim,
                        source_dataset + start_row * source_stride,
                        source_stride,
                        dim,
                        rows_to_read,
                        stream);
      raft::resource::sync_stream(res);
    }

    // identity labels
    std::iota(label_buf.data_handle(),
              label_buf.data_handle() + rows_to_read,
              static_cast<uint32_t>(start_row));
  };

  serialize_to_hnswlib_batched<T, IdxT>(
    res, os_raw, params, n_rows, dim, graph_degree_int, index_.metric(), read_batch);
}

template <typename T, HnswHierarchy hierarchy, typename CagraIndexT>
std::enable_if_t<hierarchy == HnswHierarchy::GPU && is_cagra_hnsw_export_index_v<T, CagraIndexT>,
                 std::unique_ptr<index<T>>>
from_cagra(raft::resources const& res,
           const index_params& params,
           CagraIndexT const& cagra_index,
           std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  common::nvtx::range<common::nvtx::domain::cuvs> fun_scope("hnsw::from_cagra<GPU>");
  auto stream = raft::resource::get_cuda_stream(res);
  auto num_threads =
    params.num_threads == 0 ? cuvs::core::omp::get_max_threads() : params.num_threads;

  /* Note: NNSW data layout

  offsetLevel0_      = seems to always be 0, and it would break things otherwise
  size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
  offsetData_        = size_links_level0_;
  label_offset_      = size_links_level0_ + data_size_;

  get_linklist0(i)       = data_level0_memory_ + i * size_data_per_element_ + offsetLevel0_
  getDataByInternalId(i) = data_level0_memory_ + i * size_data_per_element_ + offsetData_
  getExternalLabeLp(i)   = data_level0_memory_ + i * size_data_per_element_ + label_offset_

  Hence the layout:
      2M x uint32_t  +   1 x uint32_t        dim x T    1 x size_t
     [linked list + linked list sizes]        [data]     [label]
  */

  const T* source_dataset = nullptr;
  int64_t n_rows, dim, source_stride;
  bool device_copy;
  if (dataset.has_value()) {
    n_rows         = dataset->extent(0);
    dim            = dataset->extent(1);
    device_copy    = false;
    source_dataset = dataset->data_handle();
    source_stride  = dim;
  } else if constexpr (is_host_cagra_hnsw_export_index_v<T, CagraIndexT>) {
    RAFT_FAIL("hnsw::from_cagra<GPU> requires dataset for host CAGRA index");
  } else if (auto dataset_view = cagra_index.dataset();
             dataset_view.view().data_handle() != nullptr) {
    n_rows         = dataset_view.n_rows();
    dim            = dataset_view.dim();
    device_copy    = true;
    source_dataset = dataset_view.view().data_handle();
    source_stride  = dataset_view.stride();
  } else {
    RAFT_FAIL("hnsw::from_cagra<GPU>: No dataset provided");
  }

  // initialize HNSW index
  auto hnsw_index = std::make_unique<index_impl<T>>(dim, cagra_index.metric(), hierarchy);
  auto appr_algo  = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    hnsw_index->get_space(),
    n_rows,
    (cagra_index.graph().extent(1) + 1) / 2,
    params.ef_construction);
  appr_algo->cur_element_count = n_rows;

  // Initialize linked lists
  auto& levels = appr_algo->element_levels_;
  {
    common::nvtx::range<common::nvtx::domain::cuvs> block_scope(
      "parallel::initialize_data<%s>(%d threads)", device_copy ? "device" : "host", num_threads);
    /* Note: batching

    If the dataset is on the device, we want to copy it to HNSW in parallel to the rest of the
    initialization loop. Ideally, we could use cudaMemcpy2DAsync to do this, but this call is very
    likely to sync with the CPU, because normally the allocated host memory is paged. To avoid this,
    we use double-buffering and copy data in small batches via pinned memory. Hence, the cuda
    device-to-host copy is completely overlapped with the host loop.

    The batching is completely disabled if the source dataset is on the host.
    */
    const int64_t max_batch_size =
      device_copy ? raft::div_rounding_up_safe<int64_t>(64 * 1024 * 1024, source_stride * sizeof(T))
                  : n_rows;
    T* bufs[2]                                                  = {nullptr, nullptr};
    std::optional<raft::pinned_matrix<T, int64_t>> bufs_storage = std::nullopt;
    if (device_copy) {
      bufs_storage.emplace(
        std::move(raft::make_pinned_matrix<T, int64_t>(res, max_batch_size * 2, source_stride)));
      bufs[0] = bufs_storage->data_handle();
      bufs[1] = bufs[0] + max_batch_size * source_stride;
    }
    auto n_batches = raft::div_rounding_up_safe<int64_t>(n_rows, max_batch_size);
    for (int64_t batch_i = -1; batch_i < n_batches; batch_i++) {
      if (device_copy) {
        if (batch_i >= 0) {
          // Sync previous batch load
          raft::resource::sync_stream(res);
        }
        auto next_batch_i = batch_i + 1;
        if (next_batch_i < n_batches) {
          auto offset     = next_batch_i * max_batch_size;
          auto batch_size = std::min(max_batch_size, n_rows - offset);
          raft::copy(
            res,
            raft::make_host_vector_view(bufs[next_batch_i % 2], batch_size * source_stride),
            raft::make_device_vector_view(source_dataset + offset * source_stride,
                                          batch_size * source_stride));
        }
      }
      if (batch_i < 0) { continue; }
      const auto i0 = batch_i * max_batch_size;
      const auto i1 = std::min(i0 + max_batch_size, n_rows);
#pragma omp parallel for num_threads(num_threads)
      for (int64_t i = i0; i < i1; i++) {
        // clear the storage (TODO: it's not clear if this is necessary)
        memset(appr_algo->get_linklist0(i), 0, appr_algo->size_links_level0_);
        // copy the data section
        auto* source_ptr = device_copy ? bufs[batch_i % 2] + (i - i0) * source_stride
                                       : source_dataset + i * source_stride;
        memcpy(appr_algo->getDataByInternalId(i), source_ptr, appr_algo->data_size_);
        // As we build the index from scratch, we assign labels 1-1 in data order
        *appr_algo->getExternalLabeLp(i) = static_cast<hnswlib::labeltype>(i);
        int32_t curlevel                 = appr_algo->getRandomLevel(appr_algo->mult_);
        levels[i]                        = curlevel;
        if (curlevel) {
          appr_algo->linkLists_[i] =
            (char*)malloc(appr_algo->size_links_per_element_ * curlevel + 1);
          if (appr_algo->linkLists_[i] == nullptr)
            throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
          memset(appr_algo->linkLists_[i], 0, appr_algo->size_links_per_element_ * curlevel + 1);
        }
      }
    }
  }

  // sort the points by levels
  // build histogram
  std::vector<size_t> hist;
  std::vector<size_t> order(n_rows);
  for (int64_t i = 0; i < n_rows; i++) {
    auto pt_level = levels[i];
    while (pt_level >= static_cast<int32_t>(hist.size()))
      hist.push_back(0);
    hist[pt_level]++;
  }

  // accumulate
  std::vector<size_t> offsets(hist.size() + 1, 0);
  for (size_t i = 0; i < hist.size() - 1; i++) {
    offsets[i + 1] = offsets[i] + hist[i];
  }

  // bucket sort
  for (int64_t i = 0; i < n_rows; i++) {
    auto pt_level              = levels[i];
    order[offsets[pt_level]++] = i;
  }

  // set last point of the highest level as the entry point
  appr_algo->enterpoint_node_ = order.back();
  appr_algo->maxlevel_        = hist.size() - 1;

  // iterate over the points in the descending order of their levels
  for (size_t pt_level = hist.size() - 1; pt_level >= 1; pt_level--) {
    common::nvtx::range<common::nvtx::domain::cuvs> level_scope("level %zu", pt_level);
    auto start_idx     = offsets[pt_level - 1];
    auto end_idx       = offsets[hist.size() - 1];
    auto num_pts       = end_idx - start_idx;
    auto neighbor_size = num_pts > appr_algo->M_ ? appr_algo->M_ : num_pts - 1;
    if (num_pts <= 1) {
      // this means only 1 point in the level
      continue;
    }

    // gather points from dataset to form query set on host
    auto host_query_set = raft::make_host_matrix<T, int64_t>(num_pts, dim);
    // TODO: Use `raft::matrix::gather` when available as a public API
    // Issue: https://github.com/rapidsai/raft/issues/2572
#pragma omp parallel for num_threads(num_threads)
    for (auto i = start_idx; i < end_idx; i++) {
      auto pt_id = order[i];
      std::memcpy(
        &host_query_set(i - start_idx, 0), appr_algo->getDataByInternalId(pt_id), dim * sizeof(T));
    }

    // find neighbors of the query set
    auto host_neighbors = raft::make_host_matrix<uint32_t, int64_t>(num_pts, neighbor_size);
    all_neighbors_graph(res,
                        raft::make_const_mdspan(host_query_set.view()),
                        host_neighbors.view(),
                        cagra_index.metric());

    {
      common::nvtx::range<common::nvtx::domain::cuvs> copy_scope(
        "get_linklist(%zu, %zu)", start_idx, end_idx);
      // add points to the HNSW index upper layers
#pragma omp parallel for num_threads(num_threads)
      for (auto i = start_idx; i < end_idx; i++) {
        auto pt_id  = order[i];
        auto ll_cur = appr_algo->get_linklist(pt_id, pt_level);
        appr_algo->setListCount(ll_cur, host_neighbors.extent(1));
        auto* data     = (uint32_t*)(ll_cur + 1);
        auto neighbors = &host_neighbors(i - start_idx, 0);
        for (auto j = 0; j < host_neighbors.extent(1); j++) {
          data[j] = order[neighbors[j] + start_idx];
        }
      }
    }
  }

  auto graph_ptr = cagra_index.graph().data_handle();
  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, graph_ptr));
  bool is_host_accessible = false;
  int64_t degree          = cagra_index.graph().extent(1);
  if (attr.type == cudaMemoryTypeUnregistered) {
    is_host_accessible = true;
  } else if (attr.hostPointer != nullptr) {
    graph_ptr          = static_cast<uint32_t*>(attr.hostPointer);
    is_host_accessible = true;
  }

  // copy cagra graph to hnswlib base layer
  if (is_host_accessible) {
    common::nvtx::range<common::nvtx::domain::cuvs> copy_scope("get_linklist0<host>");
#pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < n_rows; i++) {
      auto ll_i = appr_algo->get_linklist0(i);
      appr_algo->setListCount(ll_i, degree);
      auto* data = (uint32_t*)(ll_i + 1);
      for (int64_t j = 0; j < degree; j++) {
        data[j] = graph_ptr[i * degree + j];
      }
    }
  } else {
    common::nvtx::range<common::nvtx::domain::cuvs> copy_scope("get_linklist0<device>");
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(appr_algo->get_linklist0(0) + 1,
                                    appr_algo->size_data_per_element_,
                                    graph_ptr,
                                    degree * sizeof(uint32_t),
                                    degree * sizeof(uint32_t),
                                    n_rows,
                                    cudaMemcpyDefault,
                                    raft::resource::get_cuda_stream(res)));
#pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < n_rows; i++) {
      appr_algo->setListCount(appr_algo->get_linklist0(i), degree);
    }
    raft::resource::sync_stream(res);
  }
  hnsw_index->set_index(std::move(appr_algo));
  return hnsw_index;
}

template <typename T>
size_t estimate_hnsw_host_memory(int64_t n_rows,
                                 int64_t dim,
                                 int graph_degree,
                                 cuvs::distance::DistanceType metric,
                                 HnswHierarchy hierarchy,
                                 int ef_construction)
{
  RAFT_EXPECTS(n_rows > 0 && dim > 0 && graph_degree > 0,
               "HNSW host-memory estimate requires a positive shape and graph degree");

  auto dummy_index = std::make_unique<index_impl<T>>(dim, metric, hierarchy);
  auto dummy_algo  = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
    dummy_index->get_space(), 1, (graph_degree + 1) / 2, ef_construction);

  size_t per_element = dummy_algo->size_data_per_element_;
  per_element += sizeof(void*) + sizeof(int) + sizeof(std::mutex);
  per_element += 56;  // unordered_map node + bucket slot (upper bound)
  if (hierarchy != HnswHierarchy::NONE) {
    int m_used = std::max(2, (graph_degree + 1) / 2);
    size_t size_links_per_element =
      static_cast<size_t>(m_used) * sizeof(uint32_t) + sizeof(uint32_t);
    per_element +=
      static_cast<size_t>(size_links_per_element / std::log(static_cast<double>(m_used)));
  }

  const size_t rows = static_cast<size_t>(n_rows);
  if (rows > std::numeric_limits<size_t>::max() / per_element) {
    return std::numeric_limits<size_t>::max();
  }
  return rows * per_element;
}

inline std::pair<size_t, size_t> get_available_memory(
  std::optional<double> max_host_memory_gb = std::nullopt,
  std::optional<double> max_gpu_memory_gb  = std::nullopt)
{
  size_t available_host_memory = cuvs::util::get_free_host_memory();
  if (max_host_memory_gb.has_value() && max_host_memory_gb.value() > 0) {
    const auto configured_host_memory =
      static_cast<size_t>(max_host_memory_gb.value() * (1ULL << 30));
    if (available_host_memory < configured_host_memory) {
      RAFT_LOG_WARN(
        "ACE: Actual host memory (%.2f GiB) is less than configured limit (%.2f GiB). Using "
        "actual host memory.",
        static_cast<double>(available_host_memory) / (1ULL << 30),
        max_host_memory_gb.value());
    } else {
      available_host_memory = configured_host_memory;
      RAFT_LOG_INFO("ACE: Using overridden host memory limit: %.2f GiB",
                    max_host_memory_gb.value());
    }
  }
  // Note: We use total device memory rather than free memory because RMM pools
  // and other allocators may report artificially low free memory. The assumption
  // is that the full device memory will be available for the build operation.
  size_t available_device_memory = rmm::available_device_memory().second;
  if (max_gpu_memory_gb.has_value() && max_gpu_memory_gb.value() > 0) {
    available_device_memory = static_cast<size_t>(max_gpu_memory_gb.value() * (1ULL << 30));
    RAFT_LOG_INFO("ACE: Using overridden GPU memory limit: %.2f GiB", max_gpu_memory_gb.value());
  }
  return std::make_pair(available_host_memory, available_device_memory);
}

template <typename T, typename CagraIndexT>
  requires is_cagra_hnsw_export_index_v<T, CagraIndexT>
std::unique_ptr<index<T>> from_cagra(
  raft::resources const& res,
  const index_params& params,
  CagraIndexT const& cagra_index,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  if constexpr (is_host_cagra_hnsw_export_index_v<T, CagraIndexT>) {
    if (!cagra_index.dataset_fd().has_value() && !dataset.has_value()) {
      RAFT_FAIL("hnsw::from_cagra requires dataset for host CAGRA index");
    }
  }

  // special treatment for index on disk
  if (cagra_index.dataset_fd().has_value() && cagra_index.graph_fd().has_value()) {
    // Get directory from graph file descriptor
    const auto& graph_fd = cagra_index.graph_fd();
    RAFT_EXPECTS(graph_fd.has_value() && graph_fd->is_valid(),
                 "Graph file descriptor is not available for disk-backed index");

    std::string graph_path = graph_fd->get_path();
    RAFT_EXPECTS(!graph_path.empty(), "Unable to get path from graph file descriptor");

    std::string index_directory = std::filesystem::path(graph_path).parent_path().string();
    RAFT_EXPECTS(
      std::filesystem::exists(index_directory) && std::filesystem::is_directory(index_directory),
      "Directory '%s' does not exist",
      index_directory.c_str());
    std::string index_filename =
      (std::filesystem::path(index_directory) / "hnsw_index.bin").string();
    exclusive_hnsw_output_file output(index_filename);

    serialize_to_hnswlib_from_disk(res, output.stream(), params, cagra_index);
    output.publish();

    // Create an empty HNSW index that holds the file descriptor
    auto hnsw_index =
      std::make_unique<index_impl<T>>(cagra_index.dim(), cagra_index.metric(), params.hierarchy);

    // Open file descriptor for the HNSW index file and transfer ownership to the index
    hnsw_index->set_file_descriptor(cuvs::util::file_descriptor(index_filename, O_RDONLY));

    RAFT_LOG_INFO("HNSW index written to disk at: %s", index_filename.c_str());

    return hnsw_index;
  }

  // In-memory CAGRA index: honor explicit ACE disk mode, or spill if the resulting HNSW index
  // would not fit in host memory. serialize_to_hnswlib_from_inmem avoids constructing the full
  // index in RAM (NONE/GPU only; the CPU hierarchy is not supported by the batched serializer).
  if (params.hierarchy != HnswHierarchy::CPU) {
    int64_t n_rows       = dataset.has_value() ? dataset->extent(0) : cagra_index.size();
    int64_t dim          = dataset.has_value() ? dataset->extent(1) : cagra_index.dim();
    int graph_degree_int = static_cast<int>(cagra_index.graph().extent(1));

    // Account for the contiguous level-0 storage plus hnswlib's per-element pointers, levels,
    // locks, label map, and expected upper-level links.
    size_t required_host = estimate_hnsw_host_memory<T>(n_rows,
                                                        dim,
                                                        graph_degree_int,
                                                        cagra_index.metric(),
                                                        params.hierarchy,
                                                        params.ef_construction);

    // Honor explicit disk mode and any host-memory limit from ACE params, mirroring hnsw::build.
    // The memory limit also makes the spill branch deterministically testable.
    const auto* ace_params =
      std::get_if<graph_build_params::ace_params>(&params.graph_build_params);
    const bool disk_requested                = ace_params != nullptr && ace_params->use_disk;
    std::optional<double> max_host_memory_gb = std::nullopt;
    if (ace_params != nullptr && ace_params->max_host_memory_gb > 0) {
      max_host_memory_gb = ace_params->max_host_memory_gb;
    }
    size_t available_host = get_available_memory(max_host_memory_gb).first;
    if (max_host_memory_gb.has_value()) {
      auto graph = cagra_index.graph();
      cudaPointerAttributes attributes;
      RAFT_CUDA_TRY(cudaPointerGetAttributes(&attributes, graph.data_handle()));
      const bool graph_uses_host_memory =
        attributes.type == cudaMemoryTypeUnregistered || attributes.hostPointer != nullptr;
      if (graph_uses_host_memory) {
        const size_t configured_host =
          static_cast<size_t>(max_host_memory_gb.value() * static_cast<double>(uint64_t{1} << 30));
        const size_t graph_bytes = graph.size() * sizeof(uint32_t);
        const size_t remaining_configured_host =
          graph_bytes < configured_host ? configured_host - graph_bytes : 0;
        available_host = std::min(available_host, remaining_configured_host);
      }
    }

    RAFT_LOG_INFO(
      "hnsw::from_cagra - in-memory HNSW requires ~%4.1f GB host mem, available %4.1f GB",
      required_host / 1e9,
      available_host / 1e9);

    if (disk_requested || required_host >= available_host) {
      if (disk_requested) {
        RAFT_LOG_INFO("ACE disk mode requested. Writing HNSW index to disk.");
      } else {
        RAFT_LOG_INFO("Not enough host memory for in-memory HNSW. Spilling HNSW index to disk.");
      }

      // Use the ACE build_dir if configured, otherwise fallback to system temp
      std::string index_directory;
      if (ace_params != nullptr && !ace_params->build_dir.empty()) {
        index_directory = ace_params->build_dir;
      }
      if (index_directory.empty()) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::filesystem::path candidate;
        do {
          candidate =
            std::filesystem::temp_directory_path() / ("cuvs_hnsw_" + std::to_string(gen()));
        } while (std::filesystem::exists(candidate));
        index_directory = candidate.string();
      }
      std::filesystem::create_directories(index_directory);
      RAFT_EXPECTS(
        std::filesystem::exists(index_directory) && std::filesystem::is_directory(index_directory),
        "Directory '%s' does not exist",
        index_directory.c_str());

      std::string index_filename =
        (std::filesystem::path(index_directory) / "hnsw_index.bin").string();
      exclusive_hnsw_output_file output(index_filename);

      serialize_to_hnswlib_from_inmem(res, output.stream(), params, cagra_index, dataset);
      output.publish();

      // Create an empty HNSW index that holds the file descriptor
      auto hnsw_index =
        std::make_unique<index_impl<T>>(dim, cagra_index.metric(), params.hierarchy);
      hnsw_index->set_file_descriptor(cuvs::util::file_descriptor(index_filename, O_RDONLY));

      RAFT_LOG_INFO("HNSW index written to disk at: %s", index_filename.c_str());

      return hnsw_index;
    }
  }

  if (params.hierarchy == HnswHierarchy::NONE) {
    return from_cagra<T, HnswHierarchy::NONE>(res, params, cagra_index, dataset);
  } else if (params.hierarchy == HnswHierarchy::CPU) {
    return from_cagra<T, HnswHierarchy::CPU>(res, params, cagra_index, dataset);
  } else if (params.hierarchy == HnswHierarchy::GPU) {
    return from_cagra<T, HnswHierarchy::GPU>(res, params, cagra_index, dataset);
  } else {
    RAFT_FAIL("Unsupported hierarchy type");
  }
}

template <typename T>
void extend(raft::resources const& res,
            const extend_params& params,
            raft::host_matrix_view<const T, int64_t, raft::row_major> additional_dataset,
            index<T>& idx)
{
  // If the index is disk-backed, load it into memory first
  auto* idx_impl = dynamic_cast<index_impl<T>*>(&idx);
  if (idx_impl) { idx_impl->ensure_loaded(); }

  auto* hnswlib_index = reinterpret_cast<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>*>(
    const_cast<void*>(idx.get_index()));
  auto current_element_count = hnswlib_index->getCurrentElementCount();
  auto new_element_count     = additional_dataset.extent(0);
  [[maybe_unused]] auto num_threads =
    params.num_threads == 0 ? cuvs::core::omp::get_max_threads() : params.num_threads;

  hnswlib_index->resizeIndex(current_element_count + new_element_count);
#pragma omp parallel for num_threads(num_threads)
  for (int64_t i = 0; i < additional_dataset.extent(0); i++) {
    hnswlib_index->addPoint(
      (void*)(additional_dataset.data_handle() + i * additional_dataset.extent(1)),
      current_element_count + i);
  }
}

template <typename T>
void get_search_knn_results(hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type> const* idx,
                            const T* query,
                            int k,
                            uint64_t* indices,
                            float* distances)
{
  auto result = idx->searchKnn(query, k);
  assert(result.size() >= static_cast<size_t>(k));

  for (int i = k - 1; i >= 0; --i) {
    indices[i]   = result.top().second;
    distances[i] = result.top().first;
    result.pop();
  }
}

template <typename T>
void search(raft::resources const& res,
            const search_params& params,
            const index<T>& idx,
            raft::host_matrix_view<const T, int64_t, raft::row_major> queries,
            raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances)
{
  // If the index is disk-backed, load it into memory first
  auto* idx_impl = dynamic_cast<const index_impl<T>*>(&idx);
  if (idx_impl) { idx_impl->ensure_loaded(); }

  RAFT_EXPECTS(queries.extent(0) == neighbors.extent(0) && queries.extent(0) == distances.extent(0),
               "Number of rows in output neighbors and distances matrices must equal the number of "
               "queries.");

  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "Number of columns in output neighbors and distances matrices must equal k");
  RAFT_EXPECTS(queries.extent(1) == idx.dim(),
               "Number of query dimensions should equal number of dimensions in the index.");

  idx.set_ef(params.ef);
  auto const* hnswlib_index =
    reinterpret_cast<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type> const*>(
      idx.get_index());

  // when num_threads == 0, automatically maximize parallelism
  if (params.num_threads) {
#pragma omp parallel for num_threads(params.num_threads)
    for (int64_t i = 0; i < queries.extent(0); ++i) {
      get_search_knn_results(hnswlib_index,
                             queries.data_handle() + i * queries.extent(1),
                             neighbors.extent(1),
                             neighbors.data_handle() + i * neighbors.extent(1),
                             distances.data_handle() + i * distances.extent(1));
    }
  } else {
#pragma omp parallel for
    for (int64_t i = 0; i < queries.extent(0); ++i) {
      get_search_knn_results(hnswlib_index,
                             queries.data_handle() + i * queries.extent(1),
                             neighbors.extent(1),
                             neighbors.data_handle() + i * neighbors.extent(1),
                             distances.data_handle() + i * distances.extent(1));
    }
  }
}

template <typename T>
void serialize(raft::resources const& res, const std::string& filename, const index<T>& idx)
{
  auto* idx_impl = dynamic_cast<const index_impl<T>*>(&idx);

  // Check if this is a disk-based index (created from disk-backed CAGRA)
  if (idx_impl && idx_impl->file_descriptor().has_value()) {
    // For disk-based indexes, copy the existing file to the new location
    std::string source_path = idx_impl->file_path();
    RAFT_EXPECTS(!source_path.empty(), "Disk-based index has invalid file path");
    RAFT_EXPECTS(std::filesystem::exists(source_path),
                 "Disk-based index file does not exist: %s",
                 source_path.c_str());

    // Copy the file to the new location
    std::filesystem::copy_file(
      source_path, filename, std::filesystem::copy_options::overwrite_existing);
    RAFT_LOG_INFO(
      "Copied disk-based HNSW index from %s to %s", source_path.c_str(), filename.c_str());
    return;
  }

  auto* hnswlib_index = reinterpret_cast<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>*>(
    const_cast<void*>(idx.get_index()));
  hnswlib_index->saveIndex(filename);
}

template <typename T>
void deserialize(raft::resources const& res,
                 const index_params& params,
                 const std::string& filename,
                 int dim,
                 cuvs::distance::DistanceType metric,
                 index<T>** idx)
{
  try {
    auto hnsw_index = std::make_unique<index_impl<T>>(dim, metric, params.hierarchy);
    auto appr_algo  = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
      hnsw_index->get_space(), filename);
    if (params.hierarchy == HnswHierarchy::NONE) { appr_algo->base_layer_only = true; }
    hnsw_index->set_index(std::move(appr_algo));
    *idx = hnsw_index.release();
  } catch (const std::bad_alloc& e) {
    RAFT_FAIL(
      "Failed to deserialize HNSW index from '%s': insufficient host memory. "
      "The index is too large to fit in available RAM. "
      "Consider using a machine with more memory or reducing the dataset size.",
      filename.c_str());
  }
}

/**
 * @brief Build an HNSW index on the GPU using CAGRA graph building algorithm
 *
 * This function builds an HNSW index
 * 1. Converting HNSW parameters to CAGRA parameters
 * 2. Inspect memory requirements (fall back to ACE algorithm if memory constrained)
 * 3. Building with ACE (direct partitioned HNSW) or in-memory CAGRA, then converting as needed
 */
template <typename T>
std::unique_ptr<index<T>> build(raft::resources const& res,
                                const index_params& params,
                                raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  common::nvtx::range<common::nvtx::domain::cuvs> fun_scope("hnsw::build<ACE>");
  RAFT_EXPECTS(params.M <= static_cast<size_t>(std::numeric_limits<int>::max() / 3),
               "HNSW M is too large for CAGRA graph-degree parameters");

  cuvs::neighbors::cagra::index_params cagra_params =
    cagra::index_params::from_hnsw_params(dataset.extents(),
                                          params.M,
                                          params.ef_construction,
                                          cagra::hnsw_heuristic_type::SAME_GRAPH_FOOTPRINT,
                                          params.metric);
  cagra_params.metric                  = params.metric;
  cagra_params.attach_dataset_on_build = false;

  const bool cagra_ace_explicitly_selected =
    std::holds_alternative<graph_build_params::ace_params>(params.graph_build_params);
  if (cagra_ace_explicitly_selected) {
    const auto& explicit_ace_params =
      std::get<graph_build_params::ace_params>(params.graph_build_params);
    RAFT_EXPECTS(explicit_ace_params.npartitions <= static_cast<size_t>(dataset.extent(0)),
                 "ACE: number of partitions cannot exceed dataset size");
  }
  bool cagra_ace_selected_for_memory = false;

  if (std::holds_alternative<std::monostate>(params.graph_build_params)) {
    auto [required_host, required_dev] = cuvs::neighbors::cagra::helpers::cagra_build_mem_usage(
      res, dataset.extents(), to_cuda_data_type<T>(), cagra_params);
    auto [available_host, available_dev] = get_available_memory();

    RAFT_LOG_INFO("CAGRA in memory build, required host mem %4.1f GB, GPU mem %4.1f GB",
                  required_host / 1e9,
                  required_dev / 1e9);
    RAFT_LOG_INFO("Available                       host mem %4.1f GB, GPU mem %4.1f GB",
                  available_host / 1e9,
                  available_dev / 1e9);
    if (required_host < available_host && required_dev < available_dev) {
      RAFT_LOG_INFO("We have sufficient memory to proceed with in memory build");
    } else {
      cagra_ace_selected_for_memory = true;
      RAFT_LOG_INFO("Not enough host or device memory. Falling back to ACE partitioned HNSW build");
    }
  }
  const bool cagra_ace_selected = cagra_ace_explicitly_selected || cagra_ace_selected_for_memory;
  // Partitioned ACE needs more rows than the CAGRA intermediate degree. Smaller ACE requests keep
  // the in-memory CAGRA conversion.
  const bool ace_partitioned_build_possible =
    dataset.extent(0) > static_cast<int64_t>(cagra_params.intermediate_graph_degree);

  if (cagra_ace_selected && ace_partitioned_build_possible) {
    RAFT_EXPECTS(params.hierarchy != HnswHierarchy::CPU,
                 "ACE HNSW construction does not support a CPU hierarchy");
    auto ace_params =
      std::holds_alternative<graph_build_params::ace_params>(params.graph_build_params)
        ? std::get<graph_build_params::ace_params>(params.graph_build_params)
        : graph_build_params::ace_params{};

    auto [available_host, available_device] = get_available_memory();
    if (ace_params.max_host_memory_gb > 0) {
      available_host = std::min(available_host,
                                static_cast<size_t>(ace_params.max_host_memory_gb *
                                                    static_cast<double>(uint64_t{1} << 30)));
    }
    if (ace_params.max_gpu_memory_gb > 0) {
      available_device = std::min(
        available_device,
        static_cast<size_t>(ace_params.max_gpu_memory_gb * static_cast<double>(uint64_t{1} << 30)));
    }
    cagra::detail::ace_external_plan_input external_input;
    external_input.rows                   = dataset.extent(0);
    external_input.dim                    = dataset.extent(1);
    external_input.element_size           = sizeof(T);
    external_input.M                      = params.M;
    external_input.intermediate_degree    = cagra_params.intermediate_graph_degree;
    external_input.graph_degree           = cagra_params.graph_degree;
    external_input.requested_partitions   = ace_params.npartitions;
    external_input.available_host_bytes   = available_host;
    external_input.available_device_bytes = available_device;
    uint64_t initial_partitions =
      std::min<uint64_t>(cagra::detail::external_maximum_partitions(
                           dataset.extent(0), cagra_params.intermediate_graph_degree),
                         std::max<uint64_t>(2, ace_params.npartitions));
    uint64_t initial_max_occurrences = cagra::detail::external_checked_mul(
      6,
      cagra::detail::external_div_rounding_up(dataset.extent(0), initial_partitions),
      "initial CAGRA-ACE partition occurrence count");
    auto [optimize_host, optimize_device, optimize_host_fixed, optimize_device_fixed] =
      cagra::helpers::optimize_workspace_size(initial_max_occurrences,
                                              cagra_params.graph_degree,
                                              cagra_params.intermediate_graph_degree,
                                              sizeof(uint32_t),
                                              cagra_params.guarantee_connectivity);
    external_input.optimize_host_fixed   = optimize_host_fixed;
    external_input.optimize_device_fixed = optimize_device_fixed;
    external_input.optimize_host_per_row = cagra::detail::external_div_rounding_up(
      optimize_host - optimize_host_fixed, initial_max_occurrences);
    external_input.optimize_device_per_row = cagra::detail::external_div_rounding_up(
      optimize_device - optimize_device_fixed, initial_max_occurrences);
    external_input.force_disk = true;
    external_input.hierarchy  = params.hierarchy == HnswHierarchy::GPU;
    auto external_plan        = cagra::detail::make_ace_external_plan(external_input);
    RAFT_LOG_INFO(
      "hnsw::build - using ACE partitioned HNSW build with %zu partitions, planned host/device "
      "peaks %.3f/%.3f GiB",
      static_cast<size_t>(external_plan.partitions),
      external_plan.host_peak_bytes / static_cast<double>(uint64_t{1} << 30),
      external_plan.device_peak_bytes / static_cast<double>(uint64_t{1} << 30));
    return external::build_external<T>(res,
                                       params,
                                       dataset,
                                       external_plan,
                                       ace_params,
                                       cagra_params.graph_degree,
                                       cagra_params.intermediate_graph_degree,
                                       params.ef_construction);
  }

  // Public HNSW API uses host_matrix_view; CAGRA build expects a padded dataset view.
  // Host build stores only the graph; vectors are passed separately to from_cagra below.
  cuvs::neighbors::host_padded_dataset_view<T, int64_t> host_padded_view(
    dataset, static_cast<uint32_t>(dataset.extent(1)));
  auto cagra_index = cuvs::neighbors::cagra::build(res, cagra_params, host_padded_view);

  RAFT_LOG_INFO("hnsw::build - Converting CAGRA index to HNSW format");
  return from_cagra<T>(
    res,
    params,
    cagra_index,
    cagra_index.dataset_fd().has_value() ? std::nullopt : std::make_optional(dataset));
}

}  // namespace cuvs::neighbors::hnsw::detail
