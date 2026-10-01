/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cuvs_cagra_wrapper.h"
#include <cuvs/neighbors/hnsw.hpp>
#include <raft/core/logger.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <variant>

#include "../common/ann_types.hpp"
#include "../common/blob.hpp"
#include "../common/conf.hpp"
#include "../diskann/diskann_wrapper.h"
#include "cuvs_ann_bench_utils.h"
#include <cuvs/neighbors/vamana.hpp>
#include <utils.h>

#include <memory>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan.hpp>

namespace cuvs::bench {

template <typename T, typename IdxT>
class cuvs_cagra_diskann : public algo<T>, public algo_gpu {
 public:
  using build_param       = typename cuvs_cagra<T, IdxT>::build_param;
  using search_param_base = typename algo<T>::search_param;
  using search_param      = typename diskann_memory<T>::search_param;

  cuvs_cagra_diskann(Metric metric, int dim, const build_param& param);

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const search_param_base& param, const void* filter_bitset) override;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return cagra_build_.get_sync_stream();
  }

  // to enable dataset access from GPU memory
  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHostMmap;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

  void save(const std::string& file) const override;
  void load(const std::string&) override;
  std::unique_ptr<algo<T>> copy() override
  {
    return std::make_unique<cuvs_cagra_diskann<T, IdxT>>(*this);
  }

 private:
  raft::resources handle_{};
  build_param build_param_;
  search_param search_param_;
  cuvs_cagra<T, IdxT> cagra_build_;
  std::shared_ptr<diskann_memory<T>> diskann_memory_search_;
};

template <typename T, typename IdxT>
cuvs_cagra_diskann<T, IdxT>::cuvs_cagra_diskann(Metric metric, int dim, const build_param& param)
  : algo<T>(metric, dim), build_param_{param}, cagra_build_{metric, dim, param, 1}
{
  diskann_memory_search_ = std::make_shared<cuvs::bench::diskann_memory<T>>(
    metric,
    dim,
    typename diskann_memory<T>::build_param{static_cast<uint32_t>(32), static_cast<uint32_t>(32)});
}

template <typename T, typename IdxT>
void cuvs_cagra_diskann<T, IdxT>::build(const T* dataset, size_t nrow)
{
  cagra_build_.build(dataset, nrow);
}

template <typename T, typename IdxT>
void cuvs_cagra_diskann<T, IdxT>::set_search_param(const search_param_base& param,
                                                   const void* filter_bitset)
{
  if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
  diskann_memory_search_->set_search_param(param, nullptr);
}

template <typename T, typename IdxT>
void cuvs_cagra_diskann<T, IdxT>::save(const std::string& file) const
{
  // Write graph to first index file (format from MSFT DiskANN OSS)
  std::ofstream index_of(file, std::ios::out | std::ios::binary);
  if (!index_of) { RAFT_FAIL("Cannot open file %s", file.c_str()); }

  size_t file_offset = 0;
  index_of.seekp(file_offset, index_of.beg);
  uint32_t max_degree = 0;
  size_t index_size   = 24;
  uint32_t start = static_cast<uint32_t>(rand() % (cagra_build_.get_index()->graph().extent(0)));
  size_t num_frozen_points     = 0;
  uint32_t max_observed_degree = 0;

  index_of.write((char*)&index_size, sizeof(uint64_t));
  index_of.write((char*)&max_observed_degree, sizeof(uint32_t));
  index_of.write((char*)&start, sizeof(uint32_t));
  index_of.write((char*)&num_frozen_points, sizeof(size_t));

  auto d_graph = cagra_build_.get_index()->graph();
  auto h_graph = raft::make_host_matrix<IdxT, int64_t>(d_graph.extent(0), d_graph.extent(1));
  raft::copy(h_graph.data_handle(),
             d_graph.data_handle(),
             d_graph.size(),
             raft::resource::get_cuda_stream(handle_));
  raft::resource::sync_stream(handle_);

  size_t total_edges = 0;
  size_t num_sparse  = 0;
  size_t num_single  = 0;

  for (uint32_t i = 0; i < h_graph.extent(0); i++) {
    uint32_t node_edges = 0;
    for (; node_edges < h_graph.extent(1); node_edges++) {
      if (h_graph(i, node_edges) == raft::upper_bound<IdxT>()) { break; }
    }

    if (node_edges < 3) num_sparse++;
    if (node_edges < 2) num_single++;
    total_edges += node_edges;

    index_of.write((char*)&node_edges, sizeof(uint32_t));
    if constexpr (!std::is_same_v<IdxT, uint32_t>) {
      RAFT_FAIL("serialization is only implemented for uint32_t graph");
    }
    index_of.write((char*)&h_graph(i, 0), node_edges * sizeof(uint32_t));

    max_degree = node_edges > max_degree ? (uint32_t)node_edges : max_degree;
    index_size += (size_t)(sizeof(uint32_t) * (node_edges + 1));
  }
  index_of.seekp(file_offset, index_of.beg);
  index_of.write((char*)&index_size, sizeof(uint64_t));
  index_of.write((char*)&max_degree, sizeof(uint32_t));

  RAFT_LOG_DEBUG(
    "Wrote file out, index size:%lu, max_degree:%u, num_sparse:%ld, num_single:%ld, total "
    "edges:%ld, avg degree:%f",
    index_size,
    max_degree,
    num_sparse,
    num_single,
    total_edges,
    (float)total_edges / (float)h_graph.extent(0));

  index_of.close();
  if (!index_of) { RAFT_FAIL("Error writing output %s", file.c_str()); }

  // Write the rows next to the graph; diskann::Index::load() reads them from `<file>.data`.
  // The benchmark base file is already in the same bin format, so copy it rather than pull the
  // rows out of memory - this way `save()` does not care where the dataset was allocated.
  const auto& ds_conf = configuration::singleton().get_dataset_conf();
  blob_file<T> base{ds_conf.base_file, ds_conf.subset_first_row, ds_conf.subset_size};
  int size = static_cast<int>(base.rows_limit());
  int dim  = static_cast<int>(base.n_cols());
  RAFT_EXPECTS(dim == this->dim_, "base_file dimensionality does not match the index");

  size_t header_bytes = 2 * sizeof(uint32_t);
  size_t skip_bytes   = sizeof(T) * static_cast<size_t>(base.rows_offset()) * dim;
  size_t copy_bytes   = sizeof(T) * static_cast<size_t>(size) * dim;
  RAFT_EXPECTS(std::filesystem::file_size(base.path()) >= header_bytes + skip_bytes + copy_bytes,
               "base_file is shorter than its header claims");

  std::ifstream base_in(base.path(), std::ios::in | std::ios::binary);
  if (!base_in) { RAFT_FAIL("Cannot open file %s", base.path().c_str()); }
  base_in.seekg(header_bytes + skip_bytes);

  std::string dataset_base_file = file + ".data";
  std::ofstream dataset_of(dataset_base_file, std::ios::out | std::ios::binary);
  if (!dataset_of) { RAFT_FAIL("Cannot open file %s", dataset_base_file.c_str()); }
  dataset_of.write((char*)&size, sizeof(int));
  dataset_of.write((char*)&dim, sizeof(int));
  std::copy_n(std::istreambuf_iterator<char>(base_in),
              copy_bytes,
              std::ostreambuf_iterator<char>(dataset_of));
  dataset_of.close();
  if (!base_in || !dataset_of) { RAFT_FAIL("Error writing output %s", dataset_base_file.c_str()); }
}

template <typename T, typename IdxT>
void cuvs_cagra_diskann<T, IdxT>::load(const std::string& file)
{
  diskann_memory_search_->load(file);
}

template <typename T, typename IdxT>
void cuvs_cagra_diskann<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  diskann_memory_search_->search(queries, batch_size, k, neighbors, distances);
}

}  // namespace cuvs::bench
