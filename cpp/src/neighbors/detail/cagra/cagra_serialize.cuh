/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/serialize.hpp>
#include <raft/util/cudart_utils.hpp>

#include "../../../core/nvtx.hpp"
#include "../../../util/serialize_validation.hpp"
#include "../dataset_serialize.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

constexpr int serialization_version = 5;

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] res the raft resource handle
 * @param[in] filename the file name for saving the index
 * @param[in] index_ CAGRA index
 *
 */
template <typename T, typename IdxT>
void serialize(raft::resources const& res,
               std::ostream& os,
               const index<T, IdxT>& index_,
               bool include_dataset)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("cagra::serialize");

  RAFT_EXPECTS(!index_.dataset_fd().has_value(),
               "Cannot serialize a disk-backed CAGRA index. Convert it with "
               "cuvs::neighbors::hnsw::from_cagra() and load it into memory via "
               "cuvs::neighbors::hnsw::deserialize() before serialization.");

  RAFT_LOG_DEBUG(
    "Saving CAGRA index, size %zu, dim %u", static_cast<size_t>(index_.size()), index_.dim());

  std::string dtype_string = raft::numpy_serializer::get_numpy_dtype<T>().to_string();
  dtype_string.resize(4);
  os << dtype_string;

  raft::serialize_scalar(res, os, serialization_version);
  raft::serialize_scalar(res, os, index_.size());
  raft::serialize_scalar(res, os, index_.dim());
  raft::serialize_scalar(res, os, index_.graph_degree());
  raft::serialize_scalar(res, os, index_.metric());

  raft::serialize_mdspan(res, os, index_.graph());

  include_dataset &= (index_.data().n_rows() > 0);
  bool has_source_indices = index_.source_indices().has_value();
  uint32_t content_map    = 0x1u * include_dataset + 0x2u * has_source_indices;

  raft::serialize_scalar(res, os, content_map);
  if (include_dataset) {
    RAFT_LOG_DEBUG("Saving CAGRA index with dataset");
    neighbors::detail::serialize(res, os, index_.data());
  } else {
    RAFT_LOG_DEBUG("Saving CAGRA index WITHOUT dataset");
  }

  if (has_source_indices) { raft::serialize_mdspan(res, os, index_.source_indices().value()); }
}

template <typename T, typename IdxT>
void serialize(raft::resources const& res,
               const std::string& filename,
               const index<T, IdxT>& index_,
               bool include_dataset)
{
  RAFT_EXPECTS(!index_.dataset_fd().has_value(),
               "Cannot serialize a disk-backed CAGRA index. Convert it with "
               "cuvs::neighbors::hnsw::from_cagra() and load it into memory via "
               "cuvs::neighbors::hnsw::deserialize() before serialization.");
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  detail::serialize(res, of, index_, include_dataset);

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}

template <typename T, typename IdxT>
void serialize_to_hnswlib(
  raft::resources const& res,
  std::ostream& os,
  const cuvs::neighbors::cagra::index<T, IdxT>& index_,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  // static_assert(std::is_same_v<IdxT, int> or std::is_same_v<IdxT, uint32_t>,
  //               "An hnswlib index can only be trained with int32 or uint32 IdxT");
  int dim = (dataset) ? dataset->extent(1) : index_.dim();
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("cagra::serialize");
  RAFT_LOG_DEBUG("Saving CAGRA index to hnswlib format, size %zu, dim %u",
                 static_cast<size_t>(index_.size()),
                 dim);

  // offset_level_0
  std::size_t offset_level_0 = 0;
  os.write(reinterpret_cast<char*>(&offset_level_0), sizeof(std::size_t));
  // max_element
  std::size_t max_element = index_.size();
  os.write(reinterpret_cast<char*>(&max_element), sizeof(std::size_t));
  // curr_element_count
  std::size_t curr_element_count = index_.size();
  os.write(reinterpret_cast<char*>(&curr_element_count), sizeof(std::size_t));
  // Example:M: 16, dim = 128, data_t = float, index_t = uint32_t, list_size_type = uint32_t,
  // labeltype: size_t size_data_per_element_ = M * 2 * sizeof(index_t) + sizeof(list_size_type) +
  // dim * sizeof(T) + sizeof(labeltype)
  auto size_data_per_element =
    static_cast<std::size_t>(index_.graph_degree() * sizeof(IdxT) + 4 + dim * sizeof(T) + 8);
  os.write(reinterpret_cast<char*>(&size_data_per_element), sizeof(std::size_t));
  // label_offset
  std::size_t label_offset = size_data_per_element - 8;
  os.write(reinterpret_cast<char*>(&label_offset), sizeof(std::size_t));
  // offset_data
  auto offset_data = static_cast<std::size_t>(index_.graph_degree() * sizeof(IdxT) + 4);
  os.write(reinterpret_cast<char*>(&offset_data), sizeof(std::size_t));
  // max_level
  int max_level = 1;
  os.write(reinterpret_cast<char*>(&max_level), sizeof(int));
  // entrypoint_node
  auto entrypoint_node = static_cast<int>(index_.size() / 2);
  os.write(reinterpret_cast<char*>(&entrypoint_node), sizeof(int));
  // max_M
  auto max_M = static_cast<std::size_t>(index_.graph_degree() / 2);
  os.write(reinterpret_cast<char*>(&max_M), sizeof(std::size_t));
  // max_M0
  std::size_t max_M0 = index_.graph_degree();
  os.write(reinterpret_cast<char*>(&max_M0), sizeof(std::size_t));
  // M
  auto M = static_cast<std::size_t>(index_.graph_degree() / 2);
  os.write(reinterpret_cast<char*>(&M), sizeof(std::size_t));
  // mult, can be anything
  double mult = 0.42424242;
  os.write(reinterpret_cast<char*>(&mult), sizeof(double));
  // efConstruction, can be anything
  std::size_t efConstruction = 500;
  os.write(reinterpret_cast<char*>(&efConstruction), sizeof(std::size_t));

  // Remove padding before saving the dataset
  raft::host_matrix<T, int64_t> host_dataset = raft::make_host_matrix<T, int64_t>(0, 0);
  raft::host_matrix_view<const T, int64_t> host_dataset_view;
  if (dataset) {
    host_dataset_view = *dataset;
  } else {
    auto dataset = index_.dataset();
    RAFT_EXPECTS(dataset.size() > 0,
                 "Invalid CAGRA dataset of size 0 during serialization, shape %zux%zu",
                 static_cast<size_t>(dataset.extent(0)),
                 static_cast<size_t>(dataset.extent(1)));
    host_dataset = raft::make_host_matrix<T, int64_t>(dataset.extent(0), dataset.extent(1));
    raft::copy_matrix(host_dataset.data_handle(),
                      host_dataset.extent(1),
                      dataset.data_handle(),
                      dataset.stride(0),
                      host_dataset.extent(1),
                      dataset.extent(0),
                      raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    host_dataset_view = raft::make_const_mdspan(host_dataset.view());
  }
  auto graph = index_.graph();
  auto host_graph =
    raft::make_host_matrix<IdxT, int64_t, raft::row_major>(graph.extent(0), graph.extent(1));
  raft::copy(res, host_graph.view(), graph);
  raft::resource::sync_stream(res);

  size_t d_report_offset    = index_.size() / 10;  // Report progress in 10% steps.
  size_t next_report_offset = d_report_offset;
  const auto start_clock    = std::chrono::system_clock::now();
  // Write one dataset and graph row at a time
  RAFT_EXPECTS(host_graph.stride(1) == 1, "serialize_to_hnswlib expects row_major graph");
  RAFT_EXPECTS(host_dataset_view.stride(1) == 1, "serialize_to_hnswlib expects row_major dataset");

  size_t bytes_written = 0;
  float GiB            = 1 << 30;
  for (std::size_t i = 0; i < index_.size(); i++) {
    auto graph_degree = static_cast<int>(index_.graph_degree());
    os.write(reinterpret_cast<char*>(&graph_degree), sizeof(int));

    IdxT* graph_row = &host_graph(i, 0);
    os.write(reinterpret_cast<char*>(graph_row), sizeof(IdxT) * index_.graph_degree());

    const T* data_row = &host_dataset_view(i, 0);
    os.write(reinterpret_cast<const char*>(data_row), sizeof(T) * dim);
    os.write(reinterpret_cast<char*>(&i), sizeof(std::size_t));

    bytes_written +=
      dim * sizeof(T) + index_.graph_degree() * sizeof(IdxT) + sizeof(int) + sizeof(size_t);
    const auto end_clock = std::chrono::system_clock::now();
    if (!os.good()) { RAFT_FAIL("Error writing HNSW file, row %zu", i); }
    if (i > next_report_offset) {
      next_report_offset += d_report_offset;
      const auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() *
        1e-6;
      float throughput      = bytes_written / GiB / time;
      float rows_throughput = i / time;
      float ETA             = (index_.size() - i) / rows_throughput;
      RAFT_LOG_DEBUG(
        "# Writing rows %12lu / %12lu (%3.2f %%), %3.2f GiB/sec, ETA %d:%3.1f, written %3.2f GiB\r",
        i,
        index_.size(),
        i / static_cast<double>(index_.size()) * 100,
        throughput,
        int(ETA / 60),
        std::fmod(ETA, 60.0f),
        bytes_written / GiB);
    }
  }

  for (std::size_t i = 0; i < index_.size(); i++) {
    // zeroes
    auto zero = 0;
    os.write(reinterpret_cast<char*>(&zero), sizeof(int));
  }
}

template <typename T, typename IdxT>
void serialize_to_hnswlib(
  raft::resources const& res,
  const std::string& filename,
  const cuvs::neighbors::cagra::index<T, IdxT>& index_,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  detail::serialize_to_hnswlib<T, IdxT>(res, of, index_, dataset);

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}

/** Load an index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] res the raft resource handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index_ CAGRA index
 *
 */
template <typename T, typename IdxT>
void deserialize(raft::resources const& res, std::istream& is, index<T, IdxT>* index_)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("cagra::deserialize");

  char dtype_string[4];
  RAFT_EXPECTS(is.read(dtype_string, 4), "cagra::deserialize: failed to read dtype prefix");
  RAFT_EXPECTS(cuvs::util::validate_serialized_dtype<T>(dtype_string, sizeof(dtype_string)),
               "cagra::deserialize: serialized dtype prefix does not match requested type");

  auto ver = raft::deserialize_scalar<int>(res, is);
  if (ver != serialization_version) {
    RAFT_FAIL("serialization version mismatch, expected %d, got %d ", serialization_version, ver);
  }
  auto n_rows       = raft::deserialize_scalar<IdxT>(res, is);
  auto dim          = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto graph_degree = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto metric       = raft::deserialize_scalar<cuvs::distance::DistanceType>(res, is);

  RAFT_EXPECTS(cuvs::util::is_valid_distance_type(metric),
               "cagra::deserialize: invalid metric value %d",
               static_cast<int>(metric));
  RAFT_EXPECTS(graph_degree <= cuvs::util::kMaxGraphDegree,
               "cagra::deserialize: graph_degree=%u exceeds maximum %u",
               graph_degree,
               cuvs::util::kMaxGraphDegree);
  RAFT_EXPECTS(
    cuvs::util::is_mul_no_overflow(
      static_cast<std::size_t>(n_rows), static_cast<std::size_t>(graph_degree), sizeof(IdxT)),
    "cagra::deserialize: integer overflow in n_rows*graph_degree*sizeof(IdxT) "
    "(n_rows=%lld, graph_degree=%u, sizeof(IdxT)=%zu)",
    static_cast<long long>(n_rows),
    graph_degree,
    sizeof(IdxT));

  auto graph = raft::make_host_matrix<IdxT, int64_t>(n_rows, graph_degree);
  deserialize_mdspan(res, is, graph.view());

  *index_ = index<T, IdxT>(res, metric);
  index_->update_graph(res, raft::make_const_mdspan(graph.view()));

  auto content_map = raft::deserialize_scalar<uint32_t>(res, is);
  bool has_dataset = content_map & 0x1u;
  if (has_dataset) {
    index_->update_dataset(res, cuvs::neighbors::detail::deserialize_dataset<int64_t>(res, is));
  }

  bool has_source_indices = content_map & 0x2u;
  if (has_source_indices) {
    auto source_indices = raft::make_host_vector<IdxT, int64_t>(n_rows);
    deserialize_mdspan(res, is, source_indices.view());
    index_->update_source_indices(res, raft::make_const_mdspan(source_indices.view()));
    raft::resource::sync_stream(
      res);  // Don't let the vector out of the scope before the copy is finished
  }
}

template <typename T, typename IdxT>
void deserialize(raft::resources const& res, const std::string& filename, index<T, IdxT>* index_)
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);

  if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  detail::deserialize<T, IdxT>(res, is, index_);

  is.close();
}
}  // namespace cuvs::neighbors::cagra::detail
