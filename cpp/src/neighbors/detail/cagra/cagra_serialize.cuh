/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/util/file_io.hpp>
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
#include "../../../util/kvikio_serialize.hpp"
#include "../../../util/serialize_validation.hpp"
#include "../dataset_serialize.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>

namespace cuvs::neighbors::cagra::detail {

constexpr int serialization_version = 5;

template <typename MdspanT>
void serialize_index_mdspan(raft::resources const& res, std::ostream& os, const MdspanT& mdspan)
{
  cuvs::util::detail::serialize_mdspan(res, os, mdspan);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void serialize_index_mdspan(
  raft::resources const& res,
  cuvs::util::kvikio_ofstream& os,
  const raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& mdspan)
{
  cuvs::util::detail::serialize_device_mdspan(res, os, mdspan);
}

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
template <typename OutputStream, typename T, typename IdxT>
void serialize_impl(raft::resources const& res,
                    OutputStream& os,
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

  serialize_index_mdspan(res, os, index_.graph());

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

  if (has_source_indices) { serialize_index_mdspan(res, os, index_.source_indices().value()); }
}

template <typename T, typename IdxT>
void serialize(raft::resources const& res,
               std::ostream& os,
               const index<T, IdxT>& index_,
               bool include_dataset)
{
  serialize_impl(res, os, index_, include_dataset);
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
  cuvs::util::kvikio_ofstream of(filename);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  serialize_impl(res, of, index_, include_dataset);

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}

template <typename T, typename IdxT>
void write_hnswlib_header(std::ostream& os,
                          const cuvs::neighbors::cagra::index<T, IdxT>& index_,
                          int dim)
{
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
}

inline void log_hnswlib_progress(size_t completed_rows,
                                 size_t total_rows,
                                 size_t bytes_written,
                                 const std::chrono::system_clock::time_point& start_clock,
                                 size_t& next_report_offset)
{
  if (completed_rows < next_report_offset || completed_rows == 0) { return; }

  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::system_clock::now() - start_clock)
                         .count() *
                       1e-6;
  if (elapsed <= 0) { return; }

  constexpr double gib         = double(size_t{1} << 30);
  const double throughput      = bytes_written / gib / elapsed;
  const double rows_throughput = completed_rows / elapsed;
  const double eta             = (total_rows - completed_rows) / rows_throughput;
  RAFT_LOG_DEBUG(
    "# Writing rows %12zu / %12zu (%3.2f %%), %3.2f GiB/sec, ETA %d:%3.1f, written %3.2f GiB\r",
    completed_rows,
    total_rows,
    completed_rows / static_cast<double>(total_rows) * 100,
    throughput,
    int(eta / 60),
    std::fmod(eta, 60.0),
    bytes_written / gib);

  const size_t report_interval = std::max<size_t>(1, total_rows / 10);
  next_report_offset += report_interval;
}

template <typename T, typename IdxT>
void write_hnswlib_rows_host(
  raft::resources const& res,
  std::ostream& os,
  const cuvs::neighbors::cagra::index<T, IdxT>& index_,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  const size_t n_rows        = static_cast<size_t>(index_.size());
  const int64_t dim          = dataset ? dataset->extent(1) : index_.dim();
  const int64_t graph_degree = index_.graph_degree();
  const size_t row_size =
    sizeof(int) + graph_degree * sizeof(IdxT) + dim * sizeof(T) + sizeof(size_t);
  if (n_rows == 0) { return; }

  const size_t batch_rows = std::min<size_t>(
    n_rows, std::max<size_t>(1, cuvs::util::detail::kDeviceSerializationBatchBytes / row_size));
  auto graph_buffer =
    raft::make_host_matrix<IdxT, int64_t, raft::row_major>(batch_rows, graph_degree);
  auto dataset_buffer = raft::make_host_matrix<T, int64_t, raft::row_major>(batch_rows, dim);

  const auto graph = index_.graph();
  RAFT_EXPECTS(static_cast<size_t>(graph.extent(0)) == n_rows,
               "CAGRA graph rows (%zu) do not match index size (%zu)",
               static_cast<size_t>(graph.extent(0)),
               n_rows);

  const T* device_dataset = nullptr;
  int64_t dataset_stride  = 0;
  if (dataset) {
    RAFT_EXPECTS(static_cast<size_t>(dataset->extent(0)) == n_rows,
                 "CAGRA dataset rows (%zu) do not match index size (%zu)",
                 static_cast<size_t>(dataset->extent(0)),
                 n_rows);
  } else {
    const auto dataset_view = index_.dataset();
    RAFT_EXPECTS(dataset_view.size() > 0,
                 "Invalid CAGRA dataset of size 0 during serialization, shape %zux%zu",
                 static_cast<size_t>(dataset_view.extent(0)),
                 static_cast<size_t>(dataset_view.extent(1)));
    RAFT_EXPECTS(static_cast<size_t>(dataset_view.extent(0)) == n_rows,
                 "CAGRA dataset rows (%zu) do not match index size (%zu)",
                 static_cast<size_t>(dataset_view.extent(0)),
                 n_rows);
    device_dataset = dataset_view.data_handle();
    dataset_stride = dataset_view.stride(0);
  }

  const auto stream           = raft::resource::get_cuda_stream(res);
  size_t bytes_written        = 0;
  size_t next_report          = std::max<size_t>(1, n_rows / 10);
  const auto start_clock      = std::chrono::system_clock::now();
  const auto graph_degree_int = static_cast<int>(graph_degree);

  for (size_t first_row = 0; first_row < n_rows; first_row += batch_rows) {
    const size_t rows = std::min(batch_rows, n_rows - first_row);
    raft::copy_matrix(graph_buffer.data_handle(),
                      graph_degree,
                      graph.data_handle() + first_row * graph_degree,
                      graph_degree,
                      graph_degree,
                      rows,
                      stream);
    if (!dataset) {
      raft::copy_matrix(dataset_buffer.data_handle(),
                        dim,
                        device_dataset + first_row * dataset_stride,
                        dataset_stride,
                        dim,
                        rows,
                        stream);
    }
    raft::resource::sync_stream(res);

    for (size_t batch_row = 0; batch_row < rows; ++batch_row) {
      const size_t row = first_row + batch_row;
      os.write(reinterpret_cast<const char*>(&graph_degree_int), sizeof(int));
      os.write(reinterpret_cast<const char*>(&graph_buffer(batch_row, 0)),
               graph_degree * sizeof(IdxT));

      const T* data_row =
        dataset ? dataset->data_handle() + row * dataset->stride(0) : &dataset_buffer(batch_row, 0);
      os.write(reinterpret_cast<const char*>(data_row), dim * sizeof(T));
      os.write(reinterpret_cast<const char*>(&row), sizeof(size_t));
    }

    RAFT_EXPECTS(os.good(), "Error writing HNSW file at row %zu", first_row + rows - 1);
    bytes_written += rows * row_size;
    log_hnswlib_progress(first_row + rows, n_rows, bytes_written, start_clock, next_report);
  }
}

template <typename ValueT>
__device__ void write_unaligned_value(uint8_t* output, ValueT value)
{
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
#pragma unroll
  for (size_t i = 0; i < sizeof(ValueT); ++i) {
    output[i] = bytes[i];
  }
}

template <typename T, typename IdxT>
__global__ void pack_hnswlib_rows(uint8_t* output,
                                  size_t row_size,
                                  const uint32_t* graph,
                                  const T* dataset,
                                  size_t first_row,
                                  size_t rows,
                                  uint32_t graph_degree,
                                  uint32_t dim,
                                  int64_t dataset_stride)
{
  const size_t warps_per_block = blockDim.x / warpSize;
  const size_t warp            = threadIdx.x / warpSize;
  const size_t lane            = threadIdx.x % warpSize;
  const size_t batch_row       = blockIdx.x * warps_per_block + warp;
  if (batch_row >= rows) { return; }

  const size_t row = first_row + batch_row;
  auto* row_output = output + batch_row * row_size;
  if (lane == 0) {
    write_unaligned_value(row_output, static_cast<int>(graph_degree));
    const size_t label_offset = sizeof(int) + graph_degree * sizeof(IdxT) + dim * sizeof(T);
    write_unaligned_value(row_output + label_offset, row);
  }

  const size_t graph_offset = sizeof(int);
  for (size_t col = lane; col < graph_degree; col += warpSize) {
    write_unaligned_value(row_output + graph_offset + col * sizeof(IdxT),
                          static_cast<IdxT>(graph[row * static_cast<size_t>(graph_degree) + col]));
  }

  const size_t dataset_offset = graph_offset + graph_degree * sizeof(IdxT);
  for (size_t col = lane; col < dim; col += warpSize) {
    write_unaligned_value(row_output + dataset_offset + col * sizeof(T),
                          dataset[row * static_cast<size_t>(dataset_stride) + col]);
  }
}

template <typename T, typename IdxT>
void write_hnswlib_rows_device(raft::resources const& res,
                               cuvs::util::kvikio_ofstream& os,
                               const cuvs::neighbors::cagra::index<T, IdxT>& index_)
{
  const size_t n_rows         = static_cast<size_t>(index_.size());
  const uint32_t dim          = index_.dim();
  const uint32_t graph_degree = index_.graph_degree();
  const size_t row_size =
    sizeof(int) + graph_degree * sizeof(IdxT) + dim * sizeof(T) + sizeof(size_t);
  if (n_rows == 0) { return; }

  const auto graph   = index_.graph();
  const auto dataset = index_.dataset();
  RAFT_EXPECTS(dataset.size() > 0,
               "Invalid CAGRA dataset of size 0 during serialization, shape %zux%zu",
               static_cast<size_t>(dataset.extent(0)),
               static_cast<size_t>(dataset.extent(1)));
  RAFT_EXPECTS(static_cast<size_t>(graph.extent(0)) == n_rows &&
                 static_cast<size_t>(dataset.extent(0)) == n_rows,
               "CAGRA graph and dataset rows must match index size");

  const size_t batch_rows = std::min<size_t>(
    n_rows, std::max<size_t>(1, cuvs::util::detail::kDeviceSerializationBatchBytes / row_size));
  auto output       = raft::make_device_vector<uint8_t, int64_t>(res, batch_rows * row_size);
  const auto stream = raft::resource::get_cuda_stream(res);

  size_t bytes_written          = 0;
  size_t next_report            = std::max<size_t>(1, n_rows / 10);
  const auto start_clock        = std::chrono::system_clock::now();
  constexpr int block_size      = 256;
  constexpr int warps_per_block = block_size / 32;

  for (size_t first_row = 0; first_row < n_rows; first_row += batch_rows) {
    const size_t rows   = std::min(batch_rows, n_rows - first_row);
    const size_t blocks = (rows + warps_per_block - 1) / warps_per_block;
    pack_hnswlib_rows<T, IdxT>
      <<<static_cast<unsigned int>(blocks), block_size, 0, stream>>>(output.data_handle(),
                                                                     row_size,
                                                                     graph.data_handle(),
                                                                     dataset.data_handle(),
                                                                     first_row,
                                                                     rows,
                                                                     graph_degree,
                                                                     dim,
                                                                     dataset.stride(0));
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::resource::sync_stream(res);

    const size_t bytes = rows * row_size;
    os.write_device(output.data_handle(), bytes);
    bytes_written += bytes;
    log_hnswlib_progress(first_row + rows, n_rows, bytes_written, start_clock, next_report);
  }
}

inline void write_hnswlib_empty_levels(std::ostream& os, size_t n_rows)
{
  constexpr size_t chunk_size = 16 * 1024;
  const std::array<int, chunk_size> zeros{};
  for (size_t first_row = 0; first_row < n_rows; first_row += chunk_size) {
    const size_t rows = std::min(chunk_size, n_rows - first_row);
    os.write(reinterpret_cast<const char*>(zeros.data()), rows * sizeof(int));
  }
}

template <typename T, typename IdxT>
void serialize_to_hnswlib(
  raft::resources const& res,
  std::ostream& os,
  const cuvs::neighbors::cagra::index<T, IdxT>& index_,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  const int dim = dataset ? dataset->extent(1) : index_.dim();
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("cagra::serialize");
  RAFT_LOG_DEBUG("Saving CAGRA index to hnswlib format, size %zu, dim %d",
                 static_cast<size_t>(index_.size()),
                 dim);

  write_hnswlib_header(os, index_, dim);
  write_hnswlib_rows_host(res, os, index_, dataset);
  write_hnswlib_empty_levels(os, static_cast<size_t>(index_.size()));
}

template <typename T, typename IdxT>
void serialize_to_hnswlib(
  raft::resources const& res,
  const std::string& filename,
  const cuvs::neighbors::cagra::index<T, IdxT>& index_,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  const int dim = dataset ? dataset->extent(1) : index_.dim();
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("cagra::serialize");
  RAFT_LOG_DEBUG("Saving CAGRA index to hnswlib format, size %zu, dim %d",
                 static_cast<size_t>(index_.size()),
                 dim);

  cuvs::util::kvikio_ofstream of(filename);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  write_hnswlib_header(of, index_, dim);
  if (dataset) {
    write_hnswlib_rows_host(res, of, index_, dataset);
  } else {
    write_hnswlib_rows_device(res, of, index_);
  }
  write_hnswlib_empty_levels(of, static_cast<size_t>(index_.size()));

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
void deserialize_graph(raft::resources const& res,
                       std::istream& is,
                       IdxT n_rows,
                       uint32_t graph_degree,
                       index<T, IdxT>* index_)
{
  using graph_index_type = typename index<T, IdxT>::graph_index_type;
  auto graph             = raft::make_host_matrix<graph_index_type, int64_t>(n_rows, graph_degree);
  deserialize_mdspan(res, is, graph.view());
  index_->update_graph(res, raft::make_const_mdspan(graph.view()));
}

template <typename T, typename IdxT>
void deserialize_graph(raft::resources const& res,
                       cuvs::util::kvikio_file_reader& reader,
                       IdxT n_rows,
                       uint32_t graph_degree,
                       index<T, IdxT>* index_)
{
  using graph_index_type = typename index<T, IdxT>::graph_index_type;
  auto graph = raft::make_device_matrix<graph_index_type, int64_t>(res, n_rows, graph_degree);
  cuvs::util::detail::deserialize_device_mdspan(res, reader, graph.view());
  index_->update_graph(std::move(graph));
}

template <typename T, typename IdxT>
void deserialize_source_indices(raft::resources const& res,
                                std::istream& is,
                                IdxT n_rows,
                                index<T, IdxT>* index_)
{
  auto source_indices = raft::make_host_vector<IdxT, int64_t>(n_rows);
  deserialize_mdspan(res, is, source_indices.view());
  index_->update_source_indices(res, raft::make_const_mdspan(source_indices.view()));
  raft::resource::sync_stream(res);
}

template <typename T, typename IdxT>
void deserialize_source_indices(raft::resources const& res,
                                cuvs::util::kvikio_file_reader& reader,
                                IdxT n_rows,
                                index<T, IdxT>* index_)
{
  auto source_indices = raft::make_device_vector<IdxT, int64_t>(res, n_rows);
  cuvs::util::detail::deserialize_device_mdspan(res, reader, source_indices.view());
  index_->update_source_indices(std::move(source_indices));
}

template <typename T, typename IdxT, typename Input>
void deserialize_impl(raft::resources const& res, Input& input, index<T, IdxT>* index_)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("cagra::deserialize");

  auto& is = cuvs::util::detail::input_stream(input);

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
  RAFT_EXPECTS(cuvs::util::is_mul_no_overflow(static_cast<std::size_t>(n_rows),
                                              static_cast<std::size_t>(graph_degree),
                                              sizeof(typename index<T, IdxT>::graph_index_type)),
               "cagra::deserialize: integer overflow in graph allocation "
               "(n_rows=%lld, graph_degree=%u, element_size=%zu)",
               static_cast<long long>(n_rows),
               graph_degree,
               sizeof(typename index<T, IdxT>::graph_index_type));

  *index_ = index<T, IdxT>(res, metric);
  deserialize_graph(res, input, n_rows, graph_degree, index_);

  auto content_map = raft::deserialize_scalar<uint32_t>(res, is);
  bool has_dataset = content_map & 0x1u;
  if (has_dataset) {
    index_->update_dataset(res, cuvs::neighbors::detail::deserialize_dataset<int64_t>(res, input));
  }

  bool has_source_indices = content_map & 0x2u;
  if (has_source_indices) { deserialize_source_indices(res, input, n_rows, index_); }
}

template <typename T, typename IdxT>
void deserialize(raft::resources const& res, std::istream& is, index<T, IdxT>* index_)
{
  deserialize_impl(res, is, index_);
}

template <typename T, typename IdxT>
void deserialize(raft::resources const& res, const std::string& filename, index<T, IdxT>* index_)
{
  cuvs::util::kvikio_file_reader reader(filename);
  deserialize_impl(res, reader, index_);
}
}  // namespace cuvs::neighbors::cagra::detail
