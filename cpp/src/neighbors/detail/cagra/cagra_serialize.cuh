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
#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <typename T, typename IdxT, typename CagraIndexT>
inline constexpr bool is_cagra_hnsw_serialize_index_v =
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::device_padded_index<T, IdxT>> ||
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::device_standard_index<T, IdxT>> ||
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::host_padded_index<T, IdxT>> ||
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::host_standard_index<T, IdxT>>;

template <typename T, typename IdxT, typename CagraIndexT>
inline constexpr bool is_device_cagra_hnsw_serialize_index_v =
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::device_padded_index<T, IdxT>> ||
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::device_standard_index<T, IdxT>>;

template <typename T, typename IdxT, typename CagraIndexT>
inline constexpr bool is_host_cagra_hnsw_serialize_index_v =
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::host_padded_index<T, IdxT>> ||
  std::is_same_v<CagraIndexT, cuvs::neighbors::cagra::host_standard_index<T, IdxT>>;

constexpr int serialization_version = cuvs::neighbors::cagra::cagra_serialization_version;

template <cuvs::neighbors::ann_dataset_view DatasetViewT>
constexpr auto serialized_dataset_kind_for_view() -> cuvs::neighbors::cagra::serialized_dataset_kind
{
  using kind = cuvs::neighbors::cagra::serialized_dataset_kind;
  if constexpr (cuvs::neighbors::is_device_padded_dataset_view_v<DatasetViewT>) {
    return kind::device_padded;
  } else if constexpr (cuvs::neighbors::is_device_standard_dataset_view_v<DatasetViewT>) {
    return kind::device_standard;
  } else if constexpr (cuvs::neighbors::is_host_padded_dataset_view_v<DatasetViewT>) {
    return kind::host_padded;
  } else if constexpr (cuvs::neighbors::is_host_standard_dataset_view_v<DatasetViewT>) {
    return kind::host_standard;
  } else {
    static_assert(sizeof(DatasetViewT) == 0,
                  "serialized_dataset_kind_for_view: unsupported dataset view type");
  }
}

constexpr bool is_valid_serialized_dataset_kind(std::uint32_t raw)
{
  using kind = cuvs::neighbors::cagra::serialized_dataset_kind;
  return raw <= static_cast<std::uint32_t>(kind::host_standard);
}

template <typename MdspanT>
void serialize_index_mdspan(raft::resources const& res, std::ostream& os, MdspanT const& mdspan)
{
  cuvs::util::detail::serialize_mdspan(res, os, mdspan);
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
template <typename T, typename IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
void serialize(raft::resources const& res,
               std::ostream& os,
               const cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>& index_,
               bool include_dataset)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("cagra::serialize");

  RAFT_EXPECTS(!index_.dataset_fd().has_value(),
               "Cannot serialize a disk-backed CAGRA index. Convert it with "
               "cuvs::neighbors::hnsw::from_cagra() and load it into memory via "
               "cuvs::neighbors::hnsw::deserialize() before serialization.");

  RAFT_LOG_DEBUG(
    "Saving CAGRA index, size %zu, dim %u", static_cast<size_t>(index_.size()), index_.dim());

  include_dataset &= (index_.dataset().n_rows() > 0);
  auto const dataset_kind = include_dataset ? serialized_dataset_kind_for_view<DatasetViewT>()
                                            : cuvs::neighbors::cagra::serialized_dataset_kind::none;

  std::string dtype_string = raft::numpy_serializer::get_numpy_dtype<T>().to_string();
  dtype_string.resize(4);
  os << dtype_string;

  raft::serialize_scalar(res, os, serialization_version);
  raft::serialize_scalar(res, os, static_cast<std::uint32_t>(dataset_kind));
  raft::serialize_scalar(res, os, index_.size());
  raft::serialize_scalar(res, os, index_.dim());
  raft::serialize_scalar(res, os, index_.graph_degree());
  raft::serialize_scalar(res, os, index_.metric());

  serialize_index_mdspan(res, os, index_.graph());

  bool has_source_indices = index_.source_indices().has_value();
  uint32_t content_map    = 0x1u * include_dataset + 0x2u * has_source_indices;

  raft::serialize_scalar(res, os, content_map);
  if (include_dataset) {
    RAFT_LOG_DEBUG("Saving CAGRA index with dataset");
    if constexpr (cuvs::neighbors::is_dense_row_major_dataset_view_v<DatasetViewT>) {
      neighbors::detail::serialize_cagra_dense_dataset<T, int64_t>(res, os, index_.dataset());
    } else {
      // Future dataset types (e.g. VPQ) require a new branch here and a corresponding
      // deserialize overload. Use static_assert to catch unsupported types at compile time.
      static_assert(
        sizeof(DatasetViewT) == 0,
        "serialize: dataset serialization is not yet implemented for this DatasetViewT");
    }
  } else {
    RAFT_LOG_DEBUG("Saving CAGRA index WITHOUT dataset");
  }

  if (has_source_indices) { serialize_index_mdspan(res, os, index_.source_indices().value()); }
}

template <typename T, typename IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
void serialize(raft::resources const& res,
               const std::string& filename,
               const cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>& index_,
               bool include_dataset)
{
  RAFT_EXPECTS(!index_.dataset_fd().has_value(),
               "Cannot serialize a disk-backed CAGRA index. Convert it with "
               "cuvs::neighbors::hnsw::from_cagra() and load it into memory via "
               "cuvs::neighbors::hnsw::deserialize() before serialization.");
  cuvs::util::kvikio_ofstream of(filename);

  detail::serialize(res, of, index_, include_dataset);

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}

template <typename T, typename IdxT, typename CagraIndexT>
void write_hnswlib_header(std::ostream& os, CagraIndexT const& index_, int dim)
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
                                 std::chrono::system_clock::time_point const& start_clock,
                                 size_t& next_report_offset)
{
  if (completed_rows < next_report_offset || completed_rows == 0) { return; }

  auto const elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::system_clock::now() - start_clock)
                         .count() *
                       1e-6;
  if (elapsed <= 0) { return; }

  constexpr double gib         = double(size_t{1} << 30);
  double const throughput      = bytes_written / gib / elapsed;
  double const rows_throughput = completed_rows / elapsed;
  double const eta             = (total_rows - completed_rows) / rows_throughput;
  RAFT_LOG_DEBUG(
    "# Writing rows %12zu / %12zu (%3.2f %%), %3.2f GiB/sec, ETA %d:%3.1f, written %3.2f GiB\r",
    completed_rows,
    total_rows,
    completed_rows / static_cast<double>(total_rows) * 100,
    throughput,
    int(eta / 60),
    std::fmod(eta, 60.0),
    bytes_written / gib);

  next_report_offset += std::max<size_t>(1, total_rows / 10);
}

template <typename T, typename IdxT, typename CagraIndexT>
void write_hnswlib_rows_host(
  raft::resources const& res,
  std::ostream& os,
  CagraIndexT const& index_,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  static_assert(is_cagra_hnsw_serialize_index_v<T, IdxT, CagraIndexT>,
                "serialize_to_hnswlib requires a dense device or host padded CAGRA index");

  auto const n_rows       = static_cast<size_t>(index_.size());
  auto const dim          = static_cast<int64_t>(dataset ? dataset->extent(1) : index_.dim());
  auto const graph_degree = static_cast<int64_t>(index_.graph_degree());
  auto const row_size =
    sizeof(int) + graph_degree * sizeof(IdxT) + dim * sizeof(T) + sizeof(size_t);
  if (n_rows == 0) { return; }

  auto const batch_rows = std::min<size_t>(
    n_rows, std::max<size_t>(1, cuvs::util::detail::kDeviceSerializationBatchBytes / row_size));
  auto graph_buffer =
    raft::make_host_matrix<IdxT, int64_t, raft::row_major>(batch_rows, graph_degree);
  auto dataset_buffer = raft::make_host_matrix<T, int64_t, raft::row_major>(batch_rows, dim);

  auto const graph = index_.graph();
  RAFT_EXPECTS(static_cast<size_t>(graph.extent(0)) == n_rows,
               "CAGRA graph rows (%zu) do not match index size (%zu)",
               static_cast<size_t>(graph.extent(0)),
               n_rows);

  T const* dataset_data  = nullptr;
  int64_t dataset_stride = 0;
  bool dataset_is_device = false;
  if (dataset) {
    dataset_data   = dataset->data_handle();
    dataset_stride = dataset->stride(0);
    RAFT_EXPECTS(static_cast<size_t>(dataset->extent(0)) == n_rows,
                 "CAGRA dataset rows (%zu) do not match index size (%zu)",
                 static_cast<size_t>(dataset->extent(0)),
                 n_rows);
  } else {
    auto const dataset_view = index_.dataset();
    RAFT_EXPECTS(dataset_view.n_rows() > 0,
                 "Invalid CAGRA dataset of size 0 during serialization, shape %zux%zu",
                 static_cast<size_t>(dataset_view.n_rows()),
                 static_cast<size_t>(dataset_view.dim()));
    RAFT_EXPECTS(static_cast<size_t>(dataset_view.n_rows()) == n_rows,
                 "CAGRA dataset rows (%zu) do not match index size (%zu)",
                 static_cast<size_t>(dataset_view.n_rows()),
                 n_rows);
    dataset_data      = dataset_view.view().data_handle();
    dataset_stride    = dataset_view.stride();
    dataset_is_device = is_device_cagra_hnsw_serialize_index_v<T, IdxT, CagraIndexT>;
  }

  auto const stream           = raft::resource::get_cuda_stream(res);
  auto const graph_degree_int = static_cast<int>(graph_degree);
  size_t bytes_written        = 0;
  size_t next_report          = std::max<size_t>(1, n_rows / 10);
  auto const start_clock      = std::chrono::system_clock::now();

  for (size_t first_row = 0; first_row < n_rows; first_row += batch_rows) {
    auto const rows = std::min(batch_rows, n_rows - first_row);
    raft::copy_matrix(graph_buffer.data_handle(),
                      graph_degree,
                      graph.data_handle() + first_row * graph.stride(0),
                      graph.stride(0),
                      graph_degree,
                      rows,
                      stream);
    if (dataset_is_device) {
      raft::copy_matrix(dataset_buffer.data_handle(),
                        dim,
                        dataset_data + first_row * dataset_stride,
                        dataset_stride,
                        dim,
                        rows,
                        stream);
    }
    raft::resource::sync_stream(res);

    for (size_t batch_row = 0; batch_row < rows; ++batch_row) {
      auto const row = first_row + batch_row;
      os.write(reinterpret_cast<char const*>(&graph_degree_int), sizeof(int));
      os.write(reinterpret_cast<char const*>(&graph_buffer(batch_row, 0)),
               graph_degree * sizeof(IdxT));
      T const* data_row =
        dataset_is_device ? &dataset_buffer(batch_row, 0) : dataset_data + row * dataset_stride;
      os.write(reinterpret_cast<char const*>(data_row), dim * sizeof(T));
      os.write(reinterpret_cast<char const*>(&row), sizeof(size_t));
    }
    RAFT_EXPECTS(os.good(), "Error writing HNSW file at row %zu", first_row + rows - 1);
    bytes_written += rows * row_size;
    log_hnswlib_progress(first_row + rows, n_rows, bytes_written, start_clock, next_report);
  }
}

template <typename ValueT>
__device__ void write_unaligned_value(uint8_t* output, ValueT value)
{
  auto const* bytes = reinterpret_cast<uint8_t const*>(&value);
#pragma unroll
  for (size_t i = 0; i < sizeof(ValueT); ++i) {
    output[i] = bytes[i];
  }
}

template <typename T, typename IdxT, typename GraphT>
__global__ void pack_hnswlib_rows(uint8_t* output,
                                  size_t row_size,
                                  GraphT const* graph,
                                  T const* dataset,
                                  size_t first_row,
                                  size_t rows,
                                  uint32_t graph_degree,
                                  uint32_t dim,
                                  int64_t dataset_stride)
{
  auto const warps_per_block = blockDim.x / warpSize;
  auto const warp            = threadIdx.x / warpSize;
  auto const lane            = threadIdx.x % warpSize;
  auto const batch_row       = blockIdx.x * warps_per_block + warp;
  if (batch_row >= rows) { return; }

  auto const row   = first_row + batch_row;
  auto* row_output = output + batch_row * row_size;
  if (lane == 0) {
    write_unaligned_value(row_output, static_cast<int>(graph_degree));
    auto const label_offset = sizeof(int) + graph_degree * sizeof(IdxT) + dim * sizeof(T);
    write_unaligned_value(row_output + label_offset, row);
  }

  auto const graph_offset = sizeof(int);
  for (size_t col = lane; col < graph_degree; col += warpSize) {
    write_unaligned_value(row_output + graph_offset + col * sizeof(IdxT),
                          static_cast<IdxT>(graph[row * static_cast<size_t>(graph_degree) + col]));
  }

  auto const dataset_offset = graph_offset + graph_degree * sizeof(IdxT);
  for (size_t col = lane; col < dim; col += warpSize) {
    write_unaligned_value(row_output + dataset_offset + col * sizeof(T),
                          dataset[row * static_cast<size_t>(dataset_stride) + col]);
  }
}

template <typename T, typename IdxT, typename CagraIndexT>
void write_hnswlib_rows_device(raft::resources const& res,
                               cuvs::util::kvikio_ofstream& os,
                               CagraIndexT const& index_)
{
  static_assert(is_device_cagra_hnsw_serialize_index_v<T, IdxT, CagraIndexT>,
                "write_hnswlib_rows_device requires a device-backed CAGRA index");

  auto const n_rows       = static_cast<size_t>(index_.size());
  auto const dim          = index_.dim();
  auto const graph_degree = index_.graph_degree();
  auto const row_size =
    sizeof(int) + graph_degree * sizeof(IdxT) + dim * sizeof(T) + sizeof(size_t);
  if (n_rows == 0) { return; }

  auto const graph   = index_.graph();
  auto const dataset = index_.dataset();
  RAFT_EXPECTS(dataset.n_rows() > 0,
               "Invalid CAGRA dataset of size 0 during serialization, shape %zux%zu",
               static_cast<size_t>(dataset.n_rows()),
               static_cast<size_t>(dataset.dim()));
  RAFT_EXPECTS(static_cast<size_t>(graph.extent(0)) == n_rows &&
                 static_cast<size_t>(dataset.n_rows()) == n_rows,
               "CAGRA graph and dataset rows must match index size");

  auto const batch_rows = std::min<size_t>(
    n_rows, std::max<size_t>(1, cuvs::util::detail::kDeviceSerializationBatchBytes / row_size));
  auto output       = raft::make_device_vector<uint8_t, int64_t>(res, batch_rows * row_size);
  auto const stream = raft::resource::get_cuda_stream(res);

  size_t bytes_written          = 0;
  size_t next_report            = std::max<size_t>(1, n_rows / 10);
  auto const start_clock        = std::chrono::system_clock::now();
  constexpr int block_size      = 256;
  constexpr int warps_per_block = block_size / 32;

  for (size_t first_row = 0; first_row < n_rows; first_row += batch_rows) {
    auto const rows   = std::min(batch_rows, n_rows - first_row);
    auto const blocks = (rows + warps_per_block - 1) / warps_per_block;
    pack_hnswlib_rows<T, IdxT>
      <<<static_cast<unsigned int>(blocks), block_size, 0, stream>>>(output.data_handle(),
                                                                     row_size,
                                                                     graph.data_handle(),
                                                                     dataset.view().data_handle(),
                                                                     first_row,
                                                                     rows,
                                                                     graph_degree,
                                                                     dim,
                                                                     dataset.stride());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::resource::sync_stream(res);

    auto const bytes = rows * row_size;
    os.write_device(output.data_handle(), bytes);
    bytes_written += bytes;
    log_hnswlib_progress(first_row + rows, n_rows, bytes_written, start_clock, next_report);
  }
}

inline void write_hnswlib_empty_levels(std::ostream& os, size_t n_rows)
{
  constexpr size_t chunk_size = 16 * 1024;
  std::array<int, chunk_size> const zeros{};
  for (size_t first_row = 0; first_row < n_rows; first_row += chunk_size) {
    auto const rows = std::min(chunk_size, n_rows - first_row);
    os.write(reinterpret_cast<char const*>(zeros.data()), rows * sizeof(int));
  }
}

template <typename T, typename IdxT, typename CagraIndexT>
void serialize_to_hnswlib(
  raft::resources const& res,
  std::ostream& os,
  CagraIndexT const& index_,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  static_assert(is_cagra_hnsw_serialize_index_v<T, IdxT, CagraIndexT>,
                "serialize_to_hnswlib requires a dense device or host padded CAGRA index");

  int const dim = dataset ? dataset->extent(1) : index_.dim();
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("cagra::serialize");
  RAFT_LOG_DEBUG("Saving CAGRA index to hnswlib format, size %zu, dim %d",
                 static_cast<size_t>(index_.size()),
                 dim);

  write_hnswlib_header<T, IdxT>(os, index_, dim);
  write_hnswlib_rows_host<T, IdxT>(res, os, index_, dataset);
  write_hnswlib_empty_levels(os, static_cast<size_t>(index_.size()));
}

template <typename T, typename IdxT, typename CagraIndexT>
void serialize_to_hnswlib(
  raft::resources const& res,
  const std::string& filename,
  CagraIndexT const& index_,
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset)
{
  static_assert(is_cagra_hnsw_serialize_index_v<T, IdxT, CagraIndexT>,
                "serialize_to_hnswlib requires a dense device or host padded CAGRA index");

  int const dim = dataset ? dataset->extent(1) : index_.dim();
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("cagra::serialize");
  RAFT_LOG_DEBUG("Saving CAGRA index to hnswlib format, size %zu, dim %d",
                 static_cast<size_t>(index_.size()),
                 dim);

  cuvs::util::kvikio_ofstream of(filename);

  write_hnswlib_header<T, IdxT>(of, index_, dim);
  if constexpr (is_device_cagra_hnsw_serialize_index_v<T, IdxT, CagraIndexT>) {
    if (dataset) {
      write_hnswlib_rows_host<T, IdxT>(res, of, index_, dataset);
    } else {
      write_hnswlib_rows_device<T, IdxT>(res, of, index_);
    }
  } else {
    write_hnswlib_rows_host<T, IdxT>(res, of, index_, dataset);
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
template <typename T, typename IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT, typename Input>
void deserialize_impl(
  raft::resources const& res,
  Input& input,
  cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>* index_,
  std::unique_ptr<cuvs::neighbors::owning_dataset_for_view_t<DatasetViewT>>* out_dataset = nullptr)
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
  auto const dataset_kind_raw = raft::deserialize_scalar<std::uint32_t>(res, is);
  RAFT_EXPECTS(is_valid_serialized_dataset_kind(dataset_kind_raw),
               "cagra::deserialize: invalid serialized dataset kind %u",
               dataset_kind_raw);
  auto const dataset_kind =
    static_cast<cuvs::neighbors::cagra::serialized_dataset_kind>(dataset_kind_raw);
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
  using index_t      = cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>;
  using graph_type_t = typename index_t::graph_index_type;
  RAFT_EXPECTS(cuvs::util::is_mul_no_overflow(static_cast<std::size_t>(n_rows),
                                              static_cast<std::size_t>(graph_degree),
                                              sizeof(graph_type_t)),
               "cagra::deserialize: integer overflow in graph allocation "
               "(n_rows=%lld, graph_degree=%u, element_size=%zu)",
               static_cast<long long>(n_rows),
               graph_degree,
               sizeof(graph_type_t));

  auto finish_deserialize = [&](auto graph) {
    auto content_map = raft::deserialize_scalar<uint32_t>(res, is);
    bool has_dataset = content_map & 0x1u;
    using kind       = cuvs::neighbors::cagra::serialized_dataset_kind;
    RAFT_EXPECTS(has_dataset == (dataset_kind != kind::none),
                 "cagra::deserialize: dataset kind and content map disagree");

    using owner_t = cuvs::neighbors::owning_dataset_for_view_t<DatasetViewT>;
    std::unique_ptr<owner_t> dataset_owner{};
    if (has_dataset) {
      if (out_dataset == nullptr) {
        cuvs::neighbors::detail::skip_dense_dataset<T, int64_t>(res, is);
      } else {
        auto const expected_kind = serialized_dataset_kind_for_view<DatasetViewT>();
        RAFT_EXPECTS(
          dataset_kind == expected_kind,
          "cagra::deserialize: serialized dataset kind %u does not match requested kind %u",
          dataset_kind_raw,
          static_cast<std::uint32_t>(expected_kind));
        if constexpr (cuvs::neighbors::is_device_padded_dataset_view_v<DatasetViewT>) {
          dataset_owner =
            cuvs::neighbors::detail::deserialize_padded_dataset<T, int64_t>(res, input);
        } else if constexpr (cuvs::neighbors::is_device_standard_dataset_view_v<DatasetViewT>) {
          dataset_owner =
            cuvs::neighbors::detail::deserialize_standard_dataset<T, int64_t>(res, input);
        } else if constexpr (cuvs::neighbors::is_host_padded_dataset_view_v<DatasetViewT>) {
          dataset_owner =
            cuvs::neighbors::detail::deserialize_host_padded_dataset<T, int64_t>(res, input);
        } else if constexpr (cuvs::neighbors::is_host_standard_dataset_view_v<DatasetViewT>) {
          dataset_owner =
            cuvs::neighbors::detail::deserialize_host_standard_dataset<T, int64_t>(res, input);
        } else {
          static_assert(sizeof(DatasetViewT) == 0,
                        "deserialize: dataset deserialization is not implemented for this view");
        }
      }
    }

    if (dataset_owner) {
      *index_ = index_t(
        res, metric, dataset_owner->as_dataset_view(), raft::make_const_mdspan(graph.view()));
    } else {
      *index_ = index_t(res, metric);
      if constexpr (raft::is_device_mdspan_v<decltype(graph.view())>) {
        index_->update_graph(res, std::move(graph));
      } else {
        index_->update_graph(res, raft::make_const_mdspan(graph.view()));
      }
    }

    if constexpr (raft::is_device_mdspan_v<decltype(graph.view())>) {
      if (dataset_owner) { index_->update_graph(res, std::move(graph)); }
      if (content_map & 0x2u) {
        auto source_indices = raft::make_device_vector<IdxT, int64_t>(res, n_rows);
        cuvs::util::detail::deserialize_mdspan(res, input, source_indices.view());
        index_->update_source_indices(std::move(source_indices));
      }
    } else {
      std::optional<raft::host_vector<IdxT, int64_t>> source_indices;
      if (content_map & 0x2u) {
        source_indices.emplace(raft::make_host_vector<IdxT, int64_t>(n_rows));
        raft::deserialize_mdspan(res, is, source_indices->view());
        index_->update_source_indices(res, raft::make_const_mdspan(source_indices->view()));
      }
      // Graph and source-index updates can enqueue copies from host staging. Keep both staging
      // buffers alive through this synchronization.
      raft::resource::sync_stream(res);
    }
    if (dataset_owner) { *out_dataset = std::move(dataset_owner); }
  };

  if constexpr (std::is_same_v<std::remove_cvref_t<Input>, cuvs::util::kvikio_file_reader>) {
    auto graph = raft::make_device_matrix<graph_type_t, int64_t>(res, n_rows, graph_degree);
    cuvs::util::detail::deserialize_mdspan(res, input, graph.view());
    finish_deserialize(std::move(graph));
  } else {
    auto graph = raft::make_host_matrix<graph_type_t, int64_t>(n_rows, graph_degree);
    raft::deserialize_mdspan(res, is, graph.view());
    finish_deserialize(std::move(graph));
  }
}

template <typename T, typename IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
void deserialize(
  raft::resources const& res,
  std::istream& is,
  cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>* index_,
  std::unique_ptr<cuvs::neighbors::owning_dataset_for_view_t<DatasetViewT>>* out_dataset = nullptr)
{
  deserialize_impl(res, is, index_, out_dataset);
}

template <typename T, typename IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
void deserialize(
  raft::resources const& res,
  const std::string& filename,
  cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>* index_,
  std::unique_ptr<cuvs::neighbors::owning_dataset_for_view_t<DatasetViewT>>* out_dataset = nullptr)
{
  cuvs::util::kvikio_file_reader reader(filename);
  deserialize_impl(res, reader, index_, out_dataset);
}
}  // namespace cuvs::neighbors::cagra::detail
