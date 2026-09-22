/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef CUVS_ENABLE_FLOWANN_SEARCH

#include <cuvs/neighbors/flowann.hpp>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/serialize.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace cuvs::neighbors::cagra::experimental::flowann {

namespace detail {

inline constexpr int legacy_serialization_version       = 5;
inline constexpr std::uint32_t legacy_vpq_dataset_tag   = 3;
inline constexpr std::uint32_t legacy_vpq_codebook_bits = 8;
inline constexpr int previous_serialization_version     = 1;
inline constexpr int serialization_version              = 2;

enum class serialized_dataset_kind : std::uint32_t { dense = 1, vpq_f16 = 2 };

enum class legacy_graph_scalar_layout { typed_enums, historical_unsigned_enums };

template <typename T>
inline auto is_supported_metric(serialized_dataset_kind dataset_kind,
                                cuvs::distance::DistanceType metric) -> bool
{
  using cuvs::distance::DistanceType;
  if (dataset_kind == serialized_dataset_kind::vpq_f16) {
    return metric == DistanceType::L2Expanded;
  }
  switch (metric) {
    case DistanceType::L2Expanded:
    case DistanceType::InnerProduct:
    case DistanceType::CosineExpanded:
    case DistanceType::L1: return true;
    case DistanceType::BitwiseHamming: return std::is_same_v<T, std::uint8_t>;
    default: return false;
  }
}

template <typename T>
void validate_dtype_prefix(char const* serialized_dtype, char const* component)
{
  auto expected_dtype = raft::numpy_serializer::get_numpy_dtype<T>().to_string();
  expected_dtype.resize(4);
  RAFT_EXPECTS(std::memcmp(serialized_dtype, expected_dtype.data(), 4) == 0,
               "FlowANN legacy %s dtype does not match the requested type",
               component);
}

template <typename T>
void read_and_validate_dtype(std::istream& is, char const* component)
{
  char serialized_dtype[4]{};
  RAFT_EXPECTS(static_cast<bool>(is.read(serialized_dtype, sizeof(serialized_dtype))),
               "FlowANN legacy %s is truncated before its dtype prefix",
               component);
  validate_dtype_prefix<T>(serialized_dtype, component);
}

inline auto is_legacy_combined_dtype_prefix(char const* serialized_dtype) -> bool
{
  constexpr char legacy_dtype[4] = {'|', 'u', '1', '\0'};
  return std::memcmp(serialized_dtype, legacy_dtype, sizeof(legacy_dtype)) == 0;
}

inline void validate_matrix_size(std::uint64_t rows,
                                 std::uint64_t cols,
                                 std::size_t element_size,
                                 char const* component)
{
  RAFT_EXPECTS(cols == 0 || rows <= std::numeric_limits<std::size_t>::max() / cols,
               "FlowANN legacy %s shape overflows",
               component);
  auto const elements = rows * cols;
  RAFT_EXPECTS(
    element_size == 0 || elements <= std::numeric_limits<std::size_t>::max() / element_size,
    "FlowANN legacy %s byte size overflows",
    component);
}

inline auto is_legacy_f16_dtype(raft::numpy_serializer::dtype_t dtype) -> bool
{
  // RAFT historically serialized CUDA half as kind 'e'. NumPy uses kind 'f' for float16.
  return dtype.itemsize == sizeof(std::uint16_t) && (dtype.kind == 'e' || dtype.kind == 'f') &&
         (dtype.byteorder == '<' || dtype.byteorder == '=' || dtype.byteorder == '|');
}

template <typename T>
auto read_host_matrix(std::istream& is,
                      std::uint64_t expected_rows,
                      std::uint64_t expected_cols,
                      char const* component) -> raft::host_matrix<T, int64_t>
{
  auto const header = raft::numpy_serializer::read_header(is);
  RAFT_EXPECTS(
    header.shape.size() == 2, "FlowANN legacy %s must be a two-dimensional array", component);
  RAFT_EXPECTS(!header.fortran_order, "FlowANN legacy %s must use row-major order", component);
  RAFT_EXPECTS(header.dtype == raft::numpy_serializer::get_numpy_dtype<T>(),
               "FlowANN legacy %s has an unexpected dtype",
               component);
  RAFT_EXPECTS(header.shape[0] == expected_rows && header.shape[1] == expected_cols,
               "FlowANN legacy %s has an unexpected shape",
               component);
  validate_matrix_size(expected_rows, expected_cols, sizeof(T), component);
  RAFT_EXPECTS(expected_rows <= static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()) &&
                 expected_cols <= static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()),
               "FlowANN legacy %s shape exceeds int64",
               component);

  auto result      = raft::make_host_matrix<T, int64_t>(static_cast<int64_t>(expected_rows),
                                                   static_cast<int64_t>(expected_cols));
  auto const bytes = result.size() * sizeof(T);
  RAFT_EXPECTS(static_cast<bool>(is.read(reinterpret_cast<char*>(result.data_handle()), bytes)),
               "FlowANN legacy %s payload is truncated",
               component);
  return result;
}

template <typename T>
auto read_host_vector(std::istream& is, std::uint64_t expected_size, char const* component)
  -> raft::host_vector<T, int64_t>
{
  auto const header = raft::numpy_serializer::read_header(is);
  RAFT_EXPECTS(header.shape.size() == 1, "FlowANN %s must be a one-dimensional array", component);
  RAFT_EXPECTS(!header.fortran_order, "FlowANN %s must use row-major order", component);
  RAFT_EXPECTS(header.dtype == raft::numpy_serializer::get_numpy_dtype<T>(),
               "FlowANN %s has an unexpected dtype",
               component);
  RAFT_EXPECTS(header.shape[0] == expected_size, "FlowANN %s has an unexpected shape", component);
  validate_matrix_size(expected_size, 1, sizeof(T), component);
  RAFT_EXPECTS(expected_size <= static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()),
               "FlowANN %s shape exceeds int64",
               component);

  auto result = raft::make_host_vector<T, int64_t>(static_cast<int64_t>(expected_size));
  RAFT_EXPECTS(static_cast<bool>(
                 is.read(reinterpret_cast<char*>(result.data_handle()), result.size() * sizeof(T))),
               "FlowANN %s payload is truncated",
               component);
  return result;
}

template <typename FlowannIndexT>
void serialize_common(raft::resources const& res,
                      std::ostream& os,
                      FlowannIndexT const& index,
                      serialized_dataset_kind dataset_kind)
{
  static_assert(std::is_same_v<typename FlowannIndexT::index_type, std::uint32_t>,
                "The current FlowANN format stores uint32 indices");
  using value_type = typename FlowannIndexT::value_type;
  RAFT_EXPECTS(is_supported_metric<value_type>(dataset_kind, index.metric()),
               "FlowANN index uses an unsupported dataset and distance-metric combination");
  auto dtype = raft::numpy_serializer::get_numpy_dtype<value_type>().to_string();
  dtype.resize(4);
  os.write(dtype.data(), dtype.size());
  RAFT_EXPECTS(os.good(), "FlowANN failed to write its dtype prefix");

  auto const source_indices   = index.source_indices();
  auto const subgraph_offsets = index.subgraph_offsets();
  auto const subgraph_ids     = index.subgraph_ids();
  auto const seeds            = index.seeds();
  RAFT_EXPECTS(index.size() > 0 && index.dim() > 0 && index.graph_degree() > 0,
               "FlowANN cannot serialize an empty index");
  RAFT_EXPECTS(subgraph_offsets.has_value() && subgraph_ids.has_value(),
               "FlowANN serialization requires subgraph metadata");
  RAFT_EXPECTS(index.inner_graph().extent(1) <= std::numeric_limits<std::uint32_t>::max(),
               "FlowANN inner graph row length exceeds the serialization format");
  RAFT_EXPECTS(subgraph_offsets->extent(0) <= std::numeric_limits<std::uint32_t>::max(),
               "FlowANN subgraph count exceeds the serialization format");
  RAFT_EXPECTS(!seeds.has_value() || seeds->extent(0) <= std::numeric_limits<std::uint32_t>::max(),
               "FlowANN seed count exceeds the serialization format");

  raft::serialize_scalar(res, os, serialization_version);
  raft::serialize_scalar(res, os, static_cast<std::uint32_t>(dataset_kind));
  raft::serialize_scalar(res, os, static_cast<std::uint32_t>(index.size()));
  raft::serialize_scalar(res, os, index.dim());
  raft::serialize_scalar(res, os, index.graph_degree());
  raft::serialize_scalar(res, os, index.resident_degree());
  raft::serialize_scalar(res, os, index.cross_degree());
  raft::serialize_scalar(res, os, index.metric());
  raft::serialize_scalar(res, os, index.node_per_cacheline());
  raft::serialize_scalar(res, os, index.n_bits());
  raft::serialize_scalar(res, os, static_cast<std::uint32_t>(index.inner_graph().extent(1)));
  raft::serialize_scalar(res, os, static_cast<std::uint32_t>(subgraph_offsets->extent(0)));
  raft::serialize_scalar(
    res,
    os,
    source_indices.has_value() ? static_cast<std::uint32_t>(source_indices->extent(0)) : 0);
  raft::serialize_scalar(
    res, os, seeds.has_value() ? static_cast<std::uint32_t>(seeds->extent(0)) : 0);

  raft::serialize_mdspan(res, os, index.inner_graph());
  raft::serialize_mdspan(res, os, index.cross_graph());
  if (source_indices.has_value()) { raft::serialize_mdspan(res, os, *source_indices); }
  raft::serialize_mdspan(res, os, *subgraph_offsets);
  raft::serialize_mdspan(res, os, *subgraph_ids);
  if (seeds.has_value()) { raft::serialize_mdspan(res, os, *seeds); }
}

struct serialized_graph_data {
  std::uint32_t n_rows;
  std::uint32_t dim;
  std::uint32_t graph_degree;
  std::uint32_t resident_degree;
  std::uint32_t cross_degree;
  cuvs::distance::DistanceType metric;
  std::uint16_t node_per_cacheline;
  std::uint16_t n_bits;
  serialized_dataset_kind dataset_kind;
  raft::host_matrix<std::uint8_t, int64_t, raft::row_major> inner_graph;
  raft::host_matrix<std::uint32_t, int64_t, raft::row_major> cross_graph;
  std::optional<raft::host_vector<std::uint32_t, int64_t>> source_indices;
  raft::host_vector<std::uint32_t, int64_t> subgraph_offsets;
  raft::host_vector<std::uint32_t, int64_t> subgraph_ids;
  std::optional<raft::host_vector<std::uint32_t, int64_t>> seeds;
};

template <typename T>
auto deserialize_common_after_version(raft::resources const& res,
                                      std::istream& is,
                                      serialized_dataset_kind expected_kind,
                                      int version) -> serialized_graph_data
{
  RAFT_EXPECTS(version == previous_serialization_version || version == serialization_version,
               "FlowANN serialization version mismatch: expected %d or %d, got %d",
               previous_serialization_version,
               serialization_version,
               version);
  auto const dataset_kind =
    static_cast<serialized_dataset_kind>(raft::deserialize_scalar<std::uint32_t>(res, is));
  RAFT_EXPECTS(dataset_kind == expected_kind,
               "FlowANN serialized dataset kind does not match the requested index type");
  auto const n_rows       = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const dim          = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const graph_degree = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const resident_degree =
    version == serialization_version ? raft::deserialize_scalar<std::uint32_t>(res, is) : 0u;
  auto const cross_degree       = version == serialization_version
                                    ? raft::deserialize_scalar<std::uint32_t>(res, is)
                                    : graph_degree;
  auto const metric             = raft::deserialize_scalar<cuvs::distance::DistanceType>(res, is);
  auto const node_per_cacheline = raft::deserialize_scalar<std::uint16_t>(res, is);
  auto const n_bits             = raft::deserialize_scalar<std::uint16_t>(res, is);
  auto const inner_row_bytes    = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const n_subgraphs        = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const source_count       = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const seed_count         = raft::deserialize_scalar<std::uint32_t>(res, is);

  RAFT_EXPECTS(n_rows > 0 && dim > 0 && graph_degree > 0,
               "FlowANN serialized index contains a zero dimension");
  RAFT_EXPECTS(graph_degree <= std::numeric_limits<std::uint8_t>::max(),
               "FlowANN serialized graph degree exceeds its packed count type");
  RAFT_EXPECTS(resident_degree <= graph_degree && cross_degree <= graph_degree,
               "FlowANN serialized resident or cross degree exceeds the complete graph degree");
  RAFT_EXPECTS(resident_degree + cross_degree == graph_degree,
               "FlowANN serialized graph degrees are inconsistent");
  RAFT_EXPECTS(is_supported_metric<T>(dataset_kind, metric),
               "FlowANN serialized index uses an unsupported distance metric");
  RAFT_EXPECTS(node_per_cacheline > 0 && n_bits > 0 && n_bits <= 32,
               "FlowANN serialized encoding parameters are invalid");
  RAFT_EXPECTS(
    (resident_degree == 0 && inner_row_bytes == 0) || inner_row_bytes >= node_per_cacheline,
    "FlowANN serialized inner row is too short");
  RAFT_EXPECTS(n_subgraphs > 0 && n_subgraphs <= n_rows,
               "FlowANN serialized subgraph count is invalid");
  RAFT_EXPECTS(source_count == 0 || source_count == n_rows,
               "FlowANN serialized source mapping has an invalid size");
  RAFT_EXPECTS(seed_count <= n_rows, "FlowANN serialized seed list is too large");

  auto const inner_rows = inner_row_bytes == 0
                            ? 0
                            : raft::div_rounding_up_safe<std::uint64_t>(n_rows, node_per_cacheline);
  auto inner_graph = read_host_matrix<std::uint8_t>(is, inner_rows, inner_row_bytes, "inner graph");
  auto cross_graph = read_host_matrix<std::uint32_t>(
    is, n_rows, static_cast<std::uint64_t>(cross_degree) + 1, "cross graph");

  std::optional<raft::host_vector<std::uint32_t, int64_t>> source_indices;
  if (source_count > 0) {
    source_indices.emplace(
      read_host_vector<std::uint32_t>(is, source_count, "source-index mapping"));
  }
  auto subgraph_offsets = read_host_vector<std::uint32_t>(is, n_subgraphs, "subgraph offsets");
  auto subgraph_ids     = read_host_vector<std::uint32_t>(is, n_rows, "subgraph identifiers");
  std::optional<raft::host_vector<std::uint32_t, int64_t>> seeds;
  if (seed_count > 0) { seeds.emplace(read_host_vector<std::uint32_t>(is, seed_count, "seeds")); }

  return {n_rows,
          dim,
          graph_degree,
          resident_degree,
          cross_degree,
          metric,
          node_per_cacheline,
          n_bits,
          dataset_kind,
          std::move(inner_graph),
          std::move(cross_graph),
          std::move(source_indices),
          std::move(subgraph_offsets),
          std::move(subgraph_ids),
          std::move(seeds)};
}

template <typename T>
auto deserialize_common(raft::resources const& res,
                        std::istream& is,
                        serialized_dataset_kind expected_kind) -> serialized_graph_data
{
  read_and_validate_dtype<T>(is, "index");
  auto const version = raft::deserialize_scalar<int>(res, is);
  return deserialize_common_after_version<T>(res, is, expected_kind, version);
}

template <typename FlowannIndexT>
void populate_deserialized_index(raft::resources const& res,
                                 FlowannIndexT& index,
                                 serialized_graph_data&& data)
{
  index.update_graph(res,
                     std::move(data.inner_graph),
                     std::move(data.cross_graph),
                     data.graph_degree,
                     data.resident_degree);
  if (data.source_indices.has_value()) {
    index.update_source_indices(res, raft::make_const_mdspan(data.source_indices->view()));
  }
  index.update_subgraph_layout(res,
                               raft::make_const_mdspan(data.subgraph_offsets.view()),
                               raft::make_const_mdspan(data.subgraph_ids.view()));
  if (data.seeds.has_value()) {
    index.update_seeds(res, raft::make_const_mdspan(data.seeds->view()));
  }
  raft::resource::sync_stream(res);
}

inline auto read_host_f16_bits(std::istream& is,
                               std::uint64_t expected_rows,
                               std::uint64_t expected_cols,
                               char const* component) -> raft::host_matrix<std::uint16_t, int64_t>
{
  auto const header = raft::numpy_serializer::read_header(is);
  RAFT_EXPECTS(
    header.shape.size() == 2, "FlowANN legacy %s must be a two-dimensional array", component);
  RAFT_EXPECTS(!header.fortran_order, "FlowANN legacy %s must use row-major order", component);
  RAFT_EXPECTS(is_legacy_f16_dtype(header.dtype),
               "FlowANN legacy %s must contain little-endian float16 values",
               component);
  RAFT_EXPECTS(header.shape[0] == expected_rows && header.shape[1] == expected_cols,
               "FlowANN legacy %s has an unexpected shape",
               component);
  validate_matrix_size(expected_rows, expected_cols, sizeof(std::uint16_t), component);
  RAFT_EXPECTS(expected_rows <= static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()) &&
                 expected_cols <= static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()),
               "FlowANN legacy %s shape exceeds int64",
               component);

  auto result = raft::make_host_matrix<std::uint16_t, int64_t>(static_cast<int64_t>(expected_rows),
                                                               static_cast<int64_t>(expected_cols));
  auto const bytes = result.size() * sizeof(std::uint16_t);
  RAFT_EXPECTS(static_cast<bool>(is.read(reinterpret_cast<char*>(result.data_handle()), bytes)),
               "FlowANN legacy %s payload is truncated",
               component);
  return result;
}

template <typename T, typename IdxT, ann_dataset_view DatasetViewT>
auto deserialize_legacy_graph_body(raft::resources const& res,
                                   std::istream& is,
                                   flowann::index<T, IdxT, DatasetViewT>* index,
                                   legacy_graph_scalar_layout scalar_layout)
  -> std::pair<IdxT, std::uint32_t>
{
  static_assert(std::is_same_v<IdxT, std::uint32_t>,
                "The FlowANN legacy v5 graph format stores uint32 indices");
  RAFT_EXPECTS(index != nullptr, "FlowANN deserialize requires a non-null index output");

  auto const n_rows       = raft::deserialize_scalar<IdxT>(res, is);
  auto const dim          = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const graph_degree = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const metric =
    scalar_layout == legacy_graph_scalar_layout::historical_unsigned_enums
      ? static_cast<cuvs::distance::DistanceType>(raft::deserialize_scalar<std::uint32_t>(res, is))
      : raft::deserialize_scalar<cuvs::distance::DistanceType>(res, is);
  auto const node_per_cacheline = raft::deserialize_scalar<std::uint16_t>(res, is);
  auto const n_bits             = raft::deserialize_scalar<std::uint16_t>(res, is);

  RAFT_EXPECTS(n_rows > 0, "FlowANN legacy graph must not be empty");
  RAFT_EXPECTS(dim > 0, "FlowANN legacy graph dimension must be positive");
  RAFT_EXPECTS(graph_degree > 0, "FlowANN legacy graph degree must be positive");
  RAFT_EXPECTS(node_per_cacheline > 0, "FlowANN legacy node_per_cacheline must be positive");
  RAFT_EXPECTS(n_bits > 0 && n_bits <= 32, "FlowANN legacy n_bits must be in [1, 32]");
  RAFT_EXPECTS(index->dataset().n_rows() == 0 || index->dataset().n_rows() == n_rows,
               "FlowANN graph and dataset row counts differ");
  RAFT_EXPECTS(index->dataset().dim() == 0 || index->dataset().dim() == dim,
               "FlowANN graph and dataset dimensions differ");

  auto const inner_header = raft::numpy_serializer::read_header(is);
  RAFT_EXPECTS(inner_header.shape.size() == 2,
               "FlowANN legacy inner graph must be a two-dimensional array");
  RAFT_EXPECTS(!inner_header.fortran_order, "FlowANN legacy inner graph must use row-major order");
  RAFT_EXPECTS(inner_header.dtype == raft::numpy_serializer::get_numpy_dtype<std::uint8_t>(),
               "FlowANN legacy inner graph must contain uint8 values");
  auto const inner_rows          = inner_header.shape[0];
  auto const inner_cols          = inner_header.shape[1];
  auto const expected_inner_rows = raft::div_rounding_up_safe<std::uint64_t>(
    static_cast<std::uint64_t>(n_rows), node_per_cacheline);
  RAFT_EXPECTS(inner_rows == expected_inner_rows,
               "FlowANN legacy inner graph row count does not match node_per_cacheline");
  validate_matrix_size(inner_rows, inner_cols, sizeof(std::uint8_t), "inner graph");
  RAFT_EXPECTS(inner_rows <= static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()) &&
                 inner_cols <= static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()),
               "FlowANN legacy inner graph shape exceeds int64");
  auto inner_graph = raft::make_host_matrix<std::uint8_t, int64_t>(
    static_cast<int64_t>(inner_rows), static_cast<int64_t>(inner_cols));
  RAFT_EXPECTS(static_cast<bool>(is.read(reinterpret_cast<char*>(inner_graph.data_handle()),
                                         inner_graph.size() * sizeof(std::uint8_t))),
               "FlowANN legacy inner graph payload is truncated");

  auto cross_graph = read_host_matrix<std::uint32_t>(
    is, n_rows, static_cast<std::uint64_t>(graph_degree) + 1, "cross graph");
  auto source_indices = read_host_matrix<std::uint32_t>(is, 1, n_rows, "source-index mapping");
  for (int64_t i = 0; i < source_indices.extent(1); ++i) {
    RAFT_EXPECTS(source_indices(0, i) < n_rows,
                 "FlowANN source-index mapping contains an out-of-range ID at row %ld",
                 static_cast<long>(i));
  }

  DatasetViewT dataset = index->dataset();
  *index = flowann::index<T, IdxT, DatasetViewT>(res, metric, dataset, node_per_cacheline, n_bits);
  index->update_graph(
    res, raft::make_const_mdspan(inner_graph.view()), raft::make_const_mdspan(cross_graph.view()));
  index->update_source_indices(
    res, raft::make_host_vector_view<const IdxT, int64_t>(source_indices.data_handle(), n_rows));
  raft::resource::sync_stream(res);
  return {n_rows, dim};
}

template <typename T, typename IdxT, ann_dataset_view DatasetViewT>
void deserialize_legacy_graph(raft::resources const& res,
                              std::istream& is,
                              flowann::index<T, IdxT, DatasetViewT>* index)
{
  read_and_validate_dtype<T>(is, "graph");
  auto const version = raft::deserialize_scalar<int>(res, is);
  RAFT_EXPECTS(version == legacy_serialization_version,
               "FlowANN graph serialization version mismatch: expected %d, got %d",
               legacy_serialization_version,
               version);
  (void)deserialize_legacy_graph_body(res, is, index, legacy_graph_scalar_layout::typed_enums);
}

template <typename T, typename IdxT, ann_dataset_view DatasetViewT>
void deserialize_legacy_graph(raft::resources const& res,
                              std::string const& filename,
                              flowann::index<T, IdxT, DatasetViewT>* index)
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);
  RAFT_EXPECTS(is.is_open(), "Cannot open FlowANN graph file %s", filename.c_str());
  deserialize_legacy_graph(res, is, index);
}

struct legacy_vpq_f16_data {
  std::unique_ptr<cuvs::neighbors::device_vpq_dataset<half, int64_t>> dataset;
  raft::host_vector<std::uint32_t, int64_t> subgraph_offsets;
  raft::host_vector<std::uint32_t, int64_t> subgraph_ids;
};

template <typename T>
auto deserialize_legacy_vpq_f16_body(raft::resources const& res, std::istream& is)
  -> legacy_vpq_f16_data
{
  auto const dataset_tag = raft::deserialize_scalar<std::uint32_t>(res, is);
  RAFT_EXPECTS(dataset_tag == legacy_vpq_dataset_tag,
               "FlowANN legacy dataset must contain VPQ data");
  auto const dtype = raft::deserialize_scalar<cudaDataType_t>(res, is);
  RAFT_EXPECTS(dtype == CUDA_R_16F, "FlowANN legacy VPQ codebooks must use float16");

  auto const n_rows                = raft::deserialize_scalar<int64_t>(res, is);
  auto const dim                   = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const vq_n_centers          = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const pq_n_centers          = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const pq_len                = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const serialized_row_length = raft::deserialize_scalar<std::uint32_t>(res, is);

  RAFT_EXPECTS(n_rows > 0 && dim > 0 && vq_n_centers > 0 && pq_len > 0,
               "FlowANN legacy VPQ metadata contains a zero dimension");
  RAFT_EXPECTS(vq_n_centers <= (1u << 16), "FlowANN legacy VPQ stores VQ labels in 16 bits");
  RAFT_EXPECTS(pq_n_centers == (1u << legacy_vpq_codebook_bits),
               "FlowANN legacy VPQ search requires 8-bit PQ codes");
  RAFT_EXPECTS(pq_len == 2 || pq_len == 4 || pq_len == 8,
               "FlowANN legacy VPQ pq_len must be 2, 4, or 8");
  auto const pq_dim = raft::div_rounding_up_safe<std::uint32_t>(dim, pq_len);
  RAFT_EXPECTS(serialized_row_length >= sizeof(std::uint32_t) + pq_dim,
               "FlowANN legacy VPQ encoded rows are too short");
  RAFT_EXPECTS(serialized_row_length <= std::numeric_limits<std::uint32_t>::max() - 3,
               "FlowANN legacy VPQ encoded row length cannot be aligned safely");
  auto const encoded_row_length = (serialized_row_length + 3u) & ~3u;
  validate_matrix_size(static_cast<std::uint64_t>(n_rows),
                       encoded_row_length,
                       sizeof(std::uint8_t),
                       "aligned VPQ codes");

  auto vq_bits         = read_host_f16_bits(is, vq_n_centers, dim, "VQ codebook");
  auto pq_bits         = read_host_f16_bits(is, pq_n_centers, pq_len, "PQ codebook");
  auto serialized_data = read_host_matrix<std::uint8_t>(
    is, static_cast<std::uint64_t>(n_rows), serialized_row_length, "VPQ codes");
  auto host_data = raft::make_host_matrix<std::uint8_t, int64_t>(n_rows, encoded_row_length);
  std::fill_n(host_data.data_handle(), host_data.size(), std::uint8_t{0});
  for (int64_t row = 0; row < n_rows; ++row) {
    std::copy_n(
      serialized_data.data_handle() + static_cast<std::size_t>(row) * serialized_row_length,
      serialized_row_length,
      host_data.data_handle() + static_cast<std::size_t>(row) * encoded_row_length);
  }

  auto const n_subgraphs = raft::deserialize_scalar<std::uint32_t>(res, is);
  RAFT_EXPECTS(n_subgraphs > 0, "FlowANN legacy VPQ dataset must contain subgraph offsets");
  auto offsets_matrix = read_host_matrix<std::uint32_t>(is, 1, n_subgraphs, "subgraph offsets");
  auto offsets        = raft::make_host_vector<std::uint32_t, int64_t>(n_subgraphs);
  std::copy_n(offsets_matrix.data_handle(), n_subgraphs, offsets.data_handle());
  RAFT_EXPECTS(offsets(0) == 0, "FlowANN first subgraph offset must be zero");
  for (std::uint32_t i = 1; i < n_subgraphs; ++i) {
    RAFT_EXPECTS(offsets(i - 1) <= offsets(i), "FlowANN subgraph offsets must be monotonic");
  }
  RAFT_EXPECTS(offsets(n_subgraphs - 1) < static_cast<std::uint64_t>(n_rows),
               "FlowANN subgraph offsets exceed the dataset size");

  auto subgraph_ids = raft::make_host_vector<std::uint32_t, int64_t>(n_rows);
  for (int64_t row = 0; row < n_rows; ++row) {
    auto* encoded = host_data.data_handle() + static_cast<std::size_t>(row) * encoded_row_length;
    auto const vq_label =
      static_cast<std::uint32_t>(encoded[0]) | (static_cast<std::uint32_t>(encoded[1]) << 8);
    RAFT_EXPECTS(vq_label < vq_n_centers,
                 "FlowANN VPQ row %ld has an out-of-range VQ label",
                 static_cast<long>(row));
    auto const subgraph_id =
      static_cast<std::uint32_t>(encoded[2]) | (static_cast<std::uint32_t>(encoded[3]) << 8);
    RAFT_EXPECTS(subgraph_id < n_subgraphs,
                 "FlowANN VPQ row %ld has an out-of-range subgraph identifier",
                 static_cast<long>(row));
    subgraph_ids(row) = subgraph_id;
    // Current CAGRA VPQ expects a 32-bit VQ label. Legacy FlowANN used the upper 16 bits for the
    // subgraph ID; preserve that metadata separately and normalize the row for the current
    // distance descriptor.
    encoded[2] = 0;
    encoded[3] = 0;
  }

  auto vq_code_book = raft::make_device_matrix<half, std::uint32_t>(res, vq_n_centers, dim);
  auto pq_code_book = raft::make_device_matrix<half, std::uint32_t>(res, pq_n_centers, pq_len);
  auto data = raft::make_device_matrix<std::uint8_t, int64_t>(res, n_rows, encoded_row_length);
  auto const stream = raft::resource::get_cuda_stream(res).get();
  RAFT_CUDA_TRY(cudaMemcpyAsync(vq_code_book.data_handle(),
                                vq_bits.data_handle(),
                                vq_bits.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(pq_code_book.data_handle(),
                                pq_bits.data_handle(),
                                pq_bits.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(data.data_handle(),
                                host_data.data_handle(),
                                host_data.size() * sizeof(std::uint8_t),
                                cudaMemcpyHostToDevice,
                                stream));

  return {std::make_unique<cuvs::neighbors::device_vpq_dataset<half, int64_t>>(
            std::move(vq_code_book), std::move(pq_code_book), std::move(data)),
          std::move(offsets),
          std::move(subgraph_ids)};
}

template <typename T>
auto deserialize_legacy_vpq_f16(raft::resources const& res, std::istream& is) -> legacy_vpq_f16_data
{
  read_and_validate_dtype<T>(is, "dataset");
  auto const version = raft::deserialize_scalar<int>(res, is);
  RAFT_EXPECTS(version == legacy_serialization_version,
               "FlowANN dataset serialization version mismatch: expected %d, got %d",
               legacy_serialization_version,
               version);
  return deserialize_legacy_vpq_f16_body<T>(res, is);
}

template <typename T>
auto deserialize_legacy_vpq_f16(raft::resources const& res, std::string const& filename)
  -> legacy_vpq_f16_data
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);
  RAFT_EXPECTS(is.is_open(), "Cannot open FlowANN dataset file %s", filename.c_str());
  return deserialize_legacy_vpq_f16<T>(res, is);
}

template <typename T, typename IdxT = std::uint32_t>
auto deserialize_legacy_combined_vpq_f16_after_version(raft::resources const& res,
                                                       std::istream& is,
                                                       int version) -> vpq_f16_index_bundle<T, IdxT>
{
  static_assert(std::is_same_v<IdxT, std::uint32_t>,
                "The FlowANN legacy v5 combined format stores uint32 indices");
  RAFT_EXPECTS(version == legacy_serialization_version,
               "FlowANN combined serialization version mismatch: expected %d, got %d",
               legacy_serialization_version,
               version);

  flowann::vpq_f16_index<T, IdxT> index(res);
  auto const [graph_rows, graph_dim] = deserialize_legacy_graph_body(
    res, is, &index, legacy_graph_scalar_layout::historical_unsigned_enums);
  auto const has_dataset = raft::deserialize_scalar<bool>(res, is);
  RAFT_EXPECTS(has_dataset, "FlowANN legacy combined index does not contain a dataset");
  auto dataset_data = deserialize_legacy_vpq_f16_body<T>(res, is);
  RAFT_EXPECTS(graph_rows == static_cast<IdxT>(dataset_data.dataset->n_rows()),
               "FlowANN graph and VPQ dataset row counts differ");
  RAFT_EXPECTS(graph_dim == dataset_data.dataset->dim(),
               "FlowANN graph and VPQ dataset dimensions differ");
  index.update_dataset(res, dataset_data.dataset->as_dataset_view());
  index.update_subgraph_layout(
    res,
    raft::make_host_vector_view<const std::uint32_t, int64_t>(
      dataset_data.subgraph_offsets.data_handle(), dataset_data.subgraph_offsets.extent(0)),
    raft::make_host_vector_view<const std::uint32_t, int64_t>(
      dataset_data.subgraph_ids.data_handle(), dataset_data.subgraph_ids.extent(0)));
  raft::resource::sync_stream(res);
  return {std::move(dataset_data.dataset), std::move(index)};
}

}  // namespace detail

/** Load the split graph and VPQ dataset emitted by the FlowANN legacy v5 tools. */
template <typename T, typename IdxT = std::uint32_t>
auto deserialize_vpq_f16(raft::resources const& res,
                         std::string const& graph_filename,
                         std::string const& dataset_filename) -> vpq_f16_index_bundle<T, IdxT>
{
  auto dataset_data = detail::deserialize_legacy_vpq_f16<T>(res, dataset_filename);
  flowann::vpq_f16_index<T, IdxT> index(res);
  index.update_dataset(res, dataset_data.dataset->as_dataset_view());
  detail::deserialize_legacy_graph(res, graph_filename, &index);
  RAFT_EXPECTS(index.size() == static_cast<IdxT>(dataset_data.dataset->n_rows()),
               "FlowANN graph and VPQ dataset row counts differ");
  index.update_subgraph_layout(
    res,
    raft::make_host_vector_view<const std::uint32_t, int64_t>(
      dataset_data.subgraph_offsets.data_handle(), dataset_data.subgraph_offsets.extent(0)),
    raft::make_host_vector_view<const std::uint32_t, int64_t>(
      dataset_data.subgraph_ids.data_handle(), dataset_data.subgraph_ids.extent(0)));
  raft::resource::sync_stream(res);
  return {std::move(dataset_data.dataset), std::move(index)};
}

/** Serialize a dense FlowANN index using the current self-contained format. */
template <typename T, typename IdxT = std::uint32_t>
void serialize(raft::resources const& res,
               std::ostream& os,
               device_padded_index<T, IdxT> const& index)
{
  detail::serialize_common(res, os, index, detail::serialized_dataset_kind::dense);
  raft::serialize_scalar(res, os, index.dataset().stride());
  raft::serialize_mdspan(res, os, index.dataset().view());
  RAFT_EXPECTS(os.good(), "FlowANN failed while serializing its dense dataset");
}

/** Serialize a float16-VPQ FlowANN index using the current self-contained format. */
template <typename T, typename IdxT = std::uint32_t>
void serialize(raft::resources const& res, std::ostream& os, vpq_f16_index<T, IdxT> const& index)
{
  detail::serialize_common(res, os, index, detail::serialized_dataset_kind::vpq_f16);
  auto const& dataset = index.dataset().dset();
  raft::serialize_scalar(res, os, dataset.vq_n_centers());
  raft::serialize_scalar(res, os, dataset.pq_n_centers());
  raft::serialize_scalar(res, os, dataset.pq_len());
  raft::serialize_scalar(res, os, dataset.encoded_row_length());
  raft::serialize_mdspan(res, os, raft::make_const_mdspan(dataset.vq_code_book.view()));
  raft::serialize_mdspan(res, os, raft::make_const_mdspan(dataset.pq_code_book.view()));
  raft::serialize_mdspan(res, os, raft::make_const_mdspan(dataset.data.view()));
  RAFT_EXPECTS(os.good(), "FlowANN failed while serializing its VPQ dataset");
}

/** Serialize a FlowANN index to a file using the current self-contained format. */
template <typename FlowannIndexT>
void serialize(raft::resources const& res, std::string const& filename, FlowannIndexT const& index)
{
  std::ofstream os(filename, std::ios::out | std::ios::binary);
  RAFT_EXPECTS(os.is_open(), "Cannot open FlowANN output file %s", filename.c_str());
  serialize(res, os, index);
  os.close();
  RAFT_EXPECTS(os.good(), "FlowANN failed to close output file %s", filename.c_str());
}

/** Deserialize a dense FlowANN index from the current self-contained format. */
template <typename T = float, typename IdxT = std::uint32_t>
auto deserialize_device_padded(raft::resources const& res, std::istream& is)
  -> device_padded_index_bundle<T, IdxT>
{
  static_assert(std::is_same_v<IdxT, std::uint32_t>,
                "The current FlowANN format stores uint32 indices");
  auto graph = detail::deserialize_common<T>(res, is, detail::serialized_dataset_kind::dense);
  auto const stride = raft::deserialize_scalar<std::uint32_t>(res, is);
  RAFT_EXPECTS(stride >= graph.dim, "FlowANN dense dataset stride is smaller than its dimension");
  detail::validate_matrix_size(graph.n_rows, stride, sizeof(T), "dense dataset");
  auto storage =
    raft::make_device_matrix<T, int64_t>(res, static_cast<int64_t>(graph.n_rows), stride);
  raft::deserialize_mdspan(res, is, storage.view());
  auto dataset = std::make_unique<cuvs::neighbors::device_padded_dataset<T, int64_t>>(
    std::move(storage), graph.dim);
  device_padded_index<T, IdxT> index(
    res, graph.metric, dataset->as_dataset_view(), graph.node_per_cacheline, graph.n_bits);
  detail::populate_deserialized_index(res, index, std::move(graph));
  return {std::move(dataset), std::move(index)};
}

/** Deserialize a dense FlowANN index file in the current self-contained format. */
template <typename T = float, typename IdxT = std::uint32_t>
auto deserialize_device_padded(raft::resources const& res, std::string const& filename)
  -> device_padded_index_bundle<T, IdxT>
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);
  RAFT_EXPECTS(is.is_open(), "Cannot open FlowANN index file %s", filename.c_str());
  return deserialize_device_padded<T, IdxT>(res, is);
}

/**
 * @brief Deserialize a float16-VPQ FlowANN index from a stream.
 *
 * The loader identifies the format from the four-byte dtype prefix and serialization version.
 * Current self-contained versions 1 and 2 use their typed dtype prefix. Historical combined v5
 * files use the `|u1` prefix emitted by the original FlowANN split serializer. The stream must be
 * positioned at the beginning of the index. The returned bundle owns the device VPQ dataset that
 * backs the index.
 *
 * @tparam T Requested query value type. Current formats validate `T` against their typed prefix.
 * Historical combined v5 uses an untyped `|u1` format marker, so the caller selects `T` from the
 * query types supported by FlowANN search.
 * @tparam IdxT Index type. Only `std::uint32_t` is supported by these formats.
 * @param res RAFT resources used for allocations and transfers.
 * @param is Input stream containing one complete index.
 * @return An owning VPQ dataset and the loaded FlowANN index.
 * @throws raft::logic_error if the format, dtype, dimensions, graph, or VPQ metadata is invalid.
 */
template <typename T = float, typename IdxT = std::uint32_t>
auto deserialize_vpq_f16(raft::resources const& res, std::istream& is)
  -> vpq_f16_index_bundle<T, IdxT>
{
  static_assert(std::is_same_v<IdxT, std::uint32_t>,
                "FlowANN VPQ serialization formats store uint32 indices");
  char serialized_dtype[4]{};
  RAFT_EXPECTS(static_cast<bool>(is.read(serialized_dtype, sizeof(serialized_dtype))),
               "FlowANN index is truncated before its dtype prefix");
  auto const version = raft::deserialize_scalar<int>(res, is);
  if (detail::is_legacy_combined_dtype_prefix(serialized_dtype) &&
      version == detail::legacy_serialization_version) {
    return detail::deserialize_legacy_combined_vpq_f16_after_version<T, IdxT>(res, is, version);
  }
  detail::validate_dtype_prefix<T>(serialized_dtype, "index");
  auto graph = detail::deserialize_common_after_version<T>(
    res, is, detail::serialized_dataset_kind::vpq_f16, version);
  auto const vq_n_centers = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const pq_n_centers = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const pq_len       = raft::deserialize_scalar<std::uint32_t>(res, is);
  auto const row_length   = raft::deserialize_scalar<std::uint32_t>(res, is);
  RAFT_EXPECTS(vq_n_centers > 0 && pq_n_centers > 0 && pq_len > 0 && row_length > 0,
               "FlowANN VPQ metadata contains a zero dimension");
  RAFT_EXPECTS(pq_n_centers == 256, "FlowANN search currently requires 8-bit VPQ codebooks");
  RAFT_EXPECTS(pq_len == 2 || pq_len == 4 || pq_len == 8, "FlowANN VPQ pq_len must be 2, 4, or 8");
  auto const pq_dim = raft::div_rounding_up_safe<std::uint32_t>(graph.dim, pq_len);
  auto const expected_row_length =
    sizeof(std::uint32_t) * (1 + raft::div_rounding_up_safe<std::uint32_t>(pq_dim, 4));
  RAFT_EXPECTS(row_length == expected_row_length,
               "FlowANN VPQ encoded row length does not match its dimension and pq_len");
  detail::validate_matrix_size(vq_n_centers, graph.dim, sizeof(half), "VQ codebook");
  detail::validate_matrix_size(pq_n_centers, pq_len, sizeof(half), "PQ codebook");
  detail::validate_matrix_size(graph.n_rows, row_length, sizeof(std::uint8_t), "VPQ codes");

  auto vq_code_book = raft::make_device_matrix<half, std::uint32_t>(res, vq_n_centers, graph.dim);
  auto pq_code_book = raft::make_device_matrix<half, std::uint32_t>(res, pq_n_centers, pq_len);
  auto encoded = raft::make_device_matrix<std::uint8_t, int64_t>(res, graph.n_rows, row_length);
  raft::deserialize_mdspan(res, is, vq_code_book.view());
  raft::deserialize_mdspan(res, is, pq_code_book.view());
  raft::deserialize_mdspan(res, is, encoded.view());
  auto dataset = std::make_unique<cuvs::neighbors::device_vpq_dataset<half, int64_t>>(
    std::move(vq_code_book), std::move(pq_code_book), std::move(encoded));
  RAFT_EXPECTS(dataset->dim() == graph.dim && dataset->n_rows() == graph.n_rows,
               "FlowANN VPQ dataset metadata does not match the graph");

  vpq_f16_index<T, IdxT> index(
    res, graph.metric, dataset->as_dataset_view(), graph.node_per_cacheline, graph.n_bits);
  detail::populate_deserialized_index(res, index, std::move(graph));
  return {std::move(dataset), std::move(index)};
}

/**
 * @brief Deserialize a float16-VPQ FlowANN index file.
 *
 * This overload opens `filename` and applies the same current-versus-legacy auto-detection as the
 * stream overload. The returned bundle owns the device VPQ dataset that backs the index.
 *
 * @tparam T Requested query value type. Current formats validate `T` against their typed prefix.
 * Historical combined v5 uses an untyped `|u1` format marker, so the caller selects `T` from the
 * query types supported by FlowANN search.
 * @tparam IdxT Index type. Only `std::uint32_t` is supported by these formats.
 * @param res RAFT resources used for allocations and transfers.
 * @param filename Path to a current self-contained or historical combined v5 index.
 * @return An owning VPQ dataset and the loaded FlowANN index.
 * @throws raft::logic_error if the file cannot be opened or its serialized contents are invalid.
 */
template <typename T = float, typename IdxT = std::uint32_t>
auto deserialize_vpq_f16(raft::resources const& res, std::string const& filename)
  -> vpq_f16_index_bundle<T, IdxT>
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);
  RAFT_EXPECTS(is.is_open(), "Cannot open FlowANN index file %s", filename.c_str());
  return deserialize_vpq_f16<T, IdxT>(res, is);
}

}  // namespace cuvs::neighbors::cagra::experimental::flowann

#endif  // CUVS_ENABLE_FLOWANN_SEARCH
