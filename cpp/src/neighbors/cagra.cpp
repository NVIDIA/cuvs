/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <cuvs/neighbors/common.hpp>

#include <raft/core/error.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/serialize.hpp>

#include <cstdint>
#include <fstream>
#include <istream>
#include <string>

namespace cuvs::neighbors::cagra {

graph_build_params_t index_params::graph_build_heuristic(raft::matrix_extent<int64_t> dataset,
                                                         size_t intermediate_graph_degree,
                                                         cuvs::distance::DistanceType metric,
                                                         size_t build_quality)
{
  if (dataset.extent(0) < int64_t(1e6)) {
    // Use NN descent for smaller datasets
    auto nn_descent_params =
      graph_build_params::nn_descent_params(intermediate_graph_degree, metric);
    nn_descent_params.max_iterations = 5 + build_quality;
    return nn_descent_params;
  } else {
    // Otherwise, use IVF-PQ
    auto ivf_pq_params = cuvs::neighbors::graph_build_params::ivf_pq_params(dataset, metric);
    ivf_pq_params.search_params.n_probes =
      std::round(2 + std::sqrt(ivf_pq_params.build_params.n_lists) / 20 + build_quality);
    return ivf_pq_params;
  }
}

cagra::index_params index_params::from_dataset(raft::matrix_extent<int64_t> dataset,
                                               size_t graph_degree,
                                               cuvs::distance::DistanceType metric,
                                               size_t build_quality)
{
  cagra::index_params params;
  params.metric                    = metric;
  params.graph_degree              = graph_degree;
  params.intermediate_graph_degree = graph_degree * 3 / 2;
  params.graph_build_params =
    graph_build_heuristic(dataset, params.intermediate_graph_degree, metric, build_quality);
  return params;
}

cagra::index_params index_params::from_hnsw_params(raft::matrix_extent<int64_t> dataset,
                                                   int M,
                                                   int ef_construction,
                                                   hnsw_heuristic_type heuristic,
                                                   cuvs::distance::DistanceType metric)
{
  cagra::index_params params;
  switch (heuristic) {
    case hnsw_heuristic_type::SAME_GRAPH_FOOTPRINT:
      params.graph_degree              = M * 2;
      params.intermediate_graph_degree = M * 3;
      break;
    case hnsw_heuristic_type::SIMILAR_SEARCH_PERFORMANCE:
    default:
      params.graph_degree              = 2 + M * 2 / 3;
      params.intermediate_graph_degree = M + M * ef_construction / 256;
      break;
  }
  params.graph_build_params =
    graph_build_heuristic(dataset, params.intermediate_graph_degree, metric, ef_construction / 16);
  return params;
}

namespace {

/**
 * Map the file's 4-byte NumPy dtype descriptor back to the element type that wrote it.
 *
 * Parses the descriptor rather than comparing against `get_numpy_dtype<T>()`, which has no answer
 * for `half` outside a CUDA translation unit. 'e' is how raft spells a half, as the C API's reader
 * also has to know.
 */
auto element_dtype_of(const char (&prefix)[4], const char* source) -> cudaDataType_t
{
  auto const dtype = raft::numpy_serializer::parse_descr(std::string(prefix, sizeof(prefix)));
  if (dtype.kind == 'f' && dtype.itemsize == 4) { return CUDA_R_32F; }
  if (dtype.kind == 'e' && dtype.itemsize == 2) { return CUDA_R_16F; }
  if (dtype.kind == 'i' && dtype.itemsize == 1) { return CUDA_R_8I; }
  if (dtype.kind == 'u' && dtype.itemsize == 1) { return CUDA_R_8U; }
  RAFT_FAIL("cagra::read_serialized_header: %s holds an index whose element type (%s) is not one "
            "CAGRA writes",
            source,
            dtype.to_string().c_str());
}

auto read_header(raft::resources const& res, std::istream& is, const char* source)
  -> serialized_index_header
{
  using pos_type   = std::istream::pos_type;
  using off_type   = std::istream::off_type;
  auto const start = is.tellg();
  RAFT_EXPECTS(start != pos_type{off_type{-1}},
               "cagra::read_serialized_header: %s is not seekable",
               source);

  char dtype_prefix[4];
  RAFT_EXPECTS(is.read(dtype_prefix, sizeof(dtype_prefix)),
               "cagra::read_serialized_header: failed to read the dtype prefix of %s",
               source);
  auto const dtype = element_dtype_of(dtype_prefix, source);

  auto const version = raft::deserialize_scalar<int>(res, is);
  RAFT_EXPECTS(version == cagra_serialization_version,
               "cagra::read_serialized_header: serialization version mismatch, expected %d, got %d",
               cagra_serialization_version,
               version);

  // Read after the version check: an older or newer format need not put the kind here at all.
  auto const kind_raw = raft::deserialize_scalar<std::uint32_t>(res, is);
  RAFT_EXPECTS(kind_raw <= static_cast<std::uint32_t>(serialized_dataset_kind::device_pq),
               "cagra::read_serialized_header: invalid serialized dataset kind %u in %s",
               kind_raw,
               source);

  // Rewind, so that the caller can hand the same stream to deserialize.
  is.seekg(start);
  return {dtype, static_cast<serialized_dataset_kind>(kind_raw)};
}

}  // namespace

auto read_serialized_header(raft::resources const& res, std::istream& is) -> serialized_index_header
{
  return read_header(res, is, "the stream");
}

auto read_serialized_header(raft::resources const& res, const std::string& filename)
  -> serialized_index_header
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);
  RAFT_EXPECTS(is, "cagra::read_serialized_header: cannot open %s", filename.c_str());
  return read_header(res, is, filename.c_str());
}

}  // namespace cuvs::neighbors::cagra
