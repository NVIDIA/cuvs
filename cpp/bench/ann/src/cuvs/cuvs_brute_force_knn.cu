/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <nvbench/nvbench.cuh>

#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

NVBENCH_DECLARE_TYPE_STRINGS(half, "F16", "half");

namespace cuvs::neighbors::brute_force {

auto parse_metric(std::string const& metric) -> cuvs::distance::DistanceType
{
  if (metric == "InnerProduct") { return cuvs::distance::DistanceType::InnerProduct; }
  if (metric == "L2SqrtExpanded") { return cuvs::distance::DistanceType::L2SqrtExpanded; }
  throw std::invalid_argument("Unsupported distance metric: " + metric);
}

template <typename T>
void search_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  auto const num_queries = state.get_int64("num_queries");
  auto const num_db_vecs = state.get_int64("num_db_vecs");
  auto const dim         = state.get_int64("dim");
  auto const k           = state.get_int64("k");
  auto const metric      = parse_metric(state.get_string("metric"));
  auto const layout      = state.get_string("layout");

  if (layout != "row_major" && layout != "column_major") {
    throw std::invalid_argument("Unsupported layout: " + layout);
  }

  raft::resources handle;
  auto const stream = raft::resource::get_cuda_stream(handle);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

  rmm::device_uvector<T> database(num_db_vecs * dim, stream);
  rmm::device_uvector<T> queries(num_queries * dim, stream);
  rmm::device_uvector<int64_t> neighbors(num_queries * k, stream);
  rmm::device_uvector<float> distances(num_queries * k, stream);

  raft::random::RngState rng(1234ULL);
  raft::random::uniform(handle, rng, database.data(), database.size(), T(-1.0f), T(1.0f));
  raft::random::uniform(handle, rng, queries.data(), queries.size(), T(-1.0f), T(1.0f));

  auto const neighbors_view =
    raft::make_device_matrix_view<int64_t, int64_t>(neighbors.data(), num_queries, k);
  auto const distances_view =
    raft::make_device_matrix_view<float, int64_t>(distances.data(), num_queries, k);

  index_params index_params;
  index_params.metric     = metric;
  index_params.metric_arg = 3.0;
  search_params search_params;

  state.add_element_count(num_queries);

  if (layout == "row_major") {
    auto const database_view =
      raft::make_device_matrix_view<const T, int64_t>(database.data(), num_db_vecs, dim);
    auto const queries_view =
      raft::make_device_matrix_view<const T, int64_t>(queries.data(), num_queries, dim);
    auto index = build(handle, index_params, database_view);

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      search(handle, search_params, index, queries_view, neighbors_view, distances_view);
    });
  } else {
    auto const database_view = raft::make_device_matrix_view<const T, int64_t, raft::col_major>(
      database.data(), num_db_vecs, dim);
    auto const queries_view = raft::make_device_matrix_view<const T, int64_t, raft::col_major>(
      queries.data(), num_queries, dim);
    auto index = build(handle, index_params, database_view);

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      search(handle, search_params, index, queries_view, neighbors_view, distances_view);
    });
  }
}

using value_types = nvbench::type_list<float, half>;

NVBENCH_BENCH_TYPES(search_benchmark, NVBENCH_TYPE_AXES(value_types))
  .set_name("brute_force_search")
  .add_int64_axis("num_queries", {10, 100, 1024})
  .add_int64_axis("num_db_vecs", {1000000})
  .add_int64_axis("dim", {32, 256, 1024})
  .add_int64_axis("k", {128, 1024})
  .add_string_axis("metric", {"InnerProduct", "L2SqrtExpanded"})
  .add_string_axis("layout", {"row_major", "column_major"});

}  // namespace cuvs::neighbors::brute_force
