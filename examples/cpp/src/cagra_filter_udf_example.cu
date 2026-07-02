/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>

#include <rmm/mr/pool_memory_resource.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int64_t n_rows    = 4096;
constexpr int64_t n_dim     = 32;
constexpr int64_t n_queries = 4;
constexpr int64_t k         = 8;

struct metadata_filter_context {
  const uint32_t* row_tenant_ids;
  const int64_t* row_timestamps;
  const uint32_t* row_language_ids;
  const uint64_t* row_acl_masks;

  const uint32_t* query_tenant_ids;
  const int64_t* query_min_timestamps;
  const uint64_t* query_allowed_language_masks;
  const uint64_t* query_permission_masks;
};

std::string metadata_udf_source()
{
  return R"cpp(
    struct metadata_filter_context {
      const uint32_t* row_tenant_ids;
      const int64_t* row_timestamps;
      const uint32_t* row_language_ids;
      const uint64_t* row_acl_masks;

      const uint32_t* query_tenant_ids;
      const int64_t* query_min_timestamps;
      const uint64_t* query_allowed_language_masks;
      const uint64_t* query_permission_masks;
    };

    __device__ bool tenant_filter(uint32_t query_id, source_index_t source_id, void* filter_data)
    {
      auto* ctx = static_cast<const metadata_filter_context*>(filter_data);
      return ctx->row_tenant_ids[source_id] == ctx->query_tenant_ids[query_id];
    }

    __device__ bool timestamp_filter(uint32_t query_id, source_index_t source_id, void* filter_data)
    {
      auto* ctx = static_cast<const metadata_filter_context*>(filter_data);
      return ctx->row_timestamps[source_id] >= ctx->query_min_timestamps[query_id];
    }

    __device__ bool language_acl_filter(uint32_t query_id, source_index_t source_id, void* filter_data)
    {
      auto* ctx               = static_cast<const metadata_filter_context*>(filter_data);
      const auto language_bit = uint64_t{1} << ctx->row_language_ids[source_id];
      const bool language_ok  = (ctx->query_allowed_language_masks[query_id] & language_bit) != 0;
      const bool acl_ok = (ctx->row_acl_masks[source_id] & ctx->query_permission_masks[query_id]) != 0;
      return language_ok && acl_ok;
    }
  )cpp";
}

template <typename DeviceVectorT, typename HostVectorT>
void copy_to_device(raft::device_resources const& res, DeviceVectorT& dst, HostVectorT const& src)
{
  raft::copy(dst.data_handle(), src.data(), src.size(), raft::resource::get_cuda_stream(res));
}

std::vector<uint32_t> copy_neighbors_to_host(
  raft::device_resources const& res,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors)
{
  std::vector<uint32_t> host(neighbors.size());
  raft::copy(
    host.data(), neighbors.data_handle(), host.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return host;
}

template <typename PredicateT>
void validate_and_print(char const* name,
                        raft::device_resources const& res,
                        raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
                        PredicateT is_valid)
{
  auto host_neighbors = copy_neighbors_to_host(res, neighbors);

  std::cout << name << " first query neighbors:";
  for (int64_t i = 0; i < k; ++i) {
    std::cout << " " << host_neighbors[static_cast<size_t>(i)];
  }
  std::cout << std::endl;

  for (int64_t q = 0; q < n_queries; ++q) {
    for (int64_t i = 0; i < k; ++i) {
      auto source_id = host_neighbors[static_cast<size_t>(q * k + i)];
      if (source_id < static_cast<uint32_t>(n_rows) && !is_valid(q, source_id)) {
        std::cerr << name << " produced invalid source_id=" << source_id << " for query=" << q
                  << std::endl;
        std::exit(1);
      }
    }
  }
}

}  // namespace

int main()
{
  raft::device_resources res;

  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  auto dataset = raft::make_device_matrix<float, int64_t>(res, n_rows, n_dim);
  auto queries = raft::make_device_matrix<float, int64_t>(res, n_queries, n_dim);

  raft::random::RngState rng(1234ULL);
  raft::random::uniform(res, rng, dataset.data_handle(), dataset.size(), -1.0f, 1.0f);
  raft::random::uniform(res, rng, queries.data_handle(), queries.size(), -1.0f, 1.0f);

  cuvs::neighbors::cagra::index_params index_params;
  index_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  index_params.graph_degree              = 32;
  index_params.intermediate_graph_degree = 64;
  index_params.graph_build_params = cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
    index_params.intermediate_graph_degree);

  std::cout << "Building CAGRA index" << std::endl;
  auto index =
    cuvs::neighbors::cagra::build(res, index_params, raft::make_const_mdspan(dataset.view()));

  std::vector<uint32_t> row_tenant_ids(n_rows);
  std::vector<int64_t> row_timestamps(n_rows);
  std::vector<uint32_t> row_language_ids(n_rows);
  std::vector<uint64_t> row_acl_masks(n_rows);
  for (int64_t i = 0; i < n_rows; ++i) {
    row_tenant_ids[static_cast<size_t>(i)]   = static_cast<uint32_t>(i % 4);
    row_timestamps[static_cast<size_t>(i)]   = 1'700'000'000 + i;
    row_language_ids[static_cast<size_t>(i)] = static_cast<uint32_t>(i % 8);
    row_acl_masks[static_cast<size_t>(i)]    = uint64_t{1} << (i % 16);
  }

  std::vector<uint32_t> query_tenant_ids{0, 1, 2, 3};
  std::vector<int64_t> query_min_timestamps{
    1'700'003'000, 1'700'002'000, 1'700'001'000, 1'700'000'500};
  std::vector<uint64_t> query_allowed_language_masks{(uint64_t{1} << 0) | (uint64_t{1} << 1),
                                                     (uint64_t{1} << 2) | (uint64_t{1} << 3),
                                                     (uint64_t{1} << 4) | (uint64_t{1} << 5),
                                                     (uint64_t{1} << 6) | (uint64_t{1} << 7)};
  std::vector<uint64_t> query_permission_masks{(uint64_t{1} << 0) | (uint64_t{1} << 8),
                                               (uint64_t{1} << 2) | (uint64_t{1} << 10),
                                               (uint64_t{1} << 4) | (uint64_t{1} << 12),
                                               (uint64_t{1} << 6) | (uint64_t{1} << 14)};

  auto row_tenant_ids_device       = raft::make_device_vector<uint32_t, int64_t>(res, n_rows);
  auto row_timestamps_device       = raft::make_device_vector<int64_t, int64_t>(res, n_rows);
  auto row_language_ids_device     = raft::make_device_vector<uint32_t, int64_t>(res, n_rows);
  auto row_acl_masks_device        = raft::make_device_vector<uint64_t, int64_t>(res, n_rows);
  auto query_tenant_ids_device     = raft::make_device_vector<uint32_t, int64_t>(res, n_queries);
  auto query_min_timestamps_device = raft::make_device_vector<int64_t, int64_t>(res, n_queries);
  auto query_allowed_language_masks_device =
    raft::make_device_vector<uint64_t, int64_t>(res, n_queries);
  auto query_permission_masks_device = raft::make_device_vector<uint64_t, int64_t>(res, n_queries);
  auto context_device = raft::make_device_vector<metadata_filter_context, int64_t>(res, 1);

  copy_to_device(res, row_tenant_ids_device, row_tenant_ids);
  copy_to_device(res, row_timestamps_device, row_timestamps);
  copy_to_device(res, row_language_ids_device, row_language_ids);
  copy_to_device(res, row_acl_masks_device, row_acl_masks);
  copy_to_device(res, query_tenant_ids_device, query_tenant_ids);
  copy_to_device(res, query_min_timestamps_device, query_min_timestamps);
  copy_to_device(res, query_allowed_language_masks_device, query_allowed_language_masks);
  copy_to_device(res, query_permission_masks_device, query_permission_masks);

  metadata_filter_context host_context{row_tenant_ids_device.data_handle(),
                                       row_timestamps_device.data_handle(),
                                       row_language_ids_device.data_handle(),
                                       row_acl_masks_device.data_handle(),
                                       query_tenant_ids_device.data_handle(),
                                       query_min_timestamps_device.data_handle(),
                                       query_allowed_language_masks_device.data_handle(),
                                       query_permission_masks_device.data_handle()};
  raft::copy(context_device.data_handle(), &host_context, 1, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);
  auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);

  cuvs::neighbors::cagra::search_params search_params;
  search_params.algo              = cuvs::neighbors::cagra::search_algo::MULTI_CTA;
  search_params.itopk_size        = 128;
  search_params.max_queries       = n_queries;
  search_params.thread_block_size = 256;

  auto source = metadata_udf_source();

  auto run_filter =
    [&](char const* label, char const* function_name, float filtering_rate, auto is_valid) {
      auto filter = cuvs::neighbors::filtering::udf_filter(
        source, context_device.data_handle(), filtering_rate, function_name);
      cuvs::neighbors::cagra::search(res,
                                     search_params,
                                     index,
                                     raft::make_const_mdspan(queries.view()),
                                     neighbors.view(),
                                     distances.view(),
                                     filter);
      validate_and_print(label, res, neighbors.view(), is_valid);
    };

  run_filter("tenant_filter", "tenant_filter", 0.75f, [&](int64_t query_id, uint32_t source_id) {
    return row_tenant_ids[source_id] == query_tenant_ids[query_id];
  });

  run_filter(
    "timestamp_filter", "timestamp_filter", 0.50f, [&](int64_t query_id, uint32_t source_id) {
      return row_timestamps[source_id] >= query_min_timestamps[query_id];
    });

  run_filter(
    "language_acl_filter",
    "language_acl_filter",
    0.875f,
    [&](int64_t query_id, uint32_t source_id) {
      const auto language_bit = uint64_t{1} << row_language_ids[source_id];
      const bool language_ok  = (query_allowed_language_masks[query_id] & language_bit) != 0;
      const bool acl_ok       = (row_acl_masks[source_id] & query_permission_masks[query_id]) != 0;
      return language_ok && acl_ok;
    });

  std::cout << "All CAGRA filter UDF examples produced valid filtered neighbors." << std::endl;
  return 0;
}
