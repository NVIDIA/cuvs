/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/roaring_allowlist.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>

#include <rmm/mr/pool_memory_resource.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

constexpr std::int64_t n_rows    = 1024;
constexpr std::int64_t n_dim     = 16;
constexpr std::int64_t n_queries = 4;
constexpr std::int64_t k         = 8;

}  // namespace

int main()
{
  raft::device_resources res;
  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  auto dataset = raft::make_device_matrix<float, std::int64_t>(res, n_rows, n_dim);
  auto queries = raft::make_device_matrix<float, std::int64_t>(res, n_queries, n_dim);
  raft::random::RngState rng(1234ULL);
  raft::random::uniform(res, rng, dataset.data_handle(), dataset.size(), -1.0f, 1.0f);
  raft::random::uniform(res, rng, queries.data_handle(), queries.size(), -1.0f, 1.0f);

  cuvs::neighbors::cagra::index_params index_params;
  index_params.graph_degree              = 32;
  index_params.intermediate_graph_degree = 64;
  auto padded =
    cuvs::neighbors::make_device_padded_dataset_view(res, raft::make_const_mdspan(dataset.view()));
  auto index = cuvs::neighbors::cagra::build(res, index_params, padded);

  // Each owner is independently reusable. The filter supplies the query-to-allowlist mapping by
  // retaining only zero-copy views of their already initialized device references.
  std::vector<cuvs::core::roaring_allowlist> owners;
  owners.reserve(n_queries);
  for (std::uint32_t query = 0; query < n_queries; ++query) {
    std::vector<std::uint32_t> ids;
    for (std::uint32_t row = query; row < n_rows; row += n_queries) {
      ids.push_back(row);
    }
    owners.push_back(cuvs::core::roaring_allowlist::from_ids(
      res,
      n_rows,
      raft::make_host_vector_view<const std::uint32_t, std::int64_t>(ids.data(), ids.size()),
      true));
  }

  std::vector<cuvs::core::roaring_allowlist_view> views;
  views.reserve(owners.size());
  for (auto const& owner : owners) {
    views.push_back(owner.view());
  }
  cuvs::neighbors::filtering::roaring_filter filter(res, views);
  auto const* prepared_payload = filter.device_payload();

  auto neighbors = raft::make_device_matrix<std::uint32_t, std::int64_t>(res, n_queries, k);
  auto distances = raft::make_device_matrix<float, std::int64_t>(res, n_queries, k);
  cuvs::neighbors::cagra::search_params search_params;
  search_params.algo        = cuvs::neighbors::cagra::search_algo::MULTI_CTA;
  search_params.itopk_size  = 64;
  search_params.max_queries = 2;  // also demonstrates internal query chunking

  cuvs::neighbors::cagra::search(res,
                                 search_params,
                                 index,
                                 raft::make_const_mdspan(queries.view()),
                                 neighbors.view(),
                                 distances.view(),
                                 filter);
  if (filter.device_payload() != prepared_payload) { return 1; }

  std::vector<std::uint32_t> host_neighbors(neighbors.size());
  raft::copy(host_neighbors.data(),
             neighbors.data_handle(),
             host_neighbors.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  for (std::int64_t query = 0; query < n_queries; ++query) {
    for (std::int64_t rank = 0; rank < k; ++rank) {
      auto row = host_neighbors[static_cast<std::size_t>(query * k + rank)];
      if (row >= n_rows || row % n_queries != static_cast<std::uint32_t>(query)) { return 1; }
    }
  }

  std::cout << "CAGRA reused " << owners.size()
            << " independently owned Roaring allowlists through one prepared filter payload.\\n";
  return 0;
}
