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

constexpr std::int64_t n_rows                            = 1024;
constexpr std::int64_t n_dim                             = 16;
constexpr std::int64_t n_queries                         = 4;
constexpr std::int64_t k                                 = 8;
constexpr std::uint32_t portable_cookie_no_run           = 12346;
constexpr std::uint32_t single_array_payload_byte_offset = 16;

void append_u16(std::vector<std::byte>& bytes, std::uint16_t value)
{
  bytes.push_back(static_cast<std::byte>(value & 0xffu));
  bytes.push_back(static_cast<std::byte>((value >> 8) & 0xffu));
}

void append_u32(std::vector<std::byte>& bytes, std::uint32_t value)
{
  for (int i = 0; i < 4; ++i) {
    bytes.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xffu));
  }
}

// This helper intentionally writes the simplest nonempty portable Roaring row:
//
//   bytes  0..3:  uint32 no-run cookie (12346)
//   bytes  4..7:  uint32 container count (1)
//   bytes 8..11:  { uint16 key, uint16 cardinality_minus_one }
//   bytes 12..15: uint32 payload offset, measured from byte 0 of this row
//   bytes 16..:   sorted uint16 array values
//
// Every integer is little-endian. This example's IDs are below 2^16, so they all use container key
// zero; a general encoder must split IDs by their high 16 bits and may need bitmap or run
// containers. Prefer `from_ids` when starting with IDs. `from_serialized` is intended primarily
// for interoperating with systems that already emit the standard format:
//
//   https://github.com/RoaringBitmap/RoaringFormatSpec
void append_portable_array_row(std::vector<std::byte>& bytes,
                               std::vector<std::uint32_t> const& ids,
                               std::int64_t first,
                               std::int64_t last)
{
  append_u32(bytes, portable_cookie_no_run);
  append_u32(bytes, 1);  // one container
  append_u16(bytes, 0);  // high 16-bit container key
  append_u16(bytes, static_cast<std::uint16_t>(last - first - 1));
  append_u32(bytes, single_array_payload_byte_offset);
  for (auto i = first; i < last; ++i) {
    append_u16(bytes, static_cast<std::uint16_t>(ids[static_cast<std::size_t>(i)]));
  }
}

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
  index.update_device_dataset_same_layout(res, padded);

  // Ragged construction: query q accepts rows whose row id modulo n_queries equals q.
  std::vector<std::uint32_t> ids;
  std::vector<std::int64_t> indptr{0};
  for (std::uint32_t query = 0; query < n_queries; ++query) {
    for (std::uint32_t row = query; row < n_rows; row += n_queries) {
      ids.push_back(row);
    }
    indptr.push_back(static_cast<std::int64_t>(ids.size()));
  }
  // One factory call consumes the ragged rows directly and retains every variable-length stream
  // in one owner.
  auto id_allowlists = cuvs::core::roaring_allowlist::from_ids(
    res,
    n_rows,
    raft::make_host_vector_view<const std::uint32_t, std::int64_t>(ids.data(), ids.size()),
    raft::make_host_vector_view<const std::int64_t, std::int64_t>(indptr.data(), indptr.size()),
    true);
  std::vector<cuvs::core::roaring_allowlist_view> id_views;
  for (std::int64_t query = 0; query < n_queries; ++query) {
    id_views.push_back(id_allowlists.view(query));
  }
  cuvs::neighbors::filtering::roaring_filter ids_filter(res, id_views);

  // A serialized system can keep ragged portable rows in one host buffer plus outer offsets. The
  // whole packed buffer becomes one owner even though encoded row sizes may differ.
  std::vector<std::byte> bytes;
  std::vector<std::uint64_t> byte_offsets{0};
  for (std::int64_t query = 0; query < n_queries; ++query) {
    append_portable_array_row(bytes, ids, indptr[query], indptr[query + 1]);
    byte_offsets.push_back(bytes.size());
  }

  auto serialized_allowlists = cuvs::core::roaring_allowlist::from_serialized(
    res,
    n_rows,
    raft::make_host_vector_view<const std::byte, std::int64_t>(bytes.data(), bytes.size()),
    raft::make_host_vector_view<const std::uint64_t, std::int64_t>(byte_offsets.data(),
                                                                   byte_offsets.size()));
  std::vector<cuvs::core::roaring_allowlist_view> serialized_views;
  for (std::int64_t query = 0; query < n_queries; ++query) {
    serialized_views.push_back(serialized_allowlists.view(query));
  }
  cuvs::neighbors::filtering::roaring_filter filter(res, serialized_views);
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

  std::cout << "Built " << id_allowlists.num_allowlists() << " ID allowlists and imported "
            << serialized_allowlists.num_allowlists()
            << " portable allowlists; CAGRA reused one prepared filter payload.\n";
  return ids_filter.num_queries() == n_queries ? 0 : 1;
}
