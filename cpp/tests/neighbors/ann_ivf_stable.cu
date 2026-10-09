/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/ivf_sq.hpp>
#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace cuvs::neighbors {

namespace {

constexpr int64_t kUnique  = 2000;
constexpr int64_t kCopies  = 8;
constexpr int64_t kDim     = 32;
constexpr int64_t kQueries = 500;

// Every vector appears kCopies times, so equal distances are common and the selector must break
// many ties. The queries are the first kQueries distinct vectors.
struct tied_data {
  std::vector<float> base;
  std::vector<float> queries;
};

auto make_tied_data() -> tied_data
{
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> unique(kUnique * kDim);
  for (auto& v : unique) {
    v = dist(rng);
  }
  tied_data out;
  out.base.reserve(kUnique * kCopies * kDim);
  for (int64_t i = 0; i < kUnique; i++) {
    for (int64_t c = 0; c < kCopies; c++) {
      out.base.insert(out.base.end(), unique.begin() + i * kDim, unique.begin() + (i + 1) * kDim);
    }
  }
  out.queries.assign(unique.begin(), unique.begin() + kQueries * kDim);
  return out;
}

struct result {
  std::vector<int64_t> ids;
  std::vector<float> dists;
};

// Searches the queries in batches of `batch` rows.
template <typename SearchFn>
auto search_in_batches(raft::resources const& res,
                       const std::vector<float>& queries,
                       int64_t k,
                       int64_t batch,
                       SearchFn search) -> result
{
  auto stream = raft::resource::get_cuda_stream(res);
  auto d_q    = raft::make_device_matrix<float, int64_t>(res, kQueries, kDim);
  auto d_ids  = raft::make_device_matrix<int64_t, int64_t>(res, kQueries, k);
  auto d_dist = raft::make_device_matrix<float, int64_t>(res, kQueries, k);
  raft::copy(d_q.data_handle(), queries.data(), queries.size(), stream);
  for (int64_t start = 0; start < kQueries; start += batch) {
    int64_t n = std::min(batch, kQueries - start);
    search(raft::make_device_matrix_view<const float, int64_t>(
             d_q.data_handle() + start * kDim, n, kDim),
           raft::make_device_matrix_view<int64_t, int64_t>(d_ids.data_handle() + start * k, n, k),
           raft::make_device_matrix_view<float, int64_t>(d_dist.data_handle() + start * k, n, k));
  }
  result out{std::vector<int64_t>(kQueries * k), std::vector<float>(kQueries * k)};
  raft::copy(out.ids.data(), d_ids.data_handle(), out.ids.size(), stream);
  raft::copy(out.dists.data(), d_dist.data_handle(), out.dists.size(), stream);
  raft::resource::sync_stream(res, stream);
  return out;
}

auto sorted_rows(std::vector<float> v, int64_t k) -> std::vector<float>
{
  for (int64_t r = 0; r < kQueries; r++) {
    std::sort(v.begin() + r * k, v.begin() + (r + 1) * k);
  }
  return v;
}

// `search(stable, queries, neighbors, distances)` runs one search of the index under test.
template <typename SearchFn>
void check_stable(raft::resources const& res, const tied_data& data, SearchFn search)
{
  for (int64_t k : {10, 100, 300}) {
    auto with = [&](bool stable) {
      return [&, stable](auto q, auto n, auto d) { search(stable, q, n, d); };
    };
    auto first = search_in_batches(res, data.queries, k, kQueries, with(true));
    for (int64_t batch : {kQueries, int64_t{7}, int64_t{128}}) {
      auto again = search_in_batches(res, data.queries, k, batch, with(true));
      EXPECT_EQ(first.ids, again.ids) << "k=" << k << " batch=" << batch;
      EXPECT_EQ(first.dists, again.dists) << "k=" << k << " batch=" << batch;
    }

    for (int64_t r = 0; r < kQueries; r++) {
      std::vector<int64_t> row(first.ids.begin() + r * k, first.ids.begin() + (r + 1) * k);
      std::sort(row.begin(), row.end());
      EXPECT_EQ(std::adjacent_find(row.begin(), row.end()), row.end())
        << "k=" << k << " row=" << r << " has a repeated id";
    }

    // The stable selector changes which tied candidate wins, not which distances are selected.
    auto plain = search_in_batches(res, data.queries, k, kQueries, with(false));
    EXPECT_EQ(sorted_rows(first.dists, k), sorted_rows(plain.dists, k)) << "k=" << k;
  }
}

}  // namespace

TEST(AnnIvfStable, IvfPqSameResultForEveryQueryBatch)
{
  raft::resources res;
  auto data   = make_tied_data();
  auto stream = raft::resource::get_cuda_stream(res);
  auto d_base = raft::make_device_matrix<float, int64_t>(res, kUnique * kCopies, kDim);
  raft::copy(d_base.data_handle(), data.base.data(), data.base.size(), stream);

  ivf_pq::index_params index_params;
  index_params.n_lists = 64;
  index_params.pq_dim  = 8;
  index_params.pq_bits = 8;
  auto index           = ivf_pq::build(res, index_params, raft::make_const_mdspan(d_base.view()));

  check_stable(res, data, [&](bool stable, auto queries, auto neighbors, auto distances) {
    ivf_pq::search_params params;
    params.n_probes = 16;
    params.stable   = stable;
    ivf_pq::search(res, params, index, queries, neighbors, distances);
  });
}

TEST(AnnIvfStable, IvfSqSameResultForEveryQueryBatch)
{
  raft::resources res;
  auto data   = make_tied_data();
  auto stream = raft::resource::get_cuda_stream(res);
  auto d_base = raft::make_device_matrix<float, int64_t>(res, kUnique * kCopies, kDim);
  raft::copy(d_base.data_handle(), data.base.data(), data.base.size(), stream);

  ivf_sq::index_params index_params;
  index_params.n_lists = 64;
  auto index           = ivf_sq::build(res, index_params, raft::make_const_mdspan(d_base.view()));

  check_stable(res, data, [&](bool stable, auto queries, auto neighbors, auto distances) {
    ivf_sq::search_params params;
    params.n_probes = 16;
    params.stable   = stable;
    ivf_sq::search(res, params, index, queries, neighbors, distances);
  });
}

}  // namespace cuvs::neighbors
