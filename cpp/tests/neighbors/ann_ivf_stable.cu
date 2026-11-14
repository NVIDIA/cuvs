/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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

constexpr int64_t kDim     = 32;
constexpr int64_t kQueries = 500;
constexpr uint32_t kLists  = 64;

// Each of `n_unique` vectors appears `copies` times, so equal distances are common and the selector
// must break many ties. Query `i` is unique vector `i % n_unique`.
struct tied_data {
  int64_t n_unique;
  int64_t n_rows;
  std::vector<float> base;
  std::vector<float> queries;
};

auto make_tied_data(int64_t n_unique, int64_t copies) -> tied_data
{
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> unique(n_unique * kDim);
  for (auto& v : unique) {
    v = dist(rng);
  }
  tied_data out{n_unique, n_unique * copies, {}, {}};
  out.base.reserve(out.n_rows * kDim);
  for (int64_t i = 0; i < n_unique; i++) {
    for (int64_t c = 0; c < copies; c++) {
      out.base.insert(out.base.end(), unique.begin() + i * kDim, unique.begin() + (i + 1) * kDim);
    }
  }
  for (int64_t i = 0; i < kQueries; i++) {
    auto row = unique.begin() + (i % n_unique) * kDim;
    out.queries.insert(out.queries.end(), row, row + kDim);
  }
  return out;
}

auto upload_base(raft::resources const& res, const tied_data& data)
{
  auto d_base = raft::make_device_matrix<float, int64_t>(res, data.n_rows, kDim);
  raft::copy(
    d_base.data_handle(), data.base.data(), data.base.size(), raft::resource::get_cuda_stream(res));
  return d_base;
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

    // A query that appears more than once in a batch must get the same neighbours each time.
    for (int64_t r = data.n_unique; r < kQueries; r++) {
      EXPECT_TRUE(std::equal(first.ids.begin() + r * k,
                             first.ids.begin() + (r + 1) * k,
                             first.ids.begin() + (r % data.n_unique) * k))
        << "k=" << k << " row=" << r << " differs from row " << r % data.n_unique;
    }

    // The stable selector changes which tied candidate wins, not which distances are selected.
    auto plain = search_in_batches(res, data.queries, k, kQueries, with(false));
    EXPECT_EQ(sorted_rows(first.dists, k), sorted_rows(plain.dists, k)) << "k=" << k;
  }
}

void check_ivf_sq(const tied_data& data)
{
  raft::resources res;
  auto d_base = upload_base(res, data);
  ivf_sq::index_params index_params;
  index_params.n_lists = kLists;
  auto index           = ivf_sq::build(res, index_params, raft::make_const_mdspan(d_base.view()));

  check_stable(res, data, [&](bool stable, auto queries, auto neighbors, auto distances) {
    ivf_sq::search_params params;
    params.n_probes = 16;
    params.stable   = stable;
    ivf_sq::search(res, params, index, queries, neighbors, distances);
  });
}

}  // namespace

TEST(AnnIvfStable, IvfSqSameResultForEveryQueryBatch) { check_ivf_sq(make_tied_data(2000, 8)); }

// Fewer unique vectors than lists: the lists share centres, so the coarse search also has ties.
TEST(AnnIvfStable, IvfSqFewerUniqueVectorsThanLists) { check_ivf_sq(make_tied_data(8, 2000)); }

}  // namespace cuvs::neighbors
