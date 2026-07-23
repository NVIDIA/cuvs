/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/roaring.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/roaring_filter.hpp>

#include <raft/core/bitset.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace cuvs::neighbors {

namespace {

struct RoaringFilterInputs {
  int64_t n_rows;
  int64_t dim;
  int64_t n_queries;
  int64_t k;
  double selectivity;  // chosen to exercise sparse / mid / dense dispatch
  cuvs::distance::DistanceType metric;
};

std::ostream& operator<<(std::ostream& os, const RoaringFilterInputs& p)
{
  return os << "n_rows=" << p.n_rows << " dim=" << p.dim << " q=" << p.n_queries << " k=" << p.k
            << " s=" << p.selectivity << " metric=" << (int)p.metric;
}

std::vector<uint32_t> make_filter_ids(std::mt19937& rng, int64_t n_rows, double s)
{
  std::vector<uint32_t> ids;
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int64_t i = 0; i < n_rows; ++i)
    if (u(rng) < s) ids.push_back(static_cast<uint32_t>(i));
  if (ids.size() < 32) {  // keep k satisfiable
    for (uint32_t i = 0; ids.size() < 32; i += 7)
      ids.push_back(i % static_cast<uint32_t>(n_rows));
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  }
  return ids;
}

class RoaringFilterTest : public ::testing::TestWithParam<RoaringFilterInputs> {
 protected:
  void SetUp() override
  {
    auto p      = GetParam();
    auto stream = raft::resource::get_cuda_stream(res_);

    dataset_.emplace(raft::make_device_matrix<float, int64_t>(res_, p.n_rows, p.dim));
    queries_.emplace(raft::make_device_matrix<float, int64_t>(res_, p.n_queries, p.dim));
    raft::random::RngState r1(42), r2(43);
    raft::random::uniform(
      res_,
      r1,
      raft::make_device_vector_view<float, int64_t>(dataset_->data_handle(), p.n_rows * p.dim),
      -1.0f,
      1.0f);
    raft::random::uniform(
      res_,
      r2,
      raft::make_device_vector_view<float, int64_t>(queries_->data_handle(), p.n_queries * p.dim),
      -1.0f,
      1.0f);

    brute_force::index_params ip;
    ip.metric = p.metric;
    index_.emplace(brute_force::build(res_, ip, raft::make_const_mdspan(dataset_->view())));
    raft::resource::sync_stream(res_);
  }

  // run a search with the given filter into fresh buffers
  std::pair<std::vector<int64_t>, std::vector<float>> run(const filtering::base_filter& filter)
  {
    auto p  = GetParam();
    auto nb = raft::make_device_matrix<int64_t, int64_t>(res_, p.n_queries, p.k);
    auto di = raft::make_device_matrix<float, int64_t>(res_, p.n_queries, p.k);
    brute_force::search(res_,
                        brute_force::search_params{},
                        *index_,
                        raft::make_const_mdspan(queries_->view()),
                        nb.view(),
                        di.view(),
                        filter);
    std::vector<int64_t> h_nb(p.n_queries * p.k);
    std::vector<float> h_di(p.n_queries * p.k);
    raft::update_host(
      h_nb.data(), nb.data_handle(), h_nb.size(), raft::resource::get_cuda_stream(res_));
    raft::update_host(
      h_di.data(), di.data_handle(), h_di.size(), raft::resource::get_cuda_stream(res_));
    raft::resource::sync_stream(res_);
    return {std::move(h_nb), std::move(h_di)};
  }

  // tie-tolerant: sorted per-query distance lists must match elementwise
  static void expect_same_distances(const std::vector<float>& a,
                                    const std::vector<float>& b,
                                    int64_t n_queries,
                                    int64_t k)
  {
    for (int64_t q = 0; q < n_queries; ++q) {
      std::vector<float> da(a.begin() + q * k, a.begin() + (q + 1) * k);
      std::vector<float> db(b.begin() + q * k, b.begin() + (q + 1) * k);
      std::sort(da.begin(), da.end());
      std::sort(db.begin(), db.end());
      for (int64_t j = 0; j < k; ++j) {
        ASSERT_NEAR(da[j], db[j], 1e-3 + 1e-3 * std::abs(da[j])) << "query " << q << " rank " << j;
      }
    }
  }

  raft::resources res_;
  std::optional<raft::device_matrix<float, int64_t>> dataset_;
  std::optional<raft::device_matrix<float, int64_t>> queries_;
  std::optional<brute_force::index<float, float>> index_;
};

TEST_P(RoaringFilterTest, MatchesBitsetFilter)
{
  auto p      = GetParam();
  auto stream = raft::resource::get_cuda_stream(res_);
  std::mt19937 rng(123);
  auto ids = make_filter_ids(rng, p.n_rows, p.selectivity);

  // roaring filter
  auto roaring =
    cuvs::core::from_sorted_ids(res_, ids.data(), (uint32_t)ids.size(), (uint32_t)p.n_rows);
  auto rf = filtering::roaring_filter(roaring);

  // reference: equivalent bitset filter
  raft::core::bitset<uint32_t, int64_t> bits(res_, p.n_rows, false);
  std::vector<uint32_t> h_words((p.n_rows + 31) / 32, 0);
  for (auto id : ids)
    h_words[id / 32] |= 1u << (id % 32);
  raft::update_device(bits.data(), h_words.data(), h_words.size(), stream);
  auto bf = filtering::bitset_filter<uint32_t, int64_t>(bits.view());

  auto [nb_r, di_r] = run(rf);
  auto [nb_b, di_b] = run(bf);

  expect_same_distances(di_r, di_b, p.n_queries, p.k);
  // membership: every returned id must be in the filter
  for (auto id : nb_r) {
    ASSERT_TRUE(id >= 0 && std::binary_search(ids.begin(), ids.end(), (uint32_t)id))
      << "non-member id " << id;
  }
}

TEST_P(RoaringFilterTest, MatrixMatchesBitmapFilter)
{
  auto p      = GetParam();
  auto stream = raft::resource::get_cuda_stream(res_);
  std::mt19937 rng(321);

  // one filter per query (share a few to exercise repeated bitmaps)
  std::vector<std::vector<uint32_t>> per_query(p.n_queries);
  std::vector<cuvs::core::gpu_roaring> bitmaps;
  bitmaps.reserve(p.n_queries);
  std::vector<const cuvs::core::gpu_roaring*> ptrs(p.n_queries);
  for (int64_t q = 0; q < p.n_queries; ++q) {
    per_query[q] = make_filter_ids(rng, p.n_rows, p.selectivity);
    bitmaps.emplace_back(cuvs::core::from_sorted_ids(
      res_, per_query[q].data(), (uint32_t)per_query[q].size(), (uint32_t)p.n_rows));
    ptrs[q] = &bitmaps[q];
  }
  auto rmf = filtering::roaring_matrix_filter(ptrs.data(), (uint32_t)p.n_queries);

  // reference: dense [n_queries, n_rows] bitmap filter
  int64_t total_bits = p.n_queries * p.n_rows;
  rmm::device_uvector<uint32_t> words((total_bits + 31) / 32, stream);
  std::vector<uint32_t> h_words((total_bits + 31) / 32, 0);
  for (int64_t q = 0; q < p.n_queries; ++q)
    for (auto id : per_query[q]) {
      int64_t bit = q * p.n_rows + id;
      h_words[bit / 32] |= 1u << (bit % 32);
    }
  raft::update_device(words.data(), h_words.data(), h_words.size(), stream);
  auto bview = cuvs::core::bitmap_view<uint32_t, int64_t>(words.data(), p.n_queries, p.n_rows);
  auto bmf   = filtering::bitmap_filter<uint32_t, int64_t>(bview);

  auto [nb_r, di_r] = run(rmf);
  auto [nb_b, di_b] = run(bmf);

  expect_same_distances(di_r, di_b, p.n_queries, p.k);
  for (int64_t q = 0; q < p.n_queries; ++q)
    for (int64_t j = 0; j < p.k; ++j) {
      int64_t id = nb_r[q * p.k + j];
      ASSERT_TRUE(id >= 0 &&
                  std::binary_search(per_query[q].begin(), per_query[q].end(), (uint32_t)id))
        << "query " << q << " non-member id " << id;
    }
}

// selectivities chosen to hit the sparse / gather-GEMM / dense regimes at
// both a low and a high dimension (the sparse/mid threshold is
// dimension-dependent)
const std::vector<RoaringFilterInputs> inputs = {
  // d=64: sparse (<=3%), mid, dense
  {100000, 64, 17, 10, 0.002, cuvs::distance::DistanceType::InnerProduct},
  {100000, 64, 17, 10, 0.10, cuvs::distance::DistanceType::InnerProduct},
  {100000, 64, 17, 10, 0.60, cuvs::distance::DistanceType::InnerProduct},
  {100000, 64, 17, 10, 0.10, cuvs::distance::DistanceType::L2Expanded},
  {100000, 64, 17, 10, 0.002, cuvs::distance::DistanceType::L2Expanded},
  {100000, 64, 17, 10, 0.60, cuvs::distance::DistanceType::L2Expanded},
  {100000, 64, 17, 10, 0.10, cuvs::distance::DistanceType::CosineExpanded},
  // d=512: sparse (<=0.1%), mid, dense
  {60000, 512, 9, 10, 0.0008, cuvs::distance::DistanceType::InnerProduct},
  {60000, 512, 9, 10, 0.05, cuvs::distance::DistanceType::L2Expanded},
  {60000, 512, 9, 10, 0.55, cuvs::distance::DistanceType::InnerProduct},
};

INSTANTIATE_TEST_SUITE_P(RoaringFilter, RoaringFilterTest, ::testing::ValuesIn(inputs));

}  // namespace
}  // namespace cuvs::neighbors
