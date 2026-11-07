/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include "refine_helper.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/refine.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/itertools.hpp>

#include <cuda/stream>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <thread>
#include <type_traits>
#include <vector>

namespace cuvs::neighbors {

template <typename DataT, typename DistanceT, typename IdxT>
class RefineTest : public ::testing::TestWithParam<RefineInputs<IdxT>> {
 public:
  RefineTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      data(handle_, ::testing::TestWithParam<RefineInputs<IdxT>>::GetParam())
  {
  }

 protected:
 public:  // tamas remove
  void testRefine()
  {
    std::vector<IdxT> indices(data.p.n_queries * data.p.k);
    std::vector<DistanceT> distances(data.p.n_queries * data.p.k);

    if (data.p.host_data) {
      cuvs::neighbors::refine(handle_,
                              data.dataset_host.view(),
                              data.queries_host.view(),
                              data.candidates_host.view(),
                              data.refined_indices_host.view(),
                              data.refined_distances_host.view(),
                              data.p.metric);
      raft::copy(indices.data(),
                 data.refined_indices_host.data_handle(),
                 data.refined_indices_host.size(),
                 stream_);
      raft::copy(distances.data(),
                 data.refined_distances_host.data_handle(),
                 data.refined_distances_host.size(),
                 stream_);

    } else {
      cuvs::neighbors::refine(handle_,
                              data.dataset.view(),
                              data.queries.view(),
                              data.candidates.view(),
                              data.refined_indices.view(),
                              data.refined_distances.view(),
                              data.p.metric);
      raft::update_host(distances.data(),
                        data.refined_distances.data_handle(),
                        data.refined_distances.size(),
                        stream_);
      raft::update_host(
        indices.data(), data.refined_indices.data_handle(), data.refined_indices.size(), stream_);
    }
    raft::resource::sync_stream(handle_);

    double min_recall = 1;

    ASSERT_TRUE(cuvs::neighbors::eval_neighbours(data.true_refined_indices_host,
                                                 indices,
                                                 data.true_refined_distances_host,
                                                 distances,
                                                 data.p.n_queries,
                                                 data.p.k,
                                                 0.001,
                                                 min_recall));
  }

 public:
  raft::resources handle_;
  cuda::stream_ref stream_;
  RefineHelper<DataT, DistanceT, IdxT> data;
};

const std::vector<RefineInputs<int64_t>> inputs =
  raft::util::itertools::product<RefineInputs<int64_t>>(
    {static_cast<int64_t>(137)},
    {static_cast<int64_t>(1000)},
    {static_cast<int64_t>(16)},
    {static_cast<int64_t>(1), static_cast<int64_t>(10), static_cast<int64_t>(33)},
    {static_cast<int64_t>(33)},
    {cuvs::distance::DistanceType::L2Expanded, cuvs::distance::DistanceType::InnerProduct},
    {false, true});

typedef RefineTest<float, float, std::int64_t> RefineTestF;
TEST_P(RefineTestF, AnnRefine) { this->testRefine(); }

INSTANTIATE_TEST_CASE_P(RefineTest, RefineTestF, ::testing::ValuesIn(inputs));

typedef RefineTest<uint8_t, float, std::int64_t> RefineTestF_uint8;
TEST_P(RefineTestF_uint8, AnnRefine) { this->testRefine(); }
INSTANTIATE_TEST_CASE_P(RefineTest, RefineTestF_uint8, ::testing::ValuesIn(inputs));

typedef RefineTest<int8_t, float, std::int64_t> RefineTestF_int8;
TEST_P(RefineTestF_int8, AnnRefine) { this->testRefine(); }
INSTANTIATE_TEST_CASE_P(RefineTest, RefineTestF_int8, ::testing::ValuesIn(inputs));

namespace {

// Key `refine` ranks candidates by, smaller is closer. It reports the negated key for inner
// product.
template <typename DataT>
auto reference_rank_key(DataT const* a,
                        DataT const* b,
                        int64_t dim,
                        cuvs::distance::DistanceType metric) -> float
{
  double l2 = 0, dot = 0, norm_a = 0, norm_b = 0;
  for (int64_t i = 0; i < dim; i++) {
    auto const x = static_cast<double>(a[i]);
    auto const y = static_cast<double>(b[i]);
    l2 += (x - y) * (x - y);
    dot += x * y;
    norm_a += x * x;
    norm_b += y * y;
  }
  switch (metric) {
    case cuvs::distance::DistanceType::L2Expanded: return static_cast<float>(l2);
    case cuvs::distance::DistanceType::InnerProduct: return static_cast<float>(-dot);
    default: return static_cast<float>(1.0 - dot / std::sqrt(norm_a * norm_b));
  }
}

// `refine` on host views does not touch the device, so this needs no GPU. It takes a separate path
// when there are fewer queries than threads, and all metrics have to give the same answer on both.
template <typename DataT>
void test_refine_host_matches_reference()
{
  constexpr int64_t n_rows = 64;
  constexpr int64_t dim    = 24;
  constexpr int64_t k      = 8;

  std::mt19937 rng(1234);
  auto const random_value = [&rng]() -> DataT {
    auto const value = static_cast<int>(rng() % 201) - 100;
    if constexpr (std::is_same_v<DataT, uint8_t>) {
      return static_cast<DataT>(value + 100);
    } else if constexpr (std::is_same_v<DataT, int8_t>) {
      return static_cast<DataT>(value);
    } else {
      return static_cast<DataT>(static_cast<float>(value) / 10.f);
    }
  };

  auto dataset = raft::make_host_matrix<DataT, int64_t>(n_rows, dim);
  std::generate_n(dataset.data_handle(), dataset.size(), random_value);

  raft::resources handle;
  // One query takes the small-batch path whenever more than one thread is available. More queries
  // than threads always take the other path.
  auto const n_threads = static_cast<int64_t>(std::thread::hardware_concurrency());
  for (int64_t n_queries : {int64_t{1}, n_threads + 1}) {
    auto queries    = raft::make_host_matrix<DataT, int64_t>(n_queries, dim);
    auto candidates = raft::make_host_matrix<int64_t, int64_t>(n_queries, n_rows);
    auto indices    = raft::make_host_matrix<int64_t, int64_t>(n_queries, k);
    auto distances  = raft::make_host_matrix<float, int64_t>(n_queries, k);
    std::generate_n(queries.data_handle(), queries.size(), random_value);
    // In reverse order, so that candidates left unranked do not look ranked.
    for (int64_t q = 0; q < n_queries; q++) {
      for (int64_t c = 0; c < n_rows; c++) {
        candidates(q, c) = n_rows - 1 - c;
      }
    }

    for (auto metric : {cuvs::distance::DistanceType::L2Expanded,
                        cuvs::distance::DistanceType::InnerProduct,
                        cuvs::distance::DistanceType::CosineExpanded}) {
      SCOPED_TRACE(::testing::Message()
                   << "n_queries=" << n_queries << ", metric=" << print_metric{metric});

      cuvs::neighbors::refine(handle,
                              dataset.view(),
                              queries.view(),
                              candidates.view(),
                              indices.view(),
                              distances.view(),
                              metric);

      float const sign = metric == cuvs::distance::DistanceType::InnerProduct ? -1.f : 1.f;
      for (int64_t q = 0; q < n_queries; q++) {
        std::vector<float> expected(n_rows);
        for (int64_t c = 0; c < n_rows; c++) {
          expected[c] = reference_rank_key(&queries(q, 0), &dataset(c, 0), dim, metric);
        }
        std::sort(expected.begin(), expected.end());
        for (int64_t j = 0; j < k; j++) {
          auto const tolerance = 1e-4f * (1.f + std::abs(expected[j]));
          ASSERT_GE(indices(q, j), 0);
          ASSERT_LT(indices(q, j), n_rows);
          EXPECT_NEAR(reference_rank_key(&queries(q, 0), &dataset(indices(q, j), 0), dim, metric),
                      expected[j],
                      tolerance)
            << "id " << indices(q, j) << " at query " << q << ", rank " << j;
          EXPECT_NEAR(distances(q, j), sign * expected[j], tolerance)
            << "distance at query " << q << ", rank " << j;
        }
      }
    }
  }
}

}  // namespace

TEST(RefineHost, MatchesReferenceFloat) { test_refine_host_matches_reference<float>(); }
TEST(RefineHost, MatchesReferenceHalf) { test_refine_host_matches_reference<half>(); }
TEST(RefineHost, MatchesReferenceInt8) { test_refine_host_matches_reference<int8_t>(); }
TEST(RefineHost, MatchesReferenceUint8) { test_refine_host_matches_reference<uint8_t>(); }

}  // namespace cuvs::neighbors
