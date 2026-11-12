/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_gpipnn.cuh"

#include <random>
#include <tuple>

namespace cuvs::neighbors::experimental::gpipnn {

using AnnGpipnnTestF_U32 = AnnGpipnnTest<float, float, std::uint32_t>;

TEST_P(AnnGpipnnTestF_U32, Recall) { this->testGpipnn(); }
TEST_P(AnnGpipnnTestF_U32, Carving) { this->testCarving(); }
TEST_P(AnnGpipnnTestF_U32, Accumulator) { this->testAccumulator(); }
TEST_P(AnnGpipnnTestF_U32, HalfInput) { this->testHalfInput(); }
TEST_P(AnnGpipnnTestF_U32, IntegerInput) { this->testIntegerInput(); }
TEST_P(AnnGpipnnTestF_U32, CosineScaleInvariance) { this->testCosineScaleInvariance(); }
TEST_P(AnnGpipnnTestF_U32, CosineRecall)
{
  this->testGpipnn(cuvs::distance::DistanceType::CosineExpanded);
}
TEST_P(AnnGpipnnTestF_U32, UnsupportedMetricRejected) { this->testUnsupportedMetricRejected(); }
TEST_P(AnnGpipnnTestF_U32, SelectKBoundEnforced) { this->testSelectKBoundEnforced(); }
TEST_P(AnnGpipnnTestF_U32, Determinism) { this->testDeterminism(); }

INSTANTIATE_TEST_CASE_P(AnnGpipnnTest, AnnGpipnnTestF_U32, ::testing::ValuesIn(inputs));

/** A separate, smaller instantiation: the fp32 reference here is O(n^2) on the host. */
using AnnGpipnnPrecisionTestF_U32 = AnnGpipnnTest<float, float, std::uint32_t>;

TEST_P(AnnGpipnnPrecisionTestF_U32, Fp16Precision) { this->testFp16Precision(); }

const std::vector<AnnGpipnnInputs> precision_inputs = {
  {512, 32, 32, 64, 32, 512, 10, 10, 64, 0.5, 1.0},
};

INSTANTIATE_TEST_CASE_P(AnnGpipnnPrecision,
                        AnnGpipnnPrecisionTestF_U32,
                        ::testing::ValuesIn(precision_inputs));

/**
 * A scale large enough that the leader cap binds and the oversized tail needs the extra
 * fanout-1 rounds, which the 4000-row cases never reach.
 *
 * The search effort is higher here than for the small cases. A PiPNN graph's candidate pool is
 * drawn entirely from within-leaf joins, so it has no long-range edges, and greedy search needs
 * a wider beam as the dataset grows. Measured recall at 100k is 0.77 at itopk 64 and 0.91 at
 * itopk 128, rising to 0.98 by itopk 1024.
 */
using AnnGpipnnLargeTestF_U32 = AnnGpipnnTest<float, float, std::uint32_t>;

TEST_P(AnnGpipnnLargeTestF_U32, Recall) { this->testGpipnn(); }
TEST_P(AnnGpipnnLargeTestF_U32, Carving) { this->testCarving(); }
TEST_P(AnnGpipnnLargeTestF_U32, Accumulator) { this->testAccumulator(); }
TEST_P(AnnGpipnnLargeTestF_U32, EdgeLengths) { this->testEdgeLengths(); }
// The reservoirs only overflow the output degree at this scale, which is the regime where the
// prune's occlusion rounds actually have to choose between candidates.
TEST_P(AnnGpipnnLargeTestF_U32, Determinism) { this->testDeterminism(); }

const std::vector<AnnGpipnnInputs> large_inputs = {
  {100000, 64, 64, 128, 64, 512, 1000, 10, 128, 0.85, 1.0},
};

INSTANTIATE_TEST_CASE_P(AnnGpipnnLarge, AnnGpipnnLargeTestF_U32, ::testing::ValuesIn(large_inputs));

/**
 * Data at a realistic dynamic range.
 *
 * Every other case here draws from N(0.1, 2), whose squared distances are a few thousand at
 * most, so the whole suite ran comfortably inside the scoring path's finite score range and
 * never tested what happens at the top of it. Real datasets are nowhere near that modest: SIFT
 * has integer components up to 255 in 128 dimensions, giving typical squared distances of ~1e5.
 * Scaling the generated data by 100 puts components in the hundreds and squared distances around
 * 1e7, two orders of magnitude past the largest finite half, which is where the score matrix
 * lives.
 *
 * The two assertions that matter are the ones a saturating score matrix violates outright:
 * carving must place every point in at least one leaf, and the HashPrune reservoirs must come
 * back deep enough to prune from. Both collapse when distances at the ceiling become
 * indistinguishable from the padding that shares that value; 4000 rows is ample to show it,
 * since the effect depends on the magnitude of the data and not on its size.
 */
using AnnGpipnnWideRangeTestF_U32 = AnnGpipnnTest<float, float, std::uint32_t>;

TEST_P(AnnGpipnnWideRangeTestF_U32, Carving) { this->testCarving(); }
TEST_P(AnnGpipnnWideRangeTestF_U32, Accumulator) { this->testAccumulator(); }
TEST_P(AnnGpipnnWideRangeTestF_U32, Recall) { this->testGpipnn(); }

const std::vector<AnnGpipnnInputs> wide_range_inputs = {
  {4000, 128, 64, 128, 32, 512, 100, 10, 64, 0.9, 100.0},
};

INSTANTIATE_TEST_CASE_P(AnnGpipnnWideRange,
                        AnnGpipnnWideRangeTestF_U32,
                        ::testing::ValuesIn(wide_range_inputs));

/**
 * Undersized runs merge with their neighbours but never across a top-level bucket.
 *
 * The sizes are chosen so the unrestricted merge would run past the boundary: accumulating from
 * the first run only reaches `min_size` after taking three of them, and the third belongs to the
 * next bucket.
 */
TEST(AnnGpipnnPartition, MergeStaysInsideOneBucket)
{
  raft::resources res;
  detail::runs r(raft::resource::get_cuda_stream(res));
  detail::set_layout(res,
                     r,
                     /* offsets */ {0, 4, 8, 12, 16, 20},
                     /* bucket  */ {0, 0, 1, 1, 1});

  int64_t merged = 0;
  detail::merge_undersized(res, r, /* min_size */ 10, /* max_size */ 100, merged);

  ASSERT_EQ(r.count(), 2) << "the two buckets should stay apart";
  EXPECT_EQ(r.size_of(0), 8);   // both runs of bucket 0, merged
  EXPECT_EQ(r.size_of(1), 12);  // all three of bucket 1, merged
  EXPECT_EQ(r.h_bucket[0], 0);
  EXPECT_EQ(r.h_bucket[1], 1);
  EXPECT_EQ(merged, 5);
}

/**
 * A `k` below `kFusedK` must return exactly the leading `k` of what `kFusedK` returns, and the
 * fused kernel must pick the same neighbours as the batched path.
 *
 * Two comparisons, because they can hold to different tolerances. The truncation is the stricter
 * one: it is the same kernel over the same accumulators with only the emit loop bound changed, so
 * nothing about it may move. The cross-path one cannot ask for that much. `segmented_topk` routes
 * on whether the two id arrays are one array, so the reference comes from handing it two identical
 * copies -- but that also drops the batched path from its symmetric GEMM to the general one, and
 * the different accumulation order moves the distances by an ulp. The selection itself is what has
 * to agree.
 */
TEST(AnnGpipnnFusedLeaf, MatchesBatchedPathAtEveryFusedK)
{
  if (!is_supported()) { GTEST_SKIP(); }
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  constexpr int64_t kRows    = 512;
  constexpr int64_t kDim     = 32;  // a multiple of 16, so the fused kernel is eligible
  constexpr int64_t kSegRows = 128;

  std::vector<half> h_data(kRows * kDim);
  std::vector<float> h_norms(kRows, 0.0f);
  std::mt19937 rng(42);
  std::normal_distribution<float> gauss(0.1f, 2.0f);
  for (int64_t r = 0; r < kRows; r++) {
    for (int64_t c = 0; c < kDim; c++) {
      const half v         = __float2half(gauss(rng));
      h_data[r * kDim + c] = v;
      h_norms[r] += __half2float(v) * __half2float(v);
    }
  }
  std::vector<detail::id_t> h_ids(kRows);
  std::iota(h_ids.begin(), h_ids.end(), detail::id_t{0});

  auto data  = raft::make_device_matrix<half, int64_t>(res, kRows, kDim);
  auto norms = raft::make_device_vector<float, int64_t>(res, kRows);
  auto ids   = raft::make_device_vector<detail::id_t, int64_t>(res, kRows);
  auto ids2  = raft::make_device_vector<detail::id_t, int64_t>(res, kRows);
  raft::update_device(data.data_handle(), h_data.data(), h_data.size(), stream);
  raft::update_device(norms.data_handle(), h_norms.data(), h_norms.size(), stream);
  raft::update_device(ids.data_handle(), h_ids.data(), h_ids.size(), stream);
  raft::update_device(ids2.data_handle(), h_ids.data(), h_ids.size(), stream);

  std::vector<detail::segment> segments;
  for (int64_t b = 0; b < kRows; b += kSegRows) {
    segments.push_back(detail::segment{b, kSegRows, b, kSegRows});
  }

  // The one argument that differs between the two paths is `cand_ids`: the same array fuses, a
  // copy does not.
  auto run = [&](raft::device_vector_view<const detail::id_t, int64_t> cand, int64_t k) {
    auto out_ids   = raft::make_device_matrix<detail::id_t, int64_t>(res, kRows, k);
    auto out_dists = raft::make_device_matrix<float, int64_t>(res, kRows, k);
    auto out_slots = raft::make_device_matrix<detail::id_t, int64_t>(res, kRows, k);
    detail::segmented_topk(res,
                           raft::make_const_mdspan(data.view()),
                           raft::make_const_mdspan(norms.view()),
                           raft::make_const_mdspan(ids.view()),
                           cand,
                           segments,
                           k,
                           /* exclude_self */ true,
                           out_ids.view(),
                           out_dists.view(),
                           out_slots.view());
    std::vector<detail::id_t> h_out(kRows * k);
    std::vector<float> h_dist(kRows * k);
    std::vector<detail::id_t> h_slot(kRows * k);
    raft::update_host(h_out.data(), out_ids.data_handle(), h_out.size(), stream);
    raft::update_host(h_dist.data(), out_dists.data_handle(), h_dist.size(), stream);
    raft::update_host(h_slot.data(), out_slots.data_handle(), h_slot.size(), stream);
    raft::resource::sync_stream(res);
    return std::make_tuple(h_out, h_dist, h_slot);
  };

  const auto widest = run(raft::make_const_mdspan(ids.view()), detail::kFusedK);

  for (int64_t k = 1; k <= detail::kFusedK; k++) {
    const auto plan =
      detail::plan_fused_segments(data.data_handle(), kDim, k, /* same_ids */ true, segments);
    ASSERT_EQ(plan.fused.size(), segments.size()) << "k=" << k << " never reached the fused kernel";

    const auto fused   = run(raft::make_const_mdspan(ids.view()), k);
    const auto batched = run(raft::make_const_mdspan(ids2.view()), k);

    for (int64_t r = 0; r < kRows; r++) {
      for (int64_t j = 0; j < k; j++) {
        const int64_t got  = r * k + j;
        const int64_t wide = r * detail::kFusedK + j;
        ASSERT_EQ(std::get<0>(fused)[got], std::get<0>(widest)[wide])
          << "k=" << k << " row " << r << " rank " << j << " is not kFusedK truncated";
        ASSERT_EQ(std::get<1>(fused)[got], std::get<1>(widest)[wide]) << "k=" << k;
        ASSERT_EQ(std::get<2>(fused)[got], std::get<2>(widest)[wide]) << "k=" << k;

        ASSERT_EQ(std::get<0>(fused)[got], std::get<0>(batched)[got])
          << "k=" << k << " row " << r << " rank " << j << " differs from the batched path";
        ASSERT_EQ(std::get<2>(fused)[got], std::get<2>(batched)[got]) << "k=" << k;
        ASSERT_NEAR(std::get<1>(fused)[got],
                    std::get<1>(batched)[got],
                    1e-4f * std::abs(std::get<1>(batched)[got]))
          << "k=" << k;
      }
    }
  }
}

TEST(AnnGpipnnValidation, RejectsOddDimension)
{
  if (!is_supported()) { GTEST_SKIP(); }
  raft::resources res;
  const int64_t n_rows = 256;
  const int64_t dim    = 33;

  rmm::device_uvector<float> data(n_rows * dim, raft::resource::get_cuda_stream(res));
  auto graph = raft::make_device_matrix<uint32_t, int64_t>(res, n_rows, 32);

  index_params params;
  params.graph_degree = 32;

  // An odd dimension silently loses tensor cores in the CUTLASS half path, so it is rejected
  // rather than accepted at a large and invisible cost.
  EXPECT_THROW(
    build_graph(res,
                params,
                raft::make_device_matrix_view<const float, int64_t>(data.data(), n_rows, dim),
                graph.view()),
    raft::exception);
}

}  // namespace cuvs::neighbors::experimental::gpipnn
