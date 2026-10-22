/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_gpipnn.cuh"

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

TEST(AnnGpipnnValidation, RejectsOddDimension)
{
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
