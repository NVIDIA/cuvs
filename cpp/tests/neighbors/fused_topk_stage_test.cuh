// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Included after the guarded typed/public helpers in fused_topk_tile.cu.
static_assert(!fused_topk_separate_refinement_eligible<2, half>(63, 64));
static_assert(!fused_topk_separate_refinement_eligible<2, half>(64, 64));
static_assert(fused_topk_separate_refinement_eligible<2, half>(65, 64));
static_assert(fused_topk_separate_refinement_eligible<2, half>(72, 64));
static_assert(!fused_topk_separate_refinement_eligible<2, half>(128, 1));
static_assert(fused_topk_separate_refinement_eligible<4, half>(32, 1));
static_assert(fused_topk_separate_refinement_eligible<4, half>(32, 64));
static_assert(!fused_topk_separate_refinement_eligible<2, float>(128, 1));
static_assert(!fused_topk_separate_refinement_eligible<4, float>(128, 1));
static_assert(fused_topk_separate_refinement_eligible<2, float>(32, 64));
static_assert(fused_topk_separate_refinement_eligible<4, float>(32, 64));
static_assert(!fused_topk_separate_refinement_eligible<4, half>(0, 64));
static_assert(!fused_topk_separate_refinement_eligible<4, half>(128, 0));

template <int PhysicalK, typename DataT>
void check_staged_boundaries(int logical_k, int metric_id, int dimensions = 32)
{
  constexpr int corpus_rows = 16513;  // 130 tiles: permits the S129 generic merge.
  const auto metric         = static_cast<cuvs::distance::DistanceType>(metric_id);
  std::vector<float> corpus(corpus_rows * dimensions);
  for (int row = 0; row < corpus_rows; ++row) {
    for (int col = 0; col < dimensions; ++col) {
      corpus[row * dimensions + col] = float((row * 13 + col * 7) % 41 - 20) / 16;
    }
  }
  // Pairwise coverage of both query-tile tails and every merge-family boundary.
  // This is deliberately not a new performance matrix.
  for (const auto& [rows, splits] : {std::pair{1, 1},
                                     {16, 2},
                                     {31, 3},
                                     {32, 4},
                                     {33, 8},
                                     {63, 16},
                                     {64, 17},
                                     {65, 64},
                                     {33, 128},
                                     {65, 129}}) {
    SCOPED_TRACE(::testing::Message() << "K=" << logical_k << " rows=" << rows
                                      << " splits=" << splits << " metric=" << metric_id);
    std::vector<float> queries(rows * dimensions);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < dimensions; ++col) {
        queries[row * dimensions + col] = float((row * 11 + col * 3 + 5) % 43 - 21) / 16;
      }
    }
    const auto narrow = check_typed_topk<PhysicalK, DataT, int>(
      queries, corpus, dimensions, metric, splits, logical_k, true);
    const auto wide = check_typed_topk<PhysicalK, DataT, int64_t>(
      queries, corpus, dimensions, metric, splits, logical_k, true);
    EXPECT_EQ(narrow.indices, wide.indices);
    EXPECT_EQ(narrow.distances, wide.distances);
  }
}

class FusedTopkStagedBoundaryTest : public ::testing::TestWithParam<std::tuple<int, int, bool>> {};

TEST_P(FusedTopkStagedBoundaryTest, QueryTailsMergeFamiliesAndIndexWidths)
{
  // Managed native integration gate must require enabled SM120 exports rather
  // than silently treating an unavailable staged backend as a test pass.
  if (!is_fused_topk_pk2_auto_dispatch_available()) {
    GTEST_SKIP() << "staged boundary tests target the SM120 backend";
  }
  const auto [k, metric, half_input] = GetParam();
  if (k == 2) {
    if (half_input) {
      check_staged_boundaries<2, half>(k, metric);
      // Exercise the staged side too: 72 is the first aligned half width
      // above the broad cutoff, including padded public D65.
      check_staged_boundaries<2, half>(k, metric, 72);
    } else
      check_staged_boundaries<2, float>(k, metric);
  } else {
    if (half_input)
      check_staged_boundaries<4, half>(k, metric);
    else
      check_staged_boundaries<4, float>(k, metric);
  }
}

INSTANTIATE_TEST_SUITE_P(StagedBoundaries,
                         FusedTopkStagedBoundaryTest,
                         ::testing::Combine(::testing::Values(2, 3, 4),
                                            ::testing::Values(0, 1, 2, 6),
                                            ::testing::Bool()));

TEST(FusedTopkPk24PublicTest, SamePointerOddDimensionStagedGraph)
{
  // independent=false passes the identical device pointer as query and corpus.
  // D513 tests strict-layout padding; small N keeps this a correctness check.
  for (int metric : {0, 1, 2, 6}) {
    check_public_typed_batch<2, float>(2, metric, false, 513, 257, 257);
    check_public_typed_batch<2, half>(2, metric, false, 513, 257, 257);
    for (int k : {3, 4}) {
      check_public_typed_batch<4, float>(k, metric, false, 513, 257, 257);
      check_public_typed_batch<4, half>(k, metric, false, 513, 257, 257);
    }
  }
}
