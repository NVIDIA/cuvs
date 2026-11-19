// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Included after the shared typed/public test helpers inside their namespace.

template <typename DataT>
void check_pk8_wide_stage_contract(int k, int metric_id)
{
  const auto metric = static_cast<cuvs::distance::DistanceType>(metric_id);
  for (int dimensions : {512, 520}) {
    for (const auto& [rows, splits] : {std::pair{1, 2}, {5, 3}, {33, 2}}) {
      SCOPED_TRACE(::testing::Message()
                   << "K=" << k << " D=" << dimensions << " Q=" << rows << " S=" << splits);
      std::vector<float> q(rows * dimensions), x(259 * dimensions);
      for (size_t i = 0; i < q.size(); ++i) {
        q[i] = std::sin(i * .071f) * 2;
      }
      for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::cos(i * .053f) * 2;
      }
      const auto a = check_typed_topk<8, DataT, int>(q, x, dimensions, metric, splits, k, true);
      const auto b = check_typed_topk<8, DataT, int64_t>(q, x, dimensions, metric, splits, k, true);
      EXPECT_EQ(a.indices, b.indices);
      EXPECT_EQ(a.distances, b.distances);
    }
  }
  // Public odd widths exercise copy/padding and the real staged route rather
  // than the relaxed-layout fallback of a direct odd-width invocation.
  for (int dimensions : {509, 513}) {
    check_public_typed_batch<8, DataT>(k, metric_id, true, dimensions, 33, 259);
  }
}

class FusedTopkPk8WideStageTest : public ::testing::TestWithParam<std::tuple<int, int, bool>> {};

TEST_P(FusedTopkPk8WideStageTest, MetricsLogicalPrefixesTailsIndexWidthsAndNondefaultStream)
{
  if (!is_fused_topk_pk8_auto_dispatch_available()) {
    GTEST_SKIP() << "PK8 automatic cuTile dispatch is unavailable on this device";
  }
  const auto [k, metric, half_input] = GetParam();
  if (half_input) {
    check_pk8_wide_stage_contract<half>(k, metric);
  } else {
    check_pk8_wide_stage_contract<float>(k, metric);
  }
}

INSTANTIATE_TEST_SUITE_P(WideStages,
                         FusedTopkPk8WideStageTest,
                         ::testing::Combine(::testing::Values(5, 6, 7, 8),
                                            ::testing::Values(0, 1, 2, 6),
                                            ::testing::Bool()));
