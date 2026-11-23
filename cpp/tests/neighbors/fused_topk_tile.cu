/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/neighbors/detail/cutile_topk/fused_topk_tile.hpp"
#include "fused_topk_merge_test.cuh"

#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace cuvs::neighbors::detail {
namespace {

bool fp32_topk_test_available()
{
  using namespace cuvs::detail::jit_lto;
  if (!cutile_launch_available_on_current_device()) { return false; }
  CutileRuntimeCapabilities capabilities{};
  if (!query_current_cutile_runtime_capabilities(capabilities)) { return false; }
  // Top-K exports are narrower than the shared Top-1 capability table (no SM107).
  const int arch = 10 * capabilities.cc_major + capabilities.cc_minor;
  return arch == 80 || arch == 86 || arch == 89 || arch == 90 || arch == 100 || arch == 120;
}

struct TopkCase {
  int dimensions;
  cuvs::distance::DistanceType metric;
  int num_splits;
};

class FusedTopkPk2Test : public ::testing::TestWithParam<TopkCase> {};

TEST(FusedTopkPk2PolicyTest, CeilDivisionPreservesInt32Boundary)
{
  constexpr int maximum = std::numeric_limits<int>::max();
  for (int tile : {32, 64, 128, 256}) {
    for (int extent : {1, tile - 1, tile, tile + 1, maximum - 1, maximum}) {
      EXPECT_EQ(fused_topk_pk2_ceil_div(extent, tile),
                (static_cast<int64_t>(extent) + tile - 1) / tile);
    }
  }
  static_assert(fused_topk_pk2_ceil_div(maximum, 256) == 8388608);
}
struct Pk4Case {
  int logical_k;
  int dimensions;
  cuvs::distance::DistanceType metric;
  int num_splits;
};

class FusedTopkPk4Test : public ::testing::TestWithParam<Pk4Case> {};

float reference_value(const float* query,
                      const float* corpus,
                      int dimensions,
                      cuvs::distance::DistanceType metric)
{
  float dot        = 0.0f;
  float l2         = 0.0f;
  float query_norm = 0.0f;
  float row_norm   = 0.0f;
  for (int col = 0; col < dimensions; ++col) {
    const float q = query[col];
    const float x = corpus[col];
    dot += q * x;
    const float delta = q - x;
    l2 += delta * delta;
    query_norm += q * q;
    row_norm += x * x;
  }
  switch (metric) {
    case cuvs::distance::DistanceType::InnerProduct: return dot;
    case cuvs::distance::DistanceType::L2Expanded: return l2;
    case cuvs::distance::DistanceType::L2SqrtExpanded: return std::sqrt(l2);
    case cuvs::distance::DistanceType::CosineExpanded:
      if (query_norm == 0.0f || row_norm == 0.0f) { return 1.0f; }
      return 1.0f - dot / (std::sqrt(query_norm) * std::sqrt(row_norm));
    default: return std::numeric_limits<float>::quiet_NaN();
  }
}

float rank_value(float value, cuvs::distance::DistanceType metric)
{
  return metric == cuvs::distance::DistanceType::InnerProduct ? -value : value;
}

TEST_P(FusedTopkPk2Test, RefinesReturnedCandidatesInFp32AcrossTileBoundaries)
{
  if (!fp32_topk_test_available()) { GTEST_SKIP() << "FP32 Top-K exports are unavailable"; }
  // Cross every seeded architecture's query and corpus tile boundaries.
  constexpr int query_rows                    = 129;
  constexpr int corpus_rows                   = 259;
  constexpr int physical_k                    = 2;
  const auto [dimensions, metric, num_splits] = GetParam();

  std::vector<float> queries(static_cast<size_t>(query_rows) * dimensions);
  std::vector<float> corpus(static_cast<size_t>(corpus_rows) * dimensions);
  for (int row = 0; row < query_rows; ++row) {
    for (int col = 0; col < dimensions; ++col) {
      queries[static_cast<size_t>(row) * dimensions + col] =
        std::sin(0.071f * static_cast<float>((row + 2) * (col + 3))) + 0.013f * row;
    }
  }
  for (int row = 0; row < corpus_rows; ++row) {
    for (int col = 0; col < dimensions; ++col) {
      corpus[static_cast<size_t>(row) * dimensions + col] =
        std::cos(0.053f * static_cast<float>((row + 1) * (col + 5))) + 0.009f * row;
    }
  }

  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);
  rmm::device_uvector<float> queries_d(queries.size(), stream);
  rmm::device_uvector<float> corpus_d(corpus.size(), stream);
  rmm::device_uvector<int64_t> indices_d(static_cast<size_t>(query_rows) * physical_k, stream);
  rmm::device_uvector<uint8_t> workspace_d(
    fused_topk_pk2_workspace_bytes<int64_t>(query_rows, num_splits), stream);
  rmm::device_uvector<float> distances_d(static_cast<size_t>(query_rows) * physical_k, stream);
  raft::copy(queries_d.data(), queries.data(), queries.size(), stream);
  raft::copy(corpus_d.data(), corpus.data(), corpus.size(), stream);

  const bool launched = try_fused_topk_pk2_tile<int64_t>(indices_d.data(),
                                                         distances_d.data(),
                                                         queries_d.data(),
                                                         corpus_d.data(),
                                                         int64_t{query_rows},
                                                         int64_t{corpus_rows},
                                                         static_cast<int64_t>(dimensions),
                                                         num_splits,
                                                         metric,
                                                         stream.get(),
                                                         workspace_d.data());
  ASSERT_TRUE(launched);

  std::vector<int64_t> actual_indices(static_cast<size_t>(query_rows) * physical_k);
  std::vector<float> actual_distances(static_cast<size_t>(query_rows) * physical_k);
  raft::copy(actual_indices.data(), indices_d.data(), actual_indices.size(), stream);
  raft::copy(actual_distances.data(), distances_d.data(), actual_distances.size(), stream);
  raft::resource::sync_stream(resources);

  // TF32 chooses membership; independently verify the promised direct-FP32 value and ordering for
  // whichever candidates it retained.
  for (int query = 0; query < query_rows; ++query) {
    const size_t row_offset = static_cast<size_t>(query) * physical_k;
    ASSERT_GE(actual_indices[row_offset], 0);
    ASSERT_LT(actual_indices[row_offset], corpus_rows);
    ASSERT_GE(actual_indices[row_offset + 1], 0);
    ASSERT_LT(actual_indices[row_offset + 1], corpus_rows);
    ASSERT_NE(actual_indices[row_offset], actual_indices[row_offset + 1]);
    for (int col = 0; col < physical_k; ++col) {
      const auto corpus_row = actual_indices[row_offset + col];
      const float expected =
        reference_value(queries.data() + static_cast<size_t>(query) * dimensions,
                        corpus.data() + static_cast<size_t>(corpus_row) * dimensions,
                        dimensions,
                        metric);
      EXPECT_NEAR(actual_distances[row_offset + col], expected, 2.0e-3f)
        << "query " << query << ", column " << col;
    }
    const auto left =
      std::pair{rank_value(actual_distances[row_offset], metric), actual_indices[row_offset]};
    const auto right = std::pair{rank_value(actual_distances[row_offset + 1], metric),
                                 actual_indices[row_offset + 1]};
    EXPECT_LE(left, right) << "query " << query;
  }
}

TEST_P(FusedTopkPk4Test, RefinesAndCompactsAcrossTileAndSplitBoundaries)
{
  if (!fp32_topk_test_available()) { GTEST_SKIP() << "FP32 Top-K exports are unavailable"; }
  constexpr int query_rows                               = 129;
  constexpr int corpus_rows                              = 259;
  const auto [logical_k, dimensions, metric, num_splits] = GetParam();

  std::vector<float> queries(static_cast<size_t>(query_rows) * dimensions);
  std::vector<float> corpus(static_cast<size_t>(corpus_rows) * dimensions);
  for (int row = 0; row < query_rows; ++row) {
    for (int col = 0; col < dimensions; ++col) {
      queries[static_cast<size_t>(row) * dimensions + col] =
        std::sin(0.071f * static_cast<float>((row + 2) * (col + 3))) + 0.013f * row;
    }
  }
  for (int row = 0; row < corpus_rows; ++row) {
    for (int col = 0; col < dimensions; ++col) {
      corpus[static_cast<size_t>(row) * dimensions + col] =
        std::cos(0.053f * static_cast<float>((row + 1) * (col + 5))) + 0.009f * row;
    }
  }

  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);
  rmm::device_uvector<float> queries_d(queries.size(), stream);
  rmm::device_uvector<float> corpus_d(corpus.size(), stream);
  const size_t output_count = static_cast<size_t>(query_rows) * logical_k;
  constexpr size_t padding  = 4;
  rmm::device_uvector<int64_t> indices_d(output_count + 2 * padding, stream);
  rmm::device_uvector<float> distances_d(output_count + 2 * padding, stream);
  rmm::device_uvector<uint8_t> workspace_d(
    fused_topk_pk4_workspace_bytes<int64_t>(query_rows, num_splits, logical_k), stream);
  raft::copy(queries_d.data(), queries.data(), queries.size(), stream);
  raft::copy(corpus_d.data(), corpus.data(), corpus.size(), stream);
  RAFT_CUDA_TRY(
    cudaMemsetAsync(indices_d.data(), 0x5a, indices_d.size() * sizeof(int64_t), stream.get()));
  const std::vector<float> distance_guards(distances_d.size(), -12345.0f);
  raft::copy(distances_d.data(), distance_guards.data(), distance_guards.size(), stream);

  const bool launched = try_fused_topk_pk4_tile<int64_t>(indices_d.data() + padding,
                                                         distances_d.data() + padding,
                                                         queries_d.data(),
                                                         corpus_d.data(),
                                                         int64_t{query_rows},
                                                         int64_t{corpus_rows},
                                                         static_cast<int64_t>(dimensions),
                                                         static_cast<int64_t>(logical_k),
                                                         num_splits,
                                                         metric,
                                                         stream.get(),
                                                         workspace_d.data());
  ASSERT_TRUE(launched);

  std::vector<int64_t> actual_indices(output_count + 2 * padding);
  std::vector<float> actual_distances(output_count + 2 * padding);
  raft::copy(actual_indices.data(), indices_d.data(), actual_indices.size(), stream);
  raft::copy(actual_distances.data(), distances_d.data(), actual_distances.size(), stream);
  raft::resource::sync_stream(resources);

  constexpr int64_t index_guard = 0x5a5a5a5a5a5a5a5aLL;
  for (size_t guard = 0; guard < padding; ++guard) {
    EXPECT_EQ(actual_indices[guard], index_guard);
    EXPECT_EQ(actual_indices[padding + output_count + guard], index_guard);
    EXPECT_FLOAT_EQ(actual_distances[guard], -12345.0f);
    EXPECT_FLOAT_EQ(actual_distances[padding + output_count + guard], -12345.0f);
  }
  for (int query = 0; query < query_rows; ++query) {
    const size_t row_offset = padding + static_cast<size_t>(query) * logical_k;
    for (int col = 0; col < logical_k; ++col) {
      const auto corpus_row = actual_indices[row_offset + col];
      ASSERT_GE(corpus_row, 0);
      ASSERT_LT(corpus_row, corpus_rows);
      const float expected =
        reference_value(queries.data() + static_cast<size_t>(query) * dimensions,
                        corpus.data() + static_cast<size_t>(corpus_row) * dimensions,
                        dimensions,
                        metric);
      EXPECT_NEAR(actual_distances[row_offset + col], expected, 2.0e-3f)
        << "query " << query << ", column " << col;
      if (col > 0) {
        EXPECT_NE(actual_indices[row_offset + col - 1], corpus_row);
        const auto left  = std::pair{rank_value(actual_distances[row_offset + col - 1], metric),
                                    actual_indices[row_offset + col - 1]};
        const auto right = std::pair{rank_value(actual_distances[row_offset + col], metric),
                                     actual_indices[row_offset + col]};
        EXPECT_LE(left, right) << "query " << query << ", column " << col;
      }
    }
  }
}

TEST(FusedTopkPk2Test, UsesDeterministicIndexTieBreakForZeroCosineRows)
{
  if (!fp32_topk_test_available()) { GTEST_SKIP() << "FP32 Top-K exports are unavailable"; }
  constexpr int query_rows  = 3;
  constexpr int corpus_rows = 5;
  constexpr int dimensions  = 7;
  std::vector<float> zeros(static_cast<size_t>(query_rows) * dimensions, -0.0f);
  std::vector<float> corpus(static_cast<size_t>(corpus_rows) * dimensions, 0.0f);
  for (int row = 1; row < corpus_rows; ++row) {
    corpus[static_cast<size_t>(row) * dimensions] = static_cast<float>(row);
  }

  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);
  rmm::device_uvector<float> queries_d(zeros.size(), stream);
  rmm::device_uvector<float> corpus_d(corpus.size(), stream);
  rmm::device_uvector<int> indices_d(static_cast<size_t>(query_rows) * 2, stream);
  rmm::device_uvector<float> distances_d(static_cast<size_t>(query_rows) * 2, stream);
  raft::copy(queries_d.data(), zeros.data(), zeros.size(), stream);
  raft::copy(corpus_d.data(), corpus.data(), corpus.size(), stream);

  const bool launched = try_fused_topk_pk2_tile<int>(indices_d.data(),
                                                     distances_d.data(),
                                                     queries_d.data(),
                                                     corpus_d.data(),
                                                     query_rows,
                                                     corpus_rows,
                                                     dimensions,
                                                     1,
                                                     cuvs::distance::DistanceType::CosineExpanded,
                                                     stream.get(),
                                                     nullptr);
  ASSERT_TRUE(launched);

  std::vector<int> indices(static_cast<size_t>(query_rows) * 2);
  std::vector<float> distances(static_cast<size_t>(query_rows) * 2);
  raft::copy(indices.data(), indices_d.data(), indices.size(), stream);
  raft::copy(distances.data(), distances_d.data(), distances.size(), stream);
  raft::resource::sync_stream(resources);
  for (int row = 0; row < query_rows; ++row) {
    EXPECT_EQ(indices[static_cast<size_t>(row) * 2], 0);
    EXPECT_EQ(indices[static_cast<size_t>(row) * 2 + 1], 1);
    EXPECT_FLOAT_EQ(distances[static_cast<size_t>(row) * 2], 1.0f);
    EXPECT_FLOAT_EQ(distances[static_cast<size_t>(row) * 2 + 1], 1.0f);
  }
}

TEST(FusedTopkPk2Test, OrdersDuplicateNegativeInnerProductsAcrossSplits)
{
  if (!fp32_topk_test_available()) { GTEST_SKIP() << "FP32 Top-K exports are unavailable"; }
  constexpr int query_rows  = 3;
  constexpr int corpus_rows = 259;
  constexpr int dimensions  = 7;
  constexpr int num_splits  = 2;
  std::vector<float> queries(static_cast<size_t>(query_rows) * dimensions, -0.0f);
  std::vector<float> corpus(static_cast<size_t>(corpus_rows) * dimensions, -0.0f);
  for (int row = 0; row < query_rows; ++row) {
    queries[static_cast<size_t>(row) * dimensions] = 1.0f;
  }
  for (int row = 0; row < corpus_rows; ++row) {
    corpus[static_cast<size_t>(row) * dimensions] = -4.0f;
  }
  // Identical best rows deliberately straddle the two corpus splits. Their
  // inner products are negative, and the remaining signed-zero features must
  // not disturb the deterministic index tie-break.
  corpus[0]                                     = -1.0f;
  corpus[static_cast<size_t>(258) * dimensions] = -1.0f;

  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);
  rmm::device_uvector<float> queries_d(queries.size(), stream);
  rmm::device_uvector<float> corpus_d(corpus.size(), stream);
  rmm::device_uvector<int> indices_d(static_cast<size_t>(query_rows) * 2, stream);
  rmm::device_uvector<float> distances_d(static_cast<size_t>(query_rows) * 2, stream);
  rmm::device_uvector<uint8_t> workspace_d(
    fused_topk_pk2_workspace_bytes<int>(query_rows, num_splits), stream);
  raft::copy(queries_d.data(), queries.data(), queries.size(), stream);
  raft::copy(corpus_d.data(), corpus.data(), corpus.size(), stream);

  const bool launched = try_fused_topk_pk2_tile<int>(indices_d.data(),
                                                     distances_d.data(),
                                                     queries_d.data(),
                                                     corpus_d.data(),
                                                     query_rows,
                                                     corpus_rows,
                                                     dimensions,
                                                     num_splits,
                                                     cuvs::distance::DistanceType::InnerProduct,
                                                     stream.get(),
                                                     workspace_d.data());
  ASSERT_TRUE(launched);

  std::vector<int> indices(static_cast<size_t>(query_rows) * 2);
  std::vector<float> distances(static_cast<size_t>(query_rows) * 2);
  raft::copy(indices.data(), indices_d.data(), indices.size(), stream);
  raft::copy(distances.data(), distances_d.data(), distances.size(), stream);
  raft::resource::sync_stream(resources);
  for (int row = 0; row < query_rows; ++row) {
    EXPECT_EQ(indices[static_cast<size_t>(row) * 2], 0);
    EXPECT_EQ(indices[static_cast<size_t>(row) * 2 + 1], 258);
    EXPECT_FLOAT_EQ(distances[static_cast<size_t>(row) * 2], -1.0f);
    EXPECT_FLOAT_EQ(distances[static_cast<size_t>(row) * 2 + 1], -1.0f);
  }
}

TEST(FusedTopkPk2Test, EmitsInvalidSecondSlotWhenCorpusHasOneRow)
{
  if (!fp32_topk_test_available()) { GTEST_SKIP() << "FP32 Top-K exports are unavailable"; }
  constexpr int query_rows = 3;
  constexpr int dimensions = 7;
  std::vector<float> queries(static_cast<size_t>(query_rows) * dimensions, 0.5f);
  std::vector<float> corpus(dimensions, -0.25f);

  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);
  rmm::device_uvector<float> queries_d(queries.size(), stream);
  rmm::device_uvector<float> corpus_d(corpus.size(), stream);
  rmm::device_uvector<int> indices_d(static_cast<size_t>(query_rows) * 2, stream);
  rmm::device_uvector<float> distances_d(static_cast<size_t>(query_rows) * 2, stream);
  raft::copy(queries_d.data(), queries.data(), queries.size(), stream);
  raft::copy(corpus_d.data(), corpus.data(), corpus.size(), stream);

  const bool launched = try_fused_topk_pk2_tile<int>(indices_d.data(),
                                                     distances_d.data(),
                                                     queries_d.data(),
                                                     corpus_d.data(),
                                                     query_rows,
                                                     1,
                                                     dimensions,
                                                     1,
                                                     cuvs::distance::DistanceType::L2Expanded,
                                                     stream.get(),
                                                     nullptr);
  ASSERT_TRUE(launched);

  std::vector<int> indices(static_cast<size_t>(query_rows) * 2);
  std::vector<float> distances(static_cast<size_t>(query_rows) * 2);
  raft::copy(indices.data(), indices_d.data(), indices.size(), stream);
  raft::copy(distances.data(), distances_d.data(), distances.size(), stream);
  raft::resource::sync_stream(resources);
  for (int row = 0; row < query_rows; ++row) {
    EXPECT_EQ(indices[static_cast<size_t>(row) * 2], 0);
    EXPECT_EQ(indices[static_cast<size_t>(row) * 2 + 1], -1);
    EXPECT_TRUE(std::isfinite(distances[static_cast<size_t>(row) * 2]));
    EXPECT_GT(distances[static_cast<size_t>(row) * 2 + 1], 1.0e30f);
  }
}

// Four guard elements keep each output pointer 16-byte aligned while checking
// both sides of the exact logical-width output, for either public index width.
template <typename IdxT>
void check_pk4_fixture(const std::vector<float>& queries,
                       const std::vector<float>& corpus,
                       int dimensions,
                       int logical_k,
                       int splits,
                       cuvs::distance::DistanceType metric,
                       const std::vector<int64_t>& expected_indices,
                       const std::vector<float>& expected_distances)
{
  if (!fp32_topk_test_available()) { GTEST_SKIP() << "FP32 Top-K exports are unavailable"; }
  const IdxT query_rows          = queries.size() / dimensions;
  const IdxT corpus_rows         = corpus.size() / dimensions;
  constexpr size_t padding       = 4;
  constexpr IdxT index_guard     = -12345;
  constexpr float distance_guard = -12345.0f;
  const size_t count             = static_cast<size_t>(query_rows) * logical_k;
  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);
  rmm::device_uvector<float> q(queries.size(), stream), x(corpus.size(), stream);
  rmm::device_uvector<IdxT> i(count + 2 * padding, stream);
  rmm::device_uvector<float> d(count + 2 * padding, stream);
  rmm::device_uvector<uint8_t> workspace(
    fused_topk_pk4_workspace_bytes<IdxT>(query_rows, splits, logical_k), stream);
  std::vector<IdxT> indices(i.size(), index_guard);
  std::vector<float> distances(d.size(), distance_guard);
  raft::copy(q.data(), queries.data(), queries.size(), stream);
  raft::copy(x.data(), corpus.data(), corpus.size(), stream);
  raft::copy(i.data(), indices.data(), indices.size(), stream);
  raft::copy(d.data(), distances.data(), distances.size(), stream);
  ASSERT_TRUE(try_fused_topk_pk4_tile<IdxT>(i.data() + padding,
                                            d.data() + padding,
                                            q.data(),
                                            x.data(),
                                            query_rows,
                                            corpus_rows,
                                            static_cast<IdxT>(dimensions),
                                            static_cast<IdxT>(logical_k),
                                            splits,
                                            metric,
                                            stream.get(),
                                            workspace.data()));
  raft::copy(indices.data(), i.data(), indices.size(), stream);
  raft::copy(distances.data(), d.data(), distances.size(), stream);
  raft::resource::sync_stream(resources);
  for (size_t guard = 0; guard < padding; ++guard) {
    EXPECT_EQ(indices[guard], index_guard);
    EXPECT_EQ(indices[padding + count + guard], index_guard);
    EXPECT_FLOAT_EQ(distances[guard], distance_guard);
    EXPECT_FLOAT_EQ(distances[padding + count + guard], distance_guard);
  }
  for (IdxT row = 0; row < query_rows; ++row) {
    for (int col = 0; col < logical_k; ++col) {
      const auto offset = padding + static_cast<size_t>(row) * logical_k + col;
      EXPECT_EQ(indices[offset], expected_indices[col]);
      if (std::isinf(expected_distances[col])) {
        EXPECT_TRUE(std::isinf(distances[offset]) && distances[offset] > 0.0f);
      } else {
        EXPECT_FLOAT_EQ(distances[offset], expected_distances[col]);
      }
    }
  }
}

TEST(FusedTopkPk4Test, UsesIndexTieBreakAtFourthFifthSplitBoundary)
{
  constexpr int dimensions = 7;
  std::vector<float> queries(3 * dimensions, 0.0f);
  std::vector<float> corpus(259 * dimensions, 4.0f);
  // These five tied candidates span both runtime corpus splits.
  for (int row : {0, 1, 256, 257, 258}) {
    std::fill_n(corpus.begin() + row * dimensions, dimensions, 1.0f);
  }
  for (int k : {3, 4}) {
    for (int splits : {1, 2}) {
      const std::vector<int64_t> expected{0, 1, 256, 257};
      const std::vector<float> values(4, 7.0f);
      check_pk4_fixture<int>(queries,
                             corpus,
                             dimensions,
                             k,
                             splits,
                             cuvs::distance::DistanceType::L2Expanded,
                             expected,
                             values);
      check_pk4_fixture<int64_t>(queries,
                                 corpus,
                                 dimensions,
                                 k,
                                 splits,
                                 cuvs::distance::DistanceType::L2Expanded,
                                 expected,
                                 values);
    }
  }
}

TEST(FusedTopkPk4Test, DirectFp32RefinementCanReorderAllFourCandidates)
{
  constexpr int dimensions = 7;
  std::vector<float> queries(dimensions, 0.0f), corpus(4 * dimensions, 0.0f);
  for (int row = 0; row < 4; ++row) {
    corpus[row * dimensions] = 1.0f + (3 - row) * 0.0001f;
  }
  const std::vector<int64_t> expected{3, 2, 1, 0};
  std::vector<float> values;
  for (auto index : expected) {
    values.push_back(corpus[index * dimensions] * corpus[index * dimensions]);
  }
  for (int k : {3, 4}) {
    check_pk4_fixture<int>(queries,
                           corpus,
                           dimensions,
                           k,
                           1,
                           cuvs::distance::DistanceType::L2Expanded,
                           expected,
                           values);
    check_pk4_fixture<int64_t>(queries,
                               corpus,
                               dimensions,
                               k,
                               1,
                               cuvs::distance::DistanceType::L2Expanded,
                               expected,
                               values);
  }
}

TEST(FusedTopkPk4Test, EmitsInvalidSlotsWhenCorpusIsSmallerThanLogicalK)
{
  constexpr int dimensions = 8;
  std::vector<float> queries(dimensions, 0.0f), corpus(2 * dimensions, 1.0f);
  std::fill_n(corpus.begin(), dimensions, 0.0f);
  const std::vector<int64_t> expected{0, 1, -1, -1};
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> values{0.0f, 8.0f, inf, inf};
  for (int k : {3, 4}) {
    check_pk4_fixture<int>(queries,
                           corpus,
                           dimensions,
                           k,
                           1,
                           cuvs::distance::DistanceType::L2Expanded,
                           expected,
                           values);
    check_pk4_fixture<int64_t>(queries,
                               corpus,
                               dimensions,
                               k,
                               1,
                               cuvs::distance::DistanceType::L2Expanded,
                               expected,
                               values);
  }
}

TEST(FusedTopkPk2PublicTest, ExercisesCanonicalAllNeighborsDispatch)
{
  if (!is_fused_topk_pk2_auto_dispatch_available()) {
    GTEST_SKIP() << "automatic physical-K2 dispatch is initially SM120-only";
  }
  constexpr int64_t rows       = 8192;
  constexpr int64_t dimensions = 32;
  constexpr int64_t logical_k  = 2;

  raft::resources resources;
  auto dataset      = raft::make_device_matrix<float, int64_t>(resources, rows, dimensions);
  auto neighbors    = raft::make_device_matrix<int64_t, int64_t>(resources, rows, logical_k);
  auto distances    = raft::make_device_matrix<float, int64_t>(resources, rows, logical_k);
  const auto stream = raft::resource::get_cuda_stream(resources);
  RAFT_CUDA_TRY(
    cudaMemsetAsync(dataset.data_handle(), 0, dataset.size() * sizeof(float), stream.get()));
  auto dataset_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    dataset.data_handle(), rows, dimensions);

  cuvs::neighbors::brute_force::index_params index_params;
  index_params.metric = cuvs::distance::DistanceType::L2Expanded;
  auto index          = cuvs::neighbors::brute_force::build(resources, index_params, dataset_view);
  cuvs::neighbors::brute_force::search_params search_params;
  search_params.allow_approximate = true;
  cuvs::neighbors::brute_force::search(
    resources, search_params, index, dataset_view, neighbors.view(), distances.view());

  std::vector<int64_t> actual_indices(static_cast<size_t>(rows) * logical_k);
  std::vector<float> actual_distances(static_cast<size_t>(rows) * logical_k);
  raft::copy(actual_indices.data(), neighbors.data_handle(), actual_indices.size(), stream.get());
  raft::copy(
    actual_distances.data(), distances.data_handle(), actual_distances.size(), stream.get());
  raft::resource::sync_stream(resources);
  for (int64_t row = 0; row < rows; ++row) {
    EXPECT_EQ(actual_indices[static_cast<size_t>(row) * logical_k], 0);
    EXPECT_EQ(actual_indices[static_cast<size_t>(row) * logical_k + 1], 1);
    EXPECT_FLOAT_EQ(actual_distances[static_cast<size_t>(row) * logical_k], 0.0f);
    EXPECT_FLOAT_EQ(actual_distances[static_cast<size_t>(row) * logical_k + 1], 0.0f);
  }
}

// Values are distinct in FP32 but identical after TF32 rounding. Positive query
// rows therefore expose approximate membership even after retained-value refinement.
std::vector<float> near_tie_data(int64_t rows, int64_t dimensions)
{
  std::vector<float> data(static_cast<size_t>(rows * dimensions), 0.0f);
  data[0]              = 1.0001f;
  data[dimensions]     = 1.0002f;
  data[2 * dimensions] = 1.0003f;
  return data;
}

struct TopkOutput {
  std::vector<int64_t> indices;
  std::vector<float> distances;
};

TopkOutput copy_output(raft::resources const& resources,
                       const int64_t* indices,
                       const float* distances,
                       size_t count)
{
  TopkOutput output{std::vector<int64_t>(count), std::vector<float>(count)};
  auto stream = raft::resource::get_cuda_stream(resources);
  raft::copy(output.indices.data(), indices, count, stream);
  raft::copy(output.distances.data(), distances, count, stream);
  raft::resource::sync_stream(resources);
  return output;
}

TEST(FusedTopkPk2PublicTest, ApproximateMembershipRequiresExplicitPermission)
{
  constexpr int64_t rows = 8192, dimensions = 128, k = 2;
  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);
  const auto host   = near_tie_data(rows, dimensions);
  auto data         = raft::make_device_matrix<float, int64_t>(resources, rows, dimensions);
  auto indices      = raft::make_device_matrix<int64_t, int64_t>(resources, rows, k);
  auto distances    = raft::make_device_matrix<float, int64_t>(resources, rows, k);
  raft::copy(data.data_handle(), host.data(), host.size(), stream);
  auto view =
    raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), rows, dimensions);
  cuvs::neighbors::brute_force::index_params build_params;
  build_params.metric = cuvs::distance::DistanceType::InnerProduct;
  auto index          = cuvs::neighbors::brute_force::build(resources, build_params, view);
  cuvs::neighbors::brute_force::search_params params;
  ASSERT_FALSE(params.allow_approximate);
  auto run = [&] {
    cuvs::neighbors::brute_force::search(
      resources, params, index, view, indices.view(), distances.view());
    return copy_output(resources, indices.data_handle(), distances.data_handle(), rows * k);
  };
  const auto defaults      = run();
  params.allow_approximate = false;
  const auto disabled      = run();
  EXPECT_EQ(defaults.indices, disabled.indices);
  EXPECT_EQ(defaults.distances, disabled.distances);
  // Legacy overloads must not silently enable the new backend either.
  cuvs::neighbors::brute_force::search(resources, index, view, indices.view(), distances.view());
  const auto legacy =
    copy_output(resources, indices.data_handle(), distances.data_handle(), rows * k);
  EXPECT_EQ(defaults.indices, legacy.indices);
  EXPECT_EQ(defaults.distances, legacy.distances);
  for (int row = 0; row < 3; ++row) {
    EXPECT_EQ(defaults.indices[2 * row], 2);
    EXPECT_EQ(defaults.indices[2 * row + 1], 1);
  }

  if (!is_fused_topk_pk2_auto_dispatch_available()) { return; }
  params.allow_approximate = true;
  const auto approximate   = run();
  for (int row = 0; row < 3; ++row) {
    EXPECT_EQ(approximate.indices[2 * row], 1);
    EXPECT_EQ(approximate.indices[2 * row + 1], 0);
    EXPECT_FLOAT_EQ(approximate.distances[2 * row], host[row * dimensions] * host[dimensions]);
    EXPECT_FLOAT_EQ(approximate.distances[2 * row + 1], host[row * dimensions] * host[0]);
  }
  // Confirm the direct launcher on this untimed anchor; policy eligibility alone
  // is not a claim that every opted-in call uses the backend.
  rmm::device_uvector<uint8_t> workspace(fused_topk_pk2_workspace_bytes<int64_t>(rows, 1), stream);
  ASSERT_TRUE(try_fused_topk_pk2_tile<int64_t>(indices.data_handle(),
                                               distances.data_handle(),
                                               data.data_handle(),
                                               data.data_handle(),
                                               rows,
                                               rows,
                                               dimensions,
                                               1,
                                               build_params.metric,
                                               stream.get(),
                                               workspace.data()));
  const auto direct =
    copy_output(resources, indices.data_handle(), distances.data_handle(), rows * k);
  EXPECT_EQ(approximate.indices, direct.indices);
  EXPECT_EQ(approximate.distances, direct.distances);
}

void check_default_l2_stream_contract(cuvs::distance::DistanceType metric)
{
  constexpr int64_t rows = 257, dimensions = 128, k = 2;
  rmm::cuda_stream user_stream;
  raft::resources resources;
  raft::resource::set_cuda_stream(resources, user_stream);
  raft::resource::set_cuda_stream_pool(resources, std::make_shared<rmm::cuda_stream_pool>(2));
  const auto stream = raft::resource::get_cuda_stream(resources);
  std::vector<float> host(rows * dimensions);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < dimensions; ++col) {
      // One changing coordinate keeps products and all partial sums exact in FP32,
      // independently of the fallback reduction tree; the other coordinates are nonzero.
      host[row * dimensions + col] = col == 0 ? row * 0.125f : 1.0f;
    }
  }
  auto data      = raft::make_device_matrix<float, int64_t>(resources, rows, dimensions);
  auto indices   = raft::make_device_matrix<int64_t, int64_t>(resources, rows, k);
  auto distances = raft::make_device_matrix<float, int64_t>(resources, rows, k);
  raft::copy(data.data_handle(), host.data(), host.size(), stream);
  auto view =
    raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), rows, dimensions);
  cuvs::neighbors::brute_force::index_params build_params;
  build_params.metric = metric;
  auto index          = cuvs::neighbors::brute_force::build(resources, build_params, view);
  cuvs::neighbors::brute_force::search_params params;
  ASSERT_FALSE(params.allow_approximate);
  auto run = [&](bool legacy) {
    // Fresh poison on the user stream prevents unchanged outputs from hiding a missed write.
    const std::vector<int64_t> poison_i(rows * k, -1);
    const std::vector<float> poison_d(rows * k, -123.0f);
    raft::copy(indices.data_handle(), poison_i.data(), poison_i.size(), stream);
    raft::copy(distances.data_handle(), poison_d.data(), poison_d.size(), stream);
    if (legacy) {
      cuvs::neighbors::brute_force::search(
        resources, index, view, indices.view(), distances.view());
    } else {
      cuvs::neighbors::brute_force::search(
        resources, params, index, view, indices.view(), distances.view());
    }
    // The public call joins its pool; copies and synchronization use the explicit user stream.
    return copy_output(resources, indices.data_handle(), distances.data_handle(), rows * k);
  };
  const auto defaults      = run(false);
  params.allow_approximate = false;
  const auto disabled      = run(false);
  const auto legacy        = run(true);
  EXPECT_EQ(defaults.indices, disabled.indices);
  EXPECT_EQ(defaults.indices, legacy.indices);
  EXPECT_EQ(defaults.distances, disabled.distances);
  EXPECT_EQ(defaults.distances, legacy.distances);
  const float expected =
    metric == cuvs::distance::DistanceType::L2SqrtExpanded ? 0.125f : 0.015625f;
  for (int64_t row = 0; row < rows; ++row) {
    EXPECT_EQ(defaults.indices[2 * row], row);
    EXPECT_FLOAT_EQ(defaults.distances[2 * row], 0.0f);
    EXPECT_TRUE(defaults.indices[2 * row + 1] == row - 1 ||
                defaults.indices[2 * row + 1] == row + 1);
    EXPECT_NEAR(defaults.distances[2 * row + 1], expected, 2.0e-6f);
  }
}

TEST(FusedTopkPk2PublicTest, DefaultSquaredL2PreservesNondefaultAndPoolStreams)
{
  check_default_l2_stream_contract(cuvs::distance::DistanceType::L2Expanded);
}

TEST(FusedTopkPk2PublicTest, DefaultSqrtL2PreservesNondefaultAndPoolStreams)
{
  check_default_l2_stream_contract(cuvs::distance::DistanceType::L2SqrtExpanded);
}

TEST(FusedTopkPk2PublicTest, AllNeighborsForwardsApproximatePermission)
{
  constexpr int64_t rows = 8192, dimensions = 128, k = 2;
  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);
  const auto host   = near_tie_data(rows, dimensions);
  auto data         = raft::make_device_matrix<float, int64_t>(resources, rows, dimensions);
  auto indices      = raft::make_device_matrix<int64_t, int64_t>(resources, rows, k);
  auto distances    = raft::make_device_matrix<float, int64_t>(resources, rows, k);
  raft::copy(data.data_handle(), host.data(), host.size(), stream);
  auto view =
    raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), rows, dimensions);
  cuvs::neighbors::all_neighbors::all_neighbors_params params;
  params.n_clusters = 1;
  params.metric     = cuvs::distance::DistanceType::InnerProduct;
  cuvs::neighbors::graph_build_params::brute_force_params brute_params;
  brute_params.build_params.metric = params.metric;
  for (bool allow : {false, true}) {
    brute_params.search_params.allow_approximate = allow;
    params.graph_build_params                    = brute_params;
    cuvs::neighbors::all_neighbors::build(
      resources, params, view, indices.view(), distances.view());
    const auto actual =
      copy_output(resources, indices.data_handle(), distances.data_handle(), rows * k);
    const bool fused = allow && is_fused_topk_pk2_auto_dispatch_available();
    for (int row = 0; row < 3; ++row) {
      EXPECT_EQ(actual.indices[2 * row], fused ? 1 : 2);
      EXPECT_EQ(actual.indices[2 * row + 1], fused ? 0 : 1);
    }
  }
}

TEST(FusedTopkPk2PublicTest, PermissionPreservesLayoutAndFilteredFallbacks)
{
  constexpr int64_t rows = 8192, dimensions = 32, k = 2;
  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);
  const auto host   = near_tie_data(rows, dimensions);
  // Only the first and last elements differ from one, so row-major and
  // column-major views of this allocation describe the same matrix. Borrow it
  // in a row-major index: a column-major build would transpose into a new
  // allocation and fail the self-pointer predicate before testing layout.
  std::vector<float> column_host(host.size(), 1.0f);
  column_host.front() = 1.0001f;
  column_host.back()  = 1.0003f;
  auto data           = raft::make_device_matrix<float, int64_t>(resources, rows, dimensions);
  auto query          = raft::make_device_matrix<float, int64_t>(resources, rows, dimensions);
  auto columns =
    raft::make_device_matrix<float, int64_t, raft::col_major>(resources, rows, dimensions);
  auto indices   = raft::make_device_matrix<int64_t, int64_t>(resources, rows, k);
  auto distances = raft::make_device_matrix<float, int64_t>(resources, rows, k);
  raft::copy(data.data_handle(), host.data(), host.size(), stream);
  raft::copy(query.data_handle(), host.data(), host.size(), stream);
  raft::copy(columns.data_handle(), column_host.data(), column_host.size(), stream);
  auto view =
    raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), rows, dimensions);
  auto queries =
    raft::make_device_matrix_view<const float, int64_t>(query.data_handle(), rows, dimensions);
  auto column_queries = raft::make_device_matrix_view<const float, int64_t, raft::col_major>(
    columns.data_handle(), rows, dimensions);
  auto aliased_rows =
    raft::make_device_matrix_view<const float, int64_t>(columns.data_handle(), rows, dimensions);
  cuvs::neighbors::brute_force::index_params build_params;
  build_params.metric = cuvs::distance::DistanceType::InnerProduct;
  auto index          = cuvs::neighbors::brute_force::build(resources, build_params, view);
  auto column_index   = cuvs::neighbors::brute_force::build(resources, build_params, aliased_rows);
  ASSERT_EQ(column_index.dataset().data_handle(), column_queries.data_handle());
  // All-passing bitset still exercises the filtered API, whose path is unchanged.
  cuvs::core::bitset<uint32_t, int64_t> bits(resources, rows, true);
  cuvs::neighbors::filtering::bitset_filter filter(bits.view());
  for (int path = 1; path < 3; ++path) {
    TopkOutput baseline;
    for (bool allow : {false, true}) {
      cuvs::neighbors::brute_force::search_params params;
      params.allow_approximate = allow;
      if (path == 0) {
        cuvs::neighbors::brute_force::search(
          resources, params, index, queries, indices.view(), distances.view());
      } else if (path == 1) {
        cuvs::neighbors::brute_force::search(
          resources, params, column_index, column_queries, indices.view(), distances.view());
      } else {
        cuvs::neighbors::brute_force::search(
          resources, params, index, view, indices.view(), distances.view(), filter);
      }
      const auto actual =
        copy_output(resources, indices.data_handle(), distances.data_handle(), rows * k);
      if (!allow) {
        baseline = actual;
      } else {
        EXPECT_EQ(actual.indices, baseline.indices);
        EXPECT_EQ(actual.distances, baseline.distances);
      }
      EXPECT_EQ(actual.indices[0], path == 1 ? rows - 1 : 2);
      EXPECT_EQ(actual.indices[1], path == 1 ? 0 : 1);
    }
  }
}

TEST(FusedTopkPk4PublicTest, ExercisesOptedInK3AndK4AllNeighborsDispatch)
{
  if (!is_fused_topk_pk4_auto_dispatch_available()) {
    GTEST_SKIP() << "automatic physical-K4 dispatch is initially SM120-only";
  }
  constexpr int64_t rows       = 8192;
  constexpr int64_t dimensions = 128;

  raft::resources resources;
  auto dataset      = raft::make_device_matrix<float, int64_t>(resources, rows, dimensions);
  const auto stream = raft::resource::get_cuda_stream(resources);
  RAFT_CUDA_TRY(
    cudaMemsetAsync(dataset.data_handle(), 0, dataset.size() * sizeof(float), stream.get()));
  auto dataset_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    dataset.data_handle(), rows, dimensions);

  cuvs::neighbors::brute_force::index_params index_params;
  index_params.metric = cuvs::distance::DistanceType::L2Expanded;
  auto index          = cuvs::neighbors::brute_force::build(resources, index_params, dataset_view);
  cuvs::neighbors::brute_force::search_params search_params;
  search_params.allow_approximate = true;
  for (const int64_t logical_k : {int64_t{3}, int64_t{4}}) {
    auto neighbors = raft::make_device_matrix<int64_t, int64_t>(resources, rows, logical_k);
    auto distances = raft::make_device_matrix<float, int64_t>(resources, rows, logical_k);
    cuvs::neighbors::brute_force::search(
      resources, search_params, index, dataset_view, neighbors.view(), distances.view());

    std::vector<int64_t> actual_indices(static_cast<size_t>(rows) * logical_k);
    std::vector<float> actual_distances(static_cast<size_t>(rows) * logical_k);
    raft::copy(actual_indices.data(), neighbors.data_handle(), actual_indices.size(), stream.get());
    raft::copy(
      actual_distances.data(), distances.data_handle(), actual_distances.size(), stream.get());
    raft::resource::sync_stream(resources);
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t column = 0; column < logical_k; ++column) {
        const auto offset = static_cast<size_t>(row) * logical_k + column;
        EXPECT_EQ(actual_indices[offset], column);
        EXPECT_FLOAT_EQ(actual_distances[offset], 0.0f);
      }
    }
  }
}

std::vector<float> pk4_near_tie_data(int64_t rows, int64_t dimensions)
{
  std::vector<float> host(rows * dimensions, 0.0f);
  const float values[] = {1.00008f, 1.00016f, 1.00024f, 1.00032f, 1.00040f};
  for (int row = 0; row < 5; ++row) {
    host[row * dimensions] = values[row];
  }
  return host;
}

TEST(FusedTopkPk4PublicTest, BothLogicalWidthsRequireExplicitApproximatePermission)
{
  constexpr int64_t rows = 8192, dimensions = 128;
  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);
  const auto host   = pk4_near_tie_data(rows, dimensions);
  auto data         = raft::make_device_matrix<float, int64_t>(resources, rows, dimensions);
  raft::copy(data.data_handle(), host.data(), host.size(), stream);
  auto view =
    raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), rows, dimensions);
  cuvs::neighbors::brute_force::index_params build_params;
  build_params.metric = cuvs::distance::DistanceType::InnerProduct;
  auto index          = cuvs::neighbors::brute_force::build(resources, build_params, view);
  for (int64_t k : {3, 4}) {
    auto indices   = raft::make_device_matrix<int64_t, int64_t>(resources, rows, k);
    auto distances = raft::make_device_matrix<float, int64_t>(resources, rows, k);
    cuvs::neighbors::brute_force::search_params params;
    ASSERT_FALSE(params.allow_approximate);
    auto run = [&] {
      cuvs::neighbors::brute_force::search(
        resources, params, index, view, indices.view(), distances.view());
      return copy_output(resources, indices.data_handle(), distances.data_handle(), rows * k);
    };
    const auto defaults      = run();
    params.allow_approximate = false;
    const auto disabled      = run();
    EXPECT_EQ(defaults.indices, disabled.indices);
    EXPECT_EQ(defaults.distances, disabled.distances);
    cuvs::neighbors::brute_force::search(resources, index, view, indices.view(), distances.view());
    const auto legacy =
      copy_output(resources, indices.data_handle(), distances.data_handle(), rows * k);
    EXPECT_EQ(defaults.indices, legacy.indices);
    EXPECT_EQ(defaults.distances, legacy.distances);
    for (int row = 0; row < 5; ++row) {
      for (int col = 0; col < k; ++col) {
        EXPECT_EQ(defaults.indices[row * k + col], 4 - col);
      }
    }
    if (!is_fused_topk_pk4_auto_dispatch_available()) { continue; }
    params.allow_approximate = true;
    const auto approximate   = run();
    for (int row = 0; row < 5; ++row) {
      for (int col = 0; col < k; ++col) {
        EXPECT_EQ(approximate.indices[row * k + col], 3 - col);
        EXPECT_FLOAT_EQ(approximate.distances[row * k + col],
                        host[row * dimensions] * host[(3 - col) * dimensions]);
      }
    }
    // Permission does not force routing. Confirm direct agreement on this
    // untimed eligible anchor for both compaction and physical-width output.
    rmm::device_uvector<uint8_t> workspace(fused_topk_pk4_workspace_bytes<int64_t>(rows, 1, k),
                                           stream);
    ASSERT_TRUE(try_fused_topk_pk4_tile<int64_t>(indices.data_handle(),
                                                 distances.data_handle(),
                                                 data.data_handle(),
                                                 data.data_handle(),
                                                 rows,
                                                 rows,
                                                 dimensions,
                                                 k,
                                                 1,
                                                 build_params.metric,
                                                 stream.get(),
                                                 workspace.data()));
    const auto direct =
      copy_output(resources, indices.data_handle(), distances.data_handle(), rows * k);
    EXPECT_EQ(approximate.indices, direct.indices);
    EXPECT_EQ(approximate.distances, direct.distances);
  }
}

TEST(FusedTopkPk4PublicTest, BothLogicalWidthsForwardAllNeighborsPermission)
{
  constexpr int64_t rows = 8192, dimensions = 128;
  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);
  const auto host   = pk4_near_tie_data(rows, dimensions);
  auto data         = raft::make_device_matrix<float, int64_t>(resources, rows, dimensions);
  raft::copy(data.data_handle(), host.data(), host.size(), stream);
  auto view =
    raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), rows, dimensions);
  cuvs::neighbors::all_neighbors::all_neighbors_params params;
  params.n_clusters = 1;
  params.metric     = cuvs::distance::DistanceType::InnerProduct;
  cuvs::neighbors::graph_build_params::brute_force_params brute_params;
  brute_params.build_params.metric = params.metric;
  for (int64_t k : {3, 4}) {
    auto indices   = raft::make_device_matrix<int64_t, int64_t>(resources, rows, k);
    auto distances = raft::make_device_matrix<float, int64_t>(resources, rows, k);
    for (bool allow : {false, true}) {
      brute_params.search_params.allow_approximate = allow;
      params.graph_build_params                    = brute_params;
      cuvs::neighbors::all_neighbors::build(
        resources, params, view, indices.view(), distances.view());
      const auto actual =
        copy_output(resources, indices.data_handle(), distances.data_handle(), rows * k);
      const bool fused = allow && is_fused_topk_pk4_auto_dispatch_available();
      for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < k; ++col) {
          const int expected = (fused ? 3 : 4) - col;
          EXPECT_EQ(actual.indices[row * k + col], expected);
          EXPECT_FLOAT_EQ(actual.distances[row * k + col],
                          host[row * dimensions] * host[expected * dimensions]);
        }
      }
    }
  }
}

TEST(FusedTopkPk4Test, EuclideanFinalValueTiesAgreeAcrossSplitsAndWidths)
{
  constexpr int dimensions = 2;
  std::vector<float> queries(dimensions, 0.0f), corpus(259 * dimensions, 4.0f);
  corpus[0]                    = 1.0f;
  corpus[1]                    = std::sqrt(std::ldexp(1.0f, -23));
  corpus[2]                    = 0.5f;
  corpus[3]                    = 0.0f;
  corpus[256 * dimensions]     = 1.0f;
  corpus[256 * dimensions + 1] = 0.0f;
  corpus[258 * dimensions]     = 2.0f;
  corpus[258 * dimensions + 1] = 0.0f;
  ASSERT_EQ(std::sqrt(1.0f + corpus[1] * corpus[1]), 1.0f);
  for (int k : {3, 4}) {
    for (int splits : {1, 2}) {
      const std::vector<int64_t> expected{1, 0, 256, 258};
      const std::vector<float> values{0.5f, 1.0f, 1.0f, 2.0f};
      check_pk4_fixture<int>(queries,
                             corpus,
                             dimensions,
                             k,
                             splits,
                             cuvs::distance::DistanceType::L2SqrtExpanded,
                             expected,
                             values);
      check_pk4_fixture<int64_t>(queries,
                                 corpus,
                                 dimensions,
                                 k,
                                 splits,
                                 cuvs::distance::DistanceType::L2SqrtExpanded,
                                 expected,
                                 values);
    }
  }
}

TEST(FusedTopkPk4Test, RetainsFiniteScoresAboveFormerSentinel)
{
  const std::vector<float> queries{std::ldexp(1.0f, 64)};
  const std::vector<float> corpus{-std::ldexp(2047.0f, 53), -std::ldexp(2047.0f, 52), 0.0f};
  const std::vector<int64_t> expected{2, 1, 0, -1};
  const std::vector<float> values{
    0.0f, queries[0] * corpus[1], queries[0] * corpus[0], std::numeric_limits<float>::infinity()};
  ASSERT_TRUE(std::isfinite(values[2]));
  ASSERT_GT(-values[2], 3.4e38f);
  for (int k : {3, 4}) {
    check_pk4_fixture<int>(
      queries, corpus, 1, k, 1, cuvs::distance::DistanceType::InnerProduct, expected, values);
    check_pk4_fixture<int64_t>(
      queries, corpus, 1, k, 1, cuvs::distance::DistanceType::InnerProduct, expected, values);
  }
}

TEST(FusedTopkPk2Test, RetainsFiniteScoresAboveFormerSentinel)
{
  if (!fp32_topk_test_available()) { GTEST_SKIP() << "FP32 Top-K exports are unavailable"; }
  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);
  std::vector<float> queries{std::ldexp(1.0f, 64)};
  std::vector<float> corpus{-std::ldexp(2047.0f, 53), -std::ldexp(2047.0f, 52)};
  auto q = raft::make_device_matrix<float, int64_t>(resources, 1, 1);
  auto x = raft::make_device_matrix<float, int64_t>(resources, 2, 1);
  auto i = raft::make_device_matrix<int64_t, int64_t>(resources, 1, 2);
  auto d = raft::make_device_matrix<float, int64_t>(resources, 1, 2);
  rmm::device_uvector<uint8_t> workspace(fused_topk_pk2_workspace_bytes<int64_t>(1, 1), stream);
  raft::copy(q.data_handle(), queries.data(), queries.size(), stream);
  raft::copy(x.data_handle(), corpus.data(), corpus.size(), stream);
  ASSERT_TRUE(try_fused_topk_pk2_tile<int64_t>(i.data_handle(),
                                               d.data_handle(),
                                               q.data_handle(),
                                               x.data_handle(),
                                               1,
                                               2,
                                               1,
                                               1,
                                               cuvs::distance::DistanceType::InnerProduct,
                                               stream.get(),
                                               workspace.data()));
  const auto actual = copy_output(resources, i.data_handle(), d.data_handle(), 2);
  EXPECT_EQ(actual.indices, (std::vector<int64_t>{1, 0}));
  EXPECT_FLOAT_EQ(actual.distances[0], queries[0] * corpus[1]);
  EXPECT_FLOAT_EQ(actual.distances[1], queries[0] * corpus[0]);
  EXPECT_TRUE(std::isfinite(actual.distances[1]));
  EXPECT_GT(-actual.distances[1], 3.4e38f);
}

TEST(FusedTopkPk2Test, EuclideanFinalValueTiesAgreeAcrossSplits)
{
  if (!fp32_topk_test_available()) { GTEST_SKIP() << "FP32 Top-K exports are unavailable"; }
  constexpr int64_t rows = 259, dimensions = 2;
  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);
  std::vector<float> queries(dimensions, 0.0f), corpus(rows * dimensions, 4.0f);
  // Distinct squared distances round to the same FP32 sqrt. Put the larger
  // squared distance at the lower index, across the runtime split boundary.
  corpus[0]                    = 1.0f;
  corpus[1]                    = std::sqrt(std::ldexp(1.0f, -23));
  corpus[258 * dimensions]     = 1.0f;
  corpus[258 * dimensions + 1] = 0.0f;
  ASSERT_EQ(std::sqrt(1.0f + corpus[1] * corpus[1]), 1.0f);
  auto q = raft::make_device_matrix<float, int64_t>(resources, 1, dimensions);
  auto x = raft::make_device_matrix<float, int64_t>(resources, rows, dimensions);
  auto i = raft::make_device_matrix<int64_t, int64_t>(resources, 1, 2);
  auto d = raft::make_device_matrix<float, int64_t>(resources, 1, 2);
  raft::copy(q.data_handle(), queries.data(), queries.size(), stream);
  raft::copy(x.data_handle(), corpus.data(), corpus.size(), stream);
  for (int splits : {1, 2}) {
    rmm::device_uvector<uint8_t> workspace(fused_topk_pk2_workspace_bytes<int64_t>(1, splits),
                                           stream);
    ASSERT_TRUE(try_fused_topk_pk2_tile<int64_t>(i.data_handle(),
                                                 d.data_handle(),
                                                 q.data_handle(),
                                                 x.data_handle(),
                                                 1,
                                                 rows,
                                                 dimensions,
                                                 splits,
                                                 cuvs::distance::DistanceType::L2SqrtExpanded,
                                                 stream.get(),
                                                 workspace.data()));
    const auto actual = copy_output(resources, i.data_handle(), d.data_handle(), 2);
    EXPECT_EQ(actual.indices, (std::vector<int64_t>{0, 258}));
    EXPECT_EQ(actual.distances, (std::vector<float>{1.0f, 1.0f}));
  }
}

TEST(FusedTopkPk2PolicyTest, CapabilityPolicyHasNoBenchmarkShapeAllowlist)
{
  using DT = cuvs::distance::DistanceType;
  for (int64_t q : {1, 16, 128, 257, 4096, 8191, 10001, 100000}) {
    for (int64_t n : {263, 8192, 100000}) {
      for (int64_t d : {7, 16, 31, 32, 33, 64, 127, 128, 129, 511, 512, 1025, 4097}) {
        for (auto metric :
             {DT::L2Expanded, DT::L2SqrtExpanded, DT::CosineExpanded, DT::InnerProduct}) {
          for (bool graph : {false, true}) {
            EXPECT_TRUE(is_fused_topk_pk2_candidate<float>(q, n, d, 2, metric, graph));
            EXPECT_TRUE(is_fused_topk_pk2_candidate<half>(q, n, d, 2, metric, graph));
            for (int k : {3, 4}) {
              EXPECT_TRUE(is_fused_topk_pk4_candidate<float>(q, n, d, k, metric, graph));
              EXPECT_TRUE(is_fused_topk_pk4_candidate<half>(q, n, d, k, metric, graph));
            }
          }
        }
      }
    }
  }
  EXPECT_FALSE(is_fused_topk_pk2_candidate(8192, 8192, 32, 3, DT::L2Expanded, true));
  EXPECT_FALSE(is_fused_topk_pk4_candidate(8192, 8192, 32, 2, DT::L2Expanded, true));
  EXPECT_FALSE(is_fused_topk_pk4_candidate(8192, 8192, 32, 5, DT::L2Expanded, true));
  EXPECT_EQ(fused_topk_pk4_workspace_bytes<int>(7, 1, 4), 0);
  EXPECT_EQ(fused_topk_pk4_workspace_bytes<int64_t>(7, 1, 4), 112);
  EXPECT_EQ(fused_topk_pk4_workspace_bytes<int>(7, 1, 3), 224);
  EXPECT_EQ(fused_topk_pk4_workspace_bytes<int>(7, 2, 4), 448);
}

// These fixtures use stored half values as the reference input, never the pre-cast float values.
template <int PhysicalK, typename DataT, typename IdxT>
TopkOutput check_typed_topk(const std::vector<float>& query_values,
                            const std::vector<float>& corpus_values,
                            int dimensions,
                            cuvs::distance::DistanceType metric,
                            int splits,
                            int logical_k,
                            bool nondefault_stream = false)
{
  const int rows               = query_values.size() / dimensions;
  const int count              = corpus_values.size() / dimensions;
  constexpr size_t input_guard = 16 / sizeof(DataT);
  constexpr size_t index_guard = 16 / sizeof(IdxT);
  constexpr size_t value_guard = 4;
  std::vector<DataT> q(query_values.size() + 2 * input_guard, DataT{-7});
  std::vector<DataT> x(corpus_values.size() + 2 * input_guard, DataT{-7});
  std::vector<float> stored_q(query_values.size()), stored_x(corpus_values.size());
  for (size_t i = 0; i < query_values.size(); ++i) {
    q[input_guard + i] = DataT{query_values[i]};
    stored_q[i]        = static_cast<float>(q[input_guard + i]);
  }
  for (size_t i = 0; i < corpus_values.size(); ++i) {
    x[input_guard + i] = DataT{corpus_values[i]};
    stored_x[i]        = static_cast<float>(x[input_guard + i]);
  }
  rmm::cuda_stream user_stream;
  raft::resources resources;
  if (nondefault_stream) { raft::resource::set_cuda_stream(resources, user_stream); }
  auto stream = raft::resource::get_cuda_stream(resources);
  rmm::device_uvector<DataT> qd(q.size(), stream), xd(x.size(), stream);
  std::vector<IdxT> ids(rows * logical_k + 2 * index_guard, IdxT{-7});
  std::vector<float> values(rows * logical_k + 2 * value_guard, -7.0f);
  rmm::device_uvector<IdxT> idd(ids.size(), stream);
  rmm::device_uvector<float> vd(values.size(), stream);
  const size_t workspace_bytes =
    fused_topk_workspace_bytes<PhysicalK>(static_cast<IdxT>(rows), splits, logical_k);
  std::vector<uint8_t> workspace(workspace_bytes + 32, 0xa5);
  rmm::device_uvector<uint8_t> wd(workspace.size(), stream);
  raft::copy(qd.data(), q.data(), q.size(), stream);
  raft::copy(xd.data(), x.data(), x.size(), stream);
  raft::copy(idd.data(), ids.data(), ids.size(), stream);
  raft::copy(vd.data(), values.data(), values.size(), stream);
  raft::copy(wd.data(), workspace.data(), workspace.size(), stream);
  const auto launch = [&] {
    if constexpr (PhysicalK == 2) {
      return try_fused_topk_pk2_tile<IdxT>(idd.data() + index_guard,
                                           vd.data() + value_guard,
                                           qd.data() + input_guard,
                                           xd.data() + input_guard,
                                           static_cast<IdxT>(rows),
                                           static_cast<IdxT>(count),
                                           static_cast<IdxT>(dimensions),
                                           splits,
                                           metric,
                                           stream.get(),
                                           wd.data() + 16);
    } else {
      return try_fused_topk_pk4_tile<IdxT>(idd.data() + index_guard,
                                           vd.data() + value_guard,
                                           qd.data() + input_guard,
                                           xd.data() + input_guard,
                                           static_cast<IdxT>(rows),
                                           static_cast<IdxT>(count),
                                           static_cast<IdxT>(dimensions),
                                           static_cast<IdxT>(logical_k),
                                           splits,
                                           metric,
                                           stream.get(),
                                           wd.data() + 16);
    }
  };
  EXPECT_TRUE(launch());  // Required on the managed SM120 gate, never silently skipped.
  raft::copy(ids.data(), idd.data(), ids.size(), stream);
  raft::copy(values.data(), vd.data(), values.size(), stream);
  raft::resource::sync_stream(resources);
  const auto first_ids    = ids;
  const auto first_values = values;
  EXPECT_TRUE(launch());
  raft::copy(ids.data(), idd.data(), ids.size(), stream);
  raft::copy(values.data(), vd.data(), values.size(), stream);
  std::vector<DataT> actual_q(q.size()), actual_x(x.size());
  raft::copy(actual_q.data(), qd.data(), q.size(), stream);
  raft::copy(actual_x.data(), xd.data(), x.size(), stream);
  raft::copy(workspace.data(), wd.data(), workspace.size(), stream);
  raft::resource::sync_stream(resources);
  EXPECT_EQ(std::memcmp(first_ids.data(), ids.data(), ids.size() * sizeof(IdxT)), 0);
  EXPECT_EQ(std::memcmp(first_values.data(), values.data(), values.size() * sizeof(float)), 0);
  EXPECT_EQ(std::memcmp(actual_q.data(), q.data(), q.size() * sizeof(DataT)), 0);
  EXPECT_EQ(std::memcmp(actual_x.data(), x.data(), x.size() * sizeof(DataT)), 0);
  for (size_t i = 0; i < index_guard; ++i) {
    EXPECT_EQ(ids[i], -7);
    EXPECT_EQ(ids[ids.size() - 1 - i], -7);
  }
  for (size_t i = 0; i < value_guard; ++i) {
    EXPECT_EQ(values[i], -7);
    EXPECT_EQ(values[values.size() - 1 - i], -7);
  }
  for (size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(workspace[i], 0xa5);
    EXPECT_EQ(workspace[workspace.size() - 1 - i], 0xa5);
  }
  TopkOutput out{std::vector<int64_t>(rows * logical_k), std::vector<float>(rows * logical_k)};
  for (int row = 0; row < rows; ++row) {
    for (int k = 0; k < logical_k; ++k) {
      const size_t offset   = row * logical_k + k;
      const auto id         = ids[index_guard + offset];
      const auto value      = values[value_guard + offset];
      out.indices[offset]   = id;
      out.distances[offset] = value;
      if (k >= count) {
        EXPECT_EQ(id, -1);
        EXPECT_EQ(value, std::numeric_limits<float>::infinity());
        continue;
      }
      EXPECT_GE(id, 0);
      EXPECT_LT(id, count);
      EXPECT_TRUE(std::isfinite(value));
      if (id < 0 || id >= count) { continue; }
      const float expected = reference_value(
        stored_q.data() + row * dimensions, stored_x.data() + id * dimensions, dimensions, metric);
      EXPECT_NEAR(value, expected, 1.e-6f + 2.e-5f * std::abs(expected));
      if (k) {
        EXPECT_NE(out.indices[offset - 1], id);
        EXPECT_LE(
          (std::pair{rank_value(out.distances[offset - 1], metric), out.indices[offset - 1]}),
          (std::pair{rank_value(value, metric), static_cast<int64_t>(id)}));
      }
    }
  }
  return out;
}

template <typename DataT, typename IdxT>
TopkOutput check_typed_pk2(const std::vector<float>& q,
                           const std::vector<float>& x,
                           int d,
                           cuvs::distance::DistanceType metric,
                           int splits)
{
  return check_typed_topk<2, DataT, IdxT>(q, x, d, metric, splits, 2);
}

template <typename DataT, typename IdxT>
TopkOutput check_typed_pk4(const std::vector<float>& q,
                           const std::vector<float>& x,
                           int d,
                           cuvs::distance::DistanceType metric,
                           int splits,
                           int k)
{
  return check_typed_topk<4, DataT, IdxT>(q, x, d, metric, splits, k);
}

TEST(FusedTopkPk2PolicyTest, CapabilityPolicyRejectsInvalidExtentsAndMetrics)
{
  using DT                = cuvs::distance::DistanceType;
  constexpr int64_t limit = std::numeric_limits<int>::max();
  for (int64_t invalid : {int64_t{-1}, int64_t{0}, limit + 1}) {
    EXPECT_FALSE(is_fused_topk_shape_supported(invalid, 8192, 128, DT::L2Expanded));
    EXPECT_FALSE(is_fused_topk_shape_supported(8192, invalid, 128, DT::L2Expanded));
    EXPECT_FALSE(is_fused_topk_shape_supported(8192, 8192, invalid, DT::L2Expanded));
  }
  EXPECT_TRUE(is_fused_topk_shape_supported(limit, limit, limit, DT::L2Expanded));
  for (int k : {2, 3, 4}) {
    EXPECT_FALSE(k == 2 ? is_fused_topk_pk2_candidate(8192, 8192, 128, k, DT::L1, false)
                        : is_fused_topk_pk4_candidate(8192, 8192, 128, k, DT::L1, false));
  }
}

// Public tests deliberately do not skip on unavailable backends: permission
// must still preserve fallback. Only the direct comparison is capability gated.
template <int PhysicalK, typename DataT>
void check_public_typed_batch(int k,
                              int metric_id,
                              bool independent,
                              int64_t d           = 128,
                              int64_t query_rows  = 0,
                              int64_t corpus_rows = 0)
{
  const int64_t rows = query_rows ? query_rows : (independent ? 10000 : 8192);
  const int64_t n    = corpus_rows ? corpus_rows : rows;
  const auto metric  = static_cast<cuvs::distance::DistanceType>(metric_id);
  rmm::cuda_stream user_stream;
  raft::resources resources;
  raft::resource::set_cuda_stream(resources, user_stream);
  const auto stream = raft::resource::get_cuda_stream(resources);
  std::vector<DataT> q(rows * d), x(n * d);
  std::vector<float> qf(rows * d), xf(n * d);
  for (int64_t row = 0; row < std::max(rows, n); ++row) {
    for (int col = 0; col < d; ++col) {
      const auto i = row * d + col;
      if (row < n) {
        x[i]  = DataT{float((row * 13 + col * 7) % 41 - 20) / 16};
        xf[i] = static_cast<float>(x[i]);
      }
      if (row < rows) {
        q[i]  = DataT{float((row * 11 + col * 3 + 5) % 43 - 21) / 16};
        qf[i] = static_cast<float>(q[i]);
      }
    }
  }
  auto xd   = raft::make_device_matrix<DataT, int64_t>(resources, n, d);
  auto qd   = raft::make_device_matrix<DataT, int64_t>(resources, rows, d);
  auto ids  = raft::make_device_matrix<int64_t, int64_t>(resources, rows, k);
  auto vals = raft::make_device_matrix<float, int64_t>(resources, rows, k);
  raft::copy(xd.data_handle(), x.data(), x.size(), stream);
  raft::copy(qd.data_handle(), q.data(), q.size(), stream);
  auto corpus = raft::make_device_matrix_view<const DataT, int64_t>(xd.data_handle(), n, d);
  auto query  = raft::make_device_matrix_view<const DataT, int64_t>(
    independent ? qd.data_handle() : xd.data_handle(), rows, d);
  cuvs::neighbors::brute_force::index_params build;
  build.metric = metric;
  auto index   = cuvs::neighbors::brute_force::build(resources, build, corpus);
  cuvs::neighbors::brute_force::search_params params;
  ASSERT_FALSE(params.allow_approximate);
  const auto run = [&] {
    cuvs::neighbors::brute_force::search(resources, params, index, query, ids.view(), vals.view());
    return copy_output(resources, ids.data_handle(), vals.data_handle(), rows * k);
  };
  const auto defaults      = run();
  params.allow_approximate = false;
  const auto disabled      = run();
  EXPECT_EQ(defaults.indices, disabled.indices);
  EXPECT_EQ(defaults.distances, disabled.distances);
  params.allow_approximate = true;
  const auto permitted     = run();
  const bool available     = PhysicalK == 2 ? is_fused_topk_pk2_auto_dispatch_available()
                                            : is_fused_topk_pk4_auto_dispatch_available();
  const bool eligible      = PhysicalK == 2
                               ? is_fused_topk_pk2_candidate<DataT>(rows, n, d, k, metric, !independent)
                               : is_fused_topk_pk4_candidate<DataT>(rows, n, d, k, metric, !independent);
  if (available && eligible) {
    const int splits = fused_topk_public_num_splits(
      rows, n, raft::resource::get_device_properties(resources).multiProcessorCount);
    const int padded = fused_topk_padded_dimensions<DataT>(d);
    std::vector<float> padded_x(n * padded, 0), padded_q(rows * padded, 0);
    for (int64_t row = 0; row < n; ++row) {
      std::copy_n(xf.data() + row * d, d, padded_x.data() + row * padded);
    }
    const auto& original_q = independent ? qf : xf;
    for (int64_t row = 0; row < rows; ++row) {
      std::copy_n(original_q.data() + row * d, d, padded_q.data() + row * padded);
    }
    const auto direct = check_typed_topk<PhysicalK, DataT, int64_t>(
      padded_q, padded_x, padded, metric, splits, k, true);
    EXPECT_EQ(permitted.indices, direct.indices);
    EXPECT_EQ(permitted.distances, direct.distances);
  } else {
    EXPECT_EQ(permitted.indices, disabled.indices);
    EXPECT_EQ(permitted.distances, disabled.distances);
  }
}

class FusedTopkPk24PublicTest : public ::testing::TestWithParam<std::tuple<int, int, bool, bool>> {
};

TEST_P(FusedTopkPk24PublicTest, PermissionTypedLargeBatchAndGraphContracts)
{
  const auto [k, metric, use_half, independent] = GetParam();
  if (k == 2) {
    if (use_half) {
      check_public_typed_batch<2, half>(k, metric, independent);
    } else {
      check_public_typed_batch<2, float>(k, metric, independent);
    }
  } else {
    if (use_half) {
      check_public_typed_batch<4, half>(k, metric, independent);
    } else {
      check_public_typed_batch<4, float>(k, metric, independent);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(PublicQualified,
                         FusedTopkPk24PublicTest,
                         ::testing::Combine(::testing::Values(2, 3, 4),
                                            ::testing::Values(0, 1, 2, 6),
                                            ::testing::Bool(),
                                            ::testing::Bool()));

TEST(FusedTopkPk24PublicTest, UnequalRowsAndUnlistedFeatureWidths)
{
  for (int k : {2, 3, 4}) {
    for (int metric : {0, 1, 2, 6}) {
      for (int d : {16, 33, 129}) {
        if (k == 2) {
          check_public_typed_batch<2, float>(k, metric, true, d, 129, 263);
          check_public_typed_batch<2, half>(k, metric, true, d, 129, 263);
        } else {
          check_public_typed_batch<4, float>(k, metric, true, d, 129, 263);
          check_public_typed_batch<4, half>(k, metric, true, d, 129, 263);
        }
      }
    }
  }
}

TEST(FusedTopkPk2PolicyTest, PublicSplitGeometryIsBoundedAndShapeIndependent)
{
  // Arithmetic-only checks: these synthetic spans are never dereferenced.
  auto valid =
    [](
      uintptr_t ids, uintptr_t values, uintptr_t q, uintptr_t x, int64_t rows = 2, int64_t d = 33) {
      return fused_topk_public_buffers_supported(reinterpret_cast<int64_t*>(ids),
                                                 reinterpret_cast<float*>(values),
                                                 reinterpret_cast<const float*>(q),
                                                 reinterpret_cast<const float*>(x),
                                                 rows,
                                                 rows,
                                                 d,
                                                 2);
    };
  EXPECT_TRUE(valid(0x1000, 0x1020, 0x2000, 0x2000));
  EXPECT_FALSE(valid(0x1000, 0x1010, 0x2000, 0x3000));
  EXPECT_FALSE(valid(0x1000, 0x1020, 0x1000, 0x3000));
  EXPECT_FALSE(valid(0x1000, 0x1020, 0x2000, 0x1020));
  EXPECT_FALSE(valid(0x1000, 0x1020, 0, 0x3000));
  EXPECT_FALSE(valid(0x1000, 0x1020, 0x2004, 0x3000));
  EXPECT_FALSE(valid(0x1000, 0x1020, std::numeric_limits<uintptr_t>::max() - 15, 0x3000));
  EXPECT_FALSE(valid(0x1000, 0x1020, 0x2000, 0x3000, std::numeric_limits<int64_t>::max()));
  EXPECT_EQ(fused_topk_padded_dimensions<float>(31), 32);
  EXPECT_EQ(fused_topk_padded_dimensions<float>(33), 36);
  EXPECT_EQ(fused_topk_padded_dimensions<half>(33), 40);
  EXPECT_EQ(fused_topk_padded_dimensions<float>(513), 516);
  EXPECT_EQ(fused_topk_padded_dimensions<float>(36), 36);
  EXPECT_EQ(fused_topk_padded_dimensions<half>(128), 128);
  EXPECT_EQ(fused_topk_padded_dimensions<half>(std::numeric_limits<int>::max()),
            std::numeric_limits<int>::max());
  EXPECT_EQ(fused_topk_public_num_splits(128, 100000, 188), 64);
  EXPECT_EQ(fused_topk_public_num_splits(4096, 16384, 188), 2);
  EXPECT_EQ(fused_topk_public_num_splits(8192, 8192, 188), 1);
  EXPECT_EQ(fused_topk_public_num_splits(129, 263, 188), 3);
  EXPECT_EQ(fused_topk_public_num_splits(1, 1, 188), 1);
  EXPECT_EQ(fused_topk_public_num_splits(0, 263, 188), 0);
  EXPECT_EQ(fused_topk_public_num_splits(129, 0, 188), 0);
  EXPECT_EQ(fused_topk_public_num_splits(129, 263, 0), 0);
  for (int q : {1, 16, 128, 129, 4096, 8192, std::numeric_limits<int>::max()}) {
    for (int n : {1, 127, 128, 129, 100000, std::numeric_limits<int>::max()}) {
      const int splits = fused_topk_public_num_splits(q, n, 188);
      EXPECT_GE(splits, 1);
      EXPECT_LE(splits, 128);
      EXPECT_LE(splits, fused_topk_pk2_ceil_div(n, 128));
    }
  }
}

class FusedTopkPk2HalfTest : public ::testing::TestWithParam<TopkCase> {
 protected:
  void SetUp() override
  {
    if (!is_fused_topk_pk2_auto_dispatch_available()) {
      GTEST_SKIP() << "typed FP16 exports require an enabled SM120 cuTile backend";
    }
  }
};

// Independent query/corpus batches exercise both input types using exactly
// representable dyadic values. Near-tie fixtures separately cover approximate
// membership; these tests check the public search and stream contracts.
class FusedTopkPk24IndependentTest
  : public ::testing::TestWithParam<std::tuple<int, int, int, bool>> {};

template <int PhysicalK, typename DataT>
void check_independent_batch(int logical_k, int dimensions, int metric_id)
{
  constexpr int query_rows = 257, corpus_rows = 263;
  const auto metric = static_cast<cuvs::distance::DistanceType>(metric_id);
  std::vector<float> q(query_rows * dimensions), x(corpus_rows * dimensions);
  for (int row = 0; row < query_rows; ++row) {
    for (int col = 0; col < dimensions; ++col) {
      q[row * dimensions + col] = float(((row + 3) * 7 + col * 11) % 41 - 20) / 16;
    }
  }
  for (int row = 0; row < corpus_rows; ++row) {
    for (int col = 0; col < dimensions; ++col) {
      x[row * dimensions + col] = float(((row + 5) * 13 + col * 3) % 43 - 21) / 16;
    }
  }
  for (int splits : {1, 3}) {
    SCOPED_TRACE(::testing::Message() << "splits=" << splits);
    const auto a =
      check_typed_topk<PhysicalK, DataT, int>(q, x, dimensions, metric, splits, logical_k, true);
    const auto b = check_typed_topk<PhysicalK, DataT, int64_t>(
      q, x, dimensions, metric, splits, logical_k, true);
    EXPECT_EQ(a.indices, b.indices);
    EXPECT_EQ(a.distances, b.distances);
    // Full oracle on this deliberately exact dyadic L2/IP fixture, including
    // index tie breaking. Cosine/sqrt still receive every selected-value and
    // ordering check in check_typed_topk, without inventing a universal exactness
    // guarantee for their floating-point finalization.
    if (metric_id == 0 || metric_id == 6) {
      for (int row = 0; row < query_rows; ++row) {
        std::vector<std::pair<float, int64_t>> oracle;
        for (int j = 0; j < corpus_rows; ++j) {
          const float value = reference_value(
            q.data() + row * dimensions, x.data() + j * dimensions, dimensions, metric);
          oracle.emplace_back(rank_value(value, metric), j);
        }
        std::sort(oracle.begin(), oracle.end());
        for (int k = 0; k < logical_k; ++k) {
          EXPECT_EQ(a.indices[row * logical_k + k], oracle[k].second);
        }
      }
    }
  }
}

TEST_P(FusedTopkPk24IndependentTest, NondefaultStreamTypedSplitAndTailContracts)
{
  if (!is_fused_topk_pk2_auto_dispatch_available()) {
    GTEST_SKIP() << "this typed independent-query fixture targets enabled SM120 exports";
  }
  const auto [k, dimensions, metric, use_half] = GetParam();
  if (k == 2) {
    if (use_half) {
      check_independent_batch<2, half>(k, dimensions, metric);
    } else {
      check_independent_batch<2, float>(k, dimensions, metric);
    }
  } else {
    if (use_half) {
      check_independent_batch<4, half>(k, dimensions, metric);
    } else {
      check_independent_batch<4, float>(k, dimensions, metric);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(IndependentQueries,
                         FusedTopkPk24IndependentTest,
                         ::testing::Combine(::testing::Values(2, 3, 4),
                                            ::testing::Values(32, 33),
                                            ::testing::Values(0, 1, 2, 6),
                                            ::testing::Bool()));

TEST_P(FusedTopkPk2HalfTest, StoredHalfRefinementGuardsAndBothIndexWidths)
{
  const auto [d, metric, unused] = GetParam();
  std::vector<float> q(129 * d), x(259 * d);
  for (size_t i = 0; i < q.size(); ++i) {
    q[i] = std::sin(i * .071f) * 2;
  }
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = std::cos(i * .053f) * 2;
  }
  for (int splits : {1, 2}) {
    const auto a = check_typed_pk2<half, int>(q, x, d, metric, splits);
    const auto b = check_typed_pk2<half, int64_t>(q, x, d, metric, splits);
    EXPECT_EQ(a.indices, b.indices);
    EXPECT_EQ(a.distances, b.distances);
  }
}

INSTANTIATE_TEST_SUITE_P(
  StoredHalf,
  FusedTopkPk2HalfTest,
  ::testing::Values(TopkCase{32, cuvs::distance::DistanceType::L2Expanded, 1},
                    TopkCase{33, cuvs::distance::DistanceType::L2Expanded, 1},
                    TopkCase{32, cuvs::distance::DistanceType::L2SqrtExpanded, 1},
                    TopkCase{33, cuvs::distance::DistanceType::L2SqrtExpanded, 1},
                    TopkCase{32, cuvs::distance::DistanceType::CosineExpanded, 1},
                    TopkCase{33, cuvs::distance::DistanceType::CosineExpanded, 1},
                    TopkCase{32, cuvs::distance::DistanceType::InnerProduct, 1},
                    TopkCase{33, cuvs::distance::DistanceType::InnerProduct, 1}));

TEST_F(FusedTopkPk2HalfTest, RefinesEveryRetainedCandidateBeforeOrdering)
{
  for (int d : {32, 33}) {
    std::vector<float> q(3 * d, 0), x(2 * d, 0);
    for (int row = 0; row < 3; ++row) {
      q[row * d] = 1024;
    }
    x[0] = x[d] = 1024;
    x[1]        = .125f;
    x[d + 1]    = .0625f;
    for (auto metric :
         {cuvs::distance::DistanceType::L2Expanded, cuvs::distance::DistanceType::L2SqrtExpanded}) {
      const auto a = check_typed_pk2<half, int>(q, x, d, metric, 1);
      const auto b = check_typed_pk2<half, int64_t>(q, x, d, metric, 1);
      EXPECT_EQ(a.indices, b.indices);
      EXPECT_EQ(a.distances, b.distances);
      for (int row = 0; row < 3; ++row) {
        EXPECT_EQ(a.indices[row * 2], 1);
        EXPECT_EQ(a.indices[row * 2 + 1], 0);
        EXPECT_EQ(a.distances[row * 2],
                  metric == cuvs::distance::DistanceType::L2Expanded ? .00390625f : .0625f);
        EXPECT_EQ(a.distances[row * 2 + 1],
                  metric == cuvs::distance::DistanceType::L2Expanded ? .015625f : .125f);
      }
    }
  }
}

TEST_F(FusedTopkPk2HalfTest, WideValuesSubnormalsZeroCosineAndShortCorpus)
{
  for (int d : {32, 33}) {
    for (float magnitude : {1024.0f, std::ldexp(1.0f, -24), 0.0f}) {
      std::vector<float> q(2 * d, 0), x(2 * d, 0);
      q[0] = magnitude;
      q[d] = -magnitude;
      x[0] = -magnitude;
      x[d] = magnitude;
      for (auto metric : {cuvs::distance::DistanceType::L2Expanded,
                          cuvs::distance::DistanceType::L2SqrtExpanded,
                          cuvs::distance::DistanceType::CosineExpanded,
                          cuvs::distance::DistanceType::InnerProduct}) {
        const auto a = check_typed_pk2<half, int>(q, x, d, metric, 1);
        const auto b = check_typed_pk2<half, int64_t>(q, x, d, metric, 1);
        EXPECT_EQ(a.indices, b.indices);
        EXPECT_EQ(a.distances, b.distances);
        if (magnitude == 0) { EXPECT_EQ(a.indices, (std::vector<int64_t>{0, 1, 0, 1})); }
      }
    }
    check_typed_pk2<half, int>(std::vector<float>(d, 1),
                               std::vector<float>(d, 1),
                               d,
                               cuvs::distance::DistanceType::L2Expanded,
                               1);
    check_typed_pk2<half, int64_t>(std::vector<float>(d, 1),
                                   std::vector<float>(d, 1),
                                   d,
                                   cuvs::distance::DistanceType::L2Expanded,
                                   1);
  }
}

TEST_F(FusedTopkPk2HalfTest, DtypeAndLayoutCacheTransitions)
{
  for (int d : {32, 33, 32}) {
    std::vector<float> q(3 * d, .5f), x(259 * d, .25f);
    const auto f0 =
      check_typed_pk2<float, int64_t>(q, x, d, cuvs::distance::DistanceType::InnerProduct, 2);
    const auto h =
      check_typed_pk2<half, int64_t>(q, x, d, cuvs::distance::DistanceType::InnerProduct, 2);
    const auto f1 =
      check_typed_pk2<float, int64_t>(q, x, d, cuvs::distance::DistanceType::InnerProduct, 2);
    EXPECT_EQ(f0.indices, f1.indices);
    EXPECT_EQ(f0.distances, f1.distances);
    EXPECT_EQ(h.indices, f0.indices);
    EXPECT_EQ(h.distances, f0.distances);
  }
}

TEST_F(FusedTopkPk2HalfTest, ExactWinnersAcrossCorpusSplits)
{
  for (int d : {32, 33}) {
    std::vector<float> q(3 * d, 0), x(259 * d, 0);
    for (int row = 0; row < 3; ++row) {
      q[row * d] = 1;
    }
    for (int row = 0; row < 259; ++row) {
      x[row * d] = -4;
    }
    x[0] = x[258 * d] = -1;
    const auto a =
      check_typed_pk2<half, int>(q, x, d, cuvs::distance::DistanceType::InnerProduct, 2);
    const auto b =
      check_typed_pk2<half, int64_t>(q, x, d, cuvs::distance::DistanceType::InnerProduct, 2);
    EXPECT_EQ(a.indices, b.indices);
    EXPECT_EQ(a.distances, b.distances);
    for (int row = 0; row < 3; ++row) {
      EXPECT_EQ(a.indices[row * 2], 0);
      EXPECT_EQ(a.indices[row * 2 + 1], 258);
      EXPECT_EQ(a.distances[row * 2], -1);
      EXPECT_EQ(a.distances[row * 2 + 1], -1);
    }
  }
}

TEST(FusedTopkPk2PublicTest, UnsupportedHalfMetricPreservesFallback)
{
  constexpr int64_t rows = 8192, d = 16, k = 2;
  raft::resources resources;
  auto stream = raft::resource::get_cuda_stream(resources);
  std::vector<half> host(rows * d, half{.25f});
  for (int row = 0; row < rows; ++row) {
    host[row * d] = half{static_cast<float>(row % 29) / 16};
  }
  auto data   = raft::make_device_matrix<half, int64_t>(resources, rows, d);
  auto ids    = raft::make_device_matrix<int64_t, int64_t>(resources, rows, k);
  auto values = raft::make_device_matrix<float, int64_t>(resources, rows, k);
  raft::copy(data.data_handle(), host.data(), host.size(), stream);
  auto view = raft::make_device_matrix_view<const half, int64_t>(data.data_handle(), rows, d);
  for (auto metric : {cuvs::distance::DistanceType::L1}) {
    EXPECT_FALSE(is_fused_topk_pk2_candidate<half>(rows, rows, d, k, metric, true));
    cuvs::neighbors::brute_force::index_params build;
    build.metric = metric;
    auto index   = cuvs::neighbors::brute_force::build(resources, build, view);
    cuvs::neighbors::brute_force::search_params params;
    auto run = [&] {
      cuvs::neighbors::brute_force::search(
        resources, params, index, view, ids.view(), values.view());
      return copy_output(resources, ids.data_handle(), values.data_handle(), rows * k);
    };
    EXPECT_FALSE(params.allow_approximate);
    const auto defaults      = run();
    params.allow_approximate = false;
    const auto disabled      = run();
    params.allow_approximate = true;
    const auto permitted     = run();
    EXPECT_EQ(defaults.indices, disabled.indices);
    EXPECT_EQ(defaults.indices, permitted.indices);
    EXPECT_EQ(
      std::memcmp(defaults.distances.data(), disabled.distances.data(), rows * k * sizeof(float)),
      0);
    EXPECT_EQ(
      std::memcmp(defaults.distances.data(), permitted.distances.data(), rows * k * sizeof(float)),
      0);
  }
}

TEST_F(FusedTopkPk2HalfTest, TypedByteRangesAdjacentOutputsAndInvalidCalls)
{
  raft::resources resources;
  auto stream = raft::resource::get_cuda_stream(resources);
  // Two half rows occupy128bytes, not256; outputs immediately follow both input views.
  rmm::device_uvector<uint8_t> allocation(224, stream), workspace(16, stream);
  std::vector<uint8_t> host(224, 0xa5);
  std::vector<half> inputs(64, half{1.0f});
  for (int i = 32; i < 64; ++i) {
    inputs[i] = half{2.0f};
  }
  std::memcpy(host.data(), inputs.data(), 128);
  raft::copy(allocation.data(), host.data(), host.size(), stream);
  auto* data   = reinterpret_cast<half*>(allocation.data());
  auto* ids    = reinterpret_cast<int64_t*>(allocation.data() + 128);
  auto* values = reinterpret_cast<float*>(allocation.data() + 160);
  auto call =
    [&](const half* q, const half* x, int64_t m, int64_t n, int64_t d, int splits, void* scratch) {
      return try_fused_topk_pk2_tile<int64_t>(ids,
                                              values,
                                              q,
                                              x,
                                              m,
                                              n,
                                              d,
                                              splits,
                                              cuvs::distance::DistanceType::L2Expanded,
                                              stream.get(),
                                              scratch);
    };
  ASSERT_TRUE(call(data, data, 2, 2, 32, 1, workspace.data()));
  const auto out = copy_output(resources, ids, values, 4);
  EXPECT_EQ(out.indices, (std::vector<int64_t>{0, 1, 1, 0}));
  EXPECT_EQ(out.distances, (std::vector<float>{0, 32, 0, 32}));
  EXPECT_FALSE(call(data + 1, data, 2, 2, 32, 1, workspace.data()));
  EXPECT_FALSE(call(data, data + 1, 2, 2, 32, 1, workspace.data()));
  EXPECT_FALSE(call(nullptr, data, 2, 2, 32, 1, workspace.data()));
  EXPECT_FALSE(call(data, data, 0, 2, 32, 1, workspace.data()));
  EXPECT_FALSE(call(data, data, 2, 2, 0, 1, workspace.data()));
  EXPECT_FALSE(call(data, data, 2, 2, 32, 0, workspace.data()));
  EXPECT_FALSE(call(data, data, 2, 2, 32, 65536, workspace.data()));
  EXPECT_FALSE(call(data, data, 2, 2, 32, 2, workspace.data()));
  EXPECT_FALSE(call(data, data, 2, 2, 32, 1, nullptr));
  EXPECT_FALSE(call(data, data, 2, 2, 32, 1, allocation.data()));
  EXPECT_FALSE(call(data, data, 2, 2, 32, 1, workspace.data() + 1));
  EXPECT_FALSE(
    call(data, data, int64_t{std::numeric_limits<int>::max()} + 1, 2, 32, 1, workspace.data()));
  EXPECT_FALSE(
    call(data, data, 2, int64_t{std::numeric_limits<int>::max()} + 1, 32, 1, workspace.data()));
  EXPECT_FALSE(
    call(data, data, 2, 2, int64_t{std::numeric_limits<int>::max()} + 1, 1, workspace.data()));
  raft::copy(host.data(), allocation.data(), host.size(), stream);
  raft::resource::sync_stream(resources);
  EXPECT_EQ(std::memcmp(host.data(), inputs.data(), 128), 0);
  for (size_t i = 176; i < host.size(); ++i) {
    EXPECT_EQ(host[i], 0xa5);
  }
}

TEST(FusedTopkPk4PolicyTest, MergeGridArithmeticIsBoundedAtInt32Limit)
{
  static_assert(fused_topk_merge_block_count(0) == 0);
  static_assert(fused_topk_merge_block_count(1) == 1);
  static_assert(fused_topk_merge_block_count(256) == 1);
  static_assert(fused_topk_merge_block_count(257) == 2);
  static_assert(fused_topk_merge_block_count(std::numeric_limits<int>::max()) == 65535);
  EXPECT_EQ(fused_topk_merge_block_count(std::numeric_limits<int>::max() - 255), 65535);
}

class FusedTopkPk4HalfTest : public ::testing::TestWithParam<TopkCase> {
 protected:
  void SetUp() override
  {
    if (!is_fused_topk_pk4_auto_dispatch_available()) {
      GTEST_SKIP() << "typed FP16 exports require an enabled SM120 cuTile backend";
    }
  }
};

TEST_P(FusedTopkPk4HalfTest, StoredHalfRefinementGuardsAndBothIndexWidths)
{
  const auto [d, metric, unused] = GetParam();
  std::vector<float> q(129 * d), x(259 * d);
  for (size_t i = 0; i < q.size(); ++i) {
    q[i] = std::sin(i * .071f) * 2;
  }
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = std::cos(i * .053f) * 2;
  }
  for (int k : {3, 4}) {
    for (int splits : {1, 2}) {
      const auto a = check_typed_pk4<half, int>(q, x, d, metric, splits, k);
      const auto b = check_typed_pk4<half, int64_t>(q, x, d, metric, splits, k);
      EXPECT_EQ(a.indices, b.indices);
      EXPECT_EQ(a.distances, b.distances);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
  StoredHalf,
  FusedTopkPk4HalfTest,
  ::testing::Values(TopkCase{32, cuvs::distance::DistanceType::L2Expanded, 1},
                    TopkCase{33, cuvs::distance::DistanceType::L2Expanded, 1},
                    TopkCase{32, cuvs::distance::DistanceType::L2SqrtExpanded, 1},
                    TopkCase{33, cuvs::distance::DistanceType::L2SqrtExpanded, 1},
                    TopkCase{32, cuvs::distance::DistanceType::CosineExpanded, 1},
                    TopkCase{33, cuvs::distance::DistanceType::CosineExpanded, 1},
                    TopkCase{32, cuvs::distance::DistanceType::InnerProduct, 1},
                    TopkCase{33, cuvs::distance::DistanceType::InnerProduct, 1}));

TEST_F(FusedTopkPk4HalfTest, RefinesAllFourBeforeLogicalPrefix)
{
  for (int d : {32, 33}) {
    std::vector<float> q(3 * d, 0), x(259 * d, 0);
    for (int row = 0; row < 3; ++row)
      q[row * d] = 1024;
    for (int row = 0; row < 4; ++row) {
      x[row * d]     = 1024;
      x[row * d + 1] = static_cast<float>(4 - row) / 16;
    }
    for (auto metric :
         {cuvs::distance::DistanceType::L2Expanded, cuvs::distance::DistanceType::L2SqrtExpanded}) {
      for (int k : {3, 4})
        for (int splits : {1, 2}) {
          const auto a = check_typed_pk4<half, int>(q, x, d, metric, splits, k);
          const auto b = check_typed_pk4<half, int64_t>(q, x, d, metric, splits, k);
          EXPECT_EQ(a.indices, b.indices);
          EXPECT_EQ(a.distances, b.distances);
          for (int row = 0; row < 3; ++row)
            for (int slot = 0; slot < k; ++slot) {
              EXPECT_EQ(a.indices[row * k + slot], 3 - slot);
              const float j = slot + 1;
              EXPECT_EQ(a.distances[row * k + slot],
                        metric == cuvs::distance::DistanceType::L2Expanded ? j * j / 256 : j / 16);
            }
        }
    }
  }
}

TEST_F(FusedTopkPk4HalfTest, SqrtCutoffTiesAndCrossSplitWinners)
{
  for (int d : {32, 33}) {
    std::vector<float> q(d, 0), x(259 * d, 0);
    for (int row = 0; row < 259; ++row) {
      x[row * d]     = 4;
      x[row * d + 1] = 4;
    }
    for (int row : {0, 1, 256, 258}) {
      x[row * d]     = 1;
      x[row * d + 1] = 0;
    }
    x[1]       = 3.0f / 8192;
    x[d]       = .5f;
    x[258 * d] = 2;
    for (int k : {3, 4})
      for (int splits : {1, 2}) {
        const auto a = check_typed_pk4<half, int>(
          q, x, d, cuvs::distance::DistanceType::L2SqrtExpanded, splits, k);
        const auto b = check_typed_pk4<half, int64_t>(
          q, x, d, cuvs::distance::DistanceType::L2SqrtExpanded, splits, k);
        EXPECT_EQ(a.indices, b.indices);
        EXPECT_EQ(a.distances, b.distances);
        const std::vector<int64_t> ids{1, 0, 256, 258};
        const std::vector<float> values{.5f, 1, 1, 2};
        for (int slot = 0; slot < k; ++slot) {
          EXPECT_EQ(a.indices[slot], ids[slot]);
          EXPECT_EQ(a.distances[slot], values[slot]);
        }
      }
  }
}

TEST_F(FusedTopkPk4HalfTest, DtypeAndLayoutCacheTransitions)
{
  for (int d : {32, 33, 32})
    for (int k : {3, 4, 3}) {
      std::vector<float> q(3 * d, .5f), x(259 * d, .25f);
      const auto a =
        check_typed_pk4<float, int64_t>(q, x, d, cuvs::distance::DistanceType::InnerProduct, 2, k);
      const auto b =
        check_typed_pk4<half, int64_t>(q, x, d, cuvs::distance::DistanceType::InnerProduct, 2, k);
      const auto c =
        check_typed_pk4<float, int64_t>(q, x, d, cuvs::distance::DistanceType::InnerProduct, 2, k);
      EXPECT_EQ(a.indices, b.indices);
      EXPECT_EQ(a.distances, b.distances);
      EXPECT_EQ(a.indices, c.indices);
      EXPECT_EQ(a.distances, c.distances);
    }
}

TEST_F(FusedTopkPk4HalfTest, WideSubnormalZeroAndShortCorpus)
{
  for (int d : {32, 33})
    for (int k : {3, 4}) {
      for (float magnitude : {1024.0f, std::ldexp(1.0f, -24), 0.0f}) {
        std::vector<float> q(2 * d, 0), x(4 * d, 0);
        q[0] = magnitude;
        q[d] = -magnitude;
        for (int row = 0; row < 4; ++row)
          x[row * d] = (row % 2 ? -magnitude : magnitude);
        for (auto metric : {cuvs::distance::DistanceType::L2Expanded,
                            cuvs::distance::DistanceType::L2SqrtExpanded,
                            cuvs::distance::DistanceType::CosineExpanded,
                            cuvs::distance::DistanceType::InnerProduct}) {
          const auto a = check_typed_pk4<half, int>(q, x, d, metric, 1, k);
          const auto b = check_typed_pk4<half, int64_t>(q, x, d, metric, 1, k);
          EXPECT_EQ(a.indices, b.indices);
          EXPECT_EQ(a.distances, b.distances);
        }
      }
      check_typed_pk4<half, int>(std::vector<float>(d, 1),
                                 std::vector<float>(d, 1),
                                 d,
                                 cuvs::distance::DistanceType::L2Expanded,
                                 1,
                                 k);
      check_typed_pk4<half, int64_t>(std::vector<float>(d, 1),
                                     std::vector<float>(d, 1),
                                     d,
                                     cuvs::distance::DistanceType::L2Expanded,
                                     1,
                                     k);
    }
}
TEST(FusedTopkPk4PublicTest, UnsupportedHalfMetricPreservesFallback)
{
  constexpr int64_t rows = 8192, d = 16;
  for (int64_t k : {3, 4}) {
    raft::resources resources;
    auto stream = raft::resource::get_cuda_stream(resources);
    std::vector<half> host(rows * d, half{.25f});
    for (int row = 0; row < rows; ++row) {
      host[row * d] = half{static_cast<float>(row % 29) / 16};
    }
    auto data   = raft::make_device_matrix<half, int64_t>(resources, rows, d);
    auto ids    = raft::make_device_matrix<int64_t, int64_t>(resources, rows, k);
    auto values = raft::make_device_matrix<float, int64_t>(resources, rows, k);
    raft::copy(data.data_handle(), host.data(), host.size(), stream);
    auto view = raft::make_device_matrix_view<const half, int64_t>(data.data_handle(), rows, d);
    for (auto metric : {cuvs::distance::DistanceType::L1}) {
      EXPECT_FALSE(is_fused_topk_pk4_candidate<half>(rows, rows, d, k, metric, true));
      cuvs::neighbors::brute_force::index_params build;
      build.metric = metric;
      auto index   = cuvs::neighbors::brute_force::build(resources, build, view);
      cuvs::neighbors::brute_force::search_params params;
      auto run = [&] {
        cuvs::neighbors::brute_force::search(
          resources, params, index, view, ids.view(), values.view());
        return copy_output(resources, ids.data_handle(), values.data_handle(), rows * k);
      };
      EXPECT_FALSE(params.allow_approximate);
      const auto defaults      = run();
      params.allow_approximate = false;
      const auto disabled      = run();
      params.allow_approximate = true;
      const auto permitted     = run();
      EXPECT_EQ(defaults.indices, disabled.indices);
      EXPECT_EQ(defaults.indices, permitted.indices);
      EXPECT_EQ(
        std::memcmp(defaults.distances.data(), disabled.distances.data(), rows * k * sizeof(float)),
        0);
      EXPECT_EQ(std::memcmp(
                  defaults.distances.data(), permitted.distances.data(), rows * k * sizeof(float)),
                0);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
  AlignedAndRelaxed,
  FusedTopkPk2Test,
  ::testing::Values(TopkCase{31, cuvs::distance::DistanceType::L2Expanded, 2},
                    TopkCase{32, cuvs::distance::DistanceType::L2SqrtExpanded, 1},
                    TopkCase{31, cuvs::distance::DistanceType::CosineExpanded, 2},
                    TopkCase{32, cuvs::distance::DistanceType::InnerProduct, 1}));

INSTANTIATE_TEST_SUITE_P(
  AlignedAndRelaxed,
  FusedTopkPk4Test,
  ::testing::Values(Pk4Case{3, 31, cuvs::distance::DistanceType::L2Expanded, 2},
                    Pk4Case{3, 32, cuvs::distance::DistanceType::InnerProduct, 1},
                    Pk4Case{4, 31, cuvs::distance::DistanceType::CosineExpanded, 2},
                    Pk4Case{4, 32, cuvs::distance::DistanceType::L2SqrtExpanded, 1}));

#include "fused_topk_stage_test.cuh"

}  // namespace
}  // namespace cuvs::neighbors::detail
