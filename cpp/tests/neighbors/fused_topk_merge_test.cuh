/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../src/neighbors/detail/cutile_topk/fused_topk_merge.cuh"
#include "../../src/neighbors/detail/cutile_topk/fused_topk_register_merge.cuh"
#include "../../src/neighbors/detail/cutile_topk/fused_topk_subgroup_merge.cuh"

#include <gtest/gtest.h>
#include <raft/core/error.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace cuvs::neighbors::detail {
namespace {

// Test the merge independently of approximate candidate generation. Inputs
// satisfy the producer contract: sorted local lists, globally unique valid IDs.
// A deliberately tiny launch grid forces grid-stride reuse in every variant.
template <int PhysicalK, typename IdxT>
void check_topk_merge(int logical_k, int splits, int rows, bool inner_product)
{
  constexpr float inf = std::numeric_limits<float>::infinity();
  constexpr int guard = 64;
  const size_t count  = size_t(rows) * splits * PhysicalK;
  std::vector<int> partial_ids(count, -1);
  // Invalid values must be ignored even when their payload is not +infinity.
  std::vector<float> partial_values(count, -12345.0f);
  std::vector<IdxT> expected_ids(size_t(rows) * logical_k, -1);
  std::vector<float> expected_values(expected_ids.size(), inf);
  const auto less = [inner_product](auto a, auto b) {
    const float x = inner_product ? -a.first : a.first;
    const float y = inner_product ? -b.first : b.first;
    return x < y || (x == y && a.second < b.second);
  };
  for (int row = 0; row < rows; ++row) {
    std::vector<std::pair<float, int>> all;
    for (int split = 0; split < splits; ++split) {
      std::vector<std::pair<float, int>> local;
      for (int j = 0; j < PhysicalK; ++j) {
        if (row % 7 == 0 || (row % 7 == 1 && j > 0) || (row % 7 == 2 && split % 2)) { continue; }
        const int id = split * PhysicalK + j;
        float value  = float((id * 17 + row * 3) % 11 - 5);
        if (row % 7 == 3) { value = id % 2 ? -0.0f : 0.0f; }
        if (row % 7 == 4) { value = inf; }
        if (row % 7 == 5) { value = -inf; }
        local.emplace_back(value, id);
      }
      std::sort(local.begin(), local.end(), less);
      for (size_t j = 0; j < local.size(); ++j) {
        const size_t offset    = (size_t(row) * splits + split) * PhysicalK + j;
        partial_ids[offset]    = local[j].second;
        partial_values[offset] = local[j].first;
        all.push_back(local[j]);
      }
    }
    std::sort(all.begin(), all.end(), less);
    for (int j = 0; j < logical_k && j < int(all.size()); ++j) {
      expected_ids[size_t(row) * logical_k + j]    = all[j].second;
      expected_values[size_t(row) * logical_k + j] = all[j].first;
    }
  }

  rmm::cuda_stream stream;
  rmm::device_uvector<int> ids(count, stream.view());
  rmm::device_uvector<float> values(count, stream.view());
  const size_t output_count = expected_ids.size() + 2 * guard;
  rmm::device_uvector<IdxT> output_ids(output_count, stream.view());
  rmm::device_uvector<float> output_values(output_count, stream.view());
  std::vector<IdxT> actual_ids(output_count, 1234567);
  std::vector<float> actual_values(output_count, 123456.0f);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    ids.data(), partial_ids.data(), count * sizeof(int), cudaMemcpyHostToDevice, stream.value()));
  RAFT_CUDA_TRY(cudaMemcpyAsync(values.data(),
                                partial_values.data(),
                                count * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  RAFT_CUDA_TRY(cudaMemcpyAsync(output_ids.data(),
                                actual_ids.data(),
                                output_count * sizeof(IdxT),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  RAFT_CUDA_TRY(cudaMemcpyAsync(output_values.data(),
                                actual_values.data(),
                                output_count * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  auto* out_ids    = output_ids.data() + guard;
  auto* out_values = output_values.data() + guard;
  if (splits == 2) {
    merge_fused_topk_register_heads<PhysicalK, 2><<<3, 256, 0, stream.value()>>>(
      out_ids, out_values, ids.data(), values.data(), rows, logical_k, splits, inner_product);
  } else if (splits <= 4) {
    merge_fused_topk_register_heads<PhysicalK, 4><<<3, 256, 0, stream.value()>>>(
      out_ids, out_values, ids.data(), values.data(), rows, logical_k, splits, inner_product);
  } else if (splits <= 8) {
    merge_fused_topk_subgroups<PhysicalK, 8><<<3, 256, 0, stream.value()>>>(
      out_ids, out_values, ids.data(), values.data(), rows, logical_k, splits, inner_product);
  } else if (splits <= 16) {
    merge_fused_topk_subgroups<PhysicalK, 16><<<3, 256, 0, stream.value()>>>(
      out_ids, out_values, ids.data(), values.data(), rows, logical_k, splits, inner_product);
  } else {
    merge_fused_topk_sorted_splits<PhysicalK><<<3, 32, 0, stream.value()>>>(
      out_ids, out_values, ids.data(), values.data(), rows, logical_k, splits, inner_product);
  }
  RAFT_CUDA_TRY(cudaGetLastError());
  RAFT_CUDA_TRY(cudaMemcpyAsync(actual_ids.data(),
                                output_ids.data(),
                                output_count * sizeof(IdxT),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  RAFT_CUDA_TRY(cudaMemcpyAsync(actual_values.data(),
                                output_values.data(),
                                output_count * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
  EXPECT_EQ(
    std::memcmp(actual_ids.data() + guard, expected_ids.data(), expected_ids.size() * sizeof(IdxT)),
    0);
  // Bitwise checking is intentional: preserves signed-zero and infinity payloads.
  EXPECT_EQ(
    std::memcmp(
      actual_values.data() + guard, expected_values.data(), expected_values.size() * sizeof(float)),
    0);
  for (size_t j = 0; j < output_count; ++j) {
    if (j < guard || j >= output_count - guard) {
      EXPECT_EQ(actual_ids[j], IdxT(1234567));
      EXPECT_EQ(actual_values[j], 123456.0f);
    }
  }
}

using MergeCase = std::tuple<int, int, bool>;  // Logical K, split count, int64 output.
class FusedTopkPk24MergeTest : public ::testing::TestWithParam<MergeCase> {};

TEST_P(FusedTopkPk24MergeTest, SortedListsPreserveBitsAtCapacityBoundaries)
{
  const auto [k, splits, wide] = GetParam();
  for (int rows : {7, 777}) {
    for (bool inner : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "K=" << k << " splits=" << splits << " rows=" << rows
                                        << " inner=" << inner << " wide=" << wide);
      if (k == 2) {
        if (wide) {
          check_topk_merge<2, int64_t>(k, splits, rows, inner);
        } else {
          check_topk_merge<2, int>(k, splits, rows, inner);
        }
      } else {
        if (wide) {
          check_topk_merge<4, int64_t>(k, splits, rows, inner);
        } else {
          check_topk_merge<4, int>(k, splits, rows, inner);
        }
      }
    }
  }
}

std::vector<MergeCase> topk_merge_cases()
{
  std::vector<MergeCase> result;
  for (int k : {2, 3, 4}) {
    for (int splits :
         {2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128}) {
      for (bool wide : {false, true}) {
        result.emplace_back(k, splits, wide);
      }
    }
  }
  return result;
}

INSTANTIATE_TEST_SUITE_P(CapacityBoundaries,
                         FusedTopkPk24MergeTest,
                         ::testing::ValuesIn(topk_merge_cases()));

}  // namespace
}  // namespace cuvs::neighbors::detail
