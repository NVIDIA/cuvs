/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../src/neighbors/detail/cagra/graph_core.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <gtest/gtest.h>
#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <rmm/mr/callback_memory_resource.hpp>

#include <new>

// This test targets public API exposure and basic invariants only (shapes, in-range indices).
// Detailed optimization correctness is exercised by CAGRA build tests.

namespace {

using IdxT = uint32_t;

// Helper to create a simple synthetic KNN graph (ring-like neighbors)
auto make_ring_knn_host(int64_t num_rows, int64_t kin)
{
  auto knn_graph = raft::make_host_matrix<IdxT, int64_t>(num_rows, kin);
  for (int64_t i = 0; i < num_rows; ++i) {
    for (int64_t j = 0; j < kin; ++j) {
      knn_graph(i, j) = static_cast<IdxT>((i + j + 1) % num_rows);
    }
  }
  return knn_graph;
}

TEST(CagraOptimize, HostToHostOptimizesGraph)
{
  raft::resources res;

  constexpr int64_t num_rows = 8;
  constexpr int64_t kin      = 8;
  constexpr int64_t kout     = 4;

  auto knn_graph       = make_ring_knn_host(num_rows, kin);
  auto optimized_graph = raft::make_host_matrix<IdxT, int64_t>(num_rows, kout);

  // Test the optimize API
  cuvs::neighbors::cagra::helpers::optimize(res, knn_graph.view(), optimized_graph.view());

  // Check basic invariants
  ASSERT_EQ(optimized_graph.extent(0), num_rows);
  ASSERT_EQ(optimized_graph.extent(1), kout);

  // Check that all neighbors are valid indices
  for (int64_t i = 0; i < num_rows; ++i) {
    for (int64_t j = 0; j < kout; ++j) {
      EXPECT_LT(optimized_graph(i, j), static_cast<IdxT>(num_rows));
    }
  }
}

TEST(CagraOptimize, DeviceToDeviceOptimizesGraph)
{
  raft::resources res;

  constexpr int64_t num_rows = 8;
  constexpr int64_t kin      = 8;
  constexpr int64_t kout     = 4;

  auto knn_graph_h = make_ring_knn_host(num_rows, kin);
  auto knn_graph_d = raft::make_device_matrix<IdxT, int64_t>(res, num_rows, kin);
  raft::copy(res, knn_graph_d.view(), raft::make_const_mdspan(knn_graph_h.view()));

  auto optimized_graph_d = raft::make_device_matrix<IdxT, int64_t>(res, num_rows, kout);

  cuvs::neighbors::cagra::helpers::optimize(res, knn_graph_d.view(), optimized_graph_d.view());

  ASSERT_EQ(optimized_graph_d.extent(0), num_rows);
  ASSERT_EQ(optimized_graph_d.extent(1), kout);

  auto optimized_graph_h = raft::make_host_matrix<IdxT, int64_t>(num_rows, kout);
  raft::copy(res, optimized_graph_h.view(), raft::make_const_mdspan(optimized_graph_d.view()));
  raft::resource::sync_stream(res);

  for (int64_t i = 0; i < num_rows; ++i) {
    for (int64_t j = 0; j < kout; ++j) {
      EXPECT_LT(optimized_graph_h(i, j), static_cast<IdxT>(num_rows));
    }
  }
}

// The device and host overloads must agree: same input, same optimized graph.
TEST(CagraOptimize, DeviceMatchesHost)
{
  raft::resources res;

  constexpr int64_t num_rows = 64;
  constexpr int64_t kin      = 16;
  constexpr int64_t kout     = 8;

  auto knn_graph_h = make_ring_knn_host(num_rows, kin);

  auto expected_h = raft::make_host_matrix<IdxT, int64_t>(num_rows, kout);
  cuvs::neighbors::cagra::helpers::optimize(res, knn_graph_h.view(), expected_h.view());

  auto knn_graph_d = raft::make_device_matrix<IdxT, int64_t>(res, num_rows, kin);
  raft::copy(res, knn_graph_d.view(), raft::make_const_mdspan(knn_graph_h.view()));
  auto actual_d = raft::make_device_matrix<IdxT, int64_t>(res, num_rows, kout);
  cuvs::neighbors::cagra::helpers::optimize(res, knn_graph_d.view(), actual_d.view());

  auto actual_h = raft::make_host_matrix<IdxT, int64_t>(num_rows, kout);
  raft::copy(res, actual_h.view(), raft::make_const_mdspan(actual_d.view()));
  raft::resource::sync_stream(res);

  for (int64_t i = 0; i < num_rows; ++i) {
    for (int64_t j = 0; j < kout; ++j) {
      EXPECT_EQ(actual_h(i, j), expected_h(i, j)) << "mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(CagraReverseGraph, HostFallbackMatchesDevice)
{
  raft::resources res;
  constexpr int64_t num_rows    = 64;
  constexpr int64_t degree      = 8;
  constexpr size_t column_bytes = num_rows * sizeof(IdxT);
  auto graph_h                  = make_ring_knn_host(num_rows, degree);
  auto graph_d                  = raft::make_device_matrix<IdxT, int64_t>(res, num_rows, degree);
  raft::copy(res, graph_d.view(), raft::make_const_mdspan(graph_h.view()));
  auto reverse_d       = raft::make_device_matrix<IdxT, int64_t>(res, num_rows, degree);
  auto counts_d        = raft::make_device_vector<uint32_t, int64_t>(res, num_rows);
  auto expected        = raft::make_host_matrix<IdxT, int64_t>(num_rows, degree);
  auto expected_counts = raft::make_host_vector<uint32_t, int64_t>(num_rows);
  cuvs::neighbors::cagra::detail::graph::make_reverse_graph_gpu(
    res, graph_d.view(), reverse_d.view(), counts_d.view());
  raft::copy(res, expected.view(), raft::make_const_mdspan(reverse_d.view()));
  raft::copy(res, expected_counts.view(), raft::make_const_mdspan(counts_d.view()));
  raft::resource::sync_stream(res);

  auto actual        = raft::make_host_matrix<IdxT, int64_t>(num_rows, degree);
  auto actual_counts = raft::make_host_vector<uint32_t, int64_t>(num_rows);
  auto check_host    = [&] {
    cuvs::neighbors::cagra::detail::graph::make_reverse_graph_gpu(
      res, graph_h.view(), reverse_d.view(), counts_d.view());
    raft::copy(res, actual.view(), raft::make_const_mdspan(reverse_d.view()));
    raft::copy(res, actual_counts.view(), raft::make_const_mdspan(counts_d.view()));
    raft::resource::sync_stream(res);
    for (int64_t i = 0; i < num_rows; ++i) {
      EXPECT_EQ(actual_counts(i), expected_counts(i));
      for (int64_t j = 0; j < degree; ++j) {
        EXPECT_EQ(actual(i, j), expected(i, j));
      }
    }
  };
  check_host();

  size_t rejected_allocations = 0;
  auto upstream               = rmm::mr::get_current_device_resource_ref();
  auto limited                = rmm::mr::callback_memory_resource{
    [&](cuda::stream_ref stream, size_t bytes, size_t alignment, void*) -> void* {
      if (bytes > column_bytes) {
        ++rejected_allocations;
        throw std::bad_alloc{};
      }
      return upstream.allocate(stream, bytes, alignment);
    },
    [&](cuda::stream_ref stream, void* ptr, size_t bytes, size_t alignment, void*) {
      auto mr = upstream;
      mr.deallocate(stream, ptr, bytes, alignment);
    }};
  raft::resource::set_large_workspace_resource(res, raft::mr::device_resource{limited});
  // The workspace allowance exceeds the allocator's per-allocation limit.
  raft::resource::set_workspace_resource(
    res, raft::mr::device_resource{limited}, column_bytes * degree);
  EXPECT_NO_THROW(check_host());
  EXPECT_EQ(rejected_allocations, 1);
}

}  // namespace
