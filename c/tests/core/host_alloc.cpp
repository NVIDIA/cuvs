/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

// Pinned host buffers may be freed on a different thread than the one that allocated them, e.g.
// when a binding moves the owning object across threads.
TEST(RMMHostAlloc, FreeOnAnotherThread)
{
  constexpr std::size_t kBytes = 1024;
  void* ptr                    = nullptr;
  cuvsError_t alloc_status     = CUVS_ERROR;
  std::thread([&] { alloc_status = cuvsRMMHostAlloc(&ptr, kBytes); }).join();
  ASSERT_EQ(alloc_status, CUVS_SUCCESS);
  ASSERT_NE(ptr, nullptr);

  cuvsError_t free_status = CUVS_ERROR;
  std::thread([&] { free_status = cuvsRMMHostFree(ptr, kBytes); }).join();
  EXPECT_EQ(free_status, CUVS_SUCCESS);
}

TEST(RMMHostAlloc, ConcurrentAllocAndFree)
{
  constexpr int kThreads       = 8;
  constexpr int kIters         = 50;
  constexpr std::size_t kBytes = 4096;
  std::vector<void*> ptrs(kThreads * kIters, nullptr);
  std::atomic<int> failures{0};

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < kIters; ++i) {
        if (cuvsRMMHostAlloc(&ptrs[t * kIters + i], kBytes) != CUVS_SUCCESS) { failures++; }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  threads.clear();
  ASSERT_EQ(failures.load(), 0);

  // Free every buffer from a different thread than the one that allocated it.
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      int const owner = (t + 1) % kThreads;
      for (int i = 0; i < kIters; ++i) {
        if (cuvsRMMHostFree(ptrs[owner * kIters + i], kBytes) != CUVS_SUCCESS) { failures++; }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  EXPECT_EQ(failures.load(), 0);
}
