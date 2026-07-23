/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Regression test: warpReduce with raft::add_op template disambiguation (cu132)
 *
 * Validates that raft::warpReduce(val, raft::add_op{}) correctly resolves to the
 * raft implementation rather than CUB's version. The ambiguity caused ivf_pq
 * build failures on CUDA 13.2 where CUB scan kernels call warpReduce with
 * raft::add_op.
 *
 * Fixed in: zbrad/raft@cu132 (commit d1345188)
 */

#include "../test_utils.cuh"

#include <raft/core/operators.hpp>
#include <raft/util/reduction.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

namespace cuvs::regression {

RAFT_KERNEL warp_reduce_add_op_kernel(const int* input, int* result)
{
  assert(gridDim.x == 1);
  int val = input[threadIdx.x];
  val     = raft::warpReduce(val, raft::add_op{});
  if (threadIdx.x % 32 == 0) { atomicAdd(result, val); }
}

class WarpReduceAddOpTest : public testing::TestWithParam<std::vector<int>> {
 protected:
  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  rmm::device_uvector<int> arr_d{GetParam().size(), stream};

 public:
  WarpReduceAddOpTest() { raft::update_device(arr_d.data(), GetParam().data(), GetParam().size(), stream); }

  void run()
  {
    rmm::device_scalar<int> result_d(stream);
    constexpr int block_dim = 64;
    constexpr int grid_dim  = 1;
    warp_reduce_add_op_kernel<<<grid_dim, block_dim, 0, stream>>>(arr_d.data(), result_d.data());
    stream.synchronize();
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    // 64 threads in 2 warps of 32; each warp reduces its 32 elements via add_op.
    // Both warp leaders atomicAdd their sums → total equals sum of all 64 elements.
    int expected = 0;
    for (int v : GetParam()) { expected += v; }
    ASSERT_EQ(result_d.value(stream), expected);
  }
};

// clang-format off
const std::vector<int> test_vector{
  1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 4, 1, 2,
  3, 4, 1, 2, 0, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
  1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
// clang-format on

TEST_P(WarpReduceAddOpTest, DisambiguatesFromCub) { run(); }
INSTANTIATE_TEST_SUITE_P(WarpReduceAddOpRegression, WarpReduceAddOpTest,
                         ::testing::Values(test_vector));  // NOLINT

}  // namespace cuvs::regression
