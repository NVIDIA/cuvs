/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/distance/detail/distance_ops/bitwise_hamming.cuh"
#include "../../src/distance/detail/pairwise_matrix/dispatch-inl.cuh"
#include "../../src/distance/fused_distance_nn.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace cuvs::distance {
namespace {
using Index = int64_t;
using Pair  = raft::KeyValuePair<Index, uint32_t>;

uint32_t host_hamming(const uint8_t* x, const uint8_t* y, Index k)
{
  uint32_t distance = 0;
  for (Index j = 0; j < k; ++j) {
    auto bits = static_cast<unsigned int>(x[j] ^ y[j]);
    for (; bits != 0; bits >>= 1) {
      distance += bits & 1u;
    }
  }
  return distance;
}

std::vector<uint32_t> host_pairwise(
  const std::vector<uint8_t>& x, const std::vector<uint8_t>& y, Index m, Index n, Index k)
{
  std::vector<uint32_t> result(m * n);
  for (Index row = 0; row < m; ++row) {
    for (Index col = 0; col < n; ++col) {
      result[row * n + col] = k == 0 ? 0 : host_hamming(x.data() + row * k, y.data() + col * k, k);
    }
  }
  return result;
}

std::vector<uint8_t> column_major(const std::vector<uint8_t>& x, Index rows, Index cols)
{
  std::vector<uint8_t> result(x.size());
  for (Index row = 0; row < rows; ++row) {
    for (Index col = 0; col < cols; ++col) {
      result[col * rows + row] = x[row * cols + col];
    }
  }
  return result;
}

void copy_to_device(uint8_t* destination, const std::vector<uint8_t>& source, cudaStream_t stream)
{
  if (!source.empty()) {
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(destination, source.data(), source.size(), cudaMemcpyHostToDevice, stream));
  }
}

TEST(BitwiseHammingDistance, PairwiseLayoutsAlignmentAndByteTails)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  // Unequal row counts also exercise column-major transposition and partial tiles.
  constexpr Index m = 36, n = 68;
  std::mt19937 rng(42);
  for (Index k : {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 192, 193}) {
    std::vector<uint8_t> x(m * k), y(n * k);
    for (auto& value : x) {
      value = static_cast<uint8_t>(rng());
    }
    for (auto& value : y) {
      value = static_cast<uint8_t>(rng());
    }
    auto expected = host_pairwise(x, y, m, n, k);
    for (bool row_major : {true, false}) {
      auto x_storage = row_major ? x : column_major(x, m, k);
      auto y_storage = row_major ? y : column_major(y, n, k);
      for (size_t offset : {size_t{0}, size_t{1}, size_t{2}}) {
        SCOPED_TRACE(::testing::Message()
                     << "k=" << k << " row_major=" << row_major << " offset=" << offset);
        rmm::device_uvector<uint8_t> dx(std::max<size_t>(1, x.size() + offset), stream);
        rmm::device_uvector<uint8_t> dy(std::max<size_t>(1, y.size() + offset), stream);
        rmm::device_uvector<uint32_t> result(m * n, stream);
        copy_to_device(dx.data() + offset, x_storage, stream);
        copy_to_device(dy.data() + offset, y_storage, stream);
        using Op = detail::ops::bitwise_hamming_distance_op<uint8_t, uint32_t, Index>;
        detail::pairwise_matrix_dispatch<Op, uint8_t, uint32_t, uint32_t, raft::identity_op, Index>(
          Op{k},
          m,
          n,
          k,
          dx.data() + offset,
          dy.data() + offset,
          nullptr,
          nullptr,
          result.data(),
          raft::identity_op{},
          stream,
          row_major);
        std::vector<uint32_t> actual(m * n);
        RAFT_CUDA_TRY(cudaMemcpyAsync(actual.data(),
                                      result.data(),
                                      actual.size() * sizeof(uint32_t),
                                      cudaMemcpyDeviceToHost,
                                      stream));
        raft::resource::sync_stream(handle);
        for (Index row = 0; row < m; ++row) {
          for (Index col = 0; col < n; ++col) {
            ASSERT_EQ(actual[row_major ? row * n + col : col * m + row], expected[row * n + col]);
          }
        }
      }
    }
  }
}

TEST(BitwiseHammingDistance, FusedDistancesAbove255TiesAndByteTails)
{
  raft::resources handle;
  auto stream       = raft::resource::get_cuda_stream(handle);
  constexpr Index m = 35, n = 133;
  std::mt19937 rng(42);
  for (Index k : {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 192, 193}) {
    std::vector<uint8_t> x(m * k), y(n * k);
    for (auto& value : x) {
      value = static_cast<uint8_t>(rng() & 3);
    }
    for (auto& value : y) {
      value = static_cast<uint8_t>(0xfc | (rng() & 3));
    }
    // Duplicate the best candidate in separate tiles; the smaller index must win.
    for (Index j = 0; j < k; ++j) {
      y[5 * k + j] = y[132 * k + j] = 0x0f;
    }
    auto expected = host_pairwise(x, y, m, n, k);
    for (size_t offset : {size_t{0}, size_t{1}}) {
      SCOPED_TRACE(::testing::Message() << "k=" << k << " offset=" << offset);
      rmm::device_uvector<uint8_t> dx(std::max<size_t>(1, x.size() + offset), stream);
      rmm::device_uvector<uint8_t> dy(std::max<size_t>(1, y.size() + offset), stream);
      rmm::device_uvector<Pair> result(m, stream);
      rmm::device_uvector<int> workspace(m, stream);
      copy_to_device(dx.data() + offset, x, stream);
      copy_to_device(dy.data() + offset, y, stream);
      fusedDistanceNNMinReduce<uint8_t, Pair, Index>(result.data(),
                                                     dx.data() + offset,
                                                     dy.data() + offset,
                                                     nullptr,
                                                     nullptr,
                                                     m,
                                                     n,
                                                     k,
                                                     workspace.data(),
                                                     false,
                                                     true,
                                                     true,
                                                     DistanceType::BitwiseHamming,
                                                     0,
                                                     stream);
      std::vector<Pair> actual(m);
      RAFT_CUDA_TRY(cudaMemcpyAsync(actual.data(),
                                    result.data(),
                                    actual.size() * sizeof(Pair),
                                    cudaMemcpyDeviceToHost,
                                    stream));
      raft::resource::sync_stream(handle);
      for (Index row = 0; row < m; ++row) {
        auto begin = expected.begin() + row * n;
        auto best  = std::min_element(begin, begin + n);
        ASSERT_EQ(actual[row].key, best - begin);
        ASSERT_EQ(actual[row].value, *best);
      }
      // Updating an existing minimum must preserve a better earlier result.
      for (auto& value : actual) {
        value = Pair{0, 0};
      }
      RAFT_CUDA_TRY(cudaMemcpyAsync(result.data(),
                                    actual.data(),
                                    actual.size() * sizeof(Pair),
                                    cudaMemcpyHostToDevice,
                                    stream));
      fusedDistanceNNMinReduce<uint8_t, Pair, Index>(result.data(),
                                                     dx.data() + offset,
                                                     dy.data() + offset,
                                                     nullptr,
                                                     nullptr,
                                                     m,
                                                     n,
                                                     k,
                                                     workspace.data(),
                                                     false,
                                                     false,
                                                     true,
                                                     DistanceType::BitwiseHamming,
                                                     0,
                                                     stream);
      RAFT_CUDA_TRY(cudaMemcpyAsync(actual.data(),
                                    result.data(),
                                    actual.size() * sizeof(Pair),
                                    cudaMemcpyDeviceToHost,
                                    stream));
      raft::resource::sync_stream(handle);
      for (const auto& value : actual) {
        EXPECT_EQ(value.key, 0);
        EXPECT_EQ(value.value, 0u);
      }
    }
  }
}

TEST(BitwiseHammingDistance, ScalarOutputAndExact255)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  for (Index k : {32, 192}) {
    std::vector<uint8_t> x(k, 0), y(3 * k, 0xff);
    y[2 * k - 1] = 0x7f;
    rmm::device_uvector<uint8_t> dx(x.size(), stream), dy(y.size(), stream);
    rmm::device_uvector<uint32_t> result(1, stream);
    rmm::device_uvector<int> workspace(1, stream);
    copy_to_device(dx.data(), x, stream);
    copy_to_device(dy.data(), y, stream);
    fusedDistanceNNMinReduce<uint8_t, uint32_t, Index>(result.data(),
                                                       dx.data(),
                                                       dy.data(),
                                                       nullptr,
                                                       nullptr,
                                                       1,
                                                       3,
                                                       k,
                                                       workspace.data(),
                                                       false,
                                                       true,
                                                       true,
                                                       DistanceType::BitwiseHamming,
                                                       0,
                                                       stream);
    uint32_t actual{};
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(&actual, result.data(), sizeof(actual), cudaMemcpyDeviceToHost, stream));
    raft::resource::sync_stream(handle);
    EXPECT_EQ(actual, 8 * k - 1);
  }
}

TEST(BitwiseHammingDistance, EmptyInputsAndInvalidDimensions)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  EXPECT_NO_THROW((fusedDistanceNNMinReduce<uint8_t, Pair, Index>(nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  0,
                                                                  2,
                                                                  192,
                                                                  nullptr,
                                                                  false,
                                                                  true,
                                                                  true,
                                                                  DistanceType::BitwiseHamming,
                                                                  0,
                                                                  stream)));
  using Op = detail::ops::bitwise_hamming_distance_op<uint8_t, uint32_t, Index>;
  EXPECT_THROW((Op{-1}), raft::logic_error);
  EXPECT_THROW((Op{Index(std::numeric_limits<uint32_t>::max()) / 8 + 1}), raft::logic_error);
  EXPECT_THROW((fusedDistanceNNMinReduce<uint8_t, Pair, Index>(nullptr,
                                                               nullptr,
                                                               nullptr,
                                                               nullptr,
                                                               nullptr,
                                                               0,
                                                               2,
                                                               192,
                                                               nullptr,
                                                               false,
                                                               true,
                                                               true,
                                                               DistanceType::L2Expanded,
                                                               0,
                                                               stream)),
               raft::logic_error);
  EXPECT_THROW((fusedDistanceNNMinReduce<uint8_t, Pair, Index>(nullptr,
                                                               nullptr,
                                                               nullptr,
                                                               nullptr,
                                                               nullptr,
                                                               0,
                                                               2,
                                                               192,
                                                               nullptr,
                                                               false,
                                                               true,
                                                               false,
                                                               DistanceType::BitwiseHamming,
                                                               0,
                                                               stream)),
               raft::logic_error);
  for (Index m : {0, 3}) {
    EXPECT_NO_THROW(
      (detail::pairwise_matrix_dispatch<Op, uint8_t, uint32_t, uint32_t, raft::identity_op, Index>(
        Op{192},
        m,
        0,
        192,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        raft::identity_op{},
        stream,
        true)));
  }
  rmm::device_uvector<Pair> result(1, stream);
  rmm::device_uvector<int> workspace(1, stream);
  fusedDistanceNNMinReduce<uint8_t, Pair, Index>(result.data(),
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 1,
                                                 0,
                                                 192,
                                                 workspace.data(),
                                                 false,
                                                 true,
                                                 true,
                                                 DistanceType::BitwiseHamming,
                                                 0,
                                                 stream);
  Pair actual{};
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(&actual, result.data(), sizeof(actual), cudaMemcpyDeviceToHost, stream));
  raft::resource::sync_stream(handle);
  EXPECT_EQ(actual.value, std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(actual.key, std::numeric_limits<Index>::max());
}
}  // namespace
}  // namespace cuvs::distance
