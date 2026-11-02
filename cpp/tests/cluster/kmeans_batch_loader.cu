/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/cluster/detail/kmeans_batch_loader.cuh"

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_memory_resource.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace cuvs::cluster::kmeans::detail {
namespace {

template <typename T>
__global__ void consume_batch(T const* input, T* output, std::size_t elements, int pass)
{
  // Keep consumers in flight while the loader refills the other slot. This exposes premature
  // device-slot reuse rather than relying on fast copies that could hide an event-ordering bug.
  if (threadIdx.x == 0) {
    auto const begin = clock64();
    while (clock64() - begin < 100000) {}
  }
  __syncthreads();
  for (std::size_t i = threadIdx.x; i < elements; i += blockDim.x) {
    output[i] = input[i] + T(pass);
  }
}

class readonly_mapping {
 public:
  explicit readonly_mapping(std::size_t bytes) : bytes_(bytes)
  {
    data_ = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data_ == MAP_FAILED) { throw std::runtime_error("test mmap failed"); }
  }
  ~readonly_mapping() { munmap(data_, bytes_); }
  void* data() { return data_; }
  void protect()
  {
    if (mprotect(data_, bytes_, PROT_READ) != 0) {
      throw std::runtime_error("test mprotect failed");
    }
  }

 private:
  void* data_;
  std::size_t bytes_;
};

template <typename T>
void check_host_pipeline(int rows, int batch_rows, bool prefetch, bool pinned, bool mixed = false)
{
  constexpr int features = 17;
  raft::device_resources resources;
  auto main_stream = raft::resource::get_cuda_stream(resources);
  auto copy_stream = main_stream;
  if (prefetch) {
    raft::resource::set_cuda_stream_pool(resources, std::make_shared<rmm::cuda_stream_pool>(1));
    copy_stream = raft::resource::get_stream_from_stream_pool(resources);
  }
  const std::size_t elements = static_cast<std::size_t>(rows) * features;
  readonly_mapping mapping(elements * sizeof(T));
  auto* values = static_cast<T*>(mapping.data());
  for (std::size_t i = 0; i < elements; ++i) {
    values[i] = T(i) + T(.25);
  }
  mapping.protect();

  T* host_pinned = nullptr;
  if (pinned || mixed) {
    RAFT_CUDA_TRY(cudaMallocHost(&host_pinned, elements * sizeof(T)));
    std::memcpy(host_pinned, values, elements * sizeof(T));
  }
  std::unique_ptr<T, decltype([](T* ptr) {
                    if (ptr != nullptr) { cudaFreeHost(ptr); }
                  })>
    pinned_owner(host_pinned);
  using view = raft::host_matrix_view<const T, int64_t>;
  std::vector<view> partitions;
  std::vector<int64_t> offsets;
  if (mixed) {
    const int first = rows / 3;
    partitions      = {raft::make_host_matrix_view<const T, int64_t>(nullptr, 0, features),
                       raft::make_host_matrix_view<const T, int64_t>(values, first, features),
                       raft::make_host_matrix_view<const T, int64_t>(nullptr, 0, features),
                       raft::make_host_matrix_view<const T, int64_t>(
                    host_pinned + first * features, rows - first, features)};
    offsets         = {0, 0, first, first};
  } else {
    partitions = {
      raft::make_host_matrix_view<const T, int64_t>(pinned ? host_pinned : values, rows, features)};
    offsets = {0};
  }
  rmm::device_uvector<T> output(elements, main_stream);
  std::vector<T> actual(elements);
  {
    kmeans_batch_loader<T, int64_t, false> loader(
      resources,
      partitions,
      batch_rows,
      copy_stream,
      raft::resource::get_workspace_resource_ref(resources),
      prefetch);
    cudaPointerAttributes attributes{};
    RAFT_CUDA_TRY(cudaPointerGetAttributes(&attributes, values));
    EXPECT_EQ(attributes.type, cudaMemoryTypeUnregistered);  // Never pin the complete input.
    for (int pass = 0; pass < 7; ++pass) {
      loader.start();
      for (std::size_t pos = 0; pos < loader.num_batches(); ++pos) {
        auto batch = loader.acquire(pos);
        loader.prefetch((pos + 1) % loader.num_batches());
        auto offset = (offsets[batch.partition()] + batch.offset()) * features;
        consume_batch<<<1, 128, 0, main_stream.get()>>>(
          batch.data(), output.data() + offset, batch.size() * features, pass);
        RAFT_CUDA_TRY(cudaGetLastError());
        loader.recycle(batch, (pos + (prefetch ? 2 : 1)) % loader.num_batches());
      }
      raft::update_host(actual.data(), output.data(), elements, main_stream);
      raft::resource::sync_stream(resources);
      for (std::size_t i = 0; i < elements; ++i) {
        ASSERT_EQ(actual[i], values[i] + T(pass)) << "pass=" << pass << " element=" << i;
      }
    }
  }
}

template <typename T>
class KMeansBatchLoaderTest : public ::testing::Test {};
using data_types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(KMeansBatchLoaderTest, data_types);

TYPED_TEST(KMeansBatchLoaderTest, ReadOnlyMmapRepeatedPassesAndTail)
{
  for (bool prefetch : {false, true}) {
    for (int batch : {1, 7, 19, 80}) {
      check_host_pipeline<TypeParam>(37, batch, prefetch, false);
    }
  }
}

TYPED_TEST(KMeansBatchLoaderTest, PinnedInputRetainsDirectPath)
{
  check_host_pipeline<TypeParam>(37, 7, true, true);
}

TYPED_TEST(KMeansBatchLoaderTest, MixedPartitionsAndEmptyPartitions)
{
  check_host_pipeline<TypeParam>(37, 7, true, false, true);
}

TEST(KMeansBatchLoaderCleanup, AbandonPendingCopies)
{
  raft::device_resources resources;
  raft::resource::set_cuda_stream_pool(resources, std::make_shared<rmm::cuda_stream_pool>(1));
  auto copy_stream = raft::resource::get_stream_from_stream_pool(resources);
  std::vector<float> input(113 * 17, 1.f);
  auto view = raft::make_host_matrix_view<const float, int64_t>(input.data(), 113, 17);
  for (int repeat = 0; repeat < 20; ++repeat) {
    kmeans_batch_loader<float, int64_t, false> loader(
      resources, view, 7, copy_stream, raft::resource::get_workspace_resource_ref(resources), true);
    loader.start();
    loader.prefetch(1);
    if (repeat % 2) {
      auto batch = loader.acquire(0);
      loader.recycle(batch, 2);
    }
  }
  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace
}  // namespace cuvs::cluster::kmeans::detail
