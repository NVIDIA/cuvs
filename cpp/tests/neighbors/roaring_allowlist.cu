/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/roaring_allowlist.hpp>
#include <cuvs/neighbors/common.hpp>

#include <cuco/roaring_bitmap_ref.cuh>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/kernel_launch.hpp>

#include <gtest/gtest.h>

#include <rmm/cuda_stream.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace cuvs::core {
namespace {

roaring_allowlist from_ids(raft::resources const& res,
                           std::size_t dataset_rows,
                           std::vector<std::uint32_t> const& ids,
                           bool pre_sorted = false)
{
  return roaring_allowlist::from_ids(
    res,
    dataset_rows,
    raft::make_host_vector_view<const std::uint32_t, std::int64_t>(ids.data(), ids.size()),
    pre_sorted);
}

roaring_allowlist from_device_ids(raft::resources const& res,
                                  std::size_t dataset_rows,
                                  std::vector<std::uint32_t> const& ids,
                                  bool pre_sorted = false)
{
  auto device_ids   = raft::make_device_vector<std::uint32_t, std::int64_t>(res, ids.size());
  auto const stream = raft::resource::get_cuda_stream(res);
  raft::update_device(device_ids.data_handle(), ids.data(), ids.size(), stream);
  return roaring_allowlist::from_ids(
    res,
    dataset_rows,
    raft::make_device_vector_view<const std::uint32_t, std::int64_t>(device_ids.data_handle(),
                                                                     ids.size()),
    pre_sorted);
}

using ref_type = cuco::experimental::roaring_bitmap_ref<std::uint32_t>;

__global__ void membership_probe_kernel(ref_type const* reference,
                                        bool empty,
                                        std::uint32_t const* row_ids,
                                        std::uint8_t* output,
                                        std::size_t size)
{
  auto const i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) { output[i] = !empty && reference->contains(row_ids[i]); }
}

void expect_membership(raft::resources const& res,
                       roaring_allowlist const& allowlist,
                       std::vector<std::uint32_t> const& row_ids,
                       std::vector<std::uint8_t> const& expected)
{
  ASSERT_EQ(row_ids.size(), expected.size());
  auto rows   = raft::make_device_vector<std::uint32_t, std::int64_t>(res, row_ids.size());
  auto output = raft::make_device_vector<std::uint8_t, std::int64_t>(res, expected.size());
  auto stream = raft::resource::get_cuda_stream(res);
  raft::update_device(rows.data_handle(), row_ids.data(), row_ids.size(), stream);
  constexpr std::size_t block_size = 256;
  raft::launch_kernel(stream,
                      dim3((row_ids.size() + block_size - 1) / block_size),
                      dim3(block_size),
                      membership_probe_kernel,
                      static_cast<ref_type const*>(allowlist.view().device_reference()),
                      allowlist.empty(),
                      rows.data_handle(),
                      output.data_handle(),
                      row_ids.size());

  std::vector<std::uint8_t> actual(expected.size());
  raft::update_host(actual.data(), output.data_handle(), actual.size(), stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(actual, expected);
}

TEST(RoaringAllowlist, BuildsArrayBitmapAndMultipleContainers)
{
  raft::device_resources res;

  std::vector<std::uint32_t> ids{65537, 7, 3, 131074, 5};
  auto const original = ids;
  auto array          = from_ids(res, 200000, ids);
  EXPECT_EQ(ids, original);
  EXPECT_EQ(array.dataset_rows(), 200000);
  EXPECT_EQ(array.cardinality(), 5);
  EXPECT_FALSE(array.empty());
  EXPECT_GT(array.size_bytes(), 0);
  EXPECT_TRUE(array.view().valid());
  EXPECT_EQ(array.view().cardinality(), 5);
  expect_membership(res, array, {3, 4, 7, 65537, 131073, 131074, 200000}, {1, 0, 1, 1, 0, 1, 0});

  std::vector<std::uint32_t> dense;
  for (std::uint32_t id = 0; id < 10000; id += 2) {
    dense.push_back(id);
  }
  auto bitmap = from_ids(res, 10000, dense, true);
  EXPECT_EQ(bitmap.cardinality(), 5000);
  expect_membership(res, bitmap, {0, 1, 4096, 9998, 9999}, {1, 0, 1, 1, 0});
}

TEST(RoaringAllowlist, SupportsPresortedHostAndDeviceInputs)
{
  raft::device_resources res;
  std::vector<std::uint32_t> const ids{1, 4, 7, 65536, 65539};
  auto host   = from_ids(res, 131072, ids, true);
  auto device = from_device_ids(res, 131072, ids, true);

  EXPECT_EQ(host.cardinality(), ids.size());
  EXPECT_EQ(device.cardinality(), ids.size());
  expect_membership(res, host, {0, 1, 7, 8, 65539}, {0, 1, 1, 0, 1});
  expect_membership(res, device, {0, 1, 7, 8, 65539}, {0, 1, 1, 0, 1});
}

TEST(RoaringAllowlist, HandlesEmptyAndFullUint32Domain)
{
  raft::device_resources res;

  auto empty = from_ids(res, 32, {});
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.cardinality(), 0);
  EXPECT_TRUE(empty.view().valid());
  EXPECT_EQ(empty.view().device_reference(), nullptr);
  expect_membership(res, empty, {0, 31, 32}, {0, 0, 0});

  EXPECT_THROW(from_ids(res, 0, {}), raft::logic_error);

  auto maximum =
    from_ids(res, std::uint64_t{1} << 32, {0, std::numeric_limits<std::uint32_t>::max()}, true);
  expect_membership(res, maximum, {0, 1, std::numeric_limits<std::uint32_t>::max()}, {1, 0, 1});
}

TEST(RoaringAllowlist, ViewIsZeroCopyAndSurvivesOwnerMove)
{
  raft::device_resources res;
  auto allowlist  = from_ids(res, 32, {1, 4, 7});
  auto first_view = allowlist.view();
  auto next_view  = allowlist.view();
  EXPECT_EQ(first_view.device_reference(), next_view.device_reference());

  auto moved      = std::move(allowlist);
  auto moved_view = moved.view();
  EXPECT_EQ(first_view.device_reference(), moved_view.device_reference());
  EXPECT_EQ(moved_view.cardinality(), 3);
  expect_membership(res, moved, {1, 2, 7}, {1, 0, 1});
}

TEST(RoaringAllowlist, StreamOrderedConstructionSupportsEventHandoff)
{
  rmm::cuda_stream build_stream;
  rmm::cuda_stream consume_stream;
  raft::device_resources build_res;
  raft::device_resources consume_res;
  raft::resource::set_cuda_stream(build_res, build_stream.view());
  raft::resource::set_cuda_stream(consume_res, consume_stream.view());

  auto allowlist = from_ids(build_res, 1000, {900, 100, 300, 200});

  cudaEvent_t ready{};
  RAFT_CUDA_TRY(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
  RAFT_CUDA_TRY(cudaEventRecord(ready, build_stream.value()));
  RAFT_CUDA_TRY(cudaStreamWaitEvent(consume_stream.value(), ready));
  expect_membership(consume_res, allowlist, {99, 100, 200, 300, 900}, {0, 1, 1, 1, 1});
  RAFT_CUDA_TRY(cudaEventDestroy(ready));
}

TEST(RoaringFilter, ReusesViewsUpdatesMappingsAndRejectsInvalidInputs)
{
  raft::device_resources res;
  auto first  = from_ids(res, 16, {1, 3});
  auto second = from_ids(res, 16, {2, 4, 6});
  auto empty  = from_ids(res, 16, {});

  std::array views{first.view(), second.view(), first.view()};
  cuvs::neighbors::filtering::roaring_filter filter(res, views);
  EXPECT_TRUE(filter.valid());
  EXPECT_EQ(filter.num_queries(), 3);
  EXPECT_EQ(filter.dataset_rows(), 16);
  EXPECT_EQ(filter.cardinality(0), 2);
  EXPECT_EQ(filter.cardinality(1), 3);
  EXPECT_FLOAT_EQ(filter.filtering_rate(), 0.875f);

  auto const* payload = filter.device_payload();
  auto shared_copy    = filter;
  shared_copy.set_allowlist(res, 1, empty.view());
  EXPECT_EQ(filter.device_payload(), payload);
  EXPECT_TRUE(filter.empty(1));
  EXPECT_FLOAT_EQ(filter.filtering_rate(), 0.999f);

  EXPECT_THROW(cuvs::neighbors::filtering::roaring_filter(
                 res, std::span<const cuvs::core::roaring_allowlist_view>{}),
               raft::logic_error);
  auto different_shape = from_ids(res, 17, {1});
  std::array mismatched{first.view(), different_shape.view()};
  EXPECT_THROW(cuvs::neighbors::filtering::roaring_filter(res, mismatched), raft::logic_error);
  EXPECT_THROW(filter.set_allowlist(res, 3, first.view()), raft::logic_error);
  EXPECT_THROW(filter.set_allowlist(res, 0, different_shape.view()), raft::logic_error);
}

}  // namespace
}  // namespace cuvs::core
