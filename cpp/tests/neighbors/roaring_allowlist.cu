/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/roaring_allowlist.hpp>
#include <cuvs/neighbors/common.hpp>

#include <cuco/roaring_bitmap_ref.cuh>

#include "../../bench/ann/src/cuvs/allowlist_benchmark_utils.hpp"
#include "../../bench/ann/src/cuvs/cagra_filter_benchmark_utils.hpp"

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <gtest/gtest.h>

#include <rmm/cuda_stream.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuvs::core {
namespace {

// These hand-built fixtures exercise import of the portable bytes themselves.
// Keep their field order aligned with
// https://github.com/RoaringBitmap/RoaringFormatSpec.
constexpr std::uint32_t kCookieNoRun                  = 12346;
constexpr std::uint32_t kCookieRun                    = 12347;
constexpr std::uint32_t kSingleArrayPayloadByteOffset = 16;

void append_u16(std::vector<std::byte>& out, std::uint16_t value)
{
  out.push_back(static_cast<std::byte>(value & 0xffu));
  out.push_back(static_cast<std::byte>((value >> 8) & 0xffu));
}

void append_u32(std::vector<std::byte>& out, std::uint32_t value)
{
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xffu));
  }
}

// One no-run array container: cookie, N, {key, cardinality - 1}, payload
// offset, values.
std::vector<std::byte> array_row(std::vector<std::uint16_t> const& values)
{
  std::vector<std::byte> out;
  append_u32(out, kCookieNoRun);
  append_u32(out, 1);
  append_u16(out, 0);
  append_u16(out, static_cast<std::uint16_t>(values.size() - 1));
  append_u32(out, kSingleArrayPayloadByteOffset);
  for (auto value : values) {
    append_u16(out, value);
  }
  return out;
}

// One run container: run cookie, one-byte run bitmap, descriptor, then the run
// payload. A run-cookie row with fewer than four containers has no
// container-offset table.
std::vector<std::byte> run_row(std::uint16_t start, std::uint16_t length_minus_one)
{
  std::vector<std::byte> out;
  append_u32(out, kCookieRun);
  out.push_back(std::byte{1});
  append_u16(out, 0);
  append_u16(out, length_minus_one);
  append_u16(out, 1);
  append_u16(out, start);
  append_u16(out, length_minus_one);
  return out;
}

roaring_allowlist from_ragged_ids(raft::resources const& res,
                                  std::size_t dataset_rows,
                                  std::vector<std::uint32_t> const& ids,
                                  std::vector<std::int64_t> const& indptr,
                                  bool pre_sorted = false)
{
  return roaring_allowlist::from_ids(
    res,
    dataset_rows,
    raft::make_host_vector_view<const std::uint32_t, std::int64_t>(ids.data(), ids.size()),
    raft::make_host_vector_view<const std::int64_t, std::int64_t>(indptr.data(), indptr.size()),
    pre_sorted);
}

roaring_allowlist from_device_ragged_ids(raft::resources const& res,
                                         std::size_t dataset_rows,
                                         std::vector<std::uint32_t> const& ids,
                                         std::vector<std::int64_t> const& indptr,
                                         bool pre_sorted = false)
{
  auto device_ids    = raft::make_device_vector<std::uint32_t, std::int64_t>(res, ids.size());
  auto device_indptr = raft::make_device_vector<std::int64_t, std::int64_t>(res, indptr.size());
  auto const stream  = raft::resource::get_cuda_stream(res);
  raft::update_device(device_ids.data_handle(), ids.data(), ids.size(), stream);
  raft::update_device(device_indptr.data_handle(), indptr.data(), indptr.size(), stream);
  return roaring_allowlist::from_ids(
    res,
    dataset_rows,
    raft::make_device_vector_view<const std::uint32_t, std::int64_t>(device_ids.data_handle(),
                                                                     ids.size()),
    raft::make_device_vector_view<const std::int64_t, std::int64_t>(device_indptr.data_handle(),
                                                                    indptr.size()),
    pre_sorted);
}

roaring_allowlist from_ids(raft::resources const& res,
                           std::size_t dataset_rows,
                           std::vector<std::uint32_t> const& ids,
                           bool pre_sorted = false)
{
  return from_ragged_ids(
    res, dataset_rows, ids, {0, static_cast<std::int64_t>(ids.size())}, pre_sorted);
}

roaring_allowlist from_device_ids(raft::resources const& res,
                                  std::size_t dataset_rows,
                                  std::vector<std::uint32_t> const& ids,
                                  bool pre_sorted = false)
{
  return from_device_ragged_ids(
    res, dataset_rows, ids, {0, static_cast<std::int64_t>(ids.size())}, pre_sorted);
}

roaring_allowlist import_row(raft::resources const& res,
                             std::size_t dataset_rows,
                             std::vector<std::byte> const& bytes)
{
  std::array<std::uint64_t, 2> const offsets{0, bytes.size()};
  return roaring_allowlist::from_serialized(
    res,
    dataset_rows,
    raft::make_host_vector_view<const std::byte, std::int64_t>(
      bytes.data(), static_cast<std::int64_t>(bytes.size())),
    raft::make_host_vector_view<const std::uint64_t, std::int64_t>(offsets.data(), offsets.size()));
}

std::vector<std::byte> copy_serialized_bytes(raft::resources const& res,
                                             roaring_allowlist const& allowlist)
{
  using ref_type = cuco::experimental::roaring_bitmap_ref<std::uint32_t>;
  static_assert(std::is_trivially_copyable_v<ref_type>);
  auto const view = allowlist.view(0);
  EXPECT_FALSE(view.empty());
  EXPECT_NE(view.device_reference(), nullptr);

  std::array<std::byte, sizeof(ref_type)> raw_reference{};
  auto const stream = raft::resource::get_cuda_stream(res);
  RAFT_CUDA_TRY(cudaMemcpyAsync(raw_reference.data(),
                                view.device_reference(),
                                raw_reference.size(),
                                cudaMemcpyDeviceToHost,
                                stream));
  raft::resource::sync_stream(res);
  auto const reference = std::bit_cast<ref_type>(raw_reference);
  std::vector<std::byte> bytes(reference.size_bytes());
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(bytes.data(), reference.data(), bytes.size(), cudaMemcpyDeviceToHost, stream));
  raft::resource::sync_stream(res);
  return bytes;
}

void expect_membership(raft::resources const& res,
                       roaring_allowlist const& allowlist,
                       std::vector<std::uint32_t> const& row_ids,
                       std::vector<std::uint8_t> const& expected,
                       bool asynchronous = false)
{
  ASSERT_EQ(row_ids.size(), expected.size());
  auto rows   = raft::make_device_vector<std::uint32_t, std::int64_t>(res, row_ids.size());
  auto output = raft::make_device_vector<std::uint8_t, std::int64_t>(res, expected.size());
  auto stream = raft::resource::get_cuda_stream(res);
  raft::update_device(rows.data_handle(), row_ids.data(), row_ids.size(), stream);
  if (asynchronous) {
    allowlist.contains_async(
      res,
      raft::make_device_matrix_view<const std::uint32_t, std::int64_t, raft::row_major>(
        rows.data_handle(), 1, static_cast<std::int64_t>(row_ids.size())),
      raft::make_device_matrix_view<std::uint8_t, std::int64_t, raft::row_major>(
        output.data_handle(), 1, static_cast<std::int64_t>(expected.size())));
  } else {
    allowlist.contains(
      res,
      raft::make_device_matrix_view<const std::uint32_t, std::int64_t, raft::row_major>(
        rows.data_handle(), 1, static_cast<std::int64_t>(row_ids.size())),
      raft::make_device_matrix_view<std::uint8_t, std::int64_t, raft::row_major>(
        output.data_handle(), 1, static_cast<std::int64_t>(expected.size())));
  }

  std::vector<std::uint8_t> actual(expected.size());
  raft::update_host(actual.data(), output.data_handle(), actual.size(), stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(actual, expected);
}

void expect_batch_membership(raft::resources const& res,
                             roaring_allowlist const& allowlists,
                             std::size_t columns,
                             std::vector<std::uint32_t> const& row_ids,
                             std::vector<std::uint8_t> const& expected)
{
  ASSERT_EQ(row_ids.size(), expected.size());
  ASSERT_EQ(row_ids.size(), allowlists.num_allowlists() * columns);
  auto device_rows  = raft::make_device_vector<std::uint32_t, std::int64_t>(res, row_ids.size());
  auto output       = raft::make_device_vector<std::uint8_t, std::int64_t>(res, expected.size());
  auto const stream = raft::resource::get_cuda_stream(res);
  raft::update_device(device_rows.data_handle(), row_ids.data(), row_ids.size(), stream);
  allowlists.contains(
    res,
    raft::make_device_matrix_view<const std::uint32_t, std::int64_t, raft::row_major>(
      device_rows.data_handle(),
      static_cast<std::int64_t>(allowlists.num_allowlists()),
      static_cast<std::int64_t>(columns)),
    raft::make_device_matrix_view<std::uint8_t, std::int64_t, raft::row_major>(
      output.data_handle(),
      static_cast<std::int64_t>(allowlists.num_allowlists()),
      static_cast<std::int64_t>(columns)));
  std::vector<std::uint8_t> actual(expected.size());
  raft::update_host(actual.data(), output.data_handle(), actual.size(), stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(actual, expected);
}

TEST(RoaringAllowlist, BuildsArrayRunBitmapAndMultiContainerRowsFromIds)
{
  raft::device_resources res;

  auto array = from_ids(res, 200000, {65537, 7, 3, 131074, 5});
  EXPECT_EQ(array.dataset_rows(), 200000);
  EXPECT_EQ(array.cardinality(0), 5);
  EXPECT_FALSE(array.empty(0));
  EXPECT_GT(array.size_bytes(), 0);
  expect_membership(res, array, {3, 4, 7, 65537, 131073, 131074, 200000}, {1, 0, 1, 1, 0, 1, 0});

  std::vector<std::uint32_t> consecutive;
  for (std::uint32_t id = 100; id < 300; ++id) {
    consecutive.push_back(id);
  }
  auto run = from_ids(res, 1000, consecutive);
  EXPECT_EQ(run.cardinality(0), 200);
  expect_membership(res, run, {99, 100, 199, 299, 300}, {0, 1, 1, 1, 0});

  std::vector<std::uint32_t> sparse;
  for (std::uint32_t id = 0; id < 10000; id += 2) {
    sparse.push_back(id);
  }
  auto bitmap = from_ids(res, 10000, sparse);
  EXPECT_EQ(bitmap.cardinality(0), 5000);
  expect_membership(res, bitmap, {0, 1, 8192, 9998, 9999}, {1, 0, 1, 1, 0}, true);
}

TEST(RoaringAllowlist, BuildsRaggedRowsInOneGeneralBatch)
{
  raft::device_resources res;
  constexpr std::size_t rows         = 4;
  constexpr std::size_t dataset_rows = (std::size_t{64} << 16) + 16;
  std::vector<std::uint32_t> ids;
  std::vector<std::int64_t> indptr{0};

  for (std::uint32_t i = 0; i < 64; ++i) {
    ids.push_back(i);
  }
  indptr.push_back(static_cast<std::int64_t>(ids.size()));
  indptr.push_back(static_cast<std::int64_t>(ids.size()));  // empty row
  for (std::uint32_t i = 0; i < 128; ++i) {
    ids.push_back(1000 + 2 * i);
  }
  indptr.push_back(static_cast<std::int64_t>(ids.size()));
  for (std::uint32_t i = 0; i < 64; ++i) {
    ids.push_back((i << 16) + 7);
  }
  indptr.push_back(static_cast<std::int64_t>(ids.size()));
  for (std::size_t row = 0; row < rows; ++row) {
    std::reverse(ids.begin() + indptr[row], ids.begin() + indptr[row + 1]);
  }

  auto allowlists        = from_ragged_ids(res, dataset_rows, ids, indptr);
  auto device_allowlists = from_device_ragged_ids(res, dataset_rows, ids, indptr);
  EXPECT_EQ(allowlists.num_allowlists(), rows);
  EXPECT_EQ(allowlists.dataset_rows(), dataset_rows);
  EXPECT_EQ(allowlists.total_cardinality(), ids.size());
  EXPECT_EQ(allowlists.cardinality(0), 64);
  EXPECT_TRUE(allowlists.empty(1));
  EXPECT_EQ(allowlists.cardinality(2), 128);
  EXPECT_EQ(allowlists.cardinality(3), 64);
  EXPECT_EQ(allowlists.view(1).device_reference(), nullptr);
  EXPECT_NE(allowlists.view(0).device_reference(), allowlists.view(2).device_reference());
  EXPECT_EQ(device_allowlists.total_cardinality(), ids.size());
  EXPECT_EQ(device_allowlists.size_bytes(), allowlists.size_bytes());

  expect_batch_membership(res,
                          allowlists,
                          4,
                          {0,
                           63,
                           64,
                           1000,
                           0,
                           1000,
                           dataset_rows - 1,
                           dataset_rows,
                           999,
                           1000,
                           1254,
                           1255,
                           7,
                           (std::uint32_t{63} << 16) + 7,
                           63,
                           dataset_rows},
                          {1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0});
  expect_batch_membership(res,
                          device_allowlists,
                          2,
                          {63, 64, 1, 1000, 1254, 1255, 7, (std::uint32_t{63} << 16) + 7},
                          {1, 0, 0, 0, 1, 0, 1, 1});

  std::array views{allowlists.view(0), allowlists.view(1), allowlists.view(2), allowlists.view(3)};
  cuvs::neighbors::filtering::roaring_filter filter(res, views);
  EXPECT_EQ(filter.num_queries(), rows);
  EXPECT_EQ(filter.cardinality(2), 128);
}

TEST(RoaringAllowlist, BuildsGeneralMixedContainerRowsWithOneSegmentedSort)
{
  raft::device_resources res;
  constexpr std::size_t rows         = 3;
  constexpr std::size_t width        = 5000;
  constexpr std::size_t dataset_rows = std::size_t{8} << 16;
  std::vector<std::int64_t> const indptr{0, width, 2 * width, 3 * width};
  std::vector<std::uint32_t> sorted;
  sorted.reserve(rows * width);
  for (std::uint32_t i = 0; i < width; ++i) {
    sorted.push_back(100 + i);
  }
  for (std::uint32_t i = 0; i < width; ++i) {
    sorted.push_back(2 * i);
  }
  for (std::uint32_t i = 0; i < width; ++i) {
    auto const key   = i % 8;
    auto const value = (i / 8) * 2 + 1;
    sorted.push_back((key << 16) + value);
  }
  for (std::size_t row = 0; row < rows; ++row) {
    std::sort(sorted.begin() + indptr[row], sorted.begin() + indptr[row + 1]);
  }
  auto unsorted = sorted;
  for (std::size_t row = 0; row < rows; ++row) {
    std::reverse(unsorted.begin() + indptr[row], unsorted.begin() + indptr[row + 1]);
  }

  auto general   = from_ragged_ids(res, dataset_rows, unsorted, indptr, false);
  auto presorted = from_ragged_ids(res, dataset_rows, sorted, indptr, true);
  EXPECT_EQ(general.num_allowlists(), rows);
  EXPECT_EQ(general.total_cardinality(), rows * width);
  EXPECT_EQ(general.size_bytes(), presorted.size_bytes());
  expect_batch_membership(res,
                          general,
                          5,
                          {99,
                           100,
                           5099,
                           5100,
                           9998,
                           0,
                           1,
                           8192,
                           9998,
                           9999,
                           1,
                           2,
                           (std::uint32_t{7} << 16) + 1249,
                           (std::uint32_t{7} << 16) + 1250,
                           dataset_rows},
                          {0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0});
}

TEST(RoaringAllowlist, ImportsRaggedPortableRowsIntoOnePackedOwner)
{
  raft::device_resources res;
  auto array = array_row({1, 3, 5});
  auto run   = run_row(100, 99);
  std::vector<std::byte> bytes;
  bytes.insert(bytes.end(), array.begin(), array.end());
  auto const empty_offset = bytes.size();
  bytes.insert(bytes.end(), run.begin(), run.end());
  std::array<std::uint64_t, 4> const offsets{0, array.size(), empty_offset, bytes.size()};
  auto allowlists = roaring_allowlist::from_serialized(
    res,
    1000,
    raft::make_host_vector_view<const std::byte, std::int64_t>(bytes.data(), bytes.size()),
    raft::make_host_vector_view<const std::uint64_t, std::int64_t>(offsets.data(), offsets.size()));
  EXPECT_EQ(allowlists.num_allowlists(), 3);
  EXPECT_EQ(allowlists.cardinality(0), 3);
  EXPECT_TRUE(allowlists.empty(1));
  EXPECT_EQ(allowlists.cardinality(2), 100);
  EXPECT_EQ(allowlists.view(1).device_reference(), nullptr);
  expect_batch_membership(res,
                          allowlists,
                          4,
                          {1, 2, 3, 5, 1, 3, 100, 999, 99, 100, 199, 200},
                          {1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0});
}

TEST(RoaringAllowlist, ConstructionPathsEmitByteIdenticalPortableRows)
{
  raft::device_resources res;
  constexpr std::uint32_t second_key = std::uint32_t{1} << 16;
  constexpr std::uint32_t third_key  = std::uint32_t{2} << 16;

  std::vector<std::uint32_t> ids;
  for (std::uint32_t id = 100; id < 300; ++id) {
    ids.push_back(id);  // run
  }
  ids.insert(ids.end(), {second_key + 1, second_key + 3, second_key + 5});  // array
  for (std::uint32_t id = 0; id < 10000; id += 2) {
    ids.push_back(third_key + id);  // bitmap
  }
  std::reverse(ids.begin(), ids.end());
  auto sorted = ids;
  std::sort(sorted.begin(), sorted.end());
  auto const dataset_rows = static_cast<std::size_t>(third_key) + 10000;
  auto const expected     = cuvs::bench::allowlist::make_portable_roaring_bytes(ids);

  auto host_unsorted   = from_ids(res, dataset_rows, ids);
  auto host_pre_sorted = from_ids(res, dataset_rows, sorted, true);
  auto device_unsorted = from_device_ids(res, dataset_rows, ids);
  auto device_sorted   = from_device_ids(res, dataset_rows, sorted, true);

  EXPECT_EQ(copy_serialized_bytes(res, host_unsorted), expected);
  EXPECT_EQ(copy_serialized_bytes(res, host_pre_sorted), expected);
  EXPECT_EQ(copy_serialized_bytes(res, device_unsorted), expected);
  EXPECT_EQ(copy_serialized_bytes(res, device_sorted), expected);
}

TEST(RoaringAllowlist, BenchmarkPortableSerializerImportsMixedContainerKinds)
{
  raft::device_resources res;
  constexpr std::uint32_t second_key = std::uint32_t{1} << 16;
  constexpr std::uint32_t third_key  = std::uint32_t{2} << 16;

  std::vector<std::uint32_t> ids;
  for (std::uint32_t id = 100; id < 300; ++id) {
    ids.push_back(id);  // run
  }
  ids.insert(ids.end(), {second_key + 1, second_key + 3, second_key + 5});  // array
  for (std::uint32_t id = 0; id < 10000; id += 2) {
    ids.push_back(third_key + id);  // bitmap
  }
  std::reverse(ids.begin(), ids.end());
  auto const bytes = cuvs::bench::allowlist::make_portable_roaring_bytes(ids);
  auto imported    = import_row(res, static_cast<std::size_t>(third_key) + 10000, bytes);

  EXPECT_EQ(imported.cardinality(0), ids.size());
  expect_membership(res,
                    imported,
                    {99,
                     100,
                     299,
                     300,
                     second_key + 1,
                     second_key + 2,
                     third_key,
                     third_key + 1,
                     third_key + 9998,
                     third_key + 9999},
                    {0, 1, 1, 0, 1, 0, 1, 0, 1, 0});
}

TEST(RoaringAllowlist, BuildsFromUnsortedUniqueDeviceIdsWithoutModifyingInput)
{
  raft::device_resources res;
  constexpr std::uint32_t second_key = std::uint32_t{1} << 16;
  constexpr std::uint32_t third_key  = std::uint32_t{2} << 16;

  std::vector<std::uint32_t> ids;
  for (std::uint32_t id = 100; id < 300; ++id) {
    ids.push_back(id);  // run container
  }
  ids.insert(ids.end(), {second_key + 1, second_key + 3, second_key + 5});  // array container
  for (std::uint32_t id = 0; id < 10000; id += 2) {
    ids.push_back(third_key + id);  // bitmap container
  }
  std::reverse(ids.begin(), ids.end());
  auto const original = ids;

  auto device_ids    = raft::make_device_vector<std::uint32_t, std::int64_t>(res, ids.size());
  auto device_indptr = raft::make_device_vector<std::int64_t, std::int64_t>(res, 2);
  std::array<std::int64_t, 2> const indptr{0, static_cast<std::int64_t>(ids.size())};
  auto const stream = raft::resource::get_cuda_stream(res);
  raft::update_device(device_ids.data_handle(), ids.data(), ids.size(), stream);
  raft::update_device(device_indptr.data_handle(), indptr.data(), indptr.size(), stream);
  auto allowlist =
    roaring_allowlist::from_ids(res,
                                third_key + 10000,
                                raft::make_device_vector_view<const std::uint32_t, std::int64_t>(
                                  device_ids.data_handle(), ids.size()),
                                raft::make_device_vector_view<const std::int64_t, std::int64_t>(
                                  device_indptr.data_handle(), 2));

  std::vector<std::uint32_t> unchanged(ids.size());
  raft::update_host(unchanged.data(), device_ids.data_handle(), unchanged.size(), stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(unchanged, original);
  EXPECT_EQ(allowlist.cardinality(0), 5203);
  expect_membership(res,
                    allowlist,
                    {99,
                     100,
                     299,
                     300,
                     second_key + 1,
                     second_key + 2,
                     third_key,
                     third_key + 1,
                     third_key + 9998,
                     third_key + 9999},
                    {0, 1, 1, 0, 1, 0, 1, 0, 1, 0});
}

TEST(RoaringAllowlist, PreSortedFastPathBuildsHostAndDeviceIds)
{
  raft::device_resources res;
  std::vector<std::uint32_t> const sorted_ids{1, 2, 3, 65537, 65539, 131072};
  auto host_allowlist   = from_ids(res, 131073, sorted_ids, true);
  auto device_allowlist = from_device_ids(res, 131073, sorted_ids, true);
  EXPECT_EQ(host_allowlist.cardinality(0), sorted_ids.size());
  EXPECT_EQ(device_allowlist.cardinality(0), sorted_ids.size());
  EXPECT_EQ(host_allowlist.size_bytes(), device_allowlist.size_bytes());
  expect_membership(res, host_allowlist, {0, 1, 3, 4, 65537, 65538, 131072}, {0, 1, 1, 0, 1, 0, 1});
  expect_membership(
    res, device_allowlist, {0, 1, 3, 4, 65537, 65538, 131072}, {0, 1, 1, 0, 1, 0, 1});

  auto empty = from_device_ids(res, 10, {}, true);
  EXPECT_TRUE(empty.empty(0));
  EXPECT_THROW(from_device_ids(res, 10, {0, 9, 10}, true), raft::logic_error);
}

TEST(RoaringAllowlist, DeviceFactoryHandlesEmptyAndRejectsOutOfRangeIds)
{
  raft::device_resources res;
  auto empty = from_device_ids(res, 10, {});
  EXPECT_TRUE(empty.empty(0));
  EXPECT_EQ(empty.size_bytes(), sizeof(void const*) + sizeof(std::uint8_t));
  expect_membership(res, empty, {0, 9, 10}, {0, 0, 0});

  EXPECT_THROW(from_device_ids(res, 10, {0, 9, 10}), raft::logic_error);
  EXPECT_THROW(from_device_ids(res, 10, {0, 65536, 131072}), raft::logic_error);
  std::vector<std::uint32_t> general_invalid(129);
  for (std::uint32_t i = 0; i < 128; ++i) {
    general_invalid[i] = i;
  }
  general_invalid.back() = std::numeric_limits<std::uint32_t>::max();
  EXPECT_THROW(from_device_ids(res, 1000, general_invalid), raft::logic_error);
}

TEST(RoaringAllowlist, SparseDeviceBuilderHandlesThresholdAndGeneralCrossover)
{
  raft::device_resources res;
  constexpr auto second_key = std::uint32_t{1} << 16;

  std::vector<std::uint32_t> sorted;
  sorted.reserve(129);
  for (std::uint32_t id = 100; id < 164; ++id) {
    sorted.push_back(id);  // one run container
  }
  for (std::uint32_t value = 1; value < 65; value += 2) {
    sorted.push_back(second_key + value);  // one array container
  }
  for (std::uint32_t key = 2; sorted.size() < 129; ++key) {
    sorted.push_back((key << 16) + 7);  // many one-value array containers
  }
  ASSERT_EQ(sorted.size(), 129);

  std::vector<std::uint32_t> threshold_ids(sorted.begin(), sorted.begin() + 128);
  auto reversed_threshold = threshold_ids;
  std::reverse(reversed_threshold.begin(), reversed_threshold.end());
  auto reversed_general = sorted;
  std::reverse(reversed_general.begin(), reversed_general.end());
  auto const dataset_rows = static_cast<std::size_t>(sorted.back()) + 1;

  auto sparse_unsorted  = from_device_ids(res, dataset_rows, reversed_threshold);
  auto general_unsorted = from_device_ids(res, dataset_rows, reversed_general);
  std::vector<std::uint32_t> pre_sorted_ids(sorted.begin(), sorted.begin() + 64);
  auto sparse_pre_sorted  = from_device_ids(res, dataset_rows, pre_sorted_ids, true);
  auto general_pre_sorted = from_device_ids(res, dataset_rows, sorted, true);

  EXPECT_EQ(sparse_unsorted.cardinality(0), 128);
  EXPECT_EQ(general_unsorted.cardinality(0), 129);
  EXPECT_EQ(sparse_pre_sorted.cardinality(0), 64);
  EXPECT_EQ(general_pre_sorted.cardinality(0), 129);
  EXPECT_EQ(general_unsorted.size_bytes(), general_pre_sorted.size_bytes());

  auto const sparse_last  = threshold_ids.back();
  auto const general_last = sorted.back();
  expect_membership(res,
                    sparse_unsorted,
                    {99, 100, 163, 164, second_key + 1, second_key + 2, sparse_last, general_last},
                    {0, 1, 1, 0, 1, 0, 1, 0});
  expect_membership(res,
                    general_unsorted,
                    {99, 100, 163, 164, second_key + 1, second_key + 2, sparse_last, general_last},
                    {0, 1, 1, 0, 1, 0, 1, 1});
  expect_membership(res, sparse_pre_sorted, {99, 100, 163, 164}, {0, 1, 1, 0});
  expect_membership(
    res, general_pre_sorted, {99, 100, 163, 164, second_key + 1, general_last}, {0, 1, 1, 0, 1, 1});
}

TEST(RoaringAllowlist, SparseDeviceSortPreservesMaximumUint32Id)
{
  raft::device_resources res;
  auto const max_id = std::numeric_limits<std::uint32_t>::max();
  auto allowlist =
    from_device_ids(res, static_cast<std::size_t>(std::uint64_t{1} << 32), {max_id, 0});
  EXPECT_EQ(allowlist.cardinality(0), 2);
  expect_membership(res, allowlist, {0, 1, max_id - 1, max_id}, {1, 0, 0, 1});
}

TEST(RoaringAllowlist, ViewIsZeroCopyAndSurvivesOwnerMove)
{
  raft::device_resources res;
  auto allowlist  = from_ids(res, 32, {1, 4, 7});
  auto first_view = allowlist.view(0);
  auto next_view  = allowlist.view(0);
  EXPECT_TRUE(first_view.valid());
  EXPECT_EQ(first_view.device_reference(), next_view.device_reference());
  EXPECT_NE(first_view.device_reference(), nullptr);

  auto moved      = std::move(allowlist);
  auto moved_view = moved.view(0);
  EXPECT_EQ(first_view.device_reference(), moved_view.device_reference());
  EXPECT_EQ(moved_view.cardinality(), 3);
  expect_membership(res, moved, {1, 2, 7}, {1, 0, 1});

  auto empty = from_ids(res, 32, {});
  EXPECT_TRUE(empty.view(0).valid());
  EXPECT_EQ(empty.view(0).device_reference(), nullptr);
  EXPECT_TRUE(empty.empty(0));
  EXPECT_EQ(empty.size_bytes(), sizeof(void const*) + sizeof(std::uint8_t));
  expect_membership(res, empty, {0, 31, 32}, {0, 0, 0});
}

TEST(RoaringAllowlist, StreamOrderedConstructionSupportsExplicitEventHandoff)
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
  RAFT_CUDA_TRY(cudaEventRecord(ready, raft::resource::get_cuda_stream(build_res)));
  RAFT_CUDA_TRY(cudaStreamWaitEvent(raft::resource::get_cuda_stream(consume_res), ready));
  expect_membership(consume_res, allowlist, {99, 100, 200, 900, 901}, {0, 1, 1, 1, 0}, true);

  // Serialized import has the same lifetime rule. Keep the caller-owned host bytes alive until the
  // event recorded after construction has completed, then consume the initialized reference on the
  // second stream.
  auto bytes    = array_row({2, 4, 8});
  auto imported = import_row(build_res, 1000, bytes);
  RAFT_CUDA_TRY(cudaEventRecord(ready, raft::resource::get_cuda_stream(build_res)));
  RAFT_CUDA_TRY(cudaStreamWaitEvent(raft::resource::get_cuda_stream(consume_res), ready));
  expect_membership(consume_res, imported, {1, 2, 4, 7, 8}, {0, 1, 1, 0, 1}, true);

  // Releasing an owner immediately is safe: device_uvector schedules the packed allocation's
  // deallocation after encoding and reference initialization on the construction stream.
  {
    auto temporary = from_ids(build_res, 1000, {1, 10, 100});
    EXPECT_NE(temporary.view(0).device_reference(), nullptr);
  }
  RAFT_CUDA_TRY(cudaEventRecord(ready, raft::resource::get_cuda_stream(build_res)));
  RAFT_CUDA_TRY(cudaEventSynchronize(ready));
  RAFT_CUDA_TRY(cudaEventDestroy(ready));
}

TEST(RoaringAllowlist, ImportsPortableArrayRunAndEmptyRows)
{
  raft::device_resources res;

  auto array_bytes = array_row({1, 3, 5});
  auto array       = import_row(res, 1000, array_bytes);
  EXPECT_EQ(array.cardinality(0), 3);
  expect_membership(res, array, {1, 3, 4, 5}, {1, 1, 0, 1});

  auto run_bytes = run_row(100, 99);
  auto run       = import_row(res, 1000, run_bytes);
  EXPECT_EQ(run.cardinality(0), 100);
  expect_membership(res, run, {99, 100, 199, 200}, {0, 1, 1, 0}, true);

  std::vector<std::byte> standard_empty;
  append_u32(standard_empty, kCookieNoRun);
  append_u32(standard_empty, 0);
  auto empty = import_row(res, 1000, standard_empty);
  EXPECT_TRUE(empty.empty(0));
  EXPECT_EQ(empty.view(0).device_reference(), nullptr);

  auto zero_length = import_row(res, 1000, {});
  EXPECT_TRUE(zero_length.empty(0));
}

TEST(RoaringAllowlist, RejectsSmallMalformedPortableInputs)
{
  raft::device_resources res;

  auto truncated = array_row({1, 3, 5});
  truncated.pop_back();
  EXPECT_THROW(import_row(res, 1000, truncated), raft::logic_error);

  auto bad_cookie = array_row({1});
  bad_cookie[0]   = std::byte{0};
  EXPECT_THROW(import_row(res, 1000, bad_cookie), raft::logic_error);

  auto out_of_range = array_row({1000});
  EXPECT_THROW(import_row(res, 1000, out_of_range), raft::logic_error);

  auto valid = array_row({1, 3, 5});
  auto bytes =
    raft::make_host_vector_view<const std::byte, std::int64_t>(valid.data(), valid.size());
  std::array<std::uint64_t, 1> const too_short{0};
  EXPECT_THROW(roaring_allowlist::from_serialized(
                 res,
                 1000,
                 bytes,
                 raft::make_host_vector_view<const std::uint64_t, std::int64_t>(too_short.data(),
                                                                                too_short.size())),
               raft::logic_error);
  std::array<std::uint64_t, 2> const nonzero_start{1, valid.size()};
  EXPECT_THROW(roaring_allowlist::from_serialized(
                 res,
                 1000,
                 bytes,
                 raft::make_host_vector_view<const std::uint64_t, std::int64_t>(
                   nonzero_start.data(), nonzero_start.size())),
               raft::logic_error);
  std::array<std::uint64_t, 4> const decreasing{0, valid.size(), valid.size() - 1, valid.size()};
  EXPECT_THROW(roaring_allowlist::from_serialized(
                 res,
                 1000,
                 bytes,
                 raft::make_host_vector_view<const std::uint64_t, std::int64_t>(decreasing.data(),
                                                                                decreasing.size())),
               raft::logic_error);
}

TEST(RoaringAllowlist, RejectsIdsOutsideLogicalShape)
{
  raft::device_resources res;
  EXPECT_THROW(from_ids(res, 10, {10}), raft::logic_error);
  EXPECT_THROW(from_ids(res, 0, {}), raft::logic_error);
}

TEST(RoaringAllowlist, RejectsMalformedIndptr)
{
  raft::device_resources res;
  std::vector<std::uint32_t> const ids{1, 2};
  EXPECT_THROW(from_ragged_ids(res, 10, ids, {1, 2}), raft::logic_error);
  EXPECT_THROW(from_ragged_ids(res, 10, ids, {0, 2, 1, 2}), raft::logic_error);
  EXPECT_THROW(from_ragged_ids(res, 10, ids, {0, 1}), raft::logic_error);
}

TEST(RoaringFilter, ReusesViewsAndUpdatesOneQueryOutsideSearch)
{
  raft::device_resources res;
  auto first  = from_ids(res, 16, {1, 3});
  auto second = from_ids(res, 16, {2, 4, 6});
  auto empty  = from_ids(res, 16, {});

  std::array views{first.view(0), second.view(0), first.view(0)};
  cuvs::neighbors::filtering::roaring_filter filter(res, views);
  EXPECT_TRUE(filter.valid());
  EXPECT_EQ(filter.num_queries(), 3);
  EXPECT_EQ(filter.dataset_rows(), 16);
  EXPECT_EQ(filter.cardinality(0), 2);
  EXPECT_EQ(filter.cardinality(1), 3);
  EXPECT_EQ(filter.cardinality(2), 2);
  EXPECT_FLOAT_EQ(filter.filtering_rate(), 0.875f);
  EXPECT_GT(filter.size_bytes(), 0);

  auto const* payload = filter.device_payload();
  auto shared_copy    = filter;
  shared_copy.set_allowlist(res, 1, empty.view(0));
  EXPECT_EQ(filter.device_payload(), payload);
  EXPECT_TRUE(filter.empty(1));
  EXPECT_FLOAT_EQ(filter.filtering_rate(), 0.999f);

  shared_copy.set_allowlist(res, 1, second.view(0));
  EXPECT_EQ(filter.cardinality(1), 3);
  EXPECT_EQ(filter.device_payload(), payload);
}

TEST(RoaringFilter, RejectsSmallInvalidMappings)
{
  raft::device_resources res;
  EXPECT_THROW(cuvs::neighbors::filtering::roaring_filter(
                 res, std::span<const cuvs::core::roaring_allowlist_view>{}),
               raft::logic_error);

  auto ten    = from_ids(res, 10, {1});
  auto eleven = from_ids(res, 11, {1});
  std::array mismatched{ten.view(0), eleven.view(0)};
  EXPECT_THROW(cuvs::neighbors::filtering::roaring_filter(res, mismatched), raft::logic_error);

  std::array invalid{cuvs::core::roaring_allowlist_view{}};
  EXPECT_THROW(cuvs::neighbors::filtering::roaring_filter(res, invalid), raft::logic_error);

  std::array one{ten.view(0)};
  cuvs::neighbors::filtering::roaring_filter filter(res, one);
  EXPECT_THROW(filter.set_allowlist(res, 1, ten.view(0)), raft::logic_error);
  EXPECT_THROW(filter.set_allowlist(res, 0, eleven.view(0)), raft::logic_error);
}

TEST(CagraFilterBenchmarkUtils, GeneratesExactDeterministicAllowlistLayouts)
{
  using cuvs::bench::cagra_filter::allowlist_layout;
  using cuvs::bench::cagra_filter::make_allowed_ids;

  for (auto layout : {allowlist_layout::random, allowlist_layout::runs}) {
    auto first  = make_allowed_ids(257, 73, layout, 12345);
    auto second = make_allowed_ids(257, 73, layout, 12345);
    EXPECT_EQ(first, second);
    ASSERT_EQ(first.size(), 73);

    std::sort(first.begin(), first.end());
    EXPECT_EQ(std::adjacent_find(first.begin(), first.end()), first.end());
    EXPECT_LT(first.back(), 257);
  }

  auto run = make_allowed_ids(257, 73, allowlist_layout::runs, 12345);
  std::sort(run.begin(), run.end());
  auto breaks = std::size_t{0};
  for (std::size_t i = 1; i < run.size(); ++i) {
    breaks += run[i] != run[i - 1] + 1 ? 1 : 0;
  }
  EXPECT_LE(breaks,
            1);  // a circular interval is one range, or two when it wraps at row zero

  auto random = make_allowed_ids(257, 73, allowlist_layout::random, 12345);
  std::sort(random.begin(), random.end());
  breaks = 0;
  for (std::size_t i = 1; i < random.size(); ++i) {
    breaks += random[i] != random[i - 1] + 1 ? 1 : 0;
  }
  EXPECT_GT(breaks, 1);

  EXPECT_TRUE(make_allowed_ids(257, 0, allowlist_layout::random, 12345).empty());
  EXPECT_THROW(make_allowed_ids(0, 0, allowlist_layout::random, 12345), std::invalid_argument);
  EXPECT_THROW(make_allowed_ids(10, 11, allowlist_layout::runs, 12345), std::invalid_argument);
}

TEST(CagraFilterBenchmarkUtils, SummarizesLatencySamples)
{
  auto const stats = cuvs::bench::cagra_filter::summarize({1.0, 2.0, 3.0, 4.0, 5.0});
  EXPECT_DOUBLE_EQ(stats.mean, 3.0);
  EXPECT_DOUBLE_EQ(stats.p50, 3.0);
  EXPECT_DOUBLE_EQ(stats.p95, 4.8);
  EXPECT_DOUBLE_EQ(stats.p99, 4.96);

  EXPECT_EQ(cuvs::bench::cagra_filter::cardinality_for_fraction(1'000'000, 0.001, 10), 1000);
  EXPECT_EQ(cuvs::bench::cagra_filter::allowed_percent_label(0.001), "allowed_0.1pct");
  EXPECT_EQ(cuvs::bench::cagra_filter::allowed_percent_label(0.015), "allowed_1.5pct");
}

TEST(AllowlistBenchmarkUtils, GeneratesControlledSmallLookupProfiles)
{
  using namespace cuvs::bench::allowlist;

  constexpr std::uint32_t rows      = 64;
  constexpr std::size_t cardinality = 16;
  for (auto profile : {lookup_profile::best, lookup_profile::average, lookup_profile::worst}) {
    auto first  = make_allowed_ids(rows, cardinality, profile, 12345);
    auto second = make_allowed_ids(rows, cardinality, profile, 12345);
    EXPECT_EQ(first, second);
    ASSERT_EQ(first.size(), cardinality);
    std::sort(first.begin(), first.end());
    EXPECT_EQ(std::adjacent_find(first.begin(), first.end()), first.end());
    EXPECT_LT(first.back(), rows);
  }

  auto best = make_allowed_ids(rows, cardinality, lookup_profile::best, 12345);
  EXPECT_EQ(occupied_chunks(best), 1);
  EXPECT_EQ(count_runs(best), 1);

  auto worst = make_allowed_ids(rows, cardinality, lookup_profile::worst, 12345);
  EXPECT_EQ(maximum_encoded_runs(cardinality, rows), 7);
  EXPECT_EQ(count_runs(worst), 7);

  EXPECT_THROW(make_allowed_ids(0, 0, lookup_profile::best, 12345), std::invalid_argument);
  EXPECT_THROW(make_allowed_ids(10, 11, lookup_profile::worst, 12345), std::invalid_argument);
}

TEST(AllowlistBenchmarkUtils, GeneratesBestHitsAverageDomainAndWorstMisses)
{
  using namespace cuvs::bench::allowlist;
  constexpr std::uint32_t rows      = 64;
  constexpr std::size_t cardinality = 16;
  for (auto profile : {lookup_profile::best, lookup_profile::average, lookup_profile::worst}) {
    auto ids    = make_allowed_ids(rows, cardinality, profile, 12345);
    auto probes = make_probes(rows, ids, profile, 257, 67890);
    ASSERT_EQ(probes.size(), 257);
    EXPECT_TRUE(std::all_of(probes.begin(), probes.end(), [](auto probe) { return probe < rows; }));
    std::sort(ids.begin(), ids.end());
    if (profile == lookup_profile::best) {
      EXPECT_TRUE(std::all_of(probes.begin(), probes.end(), [&](auto probe) {
        return std::binary_search(ids.begin(), ids.end(), probe);
      }));
    } else if (profile == lookup_profile::worst) {
      EXPECT_TRUE(std::none_of(probes.begin(), probes.end(), [&](auto probe) {
        return std::binary_search(ids.begin(), ids.end(), probe);
      }));
    }
  }
  EXPECT_THROW(make_probes(rows, {1}, lookup_profile::average, 0, 67890), std::invalid_argument);

  auto full_domain = make_allowed_ids(std::uint64_t{1} << 32, 1024, lookup_profile::average, 12345);
  auto repeated    = make_allowed_ids(std::uint64_t{1} << 32, 1024, lookup_profile::average, 12345);
  EXPECT_EQ(full_domain, repeated);
  std::sort(full_domain.begin(), full_domain.end());
  EXPECT_EQ(std::adjacent_find(full_domain.begin(), full_domain.end()), full_domain.end());
}

}  // namespace
}  // namespace cuvs::core
