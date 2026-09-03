/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/roaring_allowlist.hpp>

#include "nvtx.hpp"

#include <cuco/roaring_bitmap.cuh>

#include <raft/core/copy.cuh>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuda/std/cstddef>
#include <cuda/stream_ref>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace cuvs::core {
namespace {

/**
 * cuCollections Roaring ownership and encoding
 * =============================================
 *
 * cuCollections constructs a standard portable 32-bit Roaring stream directly on the GPU. A
 * 32-bit ID is divided into a high-16-bit container key and a low-16-bit value. For each key, at
 * most 4,096 values are encoded as a sorted uint16 array; larger containers use an 8 KiB bitmap.
 * ID construction does not emit run containers.
 *
 * The owning cuco object retains the encoded bytes. Its lightweight
 * cuco::experimental::roaring_bitmap_ref<uint32_t> parses the portable header once and then stores
 * container-location metadata plus pointers into those bytes. cuVS copies that reference into a
 * stable device allocation during construction. Views and CAGRA filters copy only its pointer, so
 * search never copies or reparses the payload.
 *
 * @see https://github.com/RoaringBitmap/RoaringFormatSpec
 * @see https://github.com/NVIDIA/cuCollections/pull/839
 */

using ref_type              = cuco::experimental::roaring_bitmap_ref<std::uint32_t>;
using cuco_bitmap_allocator = rmm::mr::polymorphic_allocator<cuda::std::byte>;
using cuco_bitmap_type = cuco::experimental::roaring_bitmap<std::uint32_t, cuco_bitmap_allocator>;

void validate_dataset_rows(std::size_t dataset_rows)
{
  constexpr std::uint64_t kKeyDomain = std::uint64_t{1} << 32;
  RAFT_EXPECTS(dataset_rows > 0, "dataset_rows must be greater than zero.");
  RAFT_EXPECTS(static_cast<std::uint64_t>(dataset_rows) <= kKeyDomain,
               "dataset_rows exceeds the uint32_t Roaring key domain.");
}

struct device_build_result {
  std::unique_ptr<cuco_bitmap_type> owner;
  std::size_t cardinality{};
};

device_build_result build_from_device_ids(
  raft::resources const& res,
  raft::device_vector_view<const std::uint32_t, std::int64_t> ids,
  bool pre_sorted)
{
  common::nvtx::range<common::nvtx::domain::cuvs> build_scope("roaring_allowlist::build_from_ids");
  auto const stream = raft::resource::get_cuda_stream(res);
  auto const size   = static_cast<std::size_t>(ids.extent(0));
  if (size == 0) { return {}; }

  cuco_bitmap_allocator allocator{};
  cuda::stream_ref cuco_stream{stream.value()};
  auto bitmap            = pre_sorted ? cuco_bitmap_type::from_sorted_unique_indices(
                               ids.data_handle(), ids.data_handle() + size, allocator, cuco_stream)
                                      : cuco_bitmap_type::from_indices(
                               ids.data_handle(), ids.data_handle() + size, allocator, cuco_stream);
  auto owner             = std::make_unique<cuco_bitmap_type>(std::move(bitmap));
  auto const cardinality = static_cast<std::size_t>(owner->size());
  return {std::move(owner), cardinality};
}

}  // namespace

struct roaring_allowlist::impl {
  std::unique_ptr<cuco_bitmap_type> owner;
  std::optional<ref_type> host_reference;
  rmm::device_uvector<ref_type> device_reference;
  std::size_t cardinality_{};
  std::size_t dataset_rows_{};

  impl(raft::resources const& res, std::size_t dataset_rows, device_build_result&& built)
    : owner(std::move(built.owner)),
      host_reference(owner ? std::make_optional(owner->ref()) : std::nullopt),
      device_reference(owner ? 1 : 0, raft::resource::get_cuda_stream(res)),
      cardinality_(built.cardinality),
      dataset_rows_(dataset_rows)
  {
    static_assert(std::is_trivially_copyable_v<ref_type>);
    if (host_reference) {
      raft::copy(
        device_reference.data(), &host_reference.value(), 1, raft::resource::get_cuda_stream(res));
    }
  }

  [[nodiscard]] ref_type const* reference() const noexcept
  {
    return device_reference.size() == 0 ? nullptr : device_reference.data();
  }
};

roaring_allowlist::roaring_allowlist(std::unique_ptr<impl> impl) noexcept : impl_(std::move(impl))
{
}

roaring_allowlist roaring_allowlist::from_ids(
  raft::resources const& res,
  std::size_t dataset_rows,
  raft::host_vector_view<const key_type, std::int64_t> ids,
  bool pre_sorted)
{
  validate_dataset_rows(dataset_rows);
  auto const size   = static_cast<std::size_t>(ids.extent(0));
  auto const stream = raft::resource::get_cuda_stream(res);
  rmm::device_uvector<key_type> device_ids(size, stream);
  if (size != 0) { raft::copy(device_ids.data(), ids.data_handle(), size, stream); }
  auto device_ids_view = raft::make_device_vector_view<const key_type, std::int64_t>(
    device_ids.data(), static_cast<std::int64_t>(size));
  auto built = build_from_device_ids(res, device_ids_view, pre_sorted);
  return roaring_allowlist{std::make_unique<impl>(res, dataset_rows, std::move(built))};
}

roaring_allowlist roaring_allowlist::from_ids(
  raft::resources const& res,
  std::size_t dataset_rows,
  raft::device_vector_view<const key_type, std::int64_t> ids,
  bool pre_sorted)
{
  validate_dataset_rows(dataset_rows);
  auto built = build_from_device_ids(res, ids, pre_sorted);
  return roaring_allowlist{std::make_unique<impl>(res, dataset_rows, std::move(built))};
}

roaring_allowlist::~roaring_allowlist()                                       = default;
roaring_allowlist::roaring_allowlist(roaring_allowlist&&) noexcept            = default;
roaring_allowlist& roaring_allowlist::operator=(roaring_allowlist&&) noexcept = default;

std::size_t roaring_allowlist::dataset_rows() const noexcept { return impl_->dataset_rows_; }

std::size_t roaring_allowlist::cardinality() const noexcept { return impl_->cardinality_; }

bool roaring_allowlist::empty() const noexcept { return cardinality() == 0; }

std::size_t roaring_allowlist::size_bytes() const noexcept
{
  auto bytes = impl_->device_reference.size() * sizeof(ref_type);
  if (impl_->owner) { bytes += static_cast<std::size_t>(impl_->owner->size_bytes()); }
  return bytes;
}

roaring_allowlist_view roaring_allowlist::view() const noexcept
{
  return roaring_allowlist_view{impl_->reference(), dataset_rows(), cardinality()};
}

}  // namespace cuvs::core
