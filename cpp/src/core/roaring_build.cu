/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_device_roaring_select_launchers.hpp"

#include "cub_device_select_if_launchers.hpp"
#include "cub_device_unique_launchers.hpp"

#include "cub_radix_launchers.hpp"

#include "cub_ptx_query.hpp"

#include "cub_compact_launchers.hpp"

#include "roaring_build.hpp"
#include <cuco/roaring_bitmap.cuh>
#include <cuda/std/cstddef>
#include <cuvs/core/cuda_fp16.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

namespace cuvs::detail {

using roaring_allocator = rmm::mr::polymorphic_allocator<cuda::std::byte>;

using roaring_bitmap = cuco::experimental::roaring_bitmap<uint32_t, roaring_allocator>;

struct roaring_bitmap_owner {
  explicit roaring_bitmap_owner(roaring_bitmap&& value) : bitmap(std::move(value)) {}
  roaring_bitmap bitmap;
};

void roaring_bitmap_deleter::operator()(roaring_bitmap_owner* owner) const noexcept
{
  delete owner;
}

roaring_build_result build_roaring_bitmap(const uint32_t* ids,
                                          size_t count,
                                          bool pre_sorted,
                                          cuda::stream_ref stream)
{
  roaring_allocator allocator{};
  auto bitmap = pre_sorted
                  ? roaring_bitmap::from_sorted_unique_indices(ids, ids + count, allocator, stream)
                  : roaring_bitmap::from_indices(ids, ids + count, allocator, stream);
  roaring_owner_ptr owner(new roaring_bitmap_owner(std::move(bitmap)));
  auto reference   = owner->bitmap.ref();
  auto cardinality = static_cast<size_t>(owner->bitmap.size());
  auto bytes       = static_cast<size_t>(owner->bitmap.size_bytes());
  return {std::move(owner), reference, cardinality, bytes};
}

}  // namespace cuvs::detail
