/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuco/roaring_bitmap_ref.cuh>
#include <cuda/stream_ref>
#include <memory>
#include <optional>

namespace cuvs::detail {
struct roaring_bitmap_owner;
struct roaring_bitmap_deleter {
  void operator()(roaring_bitmap_owner*) const noexcept;
};
using roaring_owner_ptr = std::unique_ptr<roaring_bitmap_owner, roaring_bitmap_deleter>;
struct roaring_build_result {
  roaring_owner_ptr owner;
  std::optional<cuco::experimental::roaring_bitmap_ref<uint32_t>> reference;
  size_t cardinality{};
  size_t storage_bytes{};
};
roaring_build_result build_roaring_bitmap(const uint32_t* ids,
                                          size_t count,
                                          bool pre_sorted,
                                          cuda::stream_ref stream);
}  // namespace cuvs::detail
