/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

namespace cuvs::neighbors::cagra::detail {

/// Device payload for linked CAGRA sample filters plus query offset for wrapped filters.
template <typename SourceIndexT>
struct cagra_sample_filter {
  void* filter_data{nullptr};
  std::uint32_t query_id_offset{0};

  __device__ __forceinline__ void* sample_filter_data() { return filter_data; }
};

}  // namespace cuvs::neighbors::cagra::detail
