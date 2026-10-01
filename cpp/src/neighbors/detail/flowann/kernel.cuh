/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "queue.cuh"

#include <neighbors/detail/cagra/compute_distance.hpp>
#include <neighbors/detail/cagra/jit_lto_kernels/cagra_filter_payload.cuh>

#include <cstdint>

namespace cuvs::neighbors::cagra::experimental::flowann::detail {

inline constexpr std::uint32_t max_search_width        = 8;
inline constexpr std::uint32_t additional_buffer_width = 2;

template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
using search_single_cta_kernel_func_t =
  void(uintptr_t,
       DistanceT*,
       std::uint32_t,
       const DataT*,
       const std::uint8_t*,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t,
       const std::uint32_t*,
       const std::uint32_t*,
       queue_view*,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t,
       const SourceIndexT*,
       unsigned,
       std::uint64_t,
       const IndexT*,
       std::uint32_t,
       const std::uint32_t*,
       const IndexT*,
       const std::uint32_t*,
       std::uint32_t,
       IndexT*,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t*,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t,
       const cuvs::neighbors::cagra::detail::dataset_descriptor_base_t<DataT, IndexT, DistanceT>*,
       IndexT,
       float,
       std::uint32_t,
       cuvs::neighbors::cagra::detail::cagra_sample_filter<SourceIndexT>);

template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
using search_multi_cta_kernel_func_t =
  void(IndexT*,
       DistanceT*,
       const cuvs::neighbors::cagra::detail::dataset_descriptor_base_t<DataT, IndexT, DistanceT>*,
       const DataT*,
       const std::uint8_t*,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t,
       const std::uint32_t*,
       const std::uint32_t*,
       queue_view*,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t,
       const SourceIndexT*,
       unsigned,
       std::uint64_t,
       const IndexT*,
       std::uint32_t,
       const std::uint32_t*,
       const IndexT*,
       const std::uint32_t*,
       std::uint32_t,
       std::uint32_t,
       IndexT*,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t,
       std::uint32_t*,
       IndexT,
       std::uint32_t,
       float,
       std::uint32_t,
       cuvs::neighbors::cagra::detail::cagra_sample_filter<SourceIndexT>);

}  // namespace cuvs::neighbors::cagra::experimental::flowann::detail
