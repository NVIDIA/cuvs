/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuvs/neighbors/common.hpp>

#include "../compute_distance.hpp"  // dataset_descriptor_base_t
#include "cagra_filter_payload.cuh"
#include "search_single_cta_device_helpers.cuh"

namespace cuvs::neighbors::cagra::detail {

// Function types for extern "C" __global__ JIT entry points — must match cudaLibraryGetKernel /
// AlgorithmLauncher::dispatch signatures exactly (see static_assert in each *_kernel.cu).

template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
using search_single_cta_kernel_func_t =
  void(uintptr_t,
       DistanceT* const,
       const std::uint32_t,
       const DataT* const,
       const IndexT* const,
       const std::uint32_t,
       const SourceIndexT*,
       const unsigned,
       const uint64_t,
       const IndexT*,
       const uint32_t,
       IndexT* const,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       std::uint32_t* const,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*,
       const IndexT,
       cagra_sample_filter<SourceIndexT>);

namespace single_cta_search {

template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
using search_single_cta_p_kernel_func_t =
  void(worker_handle_t*,
       job_desc_t<job_desc_traits<DataT, IndexT, DistanceT>>*,
       uint32_t*,
       const IndexT* const,
       const std::uint32_t,
       const SourceIndexT*,
       const unsigned,
       const uint64_t,
       const IndexT*,
       const uint32_t,
       IndexT* const,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       std::uint32_t* const,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*,
       cagra_sample_filter<SourceIndexT>);

}  // namespace single_cta_search

namespace multi_cta_search {

template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
using search_multi_cta_kernel_func_t =
  void(IndexT* const,
       DistanceT* const,
       const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*,
       const DataT* const,
       const IndexT* const,
       const std::uint32_t,
       const std::uint32_t,
       const SourceIndexT*,
       const unsigned,
       const uint64_t,
       const IndexT*,
       const std::uint32_t,
       const std::uint32_t,
       IndexT* const,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       const std::uint32_t,
       std::uint32_t* const,
       const IndexT,
       const std::uint32_t,
       cagra_sample_filter<SourceIndexT>);

}  // namespace multi_cta_search

namespace multi_kernel_search {

template <typename DataT, typename IndexT, typename DistanceT>
using random_pickup_kernel_func_t = void(const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*,
                                         const DataT* const,
                                         const std::size_t,
                                         const unsigned,
                                         const uint64_t,
                                         const IndexT*,
                                         const std::uint32_t,
                                         IndexT* const,
                                         DistanceT* const,
                                         const std::uint32_t,
                                         IndexT* const,
                                         const std::uint32_t,
                                         const IndexT);

template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
using compute_distance_to_child_nodes_kernel_func_t =
  void(const IndexT* const,
       IndexT* const,
       DistanceT* const,
       const std::size_t,
       const std::uint32_t,
       const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*,
       const IndexT* const,
       const std::uint32_t,
       const SourceIndexT*,
       const DataT*,
       IndexT* const,
       const std::uint32_t,
       IndexT* const,
       DistanceT* const,
       const std::uint32_t,
       cagra_sample_filter<SourceIndexT>);

template <typename IndexT, typename DistanceT, typename SourceIndexT>
using apply_filter_kernel_func_t = void(const SourceIndexT* const,
                                        IndexT* const,
                                        DistanceT* const,
                                        const std::size_t,
                                        const std::uint32_t,
                                        const std::uint32_t,
                                        const std::uint32_t,
                                        cagra_sample_filter<SourceIndexT>);

}  // namespace multi_kernel_search

}  // namespace cuvs::neighbors::cagra::detail
