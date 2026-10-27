/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../detail/ann_utils.cuh"  // cuvs::spatial::knn::detail::utils::mapping

#include "../../core/nvtx.hpp"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map_reduce.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/logical.h>

#include <algorithm>
#include <cstdint>

namespace cuvs::neighbors::ivf_pq::helpers {

/**
 * @brief Detect whether FP16 internal distance dtypes overflow for this dataset during search.
 *
 * Runs a small probe search against an already-built IVF-PQ index with current distance types,
 * and reports whether any returned distance is non-finite (inf/NaN).
 */
template <typename DataT, typename Accessor>
bool detect_fp16_overflow(
  raft::resources const& handle,
  const cuvs::neighbors::ivf_pq::index<int64_t>& index,
  cuvs::neighbors::ivf_pq::search_params search_params,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  uint32_t k);

}  // namespace cuvs::neighbors::ivf_pq::helpers
