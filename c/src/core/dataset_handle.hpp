/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * Helpers shared by the C API sources that create or consume `cuvsDataset` handles: the dataset
 * definitions in `dataset.cpp` and the CAGRA bindings in `neighbors/cagra.cpp`.
 */

#include <cuvs/core/dataset.h>
#include <cuvs/core/dataset/vpq.hpp>

#include <raft/core/error.hpp>

#include <cuda_fp16.h>

#include <cstdint>
#include <utility>

namespace cuvs::core::c_api {

template <typename T>
inline void destroy_typed_addr(void* ptr)
{
  delete reinterpret_cast<T*>(ptr);
}

template <typename OwnerT, typename ViewT, typename Fn>
inline void with_dataset_view(cuvsDataset_t dataset, Fn&& fn)
{
  RAFT_EXPECTS(dataset != nullptr, "null dataset handle");
  RAFT_EXPECTS(dataset->addr != 0, "null dataset storage");
  if (dataset->is_owning) {
    auto* owner = reinterpret_cast<OwnerT*>(dataset->addr);
    auto view   = std::as_const(*owner).view();
    fn(view);
  } else {
    fn(*reinterpret_cast<ViewT*>(dataset->addr));
  }
}

using device_vpq_owner_t = cuvs::device_vpq_dataset<half, int64_t>;
using device_vpq_view_t  = cuvs::device_vpq_dataset_view<half, int64_t>;

}  // namespace cuvs::core::c_api
