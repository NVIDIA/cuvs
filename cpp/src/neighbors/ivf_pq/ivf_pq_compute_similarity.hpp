/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <memory>

#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

template <typename OutT, typename LutT>
struct selected {
  dim3 grid_dim;
  dim3 block_dim;
  size_t smem_size;
  size_t device_lut_size;
  std::shared_ptr<AlgorithmLauncher> launcher;

  selected() = default;

  selected(std::shared_ptr<AlgorithmLauncher> launcher,
           dim3 grid,
           dim3 block,
           size_t smem,
           size_t device_lut)
    : grid_dim(grid),
      block_dim(block),
      smem_size(smem),
      device_lut_size(device_lut),
      launcher(std::move(launcher))
  {
  }
};

}  // namespace cuvs::neighbors::ivf_pq::detail
