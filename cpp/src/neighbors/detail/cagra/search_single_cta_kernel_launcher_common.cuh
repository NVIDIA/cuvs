/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/logger.hpp>

namespace cuvs::neighbors::cagra::detail::single_cta_search {

// Common logic for computing max_candidates and max_itopk
struct LaunchConfig {
  uint32_t max_candidates;
  uint32_t max_itopk;
  bool topk_by_bitonic_sort;
  bool bitonic_sort_and_merge_multi_warps;
};

inline LaunchConfig compute_launch_config(uint32_t num_itopk_candidates,
                                          uint32_t itopk_size,
                                          uint32_t block_size)
{
  LaunchConfig config{};

  // Compute max_candidates
  if (num_itopk_candidates <= 64) {
    config.max_candidates = 64;
  } else if (num_itopk_candidates <= 128) {
    config.max_candidates = 128;
  } else if (num_itopk_candidates <= 256) {
    config.max_candidates = 256;
  } else {
    config.max_candidates = 32;  // irrelevant, radix based topk is used
  }

  // Compute max_itopk and sort flags
  config.topk_by_bitonic_sort               = (num_itopk_candidates <= 256);
  config.bitonic_sort_and_merge_multi_warps = false;

  if (config.topk_by_bitonic_sort) {
    if (itopk_size <= 64) {
      config.max_itopk = 64;
    } else if (itopk_size <= 128) {
      config.max_itopk = 128;
    } else if (itopk_size <= 256) {
      config.max_itopk = 256;
    } else {
      config.max_itopk                          = 512;
      config.bitonic_sort_and_merge_multi_warps = (block_size >= 64);
    }
  } else {
    if (itopk_size <= 256) {
      config.max_itopk = 256;
    } else {
      config.max_itopk = 512;
    }
  }

  return config;
}

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
