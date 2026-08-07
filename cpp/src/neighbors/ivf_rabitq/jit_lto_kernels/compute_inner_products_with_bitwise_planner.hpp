/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <AlgorithmPlanner.hpp>
#include <FragmentEntry.hpp>
#include <cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

struct ComputeInnerProductsWithBitwisePlanner : AlgorithmPlanner {
  inline static LauncherJitCache launcher_jit_cache{};

  ComputeInnerProductsWithBitwisePlanner()
    : AlgorithmPlanner("compute_inner_products_with_bitwise", launcher_jit_cache)
  {
  }

  void add_entrypoint()
  {
    this->add_static_fragment<fragment_tag_compute_inner_products_with_bitwise>();
  }

  template <bool WithEx>
  void add_bitwise_emit_distances_device_function()
  {
    this->add_static_fragment<fragment_tag_bitwise_emit_distances<WithEx>>();
  }

  template <int EX_BITS>
  void add_extract_code_device_function()
  {
    this->add_static_fragment<fragment_tag_extract_code<EX_BITS>>();
  }
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
