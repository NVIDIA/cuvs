/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <AlgorithmPlanner.hpp>
#include <FragmentEntry.hpp>
#include <cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

struct ComputeInnerProductsWithBitwiseBlockSortPlanner : AlgorithmPlanner {
  inline static LauncherJitCache launcher_jit_cache{};

  ComputeInnerProductsWithBitwiseBlockSortPlanner()
    : AlgorithmPlanner("compute_inner_products_with_bitwise_block_sort", launcher_jit_cache)
  {
  }

  void add_entrypoint()
  {
    this->add_static_fragment<fragment_tag_compute_inner_products_with_bitwise_block_sort>();
  }

  template <bool WithEx>
  void add_bitwise_block_sort_emit_topk_device_function()
  {
    this->add_static_fragment<fragment_tag_bitwise_block_sort_emit_topk<WithEx>>();
  }

  template <int EX_BITS>
  void add_extract_code_device_function()
  {
    this->add_static_fragment<fragment_tag_extract_code<EX_BITS>>();
  }

  template <int NumBits>
  void add_compute_bitwise_quantized_ip_for_vec_device_function()
  {
    this->add_static_fragment<fragment_tag_compute_bitwise_quantized_ip_for_vec<NumBits>>();
  }
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
