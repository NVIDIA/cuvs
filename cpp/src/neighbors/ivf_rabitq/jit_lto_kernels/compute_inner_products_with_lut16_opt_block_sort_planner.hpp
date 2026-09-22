/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>
#include <rtcx/algorithm_planner.hpp>
#include <rtcx/fragment_entry.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

struct ComputeInnerProductsWithLut16OptBlockSortPlanner : rtcx::algorithm_planner {
  inline static rtcx::launcher_jit_cache launcher_jit_cache{};

  ComputeInnerProductsWithLut16OptBlockSortPlanner()
    : rtcx::algorithm_planner("compute_inner_products_with_lut16_opt_block_sort",
                              launcher_jit_cache)
  {
  }

  template <bool WithEx>
  void add_entrypoint()
  {
    this->add_static_fragment<
      fragment_tag_compute_inner_products_with_lut16_opt_block_sort<WithEx>>();
  }

  template <int EX_BITS>
  void add_extract_code_device_function()
  {
    this->add_static_fragment<fragment_tag_extract_code<EX_BITS>>();
  }

  void add_compute_lut_ip_for_vec_device_function()
  {
    this->add_static_fragment<fragment_tag_compute_lut_ip_for_vec<tag_lut_dtype_f16>>();
  }
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
