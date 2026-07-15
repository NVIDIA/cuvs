/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentEntry.hpp>
#include <cuvs/detail/jit_lto/ivf_rabitq/ivf_rabitq_fragments.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

struct ComputeInnerProductsWithLutPlanner : AlgorithmPlanner {
  inline static LauncherJitCache launcher_jit_cache{};

  ComputeInnerProductsWithLutPlanner()
    : AlgorithmPlanner("compute_inner_products_with_lut", launcher_jit_cache)
  {
  }

  void add_entrypoint()
  {
    this->add_static_fragment<fragment_tag_compute_inner_products_with_lut>();
  }

  template <bool WithEx>
  void add_lut_emit_distances_device_function()
  {
    this->add_static_fragment<fragment_tag_lut_emit_distances<WithEx>>();
  }

  template <int EX_BITS>
  void add_extract_code_device_function()
  {
    this->add_static_fragment<fragment_tag_extract_code<EX_BITS>>();
  }

  void add_compute_lut_ip_for_vec_device_function()
  {
    this->add_static_fragment<fragment_tag_compute_lut_ip_for_vec<tag_lut_dtype_f32>>();
  }
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
