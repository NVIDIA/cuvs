/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_inner_products_with_bitwise_block_sort_planner.hpp"
#include "compute_inner_products_with_bitwise_planner.hpp"
#include "compute_inner_products_with_lut16_opt_block_sort_planner.hpp"
#include "compute_inner_products_with_lut16_opt_planner.hpp"
#include "compute_inner_products_with_lut_block_sort_planner.hpp"
#include "compute_inner_products_with_lut_planner.hpp"

#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>

#include <cassert>
#include <memory>

namespace cuvs::neighbors::ivf_rabitq::detail {

namespace {

template <typename Planner>
inline void add_ex_bits_device_functions(Planner& planner, int ex_bits)
{
  switch (ex_bits) {
    case 1: planner.template add_extract_code_device_function<1>(); break;
    case 2: planner.template add_extract_code_device_function<2>(); break;
    case 3: planner.template add_extract_code_device_function<3>(); break;
    case 4: planner.template add_extract_code_device_function<4>(); break;
    case 5: planner.template add_extract_code_device_function<5>(); break;
    case 6: planner.template add_extract_code_device_function<6>(); break;
    case 7: planner.template add_extract_code_device_function<7>(); break;
    case 8: planner.template add_extract_code_device_function<8>(); break;
    default: assert(false);
  }
}

}  // namespace

inline std::shared_ptr<AlgorithmLauncher> make_compute_inner_products_with_lut_launcher(
  int ex_bits, bool with_ex)
{
  ComputeInnerProductsWithLutPlanner planner;
  planner.add_entrypoint();
  planner.add_compute_lut_ip_for_vec_device_function();
  if (with_ex) {
    planner.add_lut_emit_distances_device_function<true>();
    add_ex_bits_device_functions(planner, ex_bits);
  } else {
    planner.add_lut_emit_distances_device_function<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher> make_compute_inner_products_with_lut_block_sort_launcher(
  int ex_bits, bool with_ex)
{
  ComputeInnerProductsWithLutBlockSortPlanner planner;
  planner.add_entrypoint();
  planner.add_compute_lut_ip_for_vec_device_function();
  if (with_ex) {
    planner.add_lut_block_sort_emit_topk_device_function<true>();
    add_ex_bits_device_functions(planner, ex_bits);
  } else {
    planner.add_lut_block_sort_emit_topk_device_function<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher> make_compute_inner_products_with_lut16_opt_launcher(
  int ex_bits, bool with_ex)
{
  ComputeInnerProductsWithLut16OptPlanner planner;
  planner.add_entrypoint();
  planner.add_compute_lut_ip_for_vec_device_function();
  if (with_ex) {
    planner.add_lut16_opt_emit_distances_device_function<true>();
    add_ex_bits_device_functions(planner, ex_bits);
  } else {
    planner.add_lut16_opt_emit_distances_device_function<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher>
make_compute_inner_products_with_lut16_opt_block_sort_launcher(int ex_bits, bool with_ex)
{
  ComputeInnerProductsWithLut16OptBlockSortPlanner planner;
  planner.add_compute_lut_ip_for_vec_device_function();
  if (with_ex) {
    planner.add_entrypoint<true>();
    add_ex_bits_device_functions(planner, ex_bits);
  } else {
    planner.add_entrypoint<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher> make_compute_inner_products_with_bitwise_launcher(
  int ex_bits, bool with_ex)
{
  ComputeInnerProductsWithBitwisePlanner planner;
  planner.add_entrypoint();
  if (with_ex) {
    planner.add_bitwise_emit_distances_device_function<true>();
    add_ex_bits_device_functions(planner, ex_bits);
  } else {
    planner.add_bitwise_emit_distances_device_function<false>();
  }
  return planner.get_launcher();
}

inline std::shared_ptr<AlgorithmLauncher>
make_compute_inner_products_with_bitwise_block_sort_launcher(int num_bits,
                                                             int ex_bits,
                                                             bool with_ex)
{
  ComputeInnerProductsWithBitwiseBlockSortPlanner planner;
  planner.add_entrypoint();
  if (num_bits == 4) {
    planner.add_compute_bitwise_quantized_ip_for_vec_device_function<4>();
  } else {
    planner.add_compute_bitwise_quantized_ip_for_vec_device_function<8>();
  }
  if (with_ex) {
    planner.add_bitwise_block_sort_emit_topk_device_function<true>();
    add_ex_bits_device_functions(planner, ex_bits);
  } else {
    planner.add_bitwise_block_sort_emit_topk_device_function<false>();
  }
  return planner.get_launcher();
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
