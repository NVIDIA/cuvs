/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_param_parser.h"
#include "cuvs_cagra_diskann_wrapper.h"

#include <cuda/memory_resource>
#include <rmm/cuda_device.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

namespace cuvs::bench {

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::cuvs_cagra_diskann<T, IdxT>::search_param& param)
{
  if (conf.contains("L_search")) { param.L_search = conf.at("L_search"); }
}

template <typename T>
auto create_algo(const std::string& algo_name,
                 const std::string& distance,
                 int dim,
                 const nlohmann::json& conf) -> std::unique_ptr<cuvs::bench::algo<T>>
{
  [[maybe_unused]] cuvs::bench::Metric metric = parse_metric(distance);
  std::unique_ptr<cuvs::bench::algo<T>> a;

  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::uint8_t> ||
                std::is_same_v<T, std::int8_t>) {
    if (algo_name == "cuvs_cagra_diskann") {
      typename cuvs::bench::cuvs_cagra_diskann<T, uint32_t>::build_param param;
      ::parse_build_param<T, uint32_t>(conf, param);
      a = std::make_unique<cuvs::bench::cuvs_cagra_diskann<T, uint32_t>>(metric, dim, param);
    }
  }

  if (!a) { throw std::runtime_error("invalid algo: '" + algo_name + "'"); }

  return a;
}

template <typename T>
auto create_search_param(const std::string& algo_name, const nlohmann::json& conf)
  -> std::unique_ptr<typename cuvs::bench::algo<T>::search_param>
{
  if (algo_name == "cuvs_cagra_diskann") {
    auto param =
      std::make_unique<typename cuvs::bench::cuvs_cagra_diskann<T, uint32_t>::search_param>();
    parse_search_param<T, uint32_t>(conf, *param);
    return param;
  }

  throw std::runtime_error("invalid algo: '" + algo_name + "'");
}

}  // namespace cuvs::bench

REGISTER_ALGO_INSTANCE(float);
REGISTER_ALGO_INSTANCE(std::int8_t);
REGISTER_ALGO_INSTANCE(std::uint8_t);

#ifdef ANN_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
/*
[NOTE] Dear developer,

Please don't modify the content of the `main` function; this will make the behavior of the benchmark
executable differ depending on the cmake flags and will complicate the debugging. In particular,
don't try to setup an RMM memory resource here; it will anyway be modified by the memory resource
set on per-algorithm basis. For example, see `cuvs/cuvs_ann_bench_utils.h`.
*/
int main(int argc, char** argv) { return cuvs::bench::run_main(argc, argv); }
#endif
