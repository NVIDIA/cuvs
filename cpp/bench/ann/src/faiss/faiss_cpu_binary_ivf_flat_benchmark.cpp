/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../common/ann_types.hpp"
#include "faiss_cpu_binary_ivf_flat_wrapper.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cuvs::bench {

void parse_build_param(const nlohmann::json& conf, faiss_cpu_binary_ivf_flat::build_param& param)
{
  param.nlist = conf.at("nlist");
  if (conf.contains("niter")) { param.niter = conf.at("niter"); }
  if (conf.contains("seed")) { param.seed = conf.at("seed"); }
  if (conf.contains("max_points_per_centroid")) {
    param.max_points_per_centroid = conf.at("max_points_per_centroid");
  }
  if (conf.contains("build_threads")) { param.build_threads = conf.at("build_threads"); }
  if (conf.contains("coarse_query_batch_size")) {
    param.coarse_query_batch_size = conf.at("coarse_query_batch_size");
  }
}

void parse_search_param(const nlohmann::json& conf, faiss_cpu_binary_ivf_flat::search_param& param)
{
  param.nprobe    = conf.at("nprobe");
  param.k         = conf.at("k");
  param.n_queries = conf.at("n_queries");
}

template <typename T>
auto create_algo(const std::string& algo_name,
                 const std::string& distance,
                 int dim,
                 const nlohmann::json& conf) -> std::unique_ptr<cuvs::bench::algo<T>>
{
  if constexpr (std::is_same_v<T, uint8_t>) {
    if (algo_name == "faiss_cpu_binary_ivf_flat") {
      faiss_cpu_binary_ivf_flat::build_param param;
      parse_build_param(conf, param);
      return std::make_unique<faiss_cpu_binary_ivf_flat>(parse_metric(distance), dim, param);
    }
  }

  throw std::runtime_error("invalid algo: '" + algo_name + "'");
}

template <typename T>
auto create_search_param(const std::string& algo_name, const nlohmann::json& conf)
  -> std::unique_ptr<typename cuvs::bench::algo<T>::search_param>
{
  if constexpr (std::is_same_v<T, uint8_t>) {
    if (algo_name == "faiss_cpu_binary_ivf_flat") {
      auto param = std::make_unique<faiss_cpu_binary_ivf_flat::search_param>();
      parse_search_param(conf, *param);
      return param;
    }
  }

  throw std::runtime_error("invalid algo: '" + algo_name + "'");
}

}  // namespace cuvs::bench

REGISTER_ALGO_INSTANCE(float);
REGISTER_ALGO_INSTANCE(std::int8_t);
REGISTER_ALGO_INSTANCE(std::uint8_t);

#ifdef ANN_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
int main(int argc, char** argv) { return cuvs::bench::run_main(argc, argv); }
#endif
