/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../common/ann_types.hpp"
#include "../common/conf.hpp"
#include "diskann_wrapper.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace cuvs::bench {

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::diskann_memory<T>::build_param& param)
{
  param.R       = conf.at("R");
  param.L_build = conf.at("L_build");
  if (conf.contains("alpha")) { param.num_threads = conf.at("alpha"); }
  if (conf.contains("num_threads")) { param.num_threads = conf.at("num_threads"); }
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::diskann_ssd<T>::build_param& param)
{
  param.R       = conf.at("R");
  param.L_build = conf.at("L_build");
  if (conf.contains("alpha")) { param.alpha = conf.at("alpha"); }
  if (conf.contains("num_threads")) { param.num_threads = conf.at("num_threads"); }
  if (conf.contains("QD")) { param.QD = conf.at("QD"); }
  if (conf.contains("build_dram_budget_megabytes")) {
    param.build_dram_budget_megabytes = conf.at("build_dram_budget_megabytes");
  }
  if (conf.contains("search_dram_budget_megabytes")) {
    param.search_dram_budget_megabytes = conf.at("search_dram_budget_megabytes");
  }
  param.dataset_base_file = cuvs::bench::configuration::singleton().get_dataset_conf().base_file;
  for (const auto& index : cuvs::bench::configuration::singleton().get_indices()) {
    // The nlohmann::json operator== comparator compares each key and value by content. Reference:
    // https://json.nlohmann.me/api/basic_json/operator_eq/
    if (index.build_param == conf) {
      param.index_file = index.file;
      break;
    }
  }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::diskann_memory<T>::search_param& param)
{
  param.L_search = conf.at("L_search");
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::diskann_ssd<T>::search_param& param)
{
  param.L_search = conf.at("L_search");
  if (conf.contains("num_nodes_to_cache")) {
    param.num_nodes_to_cache = conf.at("num_nodes_to_cache");
  }
  if (conf.contains("beam_width")) { param.beam_width = conf.at("beam_width"); }
}

template <typename T, template <typename> class Algo>
std::unique_ptr<cuvs::bench::algo<T>> make_algo(cuvs::bench::Metric metric,
                                                int dim,
                                                const nlohmann::json& conf)
{
  typename Algo<T>::build_param param;
  parse_build_param<T>(conf, param);
  return std::make_unique<Algo<T>>(metric, dim, param);
}

template <typename T>
auto create_algo(const std::string& algo_name,
                 const std::string& distance,
                 int dim,
                 const nlohmann::json& conf) -> std::unique_ptr<cuvs::bench::algo<T>>
{
  cuvs::bench::Metric metric = parse_metric(distance);
  std::unique_ptr<cuvs::bench::algo<T>> a;

  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint8_t> ||
                std::is_same_v<T, int8_t>) {
    if (algo_name == "diskann_memory") {
      a = make_algo<T, cuvs::bench::diskann_memory>(metric, dim, conf);
    } else if (algo_name == "diskann_ssd") {
      a = make_algo<T, cuvs::bench::diskann_ssd>(metric, dim, conf);
    }
  }
  if (!a) { throw std::runtime_error("invalid algo: '" + algo_name + "'"); }

  return a;
}

template <typename T>
std::unique_ptr<typename cuvs::bench::algo<T>::search_param> create_search_param(
  const std::string& algo_name, const nlohmann::json& conf)
{
  if (algo_name == "diskann_memory") {
    auto param = std::make_unique<typename cuvs::bench::diskann_memory<T>::search_param>();
    parse_search_param<T>(conf, *param);
    return param;
  } else if (algo_name == "diskann_ssd") {
    auto param = std::make_unique<typename cuvs::bench::diskann_ssd<T>::search_param>();
    parse_search_param<T>(conf, *param);
    return param;
  }
  throw std::runtime_error("invalid algo: '" + algo_name + "'");
}

};  // namespace cuvs::bench

REGISTER_ALGO_INSTANCE(float);
REGISTER_ALGO_INSTANCE(std::int8_t);
REGISTER_ALGO_INSTANCE(std::uint8_t);

#ifdef ANN_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
int main(int argc, char** argv) { return cuvs::bench::run_main(argc, argv); }
#endif
