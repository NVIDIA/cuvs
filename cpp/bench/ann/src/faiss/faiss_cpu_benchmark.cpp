/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../common/ann_types.hpp"
#include "faiss_cpu_wrapper.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace cuvs::bench {

template <typename T>
void parse_base_build_param(const nlohmann::json& conf,
                            typename cuvs::bench::faiss_cpu<T>::build_param& param)
{
  param.nlist = conf.at("nlist");
  if (conf.contains("ratio")) { param.ratio = conf.at("ratio"); }
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::faiss_cpu_ivf_flat<T>::build_param& param)
{
  parse_base_build_param<T>(conf, param);
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::faiss_cpu_ivfpq<T>::build_param& param)
{
  parse_base_build_param<T>(conf, param);
  param.m = conf.at("M");
  if (conf.contains("usePrecomputed")) {
    param.use_precomputed = conf.at("usePrecomputed");
  } else {
    param.use_precomputed = false;
  }
  if (conf.contains("bitsPerCode")) {
    param.bits_per_code = conf.at("bitsPerCode");
  } else {
    param.bits_per_code = 8;
  }
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::faiss_cpu_ivfrabitq<T>::build_param& param)
{
  parse_base_build_param<T>(conf, param);
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::faiss_cpu_ivfsq<T>::build_param& param)
{
  parse_base_build_param<T>(conf, param);
  param.quantizer_type = conf.at("quantizer_type");
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::faiss_cpu_hnsw_flat<T>::build_param& param)
{
  param.M = conf.at("M");
  if (conf.contains("efConstruction")) {
    param.efConstruction = conf.at("efConstruction");
  } else {
    param.efConstruction = 40;
  }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::faiss_cpu<T>::search_param& param)
{
  param.nprobe = conf.at("nprobe");
  if (conf.contains("refine_ratio")) { param.refine_ratio = conf.at("refine_ratio"); }
  if (conf.contains("parallel_mode")) { param.parallel_mode = conf.at("parallel_mode"); }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::faiss_cpu_hnsw_flat<T>::search_param& param)
{
  faiss::SearchParametersHNSW p;
  if (conf.contains("efSearch")) { p.efSearch = conf.at("efSearch"); }
  if (conf.contains("check_relative_distance")) {
    p.check_relative_distance = conf.at("check_relative_distance");
  }
  if (conf.contains("bounded_queue")) { p.bounded_queue = conf.at("bounded_queue"); }
  param.p = p;
}

template <typename T, template <typename> class Algo>
auto make_algo(cuvs::bench::Metric metric, int dim, const nlohmann::json& conf)
  -> std::unique_ptr<cuvs::bench::algo<T>>
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
  std::unique_ptr<cuvs::bench::algo<T>> a;

  if constexpr (std::is_same_v<T, float>) {
    cuvs::bench::Metric metric = parse_metric(distance);
    if (algo_name == "faiss_cpu_ivf_flat") {
      a = make_algo<T, cuvs::bench::faiss_cpu_ivf_flat>(metric, dim, conf);
    } else if (algo_name == "faiss_cpu_ivf_pq") {
      a = make_algo<T, cuvs::bench::faiss_cpu_ivfpq>(metric, dim, conf);
    } else if (algo_name == "faiss_cpu_ivf_rabitq") {
      a = make_algo<T, cuvs::bench::faiss_cpu_ivfrabitq>(metric, dim, conf);
    } else if (algo_name == "faiss_cpu_ivf_sq") {
      a = make_algo<T, cuvs::bench::faiss_cpu_ivfsq>(metric, dim, conf);
    } else if (algo_name == "faiss_cpu_flat") {
      a = std::make_unique<cuvs::bench::faiss_cpu_flat<T>>(metric, dim);
    } else if (algo_name == "faiss_cpu_hnsw_flat") {
      a = make_algo<T, cuvs::bench::faiss_cpu_hnsw_flat>(metric, dim, conf);
    }
  }

  if constexpr (std::is_same_v<T, uint8_t>) {}

  if (!a) { throw std::runtime_error("invalid algo: '" + algo_name + "'"); }

  return a;
}

template <typename T>
auto create_search_param(const std::string& algo_name, const nlohmann::json& conf)
  -> std::unique_ptr<typename cuvs::bench::algo<T>::search_param>
{
  if (algo_name == "faiss_cpu_ivf_flat" || algo_name == "faiss_cpu_ivf_pq" ||
      algo_name == "faiss_cpu_ivf_rabitq" || algo_name == "faiss_cpu_ivf_sq") {
    auto param = std::make_unique<typename cuvs::bench::faiss_cpu<T>::search_param>();
    parse_search_param<T>(conf, *param);
    return param;
  } else if (algo_name == "faiss_cpu_flat") {
    auto param = std::make_unique<typename cuvs::bench::faiss_cpu<T>::search_param>();
    return param;
  } else if (algo_name == "faiss_cpu_hnsw_flat") {
    auto param = std::make_unique<typename cuvs::bench::faiss_cpu_hnsw_flat<T>::search_param>();
    parse_search_param<T>(conf, *param);
    return param;
  }
  // else
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
