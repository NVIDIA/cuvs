/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_param_parser.h"

#include <rmm/mr/per_device_resource.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace cuvs::bench {

#ifdef CUVS_ANN_BENCH_USE_CUVS_MG
void add_distribution_mode(cuvs::neighbors::distribution_mode* dist_mode,
                           const nlohmann::json& conf)
{
  if (conf.contains("distribution_mode")) {
    std::string distribution_mode = conf.at("distribution_mode");
    if (distribution_mode == "replicated") {
      *dist_mode = cuvs::neighbors::distribution_mode::REPLICATED;
    } else if (distribution_mode == "sharded") {
      *dist_mode = cuvs::neighbors::distribution_mode::SHARDED;
    } else {
      throw std::runtime_error("invalid value for distribution_mode");
    }
  } else {
    // default
    *dist_mode = cuvs::neighbors::distribution_mode::SHARDED;
  }
};

void add_merge_mode(cuvs::neighbors::sharded_merge_mode* merge_mode, const nlohmann::json& conf)
{
  if (conf.contains("merge_mode")) {
    std::string sharded_merge_mode = conf.at("merge_mode");
    if (sharded_merge_mode == "tree_merge") {
      *merge_mode = cuvs::neighbors::sharded_merge_mode::TREE_MERGE;
    } else if (sharded_merge_mode == "merge_on_root_rank") {
      *merge_mode = cuvs::neighbors::sharded_merge_mode::MERGE_ON_ROOT_RANK;
    } else {
      throw std::runtime_error("invalid value for merge_mode");
    }
  } else {
    // default
    *merge_mode = cuvs::neighbors::sharded_merge_mode::TREE_MERGE;
  }
};
#endif

template <typename T>
auto create_algo(const std::string& algo_name,
                 const std::string& distance,
                 int dim,
                 const nlohmann::json& conf) -> std::unique_ptr<cuvs::bench::algo<T>>
{
  [[maybe_unused]] cuvs::bench::Metric metric = parse_metric(distance);
  std::unique_ptr<cuvs::bench::algo<T>> a;

  if constexpr (std::is_same_v<T, float>) {
#ifdef CUVS_ANN_BENCH_USE_CUVS_BRUTE_FORCE
    if (algo_name == "raft_brute_force" || algo_name == "cuvs_brute_force") {
      a = std::make_unique<cuvs::bench::cuvs_gpu<T>>(metric, dim);
    }
#endif
  }

  if constexpr (std::is_same_v<T, uint8_t>) {}

#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint8_t> ||
                std::is_same_v<T, int8_t>) {
    if (algo_name == "raft_ivf_flat" || algo_name == "cuvs_ivf_flat") {
      typename cuvs::bench::cuvs_ivf_flat<T, int64_t>::build_param param;
      parse_build_param<T, int64_t>(conf, param);
      a = std::make_unique<cuvs::bench::cuvs_ivf_flat<T, int64_t>>(metric, dim, param);
    }
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_SQ
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
    if (algo_name == "cuvs_ivf_sq") {
      typename cuvs::bench::cuvs_ivf_sq<T>::build_param param;
      parse_build_param<T>(conf, param);
      a = std::make_unique<cuvs::bench::cuvs_ivf_sq<T>>(metric, dim, param);
    }
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_PQ
  if (algo_name == "raft_ivf_pq" || algo_name == "cuvs_ivf_pq") {
    typename cuvs::bench::cuvs_ivf_pq<T, int64_t>::build_param param;
    parse_build_param<T, int64_t>(conf, param);
    a = std::make_unique<cuvs::bench::cuvs_ivf_pq<T, int64_t>>(metric, dim, param);
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_RABITQ
  if constexpr (std::is_same_v<T, float>) {
    if (algo_name == "cuvs_ivf_rabitq") {
      typename cuvs::bench::cuvs_ivf_rabitq<T, int64_t>::build_param param;
      parse_build_param<T, int64_t>(conf, param);
      a = std::make_unique<cuvs::bench::cuvs_ivf_rabitq<T, int64_t>>(metric, dim, param);
    }
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_CAGRA
  if (algo_name == "raft_cagra" || algo_name == "cuvs_cagra") {
    typename cuvs::bench::cuvs_cagra<T, uint32_t>::build_param param;
    parse_build_param<T, uint32_t>(conf, param);
    a = std::make_unique<cuvs::bench::cuvs_cagra<T, uint32_t>>(metric, dim, param);
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_MG
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint8_t> ||
                std::is_same_v<T, int8_t>) {
    if (algo_name == "raft_mg_ivf_flat" || algo_name == "cuvs_mg_ivf_flat") {
      typename cuvs::bench::cuvs_mg_ivf_flat<T, int64_t>::build_param param;
      parse_build_param<T, int64_t>(conf, param);
      add_distribution_mode(&param.mode, conf);
      a = std::make_unique<cuvs::bench::cuvs_mg_ivf_flat<T, int64_t>>(metric, dim, param);
    }
  }

  if (algo_name == "raft_mg_ivf_pq" || algo_name == "cuvs_mg_ivf_pq") {
    typename cuvs::bench::cuvs_mg_ivf_pq<T, int64_t>::build_param param;
    parse_build_param<T, int64_t>(conf, param);
    add_distribution_mode(&param.mode, conf);
    a = std::make_unique<cuvs::bench::cuvs_mg_ivf_pq<T, int64_t>>(metric, dim, param);
  }

  if (algo_name == "raft_mg_cagra" || algo_name == "cuvs_mg_cagra") {
    typename cuvs::bench::cuvs_mg_cagra<T, uint32_t>::build_param param;
    parse_build_param<T, uint32_t>(conf, param);
    add_distribution_mode(&param.mode, conf);
    a = std::make_unique<cuvs::bench::cuvs_mg_cagra<T, uint32_t>>(metric, dim, param);
  }

#endif

  if (!a) { throw std::runtime_error("invalid algo: '" + algo_name + "'"); }

  return a;
}

template <typename T>
auto create_search_param(const std::string& algo_name, const nlohmann::json& conf)
  -> std::unique_ptr<typename cuvs::bench::algo<T>::search_param>
{
#ifdef CUVS_ANN_BENCH_USE_CUVS_BRUTE_FORCE
  if (algo_name == "raft_brute_force" || algo_name == "cuvs_brute_force") {
    auto param = std::make_unique<typename cuvs::bench::cuvs_gpu<T>::search_param>();
    return param;
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint8_t> ||
                std::is_same_v<T, int8_t>) {
    if (algo_name == "raft_ivf_flat" || algo_name == "cuvs_ivf_flat") {
      auto param =
        std::make_unique<typename cuvs::bench::cuvs_ivf_flat<T, int64_t>::search_param>();
      parse_search_param<T, int64_t>(conf, *param);
      return param;
    }
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_SQ
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
    if (algo_name == "cuvs_ivf_sq") {
      auto param = std::make_unique<typename cuvs::bench::cuvs_ivf_sq<T>::search_param>();
      parse_search_param<T>(conf, *param);
      return param;
    }
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_PQ
  if (algo_name == "raft_ivf_pq" || algo_name == "cuvs_ivf_pq") {
    auto param = std::make_unique<typename cuvs::bench::cuvs_ivf_pq<T, int64_t>::search_param>();
    parse_search_param<T, int64_t>(conf, *param);
    return param;
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_RABITQ
  if constexpr (std::is_same_v<T, float>) {
    if (algo_name == "cuvs_ivf_rabitq") {
      auto param =
        std::make_unique<typename cuvs::bench::cuvs_ivf_rabitq<T, int64_t>::search_param>();
      parse_search_param<T, int64_t>(conf, *param);
      return param;
    }
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_CAGRA
  if (algo_name == "raft_cagra" || algo_name == "cuvs_cagra") {
    auto param = std::make_unique<typename cuvs::bench::cuvs_cagra<T, uint32_t>::search_param>();
    parse_search_param<T, uint32_t>(conf, *param);
    return param;
  }
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_MG
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, uint8_t> ||
                std::is_same_v<T, int8_t>) {
    if (algo_name == "raft_mg_ivf_flat" || algo_name == "cuvs_mg_ivf_flat") {
      auto param =
        std::make_unique<typename cuvs::bench::cuvs_mg_ivf_flat<T, int64_t>::search_param>();
      parse_search_param<T, int64_t>(conf, *param);
      add_merge_mode(&param->merge_mode, conf);
      return param;
    }
  }

  if (algo_name == "raft_mg_ivf_pq" || algo_name == "cuvs_mg_ivf_pq") {
    auto param = std::make_unique<typename cuvs::bench::cuvs_mg_ivf_pq<T, int64_t>::search_param>();
    parse_search_param<T, int64_t>(conf, *param);
    add_merge_mode(&param->merge_mode, conf);
    return param;
  }

  if (algo_name == "raft_mg_cagra" || algo_name == "cuvs_mg_cagra") {
    auto param = std::make_unique<typename cuvs::bench::cuvs_mg_cagra<T, uint32_t>::search_param>();
    parse_search_param<T, uint32_t>(conf, *param);
    add_merge_mode(&param->merge_mode, conf);
    return param;
  }
#endif

  // else
  throw std::runtime_error("invalid algo: '" + algo_name + "'");
}

};  // namespace cuvs::bench

REGISTER_ALGO_INSTANCE(float);
REGISTER_ALGO_INSTANCE(half);
REGISTER_ALGO_INSTANCE(std::int8_t);
REGISTER_ALGO_INSTANCE(std::uint8_t);

#ifdef ANN_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
int main(int argc, char** argv) { return cuvs::bench::run_main(argc, argv); }
#endif
