/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../common/benchmark.hpp"
#include "cuvs_cluster_bench_param_parser.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cuvs::bench::cluster {

template <typename T>
auto create_algo(const std::string& name,
                 const nlohmann::json& param,
                 const raft::resources& res) -> std::unique_ptr<algo<T>>
{
  if (name == "cuvs_kmeans") {
    if constexpr (std::is_same_v<T, float>) {
      cuvs_kmeans::param_type p;
      parse_param(param, p);
      return std::make_unique<cuvs_kmeans>(res, p);
    } else {
      throw std::runtime_error("cuvs_kmeans supports only dtype 'float'");
    }
  }
  if (name == "cuvs_kmeans_balanced") {
    typename cuvs_kmeans_balanced<T>::param_type p;
    parse_param(param, p);
    return std::make_unique<cuvs_kmeans_balanced<T>>(res, p, param.at("n_clusters").get<int64_t>());
  }
  throw std::runtime_error("invalid algo: '" + name + "'");
}

}  // namespace cuvs::bench::cluster

int main(int argc, char** argv) { return cuvs::bench::cluster::run_main(argc, argv); }
