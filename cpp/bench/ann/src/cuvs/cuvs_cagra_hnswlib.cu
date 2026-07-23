/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_param_parser.h"
#include "cuvs_cagra_hnswlib_wrapper.h"

#include <rmm/cuda_device.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cuvs::bench {

template <typename T, typename IdxT>
auto parse_build_param(const nlohmann::json& conf) ->
  typename cuvs::bench::cuvs_cagra_hnswlib<T, IdxT>::build_param
{
  typename cuvs::bench::cuvs_cagra_hnswlib<T, IdxT>::build_param param;
  auto& hnsw_params  = param.hnsw_index_params;
  auto& cagra_params = param.cagra_build_params;
  if (conf.contains("hierarchy")) {
    if (conf.at("hierarchy") == "none") {
      hnsw_params.hierarchy = cuvs::neighbors::hnsw::HnswHierarchy::NONE;
    } else if (conf.at("hierarchy") == "cpu") {
      hnsw_params.hierarchy = cuvs::neighbors::hnsw::HnswHierarchy::CPU;
    } else if (conf.at("hierarchy") == "gpu") {
      hnsw_params.hierarchy = cuvs::neighbors::hnsw::HnswHierarchy::GPU;
    } else {
      THROW("Invalid value for hierarchy: %s", conf.at("hierarchy").get<std::string>().c_str());
    }
  } else {
    hnsw_params.hierarchy = cuvs::neighbors::hnsw::HnswHierarchy::GPU;
  }
  if (conf.contains("ef_construction")) {
    hnsw_params.ef_construction = conf.at("ef_construction");
  }
  if (conf.contains("num_threads")) { hnsw_params.num_threads = conf.at("num_threads"); }

  // Reuse the CAGRA wrapper params parser
  ::parse_build_param<T, IdxT>(conf, cagra_params);
  // If the users provides parameter M, we can use the CAGRA-HNSW heuristics to find optimal
  // parameters for the dataset and HNSW reference.
  if (conf.contains("M")) {
    // Postpone the parsing of the CAGRA build params until the dataset extents are known.
    // We the default parameters depend on the dataset extents; and we still would like to be able
    // to override them.
    cagra_params.cagra_params = [conf, hnsw_params](raft::matrix_extent<int64_t> extents,
                                                    cuvs::distance::DistanceType dist_type) {
      auto ps = cuvs::neighbors::cagra::index_params::from_hnsw_params(
        extents,
        conf.at("M"),
        hnsw_params.ef_construction,
        cuvs::neighbors::cagra::hnsw_heuristic_type::SAME_GRAPH_FOOTPRINT,
        dist_type);
      ps.metric = dist_type;
      // Parse ACE parameters if provided
      if (conf.contains("npartitions") || conf.contains("build_dir") ||
          conf.contains("ef_construction") || conf.contains("use_disk")) {
        auto ace_params = cuvs::neighbors::cagra::graph_build_params::ace_params();
        if (conf.contains("npartitions")) { ace_params.npartitions = conf.at("npartitions"); }
        if (conf.contains("build_dir")) { ace_params.build_dir = conf.at("build_dir"); }
        if (conf.contains("ef_construction")) {
          ace_params.ef_construction = conf.at("ef_construction");
        }
        if (conf.contains("use_disk")) { ace_params.use_disk = conf.at("use_disk"); }
        ps.graph_build_params = ace_params;
      }
      // NB: above, we only provide the defaults. Below we parse the explicit parameters as usual.
      ::parse_build_param<T, uint32_t>(conf, ps);
      return ps;
    };
  }
  return param;
}

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::cuvs_cagra_hnswlib<T, IdxT>::search_param& param)
{
  param.hnsw_search_param.ef = conf.at("ef");
  if (conf.contains("num_threads")) {
    param.hnsw_search_param.num_threads = conf.at("num_threads");
  }
}

template <typename T>
auto create_algo(const std::string& algo_name,
                 const std::string& distance,
                 int dim,
                 const nlohmann::json& conf) -> std::unique_ptr<cuvs::bench::algo<T>>
{
  [[maybe_unused]] cuvs::bench::Metric metric = parse_metric(distance);
  std::unique_ptr<cuvs::bench::algo<T>> a;

  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, int8_t> ||
                std::is_same_v<T, uint8_t>) {
    if (algo_name == "raft_cagra_hnswlib" || algo_name == "cuvs_cagra_hnswlib") {
      auto params = parse_build_param<T, uint32_t>(conf);
      a = std::make_unique<cuvs::bench::cuvs_cagra_hnswlib<T, uint32_t>>(metric, dim, params);
    }
  }

  if (!a) { throw std::runtime_error("invalid algo: '" + algo_name + "'"); }

  return a;
}

template <typename T>
auto create_search_param(const std::string& algo_name, const nlohmann::json& conf)
  -> std::unique_ptr<typename cuvs::bench::algo<T>::search_param>
{
  if (algo_name == "raft_cagra_hnswlib" || algo_name == "cuvs_cagra_hnswlib") {
    auto param =
      std::make_unique<typename cuvs::bench::cuvs_cagra_hnswlib<T, uint32_t>::search_param>();
    parse_search_param<T, uint32_t>(conf, *param);
    return param;
  }

  throw std::runtime_error("invalid algo: '" + algo_name + "'");
}

}  // namespace cuvs::bench

REGISTER_ALGO_INSTANCE(float);
REGISTER_ALGO_INSTANCE(half);
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
