/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../common/ann_types.hpp"

#undef WARP_SIZE
#include "faiss_gpu_wrapper.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {
nlohmann::json collect_conf_with_prefix(const nlohmann::json& conf,
                                        const std::string& prefix,
                                        bool remove_prefix = true)
{
  nlohmann::json out;
  for (auto& i : conf.items()) {
    if (i.key().compare(0, prefix.size(), prefix) == 0) {
      auto new_key = remove_prefix ? i.key().substr(prefix.size()) : i.key();
      out[new_key] = i.value();
    }
  }
  return out;
}
}  // namespace

namespace cuvs::bench {

template <typename T>
void parse_base_build_param(const nlohmann::json& conf,
                            typename cuvs::bench::faiss_gpu<T>::build_param& param)
{
  param.nlist = conf.at("nlist");
  if (conf.contains("ratio")) { param.ratio = conf.at("ratio"); }
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::faiss_gpu_ivf_flat<T>::build_param& param)
{
  parse_base_build_param<T>(conf, param);
  if (conf.contains("use_cuvs")) {
    param.use_cuvs = conf.at("use_cuvs");
  } else {
    param.use_cuvs = false;
  }
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::faiss_gpu_ivfpq<T>::build_param& param)
{
  parse_base_build_param<T>(conf, param);
  param.m = conf.at("M");
  if (conf.contains("usePrecomputed")) {
    param.use_precomputed = conf.at("usePrecomputed");
  } else {
    param.use_precomputed = false;
  }
  if (conf.contains("useFloat16")) {
    param.use_float16 = conf.at("useFloat16");
  } else {
    param.use_float16 = false;
  }
  if (conf.contains("use_cuvs")) {
    param.use_cuvs = conf.at("use_cuvs");
  } else {
    param.use_cuvs = false;
  }
  if (conf.contains("bitsPerCode")) {
    param.bitsPerCode = conf.at("bitsPerCode");
  } else {
    param.bitsPerCode = 8;
  }
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::faiss_gpu_ivfsq<T>::build_param& param)
{
  parse_base_build_param<T>(conf, param);
  param.quantizer_type = conf.at("quantizer_type");
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::faiss_gpu_cagra<T>::build_param& param)
{
  if (conf.contains("graph_degree")) {
    param.graph_degree = conf.at("graph_degree");
  } else {
    param.graph_degree = 64;
  }
  if (conf.contains("intermediate_graph_degree")) {
    param.intermediate_graph_degree = conf.at("intermediate_graph_degree");
  } else {
    param.intermediate_graph_degree = 128;
  }
  if (conf.contains("cagra_build_algo")) { param.cagra_build_algo = conf.at("cagra_build_algo"); }
  if (conf.contains("nn_descent_niter")) {
    param.nn_descent_niter = conf.at("nn_descent_niter");
  } else {
    param.nn_descent_niter = 20;
  }
  nlohmann::json ivf_pq_build_conf = collect_conf_with_prefix(conf, "b_");
  if (!ivf_pq_build_conf.empty()) {
    faiss::gpu::IVFPQBuildCagraConfig ivf_pq_build_p;

    if (ivf_pq_build_conf.contains("nlist")) {
      ivf_pq_build_p.n_lists = ivf_pq_build_conf.at("nlist");
    }
    if (ivf_pq_build_conf.contains("niter")) {
      ivf_pq_build_p.kmeans_n_iters = ivf_pq_build_conf.at("niter");
    }
    if (ivf_pq_build_conf.contains("ratio")) {
      ivf_pq_build_p.kmeans_trainset_fraction =
        1.0 / static_cast<double>(ivf_pq_build_conf.at("ratio"));
    }
    if (ivf_pq_build_conf.contains("pq_bits")) {
      ivf_pq_build_p.pq_bits = ivf_pq_build_conf.at("pq_bits");
    }
    if (ivf_pq_build_conf.contains("pq_dim")) {
      ivf_pq_build_p.pq_dim = ivf_pq_build_conf.at("pq_dim");
    }
    param.ivf_pq_build_params = std::make_shared<faiss::gpu::IVFPQBuildCagraConfig>(ivf_pq_build_p);
  }
  nlohmann::json ivf_pq_search_conf = collect_conf_with_prefix(conf, "s_");
  if (!ivf_pq_search_conf.empty()) {
    faiss::gpu::IVFPQSearchCagraConfig ivf_pq_search_p;
    if (ivf_pq_search_conf.contains("nprobe")) {
      ivf_pq_search_p.n_probes = ivf_pq_search_conf.at("nprobe");
    }
    if (ivf_pq_search_conf.contains("internalDistanceDtype")) {
      std::string type = ivf_pq_search_conf.at("internalDistanceDtype");
      if (type == "float") {
        ivf_pq_search_p.internal_distance_dtype = CUDA_R_32F;
      } else if (type == "half") {
        ivf_pq_search_p.internal_distance_dtype = CUDA_R_16F;
      } else {
        throw std::runtime_error("internalDistanceDtype: '" + type +
                                 "', should be either 'float' or 'half'");
      }
    } else {
      // set half as default type
      ivf_pq_search_p.internal_distance_dtype = CUDA_R_16F;
    }

    if (ivf_pq_search_conf.contains("smemLutDtype")) {
      std::string type = ivf_pq_search_conf.at("smemLutDtype");
      if (type == "float") {
        ivf_pq_search_p.lut_dtype = CUDA_R_32F;
      } else if (type == "half") {
        ivf_pq_search_p.lut_dtype = CUDA_R_16F;
      } else if (type == "fp8") {
        ivf_pq_search_p.lut_dtype = CUDA_R_8U;
      } else {
        throw std::runtime_error("smemLutDtype: '" + type +
                                 "', should be either 'float', 'half' or 'fp8'");
      }
    } else {
      // set half as default
      ivf_pq_search_p.lut_dtype = CUDA_R_16F;
    }
    param.ivf_pq_search_params =
      std::make_shared<faiss::gpu::IVFPQSearchCagraConfig>(ivf_pq_search_p);
  }
}

template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::faiss_gpu_cagra_hnsw<T>::build_param& param)
{
  typename cuvs::bench::faiss_gpu_cagra<T>::build_param p;
  parse_build_param<T>(conf, p);
  param.p = p;
  if (conf.contains("base_level_only")) { param.base_level_only = conf.at("base_level_only"); }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::faiss_gpu<T>::search_param& param)
{
  if (conf.contains("nprobe")) { param.nprobe = conf.at("nprobe"); }
  if (conf.contains("refine_ratio")) { param.refine_ratio = conf.at("refine_ratio"); }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::faiss_gpu_cagra<T>::search_param& param)
{
  if (conf.contains("itopk")) { param.p.itopk_size = conf.at("itopk"); }
  if (conf.contains("search_width")) { param.p.search_width = conf.at("search_width"); }
  if (conf.contains("max_iterations")) { param.p.max_iterations = conf.at("max_iterations"); }
  if (conf.contains("algo")) {
    if (conf.at("algo") == "single_cta") {
      param.p.algo = faiss::gpu::search_algo::SINGLE_CTA;
    } else if (conf.at("algo") == "multi_cta") {
      param.p.algo = faiss::gpu::search_algo::MULTI_CTA;
    } else if (conf.at("algo") == "multi_kernel") {
      param.p.algo = faiss::gpu::search_algo::MULTI_KERNEL;
    } else if (conf.at("algo") == "auto") {
      param.p.algo = faiss::gpu::search_algo::AUTO;
    } else {
      std::string tmp = conf.at("algo");
      THROW("Invalid value for algo: %s", tmp.c_str());
    }
  }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::faiss_gpu_cagra_hnsw<T>::search_param& param)
{
  if (conf.contains("efSearch")) { param.p.efSearch = conf.at("efSearch"); }
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
    if (algo_name == "faiss_gpu_ivf_flat") {
      a = make_algo<T, cuvs::bench::faiss_gpu_ivf_flat>(metric, dim, conf);
    } else if (algo_name == "faiss_gpu_ivf_pq") {
      a = make_algo<T, cuvs::bench::faiss_gpu_ivfpq>(metric, dim, conf);
    } else if (algo_name == "faiss_gpu_ivf_sq") {
      a = make_algo<T, cuvs::bench::faiss_gpu_ivfsq>(metric, dim, conf);
    } else if (algo_name == "faiss_gpu_flat") {
      a = std::make_unique<cuvs::bench::faiss_gpu_flat<T>>(metric, dim);
    } else if (algo_name == "faiss_gpu_cagra") {
      a = make_algo<T, cuvs::bench::faiss_gpu_cagra>(metric, dim, conf);
    } else if (algo_name == "faiss_gpu_cagra_hnsw") {
      a = make_algo<T, cuvs::bench::faiss_gpu_cagra_hnsw>(metric, dim, conf);
    }
  }

  if (!a) { throw std::runtime_error("invalid algo: '" + algo_name + "'"); }

  return a;
}

template <typename T>
auto create_search_param(const std::string& algo_name, const nlohmann::json& conf)
  -> std::unique_ptr<typename cuvs::bench::algo<T>::search_param>
{
  if (algo_name == "faiss_gpu_ivf_flat" || algo_name == "faiss_gpu_ivf_pq" ||
      algo_name == "faiss_gpu_ivf_sq") {
    auto param = std::make_unique<typename cuvs::bench::faiss_gpu<T>::search_param>();
    parse_search_param<T>(conf, *param);
    return param;
  } else if (algo_name == "faiss_gpu_flat") {
    auto param = std::make_unique<typename cuvs::bench::faiss_gpu<T>::search_param>();
    return param;
  } else if (algo_name == "faiss_gpu_cagra") {
    auto param = std::make_unique<typename cuvs::bench::faiss_gpu_cagra<T>::search_param>();
    parse_search_param<T>(conf, *param);
    return param;
  } else if (algo_name == "faiss_gpu_cagra_hnsw") {
    auto param = std::make_unique<typename cuvs::bench::faiss_gpu_cagra_hnsw<T>::search_param>();
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
