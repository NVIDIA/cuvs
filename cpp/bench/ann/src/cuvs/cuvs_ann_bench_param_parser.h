/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#undef WARP_SIZE
#ifdef CUVS_ANN_BENCH_USE_CUVS_BRUTE_FORCE
#include "cuvs_wrapper.h"
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT
#include "cuvs_ivf_flat_wrapper.h"
extern template class cuvs::bench::cuvs_ivf_flat<float, int64_t>;
extern template class cuvs::bench::cuvs_ivf_flat<uint8_t, int64_t>;
extern template class cuvs::bench::cuvs_ivf_flat<int8_t, int64_t>;
#endif
#if defined(CUVS_ANN_BENCH_USE_CUVS_IVF_PQ) || defined(CUVS_ANN_BENCH_USE_CUVS_CAGRA) || \
  defined(CUVS_ANN_BENCH_USE_CUVS_CAGRA_HNSWLIB) || defined(CUVS_ANN_BENCH_USE_CUVS_CAGRA_DISKANN)
#include "cuvs_ivf_pq_wrapper.h"
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_PQ
extern template class cuvs::bench::cuvs_ivf_pq<float, int64_t>;
extern template class cuvs::bench::cuvs_ivf_pq<uint8_t, int64_t>;
extern template class cuvs::bench::cuvs_ivf_pq<int8_t, int64_t>;
#endif
#if defined(CUVS_ANN_BENCH_USE_CUVS_IVF_RABITQ)
#include "cuvs_ivf_rabitq_wrapper.h"
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_RABITQ
extern template class cuvs::bench::cuvs_ivf_rabitq<float, int64_t>;
#endif
#if defined(CUVS_ANN_BENCH_USE_CUVS_CAGRA) || defined(CUVS_ANN_BENCH_USE_CUVS_CAGRA_HNSWLIB) || \
  defined(CUVS_ANN_BENCH_USE_CUVS_CAGRA_DISKANN)
#include "cuvs_cagra_wrapper.h"
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_CAGRA
extern template class cuvs::bench::cuvs_cagra<float, uint32_t>;
extern template class cuvs::bench::cuvs_cagra<half, uint32_t>;
extern template class cuvs::bench::cuvs_cagra<uint8_t, uint32_t>;
extern template class cuvs::bench::cuvs_cagra<int8_t, uint32_t>;
#endif

#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_SQ
#include "cuvs_ivf_sq_wrapper.h"
extern template class cuvs::bench::cuvs_ivf_sq<float>;
extern template class cuvs::bench::cuvs_ivf_sq<half>;
#endif
#ifdef CUVS_ANN_BENCH_USE_CUVS_MG
#include "cuvs_ivf_flat_wrapper.h"
#include "cuvs_mg_ivf_flat_wrapper.h"

#include "cuvs_ivf_pq_wrapper.h"
#include "cuvs_mg_ivf_pq_wrapper.h"

#include "cuvs_cagra_wrapper.h"
#include "cuvs_mg_cagra_wrapper.h"
#endif

template <typename ParamT>
void parse_dynamic_batching_params(const nlohmann::json& conf, ParamT& param)
{
  if (conf.is_null() || !conf.value("dynamic_batching", false)) { return; }
  param.dynamic_batching = true;
  if (conf.contains("dynamic_batching_max_batch_size")) {
    param.dynamic_batching_max_batch_size = conf.at("dynamic_batching_max_batch_size");
  }
  param.dynamic_batching_conservative_dispatch =
    conf.value("dynamic_batching_conservative_dispatch", false);
  if (conf.contains("dynamic_batching_dispatch_timeout_ms")) {
    param.dynamic_batching_dispatch_timeout_ms = conf.at("dynamic_batching_dispatch_timeout_ms");
  }
  if (conf.contains("dynamic_batching_n_queues")) {
    param.dynamic_batching_n_queues = conf.at("dynamic_batching_n_queues");
  }
  param.dynamic_batching_k =
    uint32_t(uint32_t(conf.at("k")) * float(conf.value("refine_ratio", 1.0f)));
}

#if defined(CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT) || defined(CUVS_ANN_BENCH_USE_CUVS_MG)
template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::cuvs_ivf_flat<T, IdxT>::build_param& param)
{
  param.n_lists = conf.at("nlist");
  if (conf.contains("niter")) { param.kmeans_n_iters = conf.at("niter"); }
  if (conf.contains("ratio")) {
    param.kmeans_trainset_fraction = 1.0 / static_cast<double>(conf.at("ratio"));
  }
}

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::cuvs_ivf_flat<T, IdxT>::search_param& param)
{
  param.ivf_flat_params.n_probes = conf.at("nprobe");
}
#endif

#ifdef CUVS_ANN_BENCH_USE_CUVS_IVF_SQ
template <typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::cuvs_ivf_sq<T>::build_param& param)
{
  param.n_lists = conf.at("nlist");
  if (conf.contains("niter")) { param.kmeans_n_iters = conf.at("niter"); }
  if (conf.contains("ratio")) {
    throw std::runtime_error(
      "cuvs_ivf_sq build parameter ratio was replaced by max_train_points_per_cluster");
  }
  if (conf.contains("max_train_points_per_cluster")) {
    param.max_train_points_per_cluster = conf.at("max_train_points_per_cluster");
  }
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::cuvs_ivf_sq<T>::search_param& param)
{
  param.ivf_sq_params.n_probes = conf.at("nprobe");
}
#endif

#if defined(CUVS_ANN_BENCH_USE_CUVS_IVF_PQ) || defined(CUVS_ANN_BENCH_USE_CUVS_CAGRA) ||   \
  defined(CUVS_ANN_BENCH_USE_CUVS_CAGRA_HNSWLIB) || defined(CUVS_ANN_BENCH_USE_CUVS_MG) || \
  defined(CUVS_ANN_BENCH_USE_CUVS_CAGRA_DISKANN)
template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::cuvs_ivf_pq<T, IdxT>::build_param& param)
{
  if (conf.contains("nlist")) { param.n_lists = conf.at("nlist"); }
  if (conf.contains("niter")) { param.kmeans_n_iters = conf.at("niter"); }
  if (conf.contains("ratio")) {
    param.kmeans_trainset_fraction = 1.0 / static_cast<double>(conf.at("ratio"));
  }
  if (conf.contains("pq_bits")) { param.pq_bits = conf.at("pq_bits"); }
  if (conf.contains("pq_dim")) { param.pq_dim = conf.at("pq_dim"); }
  if (conf.contains("codebook_kind")) {
    std::string kind = conf.at("codebook_kind");
    if (kind == "cluster") {
      param.codebook_kind = cuvs::neighbors::ivf_pq::codebook_gen::PER_CLUSTER;
    } else if (kind == "subspace") {
      param.codebook_kind = cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
    } else {
      throw std::runtime_error("codebook_kind: '" + kind +
                               "', should be either 'cluster' or 'subspace'");
    }
  }
}

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::cuvs_ivf_pq<T, IdxT>::search_param& param)
{
  if (conf.contains("nprobe")) { param.pq_param.n_probes = conf.at("nprobe"); }
  if (conf.contains("internalDistanceDtype")) {
    std::string type = conf.at("internalDistanceDtype");
    if (type == "float") {
      param.pq_param.internal_distance_dtype = CUDA_R_32F;
    } else if (type == "half") {
      param.pq_param.internal_distance_dtype = CUDA_R_16F;
    } else {
      throw std::runtime_error("internalDistanceDtype: '" + type +
                               "', should be either 'float' or 'half'");
    }
  } else {
    // set half as default type
    param.pq_param.internal_distance_dtype = CUDA_R_16F;
  }

  if (conf.contains("smemLutDtype")) {
    std::string type = conf.at("smemLutDtype");
    if (type == "float") {
      param.pq_param.lut_dtype = CUDA_R_32F;
    } else if (type == "half") {
      param.pq_param.lut_dtype = CUDA_R_16F;
    } else if (type == "fp8") {
      param.pq_param.lut_dtype = CUDA_R_8U;
    } else {
      throw std::runtime_error("smemLutDtype: '" + type +
                               "', should be either 'float', 'half' or 'fp8'");
    }
  } else {
    // set half as default
    param.pq_param.lut_dtype = CUDA_R_16F;
  }

  if (conf.contains("coarse_search_dtype")) {
    std::string type = conf.at("coarse_search_dtype");
    if (type == "float") {
      param.pq_param.coarse_search_dtype = CUDA_R_32F;
    } else if (type == "half") {
      param.pq_param.coarse_search_dtype = CUDA_R_16F;
    } else if (type == "int8") {
      param.pq_param.coarse_search_dtype = CUDA_R_8I;
    } else {
      throw std::runtime_error("coarse_search_dtype: '" + type +
                               "', should be either 'float', 'half' or 'int8'");
    }
  }

  if (conf.contains("max_internal_batch_size")) {
    param.pq_param.max_internal_batch_size = conf.at("max_internal_batch_size");
  }

  if (conf.contains("refine_ratio")) {
    param.refine_ratio = conf.at("refine_ratio");
    if (param.refine_ratio < 1.0f) { throw std::runtime_error("refine_ratio should be >= 1.0"); }
  }

  // enable dynamic batching
  parse_dynamic_batching_params(conf, param);
}
#endif

#if defined(CUVS_ANN_BENCH_USE_CUVS_IVF_RABITQ)
template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::cuvs_ivf_rabitq<T, IdxT>::build_param& param)
{
  if (conf.contains("nlist")) { param.n_lists = conf.at("nlist"); }
  if (conf.contains("niter")) { param.kmeans_n_iters = conf.at("niter"); }
  if (conf.contains("max_points_per_cluster")) {
    param.max_train_points_per_cluster = conf.at("max_points_per_cluster");
  }
  if (conf.contains("bits_per_dim")) { param.bits_per_dim = conf.at("bits_per_dim"); }
  if (conf.contains("fast_quantize_flag")) {
    param.fast_quantize_flag = conf.at("fast_quantize_flag");
  }
  if (conf.contains("force_streaming")) { param.force_streaming = conf.at("force_streaming"); }
}

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::cuvs_ivf_rabitq<T, IdxT>::search_param& param)
{
  if (conf.contains("nprobe")) { param.rabitq_param.n_probes = conf.at("nprobe"); }

  if (conf.contains("mode")) {
    std::string mode = conf.at("mode");
    if (mode == "lut16") {
      param.rabitq_param.mode = cuvs::neighbors::ivf_rabitq::search_mode::LUT16;
    } else if (mode == "lut32") {
      param.rabitq_param.mode = cuvs::neighbors::ivf_rabitq::search_mode::LUT32;
    } else if (mode == "quant4") {
      param.rabitq_param.mode = cuvs::neighbors::ivf_rabitq::search_mode::QUANT4;
    } else if (mode == "quant8") {
      param.rabitq_param.mode = cuvs::neighbors::ivf_rabitq::search_mode::QUANT8;
    } else {
      throw std::runtime_error("mode: '" + mode +
                               "', should be either 'lut16', 'lut32', 'quant4' or 'quant8'");
    }
  }
}
#endif

#if defined(CUVS_ANN_BENCH_USE_CUVS_CAGRA) || defined(CUVS_ANN_BENCH_USE_CUVS_CAGRA_HNSWLIB) || \
  defined(CUVS_ANN_BENCH_USE_CUVS_MG) || defined(CUVS_ANN_BENCH_USE_CUVS_CAGRA_DISKANN)
template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf, cuvs::neighbors::nn_descent::index_params& param)
{
  if (conf.contains("graph_degree")) { param.graph_degree = conf.at("graph_degree"); }
  if (conf.contains("intermediate_graph_degree")) {
    param.intermediate_graph_degree = conf.at("intermediate_graph_degree");
  }
  // we allow niter shorthand for max_iterations
  if (conf.contains("niter")) { param.max_iterations = conf.at("niter"); }
  if (conf.contains("max_iterations")) { param.max_iterations = conf.at("max_iterations"); }
  if (conf.contains("termination_threshold")) {
    param.termination_threshold = conf.at("termination_threshold");
  }
}

inline void parse_build_param(const nlohmann::json& conf, cuvs::neighbors::vpq_params& param)
{
  if (conf.contains("pq_bits")) { param.pq_bits = conf.at("pq_bits"); }
  if (conf.contains("pq_dim")) { param.pq_dim = conf.at("pq_dim"); }
  if (conf.contains("vq_n_centers")) { param.vq_n_centers = conf.at("vq_n_centers"); }
  if (conf.contains("kmeans_n_iters")) { param.kmeans_n_iters = conf.at("kmeans_n_iters"); }
  if (conf.contains("vq_kmeans_trainset_fraction")) {
    param.vq_kmeans_trainset_fraction = conf.at("vq_kmeans_trainset_fraction");
  }
  if (conf.contains("pq_kmeans_trainset_fraction")) {
    param.pq_kmeans_trainset_fraction = conf.at("pq_kmeans_trainset_fraction");
  }
  if (conf.contains("pq_kmeans_type")) {
    std::string type = conf.at("pq_kmeans_type");
    if (type == "KMeans") {
      param.pq_kmeans_type = cuvs::cluster::kmeans::kmeans_type::KMeans;
    } else if (type == "KMeansBalanced") {
      param.pq_kmeans_type = cuvs::cluster::kmeans::kmeans_type::KMeansBalanced;
    } else {
      throw std::runtime_error("pq_kmeans_type: '" + type +
                               "', should be either 'KMeans' or 'KMeansBalanced'");
    }
  }
  if (conf.contains("max_train_points_per_pq_code")) {
    param.max_train_points_per_pq_code = conf.at("max_train_points_per_pq_code");
  }
  if (conf.contains("max_train_points_per_vq_cluster")) {
    param.max_train_points_per_vq_cluster = conf.at("max_train_points_per_vq_cluster");
  }
}

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

template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf, cuvs::neighbors::cagra::index_params& params)
{
  // NB: try to avoid setting parameters implicitly, because this parsing may be used after
  // parameter heuristics are applied (e.g. in cuvs_cagra_hnswlib.cu).
  if (conf.contains("graph_degree")) { params.graph_degree = conf.at("graph_degree"); }
  if (conf.contains("intermediate_graph_degree")) {
    params.intermediate_graph_degree = conf.at("intermediate_graph_degree");
  } else {
    // Only update the intermediate graph degree if it's invalid.
    params.intermediate_graph_degree =
      std::max(params.graph_degree, params.intermediate_graph_degree);
  }

  nlohmann::json comp_search_conf = collect_conf_with_prefix(conf, "compression_");
  if (!comp_search_conf.empty()) {
    auto vpq_pams = params.compression.value_or(cuvs::neighbors::vpq_params{});
    parse_build_param(comp_search_conf, vpq_pams);
    params.compression.emplace(vpq_pams);
  }

  if (conf.contains("guarantee_connectivity")) {
    params.guarantee_connectivity = conf.at("guarantee_connectivity");
  }

  if (conf.contains("variable_graph_degree_fraction")) {
    params.variable_graph_degree_fraction = conf.at("variable_graph_degree_fraction");
  }

  // Extract build-algo-specific parameters
  nlohmann::json ivf_pq_build_conf  = collect_conf_with_prefix(conf, "ivf_pq_build_");
  nlohmann::json ivf_pq_search_conf = collect_conf_with_prefix(conf, "ivf_pq_search_");
  nlohmann::json nn_descent_conf    = collect_conf_with_prefix(conf, "nn_descent_");
  nlohmann::json ace_conf           = collect_conf_with_prefix(conf, "ace_");

  // Determine and initialize graph build algorithm.
  // Priority 1: explicit "graph_build_algo" config key.
  // Priority 2: infer from algorithm-specific prefixed config keys (only when monostate).
  // Priority 3: leave as-is (from prior heuristics or monostate for AUTO at build time).
  std::string graph_build_algo;
  if (conf.contains("graph_build_algo")) {
    graph_build_algo = conf.at("graph_build_algo");
  } else if (std::holds_alternative<std::monostate>(params.graph_build_params)) {
    if (!ivf_pq_build_conf.empty() || !ivf_pq_search_conf.empty()) {
      graph_build_algo = "IVF_PQ";
    } else if (!nn_descent_conf.empty()) {
      graph_build_algo = "NN_DESCENT";
    } else if (!ace_conf.empty()) {
      graph_build_algo = "ACE";
    }
    // else: leave as monostate → AUTO in cagra_build.cuh
  }

  if (!graph_build_algo.empty()) {
    if (graph_build_algo == "IVF_PQ") {
      if (!std::holds_alternative<cuvs::neighbors::graph_build_params::ivf_pq_params>(
            params.graph_build_params)) {
        params.graph_build_params = cuvs::neighbors::graph_build_params::ivf_pq_params{};
      }
    } else if (graph_build_algo == "NN_DESCENT") {
      if (!std::holds_alternative<cuvs::neighbors::graph_build_params::nn_descent_params>(
            params.graph_build_params)) {
        params.graph_build_params = cuvs::neighbors::graph_build_params::nn_descent_params(
          params.intermediate_graph_degree, params.metric);
      }
    } else if (graph_build_algo == "ACE") {
      if (!std::holds_alternative<cuvs::neighbors::graph_build_params::ace_params>(
            params.graph_build_params)) {
        params.graph_build_params = cuvs::neighbors::graph_build_params::ace_params{};
      }
    } else if (graph_build_algo == "ITERATIVE_SEARCH") {
      if (!std::holds_alternative<cuvs::neighbors::graph_build_params::iterative_search_params>(
            params.graph_build_params)) {
        params.graph_build_params = cuvs::neighbors::graph_build_params::iterative_search_params{};
      }
    }
  }

  // Apply build-algo-specific parameters
  std::visit(
    [&](auto& arg) {
      using U = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<U, cuvs::neighbors::graph_build_params::ivf_pq_params>) {
        parse_build_param<T, IdxT>(ivf_pq_build_conf, arg.build_params);
        typename cuvs::bench::cuvs_ivf_pq<T, IdxT>::search_param sparam;
        sparam.pq_param     = arg.search_params;
        sparam.refine_ratio = arg.refinement_rate;
        parse_search_param<T, IdxT>(ivf_pq_search_conf, sparam);
        arg.search_params   = sparam.pq_param;
        arg.refinement_rate = sparam.refine_ratio;
      } else if constexpr (std::is_same_v<U,
                                          cuvs::neighbors::graph_build_params::nn_descent_params>) {
        parse_build_param<T, IdxT>(nn_descent_conf, arg);
      }
    },
    params.graph_build_params);
}

template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::cuvs_cagra<T, IdxT>::build_param& param)
{
  if (conf.contains("num_dataset_splits")) {
    param.num_dataset_splits = conf.at("num_dataset_splits");
  }
  if (conf.contains("merge_type")) {
    std::string mt = conf.at("merge_type");
    if (mt == "PHYSICAL") {
      param.merge_type = cuvs::bench::CagraMergeType::kPhysical;
    } else if (mt == "LOGICAL") {
      param.merge_type = cuvs::bench::CagraMergeType::kLogical;
    } else {
      throw std::runtime_error("invalid value for merge_type");
    }
  }
  param.cagra_params = [conf](raft::matrix_extent<int64_t> extents,
                              cuvs::distance::DistanceType dist_type) {
    // Delayed parsing/initialization of cagra_params - it's called once the dataset shape is known
    auto cagra_params   = cuvs::neighbors::cagra::index_params{};
    cagra_params.metric = dist_type;
    if (conf.value("graph_build_algo", "") == "IVF_PQ") {
      cagra_params.graph_build_params =
        cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(extents, dist_type);
    } else if (conf.value("graph_build_algo", "") == "NN_DESCENT") {
      cagra_params.graph_build_params =
        cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
          conf.value("intermediate_graph_degree", cagra_params.intermediate_graph_degree),
          dist_type);
    } else if (conf.value("graph_build_algo", "") == "ACE") {
      cagra_params.graph_build_params = cuvs::neighbors::cagra::graph_build_params::ace_params{};
    }
    // Parse ACE parameters if provided
    nlohmann::json ace_conf = collect_conf_with_prefix(conf, "ace_");
    if (!ace_conf.empty()) {
      auto ace_params = cuvs::neighbors::cagra::graph_build_params::ace_params();
      if (ace_conf.contains("npartitions")) { ace_params.npartitions = ace_conf.at("npartitions"); }
      if (ace_conf.contains("build_dir")) { ace_params.build_dir = ace_conf.at("build_dir"); }
      if (ace_conf.contains("ef_construction")) {
        ace_params.ef_construction = ace_conf.at("ef_construction");
      }
      if (ace_conf.contains("use_disk")) { ace_params.use_disk = ace_conf.at("use_disk"); }
      cagra_params.graph_build_params = ace_params;
    }
    ::parse_build_param<T, IdxT>(conf, cagra_params);
    return cagra_params;
  };
}

cuvs::bench::AllocatorType parse_allocator(std::string mem_type)
{
  if (mem_type == "device") {
    return cuvs::bench::AllocatorType::kDevice;
  } else if (mem_type == "host_pinned") {
    return cuvs::bench::AllocatorType::kHostPinned;
  } else if (mem_type == "host_huge_page") {
    return cuvs::bench::AllocatorType::kHostHugePage;
  }
  THROW(
    "Invalid value for memory type %s, must be one of [\"device\", \"host_pinned\", "
    "\"host_huge_page\"",
    mem_type.c_str());
}

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::cuvs_cagra<T, IdxT>::search_param& param)
{
  if (conf.contains("itopk")) { param.p.itopk_size = conf.at("itopk"); }
  if (conf.contains("search_width")) { param.p.search_width = conf.at("search_width"); }
  if (conf.contains("max_iterations")) { param.p.max_iterations = conf.at("max_iterations"); }
  if (conf.contains("persistent")) { param.p.persistent = conf.at("persistent"); }
  if (conf.contains("persistent_lifetime")) {
    param.p.persistent_lifetime = conf.at("persistent_lifetime");
  }
  if (conf.contains("persistent_device_usage")) {
    param.p.persistent_device_usage = conf.at("persistent_device_usage");
  }
  if (conf.contains("thread_block_size")) {
    param.p.thread_block_size = conf.at("thread_block_size");
  }
  if (conf.contains("algo")) {
    if (conf.at("algo") == "single_cta") {
      param.p.algo = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
    } else if (conf.at("algo") == "multi_cta") {
      param.p.algo = cuvs::neighbors::cagra::search_algo::MULTI_CTA;
    } else if (conf.at("algo") == "multi_kernel") {
      param.p.algo = cuvs::neighbors::cagra::search_algo::MULTI_KERNEL;
    } else if (conf.at("algo") == "auto") {
      param.p.algo = cuvs::neighbors::cagra::search_algo::AUTO;
    } else {
      std::string tmp = conf.at("algo");
      THROW("Invalid value for algo: %s", tmp.c_str());
    }
  }
  if (conf.contains("graph_memory_type")) {
    param.graph_mem = parse_allocator(conf.at("graph_memory_type"));
  }
  if (conf.contains("internal_dataset_memory_type")) {
    param.dataset_mem = parse_allocator(conf.at("internal_dataset_memory_type"));
  }
  // Same ratio as in IVF-PQ
  param.refine_ratio = conf.value("refine_ratio", 1.0f);

  // enable dynamic batching
  parse_dynamic_batching_params(conf, param);
}
#endif
