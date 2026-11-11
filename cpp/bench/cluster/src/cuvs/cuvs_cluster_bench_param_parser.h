/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cuvs_kmeans_wrapper.h"

#include <cuvs/distance/distance.hpp>
#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>

namespace cuvs::bench::cluster {

inline auto parse_metric(const std::string& metric) -> cuvs::distance::DistanceType
{
  if (metric == "euclidean") { return cuvs::distance::DistanceType::L2Expanded; }
  if (metric == "inner_product") { return cuvs::distance::DistanceType::InnerProduct; }
  if (metric == "cosine") { return cuvs::distance::DistanceType::CosineExpanded; }
  throw std::runtime_error("metric: '" + metric +
                           "', should be one of euclidean/inner_product/cosine");
}

// Keys are the field names of the corresponding cuVS params struct.

inline void parse_param(const nlohmann::json& conf, cuvs_kmeans::param_type& param)
{
  param.n_clusters = conf.at("n_clusters");
  if (conf.contains("metric")) { param.metric = parse_metric(conf.at("metric")); }
  if (conf.contains("max_iter")) { param.max_iter = conf.at("max_iter"); }
  if (conf.contains("tol")) { param.tol = conf.at("tol"); }
  if (conf.contains("init")) {
    std::string init = conf.at("init");
    if (init == "kmeans++") {
      param.init = cuvs::cluster::kmeans::params::KMeansPlusPlus;
    } else if (init == "random") {
      param.init = cuvs::cluster::kmeans::params::Random;
    } else {
      throw std::runtime_error("init: '" + init + "', should be either 'kmeans++' or 'random'");
    }
  }
  if (conf.contains("n_init")) { param.n_init = conf.at("n_init"); }
  if (conf.contains("oversampling_factor")) {
    param.oversampling_factor = conf.at("oversampling_factor");
  }
  if (conf.contains("batch_samples")) { param.batch_samples = conf.at("batch_samples"); }
  if (conf.contains("batch_centroids")) { param.batch_centroids = conf.at("batch_centroids"); }
  if (conf.contains("init_size")) { param.init_size = conf.at("init_size"); }
  if (conf.contains("seed")) { param.rng_state.seed = conf.at("seed"); }
  // Per-iteration logging would end up inside the timed region.
  param.verbosity = rapids_logger::level_enum::warn;
}

inline void parse_param(const nlohmann::json& conf, cuvs::cluster::kmeans::balanced_params& param)
{
  if (conf.contains("metric")) { param.metric = parse_metric(conf.at("metric")); }
  if (conf.contains("n_iters")) { param.n_iters = conf.at("n_iters"); }
  if (conf.contains("balance_lower_tolerance")) {
    param.balance_lower_tolerance = conf.at("balance_lower_tolerance");
  }
  if (conf.contains("balance_upper_tolerance")) {
    param.balance_upper_tolerance = conf.at("balance_upper_tolerance");
  }
  if (conf.contains("centroid_offset")) { param.centroid_offset = conf.at("centroid_offset"); }
  if (conf.contains("donor_selection")) {
    std::string donor = conf.at("donor_selection");
    if (donor == "size_sorted") {
      param.donor_selection = cuvs::cluster::kmeans::balanced_donor_selection::SizeSorted;
    } else if (donor == "random") {
      param.donor_selection = cuvs::cluster::kmeans::balanced_donor_selection::Random;
    } else {
      throw std::runtime_error("donor_selection: '" + donor +
                               "', should be either 'size_sorted' or 'random'");
    }
  }
}

}  // namespace cuvs::bench::cluster
