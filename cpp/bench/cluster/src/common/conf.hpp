/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../ann/src/common/util.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuvs::bench::cluster {

/**
 * Benchmark configuration, read from a JSON file of the form
 *
 *   {
 *     "dataset": {"name": "...", "base_file": "...", "subset_size": 1000000, "dtype": "float"},
 *     "clustering": [{"name": "...", "algo": "cuvs_kmeans", "param": {"n_clusters": 1024}}]
 *   }
 *
 * The dataset keys are those of the ANN benchmark, so the same dataset entries can be reused.
 */
class configuration {
 public:
  struct dataset_conf {
    std::string name;
    std::string base_file;
    uint32_t subset_first_row = 0;
    uint32_t subset_size      = 0;  // 0 means all rows
    std::string dtype         = "float";
  };

  struct clustering {
    std::string name;
    std::string algo;
    nlohmann::json param;
  };

  configuration(std::istream& conf_stream, const std::string& data_prefix)
  {
    auto conf = nlohmann::json::parse(conf_stream);

    const auto& d           = conf.at("dataset");
    dataset_conf_.name      = d.at("name");
    dataset_conf_.base_file = combine_path(data_prefix, d.at("base_file"));
    if (d.contains("subset_first_row")) {
      dataset_conf_.subset_first_row = d.at("subset_first_row");
    }
    if (d.contains("subset_size")) { dataset_conf_.subset_size = d.at("subset_size"); }
    if (d.contains("dtype")) { dataset_conf_.dtype = d.at("dtype"); }

    for (const auto& c : conf.at("clustering")) {
      clusterings_.push_back({c.at("name").get<std::string>(),
                              c.at("algo").get<std::string>(),
                              c.value("param", nlohmann::json::object())});
    }
    if (clusterings_.empty()) { throw std::runtime_error("no clustering configuration found"); }
  }

  [[nodiscard]] auto get_dataset_conf() const -> const dataset_conf& { return dataset_conf_; }
  [[nodiscard]] auto get_clusterings() -> std::vector<clustering>& { return clusterings_; }

 private:
  dataset_conf dataset_conf_;
  std::vector<clustering> clusterings_;
};

}  // namespace cuvs::bench::cluster
