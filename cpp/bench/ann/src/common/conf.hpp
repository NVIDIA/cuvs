/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "util.hpp"

#include <iostream>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace cuvs::bench {

class configuration {
 public:
  struct index {
    std::string name;
    std::string algo;
    nlohmann::json build_param;
    std::string file;

    int batch_size;
    int k;
    std::vector<nlohmann::json> search_params;
  };

  struct dataset_conf {
    std::string name;
    std::string base_file;
    // use only a subset of base_file,
    // the range of rows is [subset_first_row, subset_first_row + subset_size)
    // however, subset_size = 0 means using all rows after subset_first_row
    // that is, the subset is [subset_first_row, #rows in base_file)
    uint32_t subset_first_row{0};
    uint32_t subset_size{0};
    std::string query_file;
    std::string distance;
    std::optional<std::string> groundtruth_neighbors_file{std::nullopt};

    // data type of input dataset, possible values ["float", "int8", "uint8"]
    std::string dtype;

    std::optional<double> filtering_rate{std::nullopt};
  };

  [[nodiscard]] inline auto get_dataset_conf() const -> const dataset_conf&
  {
    return dataset_conf_;
  }
  [[nodiscard]] inline auto get_dataset_conf() -> dataset_conf& { return dataset_conf_; }
  [[nodiscard]] inline auto get_indices() const -> const std::vector<index>& { return indices_; };
  [[nodiscard]] inline auto get_indices() -> std::vector<index>& { return indices_; };

  /** The benchmark initializes the configuration once and has a chance to modify it during the
   * setup. */
  static inline auto initialize(std::istream& conf_stream,
                                std::string data_prefix,
                                std::string index_prefix) -> configuration&
  {
    singleton_ =
      std::unique_ptr<configuration>(new configuration{conf_stream, data_prefix, index_prefix});
    return *singleton_;
  }

  /** Any algorithm can access the benchmark configuration as an immutable context. */
  [[nodiscard]] static inline auto singleton() -> const configuration& { return *singleton_; }

 private:
  explicit inline configuration(std::istream& conf_stream,
                                std::string data_prefix,
                                std::string index_prefix)
  {
    // to enable comments in json
    auto conf = nlohmann::json::parse(conf_stream, nullptr, true, true);

    parse_dataset(conf.at("dataset"), data_prefix);
    parse_index(conf.at("index"), conf.at("search_basic_param"), index_prefix);
  }

  inline void parse_dataset(const nlohmann::json& conf, std::string data_prefix)
  {
    dataset_conf_.name       = conf.at("name");
    dataset_conf_.base_file  = combine_path(data_prefix, conf.at("base_file"));
    dataset_conf_.query_file = combine_path(data_prefix, conf.at("query_file"));
    dataset_conf_.distance   = conf.at("distance");
    if (conf.contains("filtering_rate")) {
      dataset_conf_.filtering_rate.emplace(conf.at("filtering_rate"));
    }

    if (conf.contains("groundtruth_neighbors_file")) {
      dataset_conf_.groundtruth_neighbors_file =
        combine_path(data_prefix, conf.at("groundtruth_neighbors_file"));
    }
    if (conf.contains("subset_first_row")) {
      dataset_conf_.subset_first_row = conf.at("subset_first_row");
    }
    if (conf.contains("subset_size")) { dataset_conf_.subset_size = conf.at("subset_size"); }

    if (conf.contains("dtype")) {
      dataset_conf_.dtype = conf.at("dtype");
    } else {
      auto filename = dataset_conf_.base_file;
      if (filename.size() > 6 && filename.compare(filename.size() - 6, 6, "f16bin") == 0) {
        dataset_conf_.dtype = "half";
      } else if (filename.size() > 9 &&
                 filename.compare(filename.size() - 9, 9, "fp16.fbin") == 0) {
        dataset_conf_.dtype = "half";
      } else if (filename.size() > 4 && filename.compare(filename.size() - 4, 4, "fbin") == 0) {
        dataset_conf_.dtype = "float";
      } else if (filename.size() > 5 && filename.compare(filename.size() - 5, 5, "u8bin") == 0) {
        dataset_conf_.dtype = "uint8";
      } else if (filename.size() > 5 && filename.compare(filename.size() - 5, 5, "i8bin") == 0) {
        dataset_conf_.dtype = "int8";
      } else {
        log_error("Could not determine data type of the dataset %s", filename.c_str());
      }
    }
  }
  inline void parse_index(const nlohmann::json& index_conf,
                          const nlohmann::json& search_basic_conf,
                          std::string index_prefix)
  {
    const int batch_size = search_basic_conf.at("batch_size");
    const int k          = search_basic_conf.at("k");

    for (const auto& conf : index_conf) {
      index index;
      index.name        = conf.at("name");
      index.algo        = conf.at("algo");
      index.build_param = conf.at("build_param");
      index.file        = combine_path(index_prefix, conf.at("file"));
      index.batch_size  = batch_size;
      index.k           = k;

      for (auto param : conf.at("search_params")) {
        /*  ### Special parameters for backward compatibility ###

          - Local values of `k` and `n_queries` take priority.
          - The legacy "batch_size" renamed to `n_queries`.
          - Basic search params are used otherwise.
        */
        if (!param.contains("k")) { param["k"] = k; }
        if (!param.contains("n_queries")) {
          if (param.contains("batch_size")) {
            param["n_queries"] = param["batch_size"];
            param.erase("batch_size");
          } else {
            param["n_queries"] = batch_size;
          }
        }
        index.search_params.push_back(param);
      }

      indices_.push_back(index);
    }
  }

  dataset_conf dataset_conf_;
  std::vector<index> indices_;

  static inline std::unique_ptr<configuration> singleton_ = nullptr;
};

}  // namespace cuvs::bench
