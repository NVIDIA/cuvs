/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Helpers shared by the ANN and clustering benchmark drivers.

#include <benchmark/benchmark.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

namespace cuvs::bench {

using kv_series = std::vector<std::tuple<std::string, std::vector<nlohmann::json>>>;

inline auto apply_overrides(const std::vector<nlohmann::json>& configs,
                            const kv_series& overrides,
                            std::size_t override_idx = 0) -> std::vector<nlohmann::json>
{
  std::vector<nlohmann::json> results{};
  if (override_idx >= overrides.size()) {
    auto n = configs.size();
    for (size_t i = 0; i < n; i++) {
      auto c               = configs[i];
      c["override_suffix"] = n > 1 ? "/" + std::to_string(i) : "";
      results.push_back(c);
    }
    return results;
  }
  auto rec_configs = apply_overrides(configs, overrides, override_idx + 1);
  auto [key, vals] = overrides[override_idx];
  auto n           = vals.size();
  for (size_t i = 0; i < n; i++) {
    const auto& val = vals[i];
    for (auto rc : rec_configs) {
      if (n > 1) {
        rc["override_suffix"] =
          static_cast<std::string>(rc["override_suffix"]) + "/" + std::to_string(i);
      }
      rc[key] = val;
      results.push_back(rc);
    }
  }
  return results;
}

inline auto apply_overrides(const nlohmann::json& config,
                            const kv_series& overrides,
                            std::size_t override_idx = 0)
{
  return apply_overrides(std::vector{config}, overrides, 0);
}

inline void dump_parameters(::benchmark::State& state, nlohmann::json params)
{
  std::string label = "";
  bool label_empty  = true;
  for (auto& [key, val] : params.items()) {
    if (val.is_number()) {
      state.counters.insert({{key, val}});
    } else if (val.is_boolean()) {
      state.counters.insert({{key, val ? 1.0 : 0.0}});
    } else {
      auto kv = key + "=" + val.dump();
      if (label_empty) {
        label = kv;
      } else {
        label += "#" + kv;
      }
      label_empty = false;
    }
  }
  if (!label_empty) { state.SetLabel(label); }
}

inline auto parse_bool_flag(const char* arg, const char* pat, bool& result) -> bool
{
  if (strcmp(arg, pat) == 0) {
    result = true;
    return true;
  }
  return false;
}

inline auto parse_string_flag(const char* arg, const char* pat, std::string& result) -> bool
{
  auto n = strlen(pat);
  if (strncmp(pat, arg, n) == 0 && arg[n] == '=') {
    result = arg + n + 1;
    return true;
  }
  return false;
}

}  // namespace cuvs::bench
