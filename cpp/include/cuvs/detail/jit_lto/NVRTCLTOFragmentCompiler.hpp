/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/FragmentEntry.hpp>

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct NVRTCLTOFragmentCompiler {
  NVRTCLTOFragmentCompiler();

  std::vector<std::string> standard_compile_opts;
  std::unordered_map<std::string, std::vector<uint8_t>> cache;
  mutable std::shared_mutex cache_mutex_;

  std::unique_ptr<UDFFatbinFragment> compile(std::string const& key, std::string const& code);

 private:
  std::unique_ptr<UDFFatbinFragment> read_cache(std::string const& key) const;
};

NVRTCLTOFragmentCompiler& nvrtc_compiler();
