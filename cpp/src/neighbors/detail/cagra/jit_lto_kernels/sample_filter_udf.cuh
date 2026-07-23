/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../cagra_filter_payload.hpp"

#include <NVRTCLTOFragmentCompiler.hpp>
#include <raft/core/error.hpp>

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <typename SourceIndexT>
inline constexpr const char* cagra_udf_source_index_type_name()
{
  static_assert(std::is_same_v<SourceIndexT, std::uint32_t>,
                "CAGRA filter UDFs currently support SourceIndexT = uint32_t only");
  return "uint32_t";
}

inline std::string instantiate_cagra_sample_filter_udf(std::string const& user_source,
                                                       std::string const& function_name,
                                                       const char* source_index_type)
{
  std::ostringstream oss;
  oss << R"(
using int8_t = signed char;
using uint8_t = unsigned char;
using int32_t = int;
using uint32_t = unsigned int;
using int64_t = long long;
using uint64_t = unsigned long long;
using source_index_t = )"
      << source_index_type << R"(;

namespace cuvs::neighbors::detail {
template <typename SourceIndexT>
extern __device__ bool sample_filter(uint32_t query_id, SourceIndexT node_id, void* filter_data);
}  // namespace cuvs::neighbors::detail

)";
  oss << user_source << R"(

namespace cuvs::neighbors::detail {

template <>
__device__ bool sample_filter<source_index_t>(uint32_t query_id,
                                              source_index_t node_id,
                                              void* filter_data)
{
  return )"
      << function_name << R"((query_id, node_id, filter_data);
}

}  // namespace cuvs::neighbors::detail
)";
  return oss.str();
}

template <typename SourceIndexT, typename SampleFilterT>
std::unique_ptr<UDFFatbinFragment> make_cagra_sample_filter_udf_fragment(
  const SampleFilterT& sample_filter)
{
  const auto* udf = get_cagra_udf_filter(sample_filter);
  if (udf == nullptr) { return nullptr; }

  RAFT_EXPECTS(!udf->source.empty(), "CAGRA filter UDF source must not be empty");
  RAFT_EXPECTS(!udf->function_name.empty(), "CAGRA filter UDF function name must not be empty");

  auto code = instantiate_cagra_sample_filter_udf(
    udf->source, udf->function_name, cagra_udf_source_index_type_name<SourceIndexT>());
  std::string key = "cagra_sample_filter_udf:";
  key += cagra_udf_source_index_type_name<SourceIndexT>();
  key += ":";
  key += udf->function_name;
  key += ":";
  key += code;
  return nvrtc_compiler().compile(key, code);
}

}  // namespace cuvs::neighbors::cagra::detail
