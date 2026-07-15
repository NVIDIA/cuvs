/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include <raft/core/detail/mdspan_numpy_serializer.hpp>
#include <raft/core/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace cuvs::util {

constexpr std::uint32_t kMaxGraphDegree = 1u << 16;  // 65,536 neighbors per row
constexpr std::uint32_t kMaxIvfNLists   = 1u << 24;  // 16,777,216 inverted lists

/**
 * Multiply N non-negative integer values left-to-right and
 * return false if any intermediate product overflows.
 */
template <typename T, typename... Rest>
inline bool is_mul_no_overflow(T a, T b, Rest... rest)
{
  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>,
                "is_mul_no_overflow requires an unsigned integer type.");
  if (a != 0 && b > std::numeric_limits<T>::max() / a) { return false; }
  if constexpr (sizeof...(Rest) == 0) {
    return true;
  } else {
    return is_mul_no_overflow<T>(a * b, rest...);
  }
}

template <typename T>
inline bool validate_serialized_dtype(const char* dtype_prefix, std::size_t dtype_prefix_size)
{
  if (dtype_prefix == nullptr || dtype_prefix_size != 4) { return false; }

  auto expected_dtype = raft::detail::numpy_serializer::get_numpy_dtype<T>().to_string();
  expected_dtype.resize(dtype_prefix_size, '\0');

  return std::equal(dtype_prefix, dtype_prefix + dtype_prefix_size, expected_dtype.begin());
}

inline bool is_valid_distance_type(cuvs::distance::DistanceType m)
{
  using cuvs::distance::DistanceType;
  // Keep this in sync with the enum in cuvs/distance/distance.hpp.
  switch (m) {
    case DistanceType::L2Expanded:
    case DistanceType::L2SqrtExpanded:
    case DistanceType::CosineExpanded:
    case DistanceType::L1:
    case DistanceType::L2Unexpanded:
    case DistanceType::L2SqrtUnexpanded:
    case DistanceType::InnerProduct:
    case DistanceType::Linf:
    case DistanceType::Canberra:
    case DistanceType::LpUnexpanded:
    case DistanceType::CorrelationExpanded:
    case DistanceType::JaccardExpanded:
    case DistanceType::HellingerExpanded:
    case DistanceType::Haversine:
    case DistanceType::BrayCurtis:
    case DistanceType::JensenShannon:
    case DistanceType::HammingUnexpanded:
    case DistanceType::KLDivergence:
    case DistanceType::RusselRaoExpanded:
    case DistanceType::DiceExpanded:
    case DistanceType::BitwiseHamming:
    case DistanceType::Precomputed:
    case DistanceType::CustomUDF: return true;
    default: return false;
  }
}

inline bool is_valid_codebook_gen(cuvs::neighbors::ivf_pq::codebook_gen g)
{
  using cuvs::neighbors::ivf_pq::codebook_gen;
  switch (g) {
    case codebook_gen::PER_SUBSPACE:
    case codebook_gen::PER_CLUSTER: return true;
    default: return false;
  }
}

inline bool is_valid_list_layout(cuvs::neighbors::ivf_pq::list_layout l)
{
  using cuvs::neighbors::ivf_pq::list_layout;
  switch (l) {
    case list_layout::FLAT:
    case list_layout::INTERLEAVED: return true;
    default: return false;
  }
}

}  // namespace cuvs::util
