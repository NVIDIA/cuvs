/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>
#include <cuvs/detail/jit_lto/common_fragments.hpp>

#include "../../sample_filter.cuh"   // For none_sample_filter, bitset_filter
#include "cagra_filter_payload.hpp"  // sample-filter payload helpers

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace cuvs::neighbors::cagra::detail {

// Helper functions to get tags for JIT LTO
template <typename T>
constexpr auto get_data_type_tag()
{
  if constexpr (std::is_same_v<T, float>) { return cuvs::neighbors::detail::tag_f{}; }
  if constexpr (std::is_same_v<T, __half>) { return cuvs::neighbors::detail::tag_h{}; }
  if constexpr (std::is_same_v<T, int8_t>) { return cuvs::neighbors::detail::tag_i8{}; }
  if constexpr (std::is_same_v<T, uint8_t>) { return cuvs::neighbors::detail::tag_u8{}; }
}

template <typename T>
constexpr auto get_index_type_tag()
{
  if constexpr (std::is_same_v<T, uint32_t>) { return cuvs::neighbors::detail::tag_index_u32{}; }
}

template <typename T>
constexpr auto get_distance_type_tag()
{
  if constexpr (std::is_same_v<T, float>) { return cuvs::neighbors::cagra::detail::tag_dist_f{}; }
}

template <typename T>
constexpr auto get_source_index_type_tag()
{
  static_assert(
    std::is_same_v<T, uint32_t>,
    "CAGRA JIT LTO: no fatbins are built for this SourceIndexT (matrices only list uint32_t). "
    "Keep SourceIndexT as the graph index type (typically uint32_t). For int64 neighbor outputs, "
    "use OutputIndexT = int64_t instead.");
  return cuvs::neighbors::detail::tag_index_u32{};
}

template <typename DataTag, cuvs::distance::DistanceType metric>
struct query_type_tag_standard {
  using type = std::conditional_t<metric == cuvs::distance::DistanceType::BitwiseHamming &&
                                    std::is_same_v<DataTag, cuvs::neighbors::detail::tag_u8>,
                                  cuvs::neighbors::detail::tag_u8,
                                  cuvs::neighbors::detail::tag_f>;
};

template <typename DataTag, cuvs::distance::DistanceType metric>
using query_type_tag_standard_t = typename query_type_tag_standard<DataTag, metric>::type;

template <typename DataTag>
using query_type_tag_vpq_t = cuvs::neighbors::detail::tag_h;

template <typename DataTag>
using query_type_tag_standard_l2_t =
  query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
template <typename DataTag>
using query_type_tag_standard_inner_product_t =
  query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::InnerProduct>;
template <typename DataTag>
using query_type_tag_standard_cosine_t =
  query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::CosineExpanded>;
template <typename DataTag>
using query_type_tag_standard_hamming_t =
  query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::BitwiseHamming>;

using codebook_tag_vpq_t      = cuvs::neighbors::cagra::detail::tag_codebook_half;
using codebook_tag_standard_t = cuvs::neighbors::cagra::detail::tag_codebook_none;

// Dependent false for static_assert in sample_filter_jit_tag (CAGRA JIT).
template <typename T>
inline constexpr bool cagra_jit_sample_filter_tag_type_always_false = false;

/// Maps a host sample-filter type to the JIT fragment tag (`cuvs::neighbors::detail::tag_filter_*`)
/// for `CagraPlannerBase<..., JitTag>` / `add_sample_filter_device_function()`.
template <class SAMPLE_FILTER_T>
struct sample_filter_jit_tag {
 private:
  using DecayedFilter = std::decay_t<SAMPLE_FILTER_T>;

  struct dispatch {
    template <typename U>
    static constexpr auto f()
    {
      using namespace cuvs::neighbors::filtering;
      if constexpr (std::is_same_v<U, none_sample_filter>) {
        return cuvs::neighbors::detail::tag_filter_none{};
      } else if constexpr (is_udf_filter<U>::value) {
        return cuvs::neighbors::detail::tag_filter_udf{};
      } else if constexpr (requires { std::declval<U>().filter; }) {
        using InnerFilter = decltype(std::declval<U>().filter);
        if constexpr (is_bitset_filter<std::decay_t<InnerFilter>>::value ||
                      std::is_same_v<std::decay_t<InnerFilter>, bitset_filter<uint32_t, int64_t>> ||
                      std::is_same_v<std::decay_t<InnerFilter>,
                                     bitset_filter<uint32_t, uint32_t>>) {
          return cuvs::neighbors::detail::tag_filter_bitset{};
        } else if constexpr (is_udf_filter<std::decay_t<InnerFilter>>::value) {
          return cuvs::neighbors::detail::tag_filter_udf{};
        } else {
          static_assert(
            cagra_jit_sample_filter_tag_type_always_false<U>,
            "CAGRA JIT: sample_filter_jit_tag does not know how to link this filter. "
            "CagraSampleFilterWithQueryIdOffset<Inner> requires Inner to be a supported "
            "built-in filter or udf_filter (see cagra_filter_payload.hpp and "
            "sample_filter_utils.cuh). "
            "For a new filter kind, add a sample_filter_jit_tag branch. "
            "(SAMPLE_FILTER_T in error; check InnerFilter in compiler output.)");
        }
      } else {
        static_assert(
          cagra_jit_sample_filter_tag_type_always_false<U>,
          "CAGRA JIT: sample_filter_jit_tag: SAMPLE_FILTER_T must be cuvs::neighbors::filtering::"
          "none_sample_filter, udf_filter, or "
          "cuvs::neighbors::cagra::detail::CagraSampleFilterWithQueryIdOffset<supported filter>. "
          "Unknown wrapper type. "
          "(SAMPLE_FILTER_T in error; add a branch in sample_filter_jit_tag.)");
      }
    }
  };

 public:
  using type = decltype(dispatch::template f<DecayedFilter>());
};

template <class SAMPLE_FILTER_T>
using sample_filter_jit_tag_t = typename sample_filter_jit_tag<SAMPLE_FILTER_T>::type;

}  // namespace cuvs::neighbors::cagra::detail
