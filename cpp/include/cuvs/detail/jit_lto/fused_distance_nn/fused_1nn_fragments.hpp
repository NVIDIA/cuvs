/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <cuvs/distance/distance.hpp>

namespace cuvs::distance::detail {

struct metric_tag_ip {};
struct metric_tag_l2 {};
struct metric_tag_cos {};
struct cutile_abi_strict {};
struct cutile_abi_relaxed {};

template <int TileM, int TileN, int TileK>
struct cutile_tile_config {
  static constexpr int tile_m = TileM;
  static constexpr int tile_n = TileN;
  static constexpr int tile_k = TileK;
};

template <cuvs::distance::DistanceType Metric>
struct fused_1nn_metric_tag;

template <>
struct fused_1nn_metric_tag<cuvs::distance::DistanceType::InnerProduct> {
  using type = metric_tag_ip;
};

template <>
struct fused_1nn_metric_tag<cuvs::distance::DistanceType::L2Expanded> {
  using type = metric_tag_l2;
};

template <>
struct fused_1nn_metric_tag<cuvs::distance::DistanceType::L2SqrtExpanded> {
  using type = metric_tag_l2;
};

template <>
struct fused_1nn_metric_tag<cuvs::distance::DistanceType::CosineExpanded> {
  using type = metric_tag_cos;
};

/** Whether sqrt is applied when packing distance into KVP output. */
template <cuvs::distance::DistanceType Metric>
constexpr bool fused_1nn_apply_sqrt_at_pack(bool is_sqrt)
{
  if constexpr (Metric == cuvs::distance::DistanceType::L2Expanded ||
                Metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    return is_sqrt;
  } else {
    return false;
  }
}

template <cuvs::distance::DistanceType Metric>
using fused_1nn_metric_tag_t = typename fused_1nn_metric_tag<Metric>::type;

template <typename DataT>
struct fused_1nn_data_tag;

template <>
struct fused_1nn_data_tag<float> {
  using type = cuvs::neighbors::detail::tag_f;
};

template <>
struct fused_1nn_data_tag<half> {
  using type = cuvs::neighbors::detail::tag_h;
};

template <typename DataT>
using fused_1nn_data_tag_t = typename fused_1nn_data_tag<DataT>::type;

template <typename IdxT>
struct fused_1nn_index_tag;

template <>
struct fused_1nn_index_tag<int32_t> {
  using type = cuvs::neighbors::detail::tag_index_i32;
};

template <>
struct fused_1nn_index_tag<int64_t> {
  using type = cuvs::neighbors::detail::tag_index_i64;
};

template <typename IdxT>
using fused_1nn_index_tag_t = typename fused_1nn_index_tag<IdxT>::type;

template <typename DataTag,
          typename MetricTag,
          typename IndexTag,
          typename TileTag,
          typename AbiTag,
          typename ArchTag>
struct fragment_tag_fused_1nn_cubin {
  static constexpr int cc_major = ArchTag::cc_major;
  static constexpr int cc_minor = ArchTag::cc_minor;
};

template <typename DataTag,
          typename MetricTag,
          typename IndexTag,
          typename TileTag,
          typename AbiTag>
struct fragment_tag_fused_1nn_tileir {};

}  // namespace cuvs::distance::detail
