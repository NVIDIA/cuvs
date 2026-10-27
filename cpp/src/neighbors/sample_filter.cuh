/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../core/bitmap_csr_conversion_launchers.hpp"
#include <cuvs/neighbors/common.hpp>
#include <raft/core/bitmap.cuh>
#include <raft/core/bitset.cuh>
#include <raft/core/detail/macros.hpp>
#include <raft/sparse/convert/csr.cuh>

#include <cuda/std/tuple>

#include <cstddef>
#include <cstdint>

namespace cuvs::neighbors::filtering {

/* A filter that filters nothing. This is the default behavior. */
constexpr __forceinline__ _RAFT_HOST_DEVICE bool none_sample_filter::operator()(
  // query index
  const uint32_t query_ix,
  // the current inverted list index
  const uint32_t cluster_ix,
  // the index of the current sample inside the current inverted list
  const uint32_t sample_ix) const
{
  return true;
}

/* A filter that filters nothing. This is the default behavior. */
constexpr __forceinline__ _RAFT_HOST_DEVICE bool none_sample_filter::operator()(
  // query index
  const uint32_t query_ix,
  // the index of the current sample
  const uint32_t sample_ix) const
{
  return true;
}

template <typename filter_t, typename = void>
struct takes_three_args : std::false_type {};
template <typename filter_t>
struct takes_three_args<
  filter_t,
  std::void_t<decltype(std::declval<filter_t>()(uint32_t{}, uint32_t{}, uint32_t{}))>>
  : std::true_type {};

/**
 * @brief Filter used to convert the cluster index and sample index
 * of an IVF search into a sample index. This can be used as an
 * intermediate filter.
 *
 * @tparam index_t Indexing type
 * @tparam filter_t
 */
template <typename index_t, typename filter_t>
_RAFT_HOST_DEVICE ivf_to_sample_filter<index_t, filter_t>::ivf_to_sample_filter(
  const index_t* const* inds_ptrs, const filter_t next_filter)
  : inds_ptrs_{inds_ptrs}, next_filter_{next_filter}
{
}

/** If the original filter takes three arguments, then don't modify the arguments.
 * If the original filter takes two arguments, then we are using `inds_ptr_` to obtain the sample
 * index.
 */
template <typename index_t, typename filter_t>
inline _RAFT_HOST_DEVICE bool ivf_to_sample_filter<index_t, filter_t>::operator()(
  // query index
  const uint32_t query_ix,
  // the current inverted list index
  const uint32_t cluster_ix,
  // the index of the current sample inside the current inverted list
  const uint32_t sample_ix) const
{
  if constexpr (takes_three_args<filter_t>::value) {
    return next_filter_(query_ix, cluster_ix, sample_ix);
  } else {
    return next_filter_(query_ix, inds_ptrs_[cluster_ix][sample_ix]);
  }
}

template <typename bitset_t, typename index_t>
_RAFT_HOST_DEVICE bitset_filter<bitset_t, index_t>::bitset_filter(
  const cuvs::core::bitset_view<bitset_t, index_t> bitset_for_filtering)
  : bitset_view_{bitset_for_filtering}
{
}

template <typename bitset_t, typename index_t>
constexpr __forceinline__ _RAFT_HOST_DEVICE bool bitset_filter<bitset_t, index_t>::operator()(
  // query index
  const uint32_t query_ix,
  // the index of the current sample
  const uint32_t sample_ix) const
{
  return bitset_view_.test(sample_ix);
}

template <typename bitset_t, typename index_t>
template <typename csr_matrix_t>
void bitset_filter<bitset_t, index_t>::to_csr(raft::resources const& handle, csr_matrix_t& csr)
{
  cuvs::detail::bitset_to_csr(handle, bitset_view_, csr);
}

template <typename bitmap_t, typename index_t>
bitmap_filter<bitmap_t, index_t>::bitmap_filter(
  const cuvs::core::bitmap_view<bitmap_t, index_t> bitmap_for_filtering)
  : bitmap_view_{bitmap_for_filtering}
{
}

template <typename bitmap_t, typename index_t>
inline _RAFT_HOST_DEVICE bool bitmap_filter<bitmap_t, index_t>::operator()(
  // query index
  const uint32_t query_ix,
  // the index of the current sample
  const uint32_t sample_ix) const
{
  return bitmap_view_.test(query_ix, sample_ix);
}

template <typename bitmap_t, typename index_t>
template <typename csr_matrix_t>
void bitmap_filter<bitmap_t, index_t>::to_csr(raft::resources const& handle, csr_matrix_t& csr)
{
  cuvs::detail::bitmap_to_csr(handle, bitmap_view_, csr);
}

struct none_filter_args_t {};
using bitset_filter_args_t =
  cuda::std::tuple<const int64_t* const*, cuvs::core::bitset_view<uint32_t, int64_t>>;

struct ivf_filter_dev {
  filtering::FilterType tag_;

  union ivf_filter_dev_args_variant {
    none_filter_args_t none_filter_args;
    bitset_filter_args_t bitset_filter_args;

    _RAFT_HOST_DEVICE explicit ivf_filter_dev_args_variant(const none_filter_args_t& args)
      : none_filter_args(args)
    {
    }

    _RAFT_HOST_DEVICE explicit ivf_filter_dev_args_variant(const bitset_filter_args_t& args)
      : bitset_filter_args(args)
    {
    }
  } args_;

  _RAFT_HOST_DEVICE ivf_filter_dev(none_filter_args_t args = {})
    : tag_(FilterType::None), args_(args) {};

  _RAFT_HOST_DEVICE ivf_filter_dev(bitset_filter_args_t args)
    : tag_(FilterType::Bitset), args_(args) {};

  constexpr __forceinline__ _RAFT_HOST_DEVICE bool operator()(const uint32_t query_ix,
                                                              const uint32_t cluster_ix,
                                                              const uint32_t sample_ix)
  {
    switch (tag_) {
      case FilterType::None:
        return filtering::none_sample_filter{}(query_ix, cluster_ix, sample_ix);
      case FilterType::Bitset: {
        auto& [inds_ptrs_, bitset_view_] = args_.bitset_filter_args;
        return filtering::bitset_filter<uint32_t, int64_t>(bitset_view_)(
          query_ix, inds_ptrs_[cluster_ix][sample_ix]);
      }
      default: return true;
    }
  }
};

}  // namespace cuvs::neighbors::filtering
