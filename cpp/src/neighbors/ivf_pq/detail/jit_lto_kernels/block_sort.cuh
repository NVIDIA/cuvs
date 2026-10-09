/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../ivf_common.cuh"
#include <raft/matrix/detail/select_warpsort.cuh>

#include <type_traits>

namespace cuvs::neighbors::ivf_pq::detail {

template <int Capacity, typename T, typename IdxT, bool Stable = false>
struct pq_block_sort {
  using type =
    std::conditional_t<Stable,
                       raft::matrix::detail::select::warpsort::block_sort<
                         raft::matrix::detail::select::warpsort::warp_sort_distributed_ext_stable,
                         Capacity,
                         true,
                         T,
                         IdxT>,
                       raft::matrix::detail::select::warpsort::block_sort<
                         raft::matrix::detail::select::warpsort::warp_sort_distributed_ext,
                         Capacity,
                         true,
                         T,
                         IdxT>>;

  static auto get_mem_required(uint32_t k_max)
  {
    if (k_max == 0 || k_max > Capacity) {
      return pq_block_sort<0, T, IdxT, Stable>::get_mem_required(k_max);
    }
    if constexpr (Capacity > 1) {
      if (k_max * 2 <= Capacity) {
        return pq_block_sort<(Capacity / 2), T, IdxT, Stable>::get_mem_required(k_max);
      }
    }
    return type::queue_t::mem_required;
  }
};

template <typename T, typename IdxT, bool Stable>
struct pq_block_sort<0, T, IdxT, Stable> : ivf::detail::dummy_block_sort_t<T, IdxT> {
  using type = ivf::detail::dummy_block_sort_t<T, IdxT>;
  static auto mem_required(uint32_t) -> size_t { return 0; }
  static auto get_mem_required(uint32_t) { return mem_required; }
};

template <int Capacity, typename T, typename IdxT, bool Stable = false>
using block_sort_t = typename pq_block_sort<Capacity, T, IdxT, Stable>::type;

}  // namespace cuvs::neighbors::ivf_pq::detail
