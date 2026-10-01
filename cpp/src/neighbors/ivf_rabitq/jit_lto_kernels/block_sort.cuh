/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../ivf_common.cuh"

#include <raft/matrix/detail/select_warpsort.cuh>

namespace cuvs::neighbors::ivf_rabitq::detail {

// IVF-RaBitQ block-level top-k sort wrapper. Mirrors the pattern used by
// ivf_pq's pq_block_sort and ivf_sq's sq_block_sort: hide the warp_sort
// variant and Ascending policy that this algorithm doesn't vary, while still
// letting consumers pick the value/index types. Capacity == 0 falls back to
// the shared dummy_block_sort_t.
template <int Capacity, typename T, typename IdxT>
struct rabitq_block_sort {
  using type = raft::matrix::detail::select::warpsort::block_sort<
    raft::matrix::detail::select::warpsort::warp_sort_filtered,
    Capacity,
    /*Ascending=*/true,
    T,
    IdxT>;
};

template <typename T, typename IdxT>
struct rabitq_block_sort<0, T, IdxT>
  : ivf::detail::dummy_block_sort_t<T, IdxT, /*Ascending=*/true> {
  using type = ivf::detail::dummy_block_sort_t<T, IdxT, /*Ascending=*/true>;
};

template <int Capacity, typename T, typename IdxT>
using block_sort_t = typename rabitq_block_sort<Capacity, T, IdxT>::type;

}  // namespace cuvs::neighbors::ivf_rabitq::detail
