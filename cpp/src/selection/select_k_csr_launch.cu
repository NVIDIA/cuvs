/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <core/cub_ptx_query.hpp>
#include <core/selection_launchers.hpp>

#include <raft/sparse/matrix/detail/select_k-inl.cuh>
#include <raft/sparse/matrix/select_k.cuh>

extern template void raft::linalg::detail::map<true, int64_t, int64_t, raft::mul_const_op<int64_t>>(
  cuda::stream_ref, int64_t*, int64_t, raft::mul_const_op<int64_t>);

extern template void raft::matrix::detail::segmented_sort_by_key<float, int64_t>(
  raft::resources const&, float*, int64_t*, size_t, size_t, int64_t const*, bool);

namespace raft::matrix::detail::select::warpsort {

extern template void calc_launch_parameter<warp_sort_filtered, float, int64_t>(
  raft::resources const&, size_t, size_t, int, int*, int*);
extern template void calc_launch_parameter<warp_sort_immediate, float, int64_t>(
  raft::resources const&, size_t, size_t, int, int*, int*);
extern template void calc_launch_parameter<warp_sort_distributed, float, int64_t>(
  raft::resources const&, size_t, size_t, int, int*, int*);
extern template void calc_launch_parameter<warp_sort_distributed_ext, float, int64_t>(
  raft::resources const&, size_t, size_t, int, int*, int*);

extern template void
select_k_<warp_sort_filtered, float, int64_t, raft::matrix::detail::select::csr_layout<int64_t>>(
  bool,
  int,
  int,
  float const*,
  int64_t const*,
  int64_t const*,
  size_t,
  size_t,
  int,
  float*,
  int64_t*,
  bool,
  cuda::stream_ref,
  rmm::device_async_resource_ref);
extern template void
select_k_<warp_sort_immediate, float, int64_t, raft::matrix::detail::select::csr_layout<int64_t>>(
  bool,
  int,
  int,
  float const*,
  int64_t const*,
  int64_t const*,
  size_t,
  size_t,
  int,
  float*,
  int64_t*,
  bool,
  cuda::stream_ref,
  rmm::device_async_resource_ref);
extern template void
select_k_<warp_sort_distributed, float, int64_t, raft::matrix::detail::select::csr_layout<int64_t>>(
  bool,
  int,
  int,
  float const*,
  int64_t const*,
  int64_t const*,
  size_t,
  size_t,
  int,
  float*,
  int64_t*,
  bool,
  cuda::stream_ref,
  rmm::device_async_resource_ref);
extern template void select_k_<warp_sort_distributed_ext,
                               float,
                               int64_t,
                               raft::matrix::detail::select::csr_layout<int64_t>>(
  bool,
  int,
  int,
  float const*,
  int64_t const*,
  int64_t const*,
  size_t,
  size_t,
  int,
  float*,
  int64_t*,
  bool,
  cuda::stream_ref,
  rmm::device_async_resource_ref);
}  // namespace raft::matrix::detail::select::warpsort

extern template void raft::matrix::detail::select::radix::
  select_k<float, int64_t, 8, 512, raft::matrix::detail::select::csr_layout<int64_t>>(
    raft::resources const&,
    float const*,
    int64_t const*,
    int,
    int64_t,
    int64_t,
    float*,
    int64_t*,
    bool,
    bool,
    int64_t const*);
extern template void raft::matrix::detail::select::radix::
  select_k<float, int64_t, 11, 512, raft::matrix::detail::select::csr_layout<int64_t>>(
    raft::resources const&,
    float const*,
    int64_t const*,
    int,
    int64_t,
    int64_t,
    float*,
    int64_t*,
    bool,
    bool,
    int64_t const*);

namespace cuvs::detail {

template <typename T, typename Index>
void select_k(raft::resources const& handle,
              raft::device_csr_matrix_view<const T, Index, Index, Index> values,
              std::optional<raft::device_vector_view<const Index, Index>> indices,
              raft::device_matrix_view<T, Index, raft::row_major> selected_values,
              raft::device_matrix_view<Index, Index, raft::row_major> selected_indices,
              bool select_min,
              bool sorted,
              raft::matrix::SelectAlgo algorithm)
{
  raft::sparse::matrix::select_k<T, Index>(
    handle, values, indices, selected_values, selected_indices, select_min, sorted, algorithm);
}

template void select_k<float, int64_t>(
  raft::resources const&,
  raft::device_csr_matrix_view<const float, int64_t, int64_t, int64_t>,
  std::optional<raft::device_vector_view<const int64_t, int64_t>>,
  raft::device_matrix_view<float, int64_t, raft::row_major>,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major>,
  bool,
  bool,
  raft::matrix::SelectAlgo);

}  // namespace cuvs::detail
