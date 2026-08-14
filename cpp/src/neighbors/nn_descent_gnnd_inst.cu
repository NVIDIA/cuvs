/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/nn_descent.hpp>
#include <neighbors/detail/nn_descent_gnnd.hpp>
#include <neighbors/detail/reachability.cuh>
#include <neighbors/nn_descent.cuh>

namespace {

using data_t  = float;
using index_t = uint32_t;

}  // namespace

namespace cuvs::neighbors::nn_descent {

template class CUVS_EXPORT detail::GNND<const data_t, int>;

template void detail::GNND<const data_t, int>::build<
  cuvs::neighbors::detail::reachability::ReachabilityPostProcess<int, data_t>>(
  const data_t* data,
  const int nrow,
  int* output_graph,
  bool return_distances,
  float* output_distances,
  cuvs::neighbors::detail::reachability::ReachabilityPostProcess<int, data_t> dist_epilogue);
template void detail::GNND<const data_t, int>::local_join<
  cuvs::neighbors::detail::reachability::ReachabilityPostProcess<int, data_t>>(
  cudaStream_t stream,
  cuvs::neighbors::detail::reachability::ReachabilityPostProcess<int, data_t> dist_epilogue);

template void detail::GNND<const data_t, int>::build<raft::identity_op>(
  const data_t* data,
  const int nrow,
  int* output_graph,
  bool return_distances,
  float* output_distances,
  raft::identity_op dist_epilogue);
template void detail::GNND<const data_t, int>::local_join<raft::identity_op>(
  cudaStream_t stream, raft::identity_op dist_epilogue);

}  // namespace cuvs::neighbors::nn_descent
