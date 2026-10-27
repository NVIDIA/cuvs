/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "nn_descent.cuh"
#include "reachability.cuh"
#include <raft/linalg/map.cuh>

extern template void raft::linalg::detail::map<false, int, long, raft::const_op<int>>(
  cuda::stream_ref, int*, long, raft::const_op<int>);

extern template void raft::linalg::detail::map<false, float, size_t, raft::const_op<float>>(
  cuda::stream_ref, float*, size_t, raft::const_op<float>);

namespace cuvs::neighbors::nn_descent::detail {

}

namespace cuvs::neighbors::nn_descent::detail {

template class GnndGraph<int>;

}  // namespace cuvs::neighbors::nn_descent::detail
