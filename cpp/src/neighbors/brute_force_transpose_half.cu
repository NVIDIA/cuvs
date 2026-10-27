/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <neighbors/detail/knn_brute_force.cuh>

namespace cuvs::neighbors::detail {

void transpose_brute_force_half(raft::resources const& handle,
                                __half* in,
                                __half* out,
                                int n_rows,
                                int n_cols,
                                cudaStream_t stream)
{
  raft::linalg::transpose(handle, in, out, n_rows, n_cols, stream);
}

}  // namespace cuvs::neighbors::detail
