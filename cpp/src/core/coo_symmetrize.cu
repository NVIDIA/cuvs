/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "spectral_launchers.hpp"

#include <cuvs/core/cuda_fp16.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/device_atomics.cuh>

#include <raft/sparse/linalg/symmetrize.cuh>

namespace cuvs::detail {

template <typename T, typename Lambda>
void coo_symmetrize(raft::sparse::COO<T>* in,
                    raft::sparse::COO<T>* out,
                    Lambda reduction_op,  // two-argument reducer
                    cudaStream_t stream)

{
  return raft::sparse::linalg::coo_symmetrize(in, out, reduction_op, stream);
}

template <int TPB_X, typename T, typename IdxT, typename nnz_t, typename Lambda>
void coo_symmetrize(raft::resources const& handle,
                    raft::device_coo_matrix_view<const T, IdxT, IdxT, nnz_t> in,
                    raft::device_coo_matrix<T, IdxT, IdxT, nnz_t>& out,
                    Lambda reduction_op)  // two-argument reducer

{
  return raft::sparse::linalg::coo_symmetrize<TPB_X>(handle, in, out, reduction_op);
}

}  // namespace cuvs::detail
