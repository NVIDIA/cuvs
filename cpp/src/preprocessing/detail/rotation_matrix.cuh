/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/qr.cuh>
#include <raft/linalg/map.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstdint>

namespace cuvs::preprocessing::detail {

/**
 * Fill `rotation_matrix` with an `n_rows x n_cols` random orthogonal matrix (the identity when
 * the matrix is square and `force_random_rotation` is false), drawn from `rng`. Shared by IVF-PQ
 * and the RaBitQ quantizer.
 *
 * The matrix is the Q factor of a Gaussian matrix, read back row-major, so it is meant to be
 * applied as `y = R x`. QR does not normalize the signs of its triangular factor's diagonal, so the
 * matrix itself is not exactly Haar-distributed; applied as `y = R x` that only flips the signs of
 * output coordinates.
 */
inline void make_rotation_matrix(raft::resources const& handle,
                                 bool force_random_rotation,
                                 uint32_t n_rows,
                                 uint32_t n_cols,
                                 float* rotation_matrix,
                                 raft::random::RngState rng)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "preprocessing::make_rotation_matrix(%u * %u)", n_rows, n_cols);
  auto stream  = raft::resource::get_cuda_stream(handle);
  bool inplace = n_rows == n_cols;
  uint32_t n   = std::max(n_rows, n_cols);
  if (force_random_rotation || !inplace) {
    rmm::device_uvector<float> buf(inplace ? 0 : n * n, stream);
    float* mat = inplace ? rotation_matrix : buf.data();
    raft::random::normal(handle, rng, mat, n * n, 0.0f, 1.0f);
    raft::linalg::detail::qrGetQ_inplace(handle, mat, n, n, stream);
    if (!inplace) { raft::copy_matrix(rotation_matrix, n_cols, mat, n, n_cols, n_rows, stream); }
  } else {
    uint32_t stride = n + 1;
    auto rotation_matrix_view =
      raft::make_device_vector_view<float, uint32_t>(rotation_matrix, n * n);
    raft::linalg::map_offset(handle, rotation_matrix_view, [stride] __device__(uint32_t i) {
      return static_cast<float>(i % stride == 0u);
    });
  }
}

}  // namespace cuvs::preprocessing::detail
