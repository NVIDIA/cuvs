/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/24/25.
//

#include "rotator_gpu.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/detail/qr.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/integer_utils.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

RotatorGPU::RotatorGPU(raft::resources const& handle, uint32_t dim)
  : D(raft::round_up_safe<uint32_t>(dim, 64u)),
    rotation_matrix_(raft::make_device_matrix<float, int64_t, raft::row_major>(handle, D, D))
{
  raft::random::RngState rng(7ULL);
  raft::random::normal(handle, rng, rotation_matrix_.data_handle(), D * D, 0.0f, 1.0f);
  raft::linalg::detail::qrGetQ_inplace(
    handle, rotation_matrix_.data_handle(), D, D, raft::resource::get_cuda_stream(handle));
}

size_t RotatorGPU::size() const { return D; }

void RotatorGPU::load(raft::resources const& handle, cuvs::util::kvikio_file_reader& input)
{
  raft::resource::sync_stream(handle);
  input.read_device(rotation_matrix_.data_handle(), sizeof(float) * D * D);
}

void RotatorGPU::save(raft::resources const& handle, cuvs::util::kvikio_ofstream& output) const
{
  raft::resource::sync_stream(handle);
  output.write_device(rotation_matrix_.data_handle(), sizeof(float) * D * D);
  RAFT_EXPECTS(static_cast<bool>(output), "failed to write rotator matrix");
}

// Rotate the matrix A and store the result in RAND_A on the GPU.
// A and RAND_A are assumed to be stored in row-major order.
// A is of size N x D and P is of size D x D, so the result is N x D.
// This function uses cuBLAS to perform the matrix multiplication.
void RotatorGPU::rotate(raft::resources const& handle,
                        const float* d_A,
                        float* d_RAND_A,
                        size_t N) const
{
  // cuBLAS assumes column-major storage by default. Since our matrices are in row-major order,
  // we can perform the multiplication as:
  //   RAND_A^T = P^T * A^T
  // which is equivalent to RAND_A = A * P, if we interpret the data as row-major.
  // Note that in Cublas it is RAND_A^T in column major, which is what we want in row-major
  // Here, we use the RAFT wrapper for gemm.
  raft::linalg::gemm(
    handle,
    raft::make_device_matrix_view<float, int64_t, raft::col_major>(
      const_cast<float*>(rotation_matrix_.data_handle()), D, D),
    raft::make_device_matrix_view<float, int64_t, raft::col_major>(const_cast<float*>(d_A), D, N),
    raft::make_device_matrix_view<float, int64_t, raft::col_major>(d_RAND_A, D, N));
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
