/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/24/25.
//

#pragma once

#include "../defines.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/util/file_io.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

namespace cuvs::neighbors::ivf_rabitq::detail {

// The RotatorGPU class holds a rotation matrix (P) on the GPU. The matrix is computed
// on the CPU (using Eigen, similar to your CPU code) and then copied to device memory.
// The rotate() function uses cuBLAS to compute the product: RAND_A = A * P.
// It is assumed that A and RAND_A reside in GPU memory.
class RotatorGPU {
 public: /**
          * @brief Construct a new RotatorGPU object.
          * @param dim The original dimension; the padded dimension D is computed as
          * rd_up_to_multiple_of(dim, 64).
          *
          * The constructor generates a random rotation matrix on the CPU (using Eigen) and then
          * copies it into                    device memory in column-major order.
          */
  explicit RotatorGPU(raft::resources const& handle, uint32_t dim);

  // Disable copy assignment
  RotatorGPU& operator=(const RotatorGPU& other) = delete;

  size_t size() const;

  /**
   * @brief Load the rotation matrix from a file.
   * @param input KvikIO reader positioned at the row-major matrix.
   *
   * The function reads the D×D matrix directly into device memory.
   */
  void load(raft::resources const& handle, cuvs::util::kvikio_file_reader& input);

  /**
   * @brief Save the rotation matrix to a file.
   * @param handle Resource handle
   * @param output KvikIO output stream.
   *
   * The function writes the row-major rotation matrix directly from device memory.
   */
  void save(raft::resources const& handle, cuvs::util::kvikio_ofstream& output) const;

  // Rotate matrix A and store the result in RAND_A.
  // A and RAND_A are device pointers representing matrices of size N x D.
  // This function computes: RAND_A = A * P using cuBLAS.
  void rotate(raft::resources const& handle, const float* d_A, float* d_RAND_A, size_t N) const;

 private:
  size_t D;  // Padded dimension
  raft::device_matrix<float, int64_t, raft::row_major> rotation_matrix_;
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
