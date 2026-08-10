/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/10/25.
//

#pragma once

#include "../defines.hpp"
#include "rotator_gpu.cuh"
#include <raft/util/integer_utils.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace cuvs::neighbors::ivf_rabitq::detail {

typedef uint32_t PID;

/*!
 * @brief GPU version of DataQuantizer for ExRaBitQ.
 *
 * This class quantizes data into two sets of codes.
 * Its interface closely follows the CPU version. Later, you would replace
 * stub functions with CUDA kernels and device‐side implementations.
 */
class DataQuantizerGPU {
 public:
  // Nested struct for fast quantization factors
  struct FastQuantizeFactors {
    float const_scaling_factor_4bit;
    float const_scaling_factor_8bit;
  };

  // Static methods
  static float get_const_scaling_factors(raft::resources const& handle, size_t dim, size_t ex_bits);

  float get_const_scaling_factors_fully_gpu(size_t dim, size_t ex_bits);
  // Constructor: initialize from dimension and bit count.
  explicit DataQuantizerGPU(raft::resources const& handle, size_t dim, size_t b)
    : DIM(dim),
      D(raft::round_up_safe<size_t>(dim, 64)),
      EX_BITS(b),
      SHORT_CODE_LENGTH((D + 31) / 32),
      LONG_CODE_LENGTH(D * EX_BITS / 8)  // Simplified for now.
      ,
      FAC_NORM(1 / std::sqrt((double)D)),
      FAC_ERR(2.0 / std::sqrt((double)(D - 1))),
      fast_quantize_flag(false),
      const_scaling_factor(0.0f),
      handle_(handle)
  {
  }

  // Disable copy assignment
  DataQuantizerGPU& operator=(const DataQuantizerGPU& other) = delete;

  // Accessor functions.
  size_t short_code_length() const { return SHORT_CODE_LENGTH; }
  size_t long_code_length() const { return LONG_CODE_LENGTH; }
  // Block size for SoA layout (factors stored separately)
  size_t block_bytes() const
  {
    // 3 factors for batch: f_add, f_rescale and f_error are stored separately
    return SHORT_CODE_LENGTH * sizeof(uint32_t);
  }
  static constexpr size_t num_short_factors() { return NUM_SHORT_FACTORS; }
  const FastQuantizeFactors* get_query_scaling_factor() const { return &fast_quantize_factors; }
  FastQuantizeFactors* get_query_scaling_factor() { return &fast_quantize_factors; }
  void compute_query_scaling_factors(size_t dim)
  {
    fast_quantize_factors.const_scaling_factor_4bit = get_const_scaling_factors(handle_, dim, 3);
    fast_quantize_factors.const_scaling_factor_8bit = get_const_scaling_factors_fully_gpu(dim, 7);
  }
  void compute_quantize_scaling_factors()
  {
    const_scaling_factor = get_const_scaling_factors_fully_gpu(D, EX_BITS);
  }
  void set_quantize_scaling_factors(float value) { const_scaling_factor = value; }

  // Public configuration flag
  bool fast_quantize_flag;

  // functions to malloc temp buffers for gpu
  void alloc_buffers(size_t num_points);

  /*!
   * @brief Quantize the input data for batch layout.
   *
   * @param data Pointer to the input data (host pointer or device pointer as needed).
   * @param centroid Pointer to the corresponding centroid.
   * @param pids Vector of vector IDs.
   * @param rotator A RotatorGPU instance.
   * @param outShort Output buffer for short codes.
   * @param outLong Output buffer for long codes.
   * @param outExFactor Output buffer for extra factors.
   * @param outTemp Temporary buffer.
   */
  void quantize_batch_opt(const float* d_data,
                          const float* d_centroid,
                          const PID* d_IDs,
                          size_t num_points,
                          const RotatorGPU& rotator,
                          uint32_t* d_short_data,
                          float* short_data_factors,
                          uint8_t* d_long_code,
                          float* d_ex_factor,
                          float* d_rotated_c);

  /*!
   * @brief Quantize contiguous cluster data for batch layout (without PID gathering).
   *
   * @param d_contiguous_data Pointer to contiguous cluster data on device.
   * @param d_centroid Pointer to the corresponding centroid on device.
   * @param num_points Number of points in the cluster.
   * @param rotator A RotatorGPU instance.
   * @param d_short_data Output buffer for short codes.
   * @param short_data_factors Output buffer for short code factors.
   * @param d_long_code Output buffer for long codes.
   * @param d_ex_factor Output buffer for extra factors.
   * @param d_rotated_c Output buffer for rotated centroid.
   */
  void quantize_batch_opt_contiguous(const float* d_contiguous_data,
                                     const float* d_centroid,
                                     size_t num_points,
                                     const RotatorGPU& rotator,
                                     uint32_t* d_short_data,
                                     float* short_data_factors,
                                     uint8_t* d_long_code,
                                     float* d_ex_factor,
                                     float* d_rotated_c);

 private:
  // Dimension and quantization parameters
  size_t DIM;                // Original data dimension.
  size_t D;                  // Padded dimension (multiple of 64).
  size_t EX_BITS;            // Number of bits for ExRaBitQ.
  size_t SHORT_CODE_LENGTH;  // Number of uint32_t to store 1-bit code for a vector.
  size_t LONG_CODE_LENGTH;   // Number of uint8_t to store EX_BITS code for a vector.
  double FAC_NORM;
  double FAC_ERR;
  float const_scaling_factor;
  FastQuantizeFactors fast_quantize_factors;
  static constexpr size_t NUM_SHORT_FACTORS = 1;

  // RAFT resources
  raft::resources const& handle_;  // reusable resource handle
  rmm::cuda_stream_view stream_ =
    raft::resource::get_cuda_stream(handle_);  // CUDA stream obtained from handle_

  // Device temporary buffers for quantization
  raft::device_vector<float, int64_t> d_XP_norm =
    raft::make_device_vector<float, int64_t>(handle_, 0);
  raft::device_vector<int, int64_t> d_bin_XP = raft::make_device_vector<int, int64_t>(handle_, 0);
  raft::device_vector<float, int64_t> d_XP   = raft::make_device_vector<float, int64_t>(handle_, 0);
  raft::device_vector<float, int64_t> d_X_and_C_pad =
    raft::make_device_vector<float, int64_t>(handle_, 0);
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
