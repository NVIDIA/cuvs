/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/3/25.
//

#pragma once

#include "../defines.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace cuvs::neighbors::ivf_rabitq::detail {

class InitializerGPU {
 public:
  /**
   * @brief Constructor.
   *
   * @param d dimension of vectors
   * @param k number of centroids
   */
  explicit InitializerGPU(raft::resources const& handle, size_t d, size_t k)
    : D(d), K(k), handle_(handle)
  {
    RAFT_EXPECTS(d > 0, "InitializerGPU: d must be > 0");
    RAFT_EXPECTS(k > 0, "InitializerGPU: k must be > 0");
  }

  virtual ~InitializerGPU() = default;

  /**
   * @brief Get a getCentroidbyId's vector data. Use ID as index.
   *
   * @param id Index.
   * @return
   */
  virtual __host__ __device__ float* GetCentroid(PID id) const = 0;

  /**
   * @brief Copies centroids from cent into device memory (K * D floats).
   *
   * The copy is issued asynchronously on the internal CUDA stream. The caller must keep cent
   * valid until that stream has been synchronized (or until any subsequent work on the stream
   * consuming the centroids has been submitted).
   *
   * @param cent pointer to centroids (host or device)
   */
  virtual void AddVectors(const float* cent) = 0;

  /**
   * @brief LoadCentroids centroids' information from files.
   *
   * @param input
   * @param filename
   */
  virtual void LoadCentroids(std::ifstream& input, const char* filename) = 0;

  /**
   * @brief SaveCentroids centroids' information from files.
   *
   * @param save
   * @param filename
   */
  virtual void SaveCentroids(std::ofstream& output, const char* filename) const = 0;

 protected:
  size_t D;                        // Dimension
  size_t K;                        // Num of Centroids
  raft::resources const& handle_;  // reusable resource handle
  rmm::cuda_stream_view stream_ =
    raft::resource::get_cuda_stream(handle_);  // CUDA stream obtained from handle_
};

class FlatInitializerGPU : public InitializerGPU {
 public:
  explicit FlatInitializerGPU(raft::resources const& handle, size_t d, size_t k);

  [[nodiscard]] __host__ __device__ float* GetCentroid(PID id) const override;

  void AddVectors(const float* cent) override;

  void LoadCentroids(std::ifstream& input, const char* filename) override;

  void SaveCentroids(std::ofstream& output, const char* filename) const override;

 private:
  // D, K are inherited from parent

  raft::device_matrix<float, int64_t, raft::row_major>
    centroids_;  // Stored in GPU device memory. Points to the parent centroids' array

  [[nodiscard]] size_t data_bytes() const noexcept { return sizeof(float) * K * D; }
  [[nodiscard]] size_t data_elements() const noexcept { return K * D; }
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
