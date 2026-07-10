/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/3/25.
//

#include "initializer_gpu.cuh"

#include <raft/core/error.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

FlatInitializerGPU::FlatInitializerGPU(raft::resources const& handle, size_t d, size_t k)
  : InitializerGPU(handle, d, k),
    centroids_(raft::make_device_matrix<float, int64_t, raft::row_major>(handle_, K, D))
{
}

__host__ __device__ float* FlatInitializerGPU::GetCentroid(PID id) const
{
  return const_cast<float*>(centroids_.data_handle()) + id * D;
}

void FlatInitializerGPU::AddVectors(const float* cent)
{
  RAFT_EXPECTS(cent != nullptr, "FlatInitializerGPU::AddVectors: cent is null");
  raft::copy(centroids_.data_handle(), cent, data_elements(), stream_);
}

void FlatInitializerGPU::LoadCentroids(cuvs::util::kvikio_file_reader& input, const char*)
{
  raft::resource::sync_stream(handle_);
  input.read_device(centroids_.data_handle(), data_bytes());
}

void FlatInitializerGPU::SaveCentroids(cuvs::util::kvikio_ofstream& output,
                                       const char* filename) const
{
  raft::resource::sync_stream(handle_);
  output.write_device(centroids_.data_handle(), data_bytes());
  RAFT_EXPECTS(static_cast<bool>(output), "Failed to write centroids to file: %s", filename);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
