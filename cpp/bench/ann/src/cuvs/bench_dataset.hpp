/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Dataset input for the cluster-assignment and fit-tuning benchmarks.
 *
 * Datasets are big-ann-benchmarks style .fbin files: uint32 n_rows, uint32 dim, then row-major
 * float32. CUVS_BENCH_DATA_FBIN names the dataset (real, or synthetic from
 * scripts/generate_clustered_dataset.py) and CUVS_BENCH_CENTROIDS_FBIN names centroids trained on
 * it (see train_centroids_from_env in cuvs_cluster_assignment_bench.cu).
 */
#pragma once

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench_dataset {

struct FbinHeader {
  int64_t n_rows;
  int64_t dim;
};

inline const char* data_path() { return std::getenv("CUVS_BENCH_DATA_FBIN"); }

inline const char* centroids_path() { return std::getenv("CUVS_BENCH_CENTROIDS_FBIN"); }

inline FbinHeader read_header(const char* path)
{
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) { throw std::runtime_error(std::string("cannot open ") + path); }
  uint32_t hdr[2];
  const bool ok = std::fread(hdr, sizeof(uint32_t), 2, f) == 2;
  std::fclose(f);
  if (!ok) { throw std::runtime_error(std::string("cannot read fbin header of ") + path); }
  return {hdr[0], hdr[1]};
}

// The first n_rows rows of the file, on the host. Checks the file has them and has this dim.
inline std::vector<float> read_first_rows(const char* path, int64_t n_rows, int64_t dim)
{
  const auto hdr = read_header(path);
  if (hdr.dim != dim || hdr.n_rows < n_rows) {
    throw std::runtime_error(std::string(path) + " has fewer rows than requested or another dim");
  }
  std::vector<float> rows(static_cast<size_t>(n_rows * dim));
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) { throw std::runtime_error(std::string("cannot open ") + path); }
  const bool ok = std::fseek(f, 8, SEEK_SET) == 0 &&
                  std::fread(rows.data(), sizeof(float), rows.size(), f) == rows.size();
  std::fclose(f);
  if (!ok) { throw std::runtime_error(std::string("short read from ") + path); }
  return rows;
}

// Copies the first n_rows points of CUVS_BENCH_DATA_FBIN to device memory.
inline void load_data(raft::resources const& handle, float* dev, int64_t n_rows, int64_t dim)
{
  const auto rows = read_first_rows(data_path(), n_rows, dim);
  raft::copy(dev, rows.data(), rows.size(), raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);
}

// Copies n_clusters centroids, taken at an even stride from CUVS_BENCH_CENTROIDS_FBIN, to device
// memory, so a smaller K still spans the whole centroid set.
inline void load_centroids(raft::resources const& handle,
                           float* dev,
                           int64_t n_clusters,
                           int64_t dim)
{
  const int64_t n_total = read_header(centroids_path()).n_rows;
  if (n_total < n_clusters) { throw std::runtime_error("fewer centroids in file than requested"); }
  const auto all = read_first_rows(centroids_path(), n_total, dim);
  std::vector<float> picked(static_cast<size_t>(n_clusters * dim));
  for (int64_t c = 0; c < n_clusters; ++c) {
    const int64_t src = c * n_total / n_clusters;
    std::copy(all.begin() + src * dim, all.begin() + (src + 1) * dim, picked.begin() + c * dim);
  }
  raft::copy(dev, picked.data(), picked.size(), raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);
}

inline void write_fbin(const char* path, const float* host, int64_t n_rows, int64_t dim)
{
  std::FILE* f = std::fopen(path, "wb");
  if (f == nullptr) { throw std::runtime_error(std::string("cannot open ") + path); }
  const uint32_t hdr[2] = {static_cast<uint32_t>(n_rows), static_cast<uint32_t>(dim)};
  std::fwrite(hdr, sizeof(uint32_t), 2, f);
  std::fwrite(host, sizeof(float), static_cast<size_t>(n_rows * dim), f);
  std::fclose(f);
}

}  // namespace bench_dataset
