/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>

#include <stdint.h>
#include <sys/stat.h>

#include <cassert>
#include <fstream>

namespace cuvs::neighbors::ivf_rabitq::detail {

size_t get_filesize(const char* filename)
{
  struct stat64 stat_buf;
  int rc = stat64(filename, &stat_buf);
  return rc == 0 ? stat_buf.st_size : -1;
}

bool file_exits(const char* filename)
{
  std::ifstream f(filename);
  if (!f.good()) {
    f.close();
    return false;
  }
  f.close();
  return true;
}

template <typename T, class M>
void load_vecs(const char* filename, M& Mat)
{
  RAFT_EXPECTS(file_exits(filename), "File %s not exists", filename);

  static_assert(std::is_same_v<T, typename M::element_type>, "T must match M::element_type");

  uint32_t tmp;
  size_t file_size = get_filesize(filename);
  std::ifstream input(filename, std::ios::binary);

  input.read((char*)&tmp, sizeof(uint32_t));

  size_t cols = tmp;
  size_t rows = file_size / (cols * sizeof(T) + sizeof(uint32_t));
  Mat         = raft::make_host_matrix<T, int64_t>(rows, cols);

  input.seekg(0, input.beg);

  for (size_t i = 0; i < rows; i++) {
    input.read((char*)&tmp, sizeof(uint32_t));
    input.read((char*)&Mat(i, 0), sizeof(T) * cols);
  }

  RAFT_LOG_DEBUG("File %s loaded", filename);
  RAFT_LOG_DEBUG("Rows %zu Cols %zu", rows, cols);
  input.close();
}

// load_vecs, but duplicate k times
template <typename T, class M>
void load_vecs_k(const char* filename, M& Mat, size_t k)
{
  static_assert(std::is_same_v<T, typename M::element_type>, "T must match M::element_type");
  RAFT_EXPECTS(file_exits(filename), "File %s not exists", filename);

  uint32_t tmp;
  size_t file_size = get_filesize(filename);
  std::ifstream input(filename, std::ios::binary);

  // read `cols` under assumption of file format
  input.read(reinterpret_cast<char*>(&tmp), sizeof(uint32_t));
  const size_t cols = tmp;

  //  calculate rows under assumption of file format
  const size_t rows = file_size / (cols * sizeof(T) + sizeof(uint32_t));

  // allocate final size: rows * k
  Mat = raft::make_host_matrix<T, int64_t>(rows * k, cols);

  // return to start of file and read line by line
  input.seekg(0, input.beg);

  for (size_t i = 0; i < rows; ++i) {
    // read header for current line
    input.read(reinterpret_cast<char*>(&tmp), sizeof(uint32_t));
    // read rest of line into i-th row of output
    T* base = &Mat(i, 0);
    input.read(reinterpret_cast<char*>(base), sizeof(T) * cols);

    // create (k-1) copies
    for (size_t t = 1; t < k; ++t) {
      std::memcpy(&Mat(i + t * rows, 0), base, sizeof(T) * cols);
    }
  }

  RAFT_LOG_DEBUG("File %s loaded", filename);
  RAFT_LOG_DEBUG("Rows %zu Cols %zu", rows * k, cols);
  input.close();
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
