/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>

#include <array>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <type_traits>
#include <unistd.h>

namespace cuvs::neighbors::hnsw::detail::external {

constexpr uint32_t stage_schema_version = 1;
constexpr uint64_t stage_data_offset    = 4096;
constexpr uint64_t stage_magic          = UINT64_C(0x43555653484e5357);  // "CUVSHNSW"
constexpr uint32_t little_endian_marker = UINT32_C(0x01020304);

enum class stage_kind : uint32_t { core = 1, spill = 2, upper_vectors = 3, upper_links = 4 };
enum class element_kind : uint32_t { f32 = 1, f16 = 2, i8 = 3, u8 = 4 };

template <typename T>
constexpr element_kind element_kind_for()
{
  if constexpr (std::is_same_v<T, float>) {
    return element_kind::f32;
  } else if constexpr (sizeof(T) == 2 && !std::is_integral_v<T>) {
    return element_kind::f16;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return element_kind::i8;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return element_kind::u8;
  } else {
    static_assert(!sizeof(T), "unsupported external HNSW element type");
  }
}

struct stage_header {
  uint64_t magic                 = stage_magic;
  uint32_t version               = stage_schema_version;
  uint32_t endian                = little_endian_marker;
  uint32_t kind                  = 0;
  uint32_t element               = 0;
  uint32_t element_size          = 0;
  uint32_t dimension             = 0;
  uint32_t partition             = 0;
  uint32_t level                 = 0;
  uint64_t record_size           = 0;
  uint64_t committed_records     = 0;
  uint64_t data_offset           = stage_data_offset;
  uint64_t parameter_fingerprint = 0;
  std::array<std::byte, 56> reserved{};
};
static_assert(sizeof(stage_header) == 128);
static_assert(std::is_trivially_copyable_v<stage_header>);

inline uint64_t checked_file_add(uint64_t lhs, uint64_t rhs, std::string_view what)
{
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    throw std::overflow_error("overflow computing " + std::string{what});
  }
  return lhs + rhs;
}

inline uint64_t checked_file_mul(uint64_t lhs, uint64_t rhs, std::string_view what)
{
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    throw std::overflow_error("overflow computing " + std::string{what});
  }
  return lhs * rhs;
}

inline uint64_t expected_file_size(const stage_header& header)
{
  return checked_file_add(
    header.data_offset,
    checked_file_mul(header.committed_records, header.record_size, "stage payload size"),
    "stage file size");
}

inline void pwrite_all(int fd, const void* source, size_t bytes, uint64_t offset)
{
  auto* ptr   = static_cast<const std::byte*>(source);
  size_t done = 0;
  while (done < bytes) {
    uint64_t current = checked_file_add(offset, done, "external HNSW write offset");
    if (current > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
      throw std::overflow_error("external HNSW write offset exceeds off_t");
    }
    ssize_t written = ::pwrite(fd, ptr + done, bytes - done, static_cast<off_t>(current));
    if (written < 0) {
      if (errno == EINTR) { continue; }
      throw std::runtime_error("external HNSW pwrite failed: " + std::string{strerror(errno)});
    }
    if (written == 0) { throw std::runtime_error("external HNSW pwrite made no progress"); }
    done += static_cast<size_t>(written);
  }
}

inline void pread_all(int fd, void* destination, size_t bytes, uint64_t offset)
{
  auto* ptr   = static_cast<std::byte*>(destination);
  size_t done = 0;
  while (done < bytes) {
    uint64_t current = checked_file_add(offset, done, "external HNSW read offset");
    if (current > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
      throw std::overflow_error("external HNSW read offset exceeds off_t");
    }
    ssize_t read_bytes = ::pread(fd, ptr + done, bytes - done, static_cast<off_t>(current));
    if (read_bytes < 0) {
      if (errno == EINTR) { continue; }
      throw std::runtime_error("external HNSW pread failed: " + std::string{strerror(errno)});
    }
    if (read_bytes == 0) { throw std::runtime_error("truncated external HNSW file"); }
    done += static_cast<size_t>(read_bytes);
  }
}

inline void write_stage_header(int fd, const stage_header& header)
{
  static_assert(std::endian::native == std::endian::little,
                "external HNSW stage format currently requires a little-endian host");
  pwrite_all(fd, &header, sizeof(header), 0);
}

inline stage_header read_stage_header(int fd)
{
  stage_header header;
  pread_all(fd, &header, sizeof(header), 0);
  return header;
}

inline void validate_stage_header(const stage_header& header,
                                  stage_kind expected_kind,
                                  element_kind expected_element,
                                  uint32_t expected_element_size,
                                  uint32_t expected_dimension,
                                  uint32_t expected_partition,
                                  uint32_t expected_level,
                                  uint64_t expected_record_size,
                                  uint64_t expected_fingerprint)
{
  RAFT_EXPECTS(header.magic == stage_magic, "Invalid external HNSW stage magic");
  RAFT_EXPECTS(header.version == stage_schema_version,
               "Unsupported external HNSW stage version %u",
               header.version);
  RAFT_EXPECTS(header.endian == little_endian_marker,
               "External HNSW stage endianness does not match this host");
  RAFT_EXPECTS(header.kind == static_cast<uint32_t>(expected_kind),
               "External HNSW stage kind mismatch");
  RAFT_EXPECTS(header.element == static_cast<uint32_t>(expected_element) &&
                 header.element_size == expected_element_size,
               "External HNSW stage element type mismatch");
  RAFT_EXPECTS(header.dimension == expected_dimension, "External HNSW stage dimension mismatch");
  RAFT_EXPECTS(header.partition == expected_partition, "External HNSW stage partition mismatch");
  RAFT_EXPECTS(header.level == expected_level, "External HNSW stage level mismatch");
  RAFT_EXPECTS(header.record_size == expected_record_size,
               "External HNSW stage record size mismatch");
  RAFT_EXPECTS(header.data_offset == stage_data_offset, "External HNSW stage data offset mismatch");
  RAFT_EXPECTS(header.parameter_fingerprint == expected_fingerprint,
               "External HNSW stage parameter fingerprint mismatch");
}

inline void validate_stage_file_size(int fd, const stage_header& header)
{
  struct stat stat_buffer{};
  if (::fstat(fd, &stat_buffer) != 0) {
    throw std::runtime_error("external HNSW fstat failed: " + std::string{strerror(errno)});
  }
  RAFT_EXPECTS(stat_buffer.st_size >= 0, "Invalid external HNSW stage size");
  RAFT_EXPECTS(static_cast<uint64_t>(stat_buffer.st_size) == expected_file_size(header),
               "External HNSW stage size mismatch: expected %zu, got %zu",
               static_cast<size_t>(expected_file_size(header)),
               static_cast<size_t>(stat_buffer.st_size));
}

template <typename T>
stage_header make_stage_header(stage_kind kind,
                               uint32_t dimension,
                               uint32_t partition,
                               uint32_t level,
                               uint64_t record_size,
                               uint64_t parameter_fingerprint)
{
  stage_header header;
  header.kind                  = static_cast<uint32_t>(kind);
  header.element               = static_cast<uint32_t>(element_kind_for<T>());
  header.element_size          = sizeof(T);
  header.dimension             = dimension;
  header.partition             = partition;
  header.level                 = level;
  header.record_size           = record_size;
  header.parameter_fingerprint = parameter_fingerprint;
  return header;
}

inline uint64_t fnv1a64(const void* bytes, size_t size)
{
  constexpr uint64_t offset = UINT64_C(14695981039346656037);
  constexpr uint64_t prime  = UINT64_C(1099511628211);
  uint64_t value            = offset;
  auto* ptr                 = static_cast<const uint8_t*>(bytes);
  for (size_t i = 0; i < size; ++i) {
    value ^= ptr[i];
    value *= prime;
  }
  return value;
}

}  // namespace cuvs::neighbors::hnsw::detail::external
