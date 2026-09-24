/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "external_format.hpp"

#include <cuvs/util/file_io.hpp>

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>
#include <kvikio/file_handle.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace cuvs::neighbors::hnsw::detail::external {

constexpr size_t hnswlib_header_size = 96;
constexpr uint64_t hnsw_level_seed   = 100;

inline uint64_t splitmix64(uint64_t value)
{
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

inline uint32_t level_for_internal_id(uint32_t id, uint64_t seed, size_t M)
{
  if (M < 2) { throw std::invalid_argument("HNSW M must be at least two"); }
  constexpr uint64_t mantissa_mask = (UINT64_C(1) << 53) - 1;
  constexpr uint64_t denominator   = (UINT64_C(1) << 53) + 1;
  const uint64_t sample =
    ((splitmix64(static_cast<uint64_t>(id) ^ seed) >> 11) & mantissa_mask) + 1;
  uint64_t threshold = denominator / M;
  uint32_t level     = 0;
  while (sample <= threshold) {
    ++level;
    threshold /= M;
  }
  return level;
}

struct hierarchy_summary {
  uint32_t max_level   = 0;
  uint32_t entry_point = 0;
  std::vector<uint64_t> active_by_level;
  uint64_t total_active_occurrences = 0;
};

inline hierarchy_summary summarize_hierarchy(uint64_t rows,
                                             size_t M,
                                             uint64_t seed = hnsw_level_seed)
{
  if (rows == 0 || rows > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("invalid row count for HNSW hierarchy");
  }
  hierarchy_summary summary;
  for (uint64_t row = 0; row < rows; ++row) {
    auto level = level_for_internal_id(static_cast<uint32_t>(row), seed, M);
    if (level > summary.max_level ||
        (level == summary.max_level && static_cast<uint32_t>(row) > summary.entry_point)) {
      summary.max_level   = level;
      summary.entry_point = static_cast<uint32_t>(row);
    }
    if (summary.active_by_level.size() < level) { summary.active_by_level.resize(level, 0); }
    for (uint32_t current = 0; current < level; ++current) {
      ++summary.active_by_level[current];
      ++summary.total_active_occurrences;
    }
  }
  return summary;
}

struct hnsw_serialize_layout {
  size_t offset_level0         = 0;
  size_t rows                  = 0;
  size_t size_data_per_element = 0;
  size_t label_offset          = 0;
  size_t offset_data           = 0;
  int max_level                = 0;
  uint32_t entry_point         = 0;
  size_t maxM                  = 0;
  size_t maxM0                 = 0;
  size_t M                     = 0;
  double mult                  = 0;
  size_t ef_construction       = 0;
  size_t dimension             = 0;
  size_t element_size          = 0;
  size_t graph_degree          = 0;
  size_t upper_block_size      = 0;

  [[nodiscard]] uint64_t base_section_size() const
  {
    return checked_file_mul(rows, size_data_per_element, "HNSW base section");
  }

  [[nodiscard]] uint64_t exact_file_size(uint64_t upper_occurrences) const
  {
    uint64_t size =
      checked_file_add(hnswlib_header_size, base_section_size(), "HNSW header and base section");
    size = checked_file_add(
      size, checked_file_mul(rows, sizeof(uint32_t), "HNSW upper node headers"), "HNSW file");
    return checked_file_add(
      size,
      checked_file_mul(upper_occurrences, upper_block_size, "HNSW upper link blocks"),
      "HNSW exact file size");
  }
};

template <typename T, typename DistT>
hnsw_serialize_layout make_hnsw_serialize_layout(hnswlib::SpaceInterface<DistT>* space,
                                                 uint64_t rows,
                                                 size_t dimension,
                                                 size_t graph_degree,
                                                 size_t ef_construction,
                                                 bool hierarchy,
                                                 const hierarchy_summary& summary)
{
  if (rows == 0 || rows > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("invalid HNSW row count");
  }
  auto algorithm = std::make_unique<hnswlib::HierarchicalNSW<DistT>>(
    space, 1, (graph_degree + 1) / 2, ef_construction);
  hnsw_serialize_layout layout;
  layout.offset_level0         = algorithm->offsetLevel0_;
  layout.rows                  = static_cast<size_t>(rows);
  layout.size_data_per_element = algorithm->size_data_per_element_;
  layout.label_offset          = algorithm->label_offset_;
  layout.offset_data           = algorithm->offsetData_;
  layout.max_level             = hierarchy ? static_cast<int>(summary.max_level) : 1;
  layout.entry_point           = hierarchy ? summary.entry_point : static_cast<uint32_t>(rows / 2);
  layout.maxM                  = algorithm->maxM_;
  layout.maxM0                 = algorithm->maxM0_;
  layout.M                     = algorithm->M_;
  layout.mult                  = algorithm->mult_;
  layout.ef_construction       = algorithm->ef_construction_;
  layout.dimension             = dimension;
  layout.element_size          = sizeof(T);
  layout.graph_degree          = graph_degree;
  layout.upper_block_size      = algorithm->size_links_per_element_;

  const size_t expected_base =
    sizeof(uint32_t) + layout.maxM0 * sizeof(uint32_t) + dimension * sizeof(T) + sizeof(size_t);
  RAFT_EXPECTS(layout.size_data_per_element == expected_base, "Unexpected hnswlib base row layout");
  RAFT_EXPECTS(layout.maxM0 >= graph_degree && layout.maxM0 - graph_degree <= 1,
               "Unexpected hnswlib base degree");
  RAFT_EXPECTS(layout.upper_block_size == sizeof(uint32_t) + layout.M * sizeof(uint32_t),
               "Unexpected hnswlib upper row layout");
  return layout;
}

template <typename T, typename DistT>
hnsw_serialize_layout hnsw_serialize_layout_from_algorithm(
  const hnswlib::HierarchicalNSW<DistT>& algorithm,
  uint64_t rows,
  size_t dimension,
  size_t graph_degree)
{
  hnsw_serialize_layout layout;
  layout.offset_level0         = algorithm.offsetLevel0_;
  layout.rows                  = static_cast<size_t>(rows);
  layout.size_data_per_element = algorithm.size_data_per_element_;
  layout.label_offset          = algorithm.label_offset_;
  layout.offset_data           = algorithm.offsetData_;
  layout.max_level             = algorithm.maxlevel_;
  layout.entry_point           = algorithm.enterpoint_node_;
  layout.maxM                  = algorithm.maxM_;
  layout.maxM0                 = algorithm.maxM0_;
  layout.M                     = algorithm.M_;
  layout.mult                  = algorithm.mult_;
  layout.ef_construction       = algorithm.ef_construction_;
  layout.dimension             = dimension;
  layout.element_size          = sizeof(T);
  layout.graph_degree          = graph_degree;
  layout.upper_block_size      = algorithm.size_links_per_element_;
  return layout;
}

class sequential_file_writer {
 public:
  sequential_file_writer(const cuvs::util::file_descriptor& fd,
                         uint64_t start_offset,
                         size_t buffer_size)
    : fd_(fd), file_offset_(start_offset), buffer_(std::max<size_t>(buffer_size, 1))
  {
    const auto path = fd_.get_path();
    if (!path.empty()) { file_handle_ = std::make_unique<kvikio::FileHandle>(path, "r+"); }
  }

  sequential_file_writer(const sequential_file_writer&)            = delete;
  sequential_file_writer& operator=(const sequential_file_writer&) = delete;

  ~sequential_file_writer() noexcept
  {
    try {
      flush();
    } catch (...) {
    }
  }

  void write(const void* source, size_t bytes)
  {
    auto* source_bytes = static_cast<const std::byte*>(source);
    while (bytes != 0) {
      if (used_ == buffer_.size()) { flush(); }
      size_t chunk = std::min(bytes, buffer_.size() - used_);
      std::memcpy(buffer_.data() + used_, source_bytes, chunk);
      used_ += chunk;
      source_bytes += chunk;
      bytes -= chunk;
    }
  }

  void flush()
  {
    if (used_ == 0) { return; }
    if (file_handle_) {
      const size_t written = file_handle_->pwrite(buffer_.data(), used_, file_offset_).get();
      RAFT_EXPECTS(
        written == used_, "Incomplete HNSW write: expected %zu bytes, wrote %zu", used_, written);
    } else {
      cuvs::util::write_large_file(fd_, buffer_.data(), used_, file_offset_);
    }
    file_offset_ = checked_file_add(file_offset_, used_, "sequential HNSW output offset");
    used_        = 0;
    ++requests_;
  }

  [[nodiscard]] uint64_t position() const { return file_offset_ + used_; }
  [[nodiscard]] uint64_t request_count() const noexcept { return requests_; }

 private:
  const cuvs::util::file_descriptor& fd_;
  uint64_t file_offset_;
  std::vector<std::byte> buffer_;
  std::unique_ptr<kvikio::FileHandle> file_handle_;
  size_t used_       = 0;
  uint64_t requests_ = 0;
};

template <typename Writer>
void write_hnsw_header_fields(Writer& writer, const hnsw_serialize_layout& layout)
{
  writer.write(reinterpret_cast<const char*>(&layout.offset_level0), sizeof(layout.offset_level0));
  writer.write(reinterpret_cast<const char*>(&layout.rows), sizeof(layout.rows));
  writer.write(reinterpret_cast<const char*>(&layout.rows), sizeof(layout.rows));
  writer.write(reinterpret_cast<const char*>(&layout.size_data_per_element),
               sizeof(layout.size_data_per_element));
  writer.write(reinterpret_cast<const char*>(&layout.label_offset), sizeof(layout.label_offset));
  writer.write(reinterpret_cast<const char*>(&layout.offset_data), sizeof(layout.offset_data));
  writer.write(reinterpret_cast<const char*>(&layout.max_level), sizeof(layout.max_level));
  writer.write(reinterpret_cast<const char*>(&layout.entry_point), sizeof(layout.entry_point));
  writer.write(reinterpret_cast<const char*>(&layout.maxM), sizeof(layout.maxM));
  writer.write(reinterpret_cast<const char*>(&layout.maxM0), sizeof(layout.maxM0));
  writer.write(reinterpret_cast<const char*>(&layout.M), sizeof(layout.M));
  writer.write(reinterpret_cast<const char*>(&layout.mult), sizeof(layout.mult));
  writer.write(reinterpret_cast<const char*>(&layout.ef_construction),
               sizeof(layout.ef_construction));
}

inline void write_hnsw_header(sequential_file_writer& writer, const hnsw_serialize_layout& layout)
{
  write_hnsw_header_fields(writer, layout);
  RAFT_EXPECTS(writer.position() == hnswlib_header_size,
               "HNSW header size mismatch: expected %zu, got %zu",
               hnswlib_header_size,
               static_cast<size_t>(writer.position()));
}

template <typename T, typename Writer>
void write_hnsw_base_row(Writer& writer,
                         const hnsw_serialize_layout& layout,
                         const uint32_t* neighbors,
                         const T* vector,
                         uint32_t external_label)
{
  uint32_t count = static_cast<uint32_t>(layout.graph_degree);
  writer.write(reinterpret_cast<const char*>(&count), sizeof(count));
  writer.write(reinterpret_cast<const char*>(neighbors), layout.graph_degree * sizeof(uint32_t));
  uint32_t zero = 0;
  for (size_t index = layout.graph_degree; index < layout.maxM0; ++index) {
    writer.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
  }
  writer.write(reinterpret_cast<const char*>(vector), layout.dimension * sizeof(T));
  size_t label = external_label;
  writer.write(reinterpret_cast<const char*>(&label), sizeof(label));
}

template <typename Writer>
void write_hnsw_upper_node_header(Writer& writer,
                                  const hnsw_serialize_layout& layout,
                                  uint32_t level)
{
  uint64_t bytes = checked_file_mul(level, layout.upper_block_size, "HNSW node upper links");
  if (bytes > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("HNSW node upper link list exceeds uint32_t");
  }
  uint32_t link_list_size = static_cast<uint32_t>(bytes);
  writer.write(reinterpret_cast<const char*>(&link_list_size), sizeof(link_list_size));
}

template <typename Writer>
void write_hnsw_upper_block(Writer& writer,
                            const hnsw_serialize_layout& layout,
                            const uint32_t* neighbors,
                            uint32_t count)
{
  RAFT_EXPECTS(count <= layout.M, "HNSW upper neighbor count exceeds M");
  writer.write(reinterpret_cast<const char*>(&count), sizeof(count));
  writer.write(reinterpret_cast<const char*>(neighbors), count * sizeof(uint32_t));
  uint32_t zero = 0;
  for (size_t index = count; index < layout.M; ++index) {
    writer.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
  }
}

}  // namespace cuvs::neighbors::hnsw::detail::external
