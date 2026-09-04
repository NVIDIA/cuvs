/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// On-disk layout of the layered HNSW artifact.

#include <cstddef>
#include <cstdint>

namespace cuvs::neighbors::hnsw::detail {

enum class layered_hnsw_dtype : uint32_t {
  unknown = 0,
  float32 = 1,
  float16 = 2,
  uint8   = 3,
  int8    = 4,
};

// The layered HNSW artifact begins with this fixed-size POD header, immediately followed by
// `num_layers` layered_hnsw_layer_descriptor records and then the payload sections.
struct layered_hnsw_file_header {
  char magic[32];
  uint64_t n_rows;
  uint64_t dim;
  uint64_t M;
  uint64_t maxM;
  uint64_t maxM0;
  uint64_t ef_construction;
  uint64_t base_degree;
  uint64_t levels_bytes;
  uint64_t base_nodes_bytes;
  uint64_t base_link_row_bytes;
  uint64_t base_links_bytes;
  uint64_t upper_nodes_count;
  uint64_t upper_nodes_bytes;
  uint64_t upper_link_row_bytes;
  uint64_t upper_links_bytes;
  double mult;
  uint32_t version;
  // Element type used to construct the graph. Attached vectors may use a different type.
  uint32_t construction_dtype;
  uint32_t metric;
  uint32_t num_layers;
  int32_t maxlevel;
  int32_t enterpoint_node;
  uint32_t reserved0;
  uint32_t reserved1;
};
static_assert(sizeof(layered_hnsw_file_header) == 192,
              "layered_hnsw_file_header must keep a fixed 192-byte on-disk layout");
inline constexpr size_t layered_hnsw_file_header_version_offset =
  offsetof(layered_hnsw_file_header, version);
inline constexpr size_t layered_hnsw_file_header_construction_dtype_offset =
  offsetof(layered_hnsw_file_header, construction_dtype);

struct layered_hnsw_layer_descriptor {
  uint64_t level;
  uint64_t row_count;
  uint64_t degree;
  uint64_t node_offset;
  uint64_t link_offset;
};
static_assert(sizeof(layered_hnsw_layer_descriptor) == 40,
              "layered_hnsw_layer_descriptor must keep a fixed 40-byte on-disk layout");

constexpr const char* layered_hnsw_magic = "CUVS_HNSW_LAYERED";
constexpr uint32_t layered_hnsw_version  = 1;
constexpr size_t layered_hnsw_alignment  = 64;

}  // namespace cuvs::neighbors::hnsw::detail
