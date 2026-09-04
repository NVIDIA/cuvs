/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace cuvs::neighbors::cagra::detail {

inline uint64_t external_checked_add(uint64_t lhs, uint64_t rhs, const char* what)
{
  RAFT_EXPECTS(rhs <= std::numeric_limits<uint64_t>::max() - lhs, "overflow computing %s", what);
  return lhs + rhs;
}

inline uint64_t external_checked_mul(uint64_t lhs, uint64_t rhs, const char* what)
{
  RAFT_EXPECTS(
    lhs == 0 || rhs <= std::numeric_limits<uint64_t>::max() / lhs, "overflow computing %s", what);
  return lhs * rhs;
}

inline uint64_t external_div_rounding_up(uint64_t value, uint64_t divisor)
{
  RAFT_EXPECTS(divisor != 0, "division by zero in external HNSW plan");
  return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

inline uint64_t external_maximum_partitions(uint64_t rows, uint64_t intermediate_degree)
{
  RAFT_EXPECTS(rows > intermediate_degree,
               "external HNSW dataset is too small for one valid CAGRA partition");
  constexpr uint64_t minimum_core_rows = 1000;
  uint64_t graph_limit =
    external_checked_mul(2, rows, "maximum external partition rows") /
    external_checked_add(intermediate_degree, 1, "minimum external partition occurrence count");
  uint64_t core_limit = rows / minimum_core_rows;
  return std::max<uint64_t>(2, std::min({rows, graph_limit, core_limit}));
}

struct external_row_range {
  uint64_t start = 0;
  uint64_t count = 0;
};

inline std::vector<external_row_range> make_external_sample_ranges(uint64_t rows,
                                                                   uint64_t sample_rows,
                                                                   uint64_t max_stripes = 16)
{
  RAFT_EXPECTS(sample_rows > 0 && sample_rows <= rows && max_stripes > 0,
               "invalid external HNSW sample range request");
  uint64_t stripe_count = std::min(max_stripes, sample_rows);
  if (stripe_count == 1) { return {{(rows - sample_rows) / 2, sample_rows}}; }
  uint64_t gap_count = stripe_count - 1;
  uint64_t gap_rows  = rows - sample_rows;
  std::vector<external_row_range> ranges;
  ranges.reserve(static_cast<size_t>(stripe_count));
  uint64_t cursor = 0;
  for (uint64_t stripe = 0; stripe < stripe_count; ++stripe) {
    uint64_t count =
      sample_rows / stripe_count + static_cast<uint64_t>(stripe < sample_rows % stripe_count);
    ranges.push_back({cursor, count});
    cursor = external_checked_add(cursor, count, "external sample range");
    if (stripe + 1 < stripe_count) {
      uint64_t gap = gap_rows / gap_count + static_cast<uint64_t>(stripe < gap_rows % gap_count);
      cursor       = external_checked_add(cursor, gap, "external sample gap");
    }
  }
  RAFT_EXPECTS(cursor == rows, "external HNSW sample ranges do not cover the requested span");
  return ranges;
}

inline std::vector<external_row_range> make_external_monotonic_ranges(uint64_t rows,
                                                                      uint64_t chunk_rows)
{
  RAFT_EXPECTS(rows > 0 && chunk_rows > 0, "invalid external HNSW monotonic range request");
  std::vector<external_row_range> ranges;
  ranges.reserve(static_cast<size_t>(external_div_rounding_up(rows, chunk_rows)));
  for (uint64_t start = 0; start < rows;) {
    uint64_t count = std::min(chunk_rows, rows - start);
    ranges.push_back({start, count});
    start = external_checked_add(start, count, "external monotonic range");
  }
  return ranges;
}

struct ace_external_plan_input {
  uint64_t rows                    = 0;
  uint64_t dim                     = 0;
  uint64_t element_size            = 0;
  uint64_t index_size              = sizeof(uint32_t);
  uint64_t M                       = 0;
  uint64_t intermediate_degree     = 0;
  uint64_t graph_degree            = 0;
  uint64_t requested_partitions    = 0;
  uint64_t available_host_bytes    = 0;
  uint64_t available_device_bytes  = 0;
  uint64_t optimize_host_fixed     = 0;
  uint64_t optimize_device_fixed   = 0;
  uint64_t optimize_host_per_row   = 0;
  uint64_t optimize_device_per_row = 0;
  bool force_disk                  = false;
  bool hierarchy                   = true;
  uint32_t requested_queue_depth   = 2;
};

struct ace_external_byte_ledger {
  uint64_t source_scan                = 0;
  uint64_t centroid_sample            = 0;
  uint64_t stage_write                = 0;
  uint64_t stage_read                 = 0;
  uint64_t base_output                = 0;
  uint64_t upper_sidecar_write        = 0;
  uint64_t upper_sidecar_read         = 0;
  uint64_t final_upper_output         = 0;
  uint64_t expected_upper_occurrences = 0;

  [[nodiscard]] uint64_t logical_total() const
  {
    uint64_t total = external_checked_add(source_scan, centroid_sample, "source and sample bytes");
    total          = external_checked_add(total, stage_write, "logical byte total");
    total          = external_checked_add(total, stage_read, "logical byte total");
    total          = external_checked_add(total, base_output, "logical byte total");
    total          = external_checked_add(total, upper_sidecar_write, "logical byte total");
    total          = external_checked_add(total, upper_sidecar_read, "logical byte total");
    return external_checked_add(total, final_upper_output, "logical byte total");
  }
};

struct ace_external_plan {
  bool use_disk                         = false;
  uint64_t partitions                   = 0;
  uint64_t target_occurrences           = 0;
  uint64_t max_occurrences              = 0;
  uint64_t assignment_chunk_rows        = 0;
  uint64_t centroid_sample_rows         = 0;
  uint64_t staging_buffer_bytes         = 0;
  uint64_t preferred_buffer_bytes       = 0;
  uint64_t hnsw_output_buffer_bytes     = 0;
  uint64_t global_upper_level_max_rows  = 0;
  uint32_t queue_depth                  = 1;
  uint64_t host_budget_bytes            = 0;
  uint64_t device_budget_bytes          = 0;
  uint64_t host_peak_bytes              = 0;
  uint64_t device_peak_bytes            = 0;
  uint64_t centroid_host_peak_bytes     = 0;
  uint64_t centroid_device_peak_bytes   = 0;
  uint64_t assignment_host_peak_bytes   = 0;
  uint64_t assignment_device_peak_bytes = 0;
  uint64_t partition_host_peak_bytes    = 0;
  uint64_t partition_device_peak_bytes  = 0;
  uint64_t hierarchy_host_peak_bytes    = 0;
  uint64_t hierarchy_device_peak_bytes  = 0;
  uint64_t host_fixed_bytes             = 0;
  uint64_t device_fixed_bytes           = 0;
  uint64_t host_per_occurrence          = 0;
  uint64_t host_reader_per_occurrence   = 0;
  uint64_t device_per_occurrence        = 0;
  ace_external_byte_ledger bytes;
};

inline uint64_t estimate_materialized_cagra_ace_host_bytes(const ace_external_plan_input& in,
                                                           uint64_t partitions)
{
  RAFT_EXPECTS(partitions > 0, "CAGRA-ACE host estimate requires at least one partition");
  const uint64_t vector_bytes =
    external_checked_mul(in.dim, in.element_size, "CAGRA-ACE vector row");
  const uint64_t mapping_bytes = external_checked_mul(
    in.rows,
    external_checked_mul(4, in.index_size, "CAGRA-ACE labels and mappings row"),
    "CAGRA-ACE labels and mappings");
  const uint64_t full_graph_bytes = external_checked_mul(
    in.rows,
    external_checked_mul(in.graph_degree, in.index_size, "CAGRA-ACE final graph row"),
    "CAGRA-ACE final graph");
  const uint64_t max_occurrences = external_checked_mul(
    6, external_div_rounding_up(in.rows, partitions), "CAGRA-ACE skewed partition rows");
  const uint64_t partition_graph_bytes = external_checked_mul(
    external_checked_add(in.intermediate_degree, in.graph_degree, "CAGRA-ACE combined degree"),
    in.index_size,
    "CAGRA-ACE partition graph row");
  const uint64_t partition_bytes = external_checked_mul(
    max_occurrences,
    external_checked_add(
      external_checked_add(vector_bytes, partition_graph_bytes, "CAGRA-ACE partition row"),
      in.optimize_host_per_row,
      "CAGRA-ACE partition row with optimization"),
    "CAGRA-ACE maximum partition");
  uint64_t total =
    external_checked_add(mapping_bytes, full_graph_bytes, "CAGRA-ACE materialized graph");
  total = external_checked_add(total, in.optimize_host_fixed, "CAGRA-ACE fixed workspace");
  return external_checked_add(total, partition_bytes, "CAGRA-ACE materialized host bytes");
}

inline ace_external_byte_ledger make_external_byte_ledger(const ace_external_plan_input& in,
                                                          uint64_t centroid_sample_rows)
{
  const uint64_t vector_bytes  = external_checked_mul(in.dim, in.element_size, "vector byte size");
  const uint64_t dataset_bytes = external_checked_mul(in.rows, vector_bytes, "dataset byte size");
  const uint64_t sample_bytes =
    external_checked_mul(centroid_sample_rows, vector_bytes, "centroid sample bytes");
  const uint64_t stage_record_bytes =
    external_checked_add(vector_bytes, 2 * sizeof(uint32_t), "stage record byte size");
  const uint64_t stage_one_direction = external_checked_mul(
    external_checked_mul(in.rows, 2, "stage occurrence count"), stage_record_bytes, "stage bytes");
  const uint64_t base_row_bytes = external_checked_add(
    external_checked_add(
      sizeof(uint32_t),
      external_checked_mul(in.graph_degree, in.index_size, "base neighbor bytes"),
      "base links"),
    external_checked_add(vector_bytes, sizeof(size_t), "base vector and label"),
    "base row bytes");
  const uint64_t base_output =
    external_checked_mul(in.rows, base_row_bytes, "base HNSW output bytes");

  const uint64_t expected_upper_occurrences = in.hierarchy && in.M > 1 ? in.rows / (in.M - 1) : 0;
  const uint64_t sidecar_record_bytes =
    external_checked_add(sizeof(uint32_t),
                         external_checked_mul(in.M, in.index_size, "upper sidecar links"),
                         "upper sidecar record");
  const uint64_t sidecar_one_direction =
    external_checked_mul(expected_upper_occurrences, sidecar_record_bytes, "upper sidecar bytes");
  const uint64_t upper_node_headers =
    external_checked_mul(in.rows, sizeof(uint32_t), "upper node headers");
  const uint64_t upper_block_bytes = external_checked_add(
    sizeof(uint32_t), external_checked_mul(in.M, in.index_size, "upper links"), "upper block");
  const uint64_t upper_links =
    external_checked_mul(expected_upper_occurrences, upper_block_bytes, "upper output links");

  return {dataset_bytes,
          sample_bytes,
          stage_one_direction,
          stage_one_direction,
          base_output,
          sidecar_one_direction,
          sidecar_one_direction,
          external_checked_add(upper_node_headers, upper_links, "final upper output"),
          expected_upper_occurrences};
}

inline ace_external_plan make_ace_external_plan(const ace_external_plan_input& in)
{
  RAFT_EXPECTS(in.rows > 0 && in.dim > 0 && in.element_size > 0,
               "external HNSW plan requires a non-empty dataset");
  RAFT_EXPECTS(in.rows <= std::numeric_limits<uint32_t>::max(),
               "external HNSW build supports at most UINT32_MAX rows");
  RAFT_EXPECTS(in.requested_partitions <= in.rows,
               "ACE: number of partitions cannot exceed dataset size");
  RAFT_EXPECTS(in.dim <= std::numeric_limits<uint32_t>::max(),
               "external HNSW build supports at most UINT32_MAX dimensions");
  RAFT_EXPECTS(in.M >= 2 && in.graph_degree > 0 && in.intermediate_degree >= in.graph_degree,
               "invalid graph degrees in external HNSW plan");

  constexpr uint64_t one_mib = uint64_t{1} << 20;
  constexpr uint64_t one_gib = uint64_t{1} << 30;
  const uint64_t host_limit =
    in.available_host_bytes == 0 ? std::numeric_limits<uint64_t>::max() : in.available_host_bytes;
  const uint64_t device_limit  = in.available_device_bytes == 0
                                   ? std::numeric_limits<uint64_t>::max()
                                   : in.available_device_bytes;
  const uint64_t host_budget   = host_limit - host_limit / 5;
  const uint64_t device_budget = device_limit - device_limit / 5;
  const uint64_t vector_bytes  = external_checked_mul(in.dim, in.element_size, "vector byte size");

  ace_external_plan out;
  out.host_budget_bytes       = host_budget;
  out.device_budget_bytes     = device_budget;
  uint64_t maximum_partitions = external_maximum_partitions(in.rows, in.intermediate_degree);
  out.partitions              = std::max<uint64_t>(2, in.requested_partitions);
  out.partitions              = std::min(out.partitions, maximum_partitions);

  // Cap ACE's 1% centroid sample at 8 GiB and 25% of each memory budget.
  uint64_t sample_cap = 8 * one_gib;
  if (host_budget != std::numeric_limits<uint64_t>::max()) {
    sample_cap = std::min(sample_cap, host_budget / 4);
  }
  if (device_budget != std::numeric_limits<uint64_t>::max()) {
    sample_cap = std::min(sample_cap, device_budget / 4);
  }
  // Core plus spill, with 3x imbalance: six occurrences per row / partition count.
  out.host_per_occurrence = external_checked_add(
    external_checked_add(
      vector_bytes,
      external_checked_mul(in.graph_degree, in.index_size, "poststage graph row"),
      "resident partition row"),
    external_checked_add(
      in.optimize_host_per_row,
      external_checked_add(external_checked_mul(2, in.index_size, "resident partition mappings"),
                           in.hierarchy ? sizeof(uint8_t) : 0,
                           "resident partition mappings and hierarchy level"),
      "host optimization and resident partition metadata row"),
    "host bytes per partition occurrence");
  out.host_reader_per_occurrence = external_checked_add(
    vector_bytes,
    external_checked_mul(2, in.index_size, "reader mapping and core label row"),
    "host bytes per prefetched partition occurrence");
  out.device_per_occurrence = external_checked_add(
    external_checked_add(
      vector_bytes,
      external_checked_mul(
        external_checked_add(in.intermediate_degree, in.graph_degree, "combined graph degree"),
        in.index_size,
        "device graph rows"),
      "device partition row"),
    external_checked_add(
      in.index_size, in.optimize_device_per_row, "device mapping and optimize row"),
    "device bytes per partition occurrence");

  auto update_peaks = [&] {
    uint64_t requested_sample = std::max<uint64_t>(
      external_checked_mul(100, out.partitions, "minimum centroid samples"), in.rows / 100);
    uint64_t sample_rows = std::min(requested_sample, in.rows);
    sample_rows = std::min(sample_rows, std::max<uint64_t>(1, sample_cap / sizeof(float) / in.dim));
    RAFT_EXPECTS(sample_rows >= out.partitions,
                 "host/device cap is too small for one centroid sample per partition");
    out.centroid_sample_rows = sample_rows;
    out.host_fixed_bytes     = in.optimize_host_fixed;
    out.device_fixed_bytes   = in.optimize_device_fixed;

    out.target_occurrences = external_div_rounding_up(
      external_checked_mul(2, in.rows, "target partition occurrences"), out.partitions);
    out.max_occurrences = external_div_rounding_up(
      external_checked_mul(in.rows, 6, "skewed partition rows"), out.partitions);
    out.host_peak_bytes = external_checked_add(
      out.host_fixed_bytes,
      external_checked_mul(
        out.max_occurrences, out.host_per_occurrence, "external host partition peak"),
      "external host peak");
    out.device_peak_bytes = external_checked_add(
      out.device_fixed_bytes,
      external_checked_mul(
        out.max_occurrences, out.device_per_occurrence, "external device partition peak"),
      "external device peak");
  };
  update_peaks();

  while ((out.host_peak_bytes > host_budget || out.device_peak_bytes > device_budget) &&
         out.partitions < maximum_partitions) {
    uint64_t next =
      std::min<uint64_t>(maximum_partitions,
                         std::max(external_checked_add(out.partitions, 1, "next partition count"),
                                  external_checked_mul(out.partitions, 2, "next partition count")));
    out.partitions = next;
    update_peaks();
  }
  RAFT_EXPECTS(out.host_peak_bytes <= host_budget,
               "external HNSW host cap is below the minimum planned partition peak");
  RAFT_EXPECTS(out.device_peak_bytes <= device_budget,
               "external HNSW device cap is below the minimum planned partition peak");

  uint64_t host_headroom = host_budget - out.host_peak_bytes;
  RAFT_EXPECTS(host_headroom >= 2 * one_mib,
               "external HNSW host cap leaves less than 2 MiB for staging and output buffers");
  const uint64_t second_partition_host_bytes = external_checked_mul(
    out.max_occurrences, out.host_reader_per_occurrence, "prefetched partition host peak");
  out.queue_depth = 1;
  if (in.requested_queue_depth > 1 && second_partition_host_bytes <= host_headroom - 2 * one_mib) {
    out.queue_depth = 2;
    host_headroom -= second_partition_host_bytes;
  }
  constexpr uint64_t minimum_buffer = uint64_t{64} << 10;
  uint64_t minimum_stage_buffer     = std::max<uint64_t>(
    one_mib, external_checked_add(vector_bytes, 2 * sizeof(uint32_t), "minimum stage buffer"));
  uint64_t minimum_preferred_buffer = std::max<uint64_t>(
    minimum_buffer,
    external_checked_add(vector_bytes, 2 * sizeof(uint32_t), "minimum preferred buffer"));
  uint64_t minimum_output_buffer = std::max<uint64_t>(
    minimum_buffer,
    external_checked_mul(in.graph_degree, in.index_size, "minimum graph output buffer"));
  RAFT_EXPECTS(minimum_stage_buffer <= host_budget,
               "external HNSW host cap is too small for one stage record");
  out.staging_buffer_bytes =
    std::max(minimum_stage_buffer, std::min<uint64_t>(host_budget / 4, 256 * one_mib));
  out.preferred_buffer_bytes =
    std::max(minimum_preferred_buffer, std::min<uint64_t>(host_headroom / 8, 64 * one_mib));
  uint64_t concurrent_reader_buffer = out.queue_depth > 1 ? out.preferred_buffer_bytes : 0;
  uint64_t two_output_buffers =
    external_checked_mul(2, minimum_output_buffer, "minimum output buffers");
  RAFT_EXPECTS(two_output_buffers <= host_headroom &&
                 concurrent_reader_buffer <= host_headroom - two_output_buffers,
               "external HNSW host cap is too small for bounded I/O buffers");
  out.hnsw_output_buffer_bytes =
    std::max(minimum_output_buffer,
             std::min<uint64_t>((host_headroom - concurrent_reader_buffer) / 2, 64 * one_mib));

  uint64_t centroid_bytes = external_checked_mul(
    out.partitions, external_checked_mul(in.dim, sizeof(float), "centroid row"), "centroid bytes");
  uint64_t assignment_host_row =
    external_checked_add(external_checked_mul(in.dim, sizeof(float), "assignment host input row"),
                         2 * sizeof(uint32_t),
                         "assignment host row");
  uint64_t assignment_device_row = external_checked_add(
    external_checked_mul(in.dim, sizeof(float), "assignment device input row"),
    external_checked_add(
      external_checked_mul(out.partitions, sizeof(float), "assignment distance row"),
      2 * sizeof(float) + 2 * sizeof(uint32_t),
      "assignment device output row"),
    "assignment device row");
  RAFT_EXPECTS(out.staging_buffer_bytes < host_budget && centroid_bytes < device_budget,
               "external HNSW memory cap is too small for assignment buffers");
  uint64_t assignment_host_rows   = (host_budget - out.staging_buffer_bytes) / assignment_host_row;
  uint64_t assignment_device_rows = (device_budget - centroid_bytes) / assignment_device_row;
  out.assignment_chunk_rows =
    std::min<uint64_t>({uint64_t{32} * 1024, assignment_host_rows, assignment_device_rows});
  RAFT_EXPECTS(out.assignment_chunk_rows > 0,
               "external HNSW memory cap cannot hold one assignment row");

  uint64_t sample_float_bytes =
    external_checked_mul(out.centroid_sample_rows,
                         external_checked_mul(in.dim, sizeof(float), "centroid sample row"),
                         "centroid sample bytes");
  out.centroid_host_peak_bytes = sample_float_bytes;
  out.centroid_device_peak_bytes =
    external_checked_add(sample_float_bytes, centroid_bytes, "centroid training device peak");
  out.assignment_host_peak_bytes = external_checked_add(
    out.staging_buffer_bytes,
    external_checked_mul(out.assignment_chunk_rows, assignment_host_row, "assignment host chunk"),
    "assignment host peak");
  out.assignment_device_peak_bytes = external_checked_add(
    centroid_bytes,
    external_checked_mul(
      out.assignment_chunk_rows, assignment_device_row, "assignment device chunk"),
    "assignment device peak");
  out.partition_host_peak_bytes = external_checked_add(
    out.host_peak_bytes,
    external_checked_add(
      out.queue_depth > 1 ? second_partition_host_bytes : 0,
      external_checked_add(
        concurrent_reader_buffer,
        external_checked_mul(2, out.hnsw_output_buffer_bytes, "external output and graph buffers"),
        "external partition buffers"),
      "external queued partition buffers"),
    "external partition host peak");
  out.partition_device_peak_bytes = external_checked_add(
    out.device_peak_bytes, out.hnsw_output_buffer_bytes, "external translated graph chunk");

  out.global_upper_level_max_rows = std::max<uint64_t>(
    1,
    std::min(host_budget / 4 / std::max<uint64_t>(1, out.host_per_occurrence),
             device_budget / 4 / std::max<uint64_t>(1, out.device_per_occurrence)));
  out.hierarchy_host_peak_bytes = external_checked_add(
    external_checked_mul(
      out.global_upper_level_max_rows, out.host_per_occurrence, "external hierarchy host rows"),
    external_checked_add(
      out.preferred_buffer_bytes, out.hnsw_output_buffer_bytes, "external hierarchy host buffers"),
    "external hierarchy host peak");
  out.hierarchy_device_peak_bytes = external_checked_mul(
    out.global_upper_level_max_rows, out.device_per_occurrence, "external hierarchy device peak");
  out.host_peak_bytes   = std::max({out.centroid_host_peak_bytes,
                                    out.assignment_host_peak_bytes,
                                    out.partition_host_peak_bytes,
                                    out.hierarchy_host_peak_bytes});
  out.device_peak_bytes = std::max({out.centroid_device_peak_bytes,
                                    out.assignment_device_peak_bytes,
                                    out.partition_device_peak_bytes,
                                    out.hierarchy_device_peak_bytes});
  RAFT_EXPECTS(out.host_peak_bytes <= host_budget && out.device_peak_bytes <= device_budget,
               "external HNSW phase buffers exceed the configured memory cap");

  out.bytes = make_external_byte_ledger(in, out.centroid_sample_rows);

  out.use_disk = in.force_disk;
  return out;
}

}  // namespace cuvs::neighbors::cagra::detail
