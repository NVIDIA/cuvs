/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/util/file_io.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <string>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

// These buffers are always host memory. Use positional POSIX I/O directly to avoid
// reopening/registering the bucket with KvikIO for every buffered run.
template <bool Write>
void ace_disk_host_io(cuvs::util::file_descriptor const& fd,
                      std::conditional_t<Write, void const*, void*> data,
                      size_t bytes,
                      size_t offset)
{
  auto constexpr max_offset = static_cast<size_t>(std::numeric_limits<off_t>::max());
  RAFT_EXPECTS(offset <= max_offset && bytes <= max_offset - offset,
               "ACE: reverse-edge file offset overflow");
  auto* buffer = static_cast<std::conditional_t<Write, char const*, char*>>(data);
  while (bytes > 0) {
    auto const chunk = std::min(bytes, size_t{1} << 30);
    ssize_t transferred;
    if constexpr (Write) {
      transferred = ::pwrite(fd.get(), buffer, chunk, static_cast<off_t>(offset));
    } else {
      transferred = ::pread(fd.get(), buffer, chunk, static_cast<off_t>(offset));
    }
    if (transferred < 0 && errno == EINTR) { continue; }
    RAFT_EXPECTS(transferred >= 0,
                 "ACE: reverse-edge %s failed: %s",
                 Write ? "write" : "read",
                 std::strerror(errno));
    RAFT_EXPECTS(transferred > 0, "ACE: incomplete reverse-edge %s", Write ? "write" : "read");
    buffer += transferred;
    offset += transferred;
    bytes -= transferred;
  }
}

// Spool all edges while the pruned core rows are still in memory. A single
// bounded buffer is sorted into destination-partition runs before writing; only one bucket
// file is open at a time. Neither candidate sets nor a second full graph are kept in RAM.
template <typename IdxT>
class ace_disk_reverse_edges {
 public:
  struct record {
    IdxT destination;
    IdxT source;
    uint32_t rank;
  };

  struct incoming_edge {
    IdxT source;
    uint32_t rank;

    bool operator<(incoming_edge const& other) const
    {
      return rank < other.rank || (rank == other.rank && source < other.source);
    }
  };

  // The first floor(degree / 2) graph neighbors are protected during the merge.
  static uint32_t incoming_capacity(uint32_t degree) { return degree - degree / 2; }

  static size_t buffer_bytes(size_t available_host_memory)
  {
    return std::max(sizeof(record), std::min(size_t{8} << 20, available_host_memory / 100));
  }

  ace_disk_reverse_edges(std::string const& build_dir,
                         std::span<IdxT const> offsets,
                         uint32_t degree,
                         size_t buffer_size)
    : offsets_(raft::make_host_vector<IdxT, int64_t>(offsets.size())),
      degree_(degree),
      capacity_(std::max(size_t{1}, buffer_size / sizeof(record))),
      sizes_(raft::make_host_vector<size_t, int64_t>(offsets.empty() ? 0 : offsets.size() - 1)),
      buffer_(raft::make_host_vector<record, int64_t>(capacity_))
  {
    RAFT_EXPECTS(offsets.size() >= 2 && offsets.front() == 0 && degree_ > 0,
                 "ACE: invalid reverse-edge partition layout");
    for (size_t i = 0; i < offsets.size(); ++i) {
      RAFT_EXPECTS(i == 0 || offsets[i - 1] <= offsets[i],
                   "ACE: invalid reverse-edge partition layout");
      offsets_(i) = offsets[i];
      if (i + 1 < offsets.size()) { sizes_(i) = 0; }
    }
    directory_ = build_dir + "/reverse_edges_XXXXXX";
    RAFT_EXPECTS(::mkdtemp(directory_.data()) != nullptr,
                 "ACE: failed to create reverse-edge workspace: %s",
                 std::strerror(errno));
  }

  ace_disk_reverse_edges(ace_disk_reverse_edges const&)            = delete;
  ace_disk_reverse_edges& operator=(ace_disk_reverse_edges const&) = delete;

  ~ace_disk_reverse_edges()
  {
    // This uniquely created directory contains only this build's temporary records.
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    if (error) {
      RAFT_LOG_WARN("ACE: failed to remove reverse-edge workspace %s: %s",
                    directory_.c_str(),
                    error.message().c_str());
    }
  }

  // Rows and neighbors both use core-reordered IDs, as in the on-disk graph.
  void append_partition(size_t partition, std::span<IdxT const> graph)
  {
    RAFT_EXPECTS(partition < sizes_.size(), "ACE: invalid reverse-edge partition");
    auto const first = offsets_(partition);
    auto const last  = offsets_(partition + 1);
    RAFT_EXPECTS(graph.size() == size_t(last - first) * degree_,
                 "ACE: invalid pruned core graph size");
    for (size_t row = 0; row < size_t(last - first); ++row) {
      for (uint32_t rank = 0; rank < degree_; ++rank) {
        auto const destination = graph[row * degree_ + rank];
        RAFT_EXPECTS(
          static_cast<uint64_t>(destination) < static_cast<uint64_t>(offsets_(offsets_.size() - 1)),
          "ACE: graph contains an invalid neighbor");
        buffer_(buffer_count_++) = {destination, static_cast<IdxT>(first + row), rank};
        if (buffer_count_ == capacity_) { flush(); }
      }
    }
  }

  void merge(cuvs::util::file_descriptor const& graph_fd, size_t graph_header_size)
  {
    flush();
    auto const candidate_capacity = incoming_capacity(degree_);
    for (size_t partition = 0; partition < sizes_.size(); ++partition) {
      if (sizes_(partition) == 0) { continue; }
      auto const first = offsets_(partition);
      auto const rows  = size_t(offsets_(partition + 1) - first);

      // Keep only enough incoming candidates to fill the replaceable half of each row.
      // Buckets can be arbitrarily skewed: their size does not determine RAM usage.
      // Max-heaps preserve rank priority across buffer flushes and source-partition build order.
      // GPU rank ties are scheduling-dependent; the disk path breaks them by source ID.
      auto incoming = raft::make_host_vector<incoming_edge, int64_t>(rows * candidate_capacity);
      auto counts   = raft::make_host_vector<uint32_t, int64_t>(rows);
      for (size_t row = 0; row < rows; ++row) {
        counts(row) = 0;
      }
      cuvs::util::file_descriptor bucket(bucket_path(partition), O_RDONLY);
      for (size_t offset = 0; offset < sizes_(partition);) {
        auto const count = std::min(capacity_, sizes_(partition) - offset);
        ace_disk_host_io<false>(
          bucket, buffer_.data_handle(), count * sizeof(record), offset * sizeof(record));
        for (size_t i = 0; i < count; ++i) {
          auto const& edge = buffer_(i);
          RAFT_EXPECTS(edge.destination >= first && edge.destination < offsets_(partition + 1) &&
                         edge.rank < degree_ &&
                         static_cast<uint64_t>(edge.source) <
                           static_cast<uint64_t>(offsets_(offsets_.size() - 1)),
                       "ACE: invalid reverse-edge record");
          auto const row = size_t(edge.destination - first);
          auto* heap     = incoming.data_handle() + row * candidate_capacity;
          auto& n        = counts(row);
          incoming_edge candidate{edge.source, edge.rank};
          if (n < candidate_capacity) {
            auto index = n++;
            while (index > 0) {
              auto const parent = (index - 1) / 2;
              if (!(heap[parent] < candidate)) { break; }
              heap[index] = heap[parent];
              index       = parent;
            }
            heap[index] = candidate;
          } else if (candidate < heap[0]) {
            heap[0] = candidate;
            sift_down(heap, n);
          }
        }
        offset += count;
      }

      auto graph             = raft::make_host_vector<IdxT, int64_t>(rows * degree_);
      auto const file_offset = graph_header_size + size_t(first) * degree_ * sizeof(IdxT);
      ace_disk_host_io<false>(
        graph_fd, graph.data_handle(), graph.size() * sizeof(IdxT), file_offset);
      auto const protected_edges = degree_ / 2;
      for (size_t row = 0; row < rows; ++row) {
        auto* edges     = incoming.data_handle() + row * candidate_capacity;
        auto n          = counts(row);
        auto* neighbors = graph.data_handle() + row * degree_;
        // Match ace_merge_partition_reverse_edges: insert in reverse priority order,
        // protect the first half of the pruned row, and move rather than duplicate edges.
        while (n > 0) {
          auto const source = edges[0].source;
          edges[0]          = edges[--n];
          sift_down(edges, n);
          uint32_t existing = 0;
          while (existing < degree_ && neighbors[existing] != source) {
            ++existing;
          }
          if (existing < protected_edges) { continue; }
          auto const end = existing == degree_ ? degree_ - 1 : existing;
          for (auto rank = end; rank > protected_edges; --rank) {
            neighbors[rank] = neighbors[rank - 1];
          }
          neighbors[protected_edges] = source;
        }
      }
      ace_disk_host_io<true>(
        graph_fd, graph.data_handle(), graph.size() * sizeof(IdxT), file_offset);
      bucket.close();
      std::filesystem::remove(bucket_path(partition));
    }
    buffer_count_ = 0;
  }

 private:
  std::string bucket_path(size_t partition) const
  {
    return directory_ + "/" + std::to_string(partition);
  }

  void flush()
  {
    // Sort the fixed POD buffer without STL containers or algorithms.
    ::qsort(buffer_.data_handle(), buffer_count_, sizeof(record), [](void const* a, void const* b) {
      auto const lhs = static_cast<record const*>(a)->destination;
      auto const rhs = static_cast<record const*>(b)->destination;
      return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
    });
    size_t begin = 0;
    while (begin < buffer_count_) {
      // Upper-bound search also handles empty core partitions (repeated offsets).
      size_t low  = 0;
      size_t high = offsets_.size();
      while (low < high) {
        auto const middle = low + (high - low) / 2;
        if (offsets_(middle) <= buffer_(begin).destination) {
          low = middle + 1;
        } else {
          high = middle;
        }
      }
      auto const partition = low - 1;
      size_t end           = begin + 1;
      while (end < buffer_count_ && buffer_(end).destination < offsets_(partition + 1)) {
        ++end;
      }
      cuvs::util::file_descriptor bucket(
        bucket_path(partition), sizes_(partition) == 0 ? O_CREAT | O_EXCL | O_WRONLY : O_WRONLY);
      ace_disk_host_io<true>(bucket,
                             buffer_.data_handle() + begin,
                             (end - begin) * sizeof(record),
                             sizes_(partition) * sizeof(record));
      sizes_(partition) += end - begin;
      begin = end;
    }
    buffer_count_ = 0;
  }

  // Restore a max-heap after replacing its root. No allocation occurs per candidate.
  static void sift_down(incoming_edge* heap, uint32_t count)
  {
    if (count == 0) { return; }
    auto const value = heap[0];
    uint32_t index   = 0;
    while (index < count / 2) {
      auto child = index * 2 + 1;
      if (child + 1 < count && heap[child] < heap[child + 1]) { ++child; }
      if (!(value < heap[child])) { break; }
      heap[index] = heap[child];
      index       = child;
    }
    heap[index] = value;
  }

  raft::host_vector<IdxT, int64_t> offsets_;
  uint32_t degree_;
  size_t capacity_;
  raft::host_vector<size_t, int64_t> sizes_;
  raft::host_vector<record, int64_t> buffer_;
  size_t buffer_count_ = 0;
  std::string directory_;
};

}  // namespace cuvs::neighbors::cagra::detail
