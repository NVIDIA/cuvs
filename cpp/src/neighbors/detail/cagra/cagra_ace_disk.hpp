/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/util/file_io.hpp>
#include <raft/core/logger.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

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

// Spool cross-partition edges while the pruned core rows are still in memory. A single
// bounded buffer is sorted into destination-partition runs before writing; only one bucket
// file is open at a time. Neither candidate sets nor a second full graph are kept in RAM.
template <typename IdxT>
class ace_disk_reverse_edges {
 public:
  struct record {
    IdxT destination;
    IdxT source;
    uint32_t rank;

    // Avoid the generic device-only detail::swap overload pulled in by CAGRA's top-k code.
    friend void swap(record& a, record& b) noexcept
    {
      std::swap(a.destination, b.destination);
      std::swap(a.source, b.source);
      std::swap(a.rank, b.rank);
    }
  };

  struct incoming_edge {
    IdxT source;
    uint32_t rank;

    bool operator<(incoming_edge const& other) const
    {
      return rank < other.rank || (rank == other.rank && source < other.source);
    }
  };

  static size_t buffer_bytes(size_t available_host_memory)
  {
    return std::max(sizeof(record), std::min(size_t{8} << 20, available_host_memory / 100));
  }

  ace_disk_reverse_edges(std::string const& build_dir,
                         std::span<IdxT const> offsets,
                         uint32_t degree,
                         size_t buffer_size)
    : offsets_(offsets.begin(), offsets.end()),
      degree_(degree),
      capacity_(std::max(size_t{1}, buffer_size / sizeof(record)))
  {
    RAFT_EXPECTS(offsets_.size() >= 2 && offsets_.front() == 0 &&
                   std::is_sorted(offsets_.begin(), offsets_.end()) && degree_ > 0,
                 "ACE: invalid reverse-edge partition layout");
    sizes_.resize(offsets_.size() - 1);
    buffer_.reserve(capacity_);
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
    auto const first = offsets_.at(partition);
    auto const last  = offsets_.at(partition + 1);
    RAFT_EXPECTS(graph.size() == size_t(last - first) * degree_,
                 "ACE: invalid pruned core graph size");
    for (size_t row = 0; row < size_t(last - first); ++row) {
      for (uint32_t rank = 0; rank < degree_; ++rank) {
        auto const destination = graph[row * degree_ + rank];
        RAFT_EXPECTS(static_cast<uint64_t>(destination) < static_cast<uint64_t>(offsets_.back()),
                     "ACE: graph contains an invalid neighbor");
        if (destination >= first && destination < last) { continue; }
        buffer_.push_back({destination, static_cast<IdxT>(first + row), rank});
        if (buffer_.size() == capacity_) { flush(); }
      }
    }
  }

  void merge(cuvs::util::file_descriptor const& graph_fd, size_t graph_header_size)
  {
    flush();
    buffer_.resize(capacity_);
    for (size_t partition = 0; partition < sizes_.size(); ++partition) {
      if (sizes_[partition] == 0) { continue; }
      auto const first = offsets_[partition];
      auto const rows  = size_t(offsets_[partition + 1] - first);

      // Keep only the best degree incoming edges per row. Buckets can be arbitrarily
      // skewed: their size does not determine RAM usage. Max-heaps preserve rank priority
      // across buffer flushes and source-partition build order. GPU rank ties are
      // scheduling-dependent; the disk path breaks them deterministically by source ID.
      std::vector<incoming_edge> incoming(rows * degree_);
      std::vector<uint32_t> counts(rows, 0);
      cuvs::util::file_descriptor bucket(bucket_path(partition), O_RDONLY);
      for (size_t offset = 0; offset < sizes_[partition];) {
        auto const count = std::min(capacity_, sizes_[partition] - offset);
        ace_disk_host_io<false>(
          bucket, buffer_.data(), count * sizeof(record), offset * sizeof(record));
        for (size_t i = 0; i < count; ++i) {
          auto const& edge = buffer_[i];
          RAFT_EXPECTS(
            edge.destination >= first && edge.destination < offsets_[partition + 1] &&
              edge.rank < degree_ &&
              static_cast<uint64_t>(edge.source) < static_cast<uint64_t>(offsets_.back()),
            "ACE: invalid reverse-edge record");
          auto const row = size_t(edge.destination - first);
          auto* heap     = incoming.data() + row * degree_;
          auto& n        = counts[row];
          incoming_edge candidate{edge.source, edge.rank};
          if (n < degree_) {
            heap[n++] = candidate;
            std::push_heap(heap, heap + n);
          } else if (candidate < heap[0]) {
            std::pop_heap(heap, heap + n);
            heap[n - 1] = candidate;
            std::push_heap(heap, heap + n);
          }
        }
        offset += count;
      }

      std::vector<IdxT> graph(rows * degree_);
      auto const file_offset = graph_header_size + size_t(first) * degree_ * sizeof(IdxT);
      ace_disk_host_io<false>(graph_fd, graph.data(), graph.size() * sizeof(IdxT), file_offset);
      auto const protected_edges = degree_ / 2;
      for (size_t row = 0; row < rows; ++row) {
        auto* edges = incoming.data() + row * degree_;
        auto n      = counts[row];
        std::sort_heap(edges, edges + n);
        auto* neighbors = graph.data() + row * degree_;
        // Match ace_merge_partition_reverse_edges: insert in reverse priority order,
        // protect the first half of the pruned row, and move rather than duplicate edges.
        while (n > 0) {
          auto const source = edges[--n].source;
          auto* existing    = std::find(neighbors, neighbors + degree_, source);
          if (existing < neighbors + protected_edges) { continue; }
          auto* end = existing == neighbors + degree_ ? existing - 1 : existing;
          std::move_backward(neighbors + protected_edges, end, end + 1);
          neighbors[protected_edges] = source;
        }
      }
      ace_disk_host_io<true>(graph_fd, graph.data(), graph.size() * sizeof(IdxT), file_offset);
      bucket.close();
      std::filesystem::remove(bucket_path(partition));
    }
    buffer_.clear();
  }

 private:
  std::string bucket_path(size_t partition) const
  {
    return directory_ + "/" + std::to_string(partition);
  }

  void flush()
  {
    std::sort(buffer_.begin(), buffer_.end(), [](record const& a, record const& b) {
      return a.destination < b.destination;
    });
    size_t begin = 0;
    while (begin < buffer_.size()) {
      auto const partition =
        size_t(std::upper_bound(offsets_.begin(), offsets_.end(), buffer_[begin].destination) -
               offsets_.begin() - 1);
      size_t end = begin + 1;
      while (end < buffer_.size() && buffer_[end].destination < offsets_[partition + 1]) {
        ++end;
      }
      cuvs::util::file_descriptor bucket(
        bucket_path(partition), sizes_[partition] == 0 ? O_CREAT | O_EXCL | O_WRONLY : O_WRONLY);
      ace_disk_host_io<true>(bucket,
                             buffer_.data() + begin,
                             (end - begin) * sizeof(record),
                             sizes_[partition] * sizeof(record));
      sizes_[partition] += end - begin;
      begin = end;
    }
    buffer_.clear();
  }

  std::vector<IdxT> offsets_;
  uint32_t degree_;
  size_t capacity_;
  std::vector<size_t> sizes_;
  std::vector<record> buffer_;
  std::string directory_;
};

}  // namespace cuvs::neighbors::cagra::detail
