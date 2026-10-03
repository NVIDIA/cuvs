/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../cagra/ace_external_plan.hpp"
#include "external_format.hpp"
#include "external_translate.hpp"
#include "external_workspace.hpp"
#include "index_impl.hpp"
#include "serialize_layout.hpp"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/hnsw.hpp>
#include <cuvs/selection/select_k.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuvs::neighbors::hnsw::detail::external {

struct external_io_counters {
  uint64_t sample_read_bytes      = 0;
  uint64_t source_read_bytes      = 0;
  uint64_t stage_write_bytes      = 0;
  uint64_t stage_read_bytes       = 0;
  uint64_t sidecar_write_bytes    = 0;
  uint64_t sidecar_read_bytes     = 0;
  uint64_t output_write_bytes     = 0;
  uint64_t stage_write_requests   = 0;
  uint64_t stage_read_requests    = 0;
  uint64_t sidecar_write_requests = 0;
  uint64_t sidecar_read_requests  = 0;
  uint64_t output_write_requests  = 0;
};

inline uint64_t elapsed_milliseconds(std::chrono::steady_clock::time_point start,
                                     std::chrono::steady_clock::time_point end)
{
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}

template <typename T>
void copy_rows_as_float(const T* source, float* destination, uint64_t rows, uint64_t dimension)
{
  const uint64_t elements = checked_file_mul(rows, dimension, "float conversion elements");
  RAFT_EXPECTS(elements <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
               "Float conversion exceeds supported element count");
  if constexpr (std::is_same_v<T, float>) {
    std::memcpy(destination, source, static_cast<size_t>(elements) * sizeof(float));
  } else {
#pragma omp parallel for
    for (int64_t index = 0; index < static_cast<int64_t>(elements); ++index) {
      destination[index] = static_cast<float>(source[index]);
    }
  }
}

template <typename T>
uint64_t external_parameter_fingerprint(uint64_t rows,
                                        uint64_t dim,
                                        const index_params& params,
                                        const cagra::detail::ace_external_plan& plan,
                                        size_t graph_degree,
                                        size_t intermediate_graph_degree,
                                        size_t ace_ef_construction)
{
  std::array<uint64_t, 12> fields{rows,
                                  dim,
                                  sizeof(T),
                                  params.M,
                                  static_cast<uint64_t>(params.ef_construction),
                                  ace_ef_construction,
                                  static_cast<uint64_t>(params.metric),
                                  static_cast<uint64_t>(params.hierarchy),
                                  plan.partitions,
                                  graph_degree,
                                  intermediate_graph_degree,
                                  stage_schema_version};
  return fnv1a64(fields.data(), sizeof(fields));
}

template <typename T>
class stage_buffer_pool {
 public:
  struct file_state {
    std::filesystem::path path;
    stage_header header;
    uint64_t appended_records = 0;
    uint64_t flushed_records  = 0;
    uint64_t allocated_bytes  = stage_data_offset;
    std::vector<std::byte> buffer;
    size_t used     = 0;
    uint64_t access = 0;
  };

  stage_buffer_pool(external_workspace& workspace,
                    uint32_t dimension,
                    uint32_t partitions,
                    uint64_t total_buffer_bytes,
                    uint64_t preferred_buffer_bytes,
                    uint64_t fingerprint)
    : workspace_(workspace),
      dimension_(dimension),
      partitions_(partitions),
      record_size_(checked_file_add(2 * sizeof(uint32_t),
                                    checked_file_mul(dimension, sizeof(T), "stage vector row"),
                                    "stage record")),
      growth_extent_bytes_(
        std::clamp<uint64_t>(checked_file_mul(record_size_, 1024, "stage growth extent"),
                             uint64_t{64} << 10,
                             uint64_t{8} << 20)),
      total_buffer_bytes_(std::max<uint64_t>(total_buffer_bytes, record_size_)),
      preferred_buffer_bytes_(
        std::max<uint64_t>(record_size_, std::min(preferred_buffer_bytes, total_buffer_bytes_)))
  {
    if (partitions_ == 0) { throw std::invalid_argument("external HNSW requires partitions"); }
    // Prefer one buffer per stage file. Fewer LRU buffers make random assignment nearly one write
    // per record.
    const uint64_t file_count = checked_file_mul(partitions_, 2, "external stage file count");
    const uint64_t per_file_budget =
      std::max<uint64_t>(record_size_, total_buffer_bytes_ / file_count);
    preferred_buffer_bytes_ = std::min(preferred_buffer_bytes_, per_file_budget);
    files_.reserve(static_cast<size_t>(partitions) * 2);
    for (uint32_t partition = 0; partition < partitions; ++partition) {
      add_file(partition, false, fingerprint);
      add_file(partition, true, fingerprint);
    }
  }

  ~stage_buffer_pool() noexcept
  {
    try {
      flush_all();
    } catch (...) {
    }
  }

  void append_core(uint32_t partition, uint32_t original_label, const T* vector)
  {
    append(file_index(partition, false), original_label, 0, vector);
  }

  void append_spill(uint32_t partition,
                    uint32_t owner_partition,
                    uint32_t owner_ordinal,
                    const T* vector)
  {
    append(file_index(partition, true), owner_partition, owner_ordinal, vector);
  }

  void finalize()
  {
    flush_all();
    for (auto& state : files_) {
      cuvs::util::file_descriptor fd(state.path.string(), O_RDWR);
      state.header.committed_records = state.appended_records;
      uint64_t exact_size            = expected_file_size(state.header);
      if (exact_size > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
        throw std::overflow_error("external HNSW stage exceeds off_t");
      }
      if (::ftruncate(fd.get(), static_cast<off_t>(exact_size)) != 0) {
        throw std::runtime_error("failed to trim external HNSW stage: " +
                                 std::string{strerror(errno)});
      }
      write_stage_header(fd.get(), state.header);
      if (::fsync(fd.get()) != 0) {
        throw std::runtime_error("failed to commit external HNSW stage: " +
                                 std::string{strerror(errno)});
      }
      validate_stage_file_size(fd.get(), state.header);
      std::vector<std::byte>{}.swap(state.buffer);
    }
    resident_buffer_bytes_ = 0;
  }

  [[nodiscard]] const file_state& core(uint32_t partition) const
  {
    return files_.at(file_index(partition, false));
  }
  [[nodiscard]] const file_state& spill(uint32_t partition) const
  {
    return files_.at(file_index(partition, true));
  }
  [[nodiscard]] uint64_t record_size() const noexcept { return record_size_; }
  [[nodiscard]] uint64_t bytes_written() const noexcept { return bytes_written_; }
  [[nodiscard]] uint64_t write_requests() const noexcept { return write_requests_; }

 private:
  [[nodiscard]] size_t file_index(uint32_t partition, bool spill) const
  {
    if (partition >= partitions_) { throw std::out_of_range("external HNSW partition"); }
    return static_cast<size_t>(partition) * 2 + static_cast<size_t>(spill);
  }

  void add_file(uint32_t partition, bool spill, uint64_t fingerprint)
  {
    std::ostringstream filename;
    filename << (spill ? "spill." : "core.") << partition << ".stage";
    auto path = workspace_.private_path(filename.str());
    workspace_.create_stage_file<T>(filename.str(),
                                    spill ? stage_kind::spill : stage_kind::core,
                                    dimension_,
                                    partition,
                                    0,
                                    record_size_);
    file_state state;
    state.path   = std::move(path);
    state.header = make_stage_header<T>(spill ? stage_kind::spill : stage_kind::core,
                                        dimension_,
                                        partition,
                                        0,
                                        record_size_,
                                        fingerprint);
    files_.push_back(std::move(state));
  }

  void ensure_buffer(size_t index)
  {
    auto& state = files_[index];
    if (!state.buffer.empty()) { return; }
    while (resident_buffer_bytes_ + preferred_buffer_bytes_ > total_buffer_bytes_) {
      size_t victim = files_.size();
      for (size_t candidate = 0; candidate < files_.size(); ++candidate) {
        if (!files_[candidate].buffer.empty() && candidate != index &&
            (victim == files_.size() || files_[candidate].access < files_[victim].access)) {
          victim = candidate;
        }
      }
      if (victim == files_.size()) { break; }
      flush(victim);
      resident_buffer_bytes_ -= files_[victim].buffer.size();
      std::vector<std::byte>{}.swap(files_[victim].buffer);
    }
    state.buffer.resize(static_cast<size_t>(preferred_buffer_bytes_));
    resident_buffer_bytes_ += state.buffer.size();
  }

  void append(size_t index, uint32_t first, uint32_t second, const T* vector)
  {
    ensure_buffer(index);
    auto& state = files_[index];
    if (state.buffer.size() - state.used < record_size_) { flush(index); }
    std::byte* destination = state.buffer.data() + state.used;
    std::memcpy(destination, &first, sizeof(first));
    std::memcpy(destination + sizeof(first), &second, sizeof(second));
    std::memcpy(
      destination + 2 * sizeof(uint32_t), vector, static_cast<size_t>(dimension_) * sizeof(T));
    state.used += static_cast<size_t>(record_size_);
    ++state.appended_records;
    state.access = ++clock_;
  }

  void reserve_extent(file_state& state, int fd, uint64_t required_size)
  {
    if (required_size <= state.allocated_bytes) { return; }
    uint64_t rounded =
      checked_file_add(required_size, growth_extent_bytes_ - 1, "rounded stage growth extent");
    uint64_t new_size =
      checked_file_mul(rounded / growth_extent_bytes_, growth_extent_bytes_, "stage growth extent");
    if (new_size > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
      throw std::overflow_error("external HNSW stage allocation exceeds off_t");
    }
    int status = ::posix_fallocate(fd,
                                   static_cast<off_t>(state.allocated_bytes),
                                   static_cast<off_t>(new_size - state.allocated_bytes));
    if (status != 0) {
      throw std::runtime_error("failed to grow external HNSW stage: " +
                               std::string{strerror(status)});
    }
    state.allocated_bytes = new_size;
  }

  void flush(size_t index)
  {
    auto& state = files_[index];
    if (state.used == 0) { return; }
    cuvs::util::file_descriptor fd(state.path.string(), O_RDWR);
    uint64_t offset =
      checked_file_add(stage_data_offset,
                       checked_file_mul(state.flushed_records, record_size_, "stage append offset"),
                       "stage append offset");
    uint64_t end = checked_file_add(offset, state.used, "stage append end");
    reserve_extent(state, fd.get(), end);
    cuvs::util::write_large_file(fd, state.buffer.data(), state.used, offset);
    state.flushed_records += state.used / record_size_;
    bytes_written_ += state.used;
    ++write_requests_;
    state.used = 0;
  }

  void flush_all()
  {
    for (size_t index = 0; index < files_.size(); ++index) {
      flush(index);
    }
  }

  external_workspace& workspace_;
  uint32_t dimension_;
  uint32_t partitions_;
  uint64_t record_size_;
  uint64_t growth_extent_bytes_;
  uint64_t total_buffer_bytes_;
  uint64_t preferred_buffer_bytes_;
  std::vector<file_state> files_;
  uint64_t resident_buffer_bytes_ = 0;
  uint64_t bytes_written_         = 0;
  uint64_t write_requests_        = 0;
  uint64_t clock_                 = 0;
};

class buffered_stage_reader {
 public:
  buffered_stage_reader(const cuvs::util::file_descriptor& fd,
                        stage_header header,
                        size_t target_buffer_bytes)
    : fd_(fd),
      header_(header),
      records_per_buffer_(
        std::max<uint64_t>(1, target_buffer_bytes / std::max<uint64_t>(1, header.record_size))),
      buffer_(static_cast<size_t>(
        checked_file_mul(records_per_buffer_, header.record_size, "stage read buffer")))
  {
    const auto path = fd_.get_path();
    if (!path.empty()) { file_handle_ = std::make_unique<kvikio::FileHandle>(path, "r"); }
  }

  bool next(const std::byte*& record)
  {
    if (current_record_ == header_.committed_records) { return false; }
    if (buffer_index_ == buffered_records_) { refill(); }
    record = buffer_.data() + static_cast<size_t>(buffer_index_ * header_.record_size);
    ++buffer_index_;
    ++current_record_;
    return true;
  }

  [[nodiscard]] uint64_t bytes_read() const noexcept { return bytes_read_; }
  [[nodiscard]] uint64_t read_requests() const noexcept { return read_requests_; }

 private:
  void refill()
  {
    uint64_t remaining = header_.committed_records - current_record_;
    buffered_records_  = std::min(records_per_buffer_, remaining);
    size_t bytes =
      static_cast<size_t>(checked_file_mul(buffered_records_, header_.record_size, "stage read"));
    uint64_t offset =
      checked_file_add(header_.data_offset,
                       checked_file_mul(current_record_, header_.record_size, "stage read offset"),
                       "stage read offset");
    if (file_handle_) {
      const size_t bytes_read = file_handle_->pread(buffer_.data(), bytes, offset).get();
      RAFT_EXPECTS(bytes_read == bytes,
                   "Incomplete stage read: expected %zu bytes, read %zu",
                   bytes,
                   bytes_read);
    } else {
      cuvs::util::read_large_file(fd_, buffer_.data(), bytes, offset);
    }
    bytes_read_ += bytes;
    ++read_requests_;
    buffer_index_ = 0;
  }

  const cuvs::util::file_descriptor& fd_;
  stage_header header_;
  uint64_t records_per_buffer_;
  std::vector<std::byte> buffer_;
  std::unique_ptr<kvikio::FileHandle> file_handle_;
  uint64_t current_record_   = 0;
  uint64_t buffered_records_ = 0;
  uint64_t buffer_index_     = 0;
  uint64_t bytes_read_       = 0;
  uint64_t read_requests_    = 0;
};

template <typename T>
raft::device_matrix<float, int64_t> train_external_centroids(
  raft::resources const& res,
  raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
  const cagra::detail::ace_external_plan& plan,
  external_io_counters& io)
{
  const uint64_t rows        = dataset.extent(0);
  const uint64_t dim         = dataset.extent(1);
  const uint64_t sample_rows = plan.centroid_sample_rows;
  auto sample                = raft::make_host_matrix<float, int64_t>(sample_rows, dim);

  uint64_t copied = 0;
  for (const auto& range : cagra::detail::make_external_sample_ranges(rows, sample_rows)) {
    copy_rows_as_float(dataset.data_handle() + range.start * dim,
                       sample.data_handle() + copied * dim,
                       range.count,
                       dim);
    copied += range.count;
  }
  RAFT_EXPECTS(copied == sample_rows, "External HNSW centroid sample size mismatch");
  io.sample_read_bytes = checked_file_mul(
    sample_rows, checked_file_mul(dim, sizeof(T), "sample vector"), "sample bytes");

  auto sample_device = raft::make_device_matrix<float, int64_t>(res, sample_rows, dim);
  raft::copy(res, sample_device.view(), sample.view());
  auto centroids =
    raft::make_device_matrix<float, int64_t>(res, plan.partitions, dataset.extent(1));
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  cuvs::cluster::kmeans::fit(res, kmeans_params, sample_device.view(), centroids.view());
  return centroids;
}

template <typename T>
void assign_and_stage(raft::resources const& res,
                      raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
                      const cagra::detail::ace_external_plan& plan,
                      raft::device_matrix_view<const float, int64_t, raft::row_major> centroids,
                      stage_buffer_pool<T>& stages,
                      external_io_counters& io)
{
  const uint64_t rows       = dataset.extent(0);
  const uint64_t dim        = dataset.extent(1);
  const uint64_t partitions = plan.partitions;
  const uint64_t chunk_rows = plan.assignment_chunk_rows;

  auto host_input           = raft::make_host_matrix<float, int64_t>(chunk_rows, dim);
  auto device_input         = raft::make_device_matrix<float, int64_t>(res, chunk_rows, dim);
  auto device_distances     = raft::make_device_matrix<float, int64_t>(res, chunk_rows, partitions);
  auto device_top_distances = raft::make_device_matrix<float, int64_t>(res, chunk_rows, 2);
  auto device_top_labels    = raft::make_device_matrix<uint32_t, int64_t>(res, chunk_rows, 2);
  auto host_top_labels      = raft::make_host_matrix<uint32_t, int64_t>(chunk_rows, 2);
  std::vector<uint64_t> core_counts(partitions, 0);

  for (uint64_t base = 0; base < rows; base += chunk_rows) {
    uint64_t count = std::min(chunk_rows, rows - base);
    auto host_view =
      raft::make_host_matrix_view<float, int64_t>(host_input.data_handle(), count, dim);
    copy_rows_as_float(dataset.data_handle() + base * dim, host_view.data_handle(), count, dim);
    auto device_view =
      raft::make_device_matrix_view<float, int64_t>(device_input.data_handle(), count, dim);
    auto distance_view = raft::make_device_matrix_view<float, int64_t>(
      device_distances.data_handle(), count, partitions);
    auto top_distance_view =
      raft::make_device_matrix_view<float, int64_t>(device_top_distances.data_handle(), count, 2);
    auto top_label_view =
      raft::make_device_matrix_view<uint32_t, int64_t>(device_top_labels.data_handle(), count, 2);
    auto host_label_view =
      raft::make_host_matrix_view<uint32_t, int64_t>(host_top_labels.data_handle(), count, 2);
    raft::copy(res, device_view, host_view);
    cuvs::distance::pairwise_distance(res,
                                      raft::make_const_mdspan(device_view),
                                      centroids,
                                      distance_view,
                                      cuvs::distance::DistanceType::L2Expanded);
    cuvs::selection::select_k(res,
                              raft::make_const_mdspan(distance_view),
                              std::nullopt,
                              top_distance_view,
                              top_label_view,
                              true,
                              true);
    raft::copy(res, host_label_view, top_label_view);
    raft::resource::sync_stream(res);

    for (uint64_t local = 0; local < count; ++local) {
      uint32_t core_partition  = host_label_view(local, 0);
      uint32_t spill_partition = host_label_view(local, 1);
      RAFT_EXPECTS(core_partition < partitions && spill_partition < partitions &&
                     core_partition != spill_partition,
                   "Invalid external HNSW top-two partition assignment");
      uint64_t ordinal = core_counts[core_partition]++;
      RAFT_EXPECTS(ordinal <= std::numeric_limits<uint32_t>::max(),
                   "External HNSW core ordinal exceeds uint32_t");
      auto* vector = dataset.data_handle() + (base + local) * dim;
      stages.append_core(core_partition, static_cast<uint32_t>(base + local), vector);
      stages.append_spill(spill_partition, core_partition, static_cast<uint32_t>(ordinal), vector);
    }
  }
  io.source_read_bytes =
    checked_file_mul(rows, checked_file_mul(dim, sizeof(T), "source row"), "source scan bytes");
  io.stage_write_bytes    = stages.bytes_written();
  io.stage_write_requests = stages.write_requests();
}

template <typename T>
struct resident_partition {
  raft::host_matrix<T, int64_t> vectors;
  std::vector<uint32_t> local_to_global;
  std::vector<uint32_t> core_labels;
  uint64_t core_rows = 0;
};

inline uint32_t external_owner_id(const std::vector<uint32_t>& prefixes,
                                  uint32_t owner_partition,
                                  uint32_t owner_ordinal)
{
  uint64_t owner_next = static_cast<uint64_t>(owner_partition) + 1;
  RAFT_EXPECTS(owner_next < prefixes.size(), "Invalid external HNSW spill owner partition");
  uint64_t owner_begin = prefixes[owner_partition];
  uint64_t owner_end   = prefixes[owner_next];
  uint64_t global      = checked_file_add(owner_begin, owner_ordinal, "external HNSW owner ID");
  RAFT_EXPECTS(global < owner_end, "Invalid external HNSW spill owner ordinal");
  return static_cast<uint32_t>(global);
}

template <typename T>
resident_partition<T> read_resident_partition(
  const typename stage_buffer_pool<T>::file_state& core_state,
  const typename stage_buffer_pool<T>::file_state& spill_state,
  uint32_t partition,
  const std::vector<uint32_t>& prefixes,
  uint64_t expected_record_size,
  uint64_t fingerprint,
  uint64_t reader_buffer_bytes,
  external_io_counters& io)
{
  cuvs::util::file_descriptor core_fd(core_state.path.string(), O_RDONLY);
  cuvs::util::file_descriptor spill_fd(spill_state.path.string(), O_RDONLY);
  auto core_header  = read_stage_header(core_fd.get());
  auto spill_header = read_stage_header(spill_fd.get());
  validate_stage_header(core_header,
                        stage_kind::core,
                        element_kind_for<T>(),
                        sizeof(T),
                        core_header.dimension,
                        partition,
                        0,
                        expected_record_size,
                        fingerprint);
  validate_stage_header(spill_header,
                        stage_kind::spill,
                        element_kind_for<T>(),
                        sizeof(T),
                        spill_header.dimension,
                        partition,
                        0,
                        expected_record_size,
                        fingerprint);
  RAFT_EXPECTS(core_header.dimension == spill_header.dimension,
               "External HNSW core/spill dimension mismatch");
  validate_stage_file_size(core_fd.get(), core_header);
  validate_stage_file_size(spill_fd.get(), spill_header);

  uint64_t total_rows = checked_file_add(
    core_header.committed_records, spill_header.committed_records, "resident partition rows");
  resident_partition<T> resident{
    raft::make_host_matrix<T, int64_t>(total_rows, core_header.dimension),
    std::vector<uint32_t>(total_rows),
    std::vector<uint32_t>(core_header.committed_records),
    core_header.committed_records};

  auto read_file = [&](const cuvs::util::file_descriptor& fd,
                       const stage_header& header,
                       bool spill,
                       uint64_t destination_base) {
    buffered_stage_reader reader(
      fd, header, static_cast<size_t>(std::max<uint64_t>(1, reader_buffer_bytes)));
    const std::byte* record = nullptr;
    uint64_t row            = 0;
    while (reader.next(record)) {
      uint32_t first;
      uint32_t second;
      std::memcpy(&first, record, sizeof(first));
      std::memcpy(&second, record + sizeof(first), sizeof(second));
      std::memcpy(&resident.vectors(destination_base + row, 0),
                  record + 2 * sizeof(uint32_t),
                  static_cast<size_t>(header.dimension) * sizeof(T));
      if (spill) {
        resident.local_to_global[destination_base + row] =
          external_owner_id(prefixes, first, second);
      } else {
        uint64_t global = static_cast<uint64_t>(prefixes[partition]) + row;
        RAFT_EXPECTS(global < prefixes.back(), "Invalid external HNSW core prefix");
        resident.local_to_global[destination_base + row] = static_cast<uint32_t>(global);
        resident.core_labels[row]                        = first;
      }
      ++row;
    }
    RAFT_EXPECTS(row == header.committed_records, "External HNSW stage record count mismatch");
    io.stage_read_bytes += reader.bytes_read();
    io.stage_read_requests += reader.read_requests();
  };
  read_file(core_fd, core_header, false, 0);
  read_file(spill_fd, spill_header, true, resident.core_rows);
  return resident;
}

template <typename T>
class level_file {
 public:
  level_file(external_workspace& workspace,
             std::string filename,
             stage_kind kind,
             uint32_t dimension,
             uint32_t level,
             uint64_t record_size,
             uint64_t fingerprint,
             size_t buffer_size)
    : fd_(workspace.create_stage_file<T>(filename, kind, dimension, 0, level, record_size)),
      header_(make_stage_header<T>(kind, dimension, 0, level, record_size, fingerprint)),
      writer_(fd_, stage_data_offset, buffer_size)
  {
  }

  void append(const void* record)
  {
    writer_.write(record, static_cast<size_t>(header_.record_size));
    ++count_;
  }

  void finalize()
  {
    writer_.flush();
    header_.committed_records = count_;
    uint64_t exact            = expected_file_size(header_);
    if (::ftruncate(fd_.get(), static_cast<off_t>(exact)) != 0) {
      throw std::runtime_error("failed to finalize external HNSW level file");
    }
    write_stage_header(fd_.get(), header_);
    if (::fsync(fd_.get()) != 0) {
      throw std::runtime_error("failed to sync external HNSW level file");
    }
    validate_stage_file_size(fd_.get(), header_);
  }

  [[nodiscard]] const cuvs::util::file_descriptor& descriptor() const noexcept { return fd_; }
  [[nodiscard]] const stage_header& header() const noexcept { return header_; }
  [[nodiscard]] uint64_t count() const noexcept { return count_; }
  [[nodiscard]] uint64_t request_count() const noexcept { return writer_.request_count(); }

 private:
  cuvs::util::file_descriptor fd_;
  stage_header header_;
  sequential_file_writer writer_;
  uint64_t count_ = 0;
};

template <typename T>
void write_partitioned_level(raft::resources const& res,
                             const resident_partition<T>& resident,
                             const std::vector<uint8_t>& hierarchy_levels,
                             uint32_t level,
                             const hnsw_serialize_layout& layout,
                             cuvs::distance::DistanceType metric,
                             level_file<T>& sidecar)
{
  std::vector<uint64_t> promoted_rows;
  promoted_rows.reserve(resident.local_to_global.size() / std::max<size_t>(2, layout.M));
  for (uint64_t row = 0; row < resident.local_to_global.size(); ++row) {
    if (hierarchy_levels[row] >= level) { promoted_rows.push_back(row); }
  }

  const uint64_t count = promoted_rows.size();
  auto promoted_vectors =
    raft::make_host_matrix<T, int64_t>(count, static_cast<int64_t>(layout.dimension));
  std::vector<uint32_t> promoted_ids(count);
  std::vector<int64_t> core_promoted_position(resident.core_rows, -1);
  for (uint64_t promoted = 0; promoted < count; ++promoted) {
    uint64_t source = promoted_rows[promoted];
    std::copy(&resident.vectors(source, 0),
              &resident.vectors(source, 0) + layout.dimension,
              &promoted_vectors(promoted, 0));
    promoted_ids[promoted] = resident.local_to_global[source];
    if (source < resident.core_rows) { core_promoted_position[source] = promoted; }
  }

  uint64_t neighbor_count = count > 1 ? std::min<uint64_t>(layout.M, count - 1) : 0;
  auto neighbors          = raft::make_host_matrix<uint32_t, int64_t>(count, neighbor_count);
  if (count > 1) {
    all_neighbors_graph(
      res, raft::make_const_mdspan(promoted_vectors.view()), neighbors.view(), metric);
  }

  std::vector<uint32_t> record(1 + layout.M, std::numeric_limits<uint32_t>::max());
  for (uint64_t core = 0; core < resident.core_rows; ++core) {
    int64_t promoted = core_promoted_position[core];
    if (promoted < 0) { continue; }
    record[0] = resident.local_to_global[core];
    std::fill(record.begin() + 1, record.end(), std::numeric_limits<uint32_t>::max());
    for (uint64_t neighbor = 0; neighbor < neighbor_count; ++neighbor) {
      uint32_t local = neighbors(promoted, neighbor);
      RAFT_EXPECTS(local < promoted_ids.size(), "Invalid external HNSW upper local ID");
      record[neighbor + 1] = promoted_ids[local];
    }
    sidecar.append(record.data());
  }
}

template <typename T>
void append_global_level_vectors(const resident_partition<T>& resident,
                                 const std::vector<uint8_t>& hierarchy_levels,
                                 uint32_t level,
                                 const hnsw_serialize_layout& layout,
                                 level_file<T>& vectors)
{
  std::vector<std::byte> record(sizeof(uint32_t) + layout.dimension * sizeof(T));
  for (uint64_t core = 0; core < resident.core_rows; ++core) {
    uint32_t id = resident.local_to_global[core];
    if (hierarchy_levels[core] < level) { continue; }
    std::memcpy(record.data(), &id, sizeof(id));
    std::memcpy(
      record.data() + sizeof(id), &resident.vectors(core, 0), layout.dimension * sizeof(T));
    vectors.append(record.data());
  }
}

template <typename T>
void build_global_level(raft::resources const& res,
                        level_file<T>& vectors,
                        level_file<T>& sidecar,
                        const hnsw_serialize_layout& layout,
                        cuvs::distance::DistanceType metric,
                        uint64_t reader_buffer_bytes,
                        external_io_counters& io)
{
  vectors.finalize();
  const auto& header = vectors.header();
  buffered_stage_reader reader(vectors.descriptor(), header, reader_buffer_bytes);
  auto dataset = raft::make_host_matrix<T, int64_t>(header.committed_records, layout.dimension);
  std::vector<uint32_t> ids(header.committed_records);
  const std::byte* record = nullptr;
  uint64_t row            = 0;
  while (reader.next(record)) {
    std::memcpy(&ids[row], record, sizeof(uint32_t));
    std::memcpy(&dataset(row, 0), record + sizeof(uint32_t), layout.dimension * sizeof(T));
    ++row;
  }
  io.sidecar_read_bytes += reader.bytes_read();
  io.sidecar_read_requests += reader.read_requests();

  uint64_t neighbor_count =
    row > 1 ? std::min<uint64_t>(layout.M, static_cast<uint64_t>(row - 1)) : 0;
  auto neighbors = raft::make_host_matrix<uint32_t, int64_t>(row, neighbor_count);
  if (row > 1) {
    all_neighbors_graph(res, raft::make_const_mdspan(dataset.view()), neighbors.view(), metric);
  }
  std::vector<uint32_t> output(1 + layout.M, std::numeric_limits<uint32_t>::max());
  for (uint64_t index = 0; index < row; ++index) {
    output[0] = ids[index];
    std::fill(output.begin() + 1, output.end(), std::numeric_limits<uint32_t>::max());
    for (uint64_t neighbor = 0; neighbor < neighbor_count; ++neighbor) {
      uint32_t local = neighbors(index, neighbor);
      RAFT_EXPECTS(local < ids.size(), "Invalid external HNSW global upper local ID");
      output[neighbor + 1] = ids[local];
    }
    sidecar.append(output.data());
  }
}

template <typename T>
std::unique_ptr<index<T>> build_external(
  raft::resources const& res,
  const index_params& params,
  raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
  const cagra::detail::ace_external_plan& plan,
  const graph_build_params::ace_params& ace_params,
  size_t graph_degree,
  size_t intermediate_graph_degree,
  size_t ace_ef_construction)
{
  RAFT_EXPECTS(plan.use_disk, "External HNSW builder requires a disk plan");
  RAFT_EXPECTS(params.hierarchy != HnswHierarchy::CPU,
               "Disk HNSW construction does not support a CPU hierarchy");
  RAFT_EXPECTS(graph_degree == 2 * params.M,
               "External HNSW graph degree must equal the base-layer capacity 2*M");
  RAFT_EXPECTS(intermediate_graph_degree >= graph_degree,
               "External HNSW intermediate graph degree must be at least graph degree");
  RAFT_EXPECTS(ace_ef_construction > 0, "External HNSW CAGRA-ACE ef_construction must be positive");
  const uint64_t rows        = dataset.extent(0);
  const uint64_t dim         = dataset.extent(1);
  const uint64_t fingerprint = external_parameter_fingerprint<T>(
    rows, dim, params, plan, graph_degree, intermediate_graph_degree, ace_ef_construction);
  [[maybe_unused]] const int num_threads =
    params.num_threads == 0 ? cuvs::core::omp::get_max_threads() : params.num_threads;
  external_workspace workspace(ace_params.build_dir, fingerprint);
  external_io_counters io;
  auto build_start               = std::chrono::steady_clock::now();
  auto preflight_start           = build_start;
  uint64_t preflight_ms          = 0;
  uint64_t centroid_training_ms  = 0;
  uint64_t assignment_stage_ms   = 0;
  uint64_t partition_build_ms    = 0;
  uint64_t partition_load_ms     = 0;
  uint64_t cagra_build_ms        = 0;
  uint64_t base_serialization_ms = 0;
  uint64_t hierarchy_ms          = 0;
  uint64_t publication_ms        = 0;

  try {
    auto hierarchy = params.hierarchy == HnswHierarchy::GPU ? summarize_hierarchy(rows, params.M)
                                                            : hierarchy_summary{};
    auto hnsw_index =
      std::make_unique<index_impl<T>>(static_cast<int>(dim), params.metric, params.hierarchy);
    auto layout = make_hnsw_serialize_layout<T, typename hnsw_dist_t<T>::type>(
      hnsw_index->get_space(),
      rows,
      dim,
      graph_degree,
      params.ef_construction,
      params.hierarchy == HnswHierarchy::GPU,
      hierarchy);
    uint64_t exact_output      = layout.exact_file_size(hierarchy.total_active_occurrences);
    auto output_fd             = workspace.create_partial(exact_output);
    auto free_space            = std::filesystem::space(workspace.invocation_dir());
    uint64_t stage_record_size = checked_file_add(
      2 * sizeof(uint32_t), checked_file_mul(dim, sizeof(T), "stage vector"), "stage record");
    uint64_t stage_growth_extent =
      std::clamp<uint64_t>(checked_file_mul(stage_record_size, 1024, "stage growth extent"),
                           uint64_t{64} << 10,
                           uint64_t{8} << 20);
    uint64_t stage_extent_overhead = checked_file_mul(
      checked_file_mul(plan.partitions, 2, "stage file count"),
      checked_file_add(stage_data_offset, stage_growth_extent, "stage file overhead"),
      "stage extent overhead");
    uint64_t required_sidecar_space = 0;
    for (uint32_t level = 1; level <= hierarchy.max_level; ++level) {
      const uint64_t active = hierarchy.active_by_level[level - 1];
      const uint64_t link_record =
        checked_file_add(sizeof(uint32_t),
                         checked_file_mul(layout.M, sizeof(uint32_t), "upper links"),
                         "upper link record");
      required_sidecar_space = checked_file_add(
        required_sidecar_space,
        checked_file_add(stage_data_offset,
                         checked_file_mul(active, link_record, "upper link sidecar"),
                         "upper link sidecar file"),
        "upper sidecar high-water space");
      if (active <= plan.global_upper_level_max_rows) {
        const uint64_t vector_record =
          checked_file_add(sizeof(uint32_t),
                           checked_file_mul(dim, sizeof(T), "upper vector"),
                           "upper vector record");
        required_sidecar_space = checked_file_add(
          required_sidecar_space,
          checked_file_add(stage_data_offset,
                           checked_file_mul(active, vector_record, "upper vector sidecar"),
                           "upper vector sidecar file"),
          "upper sidecar high-water space");
      }
    }
    uint64_t required_working_space = checked_file_add(
      checked_file_add(plan.bytes.stage_write, stage_extent_overhead, "stage high-water space"),
      required_sidecar_space,
      "external HNSW working-space high-water mark");
    RAFT_EXPECTS(free_space.available >= required_working_space,
                 "External HNSW staging and hierarchy sidecars require %zu bytes after final-index "
                 "preallocation, but only %zu bytes are available",
                 static_cast<size_t>(required_working_space),
                 static_cast<size_t>(free_space.available));
    preflight_ms = elapsed_milliseconds(preflight_start, std::chrono::steady_clock::now());

    stage_buffer_pool<T> stages(workspace,
                                static_cast<uint32_t>(dim),
                                static_cast<uint32_t>(plan.partitions),
                                plan.staging_buffer_bytes,
                                plan.preferred_buffer_bytes,
                                fingerprint);
    workspace.update_manifest("centroid_training");
    auto centroid_start   = std::chrono::steady_clock::now();
    auto assignment_start = centroid_start;
    {
      auto centroids       = train_external_centroids(res, dataset, plan, io);
      centroid_training_ms = elapsed_milliseconds(centroid_start, std::chrono::steady_clock::now());

      workspace.update_manifest("assignment_and_staging");
      assignment_start = std::chrono::steady_clock::now();
      assign_and_stage(res, dataset, plan, raft::make_const_mdspan(centroids.view()), stages, io);
    }
    stages.finalize();
    io.stage_write_bytes    = stages.bytes_written();
    io.stage_write_requests = stages.write_requests();
    assignment_stage_ms = elapsed_milliseconds(assignment_start, std::chrono::steady_clock::now());

    std::vector<uint32_t> prefixes(plan.partitions + 1, 0);
    uint64_t total_core  = 0;
    uint64_t total_spill = 0;
    for (uint32_t partition = 0; partition < plan.partitions; ++partition) {
      uint64_t core  = stages.core(partition).header.committed_records;
      uint64_t spill = stages.spill(partition).header.committed_records;
      uint64_t actual_occurrences =
        checked_file_add(core, spill, "external HNSW actual partition rows");
      uint64_t required_host = checked_file_add(
        plan.host_fixed_bytes,
        checked_file_mul(
          actual_occurrences, plan.host_per_occurrence, "external HNSW actual host partition"),
        "external HNSW required host bytes");
      uint64_t required_device = checked_file_add(
        plan.device_fixed_bytes,
        checked_file_mul(
          actual_occurrences, plan.device_per_occurrence, "external HNSW actual device partition"),
        "external HNSW required device bytes");
      RAFT_EXPECTS(core + spill <= plan.max_occurrences,
                   "External HNSW partition %u has %zu rows; planned maximum is %zu. Estimated "
                   "required/available host bytes are %zu/%zu and device bytes are %zu/%zu. "
                   "Increase partitions or memory budget.",
                   partition,
                   static_cast<size_t>(actual_occurrences),
                   static_cast<size_t>(plan.max_occurrences),
                   static_cast<size_t>(required_host),
                   static_cast<size_t>(plan.host_budget_bytes),
                   static_cast<size_t>(required_device),
                   static_cast<size_t>(plan.device_budget_bytes));
      total_core += core;
      total_spill += spill;
      RAFT_EXPECTS(total_core <= std::numeric_limits<uint32_t>::max(),
                   "External HNSW core prefix exceeds uint32_t");
      prefixes[partition + 1] = static_cast<uint32_t>(total_core);
    }
    RAFT_EXPECTS(total_core == rows && total_spill == rows,
                 "External HNSW staging counts do not match the dataset");
    workspace.update_manifest("staged", total_core, total_spill);

    sequential_file_writer output(output_fd, 0, static_cast<size_t>(plan.hnsw_output_buffer_bytes));
    write_hnsw_header(output, layout);

    std::vector<std::unique_ptr<level_file<T>>> sidecars(hierarchy.max_level);
    std::vector<std::unique_ptr<level_file<T>>> global_vectors(hierarchy.max_level);
    std::vector<bool> global_level(hierarchy.max_level, false);
    size_t level_buffer_size = static_cast<size_t>(std::max<uint64_t>(
      uint64_t{4} << 10,
      std::min<uint64_t>(
        plan.preferred_buffer_bytes,
        plan.staging_buffer_bytes / std::max<uint64_t>(1, 2 * hierarchy.max_level))));
    for (uint32_t level = 1; level <= hierarchy.max_level; ++level) {
      std::ostringstream sidecar_name;
      sidecar_name << "upper." << level << ".links";
      sidecars[level - 1] =
        std::make_unique<level_file<T>>(workspace,
                                        sidecar_name.str(),
                                        stage_kind::upper_links,
                                        0,
                                        level,
                                        sizeof(uint32_t) + layout.M * sizeof(uint32_t),
                                        fingerprint,
                                        level_buffer_size);
      global_level[level - 1] =
        hierarchy.active_by_level[level - 1] <= plan.global_upper_level_max_rows;
      if (global_level[level - 1]) {
        std::ostringstream vectors_name;
        vectors_name << "upper." << level << ".vectors";
        global_vectors[level - 1] =
          std::make_unique<level_file<T>>(workspace,
                                          vectors_name.str(),
                                          stage_kind::upper_vectors,
                                          static_cast<uint32_t>(dim),
                                          level,
                                          sizeof(uint32_t) + dim * sizeof(T),
                                          fingerprint,
                                          level_buffer_size);
      }
    }

    workspace.update_manifest("partition_build", total_core, total_spill);
    auto partition_start = std::chrono::steady_clock::now();
    struct load_result {
      resident_partition<T> resident;
      external_io_counters io;
      uint64_t elapsed_ms;
    };
    auto load_partition = [&](uint32_t partition) -> load_result {
      auto load_start = std::chrono::steady_clock::now();
      external_io_counters partition_io;
      auto resident = read_resident_partition<T>(stages.core(partition),
                                                 stages.spill(partition),
                                                 partition,
                                                 prefixes,
                                                 stages.record_size(),
                                                 fingerprint,
                                                 plan.preferred_buffer_bytes,
                                                 partition_io);
      return {std::move(resident),
              partition_io,
              elapsed_milliseconds(load_start, std::chrono::steady_clock::now())};
    };
    std::optional<load_result> loaded;
    if (plan.partitions > 0) { loaded.emplace(load_partition(0)); }
    for (uint32_t partition = 0; partition < plan.partitions; ++partition) {
      RAFT_EXPECTS(loaded.has_value(), "External HNSW partition prefetch state is empty");
      std::future<load_result> next;
      if (plan.queue_depth > 1 && partition + 1 < plan.partitions) {
        next = std::async(std::launch::async, load_partition, partition + 1);
      }

      auto current = std::move(*loaded);
      loaded.reset();
      partition_load_ms += current.elapsed_ms;
      io.stage_read_bytes += current.io.stage_read_bytes;
      io.stage_read_requests += current.io.stage_read_requests;
      auto& resident = current.resident;
      if (resident.core_rows != 0) {
        RAFT_EXPECTS(resident.vectors.extent(0) > 1,
                     "External HNSW partition is too small for graph construction");

        cagra::index_params partition_params =
          cagra::index_params::from_hnsw_params(resident.vectors.view().extents(),
                                                params.M,
                                                ace_ef_construction,
                                                cagra::hnsw_heuristic_type::SAME_GRAPH_FOOTPRINT,
                                                params.metric);
        partition_params.graph_degree              = graph_degree;
        partition_params.intermediate_graph_degree = intermediate_graph_degree;
        RAFT_EXPECTS(
          resident.vectors.extent(0) >
            static_cast<int64_t>(partition_params.intermediate_graph_degree),
          "External HNSW partition %u has %zu rows, but at least %zu are required for M=%zu",
          partition,
          static_cast<size_t>(resident.vectors.extent(0)),
          partition_params.intermediate_graph_degree + 1,
          params.M);
        partition_params.attach_dataset_on_build = false;
        auto partition_dataset                   = cuvs::neighbors::make_host_standard_dataset_view(
          raft::make_const_mdspan(resident.vectors.view()));
        auto cagra_start     = std::chrono::steady_clock::now();
        auto partition_index = cagra::build(res, partition_params, partition_dataset);
        cagra_build_ms += elapsed_milliseconds(cagra_start, std::chrono::steady_clock::now());
        auto serialization_start = std::chrono::steady_clock::now();

        std::vector<uint8_t> hierarchy_levels;
        if (params.hierarchy == HnswHierarchy::GPU) {
          hierarchy_levels.resize(resident.local_to_global.size());
#pragma omp parallel for num_threads(num_threads)
          for (int64_t row = 0; row < static_cast<int64_t>(resident.local_to_global.size());
               ++row) {
            auto level =
              level_for_internal_id(resident.local_to_global[row], hnsw_level_seed, layout.M);
            hierarchy_levels[row] = static_cast<uint8_t>(level);
          }
        }

        const uint64_t actual_graph_degree = partition_index.graph().extent(1);
        RAFT_EXPECTS(actual_graph_degree == layout.graph_degree,
                     "External HNSW partition graph degree changed unexpectedly");
        const uint64_t graph_chunk_rows =
          std::max<uint64_t>(1,
                             plan.hnsw_output_buffer_bytes /
                               std::max<uint64_t>(1, actual_graph_degree * sizeof(uint32_t)));
        auto graph_chunk =
          raft::make_host_matrix<uint32_t, int64_t>(graph_chunk_rows, actual_graph_degree);
        auto device_graph_chunk =
          raft::make_device_matrix<uint32_t, int64_t>(res, graph_chunk_rows, actual_graph_degree);
        auto device_mapping =
          raft::make_device_vector<uint32_t, int64_t>(res, resident.local_to_global.size());
        raft::copy(res,
                   device_mapping.view(),
                   raft::make_host_vector_view<const uint32_t, int64_t>(
                     resident.local_to_global.data(), resident.local_to_global.size()));
        for (uint64_t base = 0; base < resident.core_rows; base += graph_chunk_rows) {
          uint64_t count = std::min<uint64_t>(graph_chunk_rows, resident.core_rows - base);
          auto host_view = raft::make_host_matrix_view<uint32_t, int64_t>(
            graph_chunk.data_handle(), count, actual_graph_degree);
          auto source_view = raft::make_device_matrix_view<const uint32_t, int64_t>(
            partition_index.graph().data_handle() + base * actual_graph_degree,
            count,
            actual_graph_degree);
          auto translated_view = raft::make_device_matrix_view<uint32_t, int64_t>(
            device_graph_chunk.data_handle(), count, actual_graph_degree);
          raft::copy(res, translated_view, source_view);
          translate_graph_ids(res,
                              translated_view.data_handle(),
                              translated_view.size(),
                              device_mapping.data_handle(),
                              device_mapping.size());
          raft::copy(res, host_view, raft::make_const_mdspan(translated_view));
          raft::resource::sync_stream(res);
          for (uint64_t local = 0; local < count; ++local) {
            for (uint64_t edge = 0; edge < actual_graph_degree; ++edge) {
              uint32_t neighbor = host_view(local, edge);
              RAFT_EXPECTS(neighbor < rows, "External HNSW partition neighbor is out of range");
            }
            uint64_t core_row = base + local;
            write_hnsw_base_row(output,
                                layout,
                                &host_view(local, 0),
                                &resident.vectors(core_row, 0),
                                resident.core_labels[core_row]);
          }
        }

        for (uint32_t level = 1; level <= hierarchy.max_level; ++level) {
          if (global_level[level - 1]) {
            append_global_level_vectors(
              resident, hierarchy_levels, level, layout, *global_vectors[level - 1]);
          } else {
            write_partitioned_level(
              res, resident, hierarchy_levels, level, layout, params.metric, *sidecars[level - 1]);
          }
        }
        base_serialization_ms +=
          elapsed_milliseconds(serialization_start, std::chrono::steady_clock::now());
      }

      if (partition + 1 < plan.partitions) {
        loaded.emplace(plan.queue_depth > 1 ? next.get() : load_partition(partition + 1));
      }
    }

    auto base_commit_start = std::chrono::steady_clock::now();
    output.flush();
    RAFT_EXPECTS(output.position() == hnswlib_header_size + layout.base_section_size(),
                 "External HNSW base section size mismatch");
    if (::fsync(output_fd.get()) != 0) {
      throw std::runtime_error("failed to commit external HNSW base section");
    }
    base_serialization_ms +=
      elapsed_milliseconds(base_commit_start, std::chrono::steady_clock::now());
    workspace.update_manifest("base_committed", total_core, total_spill, output.position());
    for (uint32_t partition = 0; partition < plan.partitions; ++partition) {
      workspace.remove_consumed(stages.core(partition).path);
      workspace.remove_consumed(stages.spill(partition).path);
    }
    partition_build_ms = elapsed_milliseconds(partition_start, std::chrono::steady_clock::now());

    auto hierarchy_start = std::chrono::steady_clock::now();
    for (uint32_t level = 1; level <= hierarchy.max_level; ++level) {
      if (global_level[level - 1]) {
        build_global_level(res,
                           *global_vectors[level - 1],
                           *sidecars[level - 1],
                           layout,
                           params.metric,
                           plan.preferred_buffer_bytes,
                           io);
        io.sidecar_write_bytes += checked_file_mul(global_vectors[level - 1]->count(),
                                                   global_vectors[level - 1]->header().record_size,
                                                   "upper vector bytes");
      }
      sidecars[level - 1]->finalize();
      RAFT_EXPECTS(sidecars[level - 1]->count() == hierarchy.active_by_level[level - 1],
                   "External HNSW upper sidecar count mismatch");
      io.sidecar_write_bytes += checked_file_mul(sidecars[level - 1]->count(),
                                                 sidecars[level - 1]->header().record_size,
                                                 "upper sidecar bytes");
      io.sidecar_write_requests += sidecars[level - 1]->request_count();
      if (global_level[level - 1]) {
        io.sidecar_write_requests += global_vectors[level - 1]->request_count();
      }
    }

    std::vector<std::unique_ptr<buffered_stage_reader>> sidecar_readers;
    sidecar_readers.reserve(hierarchy.max_level);
    size_t sidecar_reader_buffer = static_cast<size_t>(std::max<uint64_t>(
      uint64_t{4} << 10,
      std::min<uint64_t>(uint64_t{8} << 20,
                         plan.staging_buffer_bytes / std::max<uint64_t>(1, hierarchy.max_level))));
    for (uint32_t level = 1; level <= hierarchy.max_level; ++level) {
      sidecar_readers.push_back(std::make_unique<buffered_stage_reader>(
        sidecars[level - 1]->descriptor(), sidecars[level - 1]->header(), sidecar_reader_buffer));
    }
    std::vector<uint32_t> upper_record(1 + layout.M);
    for (uint64_t row = 0; row < rows; ++row) {
      uint32_t level =
        params.hierarchy == HnswHierarchy::GPU
          ? level_for_internal_id(static_cast<uint32_t>(row), hnsw_level_seed, layout.M)
          : 0;
      write_hnsw_upper_node_header(output, layout, level);
      for (uint32_t current = 1; current <= level; ++current) {
        const std::byte* record = nullptr;
        RAFT_EXPECTS(sidecar_readers[current - 1]->next(record),
                     "External HNSW upper sidecar ended early");
        std::memcpy(upper_record.data(), record, sidecars[current - 1]->header().record_size);
        RAFT_EXPECTS(upper_record[0] == row, "External HNSW upper sidecar is not in base-ID order");
        uint32_t count = 0;
        while (count < layout.M &&
               upper_record[count + 1] != std::numeric_limits<uint32_t>::max()) {
          uint32_t neighbor = upper_record[count + 1];
          RAFT_EXPECTS(neighbor < rows, "External HNSW upper neighbor is out of range");
          RAFT_EXPECTS(level_for_internal_id(neighbor, hnsw_level_seed, layout.M) >= current,
                       "External HNSW upper neighbor is not active at this level");
          ++count;
        }
        write_hnsw_upper_block(output, layout, upper_record.data() + 1, count);
      }
    }
    for (auto& reader : sidecar_readers) {
      io.sidecar_read_bytes += reader->bytes_read();
      io.sidecar_read_requests += reader->read_requests();
    }

    output.flush();
    RAFT_EXPECTS(output.position() == exact_output,
                 "External HNSW output size mismatch: expected %zu, got %zu",
                 static_cast<size_t>(exact_output),
                 static_cast<size_t>(output.position()));
    io.output_write_bytes    = output.position();
    io.output_write_requests = output.request_count();
    hierarchy_ms = elapsed_milliseconds(hierarchy_start, std::chrono::steady_clock::now());

    auto publication_start = std::chrono::steady_clock::now();
    workspace.update_manifest("publishing", total_core, total_spill, exact_output);
    workspace.publish(output_fd);

    hnsw_index->set_file_descriptor(
      cuvs::util::file_descriptor(workspace.final_path().string(), O_RDONLY));
    auto build_end = std::chrono::steady_clock::now();
    publication_ms = elapsed_milliseconds(publication_start, build_end);
    auto elapsed   = elapsed_milliseconds(build_start, build_end);
    RAFT_LOG_INFO(
      "External HNSW build complete in %ld ms: preflight %ld ms, centroid training %ld ms, "
      "assignment/staging %ld ms, partition processing %ld ms (loads %ld ms cumulative, CAGRA "
      "builds %ld ms, base serialization %ld ms), hierarchy %ld ms, publication %ld ms. Logical "
      "I/O: sample %.3f GiB, source %.3f GiB, "
      "stage write/read %.3f/%.3f GiB, sidecar write/read %.3f/%.3f GiB, output %.3f GiB; "
      "KvikIO requests stage write/read %zu/%zu, output %zu",
      elapsed,
      preflight_ms,
      centroid_training_ms,
      assignment_stage_ms,
      partition_build_ms,
      partition_load_ms,
      cagra_build_ms,
      base_serialization_ms,
      hierarchy_ms,
      publication_ms,
      io.sample_read_bytes / static_cast<double>(uint64_t{1} << 30),
      io.source_read_bytes / static_cast<double>(uint64_t{1} << 30),
      io.stage_write_bytes / static_cast<double>(uint64_t{1} << 30),
      io.stage_read_bytes / static_cast<double>(uint64_t{1} << 30),
      io.sidecar_write_bytes / static_cast<double>(uint64_t{1} << 30),
      io.sidecar_read_bytes / static_cast<double>(uint64_t{1} << 30),
      io.output_write_bytes / static_cast<double>(uint64_t{1} << 30),
      static_cast<size_t>(io.stage_write_requests),
      static_cast<size_t>(io.stage_read_requests),
      static_cast<size_t>(io.output_write_requests));
    return hnsw_index;
  } catch (const std::exception& error) {
    workspace.mark_failed("failed", error.what());
    throw;
  } catch (...) {
    workspace.mark_failed("failed", "unknown exception");
    throw;
  }
}

}  // namespace cuvs::neighbors::hnsw::detail::external
