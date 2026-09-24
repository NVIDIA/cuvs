/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "external_format.hpp"

#include <cuvs/util/file_io.hpp>

#include <raft/core/error.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace cuvs::neighbors::hnsw::detail::external {

class external_workspace {
 public:
  explicit external_workspace(std::filesystem::path build_dir, uint64_t parameter_fingerprint)
    : build_dir_(std::move(build_dir)), fingerprint_(parameter_fingerprint)
  {
    RAFT_EXPECTS(!build_dir_.empty(), "ACE build_dir must not be empty");
    std::error_code error;
    if (!std::filesystem::exists(build_dir_, error)) {
      RAFT_EXPECTS(std::filesystem::create_directories(build_dir_, error) && !error,
                   "failed to create external HNSW build directory: %s",
                   error.message().c_str());
    }
    RAFT_EXPECTS(std::filesystem::is_directory(build_dir_, error) && !error,
                 "external HNSW build_dir is not a directory");

    final_path_ = build_dir_ / "hnsw_index.bin";
    RAFT_EXPECTS(!std::filesystem::exists(final_path_),
                 "refusing to overwrite existing HNSW index: %s",
                 final_path_.c_str());

    static std::atomic<uint64_t> sequence{0};
    for (int attempt = 0; attempt < 100; ++attempt) {
      auto value = sequence.fetch_add(1, std::memory_order_relaxed);
      std::ostringstream name;
      name << ".hnsw-external-" << static_cast<unsigned long>(::getpid()) << "-" << value;
      invocation_dir_ = build_dir_ / name.str();
      if (::mkdir(invocation_dir_.c_str(), 0700) == 0) { break; }
      if (errno != EEXIST) {
        RAFT_FAIL("failed to create external HNSW workspace: %s", strerror(errno));
      }
      invocation_dir_.clear();
    }
    RAFT_EXPECTS(!invocation_dir_.empty(), "failed to create a unique external HNSW workspace");

    manifest_path_ = invocation_dir_ / "manifest.json";
    manifest_fd_ =
      cuvs::util::file_descriptor(manifest_path_.string(), O_CREAT | O_EXCL | O_RDWR, 0600);
    partial_path_ = invocation_dir_ / "hnsw_index.bin.partial";
    update_manifest("created");
  }

  external_workspace(const external_workspace&)            = delete;
  external_workspace& operator=(const external_workspace&) = delete;
  external_workspace(external_workspace&&)                 = delete;
  external_workspace& operator=(external_workspace&&)      = delete;

  // Always drop the private staging directory, including after a failed build.
  // A successfully published hnsw_index.bin is not owned here and is left in place.
  ~external_workspace() noexcept { cleanup_noexcept(); }

  [[nodiscard]] const std::filesystem::path& invocation_dir() const noexcept
  {
    return invocation_dir_;
  }
  [[nodiscard]] const std::filesystem::path& final_path() const noexcept { return final_path_; }

  [[nodiscard]] std::filesystem::path private_path(std::string_view filename) const
  {
    RAFT_EXPECTS(
      !(filename.empty() || filename.find('/') != std::string_view::npos ||
        filename.find('\\') != std::string_view::npos || filename == "." || filename == ".."),
      "invalid external HNSW private filename");
    return invocation_dir_ / filename;
  }

  cuvs::util::file_descriptor create_private_file(std::string_view filename, mode_t mode = 0600)
  {
    auto path = private_path(filename);
    cuvs::util::file_descriptor fd(path.string(), O_CREAT | O_EXCL | O_RDWR, mode);
    owned_files_.push_back(path);
    return fd;
  }

  template <typename T>
  cuvs::util::file_descriptor create_stage_file(std::string_view filename,
                                                stage_kind kind,
                                                uint32_t dimension,
                                                uint32_t partition,
                                                uint32_t level,
                                                uint64_t record_size)
  {
    auto fd = create_private_file(filename);
    auto header =
      make_stage_header<T>(kind, dimension, partition, level, record_size, fingerprint_);
    RAFT_EXPECTS(::ftruncate(fd.get(), static_cast<off_t>(stage_data_offset)) == 0,
                 "failed to initialize external HNSW stage file: %s",
                 strerror(errno));
    write_stage_header(fd.get(), header);
    return fd;
  }

  cuvs::util::file_descriptor create_partial(uint64_t exact_size)
  {
    auto fd = create_private_file("hnsw_index.bin.partial");
    RAFT_EXPECTS(exact_size <= static_cast<uint64_t>(std::numeric_limits<off_t>::max()),
                 "external HNSW output exceeds off_t");
    int status = ::posix_fallocate(fd.get(), 0, static_cast<off_t>(exact_size));
    RAFT_EXPECTS(status == 0, "failed to preallocate external HNSW output: %s", strerror(status));
    return fd;
  }

  void update_manifest(std::string_view phase,
                       uint64_t core_records           = 0,
                       uint64_t spill_records          = 0,
                       uint64_t output_bytes           = 0,
                       std::string_view failure_reason = {})
  {
    std::ostringstream json;
    json << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"parameter_fingerprint\": " << fingerprint_ << ",\n"
         << "  \"phase\": \"" << json_escape(phase) << "\",\n"
         << "  \"core_records\": " << core_records << ",\n"
         << "  \"spill_records\": " << spill_records << ",\n"
         << "  \"output_bytes\": " << output_bytes << ",\n"
         << "  \"failure_reason\": \"" << json_escape(failure_reason) << "\"\n"
         << "}\n";
    auto contents = json.str();
    RAFT_EXPECTS(::ftruncate(manifest_fd_.get(), 0) == 0,
                 "failed to truncate external HNSW manifest");
    pwrite_all(manifest_fd_.get(), contents.data(), contents.size(), 0);
    RAFT_EXPECTS(::fsync(manifest_fd_.get()) == 0, "failed to sync external HNSW manifest");
  }

  // Best-effort note before the destructor removes the private workspace.
  void mark_failed(std::string_view phase, std::string_view reason) noexcept
  {
    try {
      update_manifest(phase, 0, 0, 0, reason);
    } catch (...) {
    }
  }

  void remove_consumed(const std::filesystem::path& path)
  {
    auto owned = std::find(owned_files_.begin(), owned_files_.end(), path);
    RAFT_EXPECTS(owned != owned_files_.end(),
                 "refusing to remove a file not owned by this HNSW workspace");
    std::error_code error;
    RAFT_EXPECTS(std::filesystem::remove(path, error) && !error,
                 "failed to remove consumed external HNSW stage: %s",
                 error.message().c_str());
    owned_files_.erase(owned);
  }

  void publish(cuvs::util::file_descriptor& partial_fd)
  {
    RAFT_EXPECTS(
      ::fsync(partial_fd.get()) == 0, "failed to sync external HNSW output: %s", strerror(errno));
    partial_fd.close();

    // link(2) provides no-replace atomic publication on the same filesystem. Removing the private
    // name afterwards leaves the complete inode reachable only through hnsw_index.bin.
    if (::link(partial_path_.c_str(), final_path_.c_str()) != 0) {
      if (errno == EEXIST) {
        RAFT_FAIL("refusing to overwrite an HNSW index published concurrently");
      }
      RAFT_FAIL("failed to publish external HNSW index: %s", strerror(errno));
    }
    if (::unlink(partial_path_.c_str()) == 0) {
      auto partial = std::find(owned_files_.begin(), owned_files_.end(), partial_path_);
      if (partial != owned_files_.end()) { owned_files_.erase(partial); }
    }
    sync_directory(build_dir_);
  }

 private:
  static std::string json_escape(std::string_view value)
  {
    std::string out;
    out.reserve(value.size());
    for (char character : value) {
      switch (character) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += static_cast<unsigned char>(character) < 0x20 ? '?' : character;
      }
    }
    return out;
  }

  static void sync_directory(const std::filesystem::path& path)
  {
#ifdef O_DIRECTORY
    int descriptor = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
#else
    int descriptor = ::open(path.c_str(), O_RDONLY);
#endif
    RAFT_EXPECTS(descriptor >= 0, "failed to open directory for sync: %s", strerror(errno));
    int result = ::fsync(descriptor);
    int saved  = errno;
    ::close(descriptor);
    RAFT_EXPECTS(result == 0 || saved == EINVAL, "failed to sync directory: %s", strerror(saved));
  }

  // Drops private staging files and the invocation directory. Does not remove a published
  // hnsw_index.bin or unrelated files already in build_dir_.
  void cleanup_noexcept() noexcept
  {
    manifest_fd_.close();
    std::error_code error;
    for (const auto& path : owned_files_) {
      std::filesystem::remove(path, error);
      error.clear();
    }
    std::filesystem::remove(manifest_path_, error);
    error.clear();
    std::filesystem::remove(invocation_dir_, error);
  }

  std::filesystem::path build_dir_;
  std::filesystem::path invocation_dir_;
  std::filesystem::path manifest_path_;
  std::filesystem::path partial_path_;
  std::filesystem::path final_path_;
  uint64_t fingerprint_;
  cuvs::util::file_descriptor manifest_fd_;
  std::vector<std::filesystem::path> owned_files_;
};

}  // namespace cuvs::neighbors::hnsw::detail::external
