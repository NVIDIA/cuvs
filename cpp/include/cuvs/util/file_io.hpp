/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/serialize.hpp>

#include <algorithm>
#include <cstring>
#include <istream>
#include <limits.h>
#include <memory>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include <cuvs/core/export.hpp>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

namespace CUVS_EXPORT cuvs {
namespace util {
/**
 * @brief Streambuf that reads from a POSIX file descriptor
 */
class fd_streambuf : public std::streambuf {
  int fd_;
  std::unique_ptr<char[]> buffer_;
  size_t buffer_size_;

 protected:
  int_type underflow() override
  {
    if (gptr() < egptr()) return traits_type::to_int_type(*gptr());
    ssize_t n = ::read(fd_, buffer_.get(), buffer_size_);
    if (n <= 0) return traits_type::eof();
    setg(buffer_.get(), buffer_.get(), buffer_.get() + n);
    return traits_type::to_int_type(*gptr());
  }

 public:
  explicit fd_streambuf(int fd, size_t buffer_size = 8192)
    : fd_(fd), buffer_(new char[buffer_size]), buffer_size_(buffer_size)
  {
    setg(buffer_.get(), buffer_.get(), buffer_.get());
  }

  ~fd_streambuf()
  {
    if (fd_ != -1) ::close(fd_);
  }

  fd_streambuf(const fd_streambuf&)                = delete;
  fd_streambuf& operator=(const fd_streambuf&)     = delete;
  fd_streambuf(fd_streambuf&&) noexcept            = default;
  fd_streambuf& operator=(fd_streambuf&&) noexcept = default;
};

/**
 * @brief Istream that reads from a POSIX file descriptor
 */
class fd_istream : public std::istream {
  fd_streambuf buf_;

 public:
  explicit fd_istream(int fd) : std::istream(&buf_), buf_(fd) {}

  fd_istream(const fd_istream&)            = delete;
  fd_istream& operator=(const fd_istream&) = delete;

  fd_istream(fd_istream&& o) noexcept : std::istream(std::move(o)), buf_(std::move(o.buf_))
  {
    rdbuf(&buf_);
  }

  fd_istream& operator=(fd_istream&& o) noexcept
  {
    std::istream::operator=(std::move(o));
    buf_ = std::move(o.buf_);
    rdbuf(&buf_);
    return *this;
  }
};

/**
 * @brief RAII wrapper for POSIX file descriptors
 *
 * Manages file descriptor lifecycle with automatic cleanup.
 * Non-copyable, move-only.
 */
class file_descriptor {
 public:
  explicit file_descriptor(int fd = -1) : fd_(fd) {}

  file_descriptor(const std::string& path, int flags, mode_t mode = 0644)
    : fd_(open(path.c_str(), flags, mode)), path_(path)
  {
    if (fd_ == -1) {
      RAFT_FAIL("Failed to open file: %s (errno: %d, %s)", path.c_str(), errno, strerror(errno));
    }
  }

  file_descriptor(const file_descriptor&)            = delete;
  file_descriptor& operator=(const file_descriptor&) = delete;

  file_descriptor(file_descriptor&& other) noexcept
    : fd_{std::exchange(other.fd_, -1)}, path_{std::move(other.path_)}
  {
  }

  file_descriptor& operator=(file_descriptor&& other) noexcept
  {
    std::swap(this->fd_, other.fd_);
    std::swap(this->path_, other.path_);
    return *this;
  }

  ~file_descriptor() noexcept { close(); }

  [[nodiscard]] int get() const noexcept { return fd_; }
  [[nodiscard]] bool is_valid() const noexcept { return fd_ != -1; }

  void close() noexcept
  {
    if (fd_ != -1) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  [[nodiscard]] int release() noexcept
  {
    const int fd = fd_;
    fd_          = -1;
    return fd;
  }

  [[nodiscard]] std::string get_path() const { return path_; }

  /**
   * @brief Create an input stream from this file descriptor
   *
   * Creates an istream that reads directly from the file descriptor using POSIX read().
   * The original descriptor remains valid and unchanged (we duplicate it internally).
   * Returns the stream by value (uses move semantics).
   *
   * @return fd_istream (movable istream)
   */
  [[nodiscard]] fd_istream make_istream() const
  {
    RAFT_EXPECTS(is_valid(), "Invalid file descriptor");

    // Duplicate the fd to avoid consuming the original
    int dup_fd = dup(fd_);
    RAFT_EXPECTS(dup_fd != -1, "Failed to duplicate file descriptor");

    // Create stream that owns the duplicated fd
    // Returned by value, uses move semantics
    return fd_istream(dup_fd);
  }

 private:
  int fd_;
  std::string path_;
};

/**
 * @brief Create a numpy file with pre-allocated space and write the header.
 *
 * Opens a file, writes a numpy header for the given shape/dtype, and pre-allocates
 * space for the data. This is useful for memory-mapped or streaming writes.
 *
 * @tparam T Data type for the numpy array
 * @param path File path to create
 * @param shape Shape of the numpy array (e.g., {rows, cols} for 2D)
 * @return Pair of (file_descriptor, header_size)
 */
template <typename T>
std::pair<file_descriptor, size_t> create_numpy_file(const std::string& path,
                                                     const std::vector<size_t>& shape)
{
  // Open file
  file_descriptor fd(path, O_CREAT | O_RDWR | O_TRUNC, 0644);

  // Build header
  const auto dtype                              = raft::numpy_serializer::get_numpy_dtype<T>();
  const bool fortran_order                      = false;
  const raft::numpy_serializer::header_t header = {dtype, fortran_order, shape};

  std::stringstream ss;
  raft::numpy_serializer::write_header(ss, header);
  std::string header_str = ss.str();
  size_t header_size     = header_str.size();

  // Calculate data size from shape
  size_t data_bytes = sizeof(T);
  for (auto dim : shape) {
    data_bytes *= dim;
  }

  // Pre-allocate file space
  if (posix_fallocate(fd.get(), 0, header_size + data_bytes) != 0) {
    RAFT_FAIL("Failed to pre-allocate space for file: %s", path.c_str());
  }

  // Seek to beginning and write header
  if (lseek(fd.get(), 0, SEEK_SET) == -1) {
    RAFT_FAIL("Failed to seek to beginning of file: %s", path.c_str());
  }

  ssize_t written = write(fd.get(), header_str.data(), header_str.size());
  if (written < 0 || static_cast<size_t>(written) != header_str.size()) {
    RAFT_FAIL("Failed to write numpy header to file: %s", path.c_str());
  }

  return {std::move(fd), header_size};
}

/**
 * @brief Read large file in chunks using pread
 *
 * Reads a file in chunks to avoid issues with very large reads.
 * Uses pread for thread-safe, offset-based reading.
 *
 * @param fd File descriptor to read from
 * @param dest_ptr Destination buffer
 * @param total_bytes Total bytes to read
 * @param file_offset Starting offset in file
 */
void read_large_file(const file_descriptor& fd,
                     void* dest_ptr,
                     const size_t total_bytes,
                     const uint64_t file_offset);

/**
 * @brief Write large file in chunks using pwrite
 *
 * Writes data to a file in chunks to avoid issues with very large writes.
 * Uses pwrite for thread-safe, offset-based writing.
 *
 * @param fd File descriptor to write to
 * @param data_ptr Source data buffer
 * @param total_bytes Total bytes to write
 * @param file_offset Starting offset in file
 */
void write_large_file(const file_descriptor& fd,
                      const void* data_ptr,
                      const size_t total_bytes,
                      const uint64_t file_offset);

/**
 * @brief Buffered output stream wrapper
 *
 * Wraps an std::ostream with a buffer to improve write performance by
 * reducing the number of system calls. Automatically flushes on destruction.
 * Non-copyable, non-movable.
 */
class buffered_ofstream {
 public:
  buffered_ofstream(std::ostream* os, size_t buffer_size) : os_(os), buffer_(buffer_size), pos_(0)
  {
  }

  ~buffered_ofstream() noexcept { flush(); }

  buffered_ofstream(const buffered_ofstream& res)                      = delete;
  auto operator=(const buffered_ofstream& other) -> buffered_ofstream& = delete;
  buffered_ofstream(buffered_ofstream&& other)                         = delete;
  auto operator=(buffered_ofstream&& other) -> buffered_ofstream&      = delete;

  void flush()
  {
    if (pos_ > 0) {
      os_->write(reinterpret_cast<char*>(&buffer_.front()), pos_);
      if (!os_->good()) { RAFT_FAIL("Error writing HNSW file!"); }
      pos_ = 0;
    }
  }

  void write(const char* input, size_t size)
  {
    if (size >= buffer_.size()) {
      flush();
      os_->write(input, static_cast<std::streamsize>(size));
      if (!os_->good()) { RAFT_FAIL("Error writing HNSW file!"); }
      return;
    } else {
      if (size > buffer_.size() - pos_) { flush(); }
      std::memcpy(buffer_.data() + pos_, input, size);
      pos_ += size;
    }
  }

 private:
  std::vector<char> buffer_;
  std::ostream* os_;
  size_t pos_;
};

}  // namespace util
}  // namespace CUVS_EXPORT cuvs
