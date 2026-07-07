/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <limits>
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
 * @brief Alignment (in bytes) for the numpy data body in on-disk ACE artifacts.
 *
 * The numpy header is space-padded so that the array data begins on this boundary. This keeps the
 * file numpy-compatible (readers use the stored HEADER_LEN) while letting kvikio's O_DIRECT /
 * GPUDirect Storage path transfer the aligned interior directly. 4096 ensures support for most
 * storage devices.
 */
inline constexpr size_t kNumpyDataAlignment = 4096;

/** @brief Round @p x up to the next multiple of @p alignment (must be a power of two). */
inline constexpr size_t numpy_align_up(size_t x, size_t alignment = kNumpyDataAlignment) noexcept
{
  return (x + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Streambuf that reads from a POSIX file descriptor
 */
class fd_streambuf : public std::streambuf {
  int fd_ = -1;
  std::unique_ptr<char[]> buffer_;
  size_t buffer_size_ = 0;

 protected:
  int_type underflow() override
  {
    if (gptr() < egptr()) { return traits_type::to_int_type(*gptr()); }
    if (fd_ == -1 || !buffer_) { return traits_type::eof(); }

    ssize_t n = 0;
    do {
      n = ::read(fd_, buffer_.get(), buffer_size_);
    } while (n < 0 && errno == EINTR);
    if (n <= 0) return traits_type::eof();
    setg(buffer_.get(), buffer_.get(), buffer_.get() + n);
    return traits_type::to_int_type(*gptr());
  }

 public:
  explicit fd_streambuf(int fd, size_t buffer_size = 8192)
    : fd_(fd), buffer_(new char[buffer_size]), buffer_size_(buffer_size)
  {
    RAFT_EXPECTS(buffer_size > 0, "fd_streambuf buffer size must be greater than zero");
    setg(buffer_.get(), buffer_.get(), buffer_.get());
  }

  ~fd_streambuf() override { close(); }

  fd_streambuf(const fd_streambuf&)            = delete;
  fd_streambuf& operator=(const fd_streambuf&) = delete;

  fd_streambuf(fd_streambuf&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      buffer_(std::move(other.buffer_)),
      buffer_size_(std::exchange(other.buffer_size_, 0))
  {
    setg(buffer_.get(), buffer_.get(), buffer_.get());
  }

  fd_streambuf& operator=(fd_streambuf&& other) noexcept
  {
    if (this != &other) {
      close();
      fd_          = std::exchange(other.fd_, -1);
      buffer_      = std::move(other.buffer_);
      buffer_size_ = std::exchange(other.buffer_size_, 0);
      setg(buffer_.get(), buffer_.get(), buffer_.get());
    }
    return *this;
  }

 private:
  void close() noexcept
  {
    if (fd_ != -1) {
      ::close(fd_);
      fd_ = -1;
    }
  }
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
 * Manages file descriptor lifecycle with automatic cleanup. Used to own the lifetime of disk-backed
 * ACE artifacts and to parse their numpy headers; the bulk data transfers go through kvikio (see
 * ::read_large_file / ::write_large_file). Non-copyable, move-only.
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
 * Opens a file, writes a numpy header for the given shape/dtype, and pre-allocates space for the
 * data. The numpy header is space-padded so that the data body begins on a ::kNumpyDataAlignment
 * boundary, which keeps the file numpy-compatible (readers use the stored HEADER_LEN) while letting
 * kvikio transfer the aligned interior via O_DIRECT / GPUDirect Storage. The data region is rounded
 * up to a block boundary so that an O_DIRECT write of the trailing block stays within the
 * allocated file. The returned descriptor owns the file lifetime; bulk data is written separately
 * through kvikio (::write_large_file).
 *
 * @tparam T Data type for the numpy array
 * @param path File path to create
 * @param shape Shape of the numpy array (e.g., {rows, cols} for 2D)
 * @return Pair of (file_descriptor, header_size) where header_size is the data offset in bytes
 */
template <typename T>
std::pair<file_descriptor, size_t> create_numpy_file(const std::string& path,
                                                     const std::vector<size_t>& shape)
{
  // Open the file for the header write + preallocation. Bulk data is written via kvikio (which
  // opens its own descriptor, including an O_DIRECT one when supported).
  file_descriptor fd(path, O_CREAT | O_RDWR | O_TRUNC, 0644);

  // Build header
  const auto dtype                              = raft::numpy_serializer::get_numpy_dtype<T>();
  const bool fortran_order                      = false;
  const raft::numpy_serializer::header_t header = {dtype, fortran_order, shape};

  std::stringstream ss;
  raft::numpy_serializer::write_header(ss, header);
  std::string header_str = ss.str();

  RAFT_EXPECTS((kNumpyDataAlignment & (kNumpyDataAlignment - 1)) == 0,
               "kNumpyDataAlignment must be a power of two");

  // Re-pad the numpy v1.0 header so the data body starts on a block boundary.
  // Layout: [6 bytes magic][2 bytes version][2 bytes HEADER_LEN][dict padded to HEADER_LEN].
  RAFT_EXPECTS(header_str.size() >= 10 && static_cast<unsigned char>(header_str[6]) == 0x01,
               "Expected a numpy v1.0 header to align the data body");
  std::string dict = header_str.substr(10);
  while (!dict.empty() && (dict.back() == '\n' || dict.back() == ' ')) {
    dict.pop_back();
  }
  const size_t min_total     = 10 + dict.size() + 1;  // +1 for the terminating newline
  const size_t aligned_total = numpy_align_up(min_total);
  const size_t header_len    = aligned_total - 10;
  RAFT_EXPECTS(header_len <= 0xFFFF, "Aligned numpy v1.0 header is too large");

  std::string padded_dict = dict;
  padded_dict.append(header_len - dict.size() - 1, ' ');
  padded_dict.push_back('\n');

  std::string aligned_header = header_str.substr(0, 8);  // magic + version
  aligned_header.push_back(static_cast<char>(header_len & 0xFF));
  aligned_header.push_back(static_cast<char>((header_len >> 8) & 0xFF));
  aligned_header.append(padded_dict);
  header_str = std::move(aligned_header);

  const size_t header_size = header_str.size();

  // Calculate data size from shape
  auto checked_mul = [](size_t lhs, size_t rhs) {
    RAFT_EXPECTS(rhs == 0 || lhs <= std::numeric_limits<size_t>::max() / rhs,
                 "Numpy file size calculation overflowed");
    return lhs * rhs;
  };
  size_t data_bytes = sizeof(T);
  for (auto dim : shape) {
    data_bytes = checked_mul(data_bytes, dim);
  }

  // Pre-allocate file space. The data region is rounded up to a block boundary so read-modify-write
  // of the trailing block during an O_DIRECT write stays within the allocated file.
  const size_t padded_data_bytes = numpy_align_up(data_bytes);
  RAFT_EXPECTS(header_size <= std::numeric_limits<size_t>::max() - padded_data_bytes,
               "Numpy file size calculation overflowed");
  const size_t alloc_bytes   = header_size + padded_data_bytes;
  const int fallocate_status = posix_fallocate(fd.get(), 0, alloc_bytes);
  if (fallocate_status != 0) {
    RAFT_FAIL(
      "Failed to pre-allocate space for file %s: %s", path.c_str(), strerror(fallocate_status));
  }

  // Write the small numpy header (one-time, buffered).
  const char* hp    = header_str.data();
  size_t hremaining = header_size;
  off_t hoff        = 0;
  while (hremaining > 0) {
    const size_t chunk = std::min(hremaining, static_cast<size_t>(SSIZE_MAX));
    const ssize_t w    = ::pwrite(fd.get(), hp, chunk, hoff);
    if (w < 0 && errno == EINTR) { continue; }
    RAFT_EXPECTS(w > 0, "Failed to write numpy header to %s: %s", path.c_str(), strerror(errno));
    hp += w;
    hoff += w;
    hremaining -= static_cast<size_t>(w);
  }

  return {std::move(fd), header_size};
}

/**
 * @brief Read a region of a file into @p dest_ptr through kvikio.
 *
 * Routed through kvikio::FileHandle, so it uses GPUDirect Storage (cuFile) when @p dest_ptr is
 * device memory on a GDS-capable system, and the POSIX + threadpool backend (with O_DIRECT when
 * available) otherwise. @p dest_ptr may be host or device memory.
 *
 * @param fd          File descriptor identifying the file (its path is used to open the handle).
 * @param dest_ptr    Destination buffer (host or device).
 * @param total_bytes Total bytes to read.
 * @param file_offset Starting offset in file.
 */
void read_large_file(const file_descriptor& fd,
                     void* dest_ptr,
                     const size_t total_bytes,
                     const uint64_t file_offset);

/**
 * @brief Write @p data_ptr to a region of a (pre-created) file through kvikio.
 *
 * Counterpart of ::read_large_file. The file must already exist (e.g. created by
 * ::create_numpy_file); the handle is opened in read+write mode so the existing header and
 * preallocation are preserved. @p data_ptr may be host or device memory.
 *
 * @param fd          File descriptor identifying the file (its path is used to open the handle).
 * @param data_ptr    Source data buffer (host or device).
 * @param total_bytes Total bytes to write.
 * @param file_offset Starting offset in file.
 */
void write_large_file(const file_descriptor& fd,
                      const void* data_ptr,
                      const size_t total_bytes,
                      const uint64_t file_offset);

/**
 * @brief Sequential std::ostream backed by kvikio.
 *
 * Ordinary stream output is staged into a large host buffer and written to disk through kvikio,
 * which bypasses the page cache via O_DIRECT when supported (and falls back to buffered POSIX
 * writes otherwise). Device buffers passed to write_device() use GPUDirect Storage when available.
 * This can be passed to APIs accepting a std::ostream& for sequential output (e.g. the hnswlib
 * serializer). It supports querying the current output position, but not random-access seeking.
 * std::fstream-specific APIs such as is_open() are not part of its interface. Non-copyable,
 * non-movable.
 */
class kvikio_ofstream : public std::ostream {
 public:
  /**
   * @brief Open @p path for writing (created/truncated).
   *
   * @param path        Output file path.
   * @param buffer_size Staging-buffer capacity in bytes; full buffers are written at aligned
   *                    offsets.
   */
  explicit kvikio_ofstream(const std::string& path, size_t buffer_size = (size_t(32) << 20));

  ~kvikio_ofstream() override;

  kvikio_ofstream(const kvikio_ofstream&)            = delete;
  kvikio_ofstream& operator=(const kvikio_ofstream&) = delete;
  kvikio_ofstream(kvikio_ofstream&&)                 = delete;
  kvikio_ofstream& operator=(kvikio_ofstream&&)      = delete;

  /** @brief Flush any remaining staged bytes and close the file. */
  void close();

  /**
   * @brief Write a device-memory region at the current stream position.
   *
   * Pending host output is flushed first, then @p data is written through a compatibility-off
   * KvikIO handle so GPUDirect Storage is used when available. The operation is synchronous and
   * advances the same logical position used by ordinary std::ostream writes.
   *
   * @param data Device pointer to the source data.
   * @param size Number of bytes to write.
   * @return This stream.
   */
  kvikio_ofstream& write_device(const void* data, size_t size);

  /** @brief Total number of logical bytes written so far. */
  [[nodiscard]] size_t bytes_written() const noexcept;

 private:
  class sbuf;
  std::unique_ptr<sbuf> buf_;
};

/**
 * @brief Buffered output stream wrapper
 *
 * Wraps an std::ostream with a buffer to improve write performance by reducing the number of
 * system calls. Automatically flushes on destruction. Used by the hnswlib serializer. Non-copyable,
 * non-movable.
 */
class buffered_ofstream {
 public:
  buffered_ofstream(std::ostream* os, size_t buffer_size) : buffer_(buffer_size), os_(os), pos_(0)
  {
  }

  ~buffered_ofstream() noexcept
  {
    try {
      flush();
    } catch (...) {
    }
  }

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
