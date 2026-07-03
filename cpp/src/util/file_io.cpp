/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/util/file_io.hpp>

#include "kvikio_io.hpp"

#include <kvikio/file_handle.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sys/stat.h>
#include <vector>

namespace cuvs::util {
namespace {

struct file_identity {
  dev_t device;
  ino_t inode;
};

file_identity get_file_identity(int fd, const char* description)
{
  struct stat status{};
  RAFT_EXPECTS(::fstat(fd, &status) == 0,
               "Failed to stat %s file descriptor: %s",
               description,
               std::strerror(errno));
  return {status.st_dev, status.st_ino};
}

void expect_matching_file_identity(file_identity expected,
                                   int actual_fd,
                                   const char* actual_description,
                                   const std::string& path)
{
  RAFT_EXPECTS(actual_fd >= 0, "kvikio did not open %s file descriptor", actual_description);

  const auto actual = get_file_identity(actual_fd, actual_description);
  RAFT_EXPECTS(actual.device == expected.device && actual.inode == expected.inode,
               "File path changed while opening %s for kvikio I/O",
               path.c_str());
}

void validate_kvikio_handle_matches_fd(const file_descriptor& fd,
                                       const kvikio::FileHandle& handle,
                                       const std::string& path)
{
  const auto expected = get_file_identity(fd.get(), "source");

  const int buffered_fd = handle.fd(false);
  expect_matching_file_identity(expected, buffered_fd, "kvikio buffered", path);

  const int direct_fd = handle.fd(true);
  if (direct_fd >= 0 && direct_fd != buffered_fd) {
    expect_matching_file_identity(expected, direct_fd, "kvikio direct", path);
  }
}

void validate_kvikio_handles_match(const kvikio::FileHandle& expected,
                                   const kvikio::FileHandle& actual,
                                   const std::string& path)
{
  const auto expected_identity = get_file_identity(expected.fd(false), "stream");

  const int buffered_fd = actual.fd(false);
  expect_matching_file_identity(expected_identity, buffered_fd, "kvikio buffered", path);

  const int direct_fd = actual.fd(true);
  if (direct_fd >= 0 && direct_fd != buffered_fd) {
    expect_matching_file_identity(expected_identity, direct_fd, "kvikio direct", path);
  }
}

}  // namespace

void read_large_file(const file_descriptor& fd,
                     void* dest_ptr,
                     const size_t total_bytes,
                     const uint64_t file_offset)
{
  RAFT_EXPECTS(total_bytes > 0, "Total bytes must be greater than 0");
  RAFT_EXPECTS(dest_ptr != nullptr, "Destination pointer must not be nullptr");
  RAFT_EXPECTS(fd.is_valid(), "File descriptor must be valid");
  const std::string path = fd.get_path();
  RAFT_EXPECTS(!path.empty(), "File descriptor must have an associated path for kvikio I/O");

  // kvikio selects GPUDirect Storage (cuFile) for device destinations on a GDS-capable system, and
  // the POSIX + threadpool backend (with O_DIRECT when available) otherwise. The destination may be
  // host or device memory; kvikio detects which.
  auto handle = detail::open_kvikio_file_for_ace_io(path, "r", dest_ptr);
  validate_kvikio_handle_matches_fd(fd, handle, path);
  const size_t bytes_read = handle.pread(dest_ptr, total_bytes, file_offset).get();
  RAFT_EXPECTS(bytes_read == total_bytes,
               "Incomplete read from %s: expected %zu bytes, got %zu",
               path.c_str(),
               total_bytes,
               bytes_read);
}

void write_large_file(const file_descriptor& fd,
                      const void* data_ptr,
                      const size_t total_bytes,
                      const uint64_t file_offset)
{
  RAFT_EXPECTS(total_bytes > 0, "Total bytes must be greater than 0");
  RAFT_EXPECTS(data_ptr != nullptr, "Data pointer must not be nullptr");
  RAFT_EXPECTS(fd.is_valid(), "File descriptor must be valid");
  const std::string path = fd.get_path();
  RAFT_EXPECTS(!path.empty(), "File descriptor must have an associated path for kvikio I/O");

  // Open in read+write mode ("r+") so the existing numpy header and preallocation are preserved
  // (kvikio's "w" mode would truncate). The source may be host or device memory.
  auto handle = detail::open_kvikio_file_for_ace_io(path, "r+", data_ptr);
  validate_kvikio_handle_matches_fd(fd, handle, path);
  const size_t bytes_written = handle.pwrite(data_ptr, total_bytes, file_offset).get();
  RAFT_EXPECTS(bytes_written == total_bytes,
               "Incomplete write to %s: expected %zu bytes, wrote %zu",
               path.c_str(),
               total_bytes,
               bytes_written);
}

// std::streambuf that stages output into a large buffer and writes full buffers to disk through
// kvikio at an increasing file offset. The trailing partial buffer is written on sync()/close().
class kvikio_ofstream::sbuf : public std::streambuf {
 public:
  sbuf(const std::string& path, size_t cap)
    : path_(path), handle_(path, "w"), buffer_(std::max<size_t>(cap, kNumpyDataAlignment))
  {
    RAFT_EXPECTS(buffer_.size() <= static_cast<size_t>(std::numeric_limits<int>::max()),
                 "kvikio_ofstream buffer size must fit in std::streambuf::pbump");
    setp(buffer_.data(), buffer_.data() + buffer_.size());
  }

  ~sbuf() override
  {
    try {
      close();
    } catch (...) {
      // Swallow during destruction.
    }
  }

  void close()
  {
    if (closed_) { return; }
    flush_buffer();
    if (device_handle_) { device_handle_->close(); }
    handle_.close();
    closed_ = true;
  }

  void write_device(const void* data, size_t n)
  {
    RAFT_EXPECTS(!closed_, "kvikio_ofstream: write attempted after close");
    RAFT_EXPECTS(data != nullptr || n == 0, "kvikio_ofstream: device pointer must not be null");
    if (n == 0) { return; }

    flush_buffer();
    if (!detail::is_kvikio_device_memory(data)) {
      write_at_current_offset(handle_, data, n);
      return;
    }

    if (!device_handle_) {
      device_handle_ =
        std::make_unique<kvikio::FileHandle>(detail::open_kvikio_file_for_device_io(path_, "r+"));
      validate_kvikio_handles_match(handle_, *device_handle_, path_);
    }
    write_at_current_offset(*device_handle_, data, n);
  }

  // Total logical bytes accepted = already-written + currently-staged.
  [[nodiscard]] size_t bytes_written() const noexcept
  {
    return offset_ + static_cast<size_t>(pptr() - pbase());
  }

 protected:
  int_type overflow(int_type ch) override
  {
    RAFT_EXPECTS(!closed_, "kvikio_ofstream: write attempted after close");
    flush_buffer();
    if (!traits_type::eq_int_type(ch, traits_type::eof())) {
      *pptr() = traits_type::to_char_type(ch);
      pbump(1);
    }
    return traits_type::not_eof(ch);
  }

  std::streamsize xsputn(const char* input, std::streamsize count) override
  {
    RAFT_EXPECTS(!closed_, "kvikio_ofstream: write attempted after close");
    if (count <= 0) { return 0; }

    const auto requested = count;
    auto* current        = input;
    size_t remaining     = static_cast<size_t>(count);

    while (remaining > 0) {
      // If the caller hands us a large contiguous chunk, flush pending staged bytes and pass the
      // chunk straight to kvikio. This avoids std::streambuf's byte-at-a-time fallback path.
      if (remaining >= buffer_.size()) {
        flush_buffer();
        write_at_current_offset(handle_, current, remaining);
        return requested;
      }

      size_t available = static_cast<size_t>(epptr() - pptr());
      if (available == 0) {
        flush_buffer();
        available = static_cast<size_t>(epptr() - pptr());
      }

      const size_t n = std::min(remaining, available);
      std::memcpy(pptr(), current, n);
      pbump(static_cast<int>(n));
      current += n;
      remaining -= n;
    }

    return requested;
  }

  int sync() override
  {
    try {
      flush_buffer();
      return 0;
    } catch (...) {
      return -1;
    }
  }

  // Support tellp(): report the current output position.
  pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override
  {
    if ((which & std::ios_base::out) && dir == std::ios_base::cur && off == 0) {
      return pos_type(static_cast<off_type>(bytes_written()));
    }
    return pos_type(off_type(-1));
  }

 private:
  void flush_buffer()
  {
    const size_t n = static_cast<size_t>(pptr() - pbase());
    if (n > 0) {
      write_at_current_offset(handle_, pbase(), n);
      setp(buffer_.data(), buffer_.data() + buffer_.size());
    }
  }

  void write_at_current_offset(kvikio::FileHandle& handle, const void* data, size_t n)
  {
    RAFT_EXPECTS(!closed_, "kvikio_ofstream: write attempted after close");
    const size_t w = handle.pwrite(data, n, offset_).get();
    RAFT_EXPECTS(w == n, "kvikio_ofstream: short write (expected %zu, wrote %zu)", n, w);
    offset_ += n;
  }

  std::string path_;
  kvikio::FileHandle handle_;
  std::unique_ptr<kvikio::FileHandle> device_handle_;
  std::vector<char> buffer_;
  size_t offset_ = 0;
  bool closed_   = false;
};

kvikio_ofstream::kvikio_ofstream(const std::string& path, size_t buffer_size)
  : std::ostream(nullptr), buf_(std::make_unique<sbuf>(path, buffer_size))
{
  rdbuf(buf_.get());
}

kvikio_ofstream::~kvikio_ofstream()
{
  try {
    if (buf_) { buf_->close(); }
  } catch (...) {
    // Swallow during destruction.
  }
}

void kvikio_ofstream::close()
{
  if (buf_) { buf_->close(); }
}

kvikio_ofstream& kvikio_ofstream::write_device(const void* data, size_t size)
{
  buf_->write_device(data, size);
  return *this;
}

size_t kvikio_ofstream::bytes_written() const noexcept { return buf_ ? buf_->bytes_written() : 0; }

}  // namespace cuvs::util
