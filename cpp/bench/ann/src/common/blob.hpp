/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "ann_types.hpp"
#include "util.hpp"

#include <cerrno>
#include <linux/mman.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace cuvs::bench {

/** RAII wrapper for a file descriptor. */
struct file_descriptor {
  explicit file_descriptor(std::string file_name) : fd_{std::fopen(file_name.c_str(), "r")}
  {
    if (fd_ == nullptr) {
      throw std::runtime_error(
        "cuvs::bench::file_descriptor: failed to open a file (std::fopen): errno = " +
        std::to_string(errno) + ", " + std::string(strerror(errno)) + ". File name: " + file_name);
    }
  }

  explicit file_descriptor(size_t file_size_bytes) : fd_{std::tmpfile()}
  {
    if (fd_ == nullptr) {
      throw std::runtime_error(
        "cuvs::bench::file_descriptor: failed to open a temporary file (std::tmpfile): errno = " +
        std::to_string(errno) + ", " + std::string(strerror(errno)));
    }
    if (ftruncate(fileno(fd_), file_size_bytes) == -1) {
      throw std::runtime_error(
        "cuvs::bench::file_descriptor: failed to call `ftruncate` to allocate memory for a "
        "temporary file.");
    }
  }

  // No copies for owning struct
  file_descriptor(const file_descriptor& res)                      = delete;
  auto operator=(const file_descriptor& other) -> file_descriptor& = delete;
  // Moving is fine
  file_descriptor(file_descriptor&& other) : fd_{std::exchange(other.fd_, nullptr)} {}
  auto operator=(file_descriptor&& other) -> file_descriptor&
  {
    std::swap(this->fd_, other.fd_);
    return *this;
  }

  ~file_descriptor() noexcept
  {
    if (fd_ != nullptr) { std::fclose(fd_); }
  }

  [[nodiscard]] auto value() const -> FILE* { return fd_; }

 private:
  FILE* fd_ = nullptr;
};

class mmap_error : public std::runtime_error {
 private:
  int errno_;

 public:
  mmap_error(std::string extra_msg)
    : std::runtime_error("cuvs::bench::mmap_owner: `mmap` error: Value of errno " +
                         std::to_string(errno) + ", " + std::string(strerror(errno)) + ". " +
                         extra_msg),
      errno_(errno)
  {
  }

  [[nodiscard]] auto code() const noexcept { return errno_; }
};

/** RAII wrapper for a mmap/munmap. */
struct mmap_owner {
  /** Map a file */
  mmap_owner(
    const file_descriptor& descriptor, size_t offset, size_t size, int flags, bool writable = false)
    : ptr_{mmap_verbose(size,
                        writable ? PROT_READ | PROT_WRITE : PROT_READ,
                        flags,
                        fileno(descriptor.value()),
                        offset)},
      size_{size}
  {
  }

  /** Allocate a new memory (not backed by a file). */
  mmap_owner(size_t size, int flags)
    : ptr_{mmap_verbose(size, PROT_READ | PROT_WRITE, flags, -1, 0)}, size_{size}
  {
  }

  ~mmap_owner() noexcept
  {
    if (ptr_ != nullptr) { munmap(ptr_, size_); }
  }

  // No copies for owning struct
  mmap_owner(const mmap_owner& res)                      = delete;
  auto operator=(const mmap_owner& other) -> mmap_owner& = delete;
  // Moving is fine
  mmap_owner(mmap_owner&& other)
    : ptr_{std::exchange(other.ptr_, nullptr)}, size_{std::exchange(other.size_, 0)}
  {
  }
  auto operator=(mmap_owner&& other) -> mmap_owner&
  {
    std::swap(this->ptr_, other.ptr_);
    std::swap(this->size_, other.size_);
    return *this;
  }

  [[nodiscard]] auto data() const -> void* { return ptr_; }
  [[nodiscard]] auto size() const -> size_t { return size_; }

 private:
  void* ptr_;
  size_t size_;

  static inline auto mmap_verbose(size_t length, int prot, int flags, int fd, off_t offset) -> void*
  {
    auto ptr = mmap(nullptr, length, prot, flags, fd, offset);
    if (ptr == MAP_FAILED) {
      std::array<char, 256> buf;
      snprintf(buf.data(),
               sizeof(buf),
               "Failed call: cuvs::bench::mmap_owner:mmap(nullptr, %zu, 0x%08x, 0x%08x, %d, %zd)",
               length,
               prot,
               flags,
               fd,
               offset);
      throw mmap_error{std::string(buf.data())};
    }
    return ptr;
  }
};

/** RAII wrapper for managed memory. */
struct managed_mem_owner {
  explicit managed_mem_owner(size_t size) : size_{size}
  {
#ifndef BUILD_CPU_ONLY
    auto err_code = cudaMallocManaged(&ptr_, size_);
    if (err_code != cudaSuccess) {
      ptr_ = nullptr;
      throw std::runtime_error{
        "cuvs::bench::managed_mem_owner: call to cudaMallocManaged failed with code " +
        std::to_string(err_code)};
    }
#else
    throw std::runtime_error{
      "Device functions are not available when built with BUILD_CPU_ONLY flag."};
#endif
  }

  ~managed_mem_owner() noexcept
  {
    if (ptr_ != nullptr) {
#ifndef BUILD_CPU_ONLY
      cudaFree(ptr_);
#endif
    }
  }

  // No copies for owning struct
  managed_mem_owner(const managed_mem_owner& res)                      = delete;
  auto operator=(const managed_mem_owner& other) -> managed_mem_owner& = delete;
  // Moving is fine
  managed_mem_owner(managed_mem_owner&& other)
    : ptr_{std::exchange(other.ptr_, nullptr)}, size_{std::exchange(other.size_, 0)}
  {
  }
  auto operator=(managed_mem_owner&& other) -> managed_mem_owner&
  {
    std::swap(this->ptr_, other.ptr_);
    std::swap(this->size_, other.size_);
    return *this;
  }

  [[nodiscard]] auto data() const -> void* { return ptr_; }
  [[nodiscard]] auto size() const -> size_t { return size_; }

 private:
  void* ptr_ = nullptr;
  size_t size_;
};

/** RAII wrapper for device memory. */
struct device_mem_owner {
  explicit device_mem_owner(size_t size) : size_{size}
  {
#ifndef BUILD_CPU_ONLY
    auto err_code = cudaMalloc(&ptr_, size_);
    if (err_code != cudaSuccess) {
      ptr_ = nullptr;
      throw std::runtime_error{
        "cuvs::bench::device_mem_owner: call to cudaMalloc failed with code " +
        std::to_string(err_code)};
    }
#else
    throw std::runtime_error{
      "Device functions are not available when built with BUILD_CPU_ONLY flag."};
#endif
  }
  ~device_mem_owner() noexcept
  {
    if (ptr_ != nullptr) {
#ifndef BUILD_CPU_ONLY
      cudaFree(ptr_);
#endif
    }
  }
  // No copies for owning struct
  device_mem_owner(const device_mem_owner& res)                      = delete;
  auto operator=(const device_mem_owner& other) -> device_mem_owner& = delete;
  // Moving is fine
  device_mem_owner(device_mem_owner&& other)
    : ptr_{std::exchange(other.ptr_, nullptr)}, size_{std::exchange(other.size_, 0)}
  {
  }
  auto operator=(device_mem_owner&& other) -> device_mem_owner&
  {
    std::swap(this->ptr_, other.ptr_);
    std::swap(this->size_, other.size_);
    return *this;
  }

  [[nodiscard]] auto data() const -> void* { return ptr_; }
  [[nodiscard]] auto size() const -> size_t { return size_; }

 private:
  void* ptr_ = nullptr;
  size_t size_;
};

/** Lazy-initialized file handle. */
struct file {
  explicit file(std::string file_name) : reqsize_or_name_{std::move(file_name)} {}
  explicit file(size_t tmp_size_bytes) : reqsize_or_name_{tmp_size_bytes} {}

  // this shouldn't be necessary, but adds extra safety (make sure descriptors are not copied)
  file(const file&)                    = delete;
  auto operator=(const file&) -> file& = delete;
  file(file&&);
  auto operator=(file&&) -> file&;

  [[nodiscard]] auto descriptor() const -> const file_descriptor&
  {
    if (!descriptor_.has_value()) {
      std::visit([&d = descriptor_](auto&& x) { d.emplace(x); }, reqsize_or_name_);
    }
    return descriptor_.value();
  }

  [[nodiscard]] auto path() const -> std::string
  {
    return std::holds_alternative<std::string>(reqsize_or_name_)
             ? std::get<std::string>(reqsize_or_name_)
             : "<temporary>";
  }

  [[nodiscard]] auto size() const -> size_t
  {
    if (!size_.has_value()) {
      size_.emplace(std::visit(
        [&s = size_](auto&& x) {
          if constexpr (std::is_same_v<std::decay_t<decltype(x)>, size_t>) {
            return x;
          } else {
            struct stat statbuf;
            if (stat(x.c_str(), &statbuf) != 0) {
              throw std::runtime_error{"cuvs::bench::file::size() error: `stat` failed: " + x};
            }
            return static_cast<size_t>(statbuf.st_size);
          }
        },
        reqsize_or_name_));
    }
    return size_.value();
  }

  [[nodiscard]] auto is_temporary() const -> bool
  {
    return std::holds_alternative<size_t>(reqsize_or_name_);
  }

  void reset_lazy_state() const
  {
    descriptor_.reset();
    size_.reset();
  }

 private:
  std::variant<size_t, std::string> reqsize_or_name_;
  mutable std::optional<file_descriptor> descriptor_ = std::nullopt;
  mutable std::optional<size_t> size_                = std::nullopt;
};

// declare the move constructors explicitly outside of the class declaration to make sure if
// anything is wrong, we catch that at compile time.
inline file::file(file&&)                    = default;
inline auto file::operator=(file&&) -> file& = default;

/**
 * Lazy-initialized file handle with the size information provided by our .bin format:
 * The file always starts with two uint32_t values: [n_rows, n_cols].
 */
template <typename T>
struct blob_file : public file {
  explicit blob_file(std::string file_name, uint32_t rows_offset = 0, uint32_t rows_limit = 0)
    : file{std::move(file_name)}, rows_offset_{rows_offset}, rows_limit_{rows_limit}
  {
  }

  explicit blob_file(uint32_t n_rows, uint32_t n_cols)
    : file{sizeof(T) * n_rows * n_cols + 2 * sizeof(uint32_t)}, rows_offset_{0}, rows_limit_{0}
  {
    // NB: this forces the file descriptor, thus breaking lazy-initialization. Not sure if it's
    // worth refactoring in this case.
    std::array<uint32_t, 2> h{n_rows, n_cols};
    if (std::fwrite(h.data(), sizeof(uint32_t), h.size(), descriptor().value()) != 2) {
      throw std::runtime_error{
        "cuvs::bench::blob_file `fwrite` failed when initializing a tmp file."};
    }
    if (std::fflush(descriptor().value()) != 0) {
      throw std::runtime_error{
        "cuvs::bench::blob_file `fflush` failed with non-zero code when initializing a tmp file."};
    }
  }

  blob_file(const blob_file<T>&)                    = delete;
  auto operator=(const blob_file<T>&) -> blob_file& = delete;
  blob_file(blob_file<T>&&);
  auto operator=(blob_file<T>&&) -> blob_file<T>&;

  [[nodiscard]] auto n_rows() const -> uint32_t { return header()[0]; }
  [[nodiscard]] auto n_cols() const -> uint32_t { return header()[1]; }

  [[nodiscard]] auto rows_offset() const -> uint32_t { return rows_offset_; }
  [[nodiscard]] auto rows_limit() const -> uint32_t
  {
    auto rows_max = n_rows() - std::min(rows_offset(), n_rows());          // available rows
    return rows_limit_ == 0 ? rows_max : std::min(rows_limit_, rows_max);  // limited rows
  }

  void reset_lazy_state() const
  {
    file::reset_lazy_state();
    header_.reset();
  }

 private:
  mutable std::optional<std::array<uint32_t, 2>> header_ = std::nullopt;
  uint32_t rows_offset_;
  uint32_t rows_limit_;

  [[nodiscard]] auto header() const -> const std::array<uint32_t, 2>&
  {
    if (!header_.has_value()) {
      std::array<uint32_t, 2> h;
      std::rewind(descriptor().value());
      if (std::fread(h.data(), sizeof(uint32_t), h.size(), descriptor().value()) != 2) {
        throw std::runtime_error{"cuvs::bench::blob_file read header of bin file failed: " +
                                 path()};
      }
      header_.emplace(h);
    }
    return header_.value();
  }
};

// declare the move constructors explicitly outside of the class declaration to make sure if
// anything is wrong, we catch that at compile time.
template <typename T>
inline blob_file<T>::blob_file(blob_file<T>&&) = default;
template <typename T>
inline auto blob_file<T>::operator=(blob_file<T>&&) -> blob_file<T>& = default;

/** Lazily map or copy the file content onto host memory. */
template <typename T>
struct blob_mmap {
  explicit blob_mmap(blob_file<T>&& file,
                     bool copy_in_memory     = false,
                     HugePages hugepages_2mb = HugePages::kDisable)
    : file_{std::move(file)},
      copy_in_memory_{copy_in_memory},
      hugepages_2mb_requested_{hugepages_2mb},
      hugepages_2mb_actual_{hugepages_2mb > HugePages::kDisable}
  {
  }
  explicit blob_mmap(std::string file_name,
                     uint32_t rows_offset    = 0,
                     uint32_t rows_limit     = 0,
                     bool copy_in_memory     = false,
                     HugePages hugepages_2mb = HugePages::kDisable)
    : blob_mmap{
        blob_file<T>{std::move(file_name), rows_offset, rows_limit}, copy_in_memory, hugepages_2mb}
  {
  }

 private:
  blob_file<T> file_;
  bool copy_in_memory_;
  HugePages hugepages_2mb_requested_;
  mutable bool hugepages_2mb_actual_;

  mutable std::optional<std::tuple<mmap_owner, ptrdiff_t>> handle_;
  // Heap-allocate the once_flag so that blob_mmap remains movable (std::once_flag
  // itself is neither copyable nor movable). Multiple threads can race on the
  // first call to handle() via blob<T>::data(); without this serialization each
  // racing thread would mmap the file and the losing emplace would munmap
  // the winner's mapping AND invalidate the reference the winner had already
  // returned to its caller -- a flaky SIGSEGV in ann benchmarks.
  mutable std::unique_ptr<std::once_flag> handle_once_ = std::make_unique<std::once_flag>();

  [[nodiscard]] auto handle() const -> const std::tuple<mmap_owner, ptrdiff_t>&
  {
    std::call_once(*handle_once_, [this] {
      // Loop here so that the huge-page fallback (formerly a recursive call
      // to handle()) does not re-enter std::call_once, which would deadlock.
      while (!handle_.has_value()) {
        size_t page_size =
          hugepages_2mb_actual_ ? 1024ull * 1024ull * 2ull : sysconf(_SC_PAGE_SIZE);
        int flags = 0;
        if (hugepages_2mb_actual_) { flags |= MAP_HUGETLB | MAP_HUGE_2MB; }
        size_t data_start = sizeof(T) * file_.rows_offset() * file_.n_cols() + sizeof(uint32_t) * 2;
        size_t data_end   = sizeof(T) * file_.rows_limit() * file_.n_cols() + data_start;

        try {
          if (copy_in_memory_) {
            // Copy the content in-memory
            flags |= MAP_ANONYMOUS | MAP_PRIVATE;
            size_t size = data_end - data_start;
            mmap_owner owner{size, flags};
            std::fseek(file_.descriptor().value(), data_start, SEEK_SET);
            auto n_elems =
              static_cast<size_t>(file_.rows_limit()) * static_cast<size_t>(file_.n_cols());
            if (std::fread(owner.data(), sizeof(T), n_elems, file_.descriptor().value()) !=
                n_elems) {
              throw std::runtime_error{"cuvs::bench::blob_mmap() fread " + file_.path() +
                                       " failed"};
            }
            handle_.emplace(std::move(owner), 0);
          } else {
            // Map the file
            // If this is a temporary file, we're supposed to write to it, hence MAP_SHARED.
            flags |= file_.is_temporary() ? MAP_SHARED : MAP_PRIVATE;
            size_t mmap_start = (data_start / page_size) * page_size;
            size_t mmap_size  = data_end - mmap_start;
            handle_.emplace(
              mmap_owner{file_.descriptor(), mmap_start, mmap_size, flags, file_.is_temporary()},
              data_start - mmap_start);
          }
        } catch (const mmap_error& e) {
          bool hugepages_2mb_asked = hugepages_2mb_requested_ == HugePages::kAsk ||
                                     hugepages_2mb_requested_ == HugePages::kRequire;
          if (e.code() == EPERM && hugepages_2mb_asked && hugepages_2mb_actual_) {
            if (hugepages_2mb_requested_ == HugePages::kRequire) {
              log_warn(
                "cuvs::bench::blob_mmap: `mmap` failed to map due to EPERM, which is likely "
                "caused by the permissions issue. You either need a CAP_IPC_LOCK capability or "
                "run the program with sudo. We will try again without huge pages.");
            }
            hugepages_2mb_actual_ = false;
            continue;  // retry the loop with huge pages disabled
          }
          if (e.code() == EINVAL && hugepages_2mb_asked && hugepages_2mb_actual_ &&
              !copy_in_memory_) {
            if (hugepages_2mb_requested_ == HugePages::kRequire) {
              log_warn(
                "cuvs::bench::blob_mmap: `mmap` failed to map due to EINVAL, which is likely "
                "caused by the file system not supporting huge pages. We will try again without "
                "huge pages.");
            }
            hugepages_2mb_actual_ = false;
            continue;  // retry the loop with huge pages disabled
          }
          throw;  // The error is not due to huge pages or otherwise unrecoverable
        }
      }
    });
    return handle_.value();
  }

 public:
  [[nodiscard]] auto data() const -> T*
  {
    auto& [owner, offset] = handle();
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(owner.data()) + offset);
  }

  [[nodiscard]] auto is_in_memory() const -> bool { return copy_in_memory_; }
  [[nodiscard]] auto is_hugepage() const -> bool { return hugepages_2mb_actual_; }
  /**
   * Enabling hugepages is not always possible. For convenience, we may silently disable it in some
   * cases. This function helps to decide whether the current setting is compatible with the
   * request.R
   */
  [[nodiscard]] auto hugepage_compliant(HugePages request) const -> bool
  {
    if (request == hugepages_2mb_requested_) {
      // whatever the actual state is, the result would be the same if we recreated
      // the mapping.
      return true;
    }
    bool hp_enabled    = hugepages_2mb_actual_ && request > HugePages::kDisable;
    bool hp_disabled   = !hugepages_2mb_actual_ && request == HugePages::kDisable;
    bool hp_not_forced = !hugepages_2mb_actual_ && request == HugePages::kAsk;
    return hp_enabled || hp_disabled || hp_not_forced;
  }
  [[nodiscard]] auto n_rows() const -> uint32_t { return file_.rows_limit(); }
  [[nodiscard]] auto n_cols() const -> uint32_t { return file_.n_cols(); }
  [[nodiscard]] auto size() const -> size_t { return sizeof(T) * n_rows() * n_cols(); }
  [[nodiscard]] auto unmap() && noexcept -> blob_file<T>
  {
    // If we used mmap on a temporary file, then it was writable.
    // Then there's a chance the user wrote something to the mmap.
    // Then we must ensure the changes are visible in the file before migrating it.
    bool flush_writes = file_.is_temporary() && handle_.has_value();
    if (flush_writes) {
      auto& [owner, _] = handle();
      msync(owner.data(), owner.size(), MS_SYNC | MS_INVALIDATE);
    }
    handle_.reset();
    if (flush_writes) { std::fflush(file_.descriptor().value()); }
    return blob_file<T>{std::move(file_)};
  }
  [[nodiscard]] auto release() && noexcept -> blob_file<T> { return std::move(*this).unmap(); }

  void reset_lazy_state() const
  {
    file_.reset_lazy_state();
    hugepages_2mb_actual_ = hugepages_2mb_requested_ > HugePages::kDisable;
  }
};

template <typename T>
struct blob_pinned {
 private:
  mutable blob_mmap<T> blob_;
  mutable void* ptr_ = nullptr;

 public:
  explicit blob_pinned(blob_mmap<T>&& blob) : blob_{std::move(blob)} {}
  // First map the file and then register to CUDA.
  // NB: as per docs, huge pages are not supported.
  explicit blob_pinned(blob_file<T>&& blob, bool copy_in_memory = true)
    : blob_{std::move(blob), copy_in_memory, false}
  {
  }
  explicit blob_pinned(std::string file_name,
                       uint32_t rows_offset = 0,
                       uint32_t rows_limit  = 0,
                       bool copy_in_memory  = true)
    : blob_pinned{blob_file<T>{file_name, rows_offset, rows_limit}, copy_in_memory}
  {
  }

  ~blob_pinned() noexcept { reset_lazy_state(); }

  // No copies for owning struct
  blob_pinned(const blob_pinned& res)                      = delete;
  auto operator=(const blob_pinned& other) -> blob_pinned& = delete;
  // Moving is fine
  blob_pinned(blob_pinned&& other) : blob_{std::move(other.blob_)}, ptr_{other.ptr_}
  {
    other.ptr_ = nullptr;
  }
  auto operator=(blob_pinned&& other) -> blob_pinned&
  {
    std::swap(this->blob_, other.blob_);
    std::swap(this->ptr_, other.ptr_);
    return *this;
  }

  [[nodiscard]] auto data() const -> T*
  {
    if (ptr_ == nullptr) {
      void* ptr = reinterpret_cast<void*>(blob_.data());
#ifndef BUILD_CPU_ONLY
      int flags       = cudaHostRegisterDefault;
      auto error_code = cudaSuccess;
      if (!blob_.is_in_memory()) {
        flags      = cudaHostRegisterIoMemory | cudaHostRegisterReadOnly;
        error_code = cudaHostRegister(ptr, blob_.size(), flags);
        if (error_code == cudaErrorNotSupported) {
          // Sometimes read-only is not supported
          flags      = cudaHostRegisterIoMemory;
          error_code = cudaHostRegister(ptr, blob_.size(), error_code);
        }
      } else {
        error_code = cudaHostRegister(ptr, blob_.size(), cudaHostRegisterDefault);
      }
      if (error_code == cudaErrorInvalidValue && (!blob_.is_in_memory() || blob_.is_hugepage())) {
        auto hugepage =
          (blob_.is_hugepage() && blob_.is_in_memory()) ? HugePages::kAsk : HugePages::kDisable;
        auto file = std::move(blob_).release();
        blob_     = blob_mmap<T>{std::move(file), true, hugepage};
        return data();
      }
      if (error_code != cudaSuccess) {
        log_error(
          "cuvs::bench::blob_pinned: cudaHostRegister(%p, %zu, %d)", ptr, blob_.size(), flags);
        throw std::runtime_error{
          "cuvs::bench::blob_pinned: call to cudaHostRegister failed with code " +
          std::to_string(error_code)};
      }
#endif
      ptr_ = ptr;
    }
    return reinterpret_cast<T*>(ptr_);
  }

  [[nodiscard]] auto unpin() && noexcept -> blob_mmap<T>
  {
    // unregister the memory before passing it to a third party
    if (ptr_ != nullptr) {
#ifndef BUILD_CPU_ONLY
      cudaHostUnregister(ptr_);
#endif
      ptr_ = nullptr;
    }
    return blob_mmap<T>{std::move(blob_)};
  }
  void reset_lazy_state() const
  {
    if (ptr_ != nullptr) {
#ifndef BUILD_CPU_ONLY
      cudaHostUnregister(ptr_);
#endif
      ptr_ = nullptr;
    }
    blob_.reset_lazy_state();
  }

  [[nodiscard]] auto is_in_memory() const noexcept -> bool { return true; }
  [[nodiscard]] auto is_hugepage() const noexcept -> bool { return blob_.is_hugepage(); }
  [[nodiscard]] auto hugepage_compliant(HugePages request) const -> bool
  {
    return blob_.hugepage_compliant(request);
  }
  [[nodiscard]] auto n_rows() const -> uint32_t { return blob_.n_rows(); }
  [[nodiscard]] auto n_cols() const -> uint32_t { return blob_.n_cols(); }
  [[nodiscard]] auto size() const -> size_t { return blob_.size(); }
  [[nodiscard]] auto release() && noexcept -> blob_file<T>
  {
    return std::move(*this).unpin().release();
  }
};

template <typename OwnerType, typename T>
struct blob_copying {
 private:
  blob_mmap<T> blob_;
  mutable std::optional<OwnerType> mem_ = std::nullopt;

 public:
  explicit blob_copying(blob_mmap<T>&& blob) : blob_{std::move(blob)} {}
  // First map the file and then copy it to device; use huge pages for faster copy
  explicit blob_copying(blob_file<T>&& blob) : blob_{std::move(blob), false, HugePages::kAsk} {}
  explicit blob_copying(std::string file_name, uint32_t rows_offset = 0, uint32_t rows_limit = 0)
    : blob_copying{blob_file<T>{file_name, rows_offset, rows_limit}}
  {
  }

  [[nodiscard]] auto data() const -> T*
  {
    if (!mem_.has_value()) {
      mem_.emplace(blob_.size());
#ifndef BUILD_CPU_ONLY
      auto error_code = cudaMemcpy(mem_->data(), blob_.data(), blob_.size(), cudaMemcpyDefault);
      if (error_code != cudaSuccess) {
        throw std::runtime_error{"cuvs::bench::blob_device: call to cudaMemcpy failed with code " +
                                 std::to_string(error_code)};
      }
#endif
    }
    return reinterpret_cast<T*>(mem_->data());
  }

  [[nodiscard]] auto free() && noexcept -> blob_mmap<T>
  {
    mem_.reset();
    return blob_mmap<T>{std::move(blob_)};
  }
  void reset_lazy_state() const
  {
    mem_.reset();
    blob_.reset_lazy_state();
  }

  [[nodiscard]] auto is_in_memory() const noexcept -> bool { return true; }
  [[nodiscard]] auto is_hugepage() const noexcept -> bool { return blob_.is_hugepage(); }
  // For copying CUDA allocation, the hugepage setting is not relevant at all.
  [[nodiscard]] auto hugepage_compliant(HugePages request) const -> bool { return true; }
  [[nodiscard]] auto n_rows() const -> uint32_t { return blob_.n_rows(); }
  [[nodiscard]] auto n_cols() const -> uint32_t { return blob_.n_cols(); }
  [[nodiscard]] auto size() const -> size_t { return blob_.size(); }
  [[nodiscard]] auto release() && noexcept -> blob_file<T>
  {
    return std::move(*this).free().release();
  }
};

template <typename T>
using blob_device = blob_copying<device_mem_owner, T>;
template <typename T>
using blob_managed = blob_copying<managed_mem_owner, T>;

/**
 * @brief A blob is a single contiguous piece of data.
 *
 * It can reside in host, managed, or device memory, it can be pinned in physical memory or backed
 * by a filesystem. You can also control whether to use huge pages (2MB) when it is in host memory.
 *
 * The blob data is a one- or two-dimensional typed array (it has n_rows and n_cols properties).
 *
 * The blob tries to be lazy accessing the data. It reads or copies the data only when it is
 * requested. This allows the benchmarking executable be more resilient; for example, a blob can
 * point to a non-existent query dataset when the benchmark is in the index build mode.
 *
 * @tparam T the data type.
 */
template <typename T>
struct blob {
 private:
  using blob_type = std::variant<blob_mmap<T>, blob_pinned<T>, blob_device<T>, blob_managed<T>>;
  mutable blob_type value_;

  [[nodiscard]] auto data_mmap(bool in_memory, HugePages request_hugepages_2mb) const -> T*
  {
    if (auto* v = std::get_if<blob_mmap<T>>(&value_)) {
      if (v->is_in_memory() == in_memory && v->hugepage_compliant(request_hugepages_2mb)) {
        return v->data();
      }
    }
    blob_type tmp{std::move(value_)};
    value_ = std::visit(
      [in_memory, request_hugepages_2mb](auto&& val) {
        return blob_mmap<T>{std::move(val).release(), in_memory, request_hugepages_2mb};
      },
      std::move(tmp));

    return data();
  }

  [[nodiscard]] auto data_pinned(HugePages request_hugepages_2mb) const -> T*
  {
    // The requested type is there
    if (auto* v = std::get_if<blob_pinned<T>>(&value_)) {
      if (v->hugepage_compliant(request_hugepages_2mb)) { return v->data(); }
    }
    // If there's already an mmap allocation, we just need to pin it.
    if (auto* v = std::get_if<blob_mmap<T>>(&value_)) {
      if (v->hugepage_compliant(request_hugepages_2mb)) {
        blob_mmap<T> tmp{std::move(*v)};
        return value_.template emplace<blob_pinned<T>>(std::move(tmp)).data();
      }
    }
    // otherwise do full reset
    blob_type tmp{std::move(value_)};
    value_ = std::visit(
      [request_hugepages_2mb](auto&& val) {
        blob_mmap<T> tmp{std::move(val).release(), true, request_hugepages_2mb};
        return blob_pinned<T>{std::move(tmp)};
      },
      std::move(value_));

    return data();
  }

  [[nodiscard]] auto data_device() const -> T*
  {
    // The requested type is there
    if (auto* v = std::get_if<blob_device<T>>(&value_)) { return v->data(); }
    // otherwise do full reset
    blob_type tmp{std::move(value_)};
    value_ = std::visit([](auto&& val) { return blob_device<T>{std::move(val).release()}; },
                        std::move(tmp));
    return data();
  }

  [[nodiscard]] auto data_managed() const -> T*
  {
    // The requested type is there
    if (auto* v = std::get_if<blob_managed<T>>(&value_)) { return v->data(); }
    // otherwise do full reset
    blob_type tmp{std::move(value_)};
    value_ = std::visit([](auto&& val) { return blob_managed<T>{std::move(val).release()}; },
                        std::move(tmp));
    return data();
  }

 public:
  explicit blob(std::string file_name,
                uint32_t rows_offset    = 0,
                uint32_t rows_limit     = 0,
                bool copy_in_memory     = false,
                HugePages hugepages_2mb = HugePages::kDisable)
    : value_{std::in_place_type<blob_mmap<T>>,
             std::move(file_name),
             rows_offset,
             rows_limit,
             copy_in_memory,
             hugepages_2mb}
  {
  }

  template <typename VariantT>
  explicit blob(VariantT&& blob_variant) : value_{std::move(blob_variant)}
  {
  }

  [[nodiscard]] auto data() const -> T*
  {
    return std::visit([](auto&& val) { return val.data(); }, value_);
  }

  [[nodiscard]] auto data(MemoryType memory_type,
                          HugePages request_hugepages_2mb = HugePages::kDisable) const -> T*
  {
    switch (memory_type) {
      case MemoryType::kHost: return data_mmap(true, request_hugepages_2mb);
      case MemoryType::kHostMmap: return data_mmap(false, request_hugepages_2mb);
      case MemoryType::kHostPinned:
        if (request_hugepages_2mb > HugePages::kDisable) {
          log_error(
            "cuvs::bench::blob::data(): huge pages are requested but not supported by "
            "cudaHostRegister at the moment. We will try nevertheless...");
        }
        return data_pinned(request_hugepages_2mb);
      case MemoryType::kDevice: return data_device();    // hugepages are not relevant here
      case MemoryType::kManaged: return data_managed();  // hugepages are not relevant here
      default:
        throw std::runtime_error{"cuvs::bench::blob::data(): unexpected memory type " +
                                 std::to_string(static_cast<int>(memory_type))};
    }
  }

  [[nodiscard]] auto is_in_memory() const noexcept -> bool
  {
    return std::visit([](auto&& val) { return val.is_in_memory(); }, value_);
  }
  [[nodiscard]] auto is_hugepage() const noexcept -> bool
  {
    return std::visit([](auto&& val) { return val.is_hugepage(); }, value_);
  }
  [[nodiscard]] auto n_rows() const -> uint32_t
  {
    return std::visit([](auto&& val) { return val.n_rows(); }, value_);
  }

  [[nodiscard]] auto n_cols() const -> uint32_t
  {
    return std::visit([](auto&& val) { return val.n_cols(); }, value_);
  }
  /** Size of the blob content in bytes (doesn't include file header). */
  [[nodiscard]] auto size() const -> size_t
  {
    return std::visit([](auto&& val) { return val.size(); }, value_);
  }
  /**
   * Reset the hidden internal state, e.g. release the memory allocation or close a file descriptor.
   * This is useful in case when accessing the blob resulted in an error/invalid state, which should
   * be reproduced at a later point in time.
   */
  void reset_lazy_state() const
  {
    std::visit([](auto&& val) { val.reset_lazy_state(); }, value_);
  }
};

}  // namespace  cuvs::bench
