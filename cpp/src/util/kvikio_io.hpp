/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/logger.hpp>

#if __has_include(<kvikio/compat_mode.hpp>)
#include <kvikio/compat_mode.hpp>
#define CUVS_KVIKIO_HAS_COMPAT_MODE_HEADER 1
#else
#define CUVS_KVIKIO_HAS_COMPAT_MODE_HEADER 0
#endif

#include <kvikio/file_handle.hpp>

#if __has_include(<kvikio/utils.hpp>)
#include <kvikio/utils.hpp>
#define CUVS_KVIKIO_HAS_UTILS_HEADER 1
#else
#define CUVS_KVIKIO_HAS_UTILS_HEADER 0
#endif

#include <exception>
#include <mutex>
#include <string>

namespace cuvs::util::detail {

#if CUVS_KVIKIO_HAS_COMPAT_MODE_HEADER
template <typename Handle = kvikio::FileHandle>
auto open_kvikio_file_compat_off(const std::string& path, const std::string& flags, int)
  -> decltype(Handle(path, flags, Handle::m644, kvikio::CompatMode::OFF))
{
  return Handle(path, flags, Handle::m644, kvikio::CompatMode::OFF);
}
#endif

template <typename Handle = kvikio::FileHandle>
auto open_kvikio_file_compat_off(const std::string& path, const std::string& flags, long)
  -> decltype(Handle(path, flags, Handle::m644, false))
{
  return Handle(path, flags, Handle::m644, false);
}

inline kvikio::FileHandle open_kvikio_file_compat_off(const std::string& path,
                                                      const std::string& flags,
                                                      ...)
{
  return kvikio::FileHandle(path, flags);
}

inline kvikio::FileHandle open_kvikio_file_for_device_io(const std::string& path,
                                                         const std::string& flags)
{
  // Prefer GDS for device transfers, but retain KvikIO's automatic POSIX fallback when the
  // current system cannot open the file with compatibility mode disabled.
  try {
    return open_kvikio_file_compat_off(path, flags, 0);
  } catch (const std::exception& e) {
    static std::once_flag warning_once;
    std::call_once(warning_once, [&] {
      RAFT_LOG_WARN(
        "GDS is unavailable for %s (%s); falling back to KvikIO's default I/O mode. "
        "Further GDS fallback warnings are suppressed.",
        path.c_str(),
        e.what());
    });
    return kvikio::FileHandle(path, flags);
  }
}

inline bool is_kvikio_device_memory(const void* buffer)
{
#if CUVS_KVIKIO_HAS_UTILS_HEADER
  return buffer != nullptr && !kvikio::is_host_memory(buffer);
#else
  return false;
#endif
}

inline kvikio::FileHandle open_kvikio_file_for_ace_io(const std::string& path,
                                                      const std::string& flags,
                                                      const void* buffer)
{
  if (!is_kvikio_device_memory(buffer)) { return kvikio::FileHandle(path, flags); }
  return open_kvikio_file_for_device_io(path, flags);
}

}  // namespace cuvs::util::detail

#undef CUVS_KVIKIO_HAS_COMPAT_MODE_HEADER
#undef CUVS_KVIKIO_HAS_UTILS_HEADER
