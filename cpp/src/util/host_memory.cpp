/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/util/host_memory.hpp>

#include "host_memory.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace cuvs::util {
namespace {

struct cgroup_membership {
  detail::cgroup_version version;
  std::filesystem::path path;
};

struct cgroup_mount {
  detail::cgroup_version version;
  std::filesystem::path root;
  std::filesystem::path mount_point;
};

std::optional<size_t> parse_size(std::string_view text)
{
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
    text.remove_prefix(1);
  }
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
    text.remove_suffix(1);
  }
  if (text.empty()) { return std::nullopt; }

  size_t value            = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) { return std::nullopt; }
  return value;
}

std::optional<std::string> read_first_token(const std::filesystem::path& path)
{
  std::ifstream input(path);
  std::string token;
  if (!(input >> token)) { return std::nullopt; }
  return token;
}

bool contains_controller(std::string_view controllers, std::string_view expected)
{
  while (!controllers.empty()) {
    const auto separator  = controllers.find(',');
    const auto controller = controllers.substr(0, separator);
    if (controller == expected) { return true; }
    if (separator == std::string_view::npos) { break; }
    controllers.remove_prefix(separator + 1);
  }
  return false;
}

std::vector<std::string> split_whitespace(std::string_view text)
{
  std::istringstream stream{std::string{text}};
  std::vector<std::string> fields;
  for (std::string field; stream >> field;) {
    fields.push_back(std::move(field));
  }
  return fields;
}

std::string unescape_mount_path(std::string_view path)
{
  std::string result;
  result.reserve(path.size());
  for (size_t i = 0; i < path.size(); ++i) {
    if (path[i] == '\\' && i + 3 < path.size() && path[i + 1] >= '0' && path[i + 1] <= '7' &&
        path[i + 2] >= '0' && path[i + 2] <= '7' && path[i + 3] >= '0' && path[i + 3] <= '7') {
      const auto value =
        static_cast<char>((path[i + 1] - '0') * 64 + (path[i + 2] - '0') * 8 + (path[i + 3] - '0'));
      result.push_back(value);
      i += 3;
    } else {
      result.push_back(path[i]);
    }
  }
  return result;
}

std::vector<cgroup_membership> parse_cgroup_memberships(const std::filesystem::path& path)
{
  std::ifstream input(path);
  std::vector<cgroup_membership> memberships;
  for (std::string line; std::getline(input, line);) {
    const auto first_separator  = line.find(':');
    const auto second_separator = first_separator == std::string::npos
                                    ? std::string::npos
                                    : line.find(':', first_separator + 1);
    if (second_separator == std::string::npos) { continue; }

    const std::string_view hierarchy{line.data(), first_separator};
    const std::string_view controllers{line.data() + first_separator + 1,
                                       second_separator - first_separator - 1};
    const std::string_view cgroup_path{line.data() + second_separator + 1,
                                       line.size() - second_separator - 1};
    if (cgroup_path.empty() || cgroup_path.front() != '/') { continue; }

    if (hierarchy == "0" && controllers.empty()) {
      memberships.push_back({detail::cgroup_version::v2, std::filesystem::path{cgroup_path}});
    } else if (contains_controller(controllers, "memory")) {
      memberships.push_back({detail::cgroup_version::v1, std::filesystem::path{cgroup_path}});
    }
  }
  return memberships;
}

std::vector<cgroup_mount> parse_cgroup_mounts(const std::filesystem::path& path)
{
  std::ifstream input(path);
  std::vector<cgroup_mount> mounts;
  for (std::string line; std::getline(input, line);) {
    const auto separator = line.find(" - ");
    if (separator == std::string::npos) { continue; }

    const auto left  = split_whitespace(std::string_view{line}.substr(0, separator));
    const auto right = split_whitespace(std::string_view{line}.substr(separator + 3));
    if (left.size() < 6 || right.size() < 3) { continue; }

    std::optional<detail::cgroup_version> version;
    if (right[0] == "cgroup2") {
      version = detail::cgroup_version::v2;
    } else if (right[0] == "cgroup" && (contains_controller(left[5], "memory") ||
                                        contains_controller(right[2], "memory"))) {
      version = detail::cgroup_version::v1;
    }
    if (!version.has_value()) { continue; }

    mounts.push_back({*version,
                      std::filesystem::path{unescape_mount_path(left[3])}.lexically_normal(),
                      std::filesystem::path{unescape_mount_path(left[4])}.lexically_normal()});
  }
  return mounts;
}

bool is_path_prefix(const std::filesystem::path& prefix, const std::filesystem::path& path)
{
  return std::mismatch(prefix.begin(), prefix.end(), path.begin(), path.end()).first ==
         prefix.end();
}

std::optional<std::filesystem::path> resolve_cgroup_path(const cgroup_membership& membership,
                                                         const cgroup_mount& mount)
{
  if (membership.version != mount.version) { return std::nullopt; }

  const auto member_path = membership.path.lexically_normal();
  if (is_path_prefix(mount.root, member_path)) {
    const auto relative = member_path.lexically_relative(mount.root);
    if (relative.empty()) { return std::nullopt; }
    return relative == "." ? mount.mount_point : (mount.mount_point / relative).lexically_normal();
  }

  // In a cgroup namespace the membership path is relative to the namespace root, while mountinfo
  // can still expose the original cgroup as the mount root.
  return (mount.mount_point / member_path.relative_path()).lexically_normal();
}

bool is_v1_unlimited(size_t limit)
{
  // The v1 controller represents unlimited with a page-aligned value close to LONG_MAX.
  return limit >= static_cast<size_t>(std::numeric_limits<int64_t>::max() / 2);
}

size_t subtract_saturating(size_t value, size_t amount)
{
  return amount < value ? value - amount : 0;
}

size_t get_reclaimable_file_memory(const std::filesystem::path& path,
                                   detail::cgroup_version version)
{
  std::ifstream input(path / "memory.stat");
  std::optional<size_t> file;
  size_t shmem          = 0;
  size_t file_dirty     = 0;
  size_t file_writeback = 0;
  size_t unevictable    = 0;
  std::optional<size_t> inactive_file;
  std::optional<size_t> total_inactive_file;
  for (std::string key, value; input >> key >> value;) {
    const auto parsed_value = parse_size(value);
    if (!parsed_value.has_value()) { continue; }
    if (key == "file") {
      file = *parsed_value;
    } else if (key == "shmem") {
      shmem = *parsed_value;
    } else if (key == "file_dirty") {
      file_dirty = *parsed_value;
    } else if (key == "file_writeback") {
      file_writeback = *parsed_value;
    } else if (key == "unevictable") {
      unevictable = *parsed_value;
    } else if (key == "inactive_file") {
      inactive_file = *parsed_value;
    } else if (key == "total_inactive_file") {
      total_inactive_file = *parsed_value;
    }
  }

  // v1's total_ counter includes descendants and therefore matches hierarchical usage accounting.
  if (version == detail::cgroup_version::v1 && total_inactive_file.has_value()) {
    return *total_inactive_file;
  }

  if (version == detail::cgroup_version::v2 && file.has_value()) {
    size_t clean_file = *file;
    clean_file        = subtract_saturating(clean_file, shmem);
    clean_file        = subtract_saturating(clean_file, file_dirty);
    clean_file        = subtract_saturating(clean_file, file_writeback);
    clean_file        = subtract_saturating(clean_file, unevictable);
    return clean_file;
  }

  // Older or restricted cgroup interfaces may not expose the full v2 breakdown.
  return inactive_file.value_or(0);
}

std::optional<detail::cgroup_memory_info> inspect_cgroup_hierarchy(
  detail::cgroup_version version,
  std::filesystem::path current_path,
  const std::filesystem::path& mount_point)
{
  std::optional<detail::cgroup_memory_info> result;
  if (!is_path_prefix(mount_point, current_path)) { return result; }

  while (true) {
    const auto limit_file =
      current_path /
      (version == detail::cgroup_version::v2 ? "memory.max" : "memory.limit_in_bytes");
    const auto current_file =
      current_path /
      (version == detail::cgroup_version::v2 ? "memory.current" : "memory.usage_in_bytes");
    const auto limit_token   = read_first_token(limit_file);
    const auto current_token = read_first_token(current_file);

    if (limit_token.has_value() && *limit_token != "max" && current_token.has_value()) {
      const auto limit   = parse_size(*limit_token);
      const auto current = parse_size(*current_token);
      if (limit.has_value() && current.has_value() &&
          !(version == detail::cgroup_version::v1 && is_v1_unlimited(*limit))) {
        const size_t reclaimable_file =
          std::min(*current, get_reclaimable_file_memory(current_path, version));
        const size_t working_set = *current - reclaimable_file;
        const size_t available   = working_set < *limit ? *limit - working_set : 0;
        if (!result.has_value() || available < result->available) {
          result = detail::cgroup_memory_info{*limit,
                                              *current,
                                              reclaimable_file,
                                              working_set,
                                              available,
                                              version,
                                              current_path.string()};
        }
      }
    }

    if (current_path == mount_point) { break; }
    const auto parent = current_path.parent_path();
    if (parent == current_path || !is_path_prefix(mount_point, parent)) { break; }
    current_path = parent;
  }
  return result;
}

size_t get_system_available_memory()
{
  std::ifstream meminfo("/proc/meminfo");
  for (std::string line; std::getline(meminfo, line);) {
    constexpr std::string_view key = "MemAvailable:";
    if (line.rfind(key, 0) != 0) { continue; }

    std::istringstream value_stream{line.substr(key.size())};
    size_t kilobytes = 0;
    std::string unit;
    if (value_stream >> kilobytes >> unit && unit == "kB" &&
        kilobytes <= std::numeric_limits<size_t>::max() / 1024) {
      return kilobytes * 1024;
    }
  }
  RAFT_FAIL("Failed to get available memory from /proc/meminfo");
}

}  // namespace

namespace detail {

std::optional<cgroup_memory_info> get_cgroup_memory_info(
  const std::filesystem::path& proc_self_cgroup, const std::filesystem::path& proc_self_mountinfo)
{
  const auto memberships = parse_cgroup_memberships(proc_self_cgroup);
  const auto mounts      = parse_cgroup_mounts(proc_self_mountinfo);
  std::optional<cgroup_memory_info> result;

  for (const auto& membership : memberships) {
    for (const auto& mount : mounts) {
      const auto resolved_path = resolve_cgroup_path(membership, mount);
      if (!resolved_path.has_value()) { continue; }
      const auto candidate =
        inspect_cgroup_hierarchy(membership.version, *resolved_path, mount.mount_point);
      if (candidate.has_value() &&
          (!result.has_value() || candidate->available < result->available)) {
        result = candidate;
      }
    }
  }
  return result;
}

}  // namespace detail

host_memory_info get_host_memory_info()
{
  const size_t system_available = get_system_available_memory();
  host_memory_info result{
    system_available, system_available, std::nullopt, std::nullopt, std::nullopt, std::nullopt};
  if (const auto cgroup = detail::get_cgroup_memory_info(); cgroup.has_value()) {
    result.cgroup_limit            = cgroup->limit;
    result.cgroup_current          = cgroup->current;
    result.cgroup_reclaimable_file = cgroup->reclaimable_file;
    result.cgroup_working_set      = cgroup->working_set;
    result.available               = std::min(system_available, cgroup->available);
  }
  return result;
}

size_t get_free_host_memory()
{
  const auto memory = get_host_memory_info();
  RAFT_EXPECTS(memory.available > 0,
               "No host memory is available within the current system or cgroup limit");
  return memory.available;
}

}  // namespace cuvs::util
