/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/util/host_memory.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <unistd.h>

namespace cuvs::util::detail {
namespace {

class cgroup_fixture {
 public:
  cgroup_fixture()
    : root_(std::filesystem::temp_directory_path() /
            ("cuvs_cgroup_test_" + std::to_string(::getpid()) + "_" +
             std::to_string(reinterpret_cast<uintptr_t>(this))))
  {
    std::filesystem::create_directories(root_);
  }

  ~cgroup_fixture()
  {
    std::error_code error;
    std::filesystem::remove_all(root_, error);
  }

  [[nodiscard]] std::filesystem::path path(const std::string& name) const { return root_ / name; }

  void write(const std::filesystem::path& path, const std::string& content)
  {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    ASSERT_TRUE(output.good());
    output << content;
    ASSERT_TRUE(output.good());
  }

 private:
  std::filesystem::path root_;
};

TEST(HostMemory, CgroupV2UsesTightestAncestorHeadroom)
{
  cgroup_fixture fixture;
  const auto mount = fixture.path("cgroup2");
  const auto leaf  = mount / "tenant/job";

  fixture.write(fixture.path("self.cgroup"), "0::/tenant/job\n");
  fixture.write(
    fixture.path("self.mountinfo"),
    "29 23 0:26 / " + mount.string() + " rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n");
  fixture.write(leaf / "memory.max", "900\n");
  fixture.write(leaf / "memory.current", "800\n");
  fixture.write(leaf / "memory.stat", "file 700\ninactive_file 20\nactive_file 680\n");
  fixture.write(mount / "tenant/memory.max", "1000\n");
  fixture.write(mount / "tenant/memory.current", "900\n");
  fixture.write(mount / "tenant/memory.stat",
                "file 250\nshmem 20\nfile_dirty 10\nfile_writeback 5\nunevictable 15\n"
                "inactive_file 50\nactive_file 200\n");
  fixture.write(mount / "memory.max", "max\n");
  fixture.write(mount / "memory.current", "500\n");

  const auto result =
    get_cgroup_memory_info(fixture.path("self.cgroup"), fixture.path("self.mountinfo"));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->version, cgroup_version::v2);
  EXPECT_EQ(result->limit, 1000);
  EXPECT_EQ(result->current, 900);
  EXPECT_EQ(result->reclaimable_file, 200);
  EXPECT_EQ(result->working_set, 700);
  EXPECT_EQ(result->available, 300);
  EXPECT_EQ(result->path, (mount / "tenant").string());
}

TEST(HostMemory, CgroupV2ResolvesNamespacedRoot)
{
  cgroup_fixture fixture;
  const auto mount = fixture.path("cgroup2");
  const auto leaf  = mount / "job";

  fixture.write(fixture.path("self.cgroup"), "0::/job\n");
  fixture.write(fixture.path("self.mountinfo"),
                "29 23 0:26 /docker/container " + mount.string() +
                  " rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n");
  fixture.write(leaf / "memory.max", "2048\n");
  fixture.write(leaf / "memory.current", "512\n");

  const auto result =
    get_cgroup_memory_info(fixture.path("self.cgroup"), fixture.path("self.mountinfo"));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->version, cgroup_version::v2);
  EXPECT_EQ(result->reclaimable_file, 0);
  EXPECT_EQ(result->working_set, 512);
  EXPECT_EQ(result->available, 1536);
  EXPECT_EQ(result->path, leaf.string());
}

TEST(HostMemory, CgroupV2ProtectedFileMemoryDoesNotUnderflow)
{
  cgroup_fixture fixture;
  const auto mount = fixture.path("cgroup2");
  const auto leaf  = mount / "job";

  fixture.write(fixture.path("self.cgroup"), "0::/job\n");
  fixture.write(
    fixture.path("self.mountinfo"),
    "29 23 0:26 / " + mount.string() + " rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n");
  fixture.write(leaf / "memory.max", "100\n");
  fixture.write(leaf / "memory.current", "120\n");
  fixture.write(leaf / "memory.stat",
                "file 100\nshmem 80\nfile_dirty 30\nfile_writeback 10\nunevictable 5\n");

  const auto result =
    get_cgroup_memory_info(fixture.path("self.cgroup"), fixture.path("self.mountinfo"));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->reclaimable_file, 0);
  EXPECT_EQ(result->working_set, 120);
  EXPECT_EQ(result->available, 0);
}

TEST(HostMemory, CgroupV1UsesHierarchicalInactiveFile)
{
  cgroup_fixture fixture;
  const auto mount = fixture.path("memory");
  const auto leaf  = mount / "container";

  fixture.write(fixture.path("self.cgroup"),
                "7:cpu:/docker/container\n8:memory:/docker/container\n");
  fixture.write(fixture.path("self.mountinfo"),
                "31 23 0:28 /docker " + mount.string() +
                  " rw,nosuid,nodev,noexec,relatime - cgroup cgroup rw,memory\n");
  fixture.write(leaf / "memory.limit_in_bytes", "4096\n");
  fixture.write(leaf / "memory.usage_in_bytes", "8192\n");
  fixture.write(leaf / "memory.stat", "inactive_file 128\ntotal_inactive_file 5000\n");

  const auto result =
    get_cgroup_memory_info(fixture.path("self.cgroup"), fixture.path("self.mountinfo"));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->version, cgroup_version::v1);
  EXPECT_EQ(result->limit, 4096);
  EXPECT_EQ(result->current, 8192);
  EXPECT_EQ(result->reclaimable_file, 5000);
  EXPECT_EQ(result->working_set, 3192);
  EXPECT_EQ(result->available, 904);
}

TEST(HostMemory, UnlimitedAndMalformedLimitsAreIgnored)
{
  cgroup_fixture fixture;
  const auto mount = fixture.path("cgroup2");
  const auto leaf  = mount / "job";

  fixture.write(fixture.path("self.cgroup"), "0::/job\n");
  fixture.write(
    fixture.path("self.mountinfo"),
    "29 23 0:26 / " + mount.string() + " rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n");
  fixture.write(leaf / "memory.max", "not-a-limit\n");
  fixture.write(leaf / "memory.current", "100\n");
  fixture.write(mount / "memory.max", "max\n");
  fixture.write(mount / "memory.current", "100\n");

  EXPECT_FALSE(get_cgroup_memory_info(fixture.path("self.cgroup"), fixture.path("self.mountinfo"))
                 .has_value());
}

TEST(HostMemory, CgroupV1UnlimitedSentinelIsIgnored)
{
  cgroup_fixture fixture;
  const auto mount = fixture.path("memory");

  fixture.write(fixture.path("self.cgroup"), "8:memory:/container\n");
  fixture.write(fixture.path("self.mountinfo"),
                "31 23 0:28 / " + mount.string() +
                  " rw,nosuid,nodev,noexec,relatime - cgroup cgroup rw,memory\n");
  fixture.write(mount / "container/memory.limit_in_bytes", "9223372036854771712\n");
  fixture.write(mount / "container/memory.usage_in_bytes", "1024\n");

  EXPECT_FALSE(get_cgroup_memory_info(fixture.path("self.cgroup"), fixture.path("self.mountinfo"))
                 .has_value());
}

}  // namespace
}  // namespace cuvs::util::detail
