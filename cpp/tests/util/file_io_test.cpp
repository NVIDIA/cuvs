/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/util/file_io.hpp>

#include "../../src/util/kvikio_serialize.hpp"

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace cuvs::util {

namespace {

// Deterministic byte pattern so reads can be validated independently of the writer.
std::vector<char> make_pattern(size_t n, uint8_t seed)
{
  std::vector<char> v(n);
  for (size_t i = 0; i < n; i++) {
    v[i] = static_cast<char>((i * 131u + seed * 7u + 17u) & 0xFFu);
  }
  return v;
}

// Create a unique writable scratch directory. Prefer the current working directory (typically the
// on-disk build tree, where O_DIRECT is usually supported) and fall back to the system temp dir.
class scratch_dir {
 public:
  scratch_dir()
  {
    const std::string name = ".cuvs_file_io_test_" + std::to_string(::getpid());
    std::error_code ec;
    std::filesystem::path base = std::filesystem::current_path(ec);
    if (ec) { base = std::filesystem::temp_directory_path(); }
    path_ = base / name;
    std::filesystem::create_directories(path_, ec);
    if (ec || !std::filesystem::is_directory(path_)) {
      path_ = std::filesystem::temp_directory_path() / name;
      std::filesystem::create_directories(path_);
    }
  }

  ~scratch_dir()
  {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  [[nodiscard]] std::string file(const std::string& name) const { return (path_ / name).string(); }
  [[nodiscard]] std::string dir() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

std::vector<char> read_whole_file(const std::string& path)
{
  std::ifstream is(path, std::ios::in | std::ios::binary);
  EXPECT_TRUE(is.good()) << "cannot open " << path;
  return std::vector<char>((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
}

}  // namespace

// create_numpy_file must produce a numpy-compatible header whose data body begins on a block
// boundary (so kvikio's O_DIRECT / GDS interior is aligned), and the file must be readable back
// through the numpy deserializer.
TEST(FileIO, CreateNumpyFileAlignedHeader)
{
  scratch_dir scratch;
  const std::string path = scratch.file("aligned.npy");
  const size_t rows      = 1234;
  const size_t cols      = 17;
  auto [fd, header_size] = create_numpy_file<float>(path, {rows, cols});
  EXPECT_TRUE(fd.is_valid());
  EXPECT_EQ(header_size % kNumpyDataAlignment, 0u)
    << "numpy data body must start on a " << kNumpyDataAlignment << "-byte boundary";

  // The numpy deserializer must recover the shape from the (re-padded) header.
  auto stream = fd.make_istream();
  auto header = raft::detail::numpy_serializer::read_header(stream);
  ASSERT_EQ(header.shape.size(), 2u);
  EXPECT_EQ(header.shape[0], rows);
  EXPECT_EQ(header.shape[1], cols);
}

// write_large_file followed by read_large_file with host buffers must round-trip the data exactly.
TEST(FileIO, HostReadWriteRoundTrip)
{
  scratch_dir scratch;
  const std::string path = scratch.file("host.npy");
  const size_t rows      = 4096;
  const size_t cols      = 33;
  const size_t n         = rows * cols;

  auto [fd, header_size] = create_numpy_file<float>(path, {rows, cols});

  std::vector<float> src(n);
  for (size_t i = 0; i < n; i++) {
    src[i] = static_cast<float>(i % 1000) * 0.5f;
  }
  write_large_file(fd, src.data(), n * sizeof(float), header_size);

  std::vector<float> dst(n, -1.0f);
  read_large_file(fd, dst.data(), n * sizeof(float), header_size);

  EXPECT_EQ(src, dst);
}

// read_large_file must also fill device memory (kvikio uses GPUDirect Storage when available, and
// stages through a host bounce buffer in compatibility mode). The data must match after copying
// back to the host.
TEST(FileIO, DeviceReadRoundTrip)
{
  raft::resources res;
  scratch_dir scratch;
  const std::string path = scratch.file("device.npy");
  const size_t rows      = 5000;
  const size_t cols      = 24;
  const size_t n         = rows * cols;

  auto [fd, header_size] = create_numpy_file<float>(path, {rows, cols});

  std::vector<float> src(n);
  for (size_t i = 0; i < n; i++) {
    src[i] = static_cast<float>((i * 7) % 4096);
  }
  write_large_file(fd, src.data(), n * sizeof(float), header_size);

  auto dev = raft::make_device_vector<float, int64_t>(res, n);
  read_large_file(fd, dev.data_handle(), n * sizeof(float), header_size);

  std::vector<float> dst(n, -1.0f);
  raft::copy(dst.data(), dev.data_handle(), n, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  EXPECT_EQ(src, dst);
}

// kvikio_ofstream must reproduce exactly the bytes handed to write(), across many small writes that
// span multiple internal buffer flushes, and report the correct logical size.
TEST(FileIO, KvikioOfstreamRoundTrip)
{
  scratch_dir scratch;
  const std::string path = scratch.file("stream.bin");
  std::vector<char> data = make_pattern(5'000'003, 42);
  const size_t cap       = size_t(1) << 20;  // 1 MiB staging buffer -> several flushes

  {
    kvikio_ofstream os(path, cap);
    size_t pos = 0;
    while (pos < data.size()) {
      const size_t chunk = std::min<size_t>(7777, data.size() - pos);
      os.write(data.data() + pos, chunk);
      pos += chunk;
    }
    os.flush();
    EXPECT_EQ(os.bytes_written(), data.size());
    EXPECT_EQ(os.tellp(), std::streampos(static_cast<std::streamoff>(data.size())));
    os.close();
  }

  const std::vector<char> got = read_whole_file(path);
  ASSERT_EQ(got.size(), data.size());
  EXPECT_EQ(got, data);
}

// Verify the std::ostream substitution used by serializers, including formatted and unformatted
// sequential output and current-position queries.
TEST(FileIO, KvikioOfstreamSequentialOstreamInterface)
{
  static_assert(std::is_base_of_v<std::ostream, kvikio_ofstream>);
  static_assert(std::is_convertible_v<kvikio_ofstream&, std::ostream&>);
  static_assert(!std::is_constructible_v<kvikio_ofstream, std::ostream&>);

  scratch_dir scratch;
  const std::string path = scratch.file("stream_interface.bin");

  {
    kvikio_ofstream file(path, 4096);
    std::ostream& os = file;
    os << "value=" << 42 << '\n';
    os.write("tail", 4);
    os.seekp(0, std::ios_base::cur);
    EXPECT_TRUE(os.good());
    EXPECT_EQ(os.tellp(), std::streampos(13));
    os.flush();
    file.close();
  }

  const std::vector<char> expected{
    'v', 'a', 'l', 'u', 'e', '=', '4', '2', '\n', 't', 'a', 'i', 'l'};
  EXPECT_EQ(read_whole_file(path), expected);
}

TEST(FileIO, KvikioDeviceMdspanSerializationThroughOstream)
{
  raft::resources res;
  scratch_dir scratch;
  const std::string path = scratch.file("device_mdspan.npy");
  const int64_t size     = 4096;

  auto source_host = raft::make_host_vector<uint32_t, int64_t>(size);
  for (int64_t i = 0; i < size; ++i) {
    source_host(i) = static_cast<uint32_t>(i * 17 + 3);
  }
  auto source_device = raft::make_device_vector<uint32_t, int64_t>(res, size);
  raft::copy(res, source_device.view(), source_host.view());

  {
    kvikio_ofstream file(path);
    std::ostream& os = file;
    detail::serialize_mdspan(res, os, raft::make_const_mdspan(source_device.view()));
    file.close();
  }

  auto result = raft::make_host_vector<uint32_t, int64_t>(size);
  std::ifstream input(path, std::ios::in | std::ios::binary);
  raft::deserialize_mdspan(res, input, result.view());
  EXPECT_TRUE(std::equal(source_host.data_handle(),
                         source_host.data_handle() + source_host.size(),
                         result.data_handle()));
}

TEST(FileIO, KvikioOfstreamRejectsRandomAccessSeeking)
{
  scratch_dir scratch;
  const std::string path = scratch.file("stream_seek.bin");

  {
    kvikio_ofstream os(path, 4096);
    os << "before";
    os.seekp(0, std::ios_base::beg);
    EXPECT_TRUE(os.fail());

    os.clear();
    EXPECT_EQ(os.tellp(), std::streampos(6));
    os << "after";
    os.close();
  }

  const std::vector<char> expected{'b', 'e', 'f', 'o', 'r', 'e', 'a', 'f', 't', 'e', 'r'};
  EXPECT_EQ(read_whole_file(path), expected);
}

// A large single write must bypass the small staging buffer correctly and still round-trip.
TEST(FileIO, KvikioOfstreamLargeSingleWrite)
{
  scratch_dir scratch;
  const std::string path = scratch.file("stream_large.bin");
  std::vector<char> data = make_pattern((size_t(8) << 20) + 123, 7);

  {
    kvikio_ofstream os(path, size_t(1) << 20);
    os.write(data.data(), data.size());
    EXPECT_EQ(os.bytes_written(), data.size());
    EXPECT_EQ(os.tellp(), std::streampos(static_cast<std::streamoff>(data.size())));
    os.close();
  }

  const std::vector<char> got = read_whole_file(path);
  ASSERT_EQ(got.size(), data.size());
  EXPECT_EQ(got, data);
}

// Host stream output and direct device output must share one ordered logical file position.
TEST(FileIO, KvikioOfstreamMixedHostAndDeviceWrites)
{
  raft::resources res;
  scratch_dir scratch;
  const std::string path = scratch.file("stream_device.bin");
  const std::vector<char> prefix{'h', 'e', 'a', 'd'};
  const std::vector<char> payload = make_pattern((size_t{3} << 20) + 17, 19);
  const std::vector<char> suffix{'t', 'a', 'i', 'l'};

  auto device_payload = raft::make_device_vector<char, int64_t>(res, payload.size());
  raft::copy(device_payload.data_handle(),
             payload.data(),
             payload.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  {
    kvikio_ofstream os(path, size_t{1} << 20);
    os.write(prefix.data(), prefix.size());
    os.write_device(device_payload.data_handle(), payload.size());
    os.write(suffix.data(), suffix.size());
    EXPECT_EQ(os.bytes_written(), prefix.size() + payload.size() + suffix.size());
    os.close();
  }

  std::vector<char> expected;
  expected.reserve(prefix.size() + payload.size() + suffix.size());
  expected.insert(expected.end(), prefix.begin(), prefix.end());
  expected.insert(expected.end(), payload.begin(), payload.end());
  expected.insert(expected.end(), suffix.begin(), suffix.end());
  EXPECT_EQ(read_whole_file(path), expected);
}

// Buffered metadata reads and direct device reads must advance one shared logical position. The
// short prefix deliberately leaves unread bytes in fd_streambuf's read-ahead buffer.
TEST(FileIO, KvikioFileReaderMixedStreamAndDeviceReads)
{
  raft::resources res;
  scratch_dir scratch;
  const std::string path = scratch.file("reader_device.bin");
  const std::vector<char> prefix{'m', 'e', 't', 'a', 'd', 'a', 't', 'a'};
  const std::vector<char> payload = make_pattern((size_t{3} << 20) + 17, 31);
  const std::vector<char> suffix{'t', 'r', 'a', 'i', 'l', 'e', 'r'};

  {
    std::ofstream os(path, std::ios::out | std::ios::binary);
    ASSERT_TRUE(os.good());
    os.write(prefix.data(), prefix.size());
    os.write(payload.data(), payload.size());
    os.write(suffix.data(), suffix.size());
  }

  auto device_payload = raft::make_device_vector<char, int64_t>(res, payload.size());
  raft::resource::sync_stream(res);
  kvikio_file_reader reader(path);

  std::vector<char> prefix_read(prefix.size());
  reader.stream().read(prefix_read.data(), prefix_read.size());
  ASSERT_TRUE(reader.stream().good());
  reader.read_device(device_payload.data_handle(), payload.size());
  std::vector<char> suffix_read(suffix.size());
  reader.stream().read(suffix_read.data(), suffix_read.size());
  ASSERT_TRUE(reader.stream().good());

  std::vector<char> payload_read(payload.size());
  raft::copy(payload_read.data(),
             device_payload.data_handle(),
             payload_read.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  EXPECT_EQ(prefix_read, prefix);
  EXPECT_EQ(payload_read, payload);
  EXPECT_EQ(suffix_read, suffix);
}

// Defensive checks on the bulk helpers.
TEST(FileIO, InvalidArguments)
{
  scratch_dir scratch;
  auto [fd, header_size] = create_numpy_file<float>(scratch.file("args.npy"), {16, 4});
  char buf[8]            = {0};
  EXPECT_THROW(read_large_file(fd, buf, 0, header_size), raft::exception);
  EXPECT_THROW(write_large_file(fd, buf, 0, header_size), raft::exception);
  EXPECT_THROW(read_large_file(fd, nullptr, 8, header_size), raft::exception);

  EXPECT_NO_THROW(read_large_file(fd, buf, sizeof(buf), header_size));
  const int dup_fd = ::dup(fd.get());
  ASSERT_NE(dup_fd, -1);
  file_descriptor pathless_fd(dup_fd);
  EXPECT_THROW(read_large_file(pathless_fd, buf, sizeof(buf), header_size), raft::exception);
  EXPECT_THROW(write_large_file(pathless_fd, buf, sizeof(buf), header_size), raft::exception);
}

TEST(FileIO, RejectsReplacedPathForBulkIO)
{
  scratch_dir scratch;
  const std::string path = scratch.file("replace.npy");
  auto [fd, header_size] = create_numpy_file<float>(path, {16, 4});
  char buf[8]            = {0};

  std::error_code ec;
  ASSERT_TRUE(std::filesystem::remove(path, ec));
  ASSERT_FALSE(ec) << ec.message();

  std::ofstream replacement(path, std::ios::out | std::ios::binary);
  ASSERT_TRUE(replacement.good()) << "cannot create replacement " << path;
  std::vector<char> replacement_data(header_size + sizeof(buf), 0);
  replacement.write(replacement_data.data(), replacement_data.size());
  replacement.close();
  ASSERT_TRUE(replacement.good()) << "cannot write replacement " << path;

  EXPECT_THROW(read_large_file(fd, buf, sizeof(buf), header_size), raft::exception);
  EXPECT_THROW(write_large_file(fd, buf, sizeof(buf), header_size), raft::exception);
}

}  // namespace cuvs::util
