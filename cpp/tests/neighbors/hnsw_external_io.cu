/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/neighbors/detail/cagra/ace_external_plan.hpp"
#include "../../src/neighbors/detail/hnsw.hpp"
#include "../../src/neighbors/detail/hnsw/external_format.hpp"
#include "../../src/neighbors/detail/hnsw/external_translate.hpp"
#include "../../src/neighbors/detail/hnsw/external_workspace.hpp"
#include "../../src/neighbors/detail/hnsw/serialize_layout.hpp"

#include <gtest/gtest.h>
#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>

namespace cuvs::neighbors::hnsw::detail::external {
namespace {

class temporary_directory {
 public:
  temporary_directory()
  {
    static uint64_t sequence = 0;
    do {
      path_ = std::filesystem::temp_directory_path() /
              ("cuvs-hnsw-external-test-" + std::to_string(::getpid()) + "-" +
               std::to_string(sequence++));
    } while (!std::filesystem::create_directory(path_));
  }
  ~temporary_directory()
  {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

TEST(HnswExternalPlan, BigannByteLedger)
{
  cagra::detail::ace_external_plan_input input;
  input.rows                   = 1'000'000'000;
  input.dim                    = 128;
  input.element_size           = sizeof(uint8_t);
  input.M                      = 24;
  input.graph_degree           = 48;
  input.intermediate_degree    = 72;
  input.requested_partitions   = 64;
  input.available_host_bytes   = uint64_t{2} << 40;
  input.available_device_bytes = uint64_t{2} << 40;
  input.force_disk             = true;
  input.hierarchy              = true;

  auto plan = cagra::detail::make_ace_external_plan(input);
  auto materialized_host =
    cagra::detail::estimate_materialized_cagra_ace_host_bytes(input, plan.partitions);
  constexpr double gib = static_cast<double>(uint64_t{1} << 30);
  EXPECT_TRUE(plan.use_disk);
  EXPECT_GT(materialized_host, input.rows * input.graph_degree * sizeof(uint32_t));
  EXPECT_NEAR((plan.bytes.source_scan + plan.bytes.centroid_sample) / gib, 120.4, 0.2);
  EXPECT_NEAR((plan.bytes.stage_write + plan.bytes.stage_read) / gib, 506.6, 0.2);
  EXPECT_NEAR(plan.bytes.base_output / gib, 309.2, 0.2);
  EXPECT_NEAR((plan.bytes.upper_sidecar_write + plan.bytes.upper_sidecar_read) / gib, 8.1, 0.2);
  EXPECT_NEAR(plan.bytes.final_upper_output / gib, 7.8, 0.2);
  EXPECT_NEAR(plan.bytes.logical_total() / gib, 952.1, 0.5);
}

TEST(HnswHostMemory, EstimateIncludesRuntimeAndHierarchyStorage)
{
  constexpr int64_t rows = 1'000;
  constexpr int64_t dim  = 128;
  constexpr int degree   = 48;
  const auto base_file_bytes =
    rows * (sizeof(uint32_t) + degree * sizeof(uint32_t) + dim * sizeof(float) + sizeof(size_t));
  auto base_only = cuvs::neighbors::hnsw::detail::estimate_hnsw_host_memory<float>(
    rows, dim, degree, cuvs::distance::DistanceType::L2Expanded, HnswHierarchy::NONE, 200);
  auto with_hierarchy = cuvs::neighbors::hnsw::detail::estimate_hnsw_host_memory<float>(
    rows, dim, degree, cuvs::distance::DistanceType::L2Expanded, HnswHierarchy::GPU, 200);
  EXPECT_GT(base_only, base_file_bytes);
  EXPECT_GT(with_hierarchy, base_only);
}

TEST(HnswExternalPlan, OverflowAndHardCap)
{
  cagra::detail::ace_external_plan_input input;
  input.rows                   = std::numeric_limits<uint32_t>::max();
  input.dim                    = std::numeric_limits<uint64_t>::max();
  input.element_size           = sizeof(float);
  input.M                      = 32;
  input.graph_degree           = 64;
  input.intermediate_degree    = 96;
  input.available_host_bytes   = uint64_t{1} << 30;
  input.available_device_bytes = uint64_t{1} << 30;
  input.force_disk             = true;
  EXPECT_THROW(cagra::detail::make_ace_external_plan(input), raft::logic_error);

  EXPECT_THROW(cagra::detail::external_checked_mul(
                 std::numeric_limits<uint64_t>::max(), uint64_t{2}, "test overflow"),
               raft::logic_error);

  input.rows                   = 10'000;
  input.dim                    = 128;
  input.available_host_bytes   = 1;
  input.available_device_bytes = 1;
  EXPECT_THROW(cagra::detail::make_ace_external_plan(input), raft::logic_error);
}

TEST(HnswExternalPlan, PartitionsIncreaseMonotonically)
{
  cagra::detail::ace_external_plan_input input;
  input.rows                   = 1'000'000;
  input.dim                    = 128;
  input.element_size           = sizeof(float);
  input.M                      = 24;
  input.graph_degree           = 48;
  input.intermediate_degree    = 72;
  input.available_host_bytes   = uint64_t{8} << 30;
  input.available_device_bytes = uint64_t{8} << 30;
  input.force_disk             = true;
  auto roomy                   = cagra::detail::make_ace_external_plan(input);
  input.available_host_bytes   = uint64_t{2} << 30;
  input.available_device_bytes = uint64_t{2} << 30;
  auto tight                   = cagra::detail::make_ace_external_plan(input);
  EXPECT_GE(tight.partitions, roomy.partitions);
  EXPECT_EQ(roomy.queue_depth, 2);
  EXPECT_LE(tight.queue_depth, roomy.queue_depth);
  EXPECT_LE(tight.host_peak_bytes, input.available_host_bytes * 4 / 5);
  EXPECT_LE(tight.device_peak_bytes, input.available_device_bytes * 4 / 5);

  input.requested_partitions   = input.rows;
  input.available_host_bytes   = uint64_t{64} << 30;
  input.available_device_bytes = uint64_t{64} << 30;
  auto clamped                 = cagra::detail::make_ace_external_plan(input);
  EXPECT_LE(clamped.partitions,
            cagra::detail::external_maximum_partitions(input.rows, input.intermediate_degree));
  EXPECT_GT(clamped.target_occurrences, input.intermediate_degree);

  input.requested_partitions = input.rows + 1;
  EXPECT_THROW(cagra::detail::make_ace_external_plan(input), raft::logic_error);
}

TEST(HnswExternalPlan, SampleAndAssignmentRangesAreBoundedAndMonotonic)
{
  auto sample = cagra::detail::make_external_sample_ranges(10'003, 1'001);
  ASSERT_LE(sample.size(), 16);
  uint64_t sampled      = 0;
  uint64_t previous_end = 0;
  for (const auto& range : sample) {
    EXPECT_GT(range.count, 0);
    EXPECT_GE(range.start, previous_end);
    EXPECT_LE(range.start + range.count, 10'003);
    sampled += range.count;
    previous_end = range.start + range.count;
  }
  EXPECT_EQ(sampled, 1'001);

  auto scan        = cagra::detail::make_external_monotonic_ranges(10'003, 257);
  uint64_t scanned = 0;
  for (const auto& range : scan) {
    EXPECT_EQ(range.start, scanned);
    EXPECT_GT(range.count, 0);
    scanned += range.count;
  }
  EXPECT_EQ(scanned, 10'003);

  auto single = cagra::detail::make_external_sample_ranges(100, 1);
  ASSERT_EQ(single.size(), 1);
  EXPECT_EQ(single[0].start, 49);
  EXPECT_EQ(single[0].count, 1);
}

TEST(HnswExternalConversion, CopiesAndConvertsRows)
{
  std::array<float, 4> float_source{1.0f, -2.0f, 3.5f, 4.0f};
  std::array<float, 4> float_destination{};
  copy_rows_as_float(float_source.data(), float_destination.data(), 2, 2);
  EXPECT_EQ(float_destination, float_source);

  std::array<int8_t, 4> int_source{1, -2, 3, 4};
  std::array<float, 4> converted{};
  copy_rows_as_float(int_source.data(), converted.data(), 1, int_source.size());
  EXPECT_EQ(converted, (std::array<float, 4>{1.0f, -2.0f, 3.0f, 4.0f}));

  EXPECT_THROW(
    copy_rows_as_float(float_source.data(), float_destination.data(), uint64_t{1} << 63, 2),
    std::overflow_error);
}

TEST(HnswExternalFormat, RoundTripAndTruncation)
{
  temporary_directory directory;
  auto path = directory.path() / "core.stage";
  cuvs::util::file_descriptor fd(path.string(), O_CREAT | O_EXCL | O_RDWR, 0600);
  auto header = make_stage_header<float>(
    stage_kind::core, 3, 7, 0, 2 * sizeof(uint32_t) + 3 * sizeof(float), 123);
  header.committed_records = 1;
  ASSERT_EQ(::ftruncate(fd.get(), static_cast<off_t>(expected_file_size(header))), 0);
  write_stage_header(fd.get(), header);
  std::array<std::byte, 2 * sizeof(uint32_t) + 3 * sizeof(float)> record{};
  uint32_t label = 9;
  std::memcpy(record.data(), &label, sizeof(label));
  pwrite_all(fd.get(), record.data(), record.size(), stage_data_offset);

  auto decoded = read_stage_header(fd.get());
  EXPECT_NO_THROW(validate_stage_header(
    decoded, stage_kind::core, element_kind::f32, sizeof(float), 3, 7, 0, record.size(), 123));
  EXPECT_NO_THROW(validate_stage_file_size(fd.get(), decoded));
  ASSERT_EQ(::ftruncate(fd.get(), static_cast<off_t>(expected_file_size(header) - 1)), 0);
  EXPECT_THROW(validate_stage_file_size(fd.get(), decoded), raft::logic_error);
}

TEST(HnswExternalFormat, ShortReadIsCatchable)
{
  temporary_directory directory;
  auto path = directory.path() / "short.bin";
  cuvs::util::file_descriptor fd(path.string(), O_CREAT | O_EXCL | O_RDWR, 0600);
  uint8_t byte = 1;
  pwrite_all(fd.get(), &byte, sizeof(byte), 0);
  std::array<uint8_t, 2> output{};
  EXPECT_THROW(pread_all(fd.get(), output.data(), output.size(), 0), std::runtime_error);
}

TEST(HnswExternalFormat, WriteFailureIsCatchable)
{
  if (!std::filesystem::exists("/dev/full")) { GTEST_SKIP() << "/dev/full is unavailable"; }
  cuvs::util::file_descriptor fd("/dev/full", O_WRONLY);
  uint8_t byte = 1;
  EXPECT_THROW(pwrite_all(fd.get(), &byte, sizeof(byte), 0), std::runtime_error);
}

TEST(HnswExternalStage, RoundRobinAssignmentsRemainBuffered)
{
  temporary_directory directory;
  external_workspace workspace(directory.path(), 23);
  constexpr uint32_t partitions = 16;
  constexpr uint32_t dimension  = 4;
  stage_buffer_pool<float> stages(
    workspace, dimension, partitions, uint64_t{64} << 10, uint64_t{64} << 10, 23);
  std::array<float, dimension> vector{};
  for (uint32_t row = 0; row < 100; ++row) {
    uint32_t partition = row % partitions;
    stages.append_core(partition, row, vector.data());
    stages.append_spill((partition + 1) % partitions, partition, row / partitions, vector.data());
  }
  stages.finalize();
  EXPECT_LE(stages.write_requests(), 2 * partitions)
    << "round-robin assignment should retain one buffer per stage file";
}

TEST(HnswExternalStage, BufferedReaderReusesFileAcrossRefills)
{
  temporary_directory directory;
  external_workspace workspace(directory.path(), 29);
  constexpr uint32_t dimension = 4;
  stage_buffer_pool<float> stages(
    workspace, dimension, 1, uint64_t{64} << 10, uint64_t{64} << 10, 29);
  std::array<float, dimension> vector{};
  for (uint32_t row = 0; row < 3; ++row) {
    stages.append_core(0, row, vector.data());
  }
  stages.finalize();

  cuvs::util::file_descriptor fd(stages.core(0).path.string(), O_RDONLY);
  auto header = read_stage_header(fd.get());
  buffered_stage_reader reader(fd, header, static_cast<size_t>(header.record_size));
  const std::byte* record = nullptr;
  for (uint32_t row = 0; row < 3; ++row) {
    ASSERT_TRUE(reader.next(record));
    uint32_t label = 0;
    std::memcpy(&label, record, sizeof(label));
    EXPECT_EQ(label, row);
  }
  EXPECT_FALSE(reader.next(record));
  EXPECT_EQ(reader.read_requests(), 3);
}

TEST(HnswExternalGraph, OwnerTranslationIsPartitionLocal)
{
  std::vector<uint32_t> prefixes{0, 3, 7, 10};
  EXPECT_EQ(external_owner_id(prefixes, 0, 2), 2);
  EXPECT_EQ(external_owner_id(prefixes, 1, 3), 6);
  EXPECT_EQ(external_owner_id(prefixes, 2, 2), 9);
  EXPECT_THROW(external_owner_id(prefixes, 1, 4), raft::logic_error);
  EXPECT_THROW(external_owner_id(prefixes, 3, 0), raft::logic_error);
}

TEST(HnswExternalGraph, TranslatesIdsOnDevice)
{
  raft::resources res;
  std::array<uint32_t, 5> graph{0, 2, 1, 3, 7};
  std::array<uint32_t, 4> mapping{10, 20, 30, 40};
  auto device_graph   = raft::make_device_vector<uint32_t, int64_t>(res, graph.size());
  auto device_mapping = raft::make_device_vector<uint32_t, int64_t>(res, mapping.size());
  raft::copy(res,
             device_graph.view(),
             raft::make_host_vector_view<const uint32_t, int64_t>(graph.data(), graph.size()));
  raft::copy(res,
             device_mapping.view(),
             raft::make_host_vector_view<const uint32_t, int64_t>(mapping.data(), mapping.size()));
  translate_graph_ids(res,
                      device_graph.data_handle(),
                      device_graph.size(),
                      device_mapping.data_handle(),
                      device_mapping.size());
  raft::copy(res,
             raft::make_host_vector_view<uint32_t, int64_t>(graph.data(), graph.size()),
             raft::make_const_mdspan(device_graph.view()));
  raft::resource::sync_stream(res);
  EXPECT_EQ(graph, (std::array<uint32_t, 5>{10, 30, 20, 40, UINT32_MAX}));
}

TEST(HnswExternalWorkspace, PreservesCallerFilesAndPublishesAtomically)
{
  temporary_directory directory;
  auto sentinel = directory.path() / "sentinel.txt";
  {
    std::ofstream stream(sentinel);
    stream << "caller-owned";
  }

  std::filesystem::path failed_workspace;
  {
    external_workspace workspace(directory.path(), 17);
    failed_workspace = workspace.invocation_dir();
    auto stage       = workspace.create_private_file("owned.stage");
    uint8_t value    = 7;
    pwrite_all(stage.get(), &value, 1, 0);
    workspace.mark_failed("test", "injected");
  }
  EXPECT_TRUE(std::filesystem::exists(sentinel));
  EXPECT_FALSE(std::filesystem::exists(failed_workspace));

  {
    external_workspace workspace(directory.path(), 18);
    auto partial   = workspace.create_partial(4);
    uint32_t value = 42;
    pwrite_all(partial.get(), &value, sizeof(value), 0);
    workspace.publish(partial);
  }
  EXPECT_TRUE(std::filesystem::exists(directory.path() / "hnsw_index.bin"));
  EXPECT_TRUE(std::filesystem::exists(sentinel));
  EXPECT_THROW(external_workspace(directory.path(), 19), raft::logic_error);
}

TEST(HnswExternalHierarchy, DeterministicNestedDistribution)
{
  constexpr uint32_t rows = 1'000'000;
  uint64_t level_one      = 0;
  uint64_t level_two      = 0;
  for (uint32_t id = 0; id < rows; ++id) {
    auto first  = level_for_internal_id(id, hnsw_level_seed, 24);
    auto second = level_for_internal_id(id, hnsw_level_seed, 24);
    EXPECT_EQ(first, second);
    level_one += first >= 1;
    level_two += first >= 2;
  }
  EXPECT_NEAR(static_cast<double>(level_one) / rows, 1.0 / 24.0, 0.001);
  EXPECT_NEAR(static_cast<double>(level_two) / rows, 1.0 / (24.0 * 24.0), 0.0002);
  EXPECT_LE(level_two, level_one);
  auto summary = summarize_hierarchy(rows, 24);
  ASSERT_FALSE(summary.active_by_level.empty());
  EXPECT_EQ(summary.active_by_level[0], level_one);
}

TEST(HnswExternalLayout, WritesLoadableNoneHierarchy)
{
  temporary_directory directory;
  auto path         = directory.path() / "tiny.bin";
  auto batched_path = directory.path() / "tiny-batched.bin";
  cuvs::util::file_descriptor fd(path.string(), O_CREAT | O_EXCL | O_RDWR, 0600);
  hnswlib::L2Space<float, float> space(2);
  hierarchy_summary summary;
  auto layout = make_hnsw_serialize_layout<float>(&space, 4, 2, 2, 100, false, summary);
  auto exact  = layout.exact_file_size(0);
  ASSERT_EQ(::posix_fallocate(fd.get(), 0, static_cast<off_t>(exact)), 0);
  sequential_file_writer writer(fd, 0, 8);
  write_hnsw_header(writer, layout);
  std::array<std::array<float, 2>, 4> vectors{{{0, 0}, {1, 0}, {0, 1}, {1, 1}}};
  std::array<std::array<uint32_t, 2>, 4> neighbors{{{1, 2}, {0, 3}, {0, 3}, {1, 2}}};
  for (uint32_t row = 0; row < 4; ++row) {
    write_hnsw_base_row(writer, layout, neighbors[row].data(), vectors[row].data(), row);
  }
  for (uint32_t row = 0; row < 4; ++row) {
    write_hnsw_upper_node_header(writer, layout, 0);
  }
  writer.flush();
  EXPECT_EQ(writer.position(), exact);
  EXPECT_GT(writer.request_count(), 1);
  fd.close();

  raft::resources res;
  index_params params;
  params.hierarchy       = HnswHierarchy::NONE;
  params.ef_construction = 100;
  {
    std::ofstream batched(batched_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(batched);
    serialize_to_hnswlib_batched<float, uint32_t>(
      res,
      batched,
      params,
      vectors.size(),
      vectors[0].size(),
      neighbors[0].size(),
      cuvs::distance::DistanceType::L2Expanded,
      [&](int64_t start, int64_t count, auto graph_batch, auto dataset_batch, auto label_batch) {
        for (int64_t row = 0; row < count; ++row) {
          for (int64_t edge = 0; edge < static_cast<int64_t>(neighbors[0].size()); ++edge) {
            graph_batch(row, edge) = neighbors[start + row][edge];
          }
          for (int64_t column = 0; column < static_cast<int64_t>(vectors[0].size()); ++column) {
            dataset_batch(row, column) = vectors[start + row][column];
          }
          label_batch(row) = static_cast<uint32_t>(start + row);
        }
      });
  }
  std::ifstream shared_bytes(path, std::ios::binary);
  std::ifstream batched_bytes(batched_path, std::ios::binary);
  ASSERT_TRUE(shared_bytes && batched_bytes);
  std::vector<char> shared_contents((std::istreambuf_iterator<char>(shared_bytes)),
                                    std::istreambuf_iterator<char>());
  std::vector<char> batched_contents((std::istreambuf_iterator<char>(batched_bytes)),
                                     std::istreambuf_iterator<char>());
  EXPECT_EQ(shared_contents, batched_contents);

  EXPECT_NO_THROW({
    hnswlib::HierarchicalNSW<float> loaded(&space, path.string());
    EXPECT_EQ(loaded.getCurrentElementCount(), 4);
  });
  EXPECT_NO_THROW({
    hnswlib::HierarchicalNSW<float> loaded(&space, batched_path.string());
    EXPECT_EQ(loaded.getCurrentElementCount(), 4);
  });
}

}  // namespace
}  // namespace cuvs::neighbors::hnsw::detail::external
