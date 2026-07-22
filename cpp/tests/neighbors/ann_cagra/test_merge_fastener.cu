/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../src/neighbors/detail/cagra/cagra_merge.cuh"
#include "../naive_knn.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace cuvs::neighbors::cagra {
namespace {

template <typename T>
auto make_value(int64_t row, int64_t column) -> T
{
  int value = static_cast<int>((row * 7 + column * 3) % 23) - 11;
  if constexpr (std::is_same_v<T, half>) {
    return __float2half(static_cast<float>(value) / 8.0f);
  } else if constexpr (std::is_same_v<T, float>) {
    return static_cast<float>(value) / 8.0f;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return static_cast<uint8_t>(value + 32);
  } else {
    return static_cast<int8_t>(value);
  }
}

template <typename T>
auto make_dataset(raft::resources const& res, int64_t rows, int64_t dim, int64_t offset)
{
  auto dataset = raft::make_host_matrix<T, int64_t>(res, rows, dim);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < dim; ++j) {
      dataset(i, j) = make_value<T>(i + offset, j);
    }
  }
  return dataset;
}

auto make_ring_graph(raft::resources const& res, int64_t rows, int64_t degree)
{
  auto graph = raft::make_host_matrix<uint32_t, int64_t>(res, rows, degree);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < degree; ++j) {
      graph(i, j) = static_cast<uint32_t>((i + j + 1) % rows);
    }
  }
  return graph;
}

template <typename T>
void expect_valid_graph(index<T, uint32_t> const& merged, int64_t rows, int64_t degree)
{
  ASSERT_EQ(merged.size(), rows);
  ASSERT_EQ(merged.graph().extent(0), rows);
  ASSERT_EQ(merged.graph().extent(1), degree);
  auto host = raft::make_host_matrix<uint32_t, int64_t>(rows, degree);
  raft::resources res;
  raft::copy(host.data_handle(),
             merged.graph().data_handle(),
             host.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < degree; ++column) {
      EXPECT_LT(host(row, column), static_cast<uint32_t>(rows));
      EXPECT_NE(host(row, column), static_cast<uint32_t>(row));
    }
  }
}

template <typename T>
void expect_dataset_order(raft::resources const& res,
                          index<T, uint32_t> const& merged,
                          int64_t rows,
                          int64_t dim)
{
  ASSERT_EQ(merged.dataset().extent(0), rows);
  ASSERT_EQ(merged.dataset().extent(1), dim);
  auto host = raft::make_host_matrix<T, int64_t>(res, rows, dim);
  raft::copy(host.data_handle(),
             merged.dataset().data_handle(),
             host.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < dim; ++column) {
      EXPECT_EQ(static_cast<float>(host(row, column)),
                static_cast<float>(make_value<T>(row, column)));
    }
  }
}

template <typename T>
void run_explicit_fastener(cuvs::distance::DistanceType metric)
{
  raft::resources res;
  constexpr int64_t rows   = 48;
  constexpr int64_t dim    = 8;
  constexpr int64_t degree = 4;
  auto dataset0            = make_dataset<T>(res, rows, dim, 0);
  auto dataset1            = make_dataset<T>(res, rows, dim, rows);
  auto graph0              = make_ring_graph(res, rows, degree);
  auto graph1              = make_ring_graph(res, rows, degree);
  index<T, uint32_t> index0(
    res, metric, raft::make_const_mdspan(dataset0.view()), raft::make_const_mdspan(graph0.view()));
  index<T, uint32_t> index1(
    res, metric, raft::make_const_mdspan(dataset1.view()), raft::make_const_mdspan(graph1.view()));

  index_params params;
  params.metric                    = metric;
  params.graph_degree              = degree;
  params.intermediate_graph_degree = degree;
  params.attach_dataset_on_build   = true;
  params.guarantee_connectivity    = false;
  merge_params fastener;
  fastener.algo        = merge_algo::FASTENER;
  fastener.leaf_size   = 64;
  fastener.leaf_degree = 4;
  std::vector<index<T, uint32_t>*> indices{&index0, &index1};

  auto merged = merge(res, params, indices, fastener);
  expect_valid_graph(merged, rows * 2, degree);
  expect_dataset_order(res, merged, rows * 2, dim);
  EXPECT_TRUE(merged.data().is_owning());
  EXPECT_EQ(index0.dataset().extent(0), rows);
  EXPECT_EQ(index1.dataset().extent(0), rows);
  EXPECT_TRUE(index0.data().is_owning());
  EXPECT_TRUE(index1.data().is_owning());
  EXPECT_EQ(index0.size(), rows);
  EXPECT_EQ(index1.size(), rows);
  EXPECT_EQ(index0.dim(), dim);
  EXPECT_EQ(index1.dim(), dim);
}

TEST(CagraMergeFastener, PlanSplitIsDeterministicDenseAndCompact)
{
  using namespace detail::merge_scaffold;
  std::vector<partition_range> ranges{{0, 0, 40}, {1, 40, 300}, {2, 300, 1000}};
  split_params params{.fanout            = 2,
                      .leader_fraction   = 0.05,
                      .max_leaders       = 16,
                      .leaf_size         = 64,
                      .level             = 1,
                      .occurrence_stride = 2};

  auto plan   = plan_split(ranges, 1000, params, DETERMINISTIC_SEED);
  auto replay = plan_split(ranges, 1000, params, DETERMINISTIC_SEED);

  ASSERT_EQ(plan.parents.size(), ranges.size());
  EXPECT_TRUE(plan.parents[0].carried());
  EXPECT_FALSE(plan.parents[1].carried());
  EXPECT_FALSE(plan.parents[2].carried());

  int64_t output_cursor = 0;
  uint32_t key_cursor   = 0;
  for (size_t i = 0; i < plan.parents.size(); ++i) {
    auto const& entry = plan.parents[i];
    EXPECT_EQ(entry.output_start, output_cursor);
    EXPECT_EQ(entry.child_key_base, key_cursor);
    output_cursor += entry.output_rows(params.fanout);
    key_cursor += entry.child_count();
    if (!entry.carried()) {
      EXPECT_GE(entry.leader_count, static_cast<int32_t>(params.fanout));
      EXPECT_LE(entry.leader_count, static_cast<int32_t>(params.max_leaders));
      EXPECT_LT(entry.leader_offset, static_cast<uint32_t>(entry.size()));
    }
    EXPECT_EQ(entry.leader_offset, replay.parents[i].leader_offset);
  }
  EXPECT_EQ(plan.output_rows, output_cursor);
  EXPECT_EQ(plan.child_count, key_cursor);
  EXPECT_EQ(plan.output_rows, 40 + 260 * 2 + 700 * 2);

  std::vector<partition_range> overlapping{{0, 0, 40}, {1, 30, 100}};
  EXPECT_THROW(plan_split(overlapping, 100, params, DETERMINISTIC_SEED), raft::exception);
}

TEST(CagraMergeFastener, SplitManywayCarriesSmallParentsAndSupportsRepeatedLevels)
{
  using namespace detail::merge_scaffold;
  raft::resources res;
  auto stream            = raft::resource::get_cuda_stream(res);
  constexpr int64_t rows = 96;
  constexpr int64_t dim  = 4;
  auto host_dataset      = make_dataset<float>(res, rows, dim, 0);
  auto dataset           = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  raft::copy(dataset.data_handle(), host_dataset.data_handle(), dataset.size(), stream);

  split_context context(stream, rows);
  manyway_l2_norms_kernel<<<static_cast<int>((rows + 3) / 4), 128, 0, stream>>>(
    dataset.data_handle(), rows, dim, context.norms.data());
  RAFT_CUDA_TRY(cudaGetLastError());

  auto root   = make_root_partition(res, rows);
  auto first  = split_manyway(res,
                             raft::make_const_mdspan(dataset.view()),
                             root,
                             split_params{.fanout            = 2,
                                           .leader_fraction   = 0.01,
                                           .max_leaders       = 2,
                                           .leaf_size         = 64,
                                           .level             = 0,
                                           .occurrence_stride = 1},
                             context);
  auto second = split_manyway(res,
                              raft::make_const_mdspan(dataset.view()),
                              first,
                              split_params{.fanout            = 2,
                                           .leader_fraction   = 0.01,
                                           .max_leaders       = 2,
                                           .leaf_size         = 64,
                                           .level             = 1,
                                           .occurrence_stride = 2},
                              context);
  EXPECT_EQ(second.memberships.size(), static_cast<size_t>(rows * 4));

  int64_t cursor = 0;
  for (auto const& range : second.ranges) {
    EXPECT_EQ(range.start, cursor);
    EXPECT_GT(range.end, range.start);
    EXPECT_LE(range.end, static_cast<int64_t>(second.memberships.size()));
    cursor = range.end;
  }
  EXPECT_EQ(cursor, static_cast<int64_t>(second.memberships.size()));

  std::vector<partition_membership> records(second.memberships.size());
  raft::copy(records.data(), second.memberships.data(), records.size(), stream);
  raft::resource::sync_stream(res);
  for (auto const& record : records) {
    EXPECT_LT(record.id, static_cast<uint32_t>(rows));
    EXPECT_LT(record.occurrence, uint16_t{4});
  }

  constexpr int64_t small_rows = 32;
  auto small_root              = make_root_partition(res, small_rows);
  split_context small_context(stream, small_rows);
  auto carried = split_manyway(res,
                               raft::make_const_mdspan(dataset.view()),
                               small_root,
                               split_params{.fanout            = 3,
                                            .leader_fraction   = 0.5,
                                            .max_leaders       = 8,
                                            .leaf_size         = 64,
                                            .level             = 0,
                                            .occurrence_stride = 1},
                               small_context);
  ASSERT_EQ(carried.ranges.size(), 1);
  EXPECT_EQ(carried.memberships.size(), static_cast<size_t>(small_rows));
  std::vector<partition_membership> carried_records(carried.memberships.size());
  raft::copy(carried_records.data(), carried.memberships.data(), carried_records.size(), stream);
  raft::resource::sync_stream(res);
  for (int64_t row = 0; row < small_rows; ++row) {
    EXPECT_EQ(carried_records[row].id, static_cast<uint32_t>(row));
    EXPECT_EQ(carried_records[row].occurrence, uint16_t{0});
  }
}

TEST(CagraMergeFastener, InitializesUnwrittenScaffoldSlotsWithSelf)
{
  using namespace detail::merge_scaffold;
  raft::resources res;
  auto stream              = raft::resource::get_cuda_stream(res);
  constexpr int64_t rows   = 3;
  constexpr int64_t degree = 4;
  auto graph               = raft::make_device_matrix<uint32_t, int64_t>(res, rows, degree);

  initialize_self_scaffold_kernel<<<1, 256, 0, stream>>>(graph.data_handle(), rows, degree);
  RAFT_CUDA_TRY(cudaGetLastError());

  std::vector<uint32_t> initialized(graph.size());
  raft::copy(initialized.data(), graph.data_handle(), initialized.size(), stream);
  raft::resource::sync_stream(res);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < degree; ++column) {
      EXPECT_EQ(initialized[row * degree + column], static_cast<uint32_t>(row));
    }
  }
}

TEST(CagraMergeFastener, LeafGemmLimitsRejectOnlyUnrealisticIntegerDimensions)
{
  using namespace detail::merge_scaffold;
  EXPECT_TRUE(leaf_gemm_supported<float>(4096, 256));
  EXPECT_TRUE(leaf_gemm_supported<half>(4096, 256));
  EXPECT_TRUE(leaf_gemm_supported<int8_t>(MAX_INTEGER_LEAF_DIMENSION, 256));
  EXPECT_TRUE(leaf_gemm_supported<uint8_t>(MAX_INTEGER_LEAF_DIMENSION, 256));
  EXPECT_FALSE(leaf_gemm_supported<int8_t>(MAX_INTEGER_LEAF_DIMENSION + 1, 256));
  EXPECT_FALSE(leaf_gemm_supported<uint8_t>(MAX_INTEGER_LEAF_DIMENSION + 1, 256));

  raft::resources res;
  constexpr int64_t rows = 2;
  constexpr int64_t dim  = MAX_INTEGER_LEAF_DIMENSION + 1;
  auto dataset0          = make_dataset<int8_t>(res, rows, dim, 0);
  auto dataset1          = make_dataset<int8_t>(res, rows, dim, rows);
  auto graph0            = make_ring_graph(res, rows, 1);
  auto graph1            = make_ring_graph(res, rows, 1);
  auto metric            = cuvs::distance::DistanceType::L2Expanded;
  index<int8_t, uint32_t> index0(
    res, metric, raft::make_const_mdspan(dataset0.view()), raft::make_const_mdspan(graph0.view()));
  index<int8_t, uint32_t> index1(
    res, metric, raft::make_const_mdspan(dataset1.view()), raft::make_const_mdspan(graph1.view()));

  index_params params;
  params.metric                    = metric;
  params.graph_degree              = 1;
  params.intermediate_graph_degree = 1;
  merge_params fastener;
  fastener.algo      = merge_algo::FASTENER;
  fastener.leaf_size = 64;
  std::vector<index<int8_t, uint32_t>*> indices{&index0, &index1};

  auto result = detail::preflight_fastener(
    params, fastener, indices, cuvs::neighbors::filtering::none_sample_filter{});
  EXPECT_FALSE(result.eligible);
  EXPECT_EQ(result.reason, "integer dataset dimension exceeds the INT32 leaf GEMM limit");
  EXPECT_EQ(index0.dataset().extent(0), rows);
  EXPECT_EQ(index1.dataset().extent(0), rows);
}

TEST(CagraMergeFastener, SupportsAllScalarTypes)
{
  auto metric = cuvs::distance::DistanceType::L2Expanded;
  run_explicit_fastener<float>(metric);
  run_explicit_fastener<half>(metric);
  run_explicit_fastener<int8_t>(metric);
  run_explicit_fastener<uint8_t>(metric);
}

TEST(CagraMergeFastener, MixedDatasetOwnershipPreservesInputs)
{
  raft::resources res;
  constexpr int64_t rows   = 48;
  constexpr int64_t dim    = 8;
  constexpr int64_t degree = 4;
  auto dataset0            = make_dataset<float>(res, rows, dim, 0);
  auto dataset1            = make_dataset<float>(res, rows, dim, rows);
  auto device_dataset1     = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  raft::copy(device_dataset1.data_handle(),
             dataset1.data_handle(),
             dataset1.size(),
             raft::resource::get_cuda_stream(res));
  auto graph0 = make_ring_graph(res, rows, degree);
  auto graph1 = make_ring_graph(res, rows, degree);
  auto metric = cuvs::distance::DistanceType::L2Expanded;
  index<float, uint32_t> index0(
    res, metric, raft::make_const_mdspan(dataset0.view()), raft::make_const_mdspan(graph0.view()));
  index<float, uint32_t> index1(res,
                                metric,
                                raft::make_const_mdspan(device_dataset1.view()),
                                raft::make_const_mdspan(graph1.view()));
  ASSERT_TRUE(index0.data().is_owning());
  ASSERT_FALSE(index1.data().is_owning());

  index_params params;
  params.metric                    = metric;
  params.graph_degree              = degree;
  params.intermediate_graph_degree = degree;
  params.attach_dataset_on_build   = true;
  params.guarantee_connectivity    = false;
  merge_params fastener;
  fastener.algo        = merge_algo::FASTENER;
  fastener.leaf_size   = 64;
  fastener.leaf_degree = 4;
  std::vector<index<float, uint32_t>*> indices{&index0, &index1};

  auto merged = merge(res, params, indices, fastener);
  expect_valid_graph(merged, rows * 2, degree);
  expect_dataset_order(res, merged, rows * 2, dim);
  EXPECT_TRUE(merged.data().is_owning());
  EXPECT_EQ(index0.dataset().extent(0), rows);
  EXPECT_EQ(index1.dataset().extent(0), rows);
  EXPECT_TRUE(index0.data().is_owning());
  EXPECT_FALSE(index1.data().is_owning());
}

TEST(CagraMergeFastener, DefaultFloatMergeSearchRecallAgainstBruteForce)
{
  raft::resources res;
  auto stream                 = raft::resource::get_cuda_stream(res);
  constexpr int64_t part_rows = 256;
  constexpr int64_t rows      = 2 * part_rows;
  constexpr int64_t dim       = 16;
  constexpr int64_t queries   = 16;
  constexpr int64_t k         = 12;

  auto value = [](int64_t row, int64_t column) {
    uint64_t bits = (static_cast<uint64_t>(row) + 1) * 0x9e3779b97f4a7c15ull;
    bits ^= (static_cast<uint64_t>(column) + 1) * 0xbf58476d1ce4e5b9ull;
    bits ^= bits >> 30;
    bits *= 0x94d049bb133111ebull;
    bits ^= bits >> 31;
    return static_cast<float>(bits & 0xffffu) / 65535.0f;
  };

  auto dataset0 = raft::make_host_matrix<float, int64_t>(part_rows, dim);
  auto dataset1 = raft::make_host_matrix<float, int64_t>(part_rows, dim);
  for (int64_t row = 0; row < part_rows; ++row) {
    for (int64_t column = 0; column < dim; ++column) {
      dataset0(row, column) = value(row, column);
      dataset1(row, column) = value(row + part_rows, column);
    }
  }

  index_params partition_params;
  partition_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  partition_params.graph_degree              = 16;
  partition_params.intermediate_graph_degree = 32;
  partition_params.attach_dataset_on_build   = true;
  partition_params.guarantee_connectivity    = false;
  auto index0 = build(res, partition_params, raft::make_const_mdspan(dataset0.view()));
  auto index1 = build(res, partition_params, raft::make_const_mdspan(dataset1.view()));
  std::vector<index<float, uint32_t>*> indices{&index0, &index1};

  merge_params fastener;
  fastener.algo = merge_algo::FASTENER;
  auto merged   = merge(res, partition_params, indices, fastener);
  expect_valid_graph(merged, rows, partition_params.graph_degree);

  auto query_host = raft::make_host_matrix<float, int64_t>(queries, dim);
  for (int64_t query = 0; query < queries; ++query) {
    int64_t source = query * 31;
    for (int64_t column = 0; column < dim; ++column) {
      query_host(query, column) = value(source, column);
    }
  }
  auto query_device = raft::make_device_matrix<float, int64_t>(res, queries, dim);
  raft::copy(query_device.data_handle(), query_host.data_handle(), query_host.size(), stream);
  auto actual_indices   = raft::make_device_matrix<uint32_t, int64_t>(res, queries, k);
  auto actual_distances = raft::make_device_matrix<float, int64_t>(res, queries, k);
  search_params search_params;
  search_params.itopk_size     = 32;
  search_params.search_width   = 4;
  search_params.max_iterations = 0;
  search(res,
         search_params,
         merged,
         raft::make_const_mdspan(query_device.view()),
         actual_indices.view(),
         actual_distances.view());

  auto truth_indices   = raft::make_device_matrix<uint32_t, int64_t>(res, queries, k);
  auto truth_distances = raft::make_device_matrix<float, int64_t>(res, queries, k);
  cuvs::neighbors::naive_knn<float, float, uint32_t>(res,
                                                     truth_distances.data_handle(),
                                                     truth_indices.data_handle(),
                                                     query_device.data_handle(),
                                                     merged.dataset().data_handle(),
                                                     static_cast<uint32_t>(queries),
                                                     static_cast<uint32_t>(rows),
                                                     static_cast<uint32_t>(dim),
                                                     static_cast<uint32_t>(k),
                                                     cuvs::distance::DistanceType::L2Expanded);

  std::vector<uint32_t> actual(static_cast<size_t>(queries * k));
  std::vector<uint32_t> truth(static_cast<size_t>(queries * k));
  raft::copy(actual.data(), actual_indices.data_handle(), actual.size(), stream);
  raft::copy(truth.data(), truth_indices.data_handle(), truth.size(), stream);
  raft::resource::sync_stream(res);

  size_t matches = 0;
  for (int64_t query = 0; query < queries; ++query) {
    for (int64_t candidate = 0; candidate < k; ++candidate) {
      auto id = actual[query * k + candidate];
      for (int64_t expected = 0; expected < k; ++expected) {
        if (id == truth[query * k + expected]) {
          ++matches;
          break;
        }
      }
    }
  }
  EXPECT_GE(static_cast<double>(matches) / static_cast<double>(queries * k), 0.60);
}

TEST(CagraMergeFastener, MixedDegreesAndThreeLevelUint8OptionsProduceExactDegree)
{
  raft::resources res;
  constexpr int64_t rows = 64;
  constexpr int64_t dim  = 16;
  auto dataset0          = make_dataset<uint8_t>(res, rows, dim, 0);
  auto dataset1          = make_dataset<uint8_t>(res, rows, dim, rows);
  auto graph0            = make_ring_graph(res, rows, 6);
  auto graph1            = make_ring_graph(res, rows, 8);
  index<uint8_t, uint32_t> index0(res,
                                  cuvs::distance::DistanceType::L2Expanded,
                                  raft::make_const_mdspan(dataset0.view()),
                                  raft::make_const_mdspan(graph0.view()));
  index<uint8_t, uint32_t> index1(res,
                                  cuvs::distance::DistanceType::L2Expanded,
                                  raft::make_const_mdspan(dataset1.view()),
                                  raft::make_const_mdspan(graph1.view()));

  index_params params;
  params.metric                  = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree            = 8;
  params.attach_dataset_on_build = true;
  merge_params fastener;
  fastener.algo            = merge_algo::FASTENER;
  fastener.levels          = 3;
  fastener.root_fanout     = 2;
  fastener.lower_fanout    = 2;
  fastener.leader_fraction = 0.25;
  fastener.max_leaders     = 32;
  fastener.leaf_size       = 64;
  fastener.leaf_degree     = 8;
  std::vector<index<uint8_t, uint32_t>*> indices{&index0, &index1};

  auto merged = merge(res, params, indices, fastener);
  expect_valid_graph(merged, rows * 2, 8);
}

TEST(CagraMergeFastener, AppendCyclicallyPadsMixedInputDegrees)
{
  raft::resources res;
  auto dataset0 = make_dataset<float>(res, 2, 1, 0);
  auto dataset1 = make_dataset<float>(res, 2, 1, 2);
  auto graph0   = make_ring_graph(res, 2, 2);
  auto graph1   = make_ring_graph(res, 2, 4);
  index<float, uint32_t> index0(res,
                                cuvs::distance::DistanceType::L2Expanded,
                                raft::make_const_mdspan(dataset0.view()),
                                raft::make_const_mdspan(graph0.view()));
  index<float, uint32_t> index1(res,
                                cuvs::distance::DistanceType::L2Expanded,
                                raft::make_const_mdspan(dataset1.view()),
                                raft::make_const_mdspan(graph1.view()));
  std::vector<index<float, uint32_t>*> indices{&index0, &index1};
  std::vector<int64_t> offsets{0, 2, 4};

  auto scaffold_host        = raft::make_host_matrix<uint32_t, int64_t>(4, 1);
  uint32_t scaffold_rows[4] = {2, 3, 0, 1};
  for (int64_t row = 0; row < 4; ++row) {
    scaffold_host(row, 0) = scaffold_rows[row];
  }
  auto scaffold = raft::make_device_matrix<uint32_t, int64_t>(res, 4, 1);
  raft::copy(scaffold.data_handle(),
             scaffold_host.data_handle(),
             scaffold.size(),
             raft::resource::get_cuda_stream(res));

  auto appended = detail::merge_scaffold::append_to_input_graphs<float, uint32_t>(
    res, indices, offsets, raft::make_const_mdspan(scaffold.view()));
  ASSERT_EQ(appended.extent(0), 4);
  ASSERT_EQ(appended.extent(1), 5);

  auto host = raft::make_host_matrix<uint32_t, int64_t>(4, 5);
  raft::copy(
    host.data_handle(), appended.data_handle(), host.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // The degree-2 input repeats cyclically to the base width of 4; the degree-4 input copies
  // through shifted by its offset; the scaffold column is already global.
  uint32_t expected[4][5] = {{1, 0, 1, 0, 2}, {0, 1, 0, 1, 3}, {3, 2, 3, 2, 0}, {2, 3, 2, 3, 1}};
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t column = 0; column < 5; ++column) {
      EXPECT_EQ(host(row, column), expected[row][column]) << row << "," << column;
    }
  }
}

TEST(CagraMergeFastener, MergeSupportsOutputDegreeBelowInputDegree)
{
  raft::resources res;
  constexpr int64_t rows         = 48;
  constexpr int64_t dim          = 8;
  constexpr int64_t input_degree = 8;
  auto dataset0                  = make_dataset<float>(res, rows, dim, 0);
  auto dataset1                  = make_dataset<float>(res, rows, dim, rows);
  auto graph0                    = make_ring_graph(res, rows, input_degree);
  auto graph1                    = make_ring_graph(res, rows, input_degree);
  index<float, uint32_t> index0(res,
                                cuvs::distance::DistanceType::L2Expanded,
                                raft::make_const_mdspan(dataset0.view()),
                                raft::make_const_mdspan(graph0.view()));
  index<float, uint32_t> index1(res,
                                cuvs::distance::DistanceType::L2Expanded,
                                raft::make_const_mdspan(dataset1.view()),
                                raft::make_const_mdspan(graph1.view()));

  index_params params;
  params.metric                  = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree            = 4;
  params.attach_dataset_on_build = true;
  merge_params fastener;
  fastener.algo        = merge_algo::FASTENER;
  fastener.leaf_size   = 64;
  fastener.leaf_degree = 4;
  std::vector<index<float, uint32_t>*> indices{&index0, &index1};

  auto merged = merge(res, params, indices, fastener);
  expect_valid_graph(merged, rows * 2, params.graph_degree);
}

auto make_float_indices(raft::resources const& res,
                        cuvs::distance::DistanceType metric,
                        raft::host_matrix<float, int64_t>& dataset0,
                        raft::host_matrix<float, int64_t>& dataset1,
                        raft::host_matrix<uint32_t, int64_t>& graph0,
                        raft::host_matrix<uint32_t, int64_t>& graph1)
{
  std::vector<index<float, uint32_t>> output;
  output.emplace_back(
    res, metric, raft::make_const_mdspan(dataset0.view()), raft::make_const_mdspan(graph0.view()));
  output.emplace_back(
    res, metric, raft::make_const_mdspan(dataset1.view()), raft::make_const_mdspan(graph1.view()));
  return output;
}

TEST(CagraMergeFastener, InvalidManywayOptionsFailPreflightWithoutMutation)
{
  raft::resources res;
  constexpr int64_t rows = 32;
  constexpr int64_t dim  = 8;
  auto dataset0          = make_dataset<float>(res, rows, dim, 0);
  auto dataset1          = make_dataset<float>(res, rows, dim, rows);
  auto graph0            = make_ring_graph(res, rows, 4);
  auto graph1            = make_ring_graph(res, rows, 4);
  index_params params;
  params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree              = 4;
  params.intermediate_graph_degree = 8;
  params.attach_dataset_on_build   = true;
  auto owned = make_float_indices(res, params.metric, dataset0, dataset1, graph0, graph1);
  std::vector<index<float, uint32_t>*> indices{&owned[0], &owned[1]};

  std::vector<merge_params> invalid;
  auto add_invalid = [&](auto update) {
    merge_params candidate;
    candidate.algo = merge_algo::FASTENER;
    update(candidate);
    invalid.push_back(candidate);
  };
  add_invalid([](auto& value) { value.levels = 0; });
  add_invalid([](auto& value) { value.root_fanout = 0; });
  add_invalid([](auto& value) { value.root_fanout = 33; });
  add_invalid([](auto& value) { value.lower_fanout = 0; });
  add_invalid([](auto& value) { value.lower_fanout = 33; });
  add_invalid([](auto& value) { value.leader_fraction = 0.0; });
  add_invalid([](auto& value) { value.leader_fraction = 1.1; });
  add_invalid(
    [](auto& value) { value.leader_fraction = std::numeric_limits<double>::quiet_NaN(); });
  add_invalid([](auto& value) { value.max_leaders = 0; });
  add_invalid([](auto& value) { value.max_leaders = 8193; });
  add_invalid([](auto& value) { value.max_leaders = 2; });
  add_invalid([](auto& value) { value.leaf_size = 0; });
  add_invalid([](auto& value) { value.leaf_size = 512; });
  add_invalid([](auto& value) { value.leaf_degree = 0; });
  add_invalid([](auto& value) { value.leaf_degree = 16; });
  add_invalid([](auto& value) {
    value.levels       = 3;
    value.root_fanout  = 32;
    value.lower_fanout = 32;
  });

  for (auto const& candidate : invalid) {
    auto result = detail::preflight_fastener(
      params, candidate, indices, cuvs::neighbors::filtering::none_sample_filter{});
    EXPECT_FALSE(result.eligible) << result.reason;
  }
  EXPECT_EQ(owned[0].dataset().extent(0), rows);
  EXPECT_EQ(owned[1].dataset().extent(0), rows);
}

TEST(CagraMergeFastener, DispatchRejectsOrFallsBackBeforeMutation)
{
  raft::resources res;
  constexpr int64_t rows = 32;
  constexpr int64_t dim  = 8;
  auto dataset0          = make_dataset<float>(res, rows, dim, 0);
  auto dataset1          = make_dataset<float>(res, rows, dim, rows);
  auto graph0            = make_ring_graph(res, rows, 4);
  auto graph1            = make_ring_graph(res, rows, 4);
  index_params params;
  params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree              = 4;
  params.intermediate_graph_degree = 8;
  params.attach_dataset_on_build   = true;

  {
    auto owned = make_float_indices(res, params.metric, dataset0, dataset1, graph0, graph1);
    std::vector<index<float, uint32_t>*> indices{&owned[0], &owned[1]};
    merge_params unsupported;
    unsupported.algo      = merge_algo::FASTENER;
    unsupported.leaf_size = 512;
    EXPECT_ANY_THROW(merge(res, params, indices, unsupported));
    EXPECT_EQ(owned[0].dataset().extent(0), rows);
    EXPECT_EQ(owned[1].dataset().extent(0), rows);
  }
  {
    auto inner_product_params   = params;
    inner_product_params.metric = cuvs::distance::DistanceType::InnerProduct;
    auto owned =
      make_float_indices(res, inner_product_params.metric, dataset0, dataset1, graph0, graph1);
    std::vector<index<float, uint32_t>*> indices{&owned[0], &owned[1]};
    merge_params fastener;
    fastener.algo      = merge_algo::FASTENER;
    fastener.leaf_size = 64;
    EXPECT_ANY_THROW(merge(res, inner_product_params, indices, fastener));
    EXPECT_EQ(owned[0].dataset().extent(0), rows);
    EXPECT_EQ(owned[1].dataset().extent(0), rows);
  }
  {
    auto owned = make_float_indices(res, params.metric, dataset0, dataset1, graph0, graph1);
    std::vector<index<float, uint32_t>*> indices{&owned[0], &owned[1]};
    merge_params automatic;
    automatic.algo      = merge_algo::AUTO;
    automatic.leaf_size = 512;
    auto merged         = merge(res, params, indices, automatic);
    EXPECT_EQ(merged.size(), rows * 2);
    EXPECT_EQ(owned[0].dataset().extent(0), rows);
    EXPECT_EQ(owned[1].dataset().extent(0), rows);
  }
  {
    auto owned = make_float_indices(res, params.metric, dataset0, dataset1, graph0, graph1);
    std::vector<index<float, uint32_t>*> indices{&owned[0], &owned[1]};
    merge_params rebuild{merge_algo::REBUILD};
    auto merged = merge(res, params, indices, rebuild);
    EXPECT_EQ(merged.size(), rows * 2);
    EXPECT_EQ(owned[0].dataset().extent(0), rows);
    EXPECT_EQ(owned[1].dataset().extent(0), rows);
  }
}

TEST(CagraMergeFastener, InnerProductOrderingDiffersFromL2)
{
  raft::resources res;
  auto dataset        = raft::make_host_matrix<float, int64_t>(3, 2);
  dataset(0, 0)       = 1.0f;
  dataset(0, 1)       = 0.0f;
  dataset(1, 0)       = 2.0f;
  dataset(1, 1)       = 0.0f;
  dataset(2, 0)       = 100.0f;
  dataset(2, 1)       = 100.0f;
  auto device_dataset = raft::make_device_matrix<float, int64_t>(res, 3, 2);
  raft::copy(device_dataset.data_handle(),
             dataset.data_handle(),
             dataset.size(),
             raft::resource::get_cuda_stream(res));

  auto source  = raft::make_host_matrix<uint32_t, int64_t>(3, 2);
  source(0, 0) = 1;
  source(0, 1) = 2;
  source(1, 0) = 0;
  source(1, 1) = 2;
  source(2, 0) = 0;
  source(2, 1) = 1;
  auto l2      = raft::make_device_matrix<uint32_t, int64_t>(res, 3, 2);
  auto ip      = raft::make_device_matrix<uint32_t, int64_t>(res, 3, 2);
  raft::copy(
    l2.data_handle(), source.data_handle(), source.size(), raft::resource::get_cuda_stream(res));
  raft::copy(
    ip.data_handle(), source.data_handle(), source.size(), raft::resource::get_cuda_stream(res));

  detail::graph::sort_knn_graph_device_inplace(res,
                                               cuvs::distance::DistanceType::L2Expanded,
                                               raft::make_const_mdspan(device_dataset.view()),
                                               l2.view());
  detail::graph::sort_knn_graph_device_inplace(res,
                                               cuvs::distance::DistanceType::InnerProduct,
                                               raft::make_const_mdspan(device_dataset.view()),
                                               ip.view());
  auto l2_host = raft::make_host_matrix<uint32_t, int64_t>(3, 2);
  auto ip_host = raft::make_host_matrix<uint32_t, int64_t>(3, 2);
  raft::copy(
    l2_host.data_handle(), l2.data_handle(), l2.size(), raft::resource::get_cuda_stream(res));
  raft::copy(
    ip_host.data_handle(), ip.data_handle(), ip.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  EXPECT_EQ(l2_host(0, 0), 1);
  EXPECT_EQ(ip_host(0, 0), 2);
}

TEST(CagraMergeFastener, Uint8InnerProductSortUsesExactWideAccumulation)
{
  raft::resources res;
  constexpr int64_t dim = 1024;
  auto dataset          = raft::make_host_matrix<uint8_t, int64_t>(3, dim);
  for (int64_t d = 0; d < dim; ++d) {
    dataset(0, d) = 255;
    dataset(1, d) = 255;
    dataset(2, d) = 255;
  }
  dataset(0, 0)       = 1;
  dataset(1, 0)       = 1;
  dataset(2, 0)       = 0;
  auto device_dataset = raft::make_device_matrix<uint8_t, int64_t>(res, 3, dim);
  raft::copy(device_dataset.data_handle(),
             dataset.data_handle(),
             dataset.size(),
             raft::resource::get_cuda_stream(res));

  auto graph_host  = raft::make_host_matrix<uint32_t, int64_t>(3, 2);
  graph_host(0, 0) = 2;
  graph_host(0, 1) = 1;
  graph_host(1, 0) = 0;
  graph_host(1, 1) = 2;
  graph_host(2, 0) = 0;
  graph_host(2, 1) = 1;
  auto graph       = raft::make_device_matrix<uint32_t, int64_t>(res, 3, 2);
  raft::copy(graph.data_handle(),
             graph_host.data_handle(),
             graph.size(),
             raft::resource::get_cuda_stream(res));
  detail::graph::sort_knn_graph_device_inplace(res,
                                               cuvs::distance::DistanceType::InnerProduct,
                                               raft::make_const_mdspan(device_dataset.view()),
                                               graph.view());
  raft::copy(graph_host.data_handle(),
             graph.data_handle(),
             graph.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  EXPECT_EQ(graph_host(0, 0), 1);
}

TEST(CagraMergeFastener, AppendSortAndDedupHandleOffsetsAndPadding)
{
  raft::resources res;
  auto dataset0  = raft::make_host_matrix<float, int64_t>(2, 1);
  auto dataset1  = raft::make_host_matrix<float, int64_t>(2, 1);
  dataset0(0, 0) = 0.0f;
  dataset0(1, 0) = 2.0f;
  dataset1(0, 0) = 10.0f;
  dataset1(1, 0) = 12.0f;
  auto graph0    = make_ring_graph(res, 2, 1);
  auto graph1    = make_ring_graph(res, 2, 1);
  index<float, uint32_t> index0(res,
                                cuvs::distance::DistanceType::L2Expanded,
                                raft::make_const_mdspan(dataset0.view()),
                                raft::make_const_mdspan(graph0.view()));
  index<float, uint32_t> index1(res,
                                cuvs::distance::DistanceType::L2Expanded,
                                raft::make_const_mdspan(dataset1.view()),
                                raft::make_const_mdspan(graph1.view()));
  std::vector<index<float, uint32_t>*> indices{&index0, &index1};
  std::vector<int64_t> offsets{0, 2, 4};

  auto scaffold_host           = raft::make_host_matrix<uint32_t, int64_t>(4, 2);
  uint32_t scaffold_rows[4][2] = {{1, 3}, {1, 2}, {2, 0}, {3, 1}};
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t column = 0; column < 2; ++column) {
      scaffold_host(row, column) = scaffold_rows[row][column];
    }
  }
  auto scaffold = raft::make_device_matrix<uint32_t, int64_t>(res, 4, 2);
  raft::copy(scaffold.data_handle(),
             scaffold_host.data_handle(),
             scaffold.size(),
             raft::resource::get_cuda_stream(res));
  auto appended = detail::merge_scaffold::append_to_input_graphs<float, uint32_t>(
    res, indices, offsets, raft::make_const_mdspan(scaffold.view()));

  auto combined_host  = raft::make_host_matrix<float, int64_t>(4, 1);
  combined_host(0, 0) = 0.0f;
  combined_host(1, 0) = 2.0f;
  combined_host(2, 0) = 10.0f;
  combined_host(3, 0) = 12.0f;
  auto combined       = raft::make_device_matrix<float, int64_t>(res, 4, 1);
  raft::copy(combined.data_handle(),
             combined_host.data_handle(),
             combined.size(),
             raft::resource::get_cuda_stream(res));
  detail::graph::sort_knn_graph_device_inplace(res,
                                               cuvs::distance::DistanceType::L2Expanded,
                                               raft::make_const_mdspan(combined.view()),
                                               appended.view());
  auto output =
    detail::merge_scaffold::cap_sorted_graph(res, raft::make_const_mdspan(appended.view()), 3);
  auto host = raft::make_host_matrix<uint32_t, int64_t>(4, 3);
  raft::copy(
    host.data_handle(), output.data_handle(), host.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  uint32_t expected[4][3] = {{1, 3, 1}, {0, 2, 0}, {3, 0, 3}, {2, 1, 2}};
  for (int64_t row = 0; row < 4; ++row) {
    for (int64_t column = 0; column < 3; ++column) {
      EXPECT_EQ(host(row, column), expected[row][column]);
    }
  }
}

}  // namespace
}  // namespace cuvs::neighbors::cagra
