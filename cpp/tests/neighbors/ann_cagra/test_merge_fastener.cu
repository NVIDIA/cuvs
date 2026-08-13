/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief Component and end-to-end coverage for CAGRA Fastener merges.
 */

#include "../../../src/neighbors/detail/cagra/cagra_merge.cuh"
#include "../ann_cagra.cuh"
#include "../cagra_padded_build_helpers.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

namespace cuvs::neighbors::cagra {
namespace {

/** Padded device storage plus a view of it.
 *
 * The new dataset API makes an index hold only a *view*, so the storage must outlive every index
 * built from it. Keeping both together in one object makes that lifetime explicit. Rows are padded
 * to the CAGRA-required width and zero-filled, matching what cagra::merge produces.
 */
/** plan_split now takes a raft::host_vector; tests still express ranges as literals. */
inline auto make_ranges(raft::resources const& res,
                        std::initializer_list<detail::merge_scaffold::partition_range> items)
  -> raft::host_vector<detail::merge_scaffold::partition_range, int64_t>
{
  auto out = raft::make_host_vector<detail::merge_scaffold::partition_range, int64_t>(
    res, static_cast<int64_t>(items.size()));
  int64_t i = 0;
  for (auto const& item : items) {
    out(i++) = item;
  }
  return out;
}

template <typename T>
struct padded_storage {
  raft::device_matrix<T, int64_t> matrix;
  cuvs::neighbors::device_padded_dataset_view<T, int64_t> view;
};

template <typename T>
auto make_padded(raft::resources const& res, raft::host_matrix_view<const T, int64_t> src)
  -> padded_storage<T>
{
  auto stream       = raft::resource::get_cuda_stream(res);
  auto const dim    = static_cast<uint32_t>(src.extent(1));
  auto const stride = cuvs::neighbors::cagra_required_row_width<T>(dim, 16);
  auto matrix =
    raft::make_device_matrix<T, int64_t>(res, src.extent(0), static_cast<int64_t>(stride));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    matrix.data_handle(), 0, static_cast<size_t>(matrix.size()) * sizeof(T), stream));
  raft::copy_matrix(matrix.data_handle(),
                    static_cast<size_t>(stride),
                    src.data_handle(),
                    static_cast<size_t>(dim),
                    static_cast<size_t>(dim),
                    static_cast<size_t>(src.extent(0)),
                    stream);
  raft::resource::sync_stream(res);
  cuvs::neighbors::device_padded_dataset_view<T, int64_t> view(
    raft::make_const_mdspan(matrix.view()), dim);
  return padded_storage<T>{std::move(matrix), view};
}

/** Allocate merged-dataset storage matching what cagra::merge expects for `rows` of `dim`. */
template <typename T>
auto make_merged_storage(raft::resources const& res, int64_t rows, int64_t dim) -> padded_storage<T>
{
  auto const stride = cuvs::neighbors::cagra_required_row_width<T>(static_cast<uint32_t>(dim), 16);
  auto matrix       = raft::make_device_matrix<T, int64_t>(res, rows, static_cast<int64_t>(stride));
  cuvs::neighbors::device_padded_dataset_view<T, int64_t> view(
    raft::make_const_mdspan(matrix.view()), static_cast<uint32_t>(dim));
  return padded_storage<T>{std::move(matrix), view};
}

template <typename T>
auto make_dataset(raft::resources const& res,
                  int64_t rows,
                  int64_t dim,
                  uint64_t seed                       = 1234ULL,
                  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
{
  auto device = raft::make_device_matrix<T, int64_t>(res, rows, dim);
  raft::random::RngState rng(seed);
  InitDataset(res,
              device.data_handle(),
              static_cast<std::uint32_t>(rows),
              static_cast<std::uint32_t>(dim),
              metric,
              rng);
  auto host   = raft::make_host_matrix<T, int64_t>(res, rows, dim);
  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(host.data_handle(), device.data_handle(), host.size(), stream);
  raft::resource::sync_stream(res);
  return host;
}

template <typename T>
auto concat_host_datasets(raft::resources const& res,
                          raft::host_matrix_view<const T, int64_t> first,
                          raft::host_matrix_view<const T, int64_t> second)
{
  EXPECT_EQ(first.extent(1), second.extent(1));
  auto out =
    raft::make_host_matrix<T, int64_t>(res, first.extent(0) + second.extent(0), first.extent(1));
  std::copy_n(first.data_handle(), first.size(), out.data_handle());
  std::copy_n(second.data_handle(), second.size(), out.data_handle() + first.size());
  return out;
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
void expect_valid_graph(cagra::device_padded_index<T, uint32_t> const& merged,
                        int64_t rows,
                        int64_t degree)
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

/** Weakly-connected component count of the merged graph, treating every edge as undirected.
 * `expect_valid_graph` only checks that entries are in range and not self-loops, so it cannot see a
 * merge that left the input graphs as separate components. */
template <typename T>
auto count_connected_components(cagra::device_padded_index<T, uint32_t> const& merged) -> int64_t
{
  raft::resources res;
  int64_t rows   = merged.graph().extent(0);
  int64_t degree = merged.graph().extent(1);
  auto host      = raft::make_host_matrix<uint32_t, int64_t>(rows, degree);
  raft::copy(host.data_handle(),
             merged.graph().data_handle(),
             host.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  std::vector<int64_t> parent(static_cast<size_t>(rows));
  std::iota(parent.begin(), parent.end(), int64_t{0});
  auto find = [&parent](int64_t node) {
    while (parent[node] != node) {
      parent[node] = parent[parent[node]];
      node         = parent[node];
    }
    return node;
  };
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < degree; ++column) {
      int64_t left  = find(row);
      int64_t right = find(static_cast<int64_t>(host(row, column)));
      if (left != right) { parent[left] = right; }
    }
  }
  int64_t components = 0;
  for (int64_t row = 0; row < rows; ++row) {
    if (find(row) == row) { ++components; }
  }
  return components;
}

template <typename T>
void expect_dataset_order(raft::resources const& res,
                          cagra::device_padded_index<T, uint32_t> const& merged,
                          raft::host_matrix_view<const T, int64_t> expected)
{
  auto view = merged.dataset();
  ASSERT_EQ(view.n_rows(), expected.extent(0));
  ASSERT_EQ(static_cast<int64_t>(view.dim()), expected.extent(1));
  // The merged dataset is padded to a 16-byte row stride. For int8/uint8 (and any dim whose byte
  // width is not a multiple of 16) the stride exceeds the logical dimension, so rows must be copied
  // honouring that stride rather than as one contiguous block.
  auto host   = raft::make_host_matrix<T, int64_t>(res, expected.extent(0), expected.extent(1));
  auto stream = raft::resource::get_cuda_stream(res);
  int64_t const row_stride = static_cast<int64_t>(view.stride());
  int64_t const dim        = static_cast<int64_t>(view.dim());
  for (int64_t row = 0; row < view.n_rows(); ++row) {
    raft::copy(
      host.data_handle() + row * dim, view.view().data_handle() + row * row_stride, dim, stream);
  }
  raft::resource::sync_stream(res);
  for (int64_t row = 0; row < expected.extent(0); ++row) {
    for (int64_t column = 0; column < expected.extent(1); ++column) {
      EXPECT_EQ(static_cast<float>(host(row, column)), static_cast<float>(expected(row, column)));
    }
  }
}

template <typename T>
void expect_zero_padding(raft::resources const& res,
                         cagra::device_padded_index<T, uint32_t> const& merged)
{
  auto view            = merged.dataset();
  int64_t const rows   = view.n_rows();
  int64_t const dim    = static_cast<int64_t>(view.dim());
  int64_t const stride = static_cast<int64_t>(view.stride());
  std::vector<T> host(static_cast<std::size_t>(rows * stride));
  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(host.data(), view.view().data_handle(), host.size(), stream);
  raft::resource::sync_stream(res);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = dim; column < stride; ++column) {
      EXPECT_EQ(static_cast<float>(host[row * stride + column]), 0.0f);
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
  auto dataset0            = make_dataset<T>(res, rows, dim, 1234ULL, metric);
  auto dataset1            = make_dataset<T>(res, rows, dim, 5678ULL, metric);
  auto expected            = concat_host_datasets<T>(
    res, raft::make_const_mdspan(dataset0.view()), raft::make_const_mdspan(dataset1.view()));
  auto graph0         = make_ring_graph(res, rows, degree);
  auto graph1         = make_ring_graph(res, rows, degree);
  auto index0_storage = make_padded<T>(res, raft::make_const_mdspan(dataset0.view()));
  cagra::index<T, uint32_t> index0(
    res, metric, index0_storage.view, raft::make_const_mdspan(graph0.view()));
  auto index1_storage = make_padded<T>(res, raft::make_const_mdspan(dataset1.view()));
  cagra::index<T, uint32_t> index1(
    res, metric, index1_storage.view, raft::make_const_mdspan(graph1.view()));
  // Ownership is expressed by the view type now, so record the observable shape instead.
  auto const index0_rows = index0.dataset().n_rows();
  auto const index1_rows = index1.dataset().n_rows();

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
  std::vector<cagra::device_padded_index<T, uint32_t>*> indices{&index0, &index1};

  auto merged_storage = make_merged_storage<T>(res, rows * 2, dim);
  RAFT_CUDA_TRY(cudaMemsetAsync(merged_storage.matrix.data_handle(),
                                0xff,
                                merged_storage.matrix.size() * sizeof(T),
                                raft::resource::get_cuda_stream(res)));
  auto merged = merge(res, params, indices, merged_storage.view, fastener);
  expect_valid_graph(merged, rows * 2, degree);
  expect_dataset_order(res, merged, raft::make_const_mdspan(expected.view()));
  expect_zero_padding(res, merged);
  EXPECT_EQ(index0.dataset().n_rows(), rows);
  EXPECT_EQ(index1.dataset().n_rows(), rows);
  EXPECT_EQ(index0.dataset().n_rows(), index0_rows);
  EXPECT_EQ(index1.dataset().n_rows(), index1_rows);
  EXPECT_EQ(index0.size(), rows);
  EXPECT_EQ(index1.size(), rows);
  EXPECT_EQ(index0.dim(), dim);
  EXPECT_EQ(index1.dim(), dim);
}

/**
 * Verifies that split planning is deterministic, lays out child keys and output ranges densely,
 * carries completed parents forward, and rejects overlapping parent ranges.
 */
TEST(CagraMergeFastener, PlanSplitIsDeterministicDenseAndCompact)
{
  using namespace detail::merge_scaffold;
  raft::resources res;
  auto ranges = make_ranges(res, {{0, 0, 40}, {1, 40, 300}, {2, 300, 1000}});
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

  auto overlapping = make_ranges(res, {{0, 0, 40}, {1, 30, 100}});
  EXPECT_THROW(plan_split(overlapping, 100, params, DETERMINISTIC_SEED), raft::exception);
}

/**
 * Verifies that repeated levels use the same many-way split operation, emit valid memberships and
 * contiguous ranges, and preserve every membership of parents already within the leaf-size limit.
 */
TEST(CagraMergeFastener, SplitManywayCarriesSmallParentsAndSupportsRepeatedLevels)
{
  using namespace detail::merge_scaffold;
  raft::resources res;
  auto stream            = raft::resource::get_cuda_stream(res);
  constexpr int64_t rows = 96;
  constexpr int64_t dim  = 4;
  auto host_dataset      = make_dataset<float>(res, rows, dim);
  auto dataset           = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  raft::copy(dataset.data_handle(), host_dataset.data_handle(), dataset.size(), stream);

  split_context context(res, rows, dim);
  manyway_l2_norms_kernel<<<static_cast<int>((rows + 3) / 4), 128, 0, stream>>>(
    dataset.data_handle(), rows, dim, dim, context.norms.data_handle());
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
  for (int64_t range_index = 0; range_index < second.ranges.extent(0); ++range_index) {
    auto const& range = second.ranges(range_index);
    EXPECT_EQ(range.start, cursor);
    EXPECT_GT(range.end, range.start);
    EXPECT_LE(range.end, static_cast<int64_t>(second.memberships.size()));
    cursor = range.end;
  }
  EXPECT_EQ(cursor, static_cast<int64_t>(second.memberships.size()));

  std::vector<partition_membership> records(second.memberships.size());
  raft::copy(records.data(), second.memberships.data_handle(), records.size(), stream);
  raft::resource::sync_stream(res);
  for (auto const& record : records) {
    EXPECT_LT(record.id, static_cast<uint32_t>(rows));
    EXPECT_LT(record.occurrence, uint16_t{4});
  }

  constexpr int64_t small_rows = 32;
  auto small_root              = make_root_partition(res, small_rows);
  split_context small_context(res, small_rows, dim);
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
  raft::copy(
    carried_records.data(), carried.memberships.data_handle(), carried_records.size(), stream);
  raft::resource::sync_stream(res);
  for (int64_t row = 0; row < small_rows; ++row) {
    EXPECT_EQ(carried_records[row].id, static_cast<uint32_t>(row));
    EXPECT_EQ(carried_records[row].occurrence, uint16_t{0});
  }
}

/**
 * Verifies that scaffold slots not populated by a leaf neighbor are initialized with the source
 * row, providing a valid sentinel for later graph combination and compaction.
 */
TEST(CagraMergeFastener, InitializesUnwrittenScaffoldSlotsWithSelf)
{
  using namespace detail::merge_scaffold;
  raft::resources res;
  auto stream                        = raft::resource::get_cuda_stream(res);
  constexpr int64_t rows             = 3;
  constexpr int64_t scaffold_offset  = 2;
  constexpr int64_t scaffold_degree  = 4;
  constexpr int64_t candidate_degree = scaffold_offset + scaffold_degree;
  auto graph = raft::make_device_matrix<uint32_t, int64_t>(res, rows, candidate_degree);

  RAFT_CUDA_TRY(
    cudaMemsetAsync(graph.data_handle(), 0xff, graph.size() * sizeof(uint32_t), stream));
  launch_initialize_self_scaffold(
    res, graph.data_handle(), rows, candidate_degree, scaffold_offset, scaffold_degree);

  std::vector<uint32_t> initialized(graph.size());
  raft::copy(initialized.data(), graph.data_handle(), initialized.size(), stream);
  raft::resource::sync_stream(res);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < candidate_degree; ++column) {
      auto expected = column < scaffold_offset ? std::numeric_limits<uint32_t>::max()
                                               : static_cast<uint32_t>(row);
      EXPECT_EQ(initialized[row * candidate_degree + column], expected);
    }
  }
}

/**
 * Builds a many-way scaffold over two input ranges that share one exact duplicate vector and
 * checks that partitioning plus intra-leaf KNN place each duplicate in the other's scaffold
 * neighbors. Identical points select the same leaders, so they co-occur in the spilled children;
 * zero distance then makes them mutual cross-origin nearest neighbors inside those leaves.
 */
TEST(CagraMergeFastener, DuplicateAcrossInputsAppearInEachOthersScaffoldNeighbors)
{
  using namespace detail::merge_scaffold;
  raft::resources res;
  auto stream                    = raft::resource::get_cuda_stream(res);
  constexpr int64_t part0_rows   = 8;
  constexpr int64_t part1_rows   = 8;
  constexpr int64_t rows         = part0_rows + part1_rows;
  constexpr int64_t dim          = 8;
  constexpr uint32_t duplicate0  = 3;
  constexpr uint32_t duplicate1  = static_cast<uint32_t>(part0_rows + 5);
  constexpr uint32_t leaf_degree = 2;

  auto host = raft::make_host_matrix<float, int64_t>(res, rows, dim);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < dim; ++column) {
      host(row, column) = static_cast<float>(row * 17 + column);
    }
  }
  for (int64_t column = 0; column < dim; ++column) {
    host(duplicate1, column) = host(duplicate0, column);
  }

  auto dataset = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  raft::copy(dataset.data_handle(), host.data_handle(), dataset.size(), stream);

  // Force a real root split (rows > leaf_size). Dense leader sampling keeps each spilled child
  // small enough that make_leaves does not separate the co-assigned duplicates.
  build_params params;
  params.levels          = 1;
  params.root_fanout     = 2;
  params.lower_fanout    = 2;
  params.leader_fraction = 1.0;
  params.max_leaders     = 16;
  params.leaf_size       = 8;
  params.leaf_degree     = leaf_degree;

  std::vector<int64_t> offsets{0, part0_rows, rows};
  auto scaffold = build(res, raft::make_const_mdspan(dataset.view()), dim, offsets, params);
  ASSERT_EQ(scaffold.extent(0), rows);
  ASSERT_EQ(scaffold.extent(1), params.root_fanout * leaf_degree);

  auto scaffold_host =
    raft::make_host_matrix<uint32_t, int64_t>(res, scaffold.extent(0), scaffold.extent(1));
  raft::copy(scaffold_host.data_handle(), scaffold.data_handle(), scaffold.size(), stream);
  raft::resource::sync_stream(res);

  auto row_contains = [&](uint32_t row, uint32_t neighbor) {
    for (int64_t column = 0; column < scaffold_host.extent(1); ++column) {
      if (scaffold_host(row, column) == neighbor) { return true; }
    }
    return false;
  };

  EXPECT_TRUE(row_contains(duplicate0, duplicate1));
  EXPECT_TRUE(row_contains(duplicate1, duplicate0));
}

/**
 * Scaffold construction should be deterministic, and if future changes break that determinism,
 * they should do so deliberately.
 *
 * Everything the merge contributes ahead of graph::optimize is deterministic by design: the leader
 * sample is a fixed seed and a strided offset, assignment is a batched GEMM plus a deterministic
 * select_k, and every regroup is a stable sort. graph::optimize is not deterministic because its
 * reverse graph is built with an atomic bump allocator, so the adjacency order, and therefore the
 * pruning outcome, varies between runs. Making that deterministic is out of scope, but we still
 * want to be checking we're not accidentally introducing non-determinism.
 *
 * Sized to exercise the paths that could plausibly introduce non-determinism: enough rows for
 * several assignment tiles per parent, two configured levels, and both leader buckets.
 */
TEST(CagraMergeFastener, ScaffoldConstructionIsBitReproducible)
{
  using namespace detail::merge_scaffold;
  raft::resources res;
  auto stream                 = raft::resource::get_cuda_stream(res);
  constexpr int64_t part_rows = 4000;  // > ASSIGNMENT_TILE_ROWS, so parents span multiple tiles
  constexpr int64_t rows      = part_rows * 2;
  constexpr int64_t dim       = 32;

  auto host = make_dataset<float>(res, rows, dim, 4242ULL);
  auto data = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  raft::copy(data.data_handle(), host.data_handle(), host.size(), stream);
  raft::resource::sync_stream(res);

  std::vector<int64_t> offsets{0, part_rows, rows};
  build_params params;  // default two-level controls

  auto scaffold_bytes = [&] {
    auto scaffold = build<float>(res, raft::make_const_mdspan(data.view()), dim, offsets, params);
    std::vector<uint32_t> host_scaffold(
      static_cast<size_t>(scaffold.extent(0) * scaffold.extent(1)));
    raft::copy(host_scaffold.data(), scaffold.data_handle(), host_scaffold.size(), stream);
    raft::resource::sync_stream(res);
    return host_scaffold;
  };

  auto first  = scaffold_bytes();
  auto second = scaffold_bytes();

  ASSERT_EQ(first.size(), second.size());
  ASSERT_GT(first.size(), 0u);
  size_t differing        = 0;
  size_t first_difference = first.size();
  for (size_t i = 0; i < first.size(); ++i) {
    if (first[i] != second[i]) {
      ++differing;
      first_difference = std::min(first_difference, i);
    }
  }
  EXPECT_EQ(differing, 0u) << differing << " of " << first.size()
                           << " scaffold entries differ between identical runs, first at index "
                           << first_difference;
}

/**
 * A partition's memberships are ascending in consolidated row id: the root is the identity, tiles
 * emit in parent input order, and every regroup is a stable sort. `make_leaves` relies on this to
 * know that consecutive slicing groups a leaf by origin, so pin the invariant down explicitly --
 * a future non-stable sort should trip a test rather than silently weaken the guarantee.
 */
TEST(CagraMergeFastener, MembershipsRemainAscendingWithinEachPartition)
{
  using namespace detail::merge_scaffold;
  raft::resources res;
  auto stream            = raft::resource::get_cuda_stream(res);
  constexpr int64_t rows = 96;
  constexpr int64_t dim  = 4;
  auto host_dataset      = make_dataset<float>(res, rows, dim);
  auto dataset           = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  raft::copy(dataset.data_handle(), host_dataset.data_handle(), dataset.size(), stream);

  split_context context(res, rows, dim);
  manyway_l2_norms_kernel<<<static_cast<int>((rows + 3) / 4), 128, 0, stream>>>(
    dataset.data_handle(), rows, dim, dim, context.norms.data_handle());
  RAFT_CUDA_TRY(cudaGetLastError());

  auto partitions            = make_root_partition(res, rows);
  uint32_t occurrence_stride = 1;
  for (uint32_t level = 0; level < 2; ++level) {
    partitions = split_manyway(res,
                               raft::make_const_mdspan(dataset.view()),
                               partitions,
                               split_params{.fanout            = 2,
                                            .leader_fraction   = 0.05,
                                            .max_leaders       = 4,
                                            .leaf_size         = 16,
                                            .level             = level,
                                            .occurrence_stride = occurrence_stride},
                               context);
    occurrence_stride *= 2;
  }

  std::vector<partition_membership> records(partitions.memberships.size());
  raft::copy(records.data(), partitions.memberships.data_handle(), records.size(), stream);
  raft::resource::sync_stream(res);

  for (int64_t range_index = 0; range_index < partitions.ranges.extent(0); ++range_index) {
    auto const& range = partitions.ranges(range_index);
    for (int64_t slot = range.start + 1; slot < range.end; ++slot) {
      EXPECT_LT(records[slot - 1].id, records[slot].id)
        << "partition [" << range.start << ", " << range.end << ") is not ascending at " << slot;
    }
  }
}

/**
 * Regression for the single-origin leaf case: because memberships are ascending in row id and
 * origins are contiguous row-id blocks, every partition is origin-sorted. Slicing an oversized one
 * into consecutive chunks can therefore hand the leaf kernel a leaf drawn from a single input, and
 * that kernel skips every same-origin pair -- such a leaf emits no cross-input edges at all.
 *
 * Two identical inputs whose row counts are multiples of `leaf_size` align every consecutive chunk
 * to one origin. `levels=1` with `root_fanout=1` and a leader fraction below one row clamps to a
 * single leader, so all rows land in one oversized partition and the fallback is guaranteed to run.
 */
TEST(CagraMergeFastener, IdenticalInputsAlignedToLeafSizeGetCrossOriginScaffoldEdges)
{
  using namespace detail::merge_scaffold;
  raft::resources res;
  auto stream                    = raft::resource::get_cuda_stream(res);
  constexpr int64_t part_rows    = 128;  // two whole leaves per input
  constexpr int64_t rows         = part_rows * 2;
  constexpr int64_t dim          = 8;
  constexpr uint32_t leaf_degree = 4;

  // Both halves identical, so each row's exact duplicate sits in the other input at distance zero.
  auto host = raft::make_host_matrix<float, int64_t>(res, rows, dim);
  for (int64_t row = 0; row < part_rows; ++row) {
    for (int64_t column = 0; column < dim; ++column) {
      auto value                    = static_cast<float>(row * 17 + column);
      host(row, column)             = value;
      host(part_rows + row, column) = value;
    }
  }
  auto dataset = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  raft::copy(dataset.data_handle(), host.data_handle(), dataset.size(), stream);

  build_params params;
  params.levels          = 1;
  params.root_fanout     = 1;
  params.lower_fanout    = 1;
  params.leader_fraction = 1.0 / static_cast<double>(rows * 2);
  params.max_leaders     = 16;
  params.leaf_size       = 64;
  params.leaf_degree     = leaf_degree;

  std::vector<int64_t> offsets{0, part_rows, rows};
  auto scaffold = build(res, raft::make_const_mdspan(dataset.view()), dim, offsets, params);
  ASSERT_EQ(scaffold.extent(0), rows);
  ASSERT_EQ(scaffold.extent(1), static_cast<int64_t>(leaf_degree));

  auto scaffold_host = raft::make_host_matrix<uint32_t, int64_t>(res, rows, scaffold.extent(1));
  raft::copy(scaffold_host.data_handle(), scaffold.data_handle(), scaffold.size(), stream);
  raft::resource::sync_stream(res);

  // Slots a leaf never fills keep their self-ID prefill, so a row with no cross-origin entry is
  // exactly a row that a single-origin leaf failed to serve.
  for (int64_t row = 0; row < rows; ++row) {
    bool crosses = false;
    for (int64_t column = 0; column < scaffold_host.extent(1); ++column) {
      auto neighbor = static_cast<int64_t>(scaffold_host(row, column));
      crosses       = crosses || ((neighbor < part_rows) != (row < part_rows));
    }
    EXPECT_TRUE(crosses) << "row " << row << " has no cross-input scaffold neighbor";
  }
}

/**
 * End-to-end form of the same case: with no cross-input edge the merge returns the two input ring
 * graphs side by side, i.e. two disconnected components rather than one merged index.
 */
TEST(CagraMergeFastener, IdenticalInputsAlignedToLeafSizeStayConnected)
{
  raft::resources res;
  constexpr int64_t part_rows = 128;  // two whole leaves per input
  constexpr int64_t rows      = part_rows * 2;
  constexpr int64_t dim       = 8;
  constexpr int64_t degree    = 4;

  auto data           = make_dataset<float>(res, part_rows, dim, 1234ULL);
  auto graph0         = make_ring_graph(res, part_rows, degree);
  auto graph1         = make_ring_graph(res, part_rows, degree);
  auto metric         = cuvs::distance::DistanceType::L2Expanded;
  auto index0_storage = make_padded<float>(res, raft::make_const_mdspan(data.view()));
  cagra::index<float, uint32_t> index0(
    res, metric, index0_storage.view, raft::make_const_mdspan(graph0.view()));
  auto index1_storage = make_padded<float>(res, raft::make_const_mdspan(data.view()));
  cagra::index<float, uint32_t> index1(
    res, metric, index1_storage.view, raft::make_const_mdspan(graph1.view()));

  index_params params;
  params.metric                    = metric;
  params.graph_degree              = degree;
  params.intermediate_graph_degree = degree;
  params.attach_dataset_on_build   = true;
  params.guarantee_connectivity    = false;

  merge_params fastener;
  fastener.algo            = merge_algo::FASTENER;
  fastener.levels          = 1;
  fastener.root_fanout     = 1;
  fastener.lower_fanout    = 1;
  fastener.leader_fraction = 1.0 / static_cast<double>(rows * 2);
  fastener.max_leaders     = 16;
  fastener.leaf_size       = 64;
  fastener.leaf_degree     = 4;

  std::vector<cagra::device_padded_index<float, uint32_t>*> indices{&index0, &index1};
  auto merged_storage = make_merged_storage<float>(res, rows, dim);
  auto merged         = merge(res, params, indices, merged_storage.view, fastener);
  expect_valid_graph(merged, rows, degree);
  EXPECT_EQ(count_connected_components(merged), 1)
    << "the inputs were merged without any edge connecting them";
}

/**
 * Verifies the leaf-GEMM eligibility boundary, which is the same for every scalar type because
 * every type gathers into float leaf vectors, and confirms that a dimension that cannot fit one
 * float leaf in the GEMM workspace is rejected before either input is mutated.
 */
TEST(CagraMergeFastener, LeafGemmLimitsRejectOversizedWorkspaceDimensions)
{
  using namespace detail::merge_scaffold;
  constexpr size_t workspace_bytes = size_t{64} * 1024 * 1024;
  EXPECT_TRUE(leaf_gemm_supported(4096, 256, workspace_bytes));
  EXPECT_FALSE(leaf_gemm_supported(0, 256, workspace_bytes));
  EXPECT_FALSE(leaf_gemm_supported(-1, 256, workspace_bytes));

  // The widest leaf whose float vectors and float Gram matrix together fit the configured
  // workspace.
  constexpr uint32_t leaf_size = 256;
  constexpr int64_t widest_dim = static_cast<int64_t>(
    (workspace_bytes - size_t{leaf_size} * leaf_size * sizeof(float)) / sizeof(float) / leaf_size);
  EXPECT_TRUE(leaf_gemm_supported(widest_dim, leaf_size, workspace_bytes));
  EXPECT_FALSE(leaf_gemm_supported(widest_dim + 1, leaf_size, workspace_bytes));

  // Comfortably past the workspace, used below to check pre-mutation rejection.
  constexpr int64_t oversized_dim =
    static_cast<int64_t>(workspace_bytes / sizeof(float) / leaf_size) + 1;
  EXPECT_FALSE(leaf_gemm_supported(oversized_dim, leaf_size, workspace_bytes));

  raft::resources res;
  raft::resource::set_workspace_to_global_resource(res, workspace_bytes);
  constexpr int64_t rows = 2;
  constexpr int64_t dim  = oversized_dim;
  auto dataset0          = make_dataset<int8_t>(res, rows, dim, 1234ULL);
  auto dataset1          = make_dataset<int8_t>(res, rows, dim, 5678ULL);
  auto graph0            = make_ring_graph(res, rows, 1);
  auto graph1            = make_ring_graph(res, rows, 1);
  auto metric            = cuvs::distance::DistanceType::L2Expanded;
  auto index0_storage    = make_padded<int8_t>(res, raft::make_const_mdspan(dataset0.view()));
  cagra::index<int8_t, uint32_t> index0(
    res, metric, index0_storage.view, raft::make_const_mdspan(graph0.view()));
  auto index1_storage = make_padded<int8_t>(res, raft::make_const_mdspan(dataset1.view()));
  cagra::index<int8_t, uint32_t> index1(
    res, metric, index1_storage.view, raft::make_const_mdspan(graph1.view()));

  index_params params;
  params.metric                    = metric;
  params.graph_degree              = 1;
  params.intermediate_graph_degree = 1;
  merge_params fastener;
  fastener.algo      = merge_algo::FASTENER;
  fastener.leaf_size = leaf_size;
  std::vector<index<int8_t, uint32_t>*> indices{&index0, &index1};

  auto result = detail::preflight_fastener(
    res, params, fastener, indices, cuvs::neighbors::filtering::none_sample_filter{});
  EXPECT_FALSE(result.eligible);
  EXPECT_EQ(result.reason, "dataset dimension exceeds the leaf GEMM workspace limit");
  EXPECT_EQ(index0.dataset().n_rows(), rows);
  EXPECT_EQ(index1.dataset().n_rows(), rows);
}

/**
 * The leaf GEMM is not the only indivisible workspace shape. Assignment must also hold one parent's
 * padded leader matrix next to at least one point row, and the leader matrix can be far wider than
 * a leaf: leaders are capped at `max_leaders` while leaves are capped at MAX_LEAF_SIZE. So a
 * dimension can fit a whole leaf and still fail on a single assignment row. Preflight must cover
 * that shape too -- otherwise the configuration passes preflight and dies inside assign_bucket, by
 * which point AUTO has already committed to Fastener and cannot fall back to rebuild.
 */
TEST(CagraMergeFastener, AssignmentGemmLimitsRejectDimensionsThatFitALeaf)
{
  using namespace detail::merge_scaffold;
  constexpr size_t workspace_bytes = size_t{1} * 1024 * 1024;
  constexpr uint32_t leaf_size     = 256;  // the default, spelled out for the arithmetic below
  constexpr uint32_t max_leaders   = 1024;
  constexpr int64_t rows_per_input = 512;
  constexpr int64_t merged_rows    = rows_per_input * 2;

  // Sampling every row as a leader is what drives the leader matrix to its cap; the default
  // fraction of 0.02 would pick only 21 leaders out of these rows.
  split_params const worst_case{
    .fanout = 3, .leader_fraction = 1.0, .max_leaders = max_leaders, .leaf_size = leaf_size};

  // A leaf of this dimension fits comfortably (1024 * dim + 256 KiB Gram), while a single
  // assignment row does not: the padded leader matrix alone is 1024 * dim * 4 = 2 MiB.
  constexpr int64_t dim = 512;
  EXPECT_TRUE(leaf_gemm_supported(dim, leaf_size, workspace_bytes));
  EXPECT_FALSE(assignment_gemm_supported(dim, merged_rows, worst_case, workspace_bytes));

  // Degenerate shapes are rejected rather than wrapping around in the byte model.
  EXPECT_FALSE(assignment_gemm_supported(0, merged_rows, worst_case, workspace_bytes));
  EXPECT_FALSE(assignment_gemm_supported(-1, merged_rows, worst_case, workspace_bytes));
  EXPECT_FALSE(assignment_gemm_supported(dim, 0, worst_case, workspace_bytes));

  // The leader count follows the sample fraction and the row count, not just the cap, so the same
  // dimension is fine once no parent can reach a wide leader matrix. Bounding on max_leaders alone
  // would reject both of these.
  EXPECT_TRUE(assignment_gemm_supported(dim, 64, worst_case, workspace_bytes));
  auto sparse_leaders            = worst_case;
  sparse_leaders.leader_fraction = 0.02;
  EXPECT_TRUE(assignment_gemm_supported(dim, merged_rows, sparse_leaders, workspace_bytes));

  raft::resources res;
  raft::resource::set_workspace_to_global_resource(res, workspace_bytes);
  auto metric         = cuvs::distance::DistanceType::L2Expanded;
  auto dataset0       = make_dataset<int8_t>(res, rows_per_input, dim, 1234ULL);
  auto dataset1       = make_dataset<int8_t>(res, rows_per_input, dim, 5678ULL);
  auto graph0         = make_ring_graph(res, rows_per_input, 4);
  auto graph1         = make_ring_graph(res, rows_per_input, 4);
  auto index0_storage = make_padded<int8_t>(res, raft::make_const_mdspan(dataset0.view()));
  cagra::index<int8_t, uint32_t> index0(
    res, metric, index0_storage.view, raft::make_const_mdspan(graph0.view()));
  auto index1_storage = make_padded<int8_t>(res, raft::make_const_mdspan(dataset1.view()));
  cagra::index<int8_t, uint32_t> index1(
    res, metric, index1_storage.view, raft::make_const_mdspan(graph1.view()));

  index_params params;
  params.metric                    = metric;
  params.graph_degree              = 4;
  params.intermediate_graph_degree = 8;
  params.attach_dataset_on_build   = true;
  std::vector<index<int8_t, uint32_t>*> indices{&index0, &index1};

  merge_params fastener;
  fastener.algo            = merge_algo::FASTENER;
  fastener.leader_fraction = worst_case.leader_fraction;
  fastener.max_leaders     = worst_case.max_leaders;
  fastener.leaf_size       = worst_case.leaf_size;
  auto result              = detail::preflight_fastener(
    res, params, fastener, indices, cuvs::neighbors::filtering::none_sample_filter{});
  EXPECT_FALSE(result.eligible);
  EXPECT_EQ(result.reason, "dataset dimension exceeds the assignment GEMM workspace limit");

  // Explicit Fastener surfaces the rejection; AUTO takes the rebuild path and still produces an
  // index. Reaching rebuild at all is the regression this test guards.
  auto fastener_storage = make_merged_storage<int8_t>(res, merged_rows, dim);
  EXPECT_ANY_THROW(merge(res, params, indices, fastener_storage.view, fastener));

  auto automatic      = fastener;
  automatic.algo      = merge_algo::AUTO;
  auto merged_storage = make_merged_storage<int8_t>(res, merged_rows, dim);
  auto merged         = merge(res, params, indices, merged_storage.view, automatic);
  EXPECT_EQ(merged.size(), merged_rows);

  EXPECT_EQ(index0.dataset().n_rows(), rows_per_input);
  EXPECT_EQ(index1.dataset().n_rows(), rows_per_input);
}

/**
 * The appended candidate graph is sorted by launch_sort_knn_graph, whose kernel capacity is
 * kMaxSortDegree. Preflight must reject a combined candidate width above that limit, because the
 * sorter would otherwise fail only after consolidation and scaffold construction had already run --
 * and under AUTO, past the point where the rebuild fallback could still be taken.
 */
TEST(CagraMergeFastener, PreflightRejectsCandidateDegreeAboveSortLimit)
{
  raft::resources res;
  constexpr int64_t rows = 600;
  constexpr int64_t dim  = 8;
  auto metric            = cuvs::distance::DistanceType::L2Expanded;

  // Defaults give spill = root_fanout * lower_fanout = 6 and leaf_degree 4, i.e.
  // scaffold_degree 24.
  constexpr uint64_t scaffold_degree = 24;
  constexpr auto sort_limit          = detail::graph::kMaxSortDegree;

  auto dataset0 = make_dataset<float>(res, rows, dim, 1234ULL);
  auto dataset1 = make_dataset<float>(res, rows, dim, 5678ULL);

  auto check = [&](int64_t input_degree) {
    auto graph0         = make_ring_graph(res, rows, input_degree);
    auto graph1         = make_ring_graph(res, rows, input_degree);
    auto index0_storage = make_padded<float>(res, raft::make_const_mdspan(dataset0.view()));
    cagra::index<float, uint32_t> index0(
      res, metric, index0_storage.view, raft::make_const_mdspan(graph0.view()));
    auto index1_storage = make_padded<float>(res, raft::make_const_mdspan(dataset1.view()));
    cagra::index<float, uint32_t> index1(
      res, metric, index1_storage.view, raft::make_const_mdspan(graph1.view()));

    index_params params;
    params.metric                    = metric;
    params.graph_degree              = 64;
    params.intermediate_graph_degree = 64;
    merge_params fastener;
    fastener.algo = merge_algo::FASTENER;
    std::vector<cagra::device_padded_index<float, uint32_t>*> indices{&index0, &index1};
    return detail::preflight_fastener(
      res, params, fastener, indices, cuvs::neighbors::filtering::none_sample_filter{});
  };

  // Exactly at the limit stays eligible; one degree past it is rejected for that specific reason.
  auto at_limit = check(static_cast<int64_t>(sort_limit - scaffold_degree));
  EXPECT_TRUE(at_limit.eligible) << at_limit.reason;

  auto past_limit = check(static_cast<int64_t>(sort_limit - scaffold_degree) + 1);
  EXPECT_FALSE(past_limit.eligible);
  EXPECT_EQ(past_limit.reason,
            "the widest input graph degree plus the scaffold degree must not exceed " +
              std::to_string(sort_limit));
}

/**
 * Exercises an explicit Fastener merge for float, half, int8, and uint8 data, including graph
 * validity, concatenated dataset order, output ownership, and preservation of both input indices.
 */
TEST(CagraMergeFastener, SupportsAllScalarTypes)
{
  auto metric = cuvs::distance::DistanceType::L2Expanded;
  run_explicit_fastener<float>(metric);
  run_explicit_fastener<half>(metric);
  run_explicit_fastener<int8_t>(metric);
  run_explicit_fastener<uint8_t>(metric);
}

/** The assignment and leaf buffers must shrink to the caller's bounded workspace rather than
 * allocating against a fixed compile-time budget. */
TEST(CagraMergeFastener, BatchesWithinConfiguredWorkspace)
{
  raft::resources res;
  constexpr size_t workspace_bytes = size_t{64} * 1024;
  raft::resource::set_workspace_to_global_resource(res, workspace_bytes);

  constexpr int64_t rows   = 512;
  constexpr int64_t dim    = 16;
  constexpr int64_t degree = 8;
  auto dataset0            = make_dataset<float>(res, rows, dim, 1234ULL);
  auto dataset1            = make_dataset<float>(res, rows, dim, 5678ULL);
  auto graph0              = make_ring_graph(res, rows, degree);
  auto graph1              = make_ring_graph(res, rows, degree);
  auto metric              = cuvs::distance::DistanceType::L2Expanded;
  auto index0_storage      = make_padded<float>(res, raft::make_const_mdspan(dataset0.view()));
  cagra::index<float, uint32_t> index0(
    res, metric, index0_storage.view, raft::make_const_mdspan(graph0.view()));
  auto index1_storage = make_padded<float>(res, raft::make_const_mdspan(dataset1.view()));
  cagra::index<float, uint32_t> index1(
    res, metric, index1_storage.view, raft::make_const_mdspan(graph1.view()));

  index_params params;
  params.metric                    = metric;
  params.graph_degree              = degree;
  params.intermediate_graph_degree = degree;
  params.guarantee_connectivity    = false;
  merge_params fastener;
  fastener.algo      = merge_algo::FASTENER;
  fastener.leaf_size = 32;
  std::vector<cagra::device_padded_index<float, uint32_t>*> indices{&index0, &index1};

  auto merged_storage = make_merged_storage<float>(res, rows * 2, dim);
  auto merged         = merge(res, params, indices, merged_storage.view, fastener);
  expect_valid_graph(merged, rows * 2, degree);
  EXPECT_EQ(raft::resource::get_workspace_used_bytes(res), 0);
}

/**
 * Verifies that merging one owning and one non-owning input preserves both inputs and their
 * ownership modes while producing an owning output with the expected dataset order.
 */
TEST(CagraMergeFastener, MixedDatasetOwnershipPreservesInputs)
{
  raft::resources res;
  constexpr int64_t rows   = 48;
  constexpr int64_t dim    = 8;
  constexpr int64_t degree = 4;
  auto dataset0            = make_dataset<float>(res, rows, dim, 1234ULL);
  auto dataset1            = make_dataset<float>(res, rows, dim, 5678ULL);
  auto expected            = concat_host_datasets<float>(
    res, raft::make_const_mdspan(dataset0.view()), raft::make_const_mdspan(dataset1.view()));
  auto device_dataset0 = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  auto device_dataset1 = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  raft::copy(device_dataset0.data_handle(),
             dataset0.data_handle(),
             dataset0.size(),
             raft::resource::get_cuda_stream(res));
  raft::copy(device_dataset1.data_handle(),
             dataset1.data_handle(),
             dataset1.size(),
             raft::resource::get_cuda_stream(res));
  auto graph0         = make_ring_graph(res, rows, degree);
  auto graph1         = make_ring_graph(res, rows, degree);
  auto metric         = cuvs::distance::DistanceType::L2Expanded;
  auto index0_storage = make_padded<float>(res, raft::make_const_mdspan(dataset0.view()));
  cagra::index<float, uint32_t> index0(
    res, metric, index0_storage.view, raft::make_const_mdspan(graph0.view()));
  auto index1_storage = make_padded<float>(res, raft::make_const_mdspan(dataset1.view()));
  cagra::index<float, uint32_t> index1(
    res, metric, index1_storage.view, raft::make_const_mdspan(graph1.view()));
  // The old mixed-ownership assertions are gone: under the dataset-view API an index always holds
  // a non-owning view and ownership lives in the type, so there is no runtime ownership to probe.
  // What still matters is that merging leaves both inputs intact.
  ASSERT_EQ(index0.dataset().n_rows(), rows);
  ASSERT_EQ(index1.dataset().n_rows(), rows);

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
  std::vector<cagra::device_padded_index<float, uint32_t>*> indices{&index0, &index1};

  auto merged_storage = make_merged_storage<float>(res, rows * 2, dim);
  auto merged         = merge(res, params, indices, merged_storage.view, fastener);
  expect_valid_graph(merged, rows * 2, degree);
  expect_dataset_order(res, merged, raft::make_const_mdspan(expected.view()));
  EXPECT_EQ(merged.dataset().n_rows(), rows * 2);
  EXPECT_EQ(index0.dataset().n_rows(), rows);
  EXPECT_EQ(index1.dataset().n_rows(), rows);
  EXPECT_EQ(index0.dataset().dim(), static_cast<uint32_t>(dim));
  EXPECT_EQ(index1.dataset().dim(), static_cast<uint32_t>(dim));
}

/**
 * Reuses AnnCagraIndexMergeTest on the primary float L2 shapes from generate_inputs(): build two
 * CAGRA graphs over halves of the same InitDataset vectors, merge with default Fastener knobs, and
 * require at least 95% recall against brute force.
 */
namespace {
std::vector<AnnCagraInputs> generate_fastener_merge_recall_inputs()
{
  // Same core shape as the leading AnnCagra float L2 cases: 1000x16 database, 100 queries, k=16.
  auto inputs = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {16},
    {16},  // k
    {32},  // graph_degree
    {graph_build_algo::IVF_PQ, graph_build_algo::NN_DESCENT},
    {search_algo::AUTO},
    {10},
    {0},
    {256},
    {1},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {true},
    {false},
    {0.95},
    {std::optional<float>{std::nullopt}},
    // AnnCagraInputs::compression was removed upstream along with index_params::compression.
    {std::optional<bool>{std::nullopt}},
    {cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL});
  for (auto& input : inputs) {
    // Default Fastener controls; only the algorithm is forced so AUTO cannot fall back to rebuild.
    merge_params fastener;
    fastener.algo               = merge_algo::FASTENER;
    input.physical_merge_params = fastener;
  }
  return inputs;
}
}  // namespace

/**
 * Search recall for a Fastener merge of `n_inputs` real CAGRA indices, against exact brute-force
 * ground truth.
 *
 * The shared AnnCagraIndexMergeTest fixture always splits its database into exactly two parts and
 * is instantiated for float only, so it cannot cover a non-float dtype or a fan-in above two. This
 * builds the parts directly instead. Slicing the database into consecutive ranges and merging in
 * that order reproduces the original row order in the merged index, so ground-truth indices
 * computed over the whole database apply to the merged result without remapping.
 */
template <typename DataT>
void run_fastener_merge_recall(size_t n_inputs, double min_recall)
{
  raft::resources res;
  auto stream               = raft::resource::get_cuda_stream(res);
  constexpr int64_t rows    = 2000;
  constexpr int64_t dim     = 16;
  constexpr int64_t n_query = 100;
  constexpr uint32_t k      = 16;
  constexpr uint32_t degree = 32;
  auto metric               = cuvs::distance::DistanceType::L2Expanded;

  auto host_data    = make_dataset<DataT>(res, rows, dim, 1234ULL, metric);
  auto host_queries = make_dataset<DataT>(res, n_query, dim, 9876ULL, metric);
  auto data         = raft::make_device_matrix<DataT, int64_t>(res, rows, dim);
  auto queries      = raft::make_device_matrix<DataT, int64_t>(res, n_query, dim);
  raft::copy(data.data_handle(), host_data.data_handle(), host_data.size(), stream);
  raft::copy(queries.data_handle(), host_queries.data_handle(), host_queries.size(), stream);
  raft::resource::sync_stream(res);

  index_params build_params;
  build_params.metric                    = metric;
  build_params.graph_degree              = degree;
  build_params.intermediate_graph_degree = degree * 2;
  build_params.attach_dataset_on_build   = true;

  // Consecutive, near-equal slices so the merged row order matches the original database.
  std::vector<cuvs::neighbors::test::padded_device_matrix_for_cagra<DataT>> part_storage;
  part_storage.reserve(n_inputs);
  std::vector<cagra::device_padded_index<DataT, uint32_t>> parts;
  parts.reserve(n_inputs);
  int64_t rows_per_part = raft::ceildiv<int64_t>(rows, static_cast<int64_t>(n_inputs));
  for (size_t part = 0; part < n_inputs; ++part) {
    int64_t start = static_cast<int64_t>(part) * rows_per_part;
    int64_t count = std::min<int64_t>(rows_per_part, rows - start);
    ASSERT_GT(count, 0);
    auto slice = raft::make_device_matrix_view<const DataT, int64_t>(
      data.data_handle() + start * dim, count, dim);
    // build() takes a dataset view now, and the index keeps only a view, so the padded storage
    // must outlive every part.
    part_storage.emplace_back(res, slice);
    parts.emplace_back(cagra::build(res, build_params, part_storage.back().view));
    parts.back().update_device_dataset_same_layout(res, part_storage.back().view);
  }

  std::vector<cagra::device_padded_index<DataT, uint32_t>*> inputs;
  for (auto& part : parts) {
    inputs.push_back(&part);
  }
  merge_params fastener;
  fastener.algo       = merge_algo::FASTENER;
  auto merged_storage = make_merged_storage<DataT>(res, rows, dim);
  auto merged         = merge(res, build_params, inputs, merged_storage.view, fastener);
  ASSERT_EQ(merged.size(), rows);

  auto found_indices   = raft::make_device_matrix<uint32_t, int64_t>(res, n_query, k);
  auto found_distances = raft::make_device_matrix<float, int64_t>(res, n_query, k);
  search_params search;
  search.itopk_size = 256;
  cagra::search(res,
                search,
                merged,
                raft::make_const_mdspan(queries.view()),
                found_indices.view(),
                found_distances.view());

  auto exact_indices   = raft::make_device_matrix<uint32_t, int64_t>(res, n_query, k);
  auto exact_distances = raft::make_device_matrix<float, int64_t>(res, n_query, k);
  cuvs::neighbors::naive_knn<float, DataT, uint32_t>(res,
                                                     exact_distances.data_handle(),
                                                     exact_indices.data_handle(),
                                                     queries.data_handle(),
                                                     data.data_handle(),
                                                     n_query,
                                                     rows,
                                                     dim,
                                                     k,
                                                     metric);

  size_t total = static_cast<size_t>(n_query) * k;
  std::vector<uint32_t> found_idx(total), exact_idx(total);
  std::vector<float> found_dist(total), exact_dist(total);
  raft::copy(found_idx.data(), found_indices.data_handle(), total, stream);
  raft::copy(exact_idx.data(), exact_indices.data_handle(), total, stream);
  raft::copy(found_dist.data(), found_distances.data_handle(), total, stream);
  raft::copy(exact_dist.data(), exact_distances.data_handle(), total, stream);
  raft::resource::sync_stream(res);

  EXPECT_TRUE(cuvs::neighbors::eval_neighbours(
    exact_idx, found_idx, exact_dist, found_dist, n_query, k, 0.001, min_recall));
}

/**
 * Non-float recall coverage. The dtype tests elsewhere in this file only assert that the merged
 * graph is structurally valid; this checks that a uint8 merge is actually searchable.
 */
TEST(CagraMergeFastener, Uint8TwoWayMergeSearchRecall)
{
  run_fastener_merge_recall<uint8_t>(2, 0.85);
}

/** Recall coverage for a fan-in above two, which the shared fixture cannot express. */
TEST(CagraMergeFastener, FloatFourWayMergeSearchRecall)
{
  run_fastener_merge_recall<float>(4, 0.85);
}

/** Both dimensions at once: non-float data merged four ways. */
TEST(CagraMergeFastener, Uint8FourWayMergeSearchRecall)
{
  run_fastener_merge_recall<uint8_t>(4, 0.85);
}

typedef AnnCagraIndexMergeTest<float, float, std::uint32_t> AnnCagraFastenerMergeRecallTest;
TEST_P(AnnCagraFastenerMergeRecallTest, DefaultFloatMergeSearchRecallAgainstBruteForce)
{
  this->testCagra<uint32_t>();
}
INSTANTIATE_TEST_CASE_P(CagraMergeFastener,
                        AnnCagraFastenerMergeRecallTest,
                        ::testing::ValuesIn(generate_fastener_merge_recall_inputs()));

/**
 * Exercises a three-level, non-default uint8 merge with different input graph degrees and verifies
 * that every output row has the requested degree and contains only valid neighbor identifiers.
 */
TEST(CagraMergeFastener, MixedDegreesAndThreeLevelUint8OptionsProduceExactDegree)
{
  raft::resources res;
  constexpr int64_t rows = 64;
  constexpr int64_t dim  = 16;
  auto dataset0          = make_dataset<uint8_t>(res, rows, dim, 1234ULL);
  auto dataset1          = make_dataset<uint8_t>(res, rows, dim, 5678ULL);
  auto graph0            = make_ring_graph(res, rows, 6);
  auto graph1            = make_ring_graph(res, rows, 8);
  auto index0_storage    = make_padded<uint8_t>(res, raft::make_const_mdspan(dataset0.view()));
  cagra::index<uint8_t, uint32_t> index0(res,
                                         cuvs::distance::DistanceType::L2Expanded,
                                         index0_storage.view,
                                         raft::make_const_mdspan(graph0.view()));
  auto index1_storage = make_padded<uint8_t>(res, raft::make_const_mdspan(dataset1.view()));
  cagra::index<uint8_t, uint32_t> index1(res,
                                         cuvs::distance::DistanceType::L2Expanded,
                                         index1_storage.view,
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
  std::vector<cagra::device_padded_index<uint8_t, uint32_t>*> indices{&index0, &index1};

  auto merged_storage = make_merged_storage<uint8_t>(res, rows * 2, dim);
  auto merged         = merge(res, params, indices, merged_storage.view, fastener);
  expect_valid_graph(merged, rows * 2, 8);
}

/**
 * Verifies that input graphs with different degrees are shifted to global identifiers, cyclically
 * padded to a common width, and appended to the already-global scaffold columns in row order.
 */
TEST(CagraMergeFastener, AppendCyclicallyPadsMixedInputDegrees)
{
  raft::resources res;
  auto dataset0       = make_dataset<float>(res, 2, 1, 1234ULL);
  auto dataset1       = make_dataset<float>(res, 2, 1, 5678ULL);
  auto graph0         = make_ring_graph(res, 2, 2);
  auto graph1         = make_ring_graph(res, 2, 4);
  auto index0_storage = make_padded<float>(res, raft::make_const_mdspan(dataset0.view()));
  cagra::index<float, uint32_t> index0(res,
                                       cuvs::distance::DistanceType::L2Expanded,
                                       index0_storage.view,
                                       raft::make_const_mdspan(graph0.view()));
  auto index1_storage = make_padded<float>(res, raft::make_const_mdspan(dataset1.view()));
  cagra::index<float, uint32_t> index1(res,
                                       cuvs::distance::DistanceType::L2Expanded,
                                       index1_storage.view,
                                       raft::make_const_mdspan(graph1.view()));
  std::vector<cagra::device_padded_index<float, uint32_t>*> indices{&index0, &index1};
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

  constexpr int64_t base_degree = 4;
  auto appended                 = raft::make_device_matrix<uint32_t, int64_t>(res, 4, 5);
  raft::copy_matrix(appended.data_handle() + base_degree,
                    static_cast<std::size_t>(appended.extent(1)),
                    scaffold.data_handle(),
                    static_cast<std::size_t>(scaffold.extent(1)),
                    static_cast<std::size_t>(scaffold.extent(1)),
                    static_cast<std::size_t>(scaffold.extent(0)),
                    raft::resource::get_cuda_stream(res));
  detail::merge_scaffold::append_to_input_graphs<float, uint32_t>(
    res, indices, offsets, appended.view(), base_degree);
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

/**
 * Verifies that Fastener can merge higher-degree input graphs into a valid graph whose requested
 * output degree is smaller than either input degree.
 */
TEST(CagraMergeFastener, MergeSupportsOutputDegreeBelowInputDegree)
{
  raft::resources res;
  constexpr int64_t rows         = 48;
  constexpr int64_t dim          = 8;
  constexpr int64_t input_degree = 8;
  auto dataset0                  = make_dataset<float>(res, rows, dim, 1234ULL);
  auto dataset1                  = make_dataset<float>(res, rows, dim, 5678ULL);
  auto graph0                    = make_ring_graph(res, rows, input_degree);
  auto graph1                    = make_ring_graph(res, rows, input_degree);
  auto index0_storage = make_padded<float>(res, raft::make_const_mdspan(dataset0.view()));
  cagra::index<float, uint32_t> index0(res,
                                       cuvs::distance::DistanceType::L2Expanded,
                                       index0_storage.view,
                                       raft::make_const_mdspan(graph0.view()));
  auto index1_storage = make_padded<float>(res, raft::make_const_mdspan(dataset1.view()));
  cagra::index<float, uint32_t> index1(res,
                                       cuvs::distance::DistanceType::L2Expanded,
                                       index1_storage.view,
                                       raft::make_const_mdspan(graph1.view()));

  index_params params;
  params.metric                  = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree            = 4;
  params.attach_dataset_on_build = true;
  merge_params fastener;
  fastener.algo        = merge_algo::FASTENER;
  fastener.leaf_size   = 64;
  fastener.leaf_degree = 4;
  std::vector<cagra::device_padded_index<float, uint32_t>*> indices{&index0, &index1};

  auto merged_storage = make_merged_storage<float>(res, rows * 2, dim);
  auto merged         = merge(res, params, indices, merged_storage.view, fastener);
  expect_valid_graph(merged, rows * 2, params.graph_degree);
}

/** Indices hold only views, so the padded storage travels with them. */
struct float_index_pair {
  padded_storage<float> storage0;
  padded_storage<float> storage1;
  std::vector<cagra::device_padded_index<float, uint32_t>> indices;
};

auto make_float_indices(raft::resources const& res,
                        cuvs::distance::DistanceType metric,
                        raft::host_matrix<float, int64_t>& dataset0,
                        raft::host_matrix<float, int64_t>& dataset1,
                        raft::host_matrix<uint32_t, int64_t>& graph0,
                        raft::host_matrix<uint32_t, int64_t>& graph1) -> float_index_pair
{
  auto storage0 = make_padded<float>(res, raft::make_const_mdspan(dataset0.view()));
  auto storage1 = make_padded<float>(res, raft::make_const_mdspan(dataset1.view()));
  std::vector<cagra::device_padded_index<float, uint32_t>> output;
  output.emplace_back(res, metric, storage0.view, raft::make_const_mdspan(graph0.view()));
  output.emplace_back(res, metric, storage1.view, raft::make_const_mdspan(graph1.view()));
  return float_index_pair{std::move(storage0), std::move(storage1), std::move(output)};
}

/**
 * Exercises invalid levels, fanouts, leader settings, leaf settings, and spill width, and verifies
 * that preflight rejects each configuration without changing either input dataset.
 */
TEST(CagraMergeFastener, InvalidManywayOptionsFailPreflightWithoutMutation)
{
  raft::resources res;
  constexpr int64_t rows = 32;
  constexpr int64_t dim  = 8;
  auto dataset0          = make_dataset<float>(res, rows, dim, 1234ULL);
  auto dataset1          = make_dataset<float>(res, rows, dim, 5678ULL);
  auto graph0            = make_ring_graph(res, rows, 4);
  auto graph1            = make_ring_graph(res, rows, 4);
  index_params params;
  params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree              = 4;
  params.intermediate_graph_degree = 8;
  params.attach_dataset_on_build   = true;
  auto owned = make_float_indices(res, params.metric, dataset0, dataset1, graph0, graph1);
  std::vector<cagra::device_padded_index<float, uint32_t>*> indices{&owned.indices[0],
                                                                    &owned.indices[1]};

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
      res, params, candidate, indices, cuvs::neighbors::filtering::none_sample_filter{});
    EXPECT_FALSE(result.eligible) << result.reason;
  }
  EXPECT_EQ(owned.indices[0].dataset().n_rows(), rows);
  EXPECT_EQ(owned.indices[1].dataset().n_rows(), rows);
}

/**
 * Verifies dispatch semantics: explicit Fastener rejects unsupported settings and metrics, AUTO
 * falls back successfully, REBUILD remains available, and every path preserves its input indices.
 */
TEST(CagraMergeFastener, DispatchRejectsOrFallsBackBeforeMutation)
{
  raft::resources res;
  constexpr int64_t rows = 32;
  constexpr int64_t dim  = 8;
  auto dataset0          = make_dataset<float>(res, rows, dim, 1234ULL);
  auto dataset1          = make_dataset<float>(res, rows, dim, 5678ULL);
  auto graph0            = make_ring_graph(res, rows, 4);
  auto graph1            = make_ring_graph(res, rows, 4);
  index_params params;
  params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree              = 4;
  params.intermediate_graph_degree = 8;
  params.attach_dataset_on_build   = true;

  {
    auto owned = make_float_indices(res, params.metric, dataset0, dataset1, graph0, graph1);
    auto throwaway_storage = make_merged_storage<float>(res, rows * 2, dim);
    std::vector<cagra::device_padded_index<float, uint32_t>*> indices{&owned.indices[0],
                                                                      &owned.indices[1]};
    merge_params unsupported;
    unsupported.algo      = merge_algo::FASTENER;
    unsupported.leaf_size = 512;
    EXPECT_ANY_THROW(merge(res, params, indices, throwaway_storage.view, unsupported));
    EXPECT_EQ(owned.indices[0].dataset().n_rows(), rows);
    EXPECT_EQ(owned.indices[1].dataset().n_rows(), rows);
  }
  {
    auto inner_product_params   = params;
    inner_product_params.metric = cuvs::distance::DistanceType::InnerProduct;
    auto owned =
      make_float_indices(res, inner_product_params.metric, dataset0, dataset1, graph0, graph1);
    auto throwaway_storage = make_merged_storage<float>(res, rows * 2, dim);
    std::vector<cagra::device_padded_index<float, uint32_t>*> indices{&owned.indices[0],
                                                                      &owned.indices[1]};
    merge_params fastener;
    fastener.algo      = merge_algo::FASTENER;
    fastener.leaf_size = 64;
    EXPECT_ANY_THROW(merge(res, inner_product_params, indices, throwaway_storage.view, fastener));
    EXPECT_EQ(owned.indices[0].dataset().n_rows(), rows);
    EXPECT_EQ(owned.indices[1].dataset().n_rows(), rows);
  }
  {
    auto owned = make_float_indices(res, params.metric, dataset0, dataset1, graph0, graph1);
    auto throwaway_storage = make_merged_storage<float>(res, rows * 2, dim);
    std::vector<cagra::device_padded_index<float, uint32_t>*> indices{&owned.indices[0],
                                                                      &owned.indices[1]};
    merge_params automatic;
    automatic.algo      = merge_algo::AUTO;
    automatic.leaf_size = 512;
    auto merged         = merge(res, params, indices, throwaway_storage.view, automatic);
    EXPECT_EQ(merged.size(), rows * 2);
    EXPECT_EQ(owned.indices[0].dataset().n_rows(), rows);
    EXPECT_EQ(owned.indices[1].dataset().n_rows(), rows);
  }
  {
    auto owned = make_float_indices(res, params.metric, dataset0, dataset1, graph0, graph1);
    auto throwaway_storage = make_merged_storage<float>(res, rows * 2, dim);
    std::vector<cagra::device_padded_index<float, uint32_t>*> indices{&owned.indices[0],
                                                                      &owned.indices[1]};
    merge_params rebuild{merge_algo::REBUILD};
    auto merged_storage = make_merged_storage<float>(res, rows * 2, dim);
    auto merged         = merge(res, params, indices, merged_storage.view, rebuild);
    EXPECT_EQ(merged.size(), rows * 2);
    EXPECT_EQ(owned.indices[0].dataset().n_rows(), rows);
    EXPECT_EQ(owned.indices[1].dataset().n_rows(), rows);
  }
}

/**
 * Verifies that the shared graph sorter applies metric-specific ordering by producing different
 * first neighbors for L2 distance and inner-product similarity on the same candidate graph.
 */
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

  detail::graph::launch_sort_knn_graph(res,
                                       cuvs::distance::DistanceType::L2Expanded,
                                       device_dataset.data_handle(),
                                       static_cast<uint32_t>(device_dataset.extent(0)),
                                       static_cast<uint32_t>(device_dataset.extent(1)),
                                       l2.data_handle(),
                                       static_cast<uint32_t>(l2.extent(1)));
  detail::graph::launch_sort_knn_graph(res,
                                       cuvs::distance::DistanceType::InnerProduct,
                                       device_dataset.data_handle(),
                                       static_cast<uint32_t>(device_dataset.extent(0)),
                                       static_cast<uint32_t>(device_dataset.extent(1)),
                                       ip.data_handle(),
                                       static_cast<uint32_t>(ip.extent(1)));
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

/**
 * Verifies the graph-combination pipeline end to end: local identifiers are offset, scaffold edges
 * are appended, rows are metric-sorted, duplicates are removed, and short rows are padded.
 */
TEST(CagraMergeFastener, AppendSortAndDedupHandleOffsetsAndPadding)
{
  raft::resources res;
  auto dataset0       = raft::make_host_matrix<float, int64_t>(2, 1);
  auto dataset1       = raft::make_host_matrix<float, int64_t>(2, 1);
  dataset0(0, 0)      = 0.0f;
  dataset0(1, 0)      = 2.0f;
  dataset1(0, 0)      = 10.0f;
  dataset1(1, 0)      = 12.0f;
  auto graph0         = make_ring_graph(res, 2, 1);
  auto graph1         = make_ring_graph(res, 2, 1);
  auto index0_storage = make_padded<float>(res, raft::make_const_mdspan(dataset0.view()));
  cagra::index<float, uint32_t> index0(res,
                                       cuvs::distance::DistanceType::L2Expanded,
                                       index0_storage.view,
                                       raft::make_const_mdspan(graph0.view()));
  auto index1_storage = make_padded<float>(res, raft::make_const_mdspan(dataset1.view()));
  cagra::index<float, uint32_t> index1(res,
                                       cuvs::distance::DistanceType::L2Expanded,
                                       index1_storage.view,
                                       raft::make_const_mdspan(graph1.view()));
  std::vector<cagra::device_padded_index<float, uint32_t>*> indices{&index0, &index1};
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
  constexpr int64_t base_degree = 1;
  auto appended                 = raft::make_device_matrix<uint32_t, int64_t>(res, 4, 3);
  raft::copy_matrix(appended.data_handle() + base_degree,
                    static_cast<std::size_t>(appended.extent(1)),
                    scaffold.data_handle(),
                    static_cast<std::size_t>(scaffold.extent(1)),
                    static_cast<std::size_t>(scaffold.extent(1)),
                    static_cast<std::size_t>(scaffold.extent(0)),
                    raft::resource::get_cuda_stream(res));
  detail::merge_scaffold::append_to_input_graphs<float, uint32_t>(
    res, indices, offsets, appended.view(), base_degree);

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
  detail::graph::launch_sort_knn_graph(res,
                                       cuvs::distance::DistanceType::L2Expanded,
                                       combined.data_handle(),
                                       static_cast<uint32_t>(combined.extent(0)),
                                       static_cast<uint32_t>(combined.extent(1)),
                                       appended.data_handle(),
                                       static_cast<uint32_t>(appended.extent(1)));
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
