/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../src/neighbors/detail/vamana/vamana_init_graph.cuh"

#include <cuvs/neighbors/vamana.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace cuvs::neighbors::vamana::detail {

namespace {

using device_accessor =
  raft::host_device_accessor<cuda::std::default_accessor<const float>, raft::memory_type::device>;
using dataset_view =
  raft::mdspan<const float, raft::matrix_extent<int64_t>, raft::row_major, device_accessor>;

constexpr uint32_t kInvalid = std::numeric_limits<uint32_t>::max();

// Small integer coordinates, so that squared distances are exact in float and the device order,
// including its tie-break by id, can be checked exactly.
auto integer_dataset(int64_t n_rows, int64_t dim, unsigned seed) -> std::vector<float>
{
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> value(0, 15);
  std::vector<float> data(n_rows * dim);
  for (auto& v : data) {
    v = static_cast<float>(value(gen));
  }
  return data;
}

auto squared_distance(const std::vector<float>& data, int64_t dim, int64_t a, int64_t b) -> int64_t
{
  int64_t d = 0;
  for (int64_t k = 0; k < dim; k++) {
    const auto diff =
      static_cast<int64_t>(data[a * dim + k]) - static_cast<int64_t>(data[b * dim + k]);
    d += diff * diff;
  }
  return d;
}

// What sort_graph_by_distance should leave in one list: valid ids sorted by (distance, id), with
// self links and repeats dropped when drop_repeats is set, then invalid ids.
auto expected_list(const std::vector<float>& data,
                   int64_t dim,
                   int64_t n_rows,
                   int64_t row,
                   const uint32_t* list,
                   int degree,
                   bool drop_repeats) -> std::vector<uint32_t>
{
  std::vector<std::pair<int64_t, uint32_t>> keyed;
  for (int j = 0; j < degree; j++) {
    const uint32_t id = list[j];
    if (static_cast<int64_t>(id) >= n_rows) { continue; }
    if (drop_repeats && static_cast<int64_t>(id) == row) { continue; }
    keyed.emplace_back(squared_distance(data, dim, row, id), id);
  }
  std::sort(keyed.begin(), keyed.end());
  if (drop_repeats) { keyed.erase(std::unique(keyed.begin(), keyed.end()), keyed.end()); }
  std::vector<uint32_t> out(degree, kInvalid);
  for (size_t j = 0; j < keyed.size(); j++) {
    out[j] = keyed[j].second;
  }
  return out;
}

}  // namespace

class VamanaSortGraphByDistance : public ::testing::TestWithParam<int> {};

TEST_P(VamanaSortGraphByDistance, MatchesHostSort)
{
  const int degree         = GetParam();
  constexpr int64_t n_rows = 600;  // more than the largest degree, so lists can hold distinct ids
  constexpr int64_t dim    = 37;   // not a multiple of the warp size
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  const auto data = integer_dataset(n_rows, dim, 42 + degree);
  auto d_data     = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  raft::copy(d_data.data_handle(), data.data(), data.size(), stream);
  dataset_view dataset(d_data.data_handle(), raft::matrix_extent<int64_t>(n_rows, dim));

  // Random lists with a self link, repeats and invalid ids in the middle. Row 0 holds only
  // invalid ids and row 1 a single id repeated.
  std::mt19937 gen(7 + degree);
  std::uniform_int_distribution<uint32_t> any_row(0, n_rows - 1);
  std::vector<uint32_t> graph(n_rows * degree);
  for (int64_t row = 0; row < n_rows; row++) {
    uint32_t* list = graph.data() + row * degree;
    for (int j = 0; j < degree; j++) {
      list[j] = any_row(gen);
    }
    if (row == 0) {
      std::fill(list, list + degree, kInvalid);
      continue;
    }
    if (row == 1) {
      std::fill(list, list + degree, 5u);
      continue;
    }
    // Distinct positions for every degree tested, the smallest being 10.
    list[degree / 3]     = static_cast<uint32_t>(row);
    list[degree / 2]     = list[1];
    list[degree - 1]     = list[0];
    list[degree / 4]     = kInvalid;
    list[3 * degree / 4] = static_cast<uint32_t>(n_rows + 3);
  }

  for (bool drop_repeats : {false, true}) {
    auto d_graph = raft::make_device_matrix<uint32_t, int64_t>(res, n_rows, degree);
    raft::copy(d_graph.data_handle(), graph.data(), graph.size(), stream);
    sort_graph_by_distance<float, uint32_t, device_accessor>(
      res, dataset, d_graph.view(), drop_repeats);
    std::vector<uint32_t> sorted(graph.size());
    raft::copy(sorted.data(), d_graph.data_handle(), sorted.size(), stream);
    raft::resource::sync_stream(res);

    int64_t mismatched_rows = 0;
    for (int64_t row = 0; row < n_rows; row++) {
      const auto expected =
        expected_list(data, dim, n_rows, row, graph.data() + row * degree, degree, drop_repeats);
      const std::vector<uint32_t> actual(sorted.begin() + row * degree,
                                         sorted.begin() + (row + 1) * degree);
      if (expected != actual) { mismatched_rows++; }
    }
    EXPECT_EQ(mismatched_rows, 0) << "degree " << degree << ", drop_repeats " << drop_repeats;
  }
}

INSTANTIATE_TEST_CASE_P(VamanaInitGraph,
                        VamanaSortGraphByDistance,
                        ::testing::Values(10, 32, 64, 128, 256, 299, 512));

class VamanaInitGraphFromKnnGraph : public ::testing::TestWithParam<std::pair<int64_t, int>> {};

// The seeded graph: every list sorted by distance, without self links or repeats, with invalid ids
// only at the tail, and the vector nearest the dataset mean as the medoid.
TEST_P(VamanaInitGraphFromKnnGraph, ListsAreSortedAndDistinct)
{
  const auto [n_rows, degree] = GetParam();
  constexpr int64_t dim       = 24;
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  const auto data = integer_dataset(n_rows, dim, 1234);
  auto d_data     = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  raft::copy(d_data.data_handle(), data.data(), data.size(), stream);
  dataset_view dataset(d_data.data_handle(), raft::matrix_extent<int64_t>(n_rows, dim));

  cuvs::neighbors::vamana::index_params params;
  params.graph_degree = degree;
  auto knn_graph      = build_init_knn_graph<float, device_accessor>(res, params, dataset, degree);
  EXPECT_EQ(knn_graph.extent(1), std::min<int64_t>(2 * degree, n_rows - 1));

  auto d_graph          = raft::make_device_matrix<uint32_t, int64_t>(res, n_rows, degree);
  const uint32_t medoid = init_graph_from_knn_graph<float, uint32_t, device_accessor>(
    res, dataset, knn_graph.view(), d_graph.view());
  std::vector<uint32_t> graph(n_rows * degree);
  raft::copy(graph.data(), d_graph.data_handle(), graph.size(), stream);
  raft::resource::sync_stream(res);

  int64_t edges = 0;
  for (int64_t row = 0; row < n_rows; row++) {
    const uint32_t* list = graph.data() + row * degree;
    int valid            = 0;
    while (valid < degree && list[valid] != kInvalid) {
      valid++;
    }
    edges += valid;
    std::set<uint32_t> seen;
    for (int j = 0; j < degree; j++) {
      if (j >= valid) {
        ASSERT_EQ(list[j], kInvalid) << "row " << row << ": a valid id after an invalid one";
        continue;
      }
      ASSERT_LT(static_cast<int64_t>(list[j]), n_rows) << "row " << row;
      ASSERT_NE(static_cast<int64_t>(list[j]), row) << "row " << row << ": self link";
      ASSERT_TRUE(seen.insert(list[j]).second) << "row " << row << ": repeated id " << list[j];
      if (j > 0) {
        const auto prev = squared_distance(data, dim, row, list[j - 1]);
        const auto curr = squared_distance(data, dim, row, list[j]);
        ASSERT_TRUE(prev < curr || (prev == curr && list[j - 1] < list[j]))
          << "row " << row << ": not sorted at position " << j;
      }
    }
  }
  // CAGRA's optimize step fills every list, and the sort drops only self links and repeats.
  EXPECT_GT(edges, static_cast<int64_t>(0.9 * n_rows * degree));

  std::vector<double> mean(dim, 0.0);
  for (int64_t row = 0; row < n_rows; row++) {
    for (int64_t k = 0; k < dim; k++) {
      mean[k] += data[row * dim + k] / static_cast<double>(n_rows);
    }
  }
  auto distance_to_mean = [&](int64_t row) {
    double d = 0.0;
    for (int64_t k = 0; k < dim; k++) {
      d += (data[row * dim + k] - mean[k]) * (data[row * dim + k] - mean[k]);
    }
    return d;
  };
  double nearest = distance_to_mean(0);
  for (int64_t row = 1; row < n_rows; row++) {
    nearest = std::min(nearest, distance_to_mean(row));
  }
  ASSERT_LT(static_cast<int64_t>(medoid), n_rows);
  EXPECT_LE(distance_to_mean(medoid), nearest * (1.0 + 1e-4) + 1e-6);
}

// The last case has fewer rows than 2 * graph_degree, so the kNN degree is n_rows - 1.
INSTANTIATE_TEST_CASE_P(VamanaInitGraph,
                        VamanaInitGraphFromKnnGraph,
                        ::testing::Values(std::make_pair(int64_t{2000}, 32),
                                          std::make_pair(int64_t{2000}, 64),
                                          std::make_pair(int64_t{3000}, 128),
                                          std::make_pair(int64_t{50}, 32)));

}  // namespace cuvs::neighbors::vamana::detail
