/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/vamana.h>

#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

extern "C" cuvsVamanaGraphInitAlgo default_vamana_graph_init(void);

extern "C" cuvsError_t run_vamana(cuvsResources_t res,
                                  int64_t n_rows,
                                  int64_t n_dim,
                                  float* index_data,
                                  uint32_t graph_degree,
                                  uint32_t visited_size,
                                  cuvsVamanaGraphInitAlgo graph_init,
                                  const char* graph_file);

namespace {

constexpr int64_t kRows          = 2000;
constexpr int64_t kDim           = 32;
constexpr uint32_t kGraphDegree  = 32;
constexpr uint32_t kVisitedSize  = 64;
constexpr unsigned kInsertSeed   = 42;
constexpr int kInvalidGraphInit  = 7;

struct vamana_graph_file {
  uint32_t max_degree;
  uint32_t medoid;
  std::vector<std::vector<uint32_t>> lists;
};

// Reads the graph written by cuvsVamanaSerialize without the dataset (DiskANN graph layout).
auto read_graph_file(const std::string& path, int64_t n_rows) -> vamana_graph_file
{
  std::ifstream in(path, std::ios::binary);
  EXPECT_TRUE(in.good()) << path;
  uint64_t file_size = 0;
  uint64_t frozen    = 0;
  vamana_graph_file graph{};
  in.read(reinterpret_cast<char*>(&file_size), sizeof(file_size));
  in.read(reinterpret_cast<char*>(&graph.max_degree), sizeof(graph.max_degree));
  in.read(reinterpret_cast<char*>(&graph.medoid), sizeof(graph.medoid));
  in.read(reinterpret_cast<char*>(&frozen), sizeof(frozen));
  EXPECT_EQ(file_size, std::filesystem::file_size(path));
  graph.lists.resize(n_rows);
  for (auto& list : graph.lists) {
    uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    list.resize(count);
    in.read(reinterpret_cast<char*>(list.data()), sizeof(uint32_t) * count);
  }
  EXPECT_TRUE(in.good()) << path;
  return graph;
}

void check_graph(const vamana_graph_file& graph, int64_t n_rows)
{
  EXPECT_LE(graph.max_degree, kGraphDegree);
  EXPECT_LT(static_cast<int64_t>(graph.medoid), n_rows);
  size_t edges = 0;
  for (int64_t i = 0; i < n_rows; i++) {
    const auto& list = graph.lists[i];
    // Lists can repeat a neighbor after the reverse-edge merge, so only ids are checked here.
    EXPECT_LE(list.size(), kGraphDegree);
    for (auto id : list) {
      EXPECT_LT(static_cast<int64_t>(id), n_rows);
      EXPECT_NE(static_cast<int64_t>(id), i);
    }
    edges += list.size();
  }
  // As in the C++ tests: the graph is not always full, but less than 75% is very unlikely.
  EXPECT_GT(static_cast<double>(edges), 0.75 * n_rows * kGraphDegree);
}

// Squared distance of every row to the mean of the rows.
auto squared_distances_to_mean(const std::vector<float>& data) -> std::vector<double>
{
  std::vector<double> mean(kDim, 0.0);
  for (int64_t i = 0; i < kRows; i++) {
    for (int64_t j = 0; j < kDim; j++) {
      mean[j] += data[i * kDim + j];
    }
  }
  std::vector<double> distances(kRows, 0.0);
  for (int64_t i = 0; i < kRows; i++) {
    for (int64_t j = 0; j < kDim; j++) {
      const double diff = data[i * kDim + j] - mean[j] / kRows;
      distances[i] += diff * diff;
    }
  }
  return distances;
}

class VamanaC : public ::testing::Test {
 protected:
  void SetUp() override
  {
    stream_ = raft::resource::get_cuda_stream(handle_).get();
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    h_data_.resize(kRows * kDim);
    for (auto& v : h_data_) {
      v = dist(gen);
    }
    d_data_ = std::make_unique<rmm::device_uvector<float>>(h_data_.size(), stream_);
    raft::update_device(d_data_->data(), h_data_.data(), h_data_.size(), stream_);
    raft::resource::sync_stream(handle_);
    ASSERT_EQ(cuvsResourcesCreate(&res_), CUVS_SUCCESS);
    ASSERT_EQ(cuvsStreamSet(res_, stream_), CUVS_SUCCESS);
    graph_file_ = (std::filesystem::temp_directory_path() /
                   ("cuvs_vamana_c_test_" + std::to_string(::getpid()) + ".graph"))
                    .string();
  }

  void TearDown() override
  {
    std::filesystem::remove(graph_file_);
    cuvsResourcesDestroy(res_);
  }

  auto build(cuvsVamanaGraphInitAlgo graph_init) -> cuvsError_t
  {
    return run_vamana(res_,
                      kRows,
                      kDim,
                      d_data_->data(),
                      kGraphDegree,
                      kVisitedSize,
                      graph_init,
                      graph_file_.c_str());
  }

  raft::handle_t handle_;
  cudaStream_t stream_{};
  cuvsResources_t res_{};
  std::vector<float> h_data_;
  std::unique_ptr<rmm::device_uvector<float>> d_data_;
  std::string graph_file_;
};

}  // namespace

TEST(VamanaCParams, DefaultGraphInitIsEmpty)
{
  EXPECT_EQ(default_vamana_graph_init(), CUVS_VAMANA_GRAPH_INIT_EMPTY);
}

TEST_F(VamanaC, BuildFromEmptyGraph)
{
  // The default build draws n_rows values from rand() for the insertion order, then the medoid.
  srand(kInsertSeed);
  for (int64_t i = 0; i < kRows; i++) {
    rand();
  }
  const auto expected_medoid = static_cast<uint32_t>(rand() % kRows);

  srand(kInsertSeed);
  ASSERT_EQ(build(CUVS_VAMANA_GRAPH_INIT_EMPTY), CUVS_SUCCESS) << cuvsGetLastErrorText();
  auto graph = read_graph_file(graph_file_, kRows);
  check_graph(graph, kRows);
  EXPECT_EQ(graph.medoid, expected_medoid);
}

TEST_F(VamanaC, BuildFromNNDescentGraph)
{
  ASSERT_EQ(build(CUVS_VAMANA_GRAPH_INIT_NN_DESCENT), CUVS_SUCCESS) << cuvsGetLastErrorText();
  auto graph = read_graph_file(graph_file_, kRows);
  check_graph(graph, kRows);

  // The medoid is the vector nearest the dataset mean (the GPU sums in float, so allow for
  // rounding between near ties).
  auto distances = squared_distances_to_mean(h_data_);
  ASSERT_LT(static_cast<int64_t>(graph.medoid), kRows);
  double nearest = *std::min_element(distances.begin(), distances.end());
  EXPECT_LE(distances[graph.medoid], nearest * (1.0 + 1e-4) + 1e-6);
}

TEST_F(VamanaC, InvalidGraphInitIsRejected)
{
  EXPECT_EQ(build(static_cast<cuvsVamanaGraphInitAlgo>(kInvalidGraphInit)), CUVS_ERROR);
}
