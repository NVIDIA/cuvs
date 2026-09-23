/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ann_cagra.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/util/file_io.hpp>
#include <raft/core/host_mdarray.hpp>

#include <algorithm>
#include <cstdlib>

namespace cuvs::neighbors::cagra {
namespace {

struct AnnCagraAceInputs {
  bool use_disk;
  bool add_global_reverse_edges;
  cuvs::distance::DistanceType metric;
  int npartitions;
};

// Own the workspace even if a build throws or a fatal assertion returns early.
struct ace_test_workspace {
  std::string path = (std::filesystem::temp_directory_path() / "cuvs_cagra_ace_XXXXXX").string();
  ace_test_workspace()
  {
    RAFT_EXPECTS(::mkdtemp(path.data()) != nullptr, "Cannot create ACE test workspace");
  }
  ~ace_test_workspace()
  {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

template <typename T>
void read_ace_array(cuvs::util::file_descriptor const& fd,
                    std::vector<T>& values,
                    std::vector<size_t> const& shape)
{
  auto stream = fd.make_istream();
  stream.seekg(0);
  auto header = raft::numpy_serializer::read_header(stream);
  ASSERT_EQ(header.shape, shape);
  ASSERT_EQ(header.dtype, raft::numpy_serializer::get_numpy_dtype<T>());
  stream.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T));
  ASSERT_TRUE(stream.good());
}

template <typename T>
class AnnCagraAceTest : public ::testing::TestWithParam<AnnCagraAceInputs> {
 public:
  void testCagraAce()
  {
    auto const [use_disk, global_reverse_edges, metric, npartitions] = this->GetParam();
    // Keep the average core partition size fixed while varying the partition count.
    const uint32_t rows = 5000 * npartitions;
    constexpr int dim = 16, queries = 100, k = 10, degree = 64;
    ace_test_workspace workspace;
    raft::resources res;
    auto stream         = raft::resource::get_cuda_stream(res);
    auto dataset        = raft::make_host_matrix<T, int64_t>(rows, dim);
    auto device_data    = raft::make_device_matrix<T, int64_t>(res, rows, dim);
    auto device_queries = raft::make_device_matrix<T, int64_t>(res, queries, dim);
    raft::random::RngState rng(1234ULL);
    InitDataset(res, device_data.data_handle(), rows, dim, metric, rng);
    InitDataset(res, device_queries.data_handle(), queries, dim, metric, rng);
    raft::copy(dataset.data_handle(), device_data.data_handle(), dataset.size(), stream);
    raft::resource::sync_stream(res);

    index_params params;
    params.metric                    = metric;
    params.graph_degree              = degree;
    params.intermediate_graph_degree = 2 * degree;
    params.attach_dataset_on_build   = false;
    graph_build_params::ace_params ace;
    ace.npartitions              = npartitions;
    ace.ef_construction          = 100;
    ace.use_disk                 = use_disk;
    ace.add_global_reverse_edges = global_reverse_edges;
    ace.build_dir                = workspace.path;
    params.graph_build_params    = ace;
    auto host_view =
      cuvs::neighbors::make_host_standard_dataset_view(raft::make_const_mdspan(dataset.view()));
    auto built = cagra::build(res, params, host_view);
    ASSERT_EQ(built.size(), rows);
    ASSERT_EQ(built.graph_degree(), degree);
    ASSERT_EQ(built.graph_fd().has_value(), use_disk);
    ASSERT_EQ(built.dataset_fd().has_value(), use_disk);
    ASSERT_EQ(built.mapping_fd().has_value(), use_disk);

    auto graph = raft::make_host_matrix<uint32_t, int64_t>(rows, degree);
    if (use_disk) {
      std::vector<uint32_t> disk_graph(rows * degree), mapping(rows);
      std::vector<T> reordered(rows * dim);
      ASSERT_NO_FATAL_FAILURE(
        read_ace_array(*built.graph_fd(), disk_graph, {size_t(rows), degree}));
      ASSERT_NO_FATAL_FAILURE(read_ace_array(*built.mapping_fd(), mapping, {size_t(rows)}));
      ASSERT_NO_FATAL_FAILURE(read_ace_array(*built.dataset_fd(), reordered, {size_t(rows), dim}));
      auto sorted_mapping = mapping;
      std::sort(sorted_mapping.begin(), sorted_mapping.end());
      for (uint32_t row = 0; row < rows; ++row) {
        ASSERT_EQ(sorted_mapping[row], row);
      }
      // Restore original IDs to search the persisted graph against the original dataset.
      for (uint32_t row = 0; row < rows; ++row) {
        for (int col = 0; col < dim; ++col) {
          ASSERT_EQ(reordered[row * dim + col], dataset(mapping[row], col));
        }
        for (int rank = 0; rank < degree; ++rank) {
          auto neighbor = disk_graph[row * degree + rank];
          ASSERT_LT(neighbor, rows);
          graph(mapping[row], rank) = mapping[neighbor];
        }
      }
      // The augmented vectors and reverse-edge scratch files must be removed on success.
      EXPECT_EQ(std::distance(std::filesystem::directory_iterator(workspace.path),
                              std::filesystem::directory_iterator{}),
                3);
    } else {
      raft::copy(graph.data_handle(), built.graph().data_handle(), graph.size(), stream);
      raft::resource::sync_stream(res);
    }
    for (uint32_t row = 0; row < rows; ++row) {
      auto* first = graph.data_handle() + row * degree;
      std::vector<uint32_t> neighbors(first, first + degree);
      std::sort(neighbors.begin(), neighbors.end());
      ASSERT_LT(neighbors.back(), rows);
      EXPECT_EQ(std::find(neighbors.begin(), neighbors.end(), row), neighbors.end());
      EXPECT_EQ(std::adjacent_find(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    cuvs::neighbors::test::padded_device_matrix_for_cagra<T> padded(
      res, raft::make_const_mdspan(device_data.view()));
    device_padded_index<T, uint32_t> searchable(
      res, metric, padded.view, raft::make_const_mdspan(graph.view()));
    auto expected_ids       = raft::make_device_matrix<uint32_t, int64_t>(res, queries, k);
    auto expected_distances = raft::make_device_matrix<float, int64_t>(res, queries, k);
    cuvs::neighbors::naive_knn<float, T, uint32_t>(res,
                                                   expected_distances.data_handle(),
                                                   expected_ids.data_handle(),
                                                   device_queries.data_handle(),
                                                   device_data.data_handle(),
                                                   queries,
                                                   rows,
                                                   dim,
                                                   k,
                                                   metric);
    auto actual_ids       = raft::make_device_matrix<uint32_t, int64_t>(res, queries, k);
    auto actual_distances = raft::make_device_matrix<float, int64_t>(res, queries, k);
    search_params search;
    search.itopk_size = 128;
    cagra::search(res,
                  search,
                  searchable,
                  raft::make_const_mdspan(device_queries.view()),
                  actual_ids.view(),
                  actual_distances.view());
    std::vector<uint32_t> expected(queries * k), actual(queries * k);
    std::vector<float> expected_d(queries * k), actual_d(queries * k);
    raft::copy(expected.data(), expected_ids.data_handle(), expected.size(), stream);
    raft::copy(actual.data(), actual_ids.data_handle(), actual.size(), stream);
    raft::copy(expected_d.data(), expected_distances.data_handle(), expected_d.size(), stream);
    raft::copy(actual_d.data(), actual_distances.data_handle(), actual_d.size(), stream);
    raft::resource::sync_stream(res);
    EXPECT_TRUE(cuvs::neighbors::eval_neighbours(
      expected, actual, expected_d, actual_d, queries, k, 0.003, 0.9));
  }
};

const std::vector<AnnCagraAceInputs> inputs_ace = raft::util::itertools::product<AnnCagraAceInputs>(
  {false, true},  // use_disk
  {false, true},  // add_global_reverse_edges
  {cuvs::distance::DistanceType::L2Expanded, cuvs::distance::DistanceType::InnerProduct},
  {2, 8, 32, 64});  // npartitions

std::string ace_case_name(::testing::TestParamInfo<AnnCagraAceInputs> const& info)
{
  auto const [disk, reverse, metric, npartitions] = info.param;
  return std::string(disk ? "Disk" : "Memory") + (reverse ? "GlobalReverse" : "LocalReverse") +
         (metric == cuvs::distance::DistanceType::L2Expanded ? "L2" : "InnerProduct") +
         "Partitions" + std::to_string(npartitions);
}

}  // namespace
}  // namespace cuvs::neighbors::cagra
