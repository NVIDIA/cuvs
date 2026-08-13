/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <gtest/gtest.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include "neighbors/ann_utils.cuh"
#include "neighbors/naive_knn.cuh"
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/mg_cagra.h>
#include <cuvs/neighbors/mg_ivf_flat.h>
#include <cuvs/neighbors/mg_ivf_pq.h>

extern "C" {

typedef enum { MG_ALGO_IVF_FLAT, MG_ALGO_IVF_PQ, MG_ALGO_CAGRA } mg_algo_t;

typedef enum { MG_MODE_REPLICATED, MG_MODE_SHARDED, MG_MODE_LOCAL_THEN_DISTRIBUTED } mg_mode_t;

typedef struct {
  int64_t num_queries;
  int64_t num_db_vecs;
  int64_t dim;
  int64_t k;
  mg_mode_t mode;
  mg_algo_t algo;
  int64_t nprobe;
  int64_t nlist;
  cuvsDistanceType metric;
} mg_test_params;

int run_mg_ivf_flat_test(mg_test_params params,
                         float* index_data,
                         float* query_data,
                         float* distances_data,
                         int64_t* neighbors_data,
                         float* ref_distances_data,
                         int64_t* ref_neighbors_data);

int run_mg_ivf_pq_test(mg_test_params params,
                       float* index_data,
                       float* query_data,
                       float* distances_data,
                       int64_t* neighbors_data,
                       float* ref_distances_data,
                       int64_t* ref_neighbors_data);

int run_mg_cagra_test(mg_test_params params,
                      float* index_data,
                      float* query_data,
                      float* distances_data,
                      int64_t* neighbors_data,
                      float* ref_distances_data,
                      int64_t* ref_neighbors_data);

int generate_reference_results(mg_test_params params,
                               float* index_data,
                               float* query_data,
                               float* ref_distances_data,
                               int64_t* ref_neighbors_data);
}

template <typename T>
void generate_random_data(T* devPtr, size_t size)
{
  raft::handle_t handle;
  raft::random::RngState r(1234ULL);
  if constexpr (std::is_same_v<T, float>) {
    raft::random::uniform(handle, r, devPtr, size, T(0.1), T(2.0));
  } else {
    raft::random::uniformInt(handle, r, devPtr, size, T(1), T(20));
  }
}

template <typename T, typename IdxT>
bool eval_recall(T* query_data,
                 T* index_data,
                 IdxT* neighbors,
                 T* distances,
                 IdxT* ref_neighbors,
                 T* ref_distances,
                 size_t n_queries,
                 size_t n_rows,
                 size_t n_dim,
                 size_t n_neighbors,
                 cuvsDistanceType metric,
                 double min_recall = 0.9)
{
  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  size_t size = n_queries * n_neighbors;
  std::vector<IdxT> neighbors_h(size);
  std::vector<T> distances_h(size);
  std::vector<IdxT> neighbors_ref_h(size);
  std::vector<T> distances_ref_h(size);

  raft::copy(neighbors_h.data(), neighbors, size, stream);
  raft::copy(distances_h.data(), distances, size, stream);
  raft::copy(neighbors_ref_h.data(), ref_neighbors, size, stream);
  raft::copy(distances_ref_h.data(), ref_distances, size, stream);

  raft::resource::sync_stream(handle);

  // Evaluate recall
  return cuvs::neighbors::eval_neighbours(neighbors_ref_h,
                                          neighbors_h,
                                          distances_ref_h,
                                          distances_h,
                                          n_queries,
                                          n_neighbors,
                                          0.001,
                                          min_recall);
}

class MgCTest : public ::testing::TestWithParam<mg_test_params> {
 public:
  void SetUp() override
  {
    params = GetParam();

    raft::handle_t handle;
    auto stream = raft::resource::get_cuda_stream(handle);

    // Allocate device memory
    index_data.resize(params.num_db_vecs * params.dim, stream);
    query_data.resize(params.num_queries * params.dim, stream);
    neighbors_data.resize(params.num_queries * params.k, stream);
    distances_data.resize(params.num_queries * params.k, stream);
    ref_neighbors_data.resize(params.num_queries * params.k, stream);
    ref_distances_data.resize(params.num_queries * params.k, stream);

    // Generate random test data
    generate_random_data(index_data.data(), params.num_db_vecs * params.dim);
    generate_random_data(query_data.data(), params.num_queries * params.dim);

    raft::resource::sync_stream(handle);

    // Allocate host memory for multi-GPU tests
    index_data_host.resize(params.num_db_vecs * params.dim);
    query_data_host.resize(params.num_queries * params.dim);
    neighbors_data_host.resize(params.num_queries * params.k);
    distances_data_host.resize(params.num_queries * params.k);

    // Copy data from device to host for multi-GPU functions
    raft::copy(index_data_host.data(), index_data.data(), index_data.size(), stream);
    raft::copy(query_data_host.data(), query_data.data(), query_data.size(), stream);

    raft::resource::sync_stream(handle);
  }

  void TearDown() override {}

 protected:
  mg_test_params params;
  rmm::device_uvector<float> index_data{0, rmm::cuda_stream_default};
  rmm::device_uvector<float> query_data{0, rmm::cuda_stream_default};
  rmm::device_uvector<int64_t> neighbors_data{0, rmm::cuda_stream_default};
  rmm::device_uvector<float> distances_data{0, rmm::cuda_stream_default};
  rmm::device_uvector<int64_t> ref_neighbors_data{0, rmm::cuda_stream_default};
  rmm::device_uvector<float> ref_distances_data{0, rmm::cuda_stream_default};

  // Host memory for multi-GPU tests
  std::vector<float> index_data_host;
  std::vector<float> query_data_host;
  std::vector<int64_t> neighbors_data_host;
  std::vector<float> distances_data_host;
};

TEST_P(MgCTest, MgIvfFlatTest)
{
  auto params = GetParam();
  if (params.algo != MG_ALGO_IVF_FLAT) return;

  // Generate reference results using brute force
  int ref_result = generate_reference_results(params,
                                              index_data.data(),
                                              query_data.data(),
                                              ref_distances_data.data(),
                                              ref_neighbors_data.data());
  ASSERT_EQ(ref_result, 0) << "Failed to generate reference results";

  // Run MG IVF-Flat test (use host data for multi-GPU)
  int test_result = run_mg_ivf_flat_test(params,
                                         index_data_host.data(),
                                         query_data_host.data(),
                                         distances_data_host.data(),
                                         neighbors_data_host.data(),
                                         ref_distances_data.data(),
                                         ref_neighbors_data.data());
  ASSERT_EQ(test_result, 0) << "MG IVF-Flat test failed";

  // Copy results back from host to device for evaluation
  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  raft::copy(neighbors_data.data(), neighbors_data_host.data(), neighbors_data_host.size(), stream);
  raft::copy(distances_data.data(), distances_data_host.data(), distances_data_host.size(), stream);
  raft::resource::sync_stream(handle);

  // Evaluate recall compared to reference
  double min_recall = static_cast<double>(params.nprobe) / static_cast<double>(params.nlist);
  bool recall_ok    = eval_recall(query_data.data(),
                               index_data.data(),
                               neighbors_data.data(),
                               distances_data.data(),
                               ref_neighbors_data.data(),
                               ref_distances_data.data(),
                               params.num_queries,
                               params.num_db_vecs,
                               params.dim,
                               params.k,
                               params.metric,
                               min_recall);
  ASSERT_TRUE(recall_ok) << "Recall evaluation failed for MG IVF-Flat";
}

TEST_P(MgCTest, MgIvfPqTest)
{
  auto params = GetParam();
  if (params.algo != MG_ALGO_IVF_PQ) return;

  // Generate reference results using brute force
  int ref_result = generate_reference_results(params,
                                              index_data.data(),
                                              query_data.data(),
                                              ref_distances_data.data(),
                                              ref_neighbors_data.data());
  ASSERT_EQ(ref_result, 0) << "Failed to generate reference results";

  // Run MG IVF-PQ test (use host data for multi-GPU)
  int test_result = run_mg_ivf_pq_test(params,
                                       index_data_host.data(),
                                       query_data_host.data(),
                                       distances_data_host.data(),
                                       neighbors_data_host.data(),
                                       ref_distances_data.data(),
                                       ref_neighbors_data.data());
  ASSERT_EQ(test_result, 0) << "MG IVF-PQ test failed";

  // Copy results back from host to device for evaluation
  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  raft::copy(neighbors_data.data(), neighbors_data_host.data(), neighbors_data_host.size(), stream);
  raft::copy(distances_data.data(), distances_data_host.data(), distances_data_host.size(), stream);
  raft::resource::sync_stream(handle);

  // Evaluate recall compared to reference
  double min_recall = static_cast<double>(params.nprobe) / static_cast<double>(params.nlist);
  bool recall_ok    = eval_recall(query_data.data(),
                               index_data.data(),
                               neighbors_data.data(),
                               distances_data.data(),
                               ref_neighbors_data.data(),
                               ref_distances_data.data(),
                               params.num_queries,
                               params.num_db_vecs,
                               params.dim,
                               params.k,
                               params.metric,
                               min_recall);
  ASSERT_TRUE(recall_ok) << "Recall evaluation failed for MG IVF-PQ";
}

TEST_P(MgCTest, MgCagraTest)
{
  auto params = GetParam();
  if (params.algo != MG_ALGO_CAGRA) return;

  // Generate reference results using brute force
  int ref_result = generate_reference_results(params,
                                              index_data.data(),
                                              query_data.data(),
                                              ref_distances_data.data(),
                                              ref_neighbors_data.data());
  ASSERT_EQ(ref_result, 0) << "Failed to generate reference results";

  // Run MG CAGRA test (use host data for multi-GPU)
  int test_result = run_mg_cagra_test(params,
                                      index_data_host.data(),
                                      query_data_host.data(),
                                      distances_data_host.data(),
                                      neighbors_data_host.data(),
                                      ref_distances_data.data(),
                                      ref_neighbors_data.data());
  ASSERT_EQ(test_result, 0) << "MG CAGRA test failed";

  // Copy results back from host to device for evaluation
  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  raft::copy(neighbors_data.data(), neighbors_data_host.data(), neighbors_data_host.size(), stream);
  raft::copy(distances_data.data(), distances_data_host.data(), distances_data_host.size(), stream);
  raft::resource::sync_stream(handle);

  // Evaluate recall compared to reference
  bool recall_ok = eval_recall(query_data.data(),
                               index_data.data(),
                               neighbors_data.data(),
                               distances_data.data(),
                               ref_neighbors_data.data(),
                               ref_distances_data.data(),
                               params.num_queries,
                               params.num_db_vecs,
                               params.dim,
                               params.k,
                               params.metric,
                               0.9);
  ASSERT_TRUE(recall_ok) << "Recall evaluation failed for MG CAGRA";
}

// Test parameters that mirror the C++ test cases
INSTANTIATE_TEST_SUITE_P(MgCTests, MgCTest, ::testing::Values(
  // IVF-Flat tests
  mg_test_params{1000, 5000, 8, 16, MG_MODE_REPLICATED, MG_ALGO_IVF_FLAT, 40, 256, L2Expanded},
  mg_test_params{1000, 5000, 8, 16, MG_MODE_SHARDED, MG_ALGO_IVF_FLAT, 40, 256, L2Expanded},

  // IVF-PQ tests
  mg_test_params{1000, 5000, 8, 16, MG_MODE_REPLICATED, MG_ALGO_IVF_PQ, 40, 256, L2Expanded},
  mg_test_params{1000, 5000, 8, 16, MG_MODE_SHARDED, MG_ALGO_IVF_PQ, 40, 256, L2Expanded},

  // CAGRA tests
  mg_test_params{1000, 5000, 8, 16, MG_MODE_REPLICATED, MG_ALGO_CAGRA, 40, 256, L2Expanded},
  mg_test_params{1000, 5000, 8, 16, MG_MODE_SHARDED, MG_ALGO_CAGRA, 40, 256, L2Expanded}));
