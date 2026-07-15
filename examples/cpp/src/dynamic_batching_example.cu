/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/dynamic_batching.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <rmm/cuda_stream_pool.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <future>

// A helper to split the dataset into chunks
template <typename DeviceMatrixOrView>
auto slice_matrix(const DeviceMatrixOrView& source,
                  typename DeviceMatrixOrView::index_type offset_rows,
                  typename DeviceMatrixOrView::index_type count_rows)
{
  auto n_cols = source.extent(1);
  return raft::make_device_matrix_view<typename DeviceMatrixOrView::element_type,
                                       typename DeviceMatrixOrView::index_type>(
    const_cast<typename DeviceMatrixOrView::element_type*>(source.data_handle()) +
      offset_rows * n_cols,
    count_rows,
    n_cols);
}

// A helper to measure the execution time of a function
template <typename F, typename... Args>
void time_it(std::string label, F f, Args&&... xs)
{
  auto start = std::chrono::system_clock::now();
  f(std::forward<Args>(xs)...);
  auto end  = std::chrono::system_clock::now();
  auto t    = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  auto t_ms = double(t.count()) / 1000.0;
  std::cout << "[" << label << "] execution time: " << t_ms << " ms" << std::endl;
}

/**
 * Wrap waiting on a stream work into an async C++ future object.
 * This is similar to recording and waiting on CUDA events, but in C++11 API.
 */
struct cuda_work_completion_promise {
  cuda_work_completion_promise(const raft::resources& res)
  {
    auto* promise = new std::promise<void>;
    RAFT_CUDA_TRY(cudaLaunchHostFunc(
      raft::resource::get_cuda_stream(res), completion_callback, reinterpret_cast<void*>(promise)));
    value_ = promise->get_future();
  }

  /**
   * Waiting on the produced `future` object has the same effect as
   * cudaEventSynchronize if an event was recorded at the time of creation of
   * this promise object.
   */
  auto get_future() -> std::future<void>&& { return std::move(value_); }

 private:
  std::future<void> value_;

  static void completion_callback(void* ptr)
  {
    auto* promise = reinterpret_cast<std::promise<void>*>(ptr);
    promise->set_value();
    delete promise;
  }
};

void dynamic_batching_example(raft::resources const& res,
                              raft::device_matrix_view<const float, int64_t> dataset,
                              raft::device_matrix_view<const float, int64_t> queries)
{
  using namespace cuvs::neighbors;

  // Number of neighbors to search
  int64_t topk = 100;

  // Streaming scenario: maximum number of requests in-flight
  constexpr int64_t kMaxJobs = 1000;
  // Streaming scenario: number of concurrent CUDA streams
  constexpr int64_t kNumWorkerStreams = 5;

  // Split the queries into two subsets to run every experiment twice and thus
  // surface any initialization overheads.
  int64_t n_queries_a = queries.extent(0) / 2;
  int64_t n_queries_b = queries.extent(0) - n_queries_a;

  auto queries_a = slice_matrix(queries, 0, n_queries_a);
  auto queries_b = slice_matrix(queries, n_queries_a, n_queries_b);

  // create output arrays
  auto neighbors = raft::make_device_matrix<uint32_t>(res, queries.extent(0), topk);
  auto distances = raft::make_device_matrix<float>(res, queries.extent(0), topk);
  // slice them same as queries
  auto neighbors_a = slice_matrix(neighbors, 0, n_queries_a);
  auto distances_a = slice_matrix(distances, 0, n_queries_a);
  auto neighbors_b = slice_matrix(neighbors, n_queries_a, n_queries_b);
  auto distances_b = slice_matrix(distances, n_queries_a, n_queries_b);

  // use default index parameters
  cagra::index_params orig_index_params;

  std::cout << "Building CAGRA index (search graph)" << std::endl;
  auto orig_index = cagra::build(res, orig_index_params, dataset);

  std::cout << "CAGRA index has " << orig_index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << orig_index.graph_degree() << ", graph size ["
            << orig_index.graph().extent(0) << ", " << orig_index.graph().extent(1) << "]"
            << std::endl;

  // use default search parameters
  cagra::search_params orig_search_params;
  // get a decent recall by increasing the internal topk list
  orig_search_params.itopk_size = 512;
  orig_search_params.algo       = cagra::search_algo::SINGLE_CTA;

  // Set up dynamic batching parameters
  dynamic_batching::index_params dynb_index_params{
    /* default-initializing the parent `neighbors::index_params`
       (not used anyway) */
    {},
    /* Set the K in advance (the batcher needs to allocate buffers) */
    topk,
    /* Configure the number and the size of IO buffers */
    64,
    kNumWorkerStreams};

  // "build" the index (it's a low-cost index wrapping),
  //  that is we need to pass the original index and its search params here
  dynamic_batching::index<float, uint32_t> dynb_index(
    res, dynb_index_params, orig_index, orig_search_params);

  // You can implement job priorities by varying the deadlines of individual
  // requests
  dynamic_batching::search_params dynb_search_params;
  dynb_search_params.dispatch_timeout_ms = 0.1;

  // Define the big-batch setting as a baseline for measuring the throughput.
  auto search_batch_orig = [&res, &orig_index, &orig_search_params](
                             raft::device_matrix_view<const float, int64_t> queries,
                             raft::device_matrix_view<uint32_t, int64_t> neighbors,
                             raft::device_matrix_view<float, int64_t> distances) {
    cagra::search(res, orig_search_params, orig_index, queries, neighbors, distances);
    raft::resource::sync_stream(res);
  };

  // Launch the baseline search: check the big-batch performance
  time_it("standard/batch A", search_batch_orig, queries_a, neighbors_a, distances_a);
  time_it("standard/batch B", search_batch_orig, queries_b, neighbors_b, distances_b);

  // Streaming scenario: prepare concurrent resources
  rmm::cuda_stream_pool worker_streams{kNumWorkerStreams};
  std::vector<raft::resources> resource_pool(0);
  for (int64_t i = 0; i < kNumWorkerStreams; i++) {
    resource_pool.push_back(res);
    raft::resource::set_cuda_stream(resource_pool[i], worker_streams.get_stream(i));
  }

  // Streaming scenario:
  // send queries one-by-one, with a maximum kMaxJobs in-flight
  auto search_async_orig = [&resource_pool, &orig_index, &orig_search_params](
                             raft::device_matrix_view<const float, int64_t> queries,
                             raft::device_matrix_view<uint32_t, int64_t> neighbors,
                             raft::device_matrix_view<float, int64_t> distances) {
    auto work_size = queries.extent(0);
    std::array<std::future<void>, kMaxJobs> futures;
    for (int64_t i = 0; i < work_size + kMaxJobs; i++) {
      // wait for previous job in the same slot to finish
      if (i >= kMaxJobs) { futures[i % kMaxJobs].wait(); }
      // submit a new job
      if (i < work_size) {
        auto& res = resource_pool[i % kNumWorkerStreams];
        cagra::search(res,
                      orig_search_params,
                      orig_index,
                      slice_matrix(queries, i, 1),
                      slice_matrix(neighbors, i, 1),
                      slice_matrix(distances, i, 1));
        futures[i % kMaxJobs] = cuda_work_completion_promise(res).get_future();
      }
    }
  };

  // Streaming scenario with dynamic batching:
  // send queries one-by-one, with a maximum kMaxJobs in-flight,
  // yet allow grouping the sequential requests (subject to deadlines)
  auto search_async_dynb = [&resource_pool, &dynb_index, &dynb_search_params](
                             raft::device_matrix_view<const float, int64_t> queries,
                             raft::device_matrix_view<uint32_t, int64_t> neighbors,
                             raft::device_matrix_view<float, int64_t> distances) {
    auto work_size = queries.extent(0);
    std::array<std::future<void>, kMaxJobs> futures;
    for (int64_t i = 0; i < work_size + kMaxJobs; i++) {
      // wait for previous job in the same slot to finish
      if (i >= kMaxJobs) { futures[i % kMaxJobs].wait(); }
      // submit a new job
      if (i < work_size) {
        auto& res = resource_pool[i % kNumWorkerStreams];
        dynamic_batching::search(res,
                                 dynb_search_params,
                                 dynb_index,
                                 slice_matrix(queries, i, 1),
                                 slice_matrix(neighbors, i, 1),
                                 slice_matrix(distances, i, 1));
        futures[i % kMaxJobs] = cuda_work_completion_promise(res).get_future();
      }
    }
  };

  // Try to handle the same amount of work in the async setting using the
  // standard implementation.
  time_it("standard/async A", search_async_orig, queries_a, neighbors_a, distances_a);
  time_it("standard/async B", search_async_orig, queries_b, neighbors_b, distances_b);

  // Do the same using dynamic batching
  time_it("dynamic_batching/async A", search_async_dynb, queries_a, neighbors_a, distances_a);
  time_it("dynamic_batching/async B", search_async_dynb, queries_b, neighbors_b, distances_b);
}

int main()
{
  raft::device_resources res;

  // Set the raft resource to use a pool for internal memory allocations
  // (workspace) and limit the available workspace size.
  raft::resource::set_workspace_to_pool_resource(res, 12ull * 1024 * 1024 * 1024ull);

  // Create input arrays.
  int64_t n_samples = 1000000;
  int64_t n_dim     = 128;
  int64_t n_queries = 10000;
  auto dataset      = raft::make_device_matrix<float, int64_t>(res, n_samples, n_dim);
  auto queries      = raft::make_device_matrix<float, int64_t>(res, n_queries, n_dim);
  generate_dataset(res, dataset.view(), queries.view());

  // run the interesting part of the program
  dynamic_batching_example(
    res, raft::make_const_mdspan(dataset.view()), raft::make_const_mdspan(queries.view()));
}
