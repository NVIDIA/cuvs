/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../dynamic_batching.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>

namespace cuvs::neighbors::dynamic_batching {

namespace {

template <typename T, typename IdxT>
auto build_cagra_with_dataset(raft::resources const& res,
                              cagra::index_params const& params,
                              raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)
  -> cagra::device_padded_index<T, IdxT>
{
  auto padded = cuvs::core::make_device_padded_dataset_view(res, dataset);
  auto index  = cagra::build(res, params, padded);
  index       = cagra::update_dataset(res, std::move(index), padded);
  return index;
}

}  // namespace

using cagra_F32 = dynamic_batching_test<float,
                                        uint32_t,
                                        cagra::device_padded_index<float, uint32_t>,
                                        build_cagra_with_dataset<float, uint32_t>,
                                        cagra::search>;

using cagra_U8 = dynamic_batching_test<uint8_t,
                                       uint32_t,
                                       cagra::device_padded_index<uint8_t, uint32_t>,
                                       build_cagra_with_dataset<uint8_t, uint32_t>,
                                       cagra::search>;

template <typename fixture>
static void set_default_cagra_params(fixture& that)
{
  that.build_params_upsm.intermediate_graph_degree = 128;
  that.build_params_upsm.graph_degree              = 64;
  that.search_params_upsm.itopk_size =
    std::clamp<int64_t>(raft::bound_by_power_of_two(that.ps.k) * 16, 128, 512);
}

TEST_P(cagra_F32, single_cta)
{
  set_default_cagra_params(*this);
  search_params_upsm.algo = cagra::search_algo::SINGLE_CTA;
  build_all();
  search_all();
  check_neighbors();
}

TEST_P(cagra_F32, multi_cta)
{
  set_default_cagra_params(*this);
  search_params_upsm.algo = cagra::search_algo::MULTI_CTA;
  build_all();
  search_all();
  check_neighbors();
}

TEST_P(cagra_F32, multi_kernel)
{
  set_default_cagra_params(*this);
  search_params_upsm.algo = cagra::search_algo::MULTI_KERNEL;
  build_all();
  search_all();
  check_neighbors();
}

TEST_P(cagra_U8, defaults)
{
  set_default_cagra_params(*this);
  build_all();
  search_all();
  check_neighbors();
}

INSTANTIATE_TEST_CASE_P(dynamic_batching, cagra_F32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(dynamic_batching, cagra_U8, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::dynamic_batching

namespace cuvs::neighbors::dynamic_batching {

/**
 * If the upstream search of a batch fails, every search call in the batch must report the error
 * (rather than wait forever for the batch results).
 */
TEST(dynamic_batching_upstream_error, every_request_throws)
{
  raft::resources res;
  constexpr int64_t n_rows = 10000;
  constexpr int64_t dim    = 32;
  constexpr int64_t k      = 64;
  // Small enough that the requests are batched
  constexpr int64_t n_queries_per_request = 3;
  constexpr int n_threads                 = 8;
  constexpr int n_requests_per_thread     = 4;

  auto dataset = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  auto queries =
    raft::make_device_matrix<float, int64_t>(res, n_threads * n_queries_per_request, dim);
  raft::random::RngState rng(1234ULL);
  raft::random::uniform(res, rng, dataset.data_handle(), dataset.size(), -1.0f, 1.0f);
  raft::random::uniform(res, rng, queries.data_handle(), queries.size(), -1.0f, 1.0f);

  cagra::index_params ip;
  ip.graph_degree              = 32;
  ip.intermediate_graph_degree = 64;
  auto upstream =
    build_cagra_with_dataset<float, uint32_t>(res, ip, raft::make_const_mdspan(dataset.view()));

  // itopk_size < k makes every upstream search fail.
  cagra::search_params upstream_params;
  upstream_params.itopk_size = 32;
  dynamic_batching::index<float, uint32_t> index{
    res, dynamic_batching::index_params{{}, k, 64, 3, false}, upstream, upstream_params};
  dynamic_batching::search_params params{};

  auto neighbors =
    raft::make_device_matrix<uint32_t, int64_t>(res, n_threads * n_queries_per_request, k);
  auto distances =
    raft::make_device_matrix<float, int64_t>(res, n_threads * n_queries_per_request, k);
  rmm::cuda_stream_pool streams(n_threads);
  std::vector<std::future<int>> futures;
  for (int t = 0; t < n_threads; t++) {
    futures.push_back(std::async(std::launch::async, [&, t]() {
      raft::resources thread_res = res;
      raft::resource::set_cuda_stream(thread_res, streams.get_stream(t));
      auto offset  = t * n_queries_per_request;
      int n_errors = 0;
      for (int r = 0; r < n_requests_per_thread; r++) {
        try {
          dynamic_batching::search(
            thread_res,
            params,
            index,
            raft::make_device_matrix_view<const float, int64_t>(
              queries.data_handle() + offset * dim, n_queries_per_request, dim),
            raft::make_device_matrix_view<uint32_t, int64_t>(
              neighbors.data_handle() + offset * k, n_queries_per_request, k),
            raft::make_device_matrix_view<float, int64_t>(
              distances.data_handle() + offset * k, n_queries_per_request, k));
          raft::resource::sync_stream(thread_res);
        } catch (const raft::exception&) {
          n_errors++;
        }
      }
      return n_errors;
    }));
  }
  for (auto& f : futures) {
    ASSERT_EQ(f.get(), n_requests_per_thread);
  }
}

}  // namespace cuvs::neighbors::dynamic_batching
