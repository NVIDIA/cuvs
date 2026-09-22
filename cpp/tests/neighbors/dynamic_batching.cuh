/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "ann_utils.cuh"

#include <gtest/gtest.h>

#include <cuvs/neighbors/dynamic_batching.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/rng.cuh>

#include <rmm/mr/pool_memory_resource.hpp>

#include <cstdint>
#include <future>
#include <vector>

namespace cuvs::neighbors::dynamic_batching {

struct dynamic_batching_spec {
  int64_t n_queries                   = 1000;
  int64_t n_rows                      = 100000;
  int64_t dim                         = 128;
  int64_t k                           = 10;
  int64_t max_batch_size              = 64;
  size_t n_queues                     = 3;
  bool conservative_dispatch          = false;
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
  int64_t max_concurrent_threads      = 128;
};

inline ::std::ostream& operator<<(::std::ostream& os, const dynamic_batching_spec& p)
{
  os << "{n_queries=" << p.n_queries;
  os << ", dataset shape=" << p.n_rows << "x" << p.dim;
  os << ", metric=" << print_metric{p.metric};
  os << ", k=" << p.k;
  os << ", max_batch_size=" << p.max_batch_size;
  os << ", n_queues=" << p.n_queues;
  os << ", conservative_dispatch=" << p.conservative_dispatch;
  os << '}';
  return os;
}

template <typename DataT, typename IdxT, typename UpstreamT>
using build_function = UpstreamT(const raft::resources&,
                                 const typename UpstreamT::index_params_type&,
                                 raft::device_matrix_view<const DataT, int64_t, raft::row_major>);

template <typename DataT, typename IdxT, typename UpstreamT>
using search_function = void(const raft::resources&,
                             const typename UpstreamT::search_params_type& params,
                             const UpstreamT& index,
                             raft::device_matrix_view<const DataT, int64_t, raft::row_major>,
                             raft::device_matrix_view<IdxT, int64_t, raft::row_major>,
                             raft::device_matrix_view<float, int64_t, raft::row_major>,
                             const cuvs::neighbors::filtering::base_filter&);

template <typename DataT,
          typename IdxT,
          typename UpstreamT,
          build_function<DataT, IdxT, UpstreamT> UpstreamBuildF,
          search_function<DataT, IdxT, UpstreamT> UpstreamSearchF>
struct dynamic_batching_test : public ::testing::TestWithParam<dynamic_batching_spec> {
  using distance_type = float;
  using data_type     = DataT;
  using index_type    = IdxT;
  using upstream_type = UpstreamT;

  dynamic_batching_spec ps = ::testing::TestWithParam<dynamic_batching_spec>::GetParam();
  raft::resources res;

  // input data
  std::optional<raft::device_matrix<data_type, int64_t>> dataset            = std::nullopt;
  std::optional<raft::device_matrix<data_type, int64_t>> queries            = std::nullopt;
  std::optional<raft::device_matrix<index_type, int64_t>> neighbors_upsm    = std::nullopt;
  std::optional<raft::device_matrix<index_type, int64_t>> neighbors_dynb    = std::nullopt;
  std::optional<raft::device_matrix<distance_type, int64_t>> distances_upsm = std::nullopt;
  std::optional<raft::device_matrix<distance_type, int64_t>> distances_dynb = std::nullopt;

  // build parameters
  cuvs::neighbors::index_params build_params_base{ps.metric};
  typename upstream_type::index_params_type build_params_upsm{build_params_base};
  dynamic_batching::index_params build_params_dynb{
    build_params_base, ps.k, ps.max_batch_size, ps.n_queues, ps.conservative_dispatch};

  // search parameters
  typename upstream_type::search_params_type search_params_upsm{};
  dynamic_batching::search_params search_params_dynb{};

  // indexes
  std::optional<upstream_type> index_upsm                                  = std::nullopt;
  std::optional<dynamic_batching::index<data_type, index_type>> index_dynb = std::nullopt;

  void build_all()
  {
    index_dynb.reset();
    index_upsm.reset();
    index_upsm = UpstreamBuildF(res, build_params_upsm, dataset->view());
    index_dynb.emplace(res, build_params_dynb, index_upsm.value(), search_params_upsm);
  }

  void search_all()
  {
    // Search using upstream index - all queries at once
    UpstreamSearchF(res,
                    search_params_upsm,
                    index_upsm.value(),
                    queries->view(),
                    neighbors_upsm->view(),
                    distances_upsm->view(),
                    filtering::none_sample_filter{});
    raft::resource::sync_stream(res);

    // Search with dynamic batching
    // Streaming scenario: prepare concurrent resources
    rmm::cuda_stream_pool worker_streams(ps.max_concurrent_threads);
    std::vector<std::future<void>> futures(ps.max_concurrent_threads);
    std::vector<raft::resources> resource_pool(0);
    for (int64_t i = 0; i < ps.max_concurrent_threads; i++) {
      resource_pool.push_back(res);  // copies the resource
      raft::resource::set_cuda_stream(resource_pool[i], worker_streams.get_stream(i));
    }

    // Try multiple batch sizes in a round-robin to improve test coverage
    std::vector<int64_t> minibatch_sizes{1, 3, 7, 10};
    auto get_bs = [&minibatch_sizes](auto i) {
      return minibatch_sizes[i % minibatch_sizes.size()];
    };
    int64_t i = 0;
    for (int64_t offset = 0; offset < ps.n_queries; offset += get_bs(i++)) {
      auto bs = std::min<int64_t>(get_bs(i), ps.n_queries - offset);
      auto j  = i % ps.max_concurrent_threads;
      // wait for previous job in the same slot to finish
      if (i >= ps.max_concurrent_threads) { futures[j].wait(); }
      // submit a new job
      futures[j] = std::async(
        std::launch::async,
        [&res       = resource_pool[j],
         &params    = search_params_dynb,
         index      = index_dynb.value(),
         query_view = raft::make_device_matrix_view<data_type, int64_t>(
           queries->data_handle() + offset * ps.dim, bs, ps.dim),
         neighbors_view = raft::make_device_matrix_view<index_type, int64_t>(
           neighbors_dynb->data_handle() + offset * ps.k, bs, ps.k),
         distances_view = raft::make_device_matrix_view<distance_type, int64_t>(
           distances_dynb->data_handle() + offset * ps.k, bs, ps.k)]() {
          dynamic_batching::search(res, params, index, query_view, neighbors_view, distances_view);
        });
    }

    // finalize all resources
    for (int64_t j = 0; j < ps.max_concurrent_threads && j < i; j++) {
      futures[j].wait();
      raft::resource::sync_stream(resource_pool[j]);
    }
    raft::resource::sync_stream(res);
  }

  /*
    Check the dynamic batching generated neighbors against the upstream index. They both may be
    imperfect w.r.t. the ground truth, but they shouldn't differ too much.
   */
  void check_neighbors()
  {
    auto stream         = raft::resource::get_cuda_stream(res);
    size_t queries_size = ps.n_queries * ps.k;
    std::vector<index_type> neighbors_upsm_host(queries_size);
    std::vector<index_type> neighbors_dynb_host(queries_size);
    std::vector<distance_type> distances_upsm_host(queries_size);
    std::vector<distance_type> distances_dynb_host(queries_size);
    raft::copy(neighbors_upsm_host.data(), neighbors_upsm->data_handle(), queries_size, stream);
    raft::copy(neighbors_dynb_host.data(), neighbors_dynb->data_handle(), queries_size, stream);
    raft::copy(distances_upsm_host.data(), distances_upsm->data_handle(), queries_size, stream);
    raft::copy(distances_dynb_host.data(), distances_dynb->data_handle(), queries_size, stream);
    raft::resource::sync_stream(res);
    ASSERT_TRUE(eval_neighbours(neighbors_upsm_host,
                                neighbors_dynb_host,
                                distances_upsm_host,
                                distances_dynb_host,
                                ps.n_queries,
                                ps.k,
                                0.001,
                                0.9))
      << ps;
  }

  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<data_type, int64_t>(res, ps.n_rows, ps.dim));
    queries.emplace(raft::make_device_matrix<data_type, int64_t>(res, ps.n_queries, ps.dim));
    neighbors_upsm.emplace(raft::make_device_matrix<index_type, int64_t>(res, ps.n_queries, ps.k));
    neighbors_dynb.emplace(raft::make_device_matrix<index_type, int64_t>(res, ps.n_queries, ps.k));
    distances_upsm.emplace(
      raft::make_device_matrix<distance_type, int64_t>(res, ps.n_queries, ps.k));
    distances_dynb.emplace(
      raft::make_device_matrix<distance_type, int64_t>(res, ps.n_queries, ps.k));

    raft::random::RngState rng(666ULL);
    if constexpr (std::is_same_v<data_type, float> || std::is_same_v<data_type, half>) {
      raft::random::uniform(
        res, rng, dataset->data_handle(), dataset->size(), data_type(0.1), data_type(2.0));
      raft::random::uniform(
        res, rng, queries->data_handle(), queries->size(), data_type(0.1), data_type(2.0));
    } else {
      raft::random::uniformInt(
        res, rng, dataset->data_handle(), dataset->size(), data_type(1), data_type(20));
      raft::random::uniformInt(
        res, rng, queries->data_handle(), queries->size(), data_type(1), data_type(20));
    }
    raft::resource::sync_stream(res);
  }

  void TearDown() override
  {
    index_dynb.reset();
    index_upsm.reset();
    dataset.reset();
    queries.reset();
    neighbors_upsm.reset();
    neighbors_dynb.reset();
    distances_upsm.reset();
    distances_dynb.reset();
    raft::resource::sync_stream(res);
  }
};

inline std::vector<dynamic_batching_spec> generate_inputs()
{
  std::vector<dynamic_batching_spec> inputs{dynamic_batching_spec{}};

  for (auto alt_n_queries : {10, 50, 100}) {
    dynamic_batching_spec input{};
    input.n_queries = alt_n_queries;
    inputs.push_back(input);
  }

  for (auto alt_k : {100, 200}) {
    dynamic_batching_spec input{};
    input.k = alt_k;
    inputs.push_back(input);
  }

  for (auto alt_max_batch_size : {4, 16, 128, 256, 512, 1024}) {
    dynamic_batching_spec input{};
    input.max_batch_size = alt_max_batch_size;
    inputs.push_back(input);
  }

  for (auto alt_n_queues : {1, 2, 16, 32}) {
    dynamic_batching_spec input{};
    input.n_queues = alt_n_queues;
    inputs.push_back(input);
  }

  for (auto alt_max_concurrent_threads : {1, 2, 16, 32}) {
    dynamic_batching_spec input{};
    input.max_concurrent_threads = alt_max_concurrent_threads;
    inputs.push_back(input);
  }

  {
    auto n = inputs.size();
    for (size_t i = 0; i < n; i++) {
      auto input                  = inputs[i];
      input.conservative_dispatch = !input.conservative_dispatch;
      inputs.push_back(input);
    }
  }

  return inputs;
}

const std::vector<dynamic_batching_spec> inputs = generate_inputs();

}  // namespace cuvs::neighbors::dynamic_batching
