/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include "cagra_padded_build_helpers.cuh"
#include "naive_knn.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_resources_snmg.hpp>
#include <raft/core/resource/nccl_comm.hpp>

namespace cuvs::neighbors::mg {

enum class algo_t { IVF_FLAT, IVF_PQ, CAGRA };
enum class d_mode_t { REPLICATED, SHARDED, LOCAL_THEN_DISTRIBUTED, ROUND_ROBIN };
enum class m_mode_t { MERGE_ON_ROOT_RANK, TREE_MERGE, UNDEFINED };

struct AnnMGInputs {
  int64_t num_queries;
  int64_t num_db_vecs;
  int64_t dim;
  int64_t k;
  d_mode_t d_mode;
  m_mode_t m_mode;
  algo_t algo;
  int64_t nprobe;
  int64_t nlist;
  cuvs::distance::DistanceType metric;
  bool adaptive_centers;
};

template <typename T, typename DataT>
class AnnMGTest : public ::testing::TestWithParam<AnnMGInputs> {
 public:
  AnnMGTest()
    : clique_(),
      ps(::testing::TestWithParam<AnnMGInputs>::GetParam()),
      d_index_dataset(0, resource::get_cuda_stream(clique_)),
      d_queries(0, resource::get_cuda_stream(clique_)),
      h_index_dataset(0),
      h_queries(0)
  {
    clique_.set_memory_pool(80);
  }

  void testAnnMG()
  {
    size_t queries_size = ps.num_queries * ps.k;
    std::vector<int64_t> neighbors_ref(queries_size);
    std::vector<T> distances_ref(queries_size);
    std::vector<int64_t> neighbors_snmg_ann(queries_size);
    std::vector<T> distances_snmg_ann(queries_size);

    {
      rmm::device_uvector<T> distances_ref_dev(queries_size, resource::get_cuda_stream(clique_));
      rmm::device_uvector<int64_t> neighbors_ref_dev(queries_size,
                                                     resource::get_cuda_stream(clique_));
      cuvs::neighbors::naive_knn<T, DataT, int64_t>(clique_,
                                                    distances_ref_dev.data(),
                                                    neighbors_ref_dev.data(),
                                                    d_queries.data(),
                                                    d_index_dataset.data(),
                                                    ps.num_queries,
                                                    ps.num_db_vecs,
                                                    ps.dim,
                                                    ps.k,
                                                    ps.metric);
      update_host(distances_ref.data(),
                  distances_ref_dev.data(),
                  queries_size,
                  resource::get_cuda_stream(clique_));
      update_host(neighbors_ref.data(),
                  neighbors_ref_dev.data(),
                  queries_size,
                  resource::get_cuda_stream(clique_));
      resource::sync_stream(clique_);
    }

    int64_t n_rows_per_search_batch = 3000;  // [3000, 3000, 1000] == 7000 rows

    // IVF-Flat
    if (ps.algo == algo_t::IVF_FLAT &&
        (ps.d_mode == d_mode_t::REPLICATED || ps.d_mode == d_mode_t::SHARDED)) {
      distribution_mode d_mode;
      if (ps.d_mode == d_mode_t::REPLICATED)
        d_mode = distribution_mode::REPLICATED;
      else
        d_mode = distribution_mode::SHARDED;

      mg_index_params<ivf_flat::index_params> index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.adaptive_centers         = ps.adaptive_centers;
      index_params.add_data_on_build        = false;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;
      index_params.mode                     = d_mode;

      mg_search_params<ivf_flat::search_params> search_params;
      search_params.n_probes    = ps.nprobe;
      search_params.search_mode = LOAD_BALANCER;

      auto index_dataset = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_index_dataset.data(), ps.num_db_vecs, ps.dim);
      auto queries = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<int64_t, int64_t, row_major>(
        neighbors_snmg_ann.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, int64_t, row_major>(
        distances_snmg_ann.data(), ps.num_queries, ps.k);

      tmp_index_file index_file;
      {
        auto index = cuvs::neighbors::ivf_flat::build(clique_, index_params, index_dataset);
        cuvs::neighbors::ivf_flat::extend(clique_, index, index_dataset, std::nullopt);
        cuvs::neighbors::ivf_flat::serialize(clique_, index, index_file.filename);
      }
      auto new_index =
        cuvs::neighbors::ivf_flat::deserialize<DataT, int64_t>(clique_, index_file.filename);

      if (ps.m_mode == m_mode_t::MERGE_ON_ROOT_RANK)
        search_params.merge_mode = MERGE_ON_ROOT_RANK;
      else
        search_params.merge_mode = TREE_MERGE;

      search_params.n_rows_per_batch = n_rows_per_search_batch;
      cuvs::neighbors::ivf_flat::search(
        clique_, new_index, search_params, queries, neighbors, distances);
      resource::sync_stream(clique_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(neighbors_ref,
                                  neighbors_snmg_ann,
                                  distances_ref,
                                  distances_snmg_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(neighbors_snmg_ann.begin(), neighbors_snmg_ann.end(), 0);
      std::fill(distances_snmg_ann.begin(), distances_snmg_ann.end(), 0);
    }

    // IVF-PQ
    if (ps.algo == algo_t::IVF_PQ &&
        (ps.d_mode == d_mode_t::REPLICATED || ps.d_mode == d_mode_t::SHARDED)) {
      distribution_mode d_mode;
      if (ps.d_mode == d_mode_t::REPLICATED)
        d_mode = distribution_mode::REPLICATED;
      else
        d_mode = distribution_mode::SHARDED;

      mg_index_params<ivf_pq::index_params> index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.add_data_on_build        = false;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;
      index_params.mode                     = d_mode;

      mg_search_params<ivf_pq::search_params> search_params;
      search_params.n_probes    = ps.nprobe;
      search_params.search_mode = LOAD_BALANCER;

      auto index_dataset = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_index_dataset.data(), ps.num_db_vecs, ps.dim);
      auto queries = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<int64_t, int64_t, row_major>(
        neighbors_snmg_ann.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, int64_t, row_major>(
        distances_snmg_ann.data(), ps.num_queries, ps.k);

      tmp_index_file index_file;
      {
        auto index = cuvs::neighbors::ivf_pq::build(clique_, index_params, index_dataset);
        cuvs::neighbors::ivf_pq::extend(clique_, index, index_dataset, std::nullopt);
        cuvs::neighbors::ivf_pq::serialize(clique_, index, index_file.filename);
      }
      auto new_index =
        cuvs::neighbors::ivf_pq::deserialize<DataT, int64_t>(clique_, index_file.filename);

      if (ps.m_mode == m_mode_t::MERGE_ON_ROOT_RANK)
        search_params.merge_mode = MERGE_ON_ROOT_RANK;
      else
        search_params.merge_mode = TREE_MERGE;

      search_params.n_rows_per_batch = n_rows_per_search_batch;
      cuvs::neighbors::ivf_pq::search(
        clique_, new_index, search_params, queries, neighbors, distances);
      resource::sync_stream(clique_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(neighbors_ref,
                                  neighbors_snmg_ann,
                                  distances_ref,
                                  distances_snmg_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(neighbors_snmg_ann.begin(), neighbors_snmg_ann.end(), 0);
      std::fill(distances_snmg_ann.begin(), distances_snmg_ann.end(), 0);
    }

    // CAGRA
    if (ps.algo == algo_t::CAGRA &&
        (ps.d_mode == d_mode_t::REPLICATED || ps.d_mode == d_mode_t::SHARDED)) {
      distribution_mode d_mode;
      if (ps.d_mode == d_mode_t::REPLICATED)
        d_mode = distribution_mode::REPLICATED;
      else
        d_mode = distribution_mode::SHARDED;

      mg_index_params<cagra::index_params> index_params;
      // Host dataset uses ACE build path; must set ace_params (not ivf_pq_params).
      index_params.graph_build_params = cagra::graph_build_params::ace_params{};
      index_params.mode               = d_mode;

      mg_search_params<cagra::search_params> search_params;

      auto index_dataset = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_index_dataset.data(), ps.num_db_vecs, ps.dim);
      auto queries = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<int64_t, int64_t, row_major>(
        neighbors_snmg_ann.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, int64_t, row_major>(
        distances_snmg_ann.data(), ps.num_queries, ps.k);

      tmp_index_file index_file;
      {
        auto index = cuvs::neighbors::cagra::build(clique_, index_params, index_dataset);
        cuvs::neighbors::cagra::serialize(clique_, index, index_file.filename);
      }
      auto new_index =
        cuvs::neighbors::cagra::deserialize<DataT, uint32_t>(clique_, index_file.filename);

      if (ps.m_mode == m_mode_t::MERGE_ON_ROOT_RANK)
        search_params.merge_mode = MERGE_ON_ROOT_RANK;
      else
        search_params.merge_mode = TREE_MERGE;

      search_params.n_rows_per_batch = n_rows_per_search_batch;
      cuvs::neighbors::cagra::search(
        clique_, new_index, search_params, queries, neighbors, distances);
      resource::sync_stream(clique_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(neighbors_ref,
                                  neighbors_snmg_ann,
                                  distances_ref,
                                  distances_snmg_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(neighbors_snmg_ann.begin(), neighbors_snmg_ann.end(), 0);
      std::fill(distances_snmg_ann.begin(), distances_snmg_ann.end(), 0);
    }

    if (ps.algo == algo_t::IVF_FLAT && ps.d_mode == d_mode_t::LOCAL_THEN_DISTRIBUTED) {
      ivf_flat::index_params index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.adaptive_centers         = ps.adaptive_centers;
      index_params.add_data_on_build        = true;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;

      mg_search_params<ivf_flat::search_params> search_params;
      search_params.n_probes    = ps.nprobe;
      search_params.search_mode = LOAD_BALANCER;

      tmp_index_file index_file;
      {
        auto index_dataset = raft::make_device_matrix_view<const DataT, int64_t>(
          d_index_dataset.data(), ps.num_db_vecs, ps.dim);
        auto index = cuvs::neighbors::ivf_flat::build(clique_, index_params, index_dataset);
        ivf_flat::serialize(clique_, index_file.filename, index);
      }

      auto queries = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<int64_t, int64_t, row_major>(
        neighbors_snmg_ann.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, int64_t, row_major>(
        distances_snmg_ann.data(), ps.num_queries, ps.k);

      auto distributed_index =
        cuvs::neighbors::ivf_flat::distribute<DataT, int64_t>(clique_, index_file.filename);
      search_params.merge_mode = TREE_MERGE;

      search_params.n_rows_per_batch = n_rows_per_search_batch;
      cuvs::neighbors::ivf_flat::search(
        clique_, distributed_index, search_params, queries, neighbors, distances);

      resource::sync_stream(clique_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(neighbors_ref,
                                  neighbors_snmg_ann,
                                  distances_ref,
                                  distances_snmg_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(neighbors_snmg_ann.begin(), neighbors_snmg_ann.end(), 0);
      std::fill(distances_snmg_ann.begin(), distances_snmg_ann.end(), 0);
    }

    if (ps.algo == algo_t::IVF_PQ && ps.d_mode == d_mode_t::LOCAL_THEN_DISTRIBUTED) {
      ivf_pq::index_params index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.add_data_on_build        = true;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;

      mg_search_params<ivf_pq::search_params> search_params;
      search_params.n_probes    = ps.nprobe;
      search_params.search_mode = LOAD_BALANCER;

      tmp_index_file index_file;
      {
        auto index_dataset = raft::make_device_matrix_view<const DataT, int64_t>(
          d_index_dataset.data(), ps.num_db_vecs, ps.dim);
        auto index = cuvs::neighbors::ivf_pq::build(clique_, index_params, index_dataset);
        ivf_pq::serialize(clique_, index_file.filename, index);
      }

      auto queries = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<int64_t, int64_t, row_major>(
        neighbors_snmg_ann.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, int64_t, row_major>(
        distances_snmg_ann.data(), ps.num_queries, ps.k);

      auto distributed_index =
        cuvs::neighbors::ivf_pq::distribute<DataT, int64_t>(clique_, index_file.filename);
      search_params.merge_mode = TREE_MERGE;

      search_params.n_rows_per_batch = n_rows_per_search_batch;
      cuvs::neighbors::ivf_pq::search(
        clique_, distributed_index, search_params, queries, neighbors, distances);

      resource::sync_stream(clique_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(neighbors_ref,
                                  neighbors_snmg_ann,
                                  distances_ref,
                                  distances_snmg_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(neighbors_snmg_ann.begin(), neighbors_snmg_ann.end(), 0);
      std::fill(distances_snmg_ann.begin(), distances_snmg_ann.end(), 0);
    }

    if (ps.algo == algo_t::CAGRA && ps.d_mode == d_mode_t::LOCAL_THEN_DISTRIBUTED) {
      cagra::index_params index_params;
      index_params.graph_build_params = cagra::graph_build_params::ivf_pq_params(
        raft::matrix_extent<int64_t>(ps.num_db_vecs, ps.dim));

      mg_search_params<cagra::search_params> search_params;

      tmp_index_file index_file;
      {
        auto index_dataset = raft::make_device_matrix_view<const DataT, int64_t>(
          d_index_dataset.data(), ps.num_db_vecs, ps.dim);
        cuvs::neighbors::test::padded_device_matrix_for_cagra<DataT> padded(clique_, index_dataset);
        auto index = cuvs::neighbors::cagra::build(clique_, index_params, padded.view);
        index.update_dataset(clique_, padded.view);
        cuvs::neighbors::cagra::serialize(clique_, index_file.filename, index);
      }

      auto queries = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<int64_t, int64_t, row_major>(
        neighbors_snmg_ann.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, int64_t, row_major>(
        distances_snmg_ann.data(), ps.num_queries, ps.k);

      auto distributed_index =
        cuvs::neighbors::cagra::distribute<DataT, uint32_t>(clique_, index_file.filename);

      search_params.merge_mode = TREE_MERGE;

      search_params.n_rows_per_batch = n_rows_per_search_batch;
      cuvs::neighbors::cagra::search(
        clique_, distributed_index, search_params, queries, neighbors, distances);

      resource::sync_stream(clique_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(neighbors_ref,
                                  neighbors_snmg_ann,
                                  distances_ref,
                                  distances_snmg_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(neighbors_snmg_ann.begin(), neighbors_snmg_ann.end(), 0);
      std::fill(distances_snmg_ann.begin(), distances_snmg_ann.end(), 0);
    }

    if (ps.algo == algo_t::IVF_FLAT && ps.d_mode == d_mode_t::ROUND_ROBIN) {
      ASSERT_TRUE(ps.num_queries <= 4);

      mg_index_params<ivf_flat::index_params> index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.adaptive_centers         = ps.adaptive_centers;
      index_params.add_data_on_build        = false;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;
      index_params.mode                     = REPLICATED;

      mg_search_params<ivf_flat::search_params> search_params;
      search_params.n_probes    = ps.nprobe;
      search_params.search_mode = ROUND_ROBIN;

      auto index_dataset = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_index_dataset.data(), ps.num_db_vecs, ps.dim);
      auto small_batch_query = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);

      auto index = cuvs::neighbors::ivf_flat::build(clique_, index_params, index_dataset);
      cuvs::neighbors::ivf_flat::extend(clique_, index, index_dataset, std::nullopt);

      int n_parallel_searches = 16;
      std::vector<char> searches_correctness(n_parallel_searches);
      std::vector<int64_t> load_balancer_neighbors_snmg_ann(n_parallel_searches * ps.num_queries *
                                                            ps.k);
      std::vector<float> load_balancer_distances_snmg_ann(n_parallel_searches * ps.num_queries *
                                                          ps.k);

      // making sure that the NCCL comms are initialized before OMP section
      raft::resource::get_nccl_comms(clique_);

#pragma omp parallel for
      for (uint64_t search_idx = 0; search_idx < searches_correctness.size(); search_idx++) {
        uint64_t offset            = search_idx * ps.num_queries * ps.k;
        auto small_batch_neighbors = raft::make_host_matrix_view<int64_t, int64_t, row_major>(
          load_balancer_neighbors_snmg_ann.data() + offset, ps.num_queries, ps.k);
        auto small_batch_distances = raft::make_host_matrix_view<float, int64_t, row_major>(
          load_balancer_distances_snmg_ann.data() + offset, ps.num_queries, ps.k);

        search_params.n_rows_per_batch = n_rows_per_search_batch;
        cuvs::neighbors::ivf_flat::search(clique_,
                                          index,
                                          search_params,
                                          small_batch_query,
                                          small_batch_neighbors,
                                          small_batch_distances);

        std::vector<int64_t> small_batch_neighbors_vec(
          small_batch_neighbors.data_handle(),
          small_batch_neighbors.data_handle() + small_batch_neighbors.size());
        std::vector<float> small_batch_distances_vec(
          small_batch_distances.data_handle(),
          small_batch_distances.data_handle() + small_batch_distances.size());
        searches_correctness[search_idx] = eval_neighbours(neighbors_ref,
                                                           small_batch_neighbors_vec,
                                                           distances_ref,
                                                           small_batch_distances_vec,
                                                           ps.num_queries,
                                                           ps.k,
                                                           0.001,
                                                           0.9);
      }
      ASSERT_TRUE(std::all_of(searches_correctness.begin(),
                              searches_correctness.end(),
                              [](char val) { return val != 0; }));
    }

    if (ps.algo == algo_t::IVF_PQ && ps.d_mode == d_mode_t::ROUND_ROBIN) {
      ASSERT_TRUE(ps.num_queries <= 4);

      mg_index_params<ivf_pq::index_params> index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.add_data_on_build        = false;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;
      index_params.mode                     = REPLICATED;

      mg_search_params<ivf_pq::search_params> search_params;
      search_params.n_probes    = ps.nprobe;
      search_params.search_mode = ROUND_ROBIN;

      auto index_dataset = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_index_dataset.data(), ps.num_db_vecs, ps.dim);
      auto small_batch_query = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);

      auto index = cuvs::neighbors::ivf_pq::build(clique_, index_params, index_dataset);
      cuvs::neighbors::ivf_pq::extend(clique_, index, index_dataset, std::nullopt);

      int n_parallel_searches = 16;
      std::vector<char> searches_correctness(n_parallel_searches);
      std::vector<int64_t> load_balancer_neighbors_snmg_ann(n_parallel_searches * ps.num_queries *
                                                            ps.k);
      std::vector<float> load_balancer_distances_snmg_ann(n_parallel_searches * ps.num_queries *
                                                          ps.k);

      // making sure that the NCCL comms are initialized before OMP section
      raft::resource::get_nccl_comms(clique_);

#pragma omp parallel for
      for (uint64_t search_idx = 0; search_idx < searches_correctness.size(); search_idx++) {
        uint64_t offset            = search_idx * ps.num_queries * ps.k;
        auto small_batch_neighbors = raft::make_host_matrix_view<int64_t, int64_t, row_major>(
          load_balancer_neighbors_snmg_ann.data() + offset, ps.num_queries, ps.k);
        auto small_batch_distances = raft::make_host_matrix_view<float, int64_t, row_major>(
          load_balancer_distances_snmg_ann.data() + offset, ps.num_queries, ps.k);

        search_params.n_rows_per_batch = n_rows_per_search_batch;
        cuvs::neighbors::ivf_pq::search(clique_,
                                        index,
                                        search_params,
                                        small_batch_query,
                                        small_batch_neighbors,
                                        small_batch_distances);

        std::vector<int64_t> small_batch_neighbors_vec(
          small_batch_neighbors.data_handle(),
          small_batch_neighbors.data_handle() + small_batch_neighbors.size());
        std::vector<float> small_batch_distances_vec(
          small_batch_distances.data_handle(),
          small_batch_distances.data_handle() + small_batch_distances.size());
        searches_correctness[search_idx] = eval_neighbours(neighbors_ref,
                                                           small_batch_neighbors_vec,
                                                           distances_ref,
                                                           small_batch_distances_vec,
                                                           ps.num_queries,
                                                           ps.k,
                                                           0.001,
                                                           0.9);
      }
      ASSERT_TRUE(std::all_of(searches_correctness.begin(),
                              searches_correctness.end(),
                              [](char val) { return val != 0; }));
    }

    if (ps.algo == algo_t::CAGRA && ps.d_mode == d_mode_t::ROUND_ROBIN) {
      ASSERT_TRUE(ps.num_queries <= 4);

      mg_index_params<cagra::index_params> index_params;
      // Host dataset uses ACE build path; must set ace_params (not ivf_pq_params).
      index_params.graph_build_params = cagra::graph_build_params::ace_params{};
      index_params.mode               = REPLICATED;

      mg_search_params<cagra::search_params> search_params;
      search_params.search_mode = ROUND_ROBIN;

      auto index_dataset = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_index_dataset.data(), ps.num_db_vecs, ps.dim);
      auto small_batch_query = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);

      auto index = cuvs::neighbors::cagra::build(clique_, index_params, index_dataset);

      int n_parallel_searches = 16;
      std::vector<char> searches_correctness(n_parallel_searches);
      std::vector<int64_t> load_balancer_neighbors_snmg_ann(n_parallel_searches * ps.num_queries *
                                                            ps.k);
      std::vector<float> load_balancer_distances_snmg_ann(n_parallel_searches * ps.num_queries *
                                                          ps.k);

      // making sure that the NCCL comms are initialized before OMP section
      raft::resource::get_nccl_comms(clique_);
#pragma omp parallel for
      for (uint64_t search_idx = 0; search_idx < searches_correctness.size(); search_idx++) {
        uint64_t offset            = search_idx * ps.num_queries * ps.k;
        auto small_batch_neighbors = raft::make_host_matrix_view<int64_t, int64_t, row_major>(
          load_balancer_neighbors_snmg_ann.data() + offset, ps.num_queries, ps.k);
        auto small_batch_distances = raft::make_host_matrix_view<float, int64_t, row_major>(
          load_balancer_distances_snmg_ann.data() + offset, ps.num_queries, ps.k);

        search_params.n_rows_per_batch = n_rows_per_search_batch;
        cuvs::neighbors::cagra::search(clique_,
                                       index,
                                       search_params,
                                       small_batch_query,
                                       small_batch_neighbors,
                                       small_batch_distances);

        std::vector<int64_t> small_batch_neighbors_vec(
          small_batch_neighbors.data_handle(),
          small_batch_neighbors.data_handle() + small_batch_neighbors.size());
        std::vector<float> small_batch_distances_vec(
          small_batch_distances.data_handle(),
          small_batch_distances.data_handle() + small_batch_distances.size());
        searches_correctness[search_idx] = eval_neighbours(neighbors_ref,
                                                           small_batch_neighbors_vec,
                                                           distances_ref,
                                                           small_batch_distances_vec,
                                                           ps.num_queries,
                                                           ps.k,
                                                           0.001,
                                                           0.9);
      }
      ASSERT_TRUE(std::all_of(searches_correctness.begin(),
                              searches_correctness.end(),
                              [](char val) { return val != 0; }));
    }
  }

  void SetUp() override
  {
    d_index_dataset.resize(ps.num_db_vecs * ps.dim, resource::get_cuda_stream(clique_));
    d_queries.resize(ps.num_queries * ps.dim, resource::get_cuda_stream(clique_));
    h_index_dataset.resize(ps.num_db_vecs * ps.dim);
    h_queries.resize(ps.num_queries * ps.dim);

    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::uniform(
        clique_, r, d_index_dataset.data(), d_index_dataset.size(), DataT(0.1), DataT(2.0));
      raft::random::uniform(clique_, r, d_queries.data(), d_queries.size(), DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        clique_, r, d_index_dataset.data(), d_index_dataset.size(), DataT(1), DataT(20));
      raft::random::uniformInt(clique_, r, d_queries.data(), d_queries.size(), DataT(1), DataT(20));
    }

    raft::copy(h_index_dataset.data(),
               d_index_dataset.data(),
               d_index_dataset.size(),
               resource::get_cuda_stream(clique_));
    raft::copy(
      h_queries.data(), d_queries.data(), d_queries.size(), resource::get_cuda_stream(clique_));
    resource::sync_stream(clique_);
  }

  void TearDown() override {}

 private:
  raft::device_resources_snmg clique_;
  AnnMGInputs ps;
  std::vector<DataT> h_index_dataset;
  std::vector<DataT> h_queries;
  rmm::device_uvector<DataT> d_index_dataset;
  rmm::device_uvector<DataT> d_queries;
};

const std::vector<AnnMGInputs> inputs = {
  {7000,
   10000,
   8,
   16,
   d_mode_t::REPLICATED,
   m_mode_t::UNDEFINED,
   algo_t::IVF_FLAT,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::REPLICATED,
   m_mode_t::UNDEFINED,
   algo_t::IVF_PQ,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::REPLICATED,
   m_mode_t::UNDEFINED,
   algo_t::CAGRA,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::SHARDED,
   m_mode_t::MERGE_ON_ROOT_RANK,
   algo_t::IVF_FLAT,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::SHARDED,
   m_mode_t::MERGE_ON_ROOT_RANK,
   algo_t::IVF_PQ,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::SHARDED,
   m_mode_t::MERGE_ON_ROOT_RANK,
   algo_t::CAGRA,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::SHARDED,
   m_mode_t::TREE_MERGE,
   algo_t::IVF_FLAT,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::SHARDED,
   m_mode_t::TREE_MERGE,
   algo_t::IVF_PQ,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::SHARDED,
   m_mode_t::TREE_MERGE,
   algo_t::CAGRA,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::LOCAL_THEN_DISTRIBUTED,
   m_mode_t::UNDEFINED,
   algo_t::IVF_FLAT,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::LOCAL_THEN_DISTRIBUTED,
   m_mode_t::UNDEFINED,
   algo_t::IVF_PQ,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::LOCAL_THEN_DISTRIBUTED,
   m_mode_t::UNDEFINED,
   algo_t::CAGRA,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {3,
   10000,
   8,
   16,
   d_mode_t::ROUND_ROBIN,
   m_mode_t::UNDEFINED,
   algo_t::IVF_FLAT,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {3,
   10000,
   8,
   16,
   d_mode_t::ROUND_ROBIN,
   m_mode_t::UNDEFINED,
   algo_t::IVF_PQ,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {3,
   10000,
   8,
   16,
   d_mode_t::ROUND_ROBIN,
   m_mode_t::UNDEFINED,
   algo_t::CAGRA,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
};
}  // namespace cuvs::neighbors::mg
