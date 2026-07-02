/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include "naive_knn.cuh"

#include <cuvs/neighbors/ivf_sq.hpp>
#include <raft/core/bitset.cuh>
#include <raft/linalg/add.cuh>
#include <raft/linalg/normalize.cuh>

#include <raft/core/resource/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <numeric>

namespace cuvs::neighbors::ivf_sq {

struct test_ivf_sample_filter {
  static constexpr unsigned offset = 300;
};

template <typename IdxT>
struct AnnIvfSqInputs {
  IdxT num_queries;
  IdxT num_db_vecs;
  IdxT dim;
  IdxT k;
  IdxT nprobe;
  IdxT nlist;
  cuvs::distance::DistanceType metric;
  bool host_dataset = false;
};

template <typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const AnnIvfSqInputs<IdxT>& p)
{
  os << "{ " << p.num_queries << ", " << p.num_db_vecs << ", " << p.dim << ", " << p.k << ", "
     << p.nprobe << ", " << p.nlist << ", "
     << cuvs::neighbors::print_metric{static_cast<cuvs::distance::DistanceType>((int)p.metric)}
     << ", " << (p.host_dataset ? "host" : "device") << '}' << std::endl;
  return os;
}

template <typename T, typename DataT, typename IdxT>
class AnnIVFSQTest : public ::testing::TestWithParam<AnnIvfSqInputs<IdxT>> {
 public:
  AnnIVFSQTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnIvfSqInputs<IdxT>>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

  void testAll()
  {
    auto naive = compute_naive_knn();
    auto idx   = build_index(true);

    {
      SCOPED_TRACE("Search");
      checkSearch(idx, naive);
    }
    {
      SCOPED_TRACE("Serialize");
      checkSerialize(idx);
    }
    {
      SCOPED_TRACE("Filter");
      checkFilter(idx);
    }
    {
      SCOPED_TRACE("Extend");
      checkExtend(naive);
    }
  }

 protected:
  struct SearchResults {
    std::vector<IdxT> indices;
    std::vector<T> distances;
  };

  void checkSearch(const cuvs::neighbors::ivf_sq::index<uint8_t>& idx, const SearchResults& naive)
  {
    auto results = search_index(idx);

    float eps = 0.1;
    ASSERT_TRUE(eval_neighbours(naive.indices,
                                results.indices,
                                naive.distances,
                                results.distances,
                                ps.num_queries,
                                ps.k,
                                eps,
                                min_recall_threshold()));
  }

  void checkSerialize(const cuvs::neighbors::ivf_sq::index<uint8_t>& idx)
  {
    tmp_index_file index_file;
    cuvs::neighbors::ivf_sq::serialize(handle_, index_file.filename, idx);
    cuvs::neighbors::ivf_sq::index<uint8_t> index_loaded(handle_);
    cuvs::neighbors::ivf_sq::deserialize(handle_, index_file.filename, &index_loaded);

    ASSERT_EQ(idx.size(), index_loaded.size());
    ASSERT_EQ(idx.dim(), index_loaded.dim());
    ASSERT_EQ(idx.n_lists(), index_loaded.n_lists());

    auto results_orig   = search_index(idx);
    auto results_loaded = search_index(index_loaded);

    float eps = 0.001;
    ASSERT_TRUE(eval_neighbours(results_orig.indices,
                                results_loaded.indices,
                                results_orig.distances,
                                results_loaded.distances,
                                ps.num_queries,
                                ps.k,
                                eps,
                                1.0));
  }

  void checkExtend(const SearchResults& naive)
  {
    auto idx_empty = build_index(false);
    extend_index(&idx_empty);

    auto results = search_index(idx_empty);

    float eps = 0.1;
    ASSERT_TRUE(eval_neighbours(naive.indices,
                                results.indices,
                                naive.distances,
                                results.distances,
                                ps.num_queries,
                                ps.k,
                                eps,
                                min_recall_threshold()));
  }

  void checkFilter(const cuvs::neighbors::ivf_sq::index<uint8_t>& idx)
  {
    if (ps.num_db_vecs <= static_cast<IdxT>(test_ivf_sample_filter::offset)) {
      GTEST_SKIP() << "Skipping filter test: num_db_vecs <= filter offset";
    }

    size_t queries_size = ps.num_queries * ps.k;
    std::vector<IdxT> indices_ivfsq(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<T> distances_ivfsq(queries_size);
    std::vector<T> distances_naive(queries_size);

    {
      rmm::device_uvector<T> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      auto* database_filtered_ptr = database.data() + test_ivf_sample_filter::offset * ps.dim;
      cuvs::neighbors::naive_knn<T, DataT, IdxT>(handle_,
                                                 distances_naive_dev.data(),
                                                 indices_naive_dev.data(),
                                                 search_queries.data(),
                                                 database_filtered_ptr,
                                                 ps.num_queries,
                                                 ps.num_db_vecs - test_ivf_sample_filter::offset,
                                                 ps.dim,
                                                 ps.k,
                                                 ps.metric);
      raft::linalg::addScalar(indices_naive_dev.data(),
                              indices_naive_dev.data(),
                              IdxT(test_ivf_sample_filter::offset),
                              queries_size,
                              stream_);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      double min_recall =
        std::min(1.0, static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist));

      rmm::device_uvector<T> distances_ivfsq_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_ivfsq_dev(queries_size, stream_);

      {
        cuvs::neighbors::ivf_sq::search_params search_params;
        search_params.n_probes = ps.nprobe;

        auto removed_indices =
          raft::make_device_vector<IdxT, int64_t>(handle_, test_ivf_sample_filter::offset);
        raft::linalg::map_offset(handle_, removed_indices.view(), raft::identity_op{});
        raft::resource::sync_stream(handle_);

        cuvs::core::bitset<std::uint32_t, IdxT> removed_indices_bitset(
          handle_, removed_indices.view(), ps.num_db_vecs);
        auto bitset_filter_obj =
          cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset.view());

        auto search_queries_view = raft::make_device_matrix_view<const DataT, IdxT>(
          search_queries.data(), ps.num_queries, ps.dim);
        auto indices_out_view =
          raft::make_device_matrix_view<IdxT, IdxT>(indices_ivfsq_dev.data(), ps.num_queries, ps.k);
        auto dists_out_view =
          raft::make_device_matrix_view<T, IdxT>(distances_ivfsq_dev.data(), ps.num_queries, ps.k);

        cuvs::neighbors::ivf_sq::search(handle_,
                                        search_params,
                                        idx,
                                        search_queries_view,
                                        indices_out_view,
                                        dists_out_view,
                                        bitset_filter_obj);

        raft::update_host(
          distances_ivfsq.data(), distances_ivfsq_dev.data(), queries_size, stream_);
        raft::update_host(indices_ivfsq.data(), indices_ivfsq_dev.data(), queries_size, stream_);
        raft::resource::sync_stream(handle_);
      }
      float eps = 0.1;
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ivfsq,
                                  distances_naive,
                                  distances_ivfsq,
                                  ps.num_queries,
                                  ps.k,
                                  eps,
                                  min_recall));
    }
  }

  void SetUp() override
  {
    database.resize(ps.num_db_vecs * ps.dim, stream_);
    search_queries.resize(ps.num_queries * ps.dim, stream_);

    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same_v<DataT, float> || std::is_same_v<DataT, half>) {
      raft::random::uniform(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0));
      raft::random::uniform(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(0.1), DataT(2.0));
    }
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

  double min_recall_threshold()
  {
    return std::min(1.0, static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist));
  }

  SearchResults compute_naive_knn()
  {
    size_t queries_size = ps.num_queries * ps.k;
    SearchResults results;
    results.indices.resize(queries_size);
    results.distances.resize(queries_size);

    rmm::device_uvector<T> distances_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_dev(queries_size, stream_);
    cuvs::neighbors::naive_knn<T, DataT, IdxT>(handle_,
                                               distances_dev.data(),
                                               indices_dev.data(),
                                               search_queries.data(),
                                               database.data(),
                                               ps.num_queries,
                                               ps.num_db_vecs,
                                               ps.dim,
                                               ps.k,
                                               ps.metric);
    raft::update_host(results.distances.data(), distances_dev.data(), queries_size, stream_);
    raft::update_host(results.indices.data(), indices_dev.data(), queries_size, stream_);
    raft::resource::sync_stream(handle_);
    return results;
  }

  cuvs::neighbors::ivf_sq::index<uint8_t> build_index(bool add_data_on_build)
  {
    cuvs::neighbors::ivf_sq::index_params index_params;
    index_params.n_lists                      = ps.nlist;
    index_params.metric                       = ps.metric;
    index_params.add_data_on_build            = add_data_on_build;
    index_params.max_train_points_per_cluster = 256;

    if (!ps.host_dataset) {
      auto database_view = raft::make_device_matrix_view<const DataT, IdxT>(
        (const DataT*)database.data(), ps.num_db_vecs, ps.dim);
      return cuvs::neighbors::ivf_sq::build(handle_, index_params, database_view);
    } else {
      auto host_database = raft::make_host_matrix<DataT, IdxT>(ps.num_db_vecs, ps.dim);
      raft::copy(host_database.data_handle(), database.data(), ps.num_db_vecs * ps.dim, stream_);
      raft::resource::sync_stream(handle_);
      return cuvs::neighbors::ivf_sq::build(
        handle_, index_params, raft::make_const_mdspan(host_database.view()));
    }
  }

  void extend_index(cuvs::neighbors::ivf_sq::index<uint8_t>* idx)
  {
    if (!ps.host_dataset) {
      auto database_view = raft::make_device_matrix_view<const DataT, IdxT>(
        (const DataT*)database.data(), ps.num_db_vecs, ps.dim);
      auto vector_indices = raft::make_device_vector<IdxT, IdxT>(handle_, ps.num_db_vecs);
      raft::linalg::map_offset(handle_, vector_indices.view(), raft::identity_op{});
      raft::resource::sync_stream(handle_);

      auto indices_view = raft::make_device_vector_view<const IdxT, IdxT>(
        vector_indices.data_handle(), ps.num_db_vecs);
      cuvs::neighbors::ivf_sq::extend(
        handle_,
        database_view,
        std::make_optional<raft::device_vector_view<const IdxT, IdxT>>(indices_view),
        idx);
    } else {
      auto host_database = raft::make_host_matrix<DataT, IdxT>(ps.num_db_vecs, ps.dim);
      raft::copy(host_database.data_handle(), database.data(), ps.num_db_vecs * ps.dim, stream_);
      raft::resource::sync_stream(handle_);

      auto vector_indices = raft::make_host_vector<IdxT>(handle_, ps.num_db_vecs);
      std::iota(
        vector_indices.data_handle(), vector_indices.data_handle() + ps.num_db_vecs, IdxT(0));

      auto indices_view =
        raft::make_host_vector_view<const IdxT, IdxT>(vector_indices.data_handle(), ps.num_db_vecs);
      auto host_database_view = raft::make_host_matrix_view<const DataT, IdxT>(
        host_database.data_handle(), ps.num_db_vecs, ps.dim);
      cuvs::neighbors::ivf_sq::extend(
        handle_,
        host_database_view,
        std::make_optional<raft::host_vector_view<const IdxT, IdxT>>(indices_view),
        idx);
    }
  }

  SearchResults search_index(const cuvs::neighbors::ivf_sq::index<uint8_t>& idx)
  {
    size_t queries_size = ps.num_queries * ps.k;
    SearchResults results;
    results.indices.resize(queries_size);
    results.distances.resize(queries_size);

    cuvs::neighbors::ivf_sq::search_params search_params;
    search_params.n_probes = ps.nprobe;

    rmm::device_uvector<T> distances_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_dev(queries_size, stream_);

    auto search_queries_view = raft::make_device_matrix_view<const DataT, IdxT>(
      search_queries.data(), ps.num_queries, ps.dim);
    auto indices_out_view =
      raft::make_device_matrix_view<IdxT, IdxT>(indices_dev.data(), ps.num_queries, ps.k);
    auto dists_out_view =
      raft::make_device_matrix_view<T, IdxT>(distances_dev.data(), ps.num_queries, ps.k);

    cuvs::neighbors::ivf_sq::search(
      handle_, search_params, idx, search_queries_view, indices_out_view, dists_out_view);

    raft::update_host(results.distances.data(), distances_dev.data(), queries_size, stream_);
    raft::update_host(results.indices.data(), indices_dev.data(), queries_size, stream_);
    raft::resource::sync_stream(handle_);
    return results;
  }

  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnIvfSqInputs<IdxT> ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

const std::vector<AnnIvfSqInputs<int64_t>> inputs = {
  // num_queries, num_db_vecs, dim, k, nprobe, nlist, metric

  // ===== Dimension edge cases (all four metrics) =====
  // dim=1 (CosineExpanded excluded: requires dim > 1)
  {1000, 10000, 1, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 1, 10, 40, 1024, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 1, 10, 40, 1024, cuvs::distance::DistanceType::L2SqrtExpanded},
  // dim=2,3,4,5 (unaligned)
  {1000, 10000, 2, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 2, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 3, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 3, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 4, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 4, 16, 40, 1024, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 5, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 5, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  // dim=7,8 (around veclen=16 boundary, not a multiple of veclen)
  {1000, 10000, 7, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 7, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 8, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 8, 16, 40, 1024, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 8, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  // dim=15,16,17 (around veclen=16 boundary)
  {1000, 10000, 15, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 15, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2SqrtExpanded},
  {1000, 10000, 17, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 17, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  // dim=31,32,33 (around 2*veclen boundary)
  {1000, 10000, 31, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 31, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 32, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 32, 10, 40, 1024, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 32, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 33, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 33, 10, 40, 1024, cuvs::distance::DistanceType::InnerProduct},
  // medium dims
  {1000, 10000, 64, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 64, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 128, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 128, 10, 40, 1024, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 128, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 128, 10, 40, 1024, cuvs::distance::DistanceType::L2SqrtExpanded},
  {1000, 10000, 256, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 256, 10, 40, 1024, cuvs::distance::DistanceType::InnerProduct},
  // large dims (may exceed shared memory limits)
  {1000, 10000, 2048, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 2048, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 2049, 16, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 2049, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 2050, 16, 40, 1024, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 2050, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 4096, 20, 50, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 4096, 20, 50, 1024, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 4096, 20, 50, 1024, cuvs::distance::DistanceType::CosineExpanded},

  // ===== k edge cases =====
  {1000, 10000, 16, 1, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 1, 40, 1024, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 16, 1, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 16, 2, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 5, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 20, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 20, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 16, 50, 100, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 100, 200, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 100, 200, 1024, cuvs::distance::DistanceType::InnerProduct},

  // ===== Large k (beyond fused top-k kMaxSqScanCapacity=256, exercises materialized fallback)
  // =====
  // k=257: smallest k that forces the materialized path (Capacity clamped to 0)
  {100, 10000, 32, 257, 100, 64, cuvs::distance::DistanceType::L2Expanded},
  {100, 10000, 32, 257, 100, 64, cuvs::distance::DistanceType::InnerProduct},
  {100, 10000, 32, 257, 100, 64, cuvs::distance::DistanceType::CosineExpanded},
  // k=300: comfortably above the fused top-k threshold
  {100, 10000, 32, 300, 64, 64, cuvs::distance::DistanceType::L2Expanded},
  {100, 10000, 32, 300, 64, 64, cuvs::distance::DistanceType::InnerProduct},

  // ===== nprobe / nlist edge cases =====
  // nprobe == nlist (exhaustive probe)
  {1000, 10000, 16, 10, 64, 64, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 10, 64, 64, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 16, 10, 64, 64, cuvs::distance::DistanceType::CosineExpanded},
  // nprobe == 1 (minimal probe)
  {1000, 10000, 16, 10, 1, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 10, 1, 1024, cuvs::distance::DistanceType::CosineExpanded},
  // nprobe > nlist (clamped to nlist)
  {1000, 10000, 16, 10, 2048, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 10, 2048, 1024, cuvs::distance::DistanceType::CosineExpanded},
  // various nprobe
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 16, 10, 50, 1024, cuvs::distance::DistanceType::L2SqrtExpanded},
  {1000, 10000, 16, 10, 70, 1024, cuvs::distance::DistanceType::L2SqrtExpanded},
  // very small nlist
  {100, 10000, 16, 10, 8, 8, cuvs::distance::DistanceType::L2Expanded},
  {100, 10000, 16, 10, 8, 8, cuvs::distance::DistanceType::CosineExpanded},
  // smaller nlist
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::L2Expanded},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::InnerProduct},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::CosineExpanded},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::L2SqrtExpanded},

  // ===== Dataset size edge cases =====
  // single query
  {1, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  // very few queries
  {2, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {5, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  // very few db vectors (nlist reduced to fit)
  {100, 500, 16, 10, 40, 256, cuvs::distance::DistanceType::L2Expanded},
  {100, 500, 16, 10, 40, 256, cuvs::distance::DistanceType::CosineExpanded},
  // larger datasets
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded},
  {20, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 100000, 16, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {10000, 131072, 8, 10, 20, 1024, cuvs::distance::DistanceType::L2Expanded},
  {10000, 131072, 8, 10, 20, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {10000, 131072, 8, 10, 50, 1024, cuvs::distance::DistanceType::InnerProduct},
  {10000, 131072, 8, 10, 50, 1024, cuvs::distance::DistanceType::L2SqrtExpanded},

  // ===== Large query batches (gridDim.x > 65535) =====
  {100000, 1024, 32, 10, 64, 64, cuvs::distance::DistanceType::L2Expanded},
  {100000, 1024, 32, 10, 64, 64, cuvs::distance::DistanceType::InnerProduct},
  {100000, 1024, 32, 10, 64, 64, cuvs::distance::DistanceType::CosineExpanded},
  {100000, 1024, 32, 10, 64, 64, cuvs::distance::DistanceType::L2SqrtExpanded},
  {100000, 8712, 3, 10, 51, 66, cuvs::distance::DistanceType::L2Expanded},
  {100000, 8712, 3, 10, 51, 66, cuvs::distance::DistanceType::CosineExpanded},
  // just above the old 65535 limit
  {65536, 1024, 16, 10, 32, 64, cuvs::distance::DistanceType::L2Expanded},
  {65536, 1024, 16, 10, 32, 64, cuvs::distance::DistanceType::CosineExpanded},

  // ===== Recall-stability: same data, different query counts =====
  {20000, 8712, 3, 10, 51, 66, cuvs::distance::DistanceType::L2Expanded},
  {50000, 8712, 3, 10, 51, 66, cuvs::distance::DistanceType::L2Expanded},

  // ===== Host dataset: build + extend from host_matrix_view =====
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2SqrtExpanded, true},
  {1000, 10000, 3, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 128, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 128, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {100, 10000, 64, 10, 20, 512, cuvs::distance::DistanceType::InnerProduct, true},
};

const std::vector<AnnIvfSqInputs<int64_t>> inputs_half = {
  // num_queries, num_db_vecs, dim, k, nprobe, nlist, metric, host_dataset

  // All four metrics at a standard dimension
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2SqrtExpanded},

  // Unaligned and small dimensions
  {1000, 10000, 3, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 7, 16, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},

  // Medium / larger dimensions
  {1000, 10000, 64, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 128, 10, 40, 1024, cuvs::distance::DistanceType::InnerProduct},
  {1000, 10000, 128, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded},
  {1000, 10000, 256, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded},

  // k edge cases
  {1000, 10000, 16, 1, 40, 1024, cuvs::distance::DistanceType::L2Expanded},
  {1000, 10000, 16, 50, 100, 1024, cuvs::distance::DistanceType::L2Expanded},

  // nprobe / nlist edge cases
  {1000, 10000, 16, 10, 64, 64, cuvs::distance::DistanceType::L2Expanded},
  {100, 10000, 16, 10, 20, 512, cuvs::distance::DistanceType::CosineExpanded},

  // Host dataset
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 16, 10, 40, 1024, cuvs::distance::DistanceType::CosineExpanded, true},
  {1000, 10000, 128, 10, 40, 1024, cuvs::distance::DistanceType::InnerProduct, true},
};

}  // namespace cuvs::neighbors::ivf_sq
