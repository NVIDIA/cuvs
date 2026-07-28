/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include "naive_knn.cuh"
#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_rabitq.hpp>
#include <raft/linalg/add.cuh>

#include <thrust/sequence.h>

#include <cmath>
#include <utility>

namespace cuvs::neighbors::ivf_rabitq {

struct ivf_rabitq_inputs {
  uint32_t num_db_vecs             = 4096;
  uint32_t num_queries             = 1024;
  uint32_t dim                     = 64;
  uint32_t k                       = 10;
  uint32_t filter_pass_count       = 0;
  std::optional<double> min_recall = std::nullopt;
  // Generate signed, mean-zero data instead of the default positive-only data. Used by the
  // InnerProduct cases to produce mixed-sign inner products (which exercise both branches of the
  // sign-aware atomic threshold update). Only affects signed DataT (e.g. float).
  bool signed_data = false;

  cuvs::neighbors::ivf_rabitq::index_params index_params;
  cuvs::neighbors::ivf_rabitq::search_params search_params;

  // Set some default parameters for tests
  ivf_rabitq_inputs() { index_params.n_lists = max(32u, min(1024u, num_db_vecs / 128u)); }
};

inline auto operator<<(std::ostream& os, const ivf_rabitq::search_mode& p) -> std::ostream&
{
  switch (p) {
    case ivf_rabitq::search_mode::LUT16: os << "search_mode::LUT16"; break;
    case ivf_rabitq::search_mode::LUT32: os << "search_mode::LUT32"; break;
    case ivf_rabitq::search_mode::QUANT4: os << "search_mode::QUANT4"; break;
    case ivf_rabitq::search_mode::QUANT8: os << "search_mode::QUANT8"; break;
    default: RAFT_FAIL("unreachable code");
  }
  return os;
}

inline auto operator<<(std::ostream& os, const ivf_rabitq_inputs& p) -> std::ostream&
{
  ivf_rabitq_inputs dflt;
  bool need_comma = false;
#define PRINT_DIFF_V(spec, val)       \
  do {                                \
    if (dflt spec != p spec) {        \
      if (need_comma) { os << ", "; } \
      os << #spec << " = " << val;    \
      need_comma = true;              \
    }                                 \
  } while (0)
#define PRINT_DIFF(spec) PRINT_DIFF_V(spec, p spec)

  os << "ivf_rabitq_inputs {";
  PRINT_DIFF(.num_db_vecs);
  PRINT_DIFF(.num_queries);
  PRINT_DIFF(.dim);
  PRINT_DIFF(.k);
  PRINT_DIFF(.filter_pass_count);
  PRINT_DIFF_V(.min_recall, p.min_recall.value_or(0));
  PRINT_DIFF(.signed_data);
  PRINT_DIFF_V(.index_params.metric, static_cast<int>(p.index_params.metric));
  PRINT_DIFF(.index_params.n_lists);
  PRINT_DIFF(.index_params.bits_per_dim);
  PRINT_DIFF(.index_params.kmeans_n_iters);
  PRINT_DIFF(.index_params.fast_quantize_flag);
  PRINT_DIFF(.search_params.n_probes);
  PRINT_DIFF(.search_params.mode);
  os << "}";
  return os;
}

template <typename EvalT, typename DataT, typename IdxT>
class ivf_rabitq_test : public ::testing::TestWithParam<ivf_rabitq_inputs> {
 public:
  ivf_rabitq_test()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<ivf_rabitq_inputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

  void gen_data()
  {
    database.resize(size_t{ps.num_db_vecs} * size_t{ps.dim}, stream_);
    search_queries.resize(size_t{ps.num_queries} * size_t{ps.dim}, stream_);

    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      // Signed data (used only by InnerProduct cases) is mean-zero over [-2, 2]; the default is
      // positive-only over [0.1, 2] to match the other ANN harnesses and keep existing recall
      // thresholds valid.
      const DataT lo = (ps.signed_data && std::is_signed_v<DataT>) ? DataT(-2.0) : DataT(0.1);
      raft::random::uniform(handle_, r, database.data(), ps.num_db_vecs * ps.dim, lo, DataT(2.0));
      raft::random::uniform(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, lo, DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(1), DataT(20));
      raft::random::uniformInt(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(1), DataT(20));
    }
    raft::resource::sync_stream(handle_);
  }

  void calc_ref()
  {
    const IdxT filter_offset = static_cast<IdxT>(ps.num_db_vecs - ps.filter_pass_count);
    const bool filtered      = ps.filter_pass_count > 0;
    const IdxT ref_offset    = filtered ? filter_offset : IdxT{0};
    const uint32_t ref_size  = filtered ? ps.filter_pass_count : ps.num_db_vecs;
    if (ref_size < ps.k) {
      indices_ref.clear();
      distances_ref.clear();
      return;
    }

    size_t queries_size = size_t{ps.num_queries} * size_t{ps.k};
    rmm::device_uvector<EvalT> distances_naive_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
    cuvs::neighbors::naive_knn<EvalT, DataT, IdxT>(
      handle_,
      distances_naive_dev.data(),
      indices_naive_dev.data(),
      search_queries.data(),
      database.data() + static_cast<size_t>(ref_offset) * ps.dim,
      ps.num_queries,
      ref_size,
      ps.dim,
      ps.k,
      static_cast<cuvs::distance::DistanceType>((int)ps.index_params.metric));
    if (filtered) {
      raft::linalg::addScalar(
        indices_naive_dev.data(), indices_naive_dev.data(), ref_offset, queries_size, stream_);
    }
    distances_ref.resize(queries_size);
    raft::update_host(distances_ref.data(), distances_naive_dev.data(), queries_size, stream_);
    indices_ref.resize(queries_size);
    raft::update_host(indices_ref.data(), indices_naive_dev.data(), queries_size, stream_);
    raft::resource::sync_stream(handle_);
  }

  auto build_only()
  {
    auto ipams = ps.index_params;

    auto database_view =
      raft::make_device_matrix_view<const DataT, IdxT>(database.data(), ps.num_db_vecs, ps.dim);
    return cuvs::neighbors::ivf_rabitq::build(handle_, ipams, database_view);
  }

  auto build_only_host_input()
  {
    auto ipams = ps.index_params;

    auto host_database = raft::make_host_matrix<DataT, IdxT>(ps.num_db_vecs, ps.dim);
    raft::copy(host_database.data_handle(), database.data(), ps.num_db_vecs * ps.dim, stream_);
    auto database_view = raft::make_host_matrix_view<const DataT, IdxT>(
      host_database.data_handle(), ps.num_db_vecs, ps.dim);
    return cuvs::neighbors::ivf_rabitq::build(handle_, ipams, database_view);
  }

  auto build_serialize()
  {
    tmp_index_file index_file;
    auto idx_to_serialize = build_only();
    cuvs::neighbors::ivf_rabitq::serialize(handle_, index_file.filename, idx_to_serialize);
    cuvs::neighbors::ivf_rabitq::index<IdxT> deserialized_index(handle_);
    cuvs::neighbors::ivf_rabitq::deserialize(handle_, index_file.filename, &deserialized_index);
    return deserialized_index;
  }

  auto build_host_input_serialize()
  {
    tmp_index_file index_file;
    auto idx_to_serialize = build_only_host_input();
    cuvs::neighbors::ivf_rabitq::serialize(handle_, index_file.filename, idx_to_serialize);
    cuvs::neighbors::ivf_rabitq::index<IdxT> deserialized_index(handle_);
    cuvs::neighbors::ivf_rabitq::deserialize(handle_, index_file.filename, &deserialized_index);
    return deserialized_index;
  }

  auto build_with_forced_streaming()
  {
    tmp_index_file index_file;
    auto ipams = ps.index_params;
    // Force streaming construction even if dataset fits in GPU memory
    ipams.force_streaming = true;
    // Use batch size that ensures at least 2-3 batches for typical test sizes
    // but scales reasonably for larger datasets
    ipams.streaming_batch_size = std::max(size_t{1000}, size_t{ps.num_db_vecs / 3});

    auto host_database = raft::make_host_matrix<DataT, IdxT>(ps.num_db_vecs, ps.dim);
    raft::copy(host_database.data_handle(), database.data(), ps.num_db_vecs * ps.dim, stream_);
    auto database_view = raft::make_host_matrix_view<const DataT, IdxT>(
      host_database.data_handle(), ps.num_db_vecs, ps.dim);
    auto idx_to_serialize = cuvs::neighbors::ivf_rabitq::build(handle_, ipams, database_view);

    // Serialize and deserialize to reorganize data for efficient search
    cuvs::neighbors::ivf_rabitq::serialize(handle_, index_file.filename, idx_to_serialize);
    cuvs::neighbors::ivf_rabitq::index<IdxT> deserialized_index(handle_);
    cuvs::neighbors::ivf_rabitq::deserialize(handle_, index_file.filename, &deserialized_index);
    return deserialized_index;
  }

  template <typename BuildIndex>
  void run(BuildIndex build_index)
  {
    index<IdxT> index = build_index();

    double compression_ratio = sizeof(DataT) * 8 / ps.index_params.bits_per_dim;

    size_t queries_size = ps.num_queries * ps.k;
    std::vector<IdxT> indices_ivf_rabitq(queries_size);
    std::vector<EvalT> distances_ivf_rabitq(queries_size);

    rmm::device_uvector<EvalT> distances_ivf_rabitq_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_ivf_rabitq_dev(queries_size, stream_);

    auto query_view =
      raft::make_device_matrix_view<DataT, uint32_t>(search_queries.data(), ps.num_queries, ps.dim);
    auto inds_view = raft::make_device_matrix_view<IdxT, uint32_t>(
      indices_ivf_rabitq_dev.data(), ps.num_queries, ps.k);
    auto dists_view = raft::make_device_matrix_view<EvalT, uint32_t>(
      distances_ivf_rabitq_dev.data(), ps.num_queries, ps.k);

    cuvs::neighbors::ivf_rabitq::search(
      handle_, ps.search_params, index, query_view, inds_view, dists_view);

    raft::update_host(
      distances_ivf_rabitq.data(), distances_ivf_rabitq_dev.data(), queries_size, stream_);
    raft::update_host(
      indices_ivf_rabitq.data(), indices_ivf_rabitq_dev.data(), queries_size, stream_);
    raft::resource::sync_stream(handle_);

    // A very conservative lower bound on recall
    double min_recall = 0.5;
    // Use explicit per-test min recall value if provided.
    min_recall = ps.min_recall.value_or(min_recall);

    ASSERT_TRUE(cuvs::neighbors::eval_neighbours(indices_ref,
                                                 indices_ivf_rabitq,
                                                 distances_ref,
                                                 distances_ivf_rabitq,
                                                 ps.num_queries,
                                                 ps.k,
                                                 0.0001 * compression_ratio,
                                                 min_recall))
      << ps;
  }

  void run_filter()
  {
    ASSERT_GT(ps.filter_pass_count, 0);
    ASSERT_LE(ps.filter_pass_count, ps.num_db_vecs);
    const IdxT filter_offset = static_cast<IdxT>(ps.num_db_vecs - ps.filter_pass_count);

    index<IdxT> index         = build_serialize();
    const size_t queries_size = size_t{ps.num_queries} * ps.k;
    auto distances = raft::make_device_matrix<EvalT, IdxT>(handle_, ps.num_queries, ps.k);
    auto indices   = raft::make_device_matrix<IdxT, IdxT>(handle_, ps.num_queries, ps.k);

    auto removed_indices = raft::make_device_vector<IdxT, int64_t>(handle_, filter_offset);
    thrust::sequence(raft::resource::get_thrust_policy(handle_),
                     thrust::device_pointer_cast(removed_indices.data_handle()),
                     thrust::device_pointer_cast(removed_indices.data_handle() + filter_offset));
    cuvs::core::bitset<uint32_t, IdxT> bitset(handle_, removed_indices.view(), ps.num_db_vecs);
    auto filter = cuvs::neighbors::filtering::bitset_filter(bitset.view());

    auto queries = raft::make_device_matrix_view<const DataT, IdxT>(
      search_queries.data(), ps.num_queries, ps.dim);
    cuvs::neighbors::ivf_rabitq::search(
      handle_, ps.search_params, index, queries, indices.view(), distances.view(), filter);

    std::vector<IdxT> host_indices(queries_size);
    std::vector<EvalT> host_distances(queries_size);
    raft::update_host(host_indices.data(), indices.data_handle(), queries_size, stream_);
    raft::update_host(host_distances.data(), distances.data_handle(), queries_size, stream_);
    raft::resource::sync_stream(handle_);

    const uint32_t expected_valid = std::min(ps.filter_pass_count, ps.k);
    for (uint32_t query_ix = 0; query_ix < ps.num_queries; ++query_ix) {
      uint32_t valid_count = 0;
      uint32_t oob_count   = 0;
      std::vector<bool> seen(ps.filter_pass_count, false);
      for (uint32_t neighbor_ix = 0; neighbor_ix < ps.k; ++neighbor_ix) {
        const size_t flat_ix = size_t{query_ix} * ps.k + neighbor_ix;
        const IdxT id        = host_indices[flat_ix];
        if (id == kOutOfBoundsRecord<IdxT>) {
          ++oob_count;
          ASSERT_TRUE(std::isinf(host_distances[flat_ix])) << ps;
          continue;
        }
        ASSERT_GE(id, filter_offset) << ps;
        ASSERT_LT(id, static_cast<IdxT>(ps.num_db_vecs)) << ps;
        const size_t allowed_ix = static_cast<size_t>(id - filter_offset);
        ASSERT_FALSE(seen[allowed_ix]) << ps;
        seen[allowed_ix] = true;
        ++valid_count;
      }
      ASSERT_EQ(valid_count, expected_valid) << ps;
      ASSERT_EQ(oob_count, ps.k - expected_valid) << ps;
    }

    if (ps.filter_pass_count < ps.k) {
      RAFT_LOG_INFO("Filter validation = %u valid, %u out-of-bounds results per query.",
                    expected_valid,
                    ps.k - expected_valid);
      return;
    }

    double compression_ratio = sizeof(DataT) * 8 / ps.index_params.bits_per_dim;
    const double min_recall  = ps.min_recall.value_or(0.5);
    ASSERT_TRUE(cuvs::neighbors::eval_neighbours(indices_ref,
                                                 host_indices,
                                                 distances_ref,
                                                 host_distances,
                                                 ps.num_queries,
                                                 ps.k,
                                                 0.0001 * compression_ratio,
                                                 min_recall))
      << ps;
  }

  void SetUp() override  // NOLINT
  {
    gen_data();
    calc_ref();
  }

  void TearDown() override  // NOLINT
  {
    cudaGetLastError();
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  ivf_rabitq_inputs ps;                       // NOLINT
  rmm::device_uvector<DataT> database;        // NOLINT
  rmm::device_uvector<DataT> search_queries;  // NOLINT
  std::vector<IdxT> indices_ref;              // NOLINT
  std::vector<EvalT> distances_ref;           // NOLINT
};

template <typename EvalT, typename DataT, typename IdxT>
class ivf_rabitq_filter_test : public ivf_rabitq_test<EvalT, DataT, IdxT> {};

/* Test cases */
using test_cases_t = std::vector<ivf_rabitq_inputs>;

// concatenate parameter sets for different type
template <typename T>
auto operator+(const std::vector<T>& a, const std::vector<T>& b) -> std::vector<T>
{
  std::vector<T> res = a;
  res.insert(res.end(), b.begin(), b.end());
  return res;
}

inline auto defaults() -> test_cases_t { return {ivf_rabitq_inputs{}}; }

template <typename B, typename A, typename F>
auto map(const std::vector<A>& xs, F f) -> std::vector<B>
{
  std::vector<B> ys(xs.size());
  std::transform(xs.begin(), xs.end(), ys.begin(), f);
  return ys;
}

inline auto with_dims(const std::vector<uint32_t>& dims) -> test_cases_t
{
  return map<ivf_rabitq_inputs>(dims, [](uint32_t d) {
    ivf_rabitq_inputs x;
    x.dim = d;
    return x;
  });
}

inline auto small_dims() -> test_cases_t { return with_dims({1, 2, 3, 4, 5, 6, 7, 8}); }

inline auto big_dims() -> test_cases_t
{
  return with_dims({512, 513, 1023, 1024, 1025, 2048, 2049, 2050});
}

inline auto var_n_probes() -> test_cases_t
{
  ivf_rabitq_inputs dflt;
  std::vector<uint32_t> xs;
  for (auto x = dflt.index_params.n_lists; x >= 1; x /= 2) {
    xs.push_back(x);
  }
  return map<ivf_rabitq_inputs>(xs, [](uint32_t n_probes) {
    ivf_rabitq_inputs x;
    x.search_params.n_probes = n_probes;
    // reduce `min_recall` for low `n_probes`
    if (n_probes <= 5) { x.min_recall = 0.08 * n_probes; }
    return x;
  });
}

inline auto var_k() -> test_cases_t
{
  return map<ivf_rabitq_inputs, uint32_t>({1, 5, 10, 32, 64, 16384}, [](uint32_t k) {
    ivf_rabitq_inputs x;
    x.k           = k;
    x.num_db_vecs = max(x.num_db_vecs, k * 2);
    if (k > 64) x.num_queries = 64;  // reduce runtime of large-k tests
    return x;
  });
}

inline auto var_bits_per_dim() -> test_cases_t
{
  ivf_rabitq_inputs dflt;
  std::vector<uint32_t> xs;
  for (auto x = 1; x <= 9; ++x) {
    xs.push_back(x);
  }
  return map<ivf_rabitq_inputs>(xs, [](uint32_t bits_per_dim) {
    ivf_rabitq_inputs x;
    x.index_params.bits_per_dim = bits_per_dim;
    if (bits_per_dim == 1) { x.min_recall = 0.3; }
    return x;
  });
}

inline auto filter_cases() -> test_cases_t
{
  test_cases_t cases;
  const std::vector<search_mode> modes{
    search_mode::LUT16, search_mode::LUT32, search_mode::QUANT4, search_mode::QUANT8};
  const std::vector<cuvs::distance::DistanceType> metrics{
    cuvs::distance::DistanceType::L2Expanded, cuvs::distance::DistanceType::InnerProduct};
  // Keep the existing 70.7%-pass coverage and add an exact 1%-pass dataset.
  // For the latter, k=65 exercises the non-block path without the largely
  // forced overlap produced by selecting 96 of only 100 eligible vectors.
  for (auto [num_db_vecs, filter_pass_count] : {std::pair{1024u, 724u}, std::pair{10000u, 100u}}) {
    const uint32_t non_block_k = filter_pass_count == 100 ? 65u : 96u;
    for (auto mode : modes) {
      for (uint32_t bits_per_dim : {1u, 3u}) {
        for (uint32_t k : {10u, non_block_k}) {
          for (auto metric : metrics) {
            ivf_rabitq_inputs x;
            x.num_db_vecs               = num_db_vecs;
            x.num_queries               = 4;
            x.k                         = k;
            x.filter_pass_count         = filter_pass_count;
            x.index_params.n_lists      = 16;
            x.index_params.bits_per_dim = bits_per_dim;
            x.index_params.metric       = metric;
            x.signed_data               = metric == cuvs::distance::DistanceType::InnerProduct;
            // Large-k L2 searches should recover nearly all eligible neighbors. At small k,
            // account for coarser 1-bit quantization while requiring higher recall when only 1%
            // of the database can pass the filter. InnerProduct uses the conservative bounds from
            // the existing metric sweep until filtered recall measurements are available.
            if (metric == cuvs::distance::DistanceType::InnerProduct) {
              x.min_recall = bits_per_dim == 1 ? 0.30 : 0.50;
            } else if (k > 64) {
              x.min_recall = 0.90;
            } else if (filter_pass_count == 100) {
              x.min_recall = bits_per_dim == 1 ? 0.50 : 0.85;
            } else {
              x.min_recall = bits_per_dim == 1 ? 0.45 : 0.70;
            }
            x.search_params.n_probes = 16;
            x.search_params.mode     = mode;
            cases.push_back(x);
          }
        }
      }
    }
  }
  // Exercise under-filled output for both block-sort and non-block-sort search.
  for (auto mode : modes) {
    for (uint32_t bits_per_dim : {1u, 3u}) {
      for (uint32_t k : {10u, 65u}) {
        for (auto metric : metrics) {
          ivf_rabitq_inputs x;
          x.num_db_vecs               = 1024;
          x.num_queries               = 4;
          x.k                         = k;
          x.filter_pass_count         = 5;
          x.index_params.n_lists      = 16;
          x.index_params.bits_per_dim = bits_per_dim;
          x.index_params.metric       = metric;
          x.signed_data               = metric == cuvs::distance::DistanceType::InnerProduct;
          x.search_params.n_probes    = 16;
          x.search_params.mode        = mode;
          cases.push_back(x);
        }
      }
    }
  }
  return cases;
}

inline auto var_search_mode() -> test_cases_t
{
  ivf_rabitq_inputs dflt;
  std::vector<cuvs::neighbors::ivf_rabitq::search_mode> xs{ivf_rabitq::search_mode::LUT16,
                                                           ivf_rabitq::search_mode::LUT32,
                                                           ivf_rabitq::search_mode::QUANT4,
                                                           ivf_rabitq::search_mode::QUANT8};

  return map<ivf_rabitq_inputs>(xs, [](cuvs::neighbors::ivf_rabitq::search_mode mode) {
    ivf_rabitq_inputs x;
    x.search_params.mode = mode;
    return x;
  });
}

inline auto var_search_mode_1_bit() -> test_cases_t
{
  ivf_rabitq_inputs dflt;
  std::vector<cuvs::neighbors::ivf_rabitq::search_mode> xs{ivf_rabitq::search_mode::LUT16,
                                                           ivf_rabitq::search_mode::LUT32,
                                                           ivf_rabitq::search_mode::QUANT4,
                                                           ivf_rabitq::search_mode::QUANT8};

  return map<ivf_rabitq_inputs>(xs, [](cuvs::neighbors::ivf_rabitq::search_mode mode) {
    ivf_rabitq_inputs x;
    x.search_params.mode        = mode;
    x.index_params.bits_per_dim = 1;
    x.min_recall                = 0.3;
    return x;
  });
}

// InnerProduct metric across all four search modes. Cover both the block-sort
// (k <= kMaxTopKBlockSort == 64) and non-block-sort (k > 64) sub-paths, and both the no-ex
// (bits_per_dim == 1) and with-ex (bits_per_dim > 1) code paths.
inline auto var_metric() -> test_cases_t
{
  test_cases_t xs;
  for (auto mode : {ivf_rabitq::search_mode::LUT16,
                    ivf_rabitq::search_mode::LUT32,
                    ivf_rabitq::search_mode::QUANT4,
                    ivf_rabitq::search_mode::QUANT8}) {
    for (uint32_t k : {uint32_t{10}, uint32_t{128}}) {
      for (uint32_t bits : {uint32_t{1}, uint32_t{4}}) {
        ivf_rabitq_inputs x;
        x.index_params.metric       = cuvs::distance::DistanceType::InnerProduct;
        x.search_params.mode        = mode;
        x.k                         = k;
        x.index_params.bits_per_dim = bits;
        // Mean-zero data so inner products take both signs.
        x.signed_data = true;
        // 1-bit quantization is coarse; relax the recall bound as the other 1-bit tests do.
        if (bits == 1) { x.min_recall = 0.3; }
        xs.push_back(x);
      }
    }
  }
  return xs;
}

/* Test instantiations */

// Currently IVF-RaBitQ deserialization reorganizes data for efficient search and is required for
// producing correct results.

#define TEST_BUILD_SERIALIZE_SEARCH(type)                    \
  TEST_P(type, build_serialize_search) /* NOLINT */          \
  {                                                          \
    this->run([this]() { return this->build_serialize(); }); \
  }

#define TEST_BUILD_HOST_INPUT_SERIALIZE_SEARCH(type)                    \
  TEST_P(type, build_host_input_serialize_search) /* NOLINT */          \
  {                                                                     \
    this->run([this]() { return this->build_host_input_serialize(); }); \
  }

#define TEST_BUILD_FORCED_STREAMING(type)                                \
  TEST_P(type, build_forced_streaming) /* NOLINT */                      \
  {                                                                      \
    this->run([this]() { return this->build_with_forced_streaming(); }); \
  }

#define INSTANTIATE(type, vals) \
  INSTANTIATE_TEST_SUITE_P(IvfRabitq, type, ::testing::ValuesIn(vals)); /* NOLINT */

}  // namespace cuvs::neighbors::ivf_rabitq
