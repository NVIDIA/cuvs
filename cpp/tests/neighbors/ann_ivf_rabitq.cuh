/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include "naive_knn.cuh"
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_rabitq.hpp>

namespace cuvs::neighbors::ivf_rabitq {

struct ivf_rabitq_inputs {
  uint32_t num_db_vecs             = 4096;
  uint32_t num_queries             = 1024;
  uint32_t dim                     = 64;
  uint32_t k                       = 10;
  std::optional<double> min_recall = std::nullopt;

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
  PRINT_DIFF_V(.min_recall, p.min_recall.value_or(0));
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
      raft::random::uniform(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0));
      raft::random::uniform(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(0.1), DataT(2.0));
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
    size_t queries_size = size_t{ps.num_queries} * size_t{ps.k};
    rmm::device_uvector<EvalT> distances_naive_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
    cuvs::neighbors::naive_knn<EvalT, DataT, IdxT>(
      handle_,
      distances_naive_dev.data(),
      indices_naive_dev.data(),
      search_queries.data(),
      database.data(),
      ps.num_queries,
      ps.num_db_vecs,
      ps.dim,
      ps.k,
      static_cast<cuvs::distance::DistanceType>((int)ps.index_params.metric));
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
