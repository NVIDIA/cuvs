/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "ann_cagra.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/preprocessing/quantize/bbq.hpp>
#include <cuvs_internal/preprocessing/bbq_cpu_quantize.hpp>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra {

struct AnnCagraBbqInputs {
  int n_queries;
  int n_rows;
  int dim;
  int k;
  int graph_degree;
  cuvs::distance::DistanceType metric;
  /** Query-side layout; the code width follows from it. */
  cuvs::preprocessing::quantize::bbq::bbq_code_layout layout;
  /**
   * Document-side layout of a second, coarser quantizer. When set, the dataset carries two
   * quantizers and the asymmetric NN-descent path is used.
   */
  std::optional<cuvs::preprocessing::quantize::bbq::bbq_code_layout> second_layout;
  /** Fraction of the full-precision build's recall that the BBQ build must retain. */
  double min_recall_ratio;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnCagraBbqInputs& p)
{
  os << "{n_queries=" << p.n_queries << ", n_rows=" << p.n_rows << ", dim=" << p.dim
     << ", k=" << p.k << ", graph_degree=" << p.graph_degree << ", metric="
     << cuvs::neighbors::print_metric{static_cast<cuvs::distance::DistanceType>((int)p.metric)}
     << ", layout=" << static_cast<int>(p.layout) << ", second_layout="
     << (p.second_layout.has_value() ? static_cast<int>(*p.second_layout) : -1) << "}";
  return os;
}

class AnnCagraBbqTest : public ::testing::TestWithParam<AnnCagraBbqInputs> {
 public:
  AnnCagraBbqTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnCagraBbqInputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  /** Quantize the float database on the host and upload the codes. */
  auto quantize_database() -> cuvs::neighbors::device_bbq_dataset<float, int64_t>
  {
    std::vector<float> host_data(static_cast<size_t>(ps.n_rows) * ps.dim);
    raft::update_host(host_data.data(), database.data(), host_data.size(), stream_);
    raft::resource::sync_stream(handle_);

    return cuvs_internal::bbq::quantize_to_device(handle_,
                                                  host_data.data(),
                                                  ps.n_rows,
                                                  ps.dim,
                                                  ps.metric,
                                                  ps.layout,
                                                  ps.second_layout.value_or(ps.layout));
  }

  [[nodiscard]] auto default_index_params() const -> cagra::index_params
  {
    cagra::index_params params;
    params.metric                    = ps.metric;
    params.graph_degree              = ps.graph_degree;
    params.intermediate_graph_degree = 2 * ps.graph_degree;
    // The BBQ path always uses nn-descent; pin the dense baseline to it as well so the two builds
    // differ only in the precision of the distances driving graph construction.
    params.graph_build_params =
      cagra::graph_build_params::nn_descent_params(params.intermediate_graph_degree, ps.metric);
    return params;
  }

  /** Brute-force top-k over the current database, as the ground truth for recall. */
  auto naive_neighbours() -> std::vector<uint32_t>
  {
    size_t queries_size = static_cast<size_t>(ps.n_queries) * ps.k;
    std::vector<uint32_t> indices_naive(queries_size);
    rmm::device_uvector<float> distances_naive_dev(queries_size, stream_);
    rmm::device_uvector<uint32_t> indices_naive_dev(queries_size, stream_);
    cuvs::neighbors::naive_knn<float, float, uint32_t>(handle_,
                                                       distances_naive_dev.data(),
                                                       indices_naive_dev.data(),
                                                       search_queries.data(),
                                                       database.data(),
                                                       ps.n_queries,
                                                       ps.n_rows,
                                                       ps.dim,
                                                       ps.k,
                                                       ps.metric);
    raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
    raft::resource::sync_stream(handle_);
    return indices_naive;
  }

  /** Top-k neighbors @p index returns for the test queries. */
  template <typename IndexT>
  auto search_neighbors(IndexT const& index, cagra::search_params search_params = {})
    -> std::vector<uint32_t>
  {
    size_t queries_size = static_cast<size_t>(ps.n_queries) * ps.k;
    rmm::device_uvector<float> distances_dev(queries_size, stream_);
    rmm::device_uvector<uint32_t> indices_dev(queries_size, stream_);

    auto search_queries_view = raft::make_device_matrix_view<const float, int64_t>(
      search_queries.data(), ps.n_queries, ps.dim);
    auto indices_out_view =
      raft::make_device_matrix_view<uint32_t, int64_t>(indices_dev.data(), ps.n_queries, ps.k);
    auto dists_out_view =
      raft::make_device_matrix_view<float, int64_t>(distances_dev.data(), ps.n_queries, ps.k);

    cagra::search(
      handle_, search_params, index, search_queries_view, indices_out_view, dists_out_view);

    std::vector<uint32_t> indices_cagra(queries_size);
    raft::update_host(indices_cagra.data(), indices_dev.data(), queries_size, stream_);
    raft::resource::sync_stream(handle_);
    return indices_cagra;
  }

  template <typename IndexT>
  auto search_recall(IndexT const& index, std::vector<uint32_t> const& ground_truth) -> double
  {
    auto [recall, match_count, total_count] =
      calc_recall(ground_truth, search_neighbors(index), ps.n_queries, ps.k);
    return recall;
  }

  /**
   * The end-to-end contract: a graph built purely from quantized codes navigates the uncompressed
   * vectors nearly as well as one built from the full-precision vectors. Comparing against a dense
   * build on the same data and parameters keeps the bar independent of how much recall the search
   * configuration itself can reach.
   */
  void testSearchRecall()
  {
    // CAGRA search has no L2SqrtExpanded kernels, so this metric is only exercised by the
    // build-side tests below.
    if (ps.metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
      GTEST_SKIP() << "CAGRA search does not support L2SqrtExpanded";
    }

    auto ground_truth = naive_neighbours();
    auto database_view =
      raft::make_device_matrix_view<const float, int64_t>(database.data(), ps.n_rows, ps.dim);
    cuvs::neighbors::test::padded_device_matrix_for_cagra<float> device_padded(handle_,
                                                                               database_view);

    double reference_recall = 0.0;
    {
      auto dense_index = cagra::build(handle_, default_index_params(), device_padded.view);
      dense_index      = cagra::update_dataset(handle_, std::move(dense_index), device_padded.view);
      reference_recall = search_recall(dense_index, ground_truth);
    }

    double bbq_recall = 0.0;
    {
      // `owning_codes` backs the view held by the built index, so it must outlive it.
      auto owning_codes = quantize_database();
      auto graph_index =
        cagra::build(handle_, default_index_params(), owning_codes.as_dataset_view());
      ASSERT_EQ(graph_index.graph_size(), static_cast<uint32_t>(ps.n_rows));

      // Rebinding the uncompressed vectors is what makes the BBQ-built graph searchable.
      auto index = cagra::update_dataset(handle_, std::move(graph_index), device_padded.view);
      bbq_recall = search_recall(index, ground_truth);
    }

    const double min_recall = ps.min_recall_ratio * reference_recall;
    RAFT_LOG_INFO(
      "CAGRA BBQ build (layout=%d, second_layout=%d): recall=%f, dense reference=%f, "
      "retained=%.1f%% (required >= %.1f%%)",
      static_cast<int>(ps.layout),
      ps.second_layout.has_value() ? static_cast<int>(*ps.second_layout) : -1,
      bbq_recall,
      reference_recall,
      100.0 * bbq_recall / reference_recall,
      100.0 * ps.min_recall_ratio);

    // A broken baseline would make the ratio check meaningless.
    ASSERT_GT(reference_recall, 0.8) << "the full-precision CAGRA baseline is unexpectedly poor";
    EXPECT_GE(bbq_recall, min_recall)
      << "recall " << bbq_recall << " retains only " << 100.0 * bbq_recall / reference_recall
      << "% of the full-precision build's " << reference_recall;
  }

  /** The optimized graph has the requested shape and refers only to existing rows. */
  void testGraphShape()
  {
    auto owning_codes = quantize_database();
    auto index = cagra::build(handle_, default_index_params(), owning_codes.as_dataset_view());

    ASSERT_EQ(index.graph_size(), static_cast<uint32_t>(ps.n_rows));
    ASSERT_EQ(index.graph_degree(), static_cast<uint32_t>(ps.graph_degree));
    // The BBQ dataset view is attached by default, so the index reports the quantized shape.
    ASSERT_EQ(index.size(), static_cast<uint32_t>(ps.n_rows));
    ASSERT_EQ(index.dim(), static_cast<uint32_t>(ps.dim));

    auto graph_host = raft::make_host_matrix<uint32_t, int64_t>(ps.n_rows, ps.graph_degree);
    raft::copy(graph_host.data_handle(), index.graph().data_handle(), graph_host.size(), stream_);
    raft::resource::sync_stream(handle_);

    for (int64_t i = 0; i < ps.n_rows; i++) {
      for (int64_t j = 0; j < ps.graph_degree; j++) {
        ASSERT_LT(graph_host(i, j), static_cast<uint32_t>(ps.n_rows))
          << "graph node " << i << " has an out-of-range neighbor at position " << j;
      }
    }
  }

  /**
   * A BBQ index file holds the graph alone: the codes and their quantizers live outside the
   * index, so the restored graph only becomes searchable once a dataset is reattached, and then
   * it answers exactly as the original did.
   */
  void testSerializeRoundTrip()
  {
    if (ps.metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
      GTEST_SKIP() << "CAGRA search does not support L2SqrtExpanded";
    }

    auto database_view =
      raft::make_device_matrix_view<const float, int64_t>(database.data(), ps.n_rows, ps.dim);
    cuvs::neighbors::test::padded_device_matrix_for_cagra<float> device_padded(handle_,
                                                                               database_view);

    auto owning_codes = quantize_database();
    auto graph_index =
      cagra::build(handle_, default_index_params(), owning_codes.as_dataset_view());

    std::stringstream stored;
    cagra::serialize(handle_, stored, graph_index);

    device_bbq_index<float> restored{handle_};
    cagra::deserialize(handle_, stored, &restored);

    ASSERT_EQ(restored.size(), graph_index.size());
    ASSERT_EQ(restored.graph_size(), graph_index.graph_size());
    ASSERT_EQ(restored.graph_degree(), graph_index.graph_degree());
    EXPECT_EQ(restored.metric(), graph_index.metric());
    EXPECT_EQ(restored.dataset().n_rows(), 0);
    EXPECT_TRUE(restored.dataset().quantizers.empty());

    auto original   = cagra::update_dataset(handle_, std::move(graph_index), device_padded.view);
    auto reattached = cagra::update_dataset(handle_, std::move(restored), device_padded.view);
    // Use single-CTA kernel to check for exact search results.
    cagra::search_params search_params;
    search_params.algo = cagra::search_algo::SINGLE_CTA;
    EXPECT_EQ(search_neighbors(reattached, search_params),
              search_neighbors(original, search_params));
  }

  /** `attach_dataset_on_build = false` yields a graph without any dataset binding. */
  void testGraphOnlyBuild()
  {
    auto params                    = default_index_params();
    params.attach_dataset_on_build = false;

    auto owning_codes = quantize_database();
    auto index        = cagra::build(handle_, params, owning_codes.as_dataset_view());

    ASSERT_EQ(index.graph_size(), static_cast<uint32_t>(ps.n_rows));
    ASSERT_EQ(index.graph_degree(), static_cast<uint32_t>(ps.graph_degree));
    EXPECT_EQ(index.dataset().n_rows(), 0);
    EXPECT_TRUE(index.dataset().quantizers.empty());
  }

  /** Only NN-descent graph construction and the four BBQ metrics are accepted. */
  void testUnsupportedParams()
  {
    auto owning_codes = quantize_database();
    auto dataset      = owning_codes.as_dataset_view();

    auto ivf_pq_params               = default_index_params();
    ivf_pq_params.graph_build_params = cagra::graph_build_params::ivf_pq_params(
      raft::matrix_extent<int64_t>(ps.n_rows, ps.dim), ps.metric);
    EXPECT_THROW(cagra::build(handle_, ivf_pq_params, dataset), raft::exception);

    auto iterative_params               = default_index_params();
    iterative_params.graph_build_params = cagra::graph_build_params::iterative_search_params();
    EXPECT_THROW(cagra::build(handle_, iterative_params, dataset), raft::exception);

    auto l1_params   = default_index_params();
    l1_params.metric = cuvs::distance::DistanceType::L1;
    EXPECT_THROW(cagra::build(handle_, l1_params, dataset), raft::exception);
  }

  void SetUp() override
  {
    // nn-descent rejects packed_4b below sm_75.
    if (ps.layout == cuvs::preprocessing::quantize::bbq::bbq_code_layout::packed_4b &&
        cuvs::neighbors::device_compute_capability() < 75) {
      GTEST_SKIP() << "packed_4b requires int4 tensor cores (compute capability 7.5 or newer)";
    }
    database.resize(static_cast<size_t>(ps.n_rows) * ps.dim, stream_);
    search_queries.resize(static_cast<size_t>(ps.n_queries) * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    InitDataset(handle_, database.data(), ps.n_rows, ps.dim, ps.metric, r);
    InitDataset(handle_, search_queries.data(), ps.n_queries, ps.dim, ps.metric, r);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnCagraBbqInputs ps;
  rmm::device_uvector<float> database;
  rmm::device_uvector<float> search_queries;
};

/**
 * Search runs on exact distances, so recall mostly reflects how well a graph built from `bits`-wide
 * codes navigates. Quality degrades gracefully as the codes get coarser, so the fraction of the
 * full-precision build's recall that must be retained is looser for the narrow codes.
 */
inline const std::vector<AnnCagraBbqInputs> bbq_inputs = [] {
  using cuvs::preprocessing::quantize::bbq::bbq_code_layout;
  using opt_layout = std::optional<bbq_code_layout>;
  const std::vector<std::tuple<bbq_code_layout, opt_layout, double>> code_specs{
    {bbq_code_layout::packed_1b, opt_layout{}, 0.85},
    {bbq_code_layout::transposed_2b, opt_layout{}, 0.92},
    {bbq_code_layout::transposed_2b, opt_layout{bbq_code_layout::packed_1b}, 0.92},
    {bbq_code_layout::packed_4b, opt_layout{}, 0.95},
    {bbq_code_layout::packed_4b, opt_layout{bbq_code_layout::packed_1b}, 0.90},
    {bbq_code_layout::transposed_4b, opt_layout{bbq_code_layout::packed_1b}, 0.90},
    {bbq_code_layout::transposed_4b, opt_layout{bbq_code_layout::transposed_2b}, 0.88},
    {bbq_code_layout::packed_7b, opt_layout{}, 0.95},
    {bbq_code_layout::packed_8b, opt_layout{}, 0.95}};

  std::vector<AnnCagraBbqInputs> out;
  for (const auto& [layout, second_layout, min_recall_ratio] : code_specs) {
    const auto batch = raft::util::itertools::product<AnnCagraBbqInputs>(
      {200},   // n_queries
      {4000},  // n_rows
      {128},   // dim
      {10},    // k
      {32},    // graph_degree
      {cuvs::distance::DistanceType::L2Expanded,
       // cuvs::distance::DistanceType::L2SqrtExpanded,
       cuvs::distance::DistanceType::InnerProduct,
       cuvs::distance::DistanceType::CosineExpanded},
      {layout},
      {second_layout},
      {min_recall_ratio});
    out.insert(out.end(), batch.begin(), batch.end());
  }
  return out;
}();

}  // namespace cuvs::neighbors::cagra
