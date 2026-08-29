/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Iterative CAGRA-Q: building a CAGRA graph directly from a PQ-compressed dataset.
 *
 * This path takes a `device_vpq_dataset_view` instead of dense rows, so the caller owns
 * compression and the inner searches of the iterative build run against the compressed data.
 * It only accepts `L2Expanded`, `pq_bits == 8` and `pq_len` in {2, 4, 8}, so it does not fit
 * the dtype-templated suites in ann_cagra.cuh and lives in its own file.
 *
 * What is checked here is that a compressed dataset, freshly compressed or loaded from disk,
 * builds a usable graph, that such an index survives a trip through a file with its rows, and that
 * the constraints above are rejected rather than accepted and quietly ignored. Fidelity of the
 * dataset payload itself is covered by preprocessing/vpq_serialization.cu.
 */

#include <gtest/gtest.h>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <variant>
#include <vector>

namespace cuvs::neighbors::cagra {

using vpq_dataset_t = cuvs::neighbors::device_vpq_dataset<half, int64_t>;

namespace {

auto compress(const raft::resources& res,
              raft::device_matrix_view<const float, int64_t> dataset,
              uint32_t pq_dim,
              uint32_t pq_bits = 8) -> vpq_dataset_t
{
  cuvs::neighbors::vpq_params params;
  params.pq_dim         = pq_dim;
  params.pq_bits        = pq_bits;
  params.vq_n_centers   = 32;
  params.kmeans_n_iters = 5;  // Codebooks need to be well defined here, not optimal.
  return cuvs::preprocessing::quantize::pq::make_vpq_dataset(res, params, dataset);
}

auto iterative_params(uint32_t graph_degree = 32) -> index_params
{
  index_params params;
  params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree              = graph_degree;
  params.intermediate_graph_degree = graph_degree * 2;
  params.graph_build_params        = graph_build_params::iterative_search_params();
  return params;
}

constexpr int64_t kSearchK = 10;

/** Neighbour ids for `queries`, row-major [n_queries, kSearchK]. */
template <typename IndexT>
auto neighbor_ids(const raft::resources& res,
                  const IndexT& idx,
                  raft::device_matrix_view<const float, int64_t> queries) -> std::vector<uint32_t>
{
  const auto n_queries = queries.extent(0);
  auto neighbors       = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, kSearchK);
  auto distances       = raft::make_device_matrix<float, int64_t>(res, n_queries, kSearchK);

  search_params params;
  params.itopk_size = 64;
  search(res, params, idx, queries, neighbors.view(), distances.view());

  std::vector<uint32_t> ids(static_cast<size_t>(n_queries * kSearchK));
  raft::copy(ids.data(), neighbors.data_handle(), ids.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return ids;
}

/**
 * Fraction of queries that retrieve their own row, where the queries are dataset rows.
 *
 * A sanity signal rather than a quality metric: search quality is a benchmark's job, and an
 * iterative build varies by a few points run to run over identical input, so callers assert a
 * loose floor that only a broken dataset would miss.
 */
template <typename IndexT>
auto self_recall_at_1(const raft::resources& res,
                      const IndexT& idx,
                      raft::device_matrix_view<const float, int64_t> queries) -> double
{
  auto ids             = neighbor_ids(res, idx, queries);
  const auto n_queries = queries.extent(0);
  int64_t hits         = 0;
  for (int64_t q = 0; q < n_queries; q++) {
    if (ids[q * kSearchK] == static_cast<uint32_t>(q)) { hits++; }
  }
  return static_cast<double>(hits) / static_cast<double>(n_queries);
}

}  // namespace

struct CagraQInputs {
  int64_t n_rows;
  int64_t dim;
  uint32_t pq_dim;  // pq_len = dim / pq_dim, which must land in {2, 4, 8}
};

std::ostream& operator<<(std::ostream& os, const CagraQInputs& in)
{
  return os << "n_rows:" << in.n_rows << " dim:" << in.dim << " pq_dim:" << in.pq_dim
            << " pq_len:" << (in.dim / in.pq_dim);
}

/** Shared clustered dataset; each test compresses it itself so nothing leaks between cases. */
class CagraQCompressedTestBase : public ::testing::Test {
 protected:
  void make_dataset(int64_t n_rows, int64_t dim)
  {
    dataset_.emplace(raft::make_device_matrix<float, int64_t>(res_, n_rows, dim));
    auto labels = raft::make_device_vector<int64_t, int64_t>(res_, n_rows);
    raft::random::make_blobs<float, int64_t, raft::row_major>(res_,
                                                             dataset_->view(),
                                                             labels.view(),
                                                             5,             // clusters
                                                             std::nullopt,  // random centers
                                                             std::nullopt,  // scalar std
                                                             1.0F,          // cluster std
                                                             true,          // shuffle
                                                             -10.0F,        // center box min
                                                             10.0F,         // center box max
                                                             1234ULL);
    raft::resource::sync_stream(res_);
  }

  auto dataset() -> raft::device_matrix_view<const float, int64_t>
  {
    return raft::make_const_mdspan(dataset_->view());
  }

  /** The first rows of the dataset, reused as queries. */
  auto queries(int64_t n_queries) -> raft::device_matrix_view<const float, int64_t>
  {
    return raft::make_device_matrix_view<const float, int64_t>(
      dataset_->data_handle(), std::min(n_queries, dataset_->extent(0)), dataset_->extent(1));
  }

  void TearDown() override
  {
    dataset_.reset();
    raft::resource::sync_stream(res_);
  }

  raft::resources res_;
  std::optional<raft::device_matrix<float, int64_t>> dataset_ = std::nullopt;
};

class CagraQBuildTest : public CagraQCompressedTestBase,
                        public ::testing::WithParamInterface<CagraQInputs> {
 protected:
  void SetUp() override
  {
    params_ = GetParam();
    make_dataset(params_.n_rows, params_.dim);
  }

  CagraQInputs params_{};
};

TEST_P(CagraQBuildTest, BuildsAndSearchesAFreshlyCompressedDataset)
{
  auto compressed = compress(res_, dataset(), params_.pq_dim);
  ASSERT_EQ(compressed.pq_len(), static_cast<uint32_t>(params_.dim / params_.pq_dim));

  auto idx = cagra::build(res_, iterative_params(), compressed.as_dataset_view());
  ASSERT_EQ(idx.size(), params_.n_rows);
  ASSERT_EQ(idx.dim(), params_.dim);
  ASSERT_EQ(idx.graph_degree(), 32u);

  // Searchable straight after build: the index keeps the compressed view it was built from, so
  // unlike a dense standard-layout dataset there is nothing to attach first.
  EXPECT_GT(self_recall_at_1(res_, idx, queries(1000)), 0.5);
}

TEST_P(CagraQBuildTest, BuildsFromADeserializedDataset)
{
  auto compressed = compress(res_, dataset(), params_.pq_dim);

  // The path the benchmarks take: compress offline, store, then build from the file.
  std::stringstream stored;
  cuvs::preprocessing::quantize::pq::serialize(res_, stored, compressed);
  std::unique_ptr<vpq_dataset_t> loaded;
  cuvs::preprocessing::quantize::pq::deserialize(res_, stored, &loaded);
  ASSERT_NE(loaded, nullptr);

  auto idx = cagra::build(res_, iterative_params(), loaded->as_dataset_view());
  ASSERT_EQ(idx.size(), params_.n_rows);
  ASSERT_EQ(idx.dim(), params_.dim);
  EXPECT_GT(self_recall_at_1(res_, idx, queries(1000)), 0.5);
}

TEST_P(CagraQBuildTest, PromotesUnsetGraphBuildParamsToIterative)
{
  auto compressed = compress(res_, dataset(), params_.pq_dim);

  // Iterative search is the only construction a compressed dataset supports, so leaving
  // graph_build_params at its default must select it, not fall back to IVF-PQ or NN-descent.
  index_params params;
  params.metric       = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree = 32;
  ASSERT_TRUE(std::holds_alternative<std::monostate>(params.graph_build_params));

  auto idx = cagra::build(res_, params, compressed.as_dataset_view());
  EXPECT_EQ(idx.size(), params_.n_rows);
}

INSTANTIATE_TEST_CASE_P(CagraQBuildTests,
                        CagraQBuildTest,
                        ::testing::ValuesIn(std::vector<CagraQInputs>{
                          {2000, 64, 32},   // pq_len 2
                          {2000, 128, 32},  // pq_len 4
                          {2000, 256, 32},  // pq_len 8
                        }));

/**
 * An index over compressed rows is serialized with those rows, so that a loaded index can be
 * searched without the dense dataset it came from and without retraining the codebooks. The
 * ownership split is the usual one: the file yields an owning dataset, the index only views it.
 */
class CagraQSerializeTest : public CagraQCompressedTestBase {
 protected:
  void SetUp() override { make_dataset(n_rows, dim); }

  static constexpr int64_t n_rows  = 2000;
  static constexpr int64_t dim     = 128;
  static constexpr uint32_t pq_dim = 32;  // pq_len 4
};

TEST_F(CagraQSerializeTest, RoundTripsThroughAFileWithItsDataset)
{
  auto compressed = compress(res_, dataset(), pq_dim);
  auto idx        = cagra::build(res_, iterative_params(), compressed.as_dataset_view());
  auto before     = neighbor_ids(res_, idx, queries(500));

  std::stringstream stored;
  cagra::serialize(res_, stored, idx);

  vpq_f16_index<float> restored{res_};
  std::unique_ptr<vpq_dataset_t> owner;
  cagra::deserialize(res_, stored, &restored, &owner);

  ASSERT_NE(owner, nullptr);
  EXPECT_EQ(owner->n_rows(), compressed.n_rows());
  EXPECT_EQ(owner->dim(), compressed.dim());
  EXPECT_EQ(owner->pq_len(), compressed.pq_len());
  EXPECT_EQ(owner->pq_bits(), compressed.pq_bits());
  EXPECT_EQ(owner->vq_n_centers(), compressed.vq_n_centers());
  EXPECT_EQ(owner->encoded_row_length(), compressed.encoded_row_length());

  ASSERT_EQ(restored.size(), idx.size());
  ASSERT_EQ(restored.dim(), idx.dim());
  ASSERT_EQ(restored.graph_degree(), idx.graph_degree());
  EXPECT_EQ(restored.metric(), idx.metric());

  // Same graph over the same rows, so the results are identical rather than merely comparable.
  auto after = neighbor_ids(res_, restored, queries(500));
  ASSERT_EQ(after.size(), before.size());
  size_t mismatches = 0;
  for (size_t i = 0; i < before.size(); i++) {
    mismatches += static_cast<size_t>(after[i] != before[i]);
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " of " << before.size() << " neighbour ids changed";
}

TEST_F(CagraQSerializeTest, RefusesToLoadWithoutItsDataset)
{
  auto compressed = compress(res_, dataset(), pq_dim);
  auto idx        = cagra::build(res_, iterative_params(), compressed.as_dataset_view());

  std::stringstream stored;
  cagra::serialize(res_, stored, idx);

  // Dropping the rows on load is fine for a dense index, whose caller can attach its own copy, but
  // it would leave a VPQ index unsearchable with no way back: the rows exist nowhere else.
  vpq_f16_index<float> restored{res_};
  EXPECT_THROW(cagra::deserialize(res_, stored, &restored, nullptr), raft::exception);
}

TEST_F(CagraQSerializeTest, SerializesTheGraphAloneWhenAsked)
{
  auto compressed = compress(res_, dataset(), pq_dim);
  auto idx        = cagra::build(res_, iterative_params(), compressed.as_dataset_view());

  std::stringstream stored;
  cagra::serialize(res_, stored, idx, /* include_dataset */ false);

  vpq_f16_index<float> restored{res_};
  std::unique_ptr<vpq_dataset_t> owner;
  cagra::deserialize(res_, stored, &restored, &owner);

  // Nothing to own, and a graph that only update_dataset() can make searchable again.
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(restored.size(), idx.size());
  EXPECT_EQ(restored.graph_degree(), idx.graph_degree());
}

TEST_F(CagraQSerializeTest, RejectsLoadingACompressedIndexAsDense)
{
  auto compressed = compress(res_, dataset(), pq_dim);
  auto idx        = cagra::build(res_, iterative_params(), compressed.as_dataset_view());

  std::stringstream stored;
  cagra::serialize(res_, stored, idx);

  // The dtype prefix says float either way, so it is the recorded dataset kind that has to stop
  // the dense reader from interpreting VPQ codes as rows of floats.
  device_padded_index<float> dense{res_};
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<float, int64_t>> dense_owner;
  EXPECT_THROW(cagra::deserialize(res_, stored, &dense, &dense_owner), raft::exception);
}

/** The constraints the VPQ build overload documents, each of which must be rejected loudly. */
class CagraQContractTest : public CagraQCompressedTestBase {
 protected:
  void SetUp() override { make_dataset(n_rows, dim); }

  static constexpr int64_t n_rows = 1000;
  static constexpr int64_t dim    = 64;
};

TEST_F(CagraQContractTest, RejectsNonIterativeGraphBuilder)
{
  auto compressed = compress(res_, dataset(), 32);
  auto params     = iterative_params();
  params.graph_build_params =
    graph_build_params::nn_descent_params(params.intermediate_graph_degree);
  EXPECT_THROW(cagra::build(res_, params, compressed.as_dataset_view()), raft::exception);
}

TEST_F(CagraQContractTest, RejectsMetricOtherThanL2Expanded)
{
  auto compressed = compress(res_, dataset(), 32);
  auto params     = iterative_params();
  params.metric   = cuvs::distance::DistanceType::InnerProduct;
  EXPECT_THROW(cagra::build(res_, params, compressed.as_dataset_view()), raft::exception);
}

TEST_F(CagraQContractTest, RejectsPqBitsOtherThan8)
{
  auto compressed = compress(res_, dataset(), 32, /* pq_bits */ 6);
  ASSERT_EQ(compressed.pq_bits(), 6u);
  EXPECT_THROW(cagra::build(res_, iterative_params(), compressed.as_dataset_view()),
               raft::exception);
}

TEST_F(CagraQContractTest, RejectsPqLenOutsideSupportedSet)
{
  auto compressed = compress(res_, dataset(), /* pq_dim */ 4);  // pq_len = 64 / 4 = 16
  ASSERT_EQ(compressed.pq_len(), 16u);
  EXPECT_THROW(cagra::build(res_, iterative_params(), compressed.as_dataset_view()),
               raft::exception);
}

TEST_F(CagraQContractTest, RejectsEmptyDataset)
{
  // Hand-built rather than compressed, since make_vpq_dataset rejects an empty input of its own
  // accord. Every other constraint is satisfied so that only the emptiness can trip.
  const auto width  = static_cast<uint32_t>(dim);
  auto vq_code_book = raft::make_device_matrix<half, uint32_t, raft::row_major>(res_, 1, width);
  auto pq_code_book = raft::make_device_matrix<half, uint32_t, raft::row_major>(res_, 256, 2);
  auto codes = raft::make_device_matrix<uint8_t, int64_t, raft::row_major>(res_, 0, 4 + dim / 2);
  vpq_dataset_t empty{std::move(vq_code_book), std::move(pq_code_book), std::move(codes)};
  ASSERT_EQ(empty.n_rows(), 0);

  EXPECT_THROW(cagra::build(res_, iterative_params(), empty.as_dataset_view()), raft::exception);
}

}  // namespace cuvs::neighbors::cagra
