/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Serializing a CAGRA index whose dataset is PQ-compressed (CAGRA-Q).
 *
 * Such an index cannot be saved by the dtype-templated suites in ann_cagra.cuh: its rows are VPQ
 * codes rather than values of `DataT`, it only searches with `L2Expanded`, `pq_bits == 8` and
 * `pq_len` in {2, 4, 8}, and it is assembled rather than built, since `cagra::build` produces dense
 * indices only. The assembly here is the usual one: a graph from a dense build, a dataset
 * compressed separately, and an index that views both.
 *
 * What is checked is that the compressed rows travel with the index, so a loaded index searches on
 * its own without the dense dataset it came from and without retraining codebooks, and that the
 * cases where they cannot travel fail loudly. Fidelity of the dataset payload itself is covered by
 * preprocessing/vpq_serialization.cu.
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
#include <vector>

namespace cuvs::neighbors::cagra {

using vpq_dataset_t = cuvs::neighbors::device_vpq_dataset<half, int64_t>;

namespace {

constexpr int64_t kSearchK      = 10;
constexpr uint32_t kGraphDegree = 32;

auto compress(const raft::resources& res,
              raft::device_matrix_view<const float, int64_t> dataset,
              uint32_t pq_dim) -> vpq_dataset_t
{
  cuvs::neighbors::vpq_params params;
  params.pq_dim         = pq_dim;
  params.pq_bits        = 8;
  params.vq_n_centers   = 32;
  params.kmeans_n_iters = 5;  // Codebooks need to be well defined here, not optimal.
  return cuvs::preprocessing::quantize::pq::make_vpq_dataset(res, params, dataset);
}

/** A dense index built over the same rows, kept alive only to lend its graph. */
auto build_graph_source(const raft::resources& res,
                        raft::device_matrix_view<const float, int64_t> dataset)
  -> device_standard_index<float, uint32_t>
{
  index_params params;
  params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree              = kGraphDegree;
  params.intermediate_graph_degree = kGraphDegree * 2;
  return cagra::build(res, params, cuvs::neighbors::make_device_standard_dataset_view(dataset));
}

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
 * A sanity signal rather than a quality metric: it is here so that comparing neighbour ids before
 * and after a round trip compares useful answers rather than two copies of the same nonsense.
 */
auto self_recall_at_1(const std::vector<uint32_t>& ids) -> double
{
  const size_t n_queries = ids.size() / kSearchK;
  size_t hits            = 0;
  for (size_t q = 0; q < n_queries; q++) {
    hits += static_cast<size_t>(ids[q * kSearchK] == static_cast<uint32_t>(q));
  }
  return static_cast<double>(hits) / static_cast<double>(n_queries);
}

}  // namespace

/**
 * An index over compressed rows is serialized with those rows. The ownership split is the usual
 * one: the file yields an owning dataset, the index only views it.
 */
class CagraVpqSerializeTest : public ::testing::Test {
 protected:
  void SetUp() override
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

  void TearDown() override
  {
    dataset_.reset();
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

  static constexpr int64_t n_rows  = 2000;
  static constexpr int64_t dim     = 128;
  static constexpr uint32_t pq_dim = 32;  // pq_len 4

  raft::resources res_;
  std::optional<raft::device_matrix<float, int64_t>> dataset_ = std::nullopt;
};

TEST_F(CagraVpqSerializeTest, RoundTripsThroughAFileWithItsDataset)
{
  auto compressed   = compress(res_, dataset(), pq_dim);
  auto graph_source = build_graph_source(res_, dataset());
  vpq_f16_index<float> idx{res_,
                           cuvs::distance::DistanceType::L2Expanded,
                           compressed.as_dataset_view(),
                           graph_source.graph()};

  auto before = neighbor_ids(res_, idx, queries(500));
  ASSERT_GT(self_recall_at_1(before), 0.5);

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

TEST_F(CagraVpqSerializeTest, RefusesToLoadWithoutItsDataset)
{
  auto compressed   = compress(res_, dataset(), pq_dim);
  auto graph_source = build_graph_source(res_, dataset());
  vpq_f16_index<float> idx{res_,
                           cuvs::distance::DistanceType::L2Expanded,
                           compressed.as_dataset_view(),
                           graph_source.graph()};

  std::stringstream stored;
  cagra::serialize(res_, stored, idx);

  // Dropping the rows on load is fine for a dense index, whose caller can attach its own copy, but
  // it would leave a VPQ index unsearchable with no way back: the rows exist nowhere else.
  vpq_f16_index<float> restored{res_};
  EXPECT_THROW(cagra::deserialize(res_, stored, &restored, nullptr), raft::exception);
}

TEST_F(CagraVpqSerializeTest, SerializesTheGraphAloneWhenAsked)
{
  auto compressed   = compress(res_, dataset(), pq_dim);
  auto graph_source = build_graph_source(res_, dataset());
  vpq_f16_index<float> idx{res_,
                           cuvs::distance::DistanceType::L2Expanded,
                           compressed.as_dataset_view(),
                           graph_source.graph()};

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

TEST_F(CagraVpqSerializeTest, RejectsLoadingACompressedIndexAsDense)
{
  auto compressed   = compress(res_, dataset(), pq_dim);
  auto graph_source = build_graph_source(res_, dataset());
  vpq_f16_index<float> idx{res_,
                           cuvs::distance::DistanceType::L2Expanded,
                           compressed.as_dataset_view(),
                           graph_source.graph()};

  std::stringstream stored;
  cagra::serialize(res_, stored, idx);

  // The dtype prefix says float either way, so it is the recorded dataset kind that has to stop the
  // dense reader from interpreting VPQ codes as rows of floats.
  device_padded_index<float> dense{res_};
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<float, int64_t>> dense_owner;
  EXPECT_THROW(cagra::deserialize(res_, stored, &dense, &dense_owner), raft::exception);
}

}  // namespace cuvs::neighbors::cagra
