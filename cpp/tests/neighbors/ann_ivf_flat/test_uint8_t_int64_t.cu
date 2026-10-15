/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <raft/core/serialize.hpp>

#include <algorithm>
#include <random>
#include <sstream>

#include "../ann_ivf_flat.cuh"

namespace cuvs::neighbors::ivf_flat {

CUVS_METRIC(binary_hamming, { acc += __popc(static_cast<unsigned>(x.raw() ^ y.raw())); })

typedef AnnIVFFlatTest<float, uint8_t, int64_t> AnnIVFFlatTestF_uint8;
TEST_P(AnnIVFFlatTestF_uint8, AnnIVFFlat)
{
  this->testIVFFlat();
  this->testPacker();
  this->testFilter();
}

INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest, AnnIVFFlatTestF_uint8, ::testing::ValuesIn(inputs));

// Check exact distances rather than tied IDs: Hamming's kth neighbor often has many ties.
TEST(BinaryIvfFlatRegression, ExhaustiveScalarAndVectorizedScan)
{
  raft::resources handle;
  auto stream                 = raft::resource::get_cuda_stream(handle);
  constexpr int64_t n_rows    = 1536;
  constexpr int64_t n_queries = 3;
  std::mt19937 rng(42);
  for (int64_t dim : {1, 3, 16, 192}) {
    SCOPED_TRACE(dim);
    std::vector<uint8_t> data(n_rows * dim);
    std::vector<uint8_t> queries(n_queries * dim);
    for (auto& x : data) {
      x = static_cast<uint8_t>(rng());
    }
    for (auto& x : queries) {
      x = static_cast<uint8_t>(rng());
    }
    auto data_dev    = raft::make_device_matrix<uint8_t, int64_t>(handle, n_rows, dim);
    auto queries_dev = raft::make_device_matrix<uint8_t, int64_t>(handle, n_queries, dim);
    raft::update_device(data_dev.data_handle(), data.data(), data.size(), stream);
    raft::update_device(queries_dev.data_handle(), queries.data(), queries.size(), stream);
    index_params params;
    params.metric                   = cuvs::distance::DistanceType::BitwiseHamming;
    params.n_lists                  = 16;
    params.kmeans_n_iters           = 3;
    params.kmeans_trainset_fraction = 1.0;
    auto idx = build(handle, params, raft::make_const_mdspan(data_dev.view()));
    ASSERT_EQ(idx.dim(), dim);
    ASSERT_EQ(idx.size(), n_rows);
    ASSERT_TRUE(idx.binary_index());
    ASSERT_EQ(idx.centers().size(), 0);

    // Verify packed-center persistence independently of the search itself.
    std::stringstream bytes;
    serialize(handle, bytes, idx);
    index<uint8_t, int64_t> restored(handle);
    deserialize(handle, bytes, &restored);
    ASSERT_TRUE(cuvs::devArrMatch(idx.binary_centers().data_handle(),
                                  restored.binary_centers().data_handle(),
                                  idx.binary_centers().size(),
                                  cuvs::Compare<uint8_t>(),
                                  stream.get()));
    index<uint8_t, int64_t> moved(std::move(restored));
    restored = std::move(moved);
    search_params search_params;
    search_params.n_probes = params.n_lists;
    for (int64_t k : {10, 100, 1025, 11, 1026}) {
      SCOPED_TRACE(k);
      search_params.metric_udf =
        (k == 11 || k == 1026) ? std::make_optional(binary_hamming_udf()) : std::nullopt;
      auto ids_dev       = raft::make_device_matrix<int64_t, int64_t>(handle, n_queries, k);
      auto distances_dev = raft::make_device_matrix<float, int64_t>(handle, n_queries, k);
      search(handle,
             search_params,
             restored,
             raft::make_const_mdspan(queries_dev.view()),
             ids_dev.view(),
             distances_dev.view());
      std::vector<int64_t> ids(n_queries * k);
      std::vector<float> distances(n_queries * k);
      raft::update_host(ids.data(), ids_dev.data_handle(), ids.size(), stream);
      raft::update_host(distances.data(), distances_dev.data_handle(), distances.size(), stream);
      raft::resource::sync_stream(handle);
      for (int64_t q = 0; q < n_queries; ++q) {
        std::vector<int> expected(n_rows);
        for (int64_t row = 0; row < n_rows; ++row) {
          for (int64_t col = 0; col < dim; ++col) {
            expected[row] +=
              __builtin_popcount(unsigned(data[row * dim + col] ^ queries[q * dim + col]));
          }
        }
        auto sorted = expected;
        std::sort(sorted.begin(), sorted.end());
        std::vector<bool> seen(n_rows);
        auto actual =
          std::vector<float>(distances.begin() + q * k, distances.begin() + (q + 1) * k);
        std::sort(actual.begin(), actual.end());
        for (int64_t j = 0; j < k; ++j) {
          auto id = ids[q * k + j];
          ASSERT_GE(id, 0);
          ASSERT_LT(id, n_rows);
          ASSERT_FALSE(seen[id]);
          seen[id] = true;
          ASSERT_EQ(distances[q * k + j], expected[id]);
          ASSERT_EQ(actual[j], sorted[j]);
        }
      }
    }
  }
}

TEST(BinaryIvfFlatRegression, AdaptiveCountsSurviveCloneAndSerialization)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  const std::vector<uint8_t> initial{1, 1, 1, 0};
  auto data = raft::make_device_matrix<uint8_t, int64_t>(handle, 4, 1);
  raft::update_device(data.data_handle(), initial.data(), initial.size(), stream);
  index_params params;
  params.metric                   = cuvs::distance::DistanceType::BitwiseHamming;
  params.n_lists                  = 1;
  params.adaptive_centers         = true;
  params.kmeans_n_iters           = 3;
  params.kmeans_trainset_fraction = 1.0;
  auto idx                        = build(handle, params, raft::make_const_mdspan(data.view()));
  std::stringstream bytes;
  serialize(handle, bytes, idx);
  index<uint8_t, int64_t> restored(handle);
  deserialize(handle, bytes, &restored);
  ASSERT_EQ(restored.binary_center_counts().extent(1), 8);
  ASSERT_TRUE(cuvs::devArrMatch(idx.binary_center_counts().data_handle(),
                                restored.binary_center_counts().data_handle(),
                                idx.binary_center_counts().size(),
                                cuvs::Compare<uint32_t>(),
                                stream.get()));
  auto more = raft::make_device_matrix<uint8_t, int64_t>(handle, 3, 1);
  auto ids  = raft::make_device_vector<int64_t, int64_t>(handle, 3);
  const std::vector<uint8_t> more_host{0, 0, 0};
  const std::vector<int64_t> ids_host{4, 5, 6};
  raft::update_device(more.data_handle(), more_host.data(), more_host.size(), stream);
  raft::update_device(ids.data_handle(), ids_host.data(), ids_host.size(), stream);
  const std::optional<raft::device_vector_view<const int64_t, int64_t>> ids_view =
    raft::make_const_mdspan(ids.view());
  // The returned-index overload exercises cloning of both center representations.
  auto extended = extend(handle, raft::make_const_mdspan(more.view()), ids_view, restored);
  uint8_t center;
  uint8_t old_center;
  uint32_t count;
  raft::update_host(&center, extended.binary_centers().data_handle(), 1, stream);
  raft::update_host(&old_center, restored.binary_centers().data_handle(), 1, stream);
  raft::update_host(&count, extended.binary_center_counts().data_handle(), 1, stream);
  raft::resource::sync_stream(handle);
  EXPECT_EQ(extended.size(), 7);
  EXPECT_EQ(restored.size(), 4);
  EXPECT_EQ(old_center, 1);
  EXPECT_EQ(center, 0);  // Three one bits out of seven, not a majority.
  EXPECT_EQ(count, 3);
}

TEST(BinaryIvfFlatRegression, RejectNonByteInputTypes)
{
  raft::resources handle;
  index_params params;
  params.metric  = cuvs::distance::DistanceType::BitwiseHamming;
  params.n_lists = 1;
  EXPECT_THROW((index<float, int64_t>(handle, params, 16)), raft::logic_error);
  EXPECT_THROW((index<int8_t, int64_t>(handle, params, 16)), raft::logic_error);
  EXPECT_THROW((index<half, int64_t>(handle, params, 16)), raft::logic_error);
  params.adaptive_centers = true;
  EXPECT_THROW((index<uint8_t, int64_t>(handle, params, std::numeric_limits<uint32_t>::max())),
               raft::logic_error);
}

TEST(BinaryIvfFlatRegression, AdaptiveHostBatchesWithCopyStream)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
  constexpr int64_t n_rows = 65539;  // Cross the host staging boundary of 65536 rows.
  auto data                = raft::make_host_matrix<uint8_t, int64_t>(n_rows, 3);
  for (int64_t row = 0; row < n_rows; ++row) {
    data(row, 0) = row % 3 == 0 ? 0xff : 0;
    data(row, 1) = row % 3 == 0 ? 0 : 0xff;
    data(row, 2) = row % 2 == 0 ? 0x55 : 0xaa;
  }
  index_params params;
  params.metric           = cuvs::distance::DistanceType::BitwiseHamming;
  params.n_lists          = 1;
  params.adaptive_centers = true;
  params.kmeans_n_iters   = 3;
  auto idx                = build(handle, params, raft::make_const_mdspan(data.view()));
  std::vector<uint8_t> center(3);
  std::vector<uint32_t> counts(24);
  raft::update_host(center.data(), idx.binary_centers().data_handle(), center.size(), stream);
  raft::update_host(counts.data(), idx.binary_center_counts().data_handle(), counts.size(), stream);
  raft::resource::sync_stream(handle);
  EXPECT_EQ(idx.size(), n_rows);
  for (int64_t col = 0; col < 3; ++col) {
    uint8_t expected = 0;
    for (int bit = 0; bit < 8; ++bit) {
      int sum = 0;
      for (int64_t row = 0; row < n_rows; ++row) {
        sum += ((data(row, col) >> bit) & 1) ? 1 : -1;
      }
      if (sum > 0) { expected |= uint8_t(1 << bit); }
      EXPECT_EQ(counts[col * 8 + bit], (sum + n_rows) / 2);
    }
    EXPECT_EQ(center[col], expected);
  }
}

TEST(BinaryIvfFlatRegression, AdaptiveExactTieAfterTrainingOnly)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  // mean = 7 / 13 rounds upward in float; reconstructing its sum would make a tie positive.
  std::vector<uint8_t> initial(13);
  std::fill(initial.begin(), initial.begin() + 10, 1);
  auto data = raft::make_device_matrix<uint8_t, int64_t>(handle, 13, 1);
  raft::update_device(data.data_handle(), initial.data(), initial.size(), stream);
  index_params params;
  params.metric                   = cuvs::distance::DistanceType::BitwiseHamming;
  params.n_lists                  = 1;
  params.adaptive_centers         = true;
  params.add_data_on_build        = false;
  params.kmeans_n_iters           = 3;
  params.kmeans_trainset_fraction = 1.0;
  auto idx                        = build(handle, params, raft::make_const_mdspan(data.view()));
  std::vector<uint32_t> counts(8);
  raft::update_host(counts.data(), idx.binary_center_counts().data_handle(), counts.size(), stream);
  raft::resource::sync_stream(handle);
  EXPECT_EQ(idx.size(), 0);
  EXPECT_EQ(counts, std::vector<uint32_t>(8, 0));
  const std::optional<raft::device_vector_view<const int64_t, int64_t>> no_ids = std::nullopt;
  extend(handle, raft::make_const_mdspan(data.view()), no_ids, &idx);
  auto more = raft::make_device_matrix<uint8_t, int64_t>(handle, 7, 1);
  auto ids  = raft::make_device_vector<int64_t, int64_t>(handle, 7);
  const std::vector<uint8_t> more_host(7, 0);
  const std::vector<int64_t> ids_host{13, 14, 15, 16, 17, 18, 19};
  raft::update_device(more.data_handle(), more_host.data(), more_host.size(), stream);
  raft::update_device(ids.data_handle(), ids_host.data(), ids_host.size(), stream);
  const std::optional<raft::device_vector_view<const int64_t, int64_t>> ids_view =
    raft::make_const_mdspan(ids.view());
  extend(handle, raft::make_const_mdspan(more.view()), ids_view, &idx);
  uint8_t center;
  raft::update_host(&center, idx.binary_centers().data_handle(), 1, stream);
  raft::update_host(counts.data(), idx.binary_center_counts().data_handle(), counts.size(), stream);
  raft::resource::sync_stream(handle);
  EXPECT_EQ(idx.size(), 20);
  EXPECT_EQ(counts[0], 10);
  EXPECT_EQ(center, 0);  // Exact ties use the same > 0 convention as binary quantization.
}

TEST(BinaryIvfFlatRegression, LegacySerializationCompatibility)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  const std::vector<uint8_t> data_host{1, 2, 3, 4};
  auto data = raft::make_device_matrix<uint8_t, int64_t>(handle, 4, 1);
  raft::update_device(data.data_handle(), data_host.data(), data_host.size(), stream);
  for (auto metric :
       {cuvs::distance::DistanceType::L2Expanded, cuvs::distance::DistanceType::BitwiseHamming}) {
    index_params params;
    params.metric                   = metric;
    params.n_lists                  = 1;
    params.kmeans_n_iters           = 3;
    params.kmeans_trainset_fraction = 1.0;
    auto idx                        = build(handle, params, raft::make_const_mdspan(data.view()));
    std::stringstream serialized;
    serialize(handle, serialized, idx);
    char dtype[4];
    serialized.read(dtype, 4);
    EXPECT_EQ(raft::deserialize_scalar<int>(handle, serialized), 6);
    for (int version : {4, 5}) {
      // Version 4 and the original binary version 5 stored padded list lengths and IDs.
      // Upstream's nonbinary version 5 stores the actual list length with padded vector data.
      const uint32_t stored_size = version == 4 || idx.binary_index() ? 32 : 4;
      std::stringstream legacy;
      legacy.write(dtype, 4);
      raft::serialize_scalar(handle, legacy, version);
      raft::serialize_scalar(handle, legacy, idx.size());
      raft::serialize_scalar(handle, legacy, idx.dim());
      raft::serialize_scalar(handle, legacy, idx.n_lists());
      raft::serialize_scalar(handle, legacy, idx.metric());
      raft::serialize_scalar(handle, legacy, idx.adaptive_centers());
      raft::serialize_scalar(handle, legacy, idx.conservative_memory_allocation());
      index<uint8_t, int64_t> restored(handle);
      if (version == 4 && idx.binary_index()) {
        EXPECT_THROW(deserialize(handle, legacy, &restored), raft::logic_error);
        continue;
      }
      if (idx.binary_index()) {
        raft::serialize_mdspan(handle, legacy, idx.binary_centers());
      } else {
        raft::serialize_mdspan(handle, legacy, idx.centers());
      }
      raft::serialize_scalar(handle, legacy, idx.center_norms().has_value());
      if (idx.center_norms()) { raft::serialize_mdspan(handle, legacy, *idx.center_norms()); }
      raft::serialize_mdspan(handle, legacy, idx.list_sizes());
      raft::serialize_scalar(handle, legacy, stored_size);
      raft::serialize_mdspan(
        handle,
        legacy,
        raft::make_device_matrix_view<const uint8_t, uint32_t>(idx.lists()[0]->data_ptr(), 32, 1));
      raft::serialize_mdspan(handle,
                             legacy,
                             raft::make_device_vector_view<const int64_t, uint32_t>(
                               idx.lists()[0]->indices_ptr(), stored_size));
      deserialize(handle, legacy, &restored);
      ASSERT_EQ(restored.size(), idx.size());
      ASSERT_EQ(restored.metric(), idx.metric());
      if (idx.binary_index()) {
        ASSERT_TRUE(cuvs::devArrMatch(idx.binary_centers().data_handle(),
                                      restored.binary_centers().data_handle(),
                                      idx.binary_centers().size(),
                                      cuvs::Compare<uint8_t>(),
                                      stream.get()));
      } else {
        ASSERT_TRUE(cuvs::devArrMatch(idx.centers().data_handle(),
                                      restored.centers().data_handle(),
                                      idx.centers().size(),
                                      cuvs::Compare<float>(),
                                      stream.get()));
      }
      ASSERT_TRUE(cuvs::devArrMatch(idx.lists()[0]->indices_ptr(),
                                    restored.lists()[0]->indices_ptr(),
                                    data_host.size(),
                                    cuvs::Compare<int64_t>(),
                                    stream.get()));
    }
  }
}

}  // namespace cuvs::neighbors::ivf_flat
