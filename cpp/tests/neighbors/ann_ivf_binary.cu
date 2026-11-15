/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <cuvs/core/bitset.hpp>
#include <cuvs/core/dataset/binary.hpp>
#include <cuvs/neighbors/ivf_binary.hpp>
#include <cuvs/preprocessing/quantize/binary.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/init.cuh>
#include <raft/util/integer_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <random>
#include <sstream>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace cuvs::neighbors::ivf_binary {
namespace {

namespace binary = cuvs::preprocessing::quantize::binary;

template <typename T>
auto to_device(raft::resources const& handle,
               const std::vector<T>& host,
               int64_t n_rows,
               int64_t dim)
{
  auto out = raft::make_device_matrix<T, int64_t>(handle, n_rows, dim);
  raft::update_device(
    out.data_handle(), host.data(), host.size(), raft::resource::get_cuda_stream(handle));
  return out;
}

template <typename T>
auto to_host(raft::resources const& handle, const T* device, size_t n)
{
  std::vector<T> out(n);
  raft::update_host(out.data(), device, n, raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);
  return out;
}

// Codes with the zero threshold, 8 bits per byte
auto zero_threshold_dataset(const raft::device_matrix<uint8_t, int64_t>& codes)
  -> device_binary_dataset_view<float, int64_t>
{
  // The zero threshold is implied by the dimension alone
  return {raft::make_const_mdspan(codes.view()),
          binary::quantizer_view<const float>{uint32_t(codes.extent(1) * 8)}};
}

auto hamming(const uint8_t* a, const uint8_t* b, int64_t n_bytes) -> float
{
  int d = 0;
  for (int64_t i = 0; i < n_bytes; ++i) {
    d += __builtin_popcount(unsigned(a[i] ^ b[i]));
  }
  return float(d);
}

index_params small_params(uint32_t n_lists)
{
  index_params params;
  params.n_lists                  = n_lists;
  params.kmeans_n_iters           = 3;
  params.kmeans_trainset_fraction = 1.0;
  return params;
}

TEST(IvfBinary, ExhaustiveSearchMatchesBruteForce)
{
  raft::resources handle;
  constexpr int64_t n_rows    = 1536;
  constexpr int64_t n_queries = 3;
  std::mt19937 rng(42);
  std::normal_distribution<float> normal(0.5f, 1.0f);
  for (uint32_t dim : {1u, 3u, 16u, 100u, 192u}) {
    for (auto threshold : {binary::bit_threshold::zero, binary::bit_threshold::mean}) {
      SCOPED_TRACE(dim);
      const int64_t n_bytes = (dim + 7) / 8;
      std::vector<float> data(n_rows * dim);
      std::vector<float> queries(n_queries * dim);
      for (auto& x : data) {
        x = normal(rng);
      }
      for (auto& x : queries) {
        x = normal(rng);
      }
      auto data_dev    = to_device(handle, data, n_rows, dim);
      auto queries_dev = to_device(handle, queries, n_queries, dim);

      binary::params quantizer_params;
      quantizer_params.threshold = threshold;
      auto quantizer =
        binary::train(handle, quantizer_params, raft::make_const_mdspan(data_dev.view()));
      auto codes       = raft::make_device_matrix<uint8_t, int64_t>(handle, n_rows, n_bytes);
      auto query_codes = raft::make_device_matrix<uint8_t, int64_t>(handle, n_queries, n_bytes);
      binary::transform(handle, quantizer, raft::make_const_mdspan(data_dev.view()), codes.view());
      ASSERT_EQ(quantizer.dim(), dim);
      const device_binary_dataset_view<float, int64_t> dataset{
        raft::make_const_mdspan(codes.view()), quantizer.view()};

      auto params = small_params(16);
      auto idx    = build(handle, params, dataset);
      ASSERT_EQ(idx.dim(), dim);
      ASSERT_EQ(idx.storage_dim(), n_bytes);
      ASSERT_EQ(idx.size(), n_rows);

      auto codes_host = to_host(handle, codes.data_handle(), codes.size());

      search_params search_params;
      search_params.n_probes = params.n_lists;
      // Query codes may be produced using a different quantizer from the build dataset.
      for (auto query_threshold : {binary::bit_threshold::zero, binary::bit_threshold::mean}) {
        binary::params query_quantizer_params;
        query_quantizer_params.threshold = query_threshold;
        auto query_quantizer =
          binary::train(handle, query_quantizer_params, raft::make_const_mdspan(data_dev.view()));
        binary::transform(
          handle, query_quantizer, raft::make_const_mdspan(queries_dev.view()), query_codes.view());
        auto query_codes_host = to_host(handle, query_codes.data_handle(), query_codes.size());
        // Include a k larger than the local top-k capacity.
        for (int64_t k : {10, 100, 1025}) {
          SCOPED_TRACE(k);
          auto ids_dev       = raft::make_device_matrix<int64_t, int64_t>(handle, n_queries, k);
          auto distances_dev = raft::make_device_matrix<float, int64_t>(handle, n_queries, k);
          search(handle,
                 search_params,
                 idx,
                 raft::make_const_mdspan(query_codes.view()),
                 ids_dev.view(),
                 distances_dev.view());
          auto ids       = to_host(handle, ids_dev.data_handle(), ids_dev.size());
          auto distances = to_host(handle, distances_dev.data_handle(), distances_dev.size());
          for (int64_t q = 0; q < n_queries; ++q) {
            std::vector<float> expected(n_rows);
            for (int64_t row = 0; row < n_rows; ++row) {
              expected[row] =
                hamming(&codes_host[row * n_bytes], &query_codes_host[q * n_bytes], n_bytes);
            }
            auto sorted = expected;
            std::sort(sorted.begin(), sorted.end());
            std::vector<bool> seen(n_rows);
            std::vector<float> actual(distances.begin() + q * k, distances.begin() + (q + 1) * k);
            std::sort(actual.begin(), actual.end());
            for (int64_t j = 0; j < k; ++j) {
              // Compare distances rather than ids: the k-th neighbor often has many ties
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
  }
}

TEST(IvfBinary, SerializationRoundTrip)
{
  raft::resources handle;
  auto stream                 = raft::resource::get_cuda_stream(handle);
  constexpr int64_t n_rows    = 512;
  constexpr int64_t n_bytes   = 13;
  constexpr int64_t n_queries = 4;
  std::mt19937 rng(7);
  std::vector<uint8_t> data(n_rows * n_bytes);
  std::vector<uint8_t> queries(n_queries * n_bytes);
  for (auto& x : data) {
    x = static_cast<uint8_t>(rng());
  }
  for (auto& x : queries) {
    x = static_cast<uint8_t>(rng());
  }
  const uint32_t dim = 100;
  // The unused high bits of the final byte are not part of the logical vector.
  for (int64_t row = 0; row < n_rows; ++row) {
    data[(row + 1) * n_bytes - 1] &= (1u << (dim % 8)) - 1;
  }
  for (int64_t row = 0; row < n_queries; ++row) {
    queries[(row + 1) * n_bytes - 1] &= (1u << (dim % 8)) - 1;
  }
  auto codes       = to_device(handle, data, n_rows, n_bytes);
  auto query_codes = to_device(handle, queries, n_queries, n_bytes);
  const device_binary_dataset_view<float, int64_t> dataset{
    raft::make_const_mdspan(codes.view()), binary::quantizer_view<const float>{dim}};
  auto encoded_queries = raft::make_const_mdspan(query_codes.view());

  auto params             = small_params(8);
  params.adaptive_centers = true;
  auto idx                = build(handle, params, dataset);

  std::stringstream bytes;
  serialize(handle, bytes, idx);
  index<int64_t> restored(handle);
  deserialize(handle, bytes, &restored);
  ASSERT_EQ(restored.dim(), dim);
  ASSERT_EQ(restored.size(), n_rows);
  ASSERT_EQ(restored.n_lists(), params.n_lists);
  ASSERT_TRUE(restored.adaptive_centers());
  ASSERT_TRUE(cuvs::devArrMatch(idx.centers().data_handle(),
                                restored.centers().data_handle(),
                                idx.centers().size(),
                                cuvs::Compare<uint8_t>(),
                                stream.get()));
  ASSERT_TRUE(cuvs::devArrMatch(idx.center_counts().data_handle(),
                                restored.center_counts().data_handle(),
                                idx.center_counts().size(),
                                cuvs::Compare<uint32_t>(),
                                stream.get()));

  constexpr int64_t k = 10;
  search_params search_params;
  search_params.n_probes = params.n_lists;
  auto distances_a       = raft::make_device_matrix<float, int64_t>(handle, n_queries, k);
  auto distances_b       = raft::make_device_matrix<float, int64_t>(handle, n_queries, k);
  auto ids               = raft::make_device_matrix<int64_t, int64_t>(handle, n_queries, k);
  search(handle, search_params, idx, encoded_queries, ids.view(), distances_a.view());
  search(handle, search_params, restored, encoded_queries, ids.view(), distances_b.view());
  ASSERT_TRUE(cuvs::devArrMatch(distances_a.data_handle(),
                                distances_b.data_handle(),
                                distances_a.size(),
                                cuvs::Compare<float>(),
                                stream.get()));
}

TEST(IvfBinary, AdaptiveCentersKeepExactCounts)
{
  raft::resources handle;
  auto stream             = raft::resource::get_cuda_stream(handle);
  auto data               = to_device(handle, std::vector<uint8_t>{1, 1, 1, 0}, 4, 1);
  auto params             = small_params(1);
  params.adaptive_centers = true;
  auto idx                = build(handle, params, zero_threshold_dataset(data));
  ASSERT_EQ(idx.center_counts().extent(1), 8);
  ASSERT_EQ(to_host(handle, idx.centers().data_handle(), 1)[0], 1);
  ASSERT_EQ(to_host(handle, idx.center_counts().data_handle(), 1)[0], 3);

  std::stringstream bytes;
  serialize(handle, bytes, idx);
  index<int64_t> restored(handle);
  deserialize(handle, bytes, &restored);

  auto more = to_device(handle, std::vector<uint8_t>{0, 0, 0}, 3, 1);
  auto ids  = raft::make_device_vector<int64_t, int64_t>(handle, 3);
  const std::vector<int64_t> ids_host{4, 5, 6};
  raft::update_device(ids.data_handle(), ids_host.data(), ids_host.size(), stream);
  extend(handle,
         zero_threshold_dataset(more),
         std::make_optional(raft::make_const_mdspan(ids.view())),
         &restored);
  EXPECT_EQ(restored.size(), 7);
  EXPECT_EQ(to_host(handle, restored.centers().data_handle(), 1)[0], 0);  // 3 ones out of 7
  EXPECT_EQ(to_host(handle, restored.center_counts().data_handle(), 1)[0], 3);
  EXPECT_EQ(idx.size(), 4);
}

TEST(IvfBinary, AdaptiveExactTieAfterTrainingOnly)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  // mean = 7 / 13 rounds upward in float; reconstructing its sum would make a tie positive.
  std::vector<uint8_t> initial(13);
  std::fill(initial.begin(), initial.begin() + 10, 1);
  auto data                = to_device(handle, initial, 13, 1);
  auto params              = small_params(1);
  params.adaptive_centers  = true;
  params.add_data_on_build = false;
  auto idx                 = build(handle, params, zero_threshold_dataset(data));
  EXPECT_EQ(idx.size(), 0);
  EXPECT_EQ(to_host(handle, idx.center_counts().data_handle(), 8), std::vector<uint32_t>(8, 0));

  extend(handle, zero_threshold_dataset(data), std::nullopt, &idx);
  auto more = to_device(handle, std::vector<uint8_t>(7, 0), 7, 1);
  auto ids  = raft::make_device_vector<int64_t, int64_t>(handle, 7);
  const std::vector<int64_t> ids_host{13, 14, 15, 16, 17, 18, 19};
  raft::update_device(ids.data_handle(), ids_host.data(), ids_host.size(), stream);
  extend(handle,
         zero_threshold_dataset(more),
         std::make_optional(raft::make_const_mdspan(ids.view())),
         &idx);
  EXPECT_EQ(idx.size(), 20);
  EXPECT_EQ(to_host(handle, idx.center_counts().data_handle(), 1)[0], 10);
  EXPECT_EQ(to_host(handle, idx.centers().data_handle(), 1)[0], 0);  // exact ties are zeros
}

TEST(IvfBinary, RejectsMismatchedInputs)
{
  raft::resources handle;
  auto data = to_device(handle, std::vector<uint8_t>(32, 0x55), 32, 1);
  auto idx  = build(handle, small_params(1), zero_threshold_dataset(data));
  auto ids  = raft::make_device_matrix<int64_t, int64_t>(handle, 1, 1);
  auto dist = raft::make_device_matrix<float, int64_t>(handle, 1, 1);

  // The row length of the codes must match the dimension
  const device_binary_dataset_view<float, int64_t> bad_row_length{
    raft::make_const_mdspan(data.view()), binary::quantizer_view<const float>{17}};
  EXPECT_THROW(build(handle, small_params(1), bad_row_length), raft::logic_error);
  EXPECT_THROW(extend(handle, bad_row_length, std::nullopt, &idx), raft::logic_error);

  // Reject a different logical dimension, even when the encoded byte width matches.
  const device_binary_dataset_view<float, int64_t> other_dim{
    raft::make_const_mdspan(data.view()), binary::quantizer_view<const float>{7}};
  EXPECT_THROW(extend(handle, other_dim, std::nullopt, &idx), raft::logic_error);

  // Query width is measured in packed bytes.
  auto queries = raft::make_device_matrix<uint8_t, int64_t>(handle, 1, 2);
  EXPECT_THROW(search(handle,
                      search_params{},
                      idx,
                      raft::make_const_mdspan(queries.view()),
                      ids.view(),
                      dist.view()),
               raft::logic_error);
}

TEST(IvfBinary, ExtendAcceptsDifferentThresholds)
{
  raft::resources handle;
  auto data            = to_device(handle, std::vector<uint8_t>{1, 3}, 2, 1);
  auto build_threshold = raft::make_device_vector<float, int64_t>(handle, 8);
  raft::matrix::fill(handle, build_threshold.view(), 0.0f);
  const device_binary_dataset_view<float, int64_t> dataset{
    raft::make_const_mdspan(data.view()),
    binary::quantizer_view<const float>{raft::make_const_mdspan(build_threshold.view())}};
  auto params             = small_params(1);
  params.adaptive_centers = true;
  auto idx                = build(handle, params, dataset);

  auto more          = to_device(handle, std::vector<uint8_t>{0xf0}, 1, 1);
  auto new_threshold = raft::make_device_vector<float, int64_t>(handle, 8);
  raft::matrix::fill(handle, new_threshold.view(), 1.0f);
  const device_binary_dataset_view<float, int64_t> new_vectors{
    raft::make_const_mdspan(more.view()),
    binary::quantizer_view<const float>{raft::make_const_mdspan(new_threshold.view())}};
  auto new_ids = to_device(handle, std::vector<int64_t>{2}, 1, 1);
  extend(handle,
         new_vectors,
         std::make_optional(
           raft::make_device_vector_view<const int64_t, int64_t>(new_ids.data_handle(), 1)),
         &idx);
  ASSERT_EQ(idx.size(), 3);
  EXPECT_EQ(to_host(handle, idx.center_counts().data_handle(), 8),
            (std::vector<uint32_t>{2, 1, 0, 0, 1, 1, 1, 1}));
  // Extending does not replace the quantizer metadata captured at build time.
  ASSERT_EQ(idx.quantizer().dim(), 8);
  EXPECT_EQ(to_host(handle, std::get<1>(idx.quantizer().value).data_handle(), 8),
            std::vector<float>(8, 0.0f));

  auto ids       = raft::make_device_matrix<int64_t, int64_t>(handle, 1, 3);
  auto distances = raft::make_device_matrix<float, int64_t>(handle, 1, 3);
  search_params search_params;
  search_params.n_probes = 1;
  search(
    handle, search_params, idx, raft::make_const_mdspan(more.view()), ids.view(), distances.view());
  auto result_ids       = to_host(handle, ids.data_handle(), 3);
  auto result_distances = to_host(handle, distances.data_handle(), 3);
  const std::vector<float> expected{5, 6, 0};
  for (int64_t i = 0; i < 3; ++i) {
    ASSERT_GE(result_ids[i], 0);
    ASSERT_LT(result_ids[i], 3);
    EXPECT_EQ(result_distances[i], expected[result_ids[i]]);
  }
  std::sort(result_ids.begin(), result_ids.end());
  EXPECT_EQ(result_ids, (std::vector<int64_t>{0, 1, 2}));
}

TEST(IvfBinary, SearchHonorsSampleFilter)
{
  raft::resources handle;
  constexpr int64_t n_rows    = 1536;
  constexpr int64_t n_bytes   = 8;
  constexpr int64_t n_removed = 512;
  constexpr int64_t k         = 10;
  std::mt19937 rng(3);
  std::vector<uint8_t> data(n_rows * n_bytes);
  std::vector<uint8_t> query(n_bytes);
  for (auto& x : data) {
    x = static_cast<uint8_t>(rng());
  }
  for (auto& x : query) {
    x = static_cast<uint8_t>(rng());
  }
  auto codes       = to_device(handle, data, n_rows, n_bytes);
  auto query_codes = to_device(handle, query, 1, n_bytes);
  auto params      = small_params(16);
  auto idx         = build(handle, params, zero_threshold_dataset(codes));

  auto removed = raft::make_device_vector<int64_t, int64_t>(handle, n_removed);
  raft::linalg::map_offset(handle, removed.view(), raft::identity_op{});
  raft::resource::sync_stream(handle);
  cuvs::core::bitset<uint32_t, int64_t> removed_bitset(handle, removed.view(), n_rows);
  auto filter = cuvs::neighbors::filtering::bitset_filter(removed_bitset.view());

  search_params search_params;
  search_params.n_probes = params.n_lists;
  auto ids_dev           = raft::make_device_matrix<int64_t, int64_t>(handle, 1, k);
  auto distances_dev     = raft::make_device_matrix<float, int64_t>(handle, 1, k);
  search(handle,
         search_params,
         idx,
         raft::make_const_mdspan(query_codes.view()),
         ids_dev.view(),
         distances_dev.view(),
         filter);
  auto ids       = to_host(handle, ids_dev.data_handle(), k);
  auto distances = to_host(handle, distances_dev.data_handle(), k);

  std::vector<float> expected;
  for (int64_t row = n_removed; row < n_rows; ++row) {
    expected.push_back(hamming(&data[row * n_bytes], query.data(), n_bytes));
  }
  std::sort(expected.begin(), expected.end());
  for (int64_t j = 0; j < k; ++j) {
    ASSERT_GE(ids[j], n_removed);
    ASSERT_EQ(distances[j], hamming(&data[ids[j] * n_bytes], query.data(), n_bytes));
  }
  std::sort(distances.begin(), distances.end());
  for (int64_t j = 0; j < k; ++j) {
    ASSERT_EQ(distances[j], expected[j]);
  }
}

TEST(IvfBinary, OwningDatasetViews)
{
  raft::resources handle;
  constexpr int64_t n_rows  = 64;
  constexpr uint32_t dim    = 100;
  constexpr int64_t n_bytes = raft::div_rounding_up_safe<uint32_t>(dim, 8);
  using dataset_type        = device_binary_dataset<float, int64_t>;

  auto codes = to_device(handle, std::vector<uint8_t>(n_rows * n_bytes, 0x5a), n_rows, n_bytes);
  // The zero threshold is the dimension alone
  dataset_type dataset{std::move(codes), dataset_type::manifest_type{dim}};
  const auto& const_dataset = dataset;

  // The dimension is the number of bits, not the padded byte width
  ASSERT_EQ(const_dataset.n_rows(), n_rows);
  ASSERT_EQ(const_dataset.dim(), dim);
  ASSERT_EQ(const_dataset.data_const_view().extent(1), n_bytes);

  // A mutable view decays to the read-only one the API takes
  device_binary_dataset_view<float, int64_t> view = dataset.view();
  ASSERT_EQ(view.n_rows(), n_rows);
  ASSERT_EQ(view.dim(), dim);
  ASSERT_TRUE(std::holds_alternative<uint32_t>(view.const_manifest_view().value));

  auto idx = build(handle, small_params(4), const_dataset.view());
  ASSERT_EQ(idx.dim(), dim);
  ASSERT_EQ(idx.size(), n_rows);

  // With the thresholds, the dimension is their number
  dataset_type with_threshold{
    to_device(handle, std::vector<uint8_t>(n_rows * n_bytes, 0x5a), n_rows, n_bytes),
    dataset_type::manifest_type{raft::make_device_vector<float, int64_t>(handle, dim)}};
  ASSERT_EQ(with_threshold.dim(), dim);
  auto idx_with_threshold = build(handle, small_params(4), with_threshold.view());
  ASSERT_EQ(idx_with_threshold.dim(), dim);
  ASSERT_NE(std::get_if<1>(&idx_with_threshold.quantizer().value), nullptr);
}

// The views have the constness semantics of an mdspan
static_assert(
  std::is_convertible_v<binary::quantizer_view<float>, binary::quantizer_view<const float>>);
static_assert(
  !std::is_convertible_v<binary::quantizer_view<const float>, binary::quantizer_view<float>>);
static_assert(std::is_same_v<decltype(std::declval<const device_binary_dataset<float, int64_t>&>()
                                        .view()
                                        .const_manifest_view()),
                             binary::quantizer_view<const float>>);
static_assert(
  std::is_same_v<
    decltype(std::declval<device_binary_dataset<float, int64_t>&>().view().manifest_view()),
    binary::quantizer_view<float>>);

}  // namespace
}  // namespace cuvs::neighbors::ivf_binary
