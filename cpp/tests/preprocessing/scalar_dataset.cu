/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/dataset/sq.hpp>
#include <gtest/gtest.h>
#include <raft/core/device_resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuvs::preprocessing::quantize::scalar {

template <typename T>
auto copy_to_host(raft::resources const& res, const T* data, int64_t size)
{
  std::vector<T> result(size);
  auto stream = raft::resource::get_cuda_stream(res);
  raft::update_host(result.data(), data, size, stream);
  raft::resource::sync_stream(res);
  return result;
}

TEST(ScalarDataset, CorpusAndQueriesShareQuantizer)
{
  raft::device_resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<float> corpus{0.f, 10.f, 9.f, 254.f, 14.f, 9.f};
  auto input = raft::make_device_matrix<float, int64_t>(res, 2, 3);
  raft::update_device(input.data_handle(), corpus.data(), corpus.size(), stream);
  auto encoded = build_per_dim(res, raft::make_const_mdspan(input.view()));
  ASSERT_EQ(encoded.n_rows(), 2);
  ASSERT_EQ(encoded.dim(), 3u);
  auto quantizer = encoded.const_manifest_view();
  EXPECT_FLOAT_EQ(quantizer.scale, 1.f);
  EXPECT_EQ(copy_to_host(res, quantizer.offsets.data_handle(), 3),
            (std::vector<float>{127.f, 12.f, 9.f}));
  EXPECT_EQ(copy_to_host(res, encoded.data_const_view().data_handle(), 6),
            (std::vector<int8_t>{-127, -2, 0, 127, 2, 0}));

  // The corpus manifest, rather than query extrema, controls ties and saturation.
  std::vector<float> queries{127.5f, 11.5f, 1000.f, 128.5f, 9.5f, -1000.f};
  raft::update_device(input.data_handle(), queries.data(), queries.size(), stream);
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 2, 3);
  transform(res, quantizer, raft::make_const_mdspan(input.view()), codes.view());
  EXPECT_EQ(copy_to_host(res, codes.data_handle(), 6),
            (std::vector<int8_t>{0, 0, 127, 2, -2, -127}));
  inverse_transform(res, quantizer, raft::make_const_mdspan(codes.view()), input.view());
  EXPECT_EQ(copy_to_host(res, input.data_handle(), 6),
            (std::vector<float>{127.f, 12.f, 136.f, 129.f, 10.f, -118.f}));

  // Ownership moves without copying codes or offsets, and views preserve constness.
  static_assert(!std::is_copy_constructible_v<device_per_dim_sq_dataset>);
  auto moved                                            = std::move(encoded);
  device_per_dim_sq_dataset::const_view_type const_view = moved.view();
  static_assert(std::is_const_v<typename decltype(const_view.data_view())::element_type>);
  EXPECT_EQ(const_view.const_manifest_view().offsets.data_handle(),
            quantizer.offsets.data_handle());
  EXPECT_EQ(const_view.n_rows(), 2);
  EXPECT_EQ(copy_to_host(res, const_view.data_const_view().data_handle(), 6),
            (std::vector<int8_t>{-127, -2, 0, 127, 2, 0}));
}

TEST(ScalarDataset, FullCorpusExtremaAndChunkedEncoding)
{
  raft::device_resources res;
  constexpr int64_t rows = 4097, dim = 5;
  std::vector<float> corpus(rows * dim, 4.f);
  corpus[0]                    = -250.f;
  corpus[(rows - 1) * dim]     = 258.f;
  corpus[1]                    = 2.f;
  corpus[(rows - 1) * dim + 1] = 6.f;
  auto input                   = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  raft::update_device(
    input.data_handle(), corpus.data(), corpus.size(), raft::resource::get_cuda_stream(res));
  auto quantizer = train_per_dim(res, raft::make_const_mdspan(input.view()), params{1.f});
  EXPECT_FLOAT_EQ(quantizer.scale, 2.f);
  EXPECT_EQ(copy_to_host(res, quantizer.offsets.data_handle(), dim),
            (std::vector<float>(dim, 4.f)));
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, rows, dim);
  for (int64_t start = 0; start < rows; start += 1024) {
    auto n = std::min(int64_t{1024}, rows - start);
    transform(
      res,
      quantizer.view(),
      raft::make_device_matrix_view<const float, int64_t>(
        input.data_handle() + start * dim, n, dim),
      raft::make_device_matrix_view<int8_t, int64_t>(codes.data_handle() + start * dim, n, dim));
  }
  auto result = copy_to_host(res, codes.data_handle(), rows * dim);
  EXPECT_EQ(result.front(), -127);
  EXPECT_EQ(result[(rows - 1) * dim], 127);
  EXPECT_EQ(result[1], -1);
  EXPECT_EQ(result[(rows - 1) * dim + 1], 1);
  EXPECT_EQ(result[1024 * dim + 2], 0);
}

TEST(ScalarDataset, PerDimensionQuantiles)
{
  raft::device_resources res;
  constexpr int64_t rows = 1000, dim = 3;
  std::vector<float> corpus(rows * dim);
  for (int64_t i = 0; i < rows; ++i) {
    corpus[i * dim]     = static_cast<float>(i);
    corpus[i * dim + 1] = 10000.f + 2.f * static_cast<float>(rows - i - 1);
    corpus[i * dim + 2] = 9.f;
  }
  auto input = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  raft::update_device(
    input.data_handle(), corpus.data(), corpus.size(), raft::resource::get_cuda_stream(res));
  auto encoded   = build_per_dim(res, raft::make_const_mdspan(input.view()));
  auto quantizer = encoded.const_manifest_view();
  EXPECT_EQ(copy_to_host(res, quantizer.offsets.data_handle(), dim),
            (std::vector<float>{499.5f, 10999.f, 9.f}));
  EXPECT_FLOAT_EQ(quantizer.scale, 991.f / 127.f);
  auto codes = copy_to_host(res, encoded.data_const_view().data_handle(), rows * dim);
  EXPECT_EQ(codes[1], 127);
  EXPECT_EQ(codes[(rows - 1) * dim + 1], -127);
  EXPECT_EQ(codes[2], 0);
  auto trimmed = train_per_dim(res, raft::make_const_mdspan(input.view()), params{0.5f});
  EXPECT_FLOAT_EQ(trimmed.scale, 499.f / 127.f);
  for (float q : {0.f, -1.f, 1.1f, std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_THROW(train_per_dim(res, raft::make_const_mdspan(input.view()), params{q}),
                 raft::logic_error);
  }
}

TEST(ScalarDataset, ConstantCorpus)
{
  raft::device_resources res;
  std::vector<float> corpus{10.f, -7.f, 10.f, -7.f};
  auto input = raft::make_device_matrix<float, int64_t>(res, 2, 2);
  raft::update_device(
    input.data_handle(), corpus.data(), corpus.size(), raft::resource::get_cuda_stream(res));
  auto encoded = build_per_dim(res, raft::make_const_mdspan(input.view()));
  EXPECT_FLOAT_EQ(encoded.const_manifest_view().scale, 1.f);
  EXPECT_EQ(copy_to_host(res, encoded.data_const_view().data_handle(), 4),
            (std::vector<int8_t>(4, 0)));
  inverse_transform(res, encoded.const_manifest_view(), encoded.data_const_view(), input.view());
  EXPECT_EQ(copy_to_host(res, input.data_handle(), 4), corpus);
}

TEST(ScalarDataset, ExtremeFiniteCorpus)
{
  raft::device_resources res;
  float const max = std::numeric_limits<float>::max();
  std::vector<float> corpus{-max, max};
  auto input = raft::make_device_matrix<float, int64_t>(res, 2, 1);
  raft::update_device(
    input.data_handle(), corpus.data(), corpus.size(), raft::resource::get_cuda_stream(res));
  auto encoded = build_per_dim(res, raft::make_const_mdspan(input.view()));
  EXPECT_EQ(copy_to_host(res, encoded.data_const_view().data_handle(), 2),
            (std::vector<int8_t>{-127, 127}));
  inverse_transform(res, encoded.const_manifest_view(), encoded.data_const_view(), input.view());
  auto decoded = copy_to_host(res, input.data_handle(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_TRUE(std::isfinite(decoded[i]));
    EXPECT_FLOAT_EQ(decoded[i], corpus[i]);
  }
}

TEST(ScalarDataset, CustomParametersAndValidation)
{
  raft::device_resources res;
  auto offsets = raft::make_device_vector<float, int64_t>(res, 2);
  std::vector<float> custom_offsets{100.f, -100.f};
  raft::update_device(offsets.data_handle(),
                      custom_offsets.data(),
                      custom_offsets.size(),
                      raft::resource::get_cuda_stream(res));
  per_dim_quantizer_const_view quantizer{raft::make_const_mdspan(offsets.view()), 2.f};
  auto input = raft::make_device_matrix<float, int64_t>(res, 1, 2);
  std::vector<float> values{104.f, -106.f};
  raft::update_device(
    input.data_handle(), values.data(), values.size(), raft::resource::get_cuda_stream(res));
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 1, 2);
  transform(res, quantizer, raft::make_const_mdspan(input.view()), codes.view());
  EXPECT_EQ(copy_to_host(res, codes.data_handle(), 2), (std::vector<int8_t>{2, -3}));
  auto wrong_shape = raft::make_device_matrix<int8_t, int64_t>(res, 2, 1);
  EXPECT_THROW(transform(res, quantizer, raft::make_const_mdspan(input.view()), wrong_shape.view()),
               raft::logic_error);
  for (float scale : {0.f, -1.f, std::numeric_limits<float>::infinity()}) {
    quantizer.scale = scale;
    EXPECT_THROW(transform(res, quantizer, raft::make_const_mdspan(input.view()), codes.view()),
                 raft::logic_error);
  }
  EXPECT_THROW(
    train_per_dim(res, raft::make_device_matrix_view<const float, int64_t>(nullptr, 0, 2)),
    raft::logic_error);
}

TEST(ScalarDataset, GlobalDatasetStillWorks)
{
  raft::device_resources res;
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 1, 2);
  device_sq_dataset<float, int64_t, float> dataset{std::move(codes), quantizer<float>{-1.f, 1.f}};
  decltype(dataset)::const_view_type view = dataset.view();
  EXPECT_EQ(view.dim(), 2u);
  EXPECT_FLOAT_EQ(view.const_manifest_view().min_, -1.f);
  EXPECT_FLOAT_EQ(view.const_manifest_view().max_, 1.f);
}

}  // namespace cuvs::preprocessing::quantize::scalar
