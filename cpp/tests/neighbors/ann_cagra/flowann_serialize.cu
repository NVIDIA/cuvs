/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/flowann_serialize.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

namespace flowann = cuvs::neighbors::cagra::experimental::flowann;

namespace {

template <typename T>
void write_dtype_prefix(std::ostream& os)
{
  auto dtype = raft::numpy_serializer::get_numpy_dtype<T>().to_string();
  dtype.resize(4);
  os.write(dtype.data(), dtype.size());
}

template <typename T>
void write_host_matrix(std::ostream& os, raft::host_matrix<T, int64_t> const& matrix)
{
  raft::numpy_serializer::write_header(
    os,
    {raft::numpy_serializer::get_numpy_dtype<T>(),
     false,
     {static_cast<std::uint64_t>(matrix.extent(0)), static_cast<std::uint64_t>(matrix.extent(1))}});
  os.write(reinterpret_cast<char const*>(matrix.data_handle()), matrix.size() * sizeof(T));
}

void write_host_f16_bits(std::ostream& os, raft::host_matrix<std::uint16_t, int64_t> const& matrix)
{
  raft::numpy_serializer::write_header(
    os,
    {{'<', 'e', sizeof(std::uint16_t)},
     false,
     {static_cast<std::uint64_t>(matrix.extent(0)), static_cast<std::uint64_t>(matrix.extent(1))}});
  os.write(reinterpret_cast<char const*>(matrix.data_handle()),
           matrix.size() * sizeof(std::uint16_t));
}

auto f16_bits(float value) -> std::uint16_t
{
  auto encoded = __float2half(value);
  std::uint16_t result{};
  static_assert(sizeof(result) == sizeof(encoded));
  std::memcpy(&result, &encoded, sizeof(result));
  return result;
}

void write_legacy_dataset_body(raft::resources const& res,
                               std::ostream& stream,
                               bool invalid_vq_label = false)
{
  constexpr int64_t rows                 = 4;
  constexpr std::uint32_t dim            = 4;
  constexpr std::uint32_t vq_centers     = 2;
  constexpr std::uint32_t pq_centers     = 256;
  constexpr std::uint32_t pq_len         = 2;
  constexpr std::uint32_t encoded_length = 6;

  raft::serialize_scalar(res, stream, flowann::detail::legacy_vpq_dataset_tag);
  raft::serialize_scalar(res, stream, CUDA_R_16F);
  raft::serialize_scalar(res, stream, rows);
  raft::serialize_scalar(res, stream, dim);
  raft::serialize_scalar(res, stream, vq_centers);
  raft::serialize_scalar(res, stream, pq_centers);
  raft::serialize_scalar(res, stream, pq_len);
  raft::serialize_scalar(res, stream, encoded_length);

  auto vq = raft::make_host_matrix<std::uint16_t, int64_t>(vq_centers, dim);
  auto pq = raft::make_host_matrix<std::uint16_t, int64_t>(pq_centers, pq_len);
  std::fill_n(vq.data_handle(), vq.size(), f16_bits(0.0f));
  std::fill_n(pq.data_handle(), pq.size(), f16_bits(0.0f));
  vq(1, 0) = f16_bits(10.0f);
  pq(1, 0) = f16_bits(1.0f);
  write_host_f16_bits(stream, vq);
  write_host_f16_bits(stream, pq);

  auto codes = raft::make_host_matrix<std::uint8_t, int64_t>(rows, encoded_length);
  for (int64_t row = 0; row < rows; ++row) {
    codes(row, 0) = static_cast<std::uint8_t>(invalid_vq_label && row == 0 ? vq_centers : row / 2);
    codes(row, 1) = 0;
    codes(row, 2) = static_cast<std::uint8_t>(row / 2);
    codes(row, 3) = 0;
    codes(row, 4) = static_cast<std::uint8_t>(row % 2);
    codes(row, 5) = 0;
  }
  write_host_matrix(stream, codes);

  constexpr std::uint32_t n_subgraphs = 2;
  raft::serialize_scalar(res, stream, n_subgraphs);
  auto offsets  = raft::make_host_matrix<std::uint32_t, int64_t>(1, n_subgraphs);
  offsets(0, 0) = 0;
  offsets(0, 1) = 2;
  write_host_matrix(stream, offsets);
}

auto make_legacy_dataset(raft::resources const& res,
                         bool truncate         = false,
                         bool invalid_vq_label = false) -> std::stringstream
{
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  write_dtype_prefix<float>(stream);
  raft::serialize_scalar(res, stream, flowann::detail::legacy_serialization_version);
  write_legacy_dataset_body(res, stream, invalid_vq_label);
  auto bytes = stream.str();
  if (truncate) { bytes.resize(bytes.size() - 1); }
  return std::stringstream(bytes, std::ios::in | std::ios::out | std::ios::binary);
}

enum class legacy_metric_header { typed_enum, historical_unsigned_enum };

void write_legacy_graph_body(raft::resources const& res,
                             std::ostream& stream,
                             legacy_metric_header metric_header = legacy_metric_header::typed_enum)
{
  constexpr std::uint32_t rows       = 4;
  constexpr std::uint32_t dim        = 4;
  constexpr std::uint32_t degree     = 2;
  constexpr std::uint16_t nodes_per  = 2;
  constexpr std::uint16_t index_bits = 2;

  raft::serialize_scalar(res, stream, rows);
  raft::serialize_scalar(res, stream, dim);
  raft::serialize_scalar(res, stream, degree);
  if (metric_header == legacy_metric_header::historical_unsigned_enum) {
    raft::serialize_scalar(
      res, stream, static_cast<std::uint32_t>(cuvs::distance::DistanceType::L2Expanded));
  } else {
    raft::serialize_scalar(res, stream, cuvs::distance::DistanceType::L2Expanded);
  }
  raft::serialize_scalar(res, stream, nodes_per);
  raft::serialize_scalar(res, stream, index_bits);

  auto inner  = raft::make_host_matrix<std::uint8_t, int64_t>(2, 4);
  inner(0, 0) = 1;
  inner(0, 1) = 1;
  inner(0, 2) = 1;
  inner(0, 3) = 0;
  inner(1, 0) = 1;
  inner(1, 1) = 1;
  inner(1, 2) = 1;
  inner(1, 3) = 0;
  write_host_matrix(stream, inner);

  auto cross = raft::make_host_matrix<std::uint32_t, int64_t>(rows, degree + 1);
  for (std::uint32_t row = 0; row < rows; ++row) {
    cross(row, 0) = 1;
    cross(row, 1) = (row + 2) % rows;
    cross(row, 2) = 0;
  }
  write_host_matrix(stream, cross);

  auto mapping  = raft::make_host_matrix<std::uint32_t, int64_t>(1, rows);
  mapping(0, 0) = 3;
  mapping(0, 1) = 2;
  mapping(0, 2) = 1;
  mapping(0, 3) = 0;
  write_host_matrix(stream, mapping);
}

auto make_legacy_graph(raft::resources const& res) -> std::stringstream
{
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  write_dtype_prefix<float>(stream);
  raft::serialize_scalar(res, stream, flowann::detail::legacy_serialization_version);
  write_legacy_graph_body(res, stream);
  stream.seekg(0);
  return stream;
}

auto make_legacy_combined_vpq(raft::resources const& res) -> std::stringstream
{
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  char const historical_dtype[4] = {'|', 'u', '1', '\0'};
  stream.write(historical_dtype, sizeof(historical_dtype));
  raft::serialize_scalar(res, stream, flowann::detail::legacy_serialization_version);
  write_legacy_graph_body(res, stream, legacy_metric_header::historical_unsigned_enum);
  raft::serialize_scalar(res, stream, true);
  write_legacy_dataset_body(res, stream);
  stream.seekg(0);
  return stream;
}

struct compact_test_graph {
  raft::host_matrix<std::uint8_t, int64_t> inner;
  raft::host_matrix<std::uint32_t, int64_t> cross;
};

template <typename IndexT>
auto install_compact_test_graph(raft::resources const& res, IndexT& index) -> compact_test_graph
{
  auto inner = raft::make_host_matrix<std::uint8_t, int64_t>(2, 3);
  std::fill_n(inner.data_handle(), inner.size(), std::uint8_t{0});
  inner(0, 0) = 1;
  inner(0, 1) = 1;
  inner(1, 0) = 1;
  inner(1, 1) = 1;
  auto cross  = raft::make_host_matrix<std::uint32_t, int64_t>(4, 2);
  for (std::uint32_t row = 0; row < 4; ++row) {
    cross(row, 0) = 1;
    cross(row, 1) = (row + 1) % 4;
  }
  index.update_graph(
    res, raft::make_const_mdspan(inner.view()), raft::make_const_mdspan(cross.view()), 2, 1);
  return {std::move(inner), std::move(cross)};
}

TEST(FlowannSerialize, LoadsAndNormalizesLegacyVpq)
{
  raft::resources res;
  auto stream = make_legacy_dataset(res);
  auto data   = flowann::detail::deserialize_legacy_vpq_f16<float>(res, stream);
  raft::resource::sync_stream(res);

  ASSERT_EQ(data.dataset->n_rows(), 4);
  ASSERT_EQ(data.dataset->dim(), 4);
  ASSERT_EQ(data.subgraph_offsets.extent(0), 2);
  EXPECT_EQ(data.subgraph_offsets(0), 0);
  EXPECT_EQ(data.subgraph_offsets(1), 2);
  EXPECT_EQ(data.subgraph_ids(0), 0);
  EXPECT_EQ(data.subgraph_ids(1), 0);
  EXPECT_EQ(data.subgraph_ids(2), 1);
  EXPECT_EQ(data.subgraph_ids(3), 1);

  auto host_codes =
    raft::make_host_matrix<std::uint8_t, int64_t>(4, data.dataset->encoded_row_length());
  raft::copy(host_codes.data_handle(),
             data.dataset->data.data_handle(),
             host_codes.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  for (int64_t row = 0; row < host_codes.extent(0); ++row) {
    EXPECT_EQ(host_codes(row, 2), 0);
    EXPECT_EQ(host_codes(row, 3), 0);
  }
}

TEST(FlowannSerialize, RejectsTruncatedLegacyVpq)
{
  raft::resources res;
  auto stream = make_legacy_dataset(res, true);
  EXPECT_ANY_THROW(flowann::detail::deserialize_legacy_vpq_f16<float>(res, stream));
}

TEST(FlowannSerialize, RejectsOutOfRangeLegacyVqLabel)
{
  raft::resources res;
  auto stream = make_legacy_dataset(res, false, true);
  EXPECT_ANY_THROW(flowann::detail::deserialize_legacy_vpq_f16<float>(res, stream));
}

TEST(FlowannSerialize, LoadsLegacyTieredGraph)
{
  raft::resources res;
  auto stream = make_legacy_graph(res);
  flowann::device_padded_index<float> index(res);
  flowann::detail::deserialize_legacy_graph(res, stream, &index);
  EXPECT_EQ(index.size(), 4);
  EXPECT_EQ(index.graph_degree(), 2);
  EXPECT_EQ(index.node_per_cacheline(), 2);
  EXPECT_EQ(index.n_bits(), 2);
  ASSERT_TRUE(index.source_indices().has_value());
}

TEST(FlowannSerialize, LoadsLegacyCombinedV5Index)
{
  raft::resources res;
  auto stream = make_legacy_combined_vpq(res);
  auto bundle = flowann::deserialize_vpq_f16<float>(res, stream);
  raft::resource::sync_stream(res);

  EXPECT_EQ(bundle.index.size(), 4);
  EXPECT_EQ(bundle.index.dim(), 4);
  EXPECT_EQ(bundle.index.graph_degree(), 2);
  EXPECT_EQ(bundle.index.node_per_cacheline(), 2);
  EXPECT_EQ(bundle.index.n_bits(), 2);
  ASSERT_TRUE(bundle.index.source_indices().has_value());
  ASSERT_TRUE(bundle.index.subgraph_offsets().has_value());
  ASSERT_TRUE(bundle.index.subgraph_ids().has_value());
  EXPECT_EQ(bundle.index.subgraph_offsets()->extent(0), 2);
  EXPECT_EQ(bundle.index.subgraph_ids()->extent(0), 4);
}

TEST(FlowannSerialize, RoundTripsCurrentVpqFormat)
{
  raft::resources res;
  constexpr std::uint32_t rows       = 4;
  constexpr std::uint32_t dim        = 4;
  constexpr std::uint32_t vq_centers = 2;
  constexpr std::uint32_t pq_centers = 256;
  constexpr std::uint32_t pq_len     = 2;
  constexpr std::uint32_t row_length = 8;

  auto host_vq      = raft::make_host_matrix<half, int64_t>(vq_centers, dim);
  auto host_pq      = raft::make_host_matrix<half, int64_t>(pq_centers, pq_len);
  auto host_encoded = raft::make_host_matrix<std::uint8_t, int64_t>(rows, row_length);
  std::fill_n(host_vq.data_handle(), host_vq.size(), __float2half(0.0f));
  std::fill_n(host_pq.data_handle(), host_pq.size(), __float2half(0.0f));
  std::fill_n(host_encoded.data_handle(), host_encoded.size(), std::uint8_t{0});
  host_vq(1, 0) = __float2half(10.0f);
  host_pq(1, 0) = __float2half(1.0f);
  for (std::uint32_t row = 0; row < rows; ++row) {
    host_encoded(row, 0) = static_cast<std::uint8_t>(row / 2);
    host_encoded(row, 4) = static_cast<std::uint8_t>(row % 2);
  }

  auto device_vq      = raft::make_device_matrix<half, std::uint32_t>(res, vq_centers, dim);
  auto device_pq      = raft::make_device_matrix<half, std::uint32_t>(res, pq_centers, pq_len);
  auto device_encoded = raft::make_device_matrix<std::uint8_t, int64_t>(res, rows, row_length);
  auto const stream   = raft::resource::get_cuda_stream(res);
  raft::copy(device_vq.data_handle(), host_vq.data_handle(), host_vq.size(), stream);
  raft::copy(device_pq.data_handle(), host_pq.data_handle(), host_pq.size(), stream);
  raft::copy(device_encoded.data_handle(), host_encoded.data_handle(), host_encoded.size(), stream);
  auto dataset = std::make_unique<cuvs::neighbors::device_vpq_dataset<half, int64_t>>(
    std::move(device_vq), std::move(device_pq), std::move(device_encoded));
  flowann::vpq_f16_index<float> index(
    res, cuvs::distance::DistanceType::L2Expanded, dataset->as_dataset_view(), 2, 2);

  auto graph = install_compact_test_graph(res, index);

  auto source_indices = raft::make_host_vector<std::uint32_t, int64_t>(rows);
  auto subgraph_ids   = raft::make_host_vector<std::uint32_t, int64_t>(rows);
  for (std::uint32_t row = 0; row < rows; ++row) {
    source_indices(row) = rows - row - 1;
    subgraph_ids(row)   = row / 2;
  }
  index.update_source_indices(res, raft::make_const_mdspan(source_indices.view()));
  auto offsets = raft::make_host_vector<std::uint32_t, int64_t>(2);
  offsets(0)   = 0;
  offsets(1)   = 2;
  index.update_subgraph_layout(
    res, raft::make_const_mdspan(offsets.view()), raft::make_const_mdspan(subgraph_ids.view()));
  auto seeds = raft::make_host_vector<std::uint32_t, int64_t>(2);
  seeds(0)   = 1;
  seeds(1)   = 2;
  index.update_seeds(res, raft::make_const_mdspan(seeds.view()));

  std::stringstream serialized(std::ios::in | std::ios::out | std::ios::binary);
  flowann::serialize(res, serialized, index);
  serialized.seekg(0);
  auto restored = flowann::deserialize_vpq_f16<float>(res, serialized);
  raft::resource::sync_stream(res);

  EXPECT_EQ(restored.index.size(), rows);
  EXPECT_EQ(restored.index.dim(), dim);
  EXPECT_EQ(restored.index.graph_degree(), 2);
  EXPECT_EQ(restored.index.resident_degree(), 1);
  EXPECT_EQ(restored.index.cross_degree(), 1);
  EXPECT_EQ(restored.index.metric(), cuvs::distance::DistanceType::L2Expanded);
  EXPECT_EQ(restored.index.node_per_cacheline(), 2);
  EXPECT_EQ(restored.index.n_bits(), 2);
  ASSERT_TRUE(restored.index.source_indices().has_value());
  ASSERT_TRUE(restored.index.subgraph_offsets().has_value());
  ASSERT_TRUE(restored.index.subgraph_ids().has_value());
  ASSERT_TRUE(restored.index.seeds().has_value());
  EXPECT_EQ(restored.index.source_indices()->extent(0), rows);
  EXPECT_EQ(restored.index.subgraph_offsets()->extent(0), 2);
  EXPECT_EQ(restored.index.subgraph_ids()->extent(0), rows);
  EXPECT_EQ(restored.index.seeds()->extent(0), 2);

  auto restored_inner          = raft::make_host_matrix<std::uint8_t, int64_t>(2, 3);
  auto restored_source_indices = raft::make_host_vector<std::uint32_t, int64_t>(rows);
  auto restored_offsets        = raft::make_host_vector<std::uint32_t, int64_t>(2);
  auto restored_subgraph_ids   = raft::make_host_vector<std::uint32_t, int64_t>(rows);
  auto restored_seeds          = raft::make_host_vector<std::uint32_t, int64_t>(2);
  auto restored_vq             = raft::make_host_matrix<half, int64_t>(vq_centers, dim);
  auto restored_pq             = raft::make_host_matrix<half, int64_t>(pq_centers, pq_len);
  auto restored_encoded        = raft::make_host_matrix<std::uint8_t, int64_t>(rows, row_length);
  raft::copy(restored_inner.data_handle(),
             restored.index.inner_graph().data_handle(),
             restored_inner.size(),
             stream);
  raft::copy(restored_source_indices.data_handle(),
             restored.index.source_indices()->data_handle(),
             restored_source_indices.size(),
             stream);
  raft::copy(restored_offsets.data_handle(),
             restored.index.subgraph_offsets()->data_handle(),
             restored_offsets.size(),
             stream);
  raft::copy(restored_subgraph_ids.data_handle(),
             restored.index.subgraph_ids()->data_handle(),
             restored_subgraph_ids.size(),
             stream);
  raft::copy(restored_seeds.data_handle(),
             restored.index.seeds()->data_handle(),
             restored_seeds.size(),
             stream);
  raft::copy(restored_vq.data_handle(),
             restored.dataset->vq_code_book.data_handle(),
             restored_vq.size(),
             stream);
  raft::copy(restored_pq.data_handle(),
             restored.dataset->pq_code_book.data_handle(),
             restored_pq.size(),
             stream);
  raft::copy(restored_encoded.data_handle(),
             restored.dataset->data.data_handle(),
             restored_encoded.size(),
             stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(std::memcmp(graph.inner.data_handle(),
                        restored_inner.data_handle(),
                        graph.inner.size() * sizeof(std::uint8_t)),
            0);
  EXPECT_EQ(std::memcmp(graph.cross.data_handle(),
                        restored.index.cross_graph().data_handle(),
                        graph.cross.size() * sizeof(std::uint32_t)),
            0);
  EXPECT_EQ(std::memcmp(source_indices.data_handle(),
                        restored_source_indices.data_handle(),
                        source_indices.size() * sizeof(std::uint32_t)),
            0);
  EXPECT_EQ(std::memcmp(offsets.data_handle(),
                        restored_offsets.data_handle(),
                        offsets.size() * sizeof(std::uint32_t)),
            0);
  EXPECT_EQ(std::memcmp(subgraph_ids.data_handle(),
                        restored_subgraph_ids.data_handle(),
                        subgraph_ids.size() * sizeof(std::uint32_t)),
            0);
  EXPECT_EQ(
    std::memcmp(
      seeds.data_handle(), restored_seeds.data_handle(), seeds.size() * sizeof(std::uint32_t)),
    0);
  EXPECT_EQ(
    std::memcmp(host_vq.data_handle(), restored_vq.data_handle(), host_vq.size() * sizeof(half)),
    0);
  EXPECT_EQ(
    std::memcmp(host_pq.data_handle(), restored_pq.data_handle(), host_pq.size() * sizeof(half)),
    0);
  EXPECT_EQ(std::memcmp(host_encoded.data_handle(),
                        restored_encoded.data_handle(),
                        host_encoded.size() * sizeof(std::uint8_t)),
            0);
}

TEST(FlowannSerialize, RoundTripsCompactDegreeMetadata)
{
  raft::resources res;
  auto host_dataset = raft::make_host_matrix<float, int64_t>(4, 4);
  std::fill_n(host_dataset.data_handle(), host_dataset.size(), 0.0f);
  auto dataset = cuvs::neighbors::make_device_padded_dataset(res, host_dataset.view());
  flowann::device_padded_index<float> index(
    res, cuvs::distance::DistanceType::L2Expanded, dataset->as_dataset_view(), 2, 2);

  install_compact_test_graph(res, index);
  auto offsets = raft::make_host_vector<std::uint32_t, int64_t>(1);
  auto ids     = raft::make_host_vector<std::uint32_t, int64_t>(4);
  offsets(0)   = 0;
  std::fill_n(ids.data_handle(), ids.size(), std::uint32_t{0});
  index.update_subgraph_layout(
    res, raft::make_const_mdspan(offsets.view()), raft::make_const_mdspan(ids.view()));

  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  flowann::serialize(res, stream, index);
  stream.seekg(0);
  auto restored = flowann::deserialize_device_padded<float>(res, stream);

  EXPECT_EQ(restored.index.graph_degree(), 2);
  EXPECT_EQ(restored.index.resident_degree(), 1);
  EXPECT_EQ(restored.index.cross_degree(), 1);
  EXPECT_EQ(restored.index.cross_graph().extent(1), 2);
}

TEST(FlowannSerialize, LoadsPreviousSelfContainedFormat)
{
  raft::resources res;
  constexpr std::uint32_t rows       = 4;
  constexpr std::uint32_t dim        = 4;
  constexpr std::uint32_t degree     = 2;
  constexpr std::uint16_t nodes_per  = 2;
  constexpr std::uint16_t index_bits = 2;
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  write_dtype_prefix<float>(stream);
  raft::serialize_scalar(res, stream, flowann::detail::previous_serialization_version);
  raft::serialize_scalar(
    res, stream, static_cast<std::uint32_t>(flowann::detail::serialized_dataset_kind::dense));
  raft::serialize_scalar(res, stream, rows);
  raft::serialize_scalar(res, stream, dim);
  raft::serialize_scalar(res, stream, degree);
  raft::serialize_scalar(res, stream, cuvs::distance::DistanceType::L2Expanded);
  raft::serialize_scalar(res, stream, nodes_per);
  raft::serialize_scalar(res, stream, index_bits);
  raft::serialize_scalar(res, stream, std::uint32_t{3});
  raft::serialize_scalar(res, stream, std::uint32_t{1});
  raft::serialize_scalar(res, stream, std::uint32_t{0});
  raft::serialize_scalar(res, stream, std::uint32_t{0});

  auto inner = raft::make_host_matrix<std::uint8_t, int64_t>(2, 3);
  std::fill_n(inner.data_handle(), inner.size(), std::uint8_t{0});
  inner(0, 0) = 1;
  inner(0, 1) = 1;
  inner(1, 0) = 1;
  inner(1, 1) = 1;
  auto cross  = raft::make_host_matrix<std::uint32_t, int64_t>(rows, degree + 1);
  for (std::uint32_t row = 0; row < rows; ++row) {
    cross(row, 0) = 1;
    cross(row, 1) = (row + 1) % rows;
    cross(row, 2) = 0;
  }
  auto offsets = raft::make_host_vector<std::uint32_t, int64_t>(1);
  auto ids     = raft::make_host_vector<std::uint32_t, int64_t>(rows);
  offsets(0)   = 0;
  std::fill_n(ids.data_handle(), ids.size(), std::uint32_t{0});
  raft::serialize_mdspan(res, stream, inner.view());
  raft::serialize_mdspan(res, stream, cross.view());
  raft::serialize_mdspan(res, stream, offsets.view());
  raft::serialize_mdspan(res, stream, ids.view());

  auto dataset = raft::make_host_matrix<float, int64_t>(rows, dim);
  std::fill_n(dataset.data_handle(), dataset.size(), 0.0f);
  raft::serialize_scalar(res, stream, dim);
  raft::serialize_mdspan(res, stream, dataset.view());
  stream.seekg(0);

  auto restored = flowann::deserialize_device_padded<float>(res, stream);
  EXPECT_EQ(restored.index.graph_degree(), degree);
  EXPECT_EQ(restored.index.resident_degree(), 0);
  EXPECT_EQ(restored.index.cross_degree(), degree);
}

TEST(FlowannSerialize, SearchesLegacyCombinedVpqWithAndWithoutFixedSeeds)
{
  raft::resources res;
  auto combined_stream = make_legacy_combined_vpq(res);
  auto bundle          = flowann::deserialize_vpq_f16<float>(res, combined_stream);
  auto& index          = bundle.index;

  auto host_queries = raft::make_host_matrix<float, int64_t>(2, 4);
  std::fill_n(host_queries.data_handle(), host_queries.size(), 0.0f);
  host_queries(0, 0) = 0.1f;
  host_queries(1, 0) = 10.9f;
  auto queries       = raft::make_device_matrix<float, int64_t>(res, 2, 4);
  raft::copy(queries.data_handle(),
             host_queries.data_handle(),
             host_queries.size(),
             raft::resource::get_cuda_stream(res));
  auto neighbors = raft::make_device_matrix<std::uint32_t, int64_t>(res, 2, 1);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 2, 1);
  flowann::search_context context(index.cross_graph());

  flowann::search_params params{};
  params.algo           = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
  params.itopk_size     = 32;
  params.search_width   = 1;
  params.max_iterations = 8;
  params.max_queries    = 2;
  params.smem_dtype     = cuvs::neighbors::cagra::internal_dtype::F16;

  for (bool use_seeds : {false, true}) {
    if (use_seeds) {
      auto seeds = raft::make_host_vector<std::uint32_t, int64_t>(2);
      seeds(0)   = 1;
      seeds(1)   = 2;
      index.update_seeds(
        res, raft::make_host_vector_view<const std::uint32_t, int64_t>(seeds.data_handle(), 2));
      params.num_seeds = 2;
    }

    flowann::search(res,
                    params,
                    index,
                    context,
                    raft::make_const_mdspan(queries.view()),
                    neighbors.view(),
                    distances.view());
    auto host_neighbors = raft::make_host_matrix<std::uint32_t, int64_t>(2, 1);
    auto host_distances = raft::make_host_matrix<float, int64_t>(2, 1);
    raft::copy(host_neighbors.data_handle(),
               neighbors.data_handle(),
               neighbors.size(),
               raft::resource::get_cuda_stream(res));
    raft::copy(host_distances.data_handle(),
               distances.data_handle(),
               distances.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    EXPECT_EQ(host_neighbors(0, 0), 3);
    EXPECT_EQ(host_neighbors(1, 0), 0);
    EXPECT_NEAR(host_distances(0, 0), 0.01f, 1e-3f);
    EXPECT_NEAR(host_distances(1, 0), 0.01f, 1e-3f);
  }
}

}  // namespace
