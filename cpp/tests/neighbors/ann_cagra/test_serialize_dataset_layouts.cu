/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../../../src/neighbors/detail/dataset_serialize.hpp"
#include "../cagra_padded_build_helpers.cuh"
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra {

/**
 * The serialized dataset payload holds only the logical rows. Loading it into a densely packed
 * (standard) dataset must allocate and fill exactly rows * dim elements, and loading it into a
 * padded dataset must restore the zero padding. The dimension is chosen so that the padded row
 * stride is greater than the dimension.
 */
class cagra_serialize_dataset_layouts_test : public ::testing::Test {
 protected:
  using data_type  = float;
  using index_type = uint32_t;

  static constexpr int64_t n_rows = 513;
  static constexpr int64_t n_dim  = 7;

  void SetUp() override
  {
    dataset_ = std::make_unique<raft::device_matrix<data_type, int64_t>>(
      raft::make_device_matrix<data_type, int64_t>(res_, n_rows, n_dim));
    raft::random::RngState rng(42ULL);
    raft::random::uniform(
      res_, rng, dataset_->data_handle(), n_rows * n_dim, data_type(-1), data_type(1));
    host_rows_.resize(n_rows * n_dim);
    raft::copy(host_rows_.data(),
               dataset_->data_handle(),
               host_rows_.size(),
               raft::resource::get_cuda_stream(res_));
    raft::resource::sync_stream(res_);
  }

  auto serialize_padded_index(std::ostream& os)
  {
    cagra::index_params params;
    params.graph_degree              = 16;
    params.intermediate_graph_degree = 32;
    cuvs::neighbors::test::padded_device_matrix_for_cagra<data_type> padded(
      res_, raft::make_const_mdspan(dataset_->view()));
    auto const stride = static_cast<int64_t>(padded.view.data_const_view().stride(0));
    EXPECT_GT(stride, n_dim);
    auto index = cagra::build(res_, params, padded.view);
    cagra::serialize(res_, os, index, true);
    raft::resource::sync_stream(res_);
    return stride;
  }

  template <typename ViewT>
  void check_rows(ViewT const& view, int64_t expected_stride)
  {
    auto const data = view.data_const_view();
    ASSERT_EQ(data.extent(0), n_rows);
    ASSERT_EQ(data.extent(1), n_dim);
    ASSERT_EQ(static_cast<int64_t>(data.stride(0)), expected_stride);
    std::vector<data_type> loaded(n_rows * expected_stride);
    raft::copy(
      loaded.data(), data.data_handle(), loaded.size(), raft::resource::get_cuda_stream(res_));
    raft::resource::sync_stream(res_);
    for (int64_t i = 0; i < n_rows; i++) {
      for (int64_t j = 0; j < expected_stride; j++) {
        auto const expected = j < n_dim ? host_rows_[i * n_dim + j] : data_type(0);
        ASSERT_EQ(loaded[i * expected_stride + j], expected) << "row " << i << ", column " << j;
      }
    }
  }

  raft::resources res_;
  std::unique_ptr<raft::device_matrix<data_type, int64_t>> dataset_;
  std::vector<data_type> host_rows_;
};

TEST_F(cagra_serialize_dataset_layouts_test, load_into_standard_dataset)
{
  cagra::index_params params;
  params.graph_degree              = 16;
  params.intermediate_graph_degree = 32;
  auto standard_view =
    cuvs::neighbors::make_device_standard_dataset_view(raft::make_const_mdspan(dataset_->view()));
  auto standard_index = cagra::build(res_, params, standard_view);
  std::stringstream ss;
  cagra::serialize(res_, ss, standard_index, true);
  raft::resource::sync_stream(res_);

  cagra::device_standard_index<data_type, index_type> index(res_);
  std::unique_ptr<cuvs::neighbors::device_standard_dataset<data_type, int64_t>> loaded;
  cagra::deserialize(res_, ss, &index, &loaded);
  ASSERT_NE(loaded, nullptr);
  check_rows(loaded->view(), n_dim);
}

TEST_F(cagra_serialize_dataset_layouts_test, load_wide_stride_payload_into_standard_dataset)
{
  // Files written from a dataset with a wider row pitch carry that pitch as a scalar. A standard
  // dataset is densely packed, so it has to ignore the pitch and read exactly rows * dim elements.
  cuvs::neighbors::test::padded_device_matrix_for_cagra<data_type> padded(
    res_, raft::make_const_mdspan(dataset_->view()));
  auto const stride = static_cast<int64_t>(padded.view.data_const_view().stride(0));
  ASSERT_GT(stride, n_dim);

  std::stringstream ss;
  cuvs::neighbors::detail::serialize_cagra_dense_dataset<data_type, int64_t>(res_, ss, padded.view);
  raft::resource::sync_stream(res_);

  auto loaded = cuvs::neighbors::detail::deserialize_standard_dataset<data_type, int64_t>(res_, ss);
  ASSERT_NE(loaded, nullptr);
  check_rows(loaded->view(), n_dim);
}

TEST_F(cagra_serialize_dataset_layouts_test, load_into_padded_dataset)
{
  std::stringstream ss;
  auto const stride = serialize_padded_index(ss);

  cagra::device_padded_index<data_type, index_type> index(res_);
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<data_type, int64_t>> loaded;
  cagra::deserialize(res_, ss, &index, &loaded);
  ASSERT_NE(loaded, nullptr);
  check_rows(loaded->view(), stride);
}

TEST_F(cagra_serialize_dataset_layouts_test, standard_view_requires_dense_rows)
{
  auto const stride  = 8;
  auto padded_buffer = raft::make_device_matrix<data_type, int64_t>(res_, n_rows, stride);
  auto strided_view  = raft::make_device_strided_matrix_view<const data_type, int64_t>(
    padded_buffer.data_handle(), n_rows, n_dim, stride);
  ASSERT_THROW(cuvs::neighbors::make_device_standard_dataset_view(strided_view), raft::exception);

  auto dense_view = raft::make_device_strided_matrix_view<const data_type, int64_t>(
    padded_buffer.data_handle(), n_rows, n_dim, n_dim);
  ASSERT_NO_THROW(cuvs::neighbors::make_device_standard_dataset_view(dense_view));
}

}  // namespace cuvs::neighbors::cagra
