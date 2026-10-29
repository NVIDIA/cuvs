/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/dataset/affine.hpp>
#include <cuvs/core/dataset/bbq.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <array>
#include <type_traits>
#include <utility>

namespace cuvs {

namespace {

constexpr int64_t kRows = 7;
constexpr int64_t kDim  = 32;

using device_affine_t = device_affine_dataset<float, int64_t>;
using host_affine_t   = host_affine_dataset<float, int64_t>;
using device_bbq_t    = device_bbq_dataset<float, int64_t>;

static_assert(!device_affine_t::is_compound, "a dense affine dataset has no manifest");
static_assert(device_bbq_t::is_compound, "BBQ keeps the centroid and the layout in its manifest");
static_assert(neighbors::dataset_view_is_device_accessible_v<device_affine_t::view_type>);
static_assert(neighbors::dataset_view_is_device_accessible_v<device_affine_t::const_view_type>);
static_assert(!neighbors::dataset_view_is_device_accessible_v<host_affine_t::const_view_type>);
static_assert(neighbors::is_device_bbq_dataset_view_v<device_bbq_t::const_view_type>);
static_assert(!neighbors::is_bbq_dataset_view_v<device_affine_t::const_view_type>);
static_assert(
  std::is_constructible_v<device_affine_t::const_view_type, device_affine_t::view_type>);
static_assert(std::is_constructible_v<device_bbq_t::const_view_type, device_bbq_t::view_type>);
static_assert(
  std::is_same_v<device_bbq_t::manifest_type, preprocessing::quantize::bbq::quantizer<float>>);
static_assert(std::is_convertible_v<preprocessing::quantize::bbq::quantizer_view<float>,
                                    preprocessing::quantize::bbq::quantizer_view<const float>>);
static_assert(std::is_nothrow_move_constructible_v<device_affine_t>);
static_assert(std::is_nothrow_move_constructible_v<device_bbq_t>);
static_assert(std::is_trivially_copyable_v<device_bbq_t::const_view_type>,
              "kernels take the view by value");

__global__ void read_bbq_view(device_bbq_t::const_view_type view, float* out)
{
  out[0] = static_cast<float>(view.n_rows());
  out[1] = static_cast<float>(view.dim());
  out[2] = static_cast<float>(view.const_manifest_view().encoded_row_length());
  out[3] = view.const_manifest_view().centroid_norm_sq;
}

TEST(AffineDataset, OwningToViews)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  device_affine_t dataset{{raft::make_device_matrix<float, int64_t>(res, kRows, kDim),
                           raft::make_device_vector<float, int64_t>(res, kRows),
                           raft::make_device_vector<float, int64_t>(res, kRows)}};
  EXPECT_EQ(dataset.n_rows(), kRows);
  EXPECT_EQ(dataset.dim(), kDim);

  auto view = dataset.view();
  EXPECT_EQ(view.n_rows(), kRows);
  EXPECT_EQ(view.dim(), kDim);
  EXPECT_EQ(view.data_view().rows.data_handle(), dataset.data_view().rows.data_handle());
  EXPECT_EQ(view.data_view().scale.data_handle(), dataset.data_view().scale.data_handle());
  EXPECT_EQ(view.data_view().shift.data_handle(), dataset.data_view().shift.data_handle());
  EXPECT_NE(view.data_view().scale.data_handle(), view.data_view().shift.data_handle());
  EXPECT_EQ(view.data_view().scale.extent(0), kRows);
  EXPECT_EQ(view.data_view().shift.extent(0), kRows);

  // The read-only flavors: of the owning dataset, and the promotion of a mutable view.
  device_affine_t::const_view_type from_const_dataset = std::as_const(dataset).view();
  device_affine_t::const_view_type promoted           = view;
  for (const auto& const_view : {from_const_dataset, promoted}) {
    EXPECT_EQ(const_view.n_rows(), kRows);
    EXPECT_EQ(const_view.dim(), kDim);
    auto data = const_view.data_const_view();
    EXPECT_EQ(data.rows.data_handle(), dataset.data_view().rows.data_handle());
    EXPECT_EQ(data.scale.data_handle(), dataset.data_view().scale.data_handle());
    EXPECT_EQ(data.shift.data_handle(), dataset.data_view().shift.data_handle());
  }
  raft::resource::sync_stream(res);
  (void)stream;
}

TEST(AffineDataset, HostMemory)
{
  host_affine_t dataset{{raft::make_host_matrix<float, int64_t>(kRows, kDim),
                         raft::make_host_vector<float, int64_t>(kRows),
                         raft::make_host_vector<float, int64_t>(kRows)}};
  EXPECT_EQ(dataset.n_rows(), kRows);
  EXPECT_EQ(dataset.dim(), kDim);
  EXPECT_FALSE(neighbors::dataset_view_is_device_accessible_v<decltype(dataset.view())>);
}

// The BBQ codes are an `affine_data`: `scale` = dequant_delta and `shift` = additional_corrections,
// the per-row terms of the distance estimate; everything else the kernels read is next to them.
TEST(AffineDataset, BbqPlugsIntoAffine)
{
  raft::resources res;
  using layout_t = preprocessing::quantize::bbq::bbq_code_layout;
  const auto len = preprocessing::quantize::bbq::get_encoded_row_length(kDim, layout_t::packed_1b);
  auto make_vec  = [&](int64_t n) { return raft::make_device_vector<float, int64_t>(res, n); };

  device_bbq_t::data_type data{
    {raft::make_device_matrix<uint8_t, int64_t>(res, kRows, len), make_vec(kRows), make_vec(kRows)},
    make_vec(kRows),
    make_vec(kRows),
    make_vec(kRows)};
  device_bbq_t::manifest_type manifest{
    make_vec(kDim), layout_t::packed_1b, distance::DistanceType::L2Expanded, 2.5f};
  device_bbq_t dataset{std::move(data), std::move(manifest)};

  EXPECT_EQ(dataset.n_rows(), kRows);
  EXPECT_EQ(dataset.dim(), kDim);

  const auto view = std::as_const(dataset).view();
  EXPECT_EQ(view.n_rows(), kRows);
  EXPECT_EQ(view.dim(), kDim);
  EXPECT_EQ(view.const_manifest_view().dim(), static_cast<uint32_t>(kDim));
  EXPECT_EQ(view.const_manifest_view().encoded_row_length(), static_cast<uint32_t>(len));
  preprocessing::quantize::bbq::quantizer_view<const float> from_mutable = dataset.manifest_view();
  EXPECT_EQ(from_mutable.centroid.data_handle(), view.const_manifest_view().centroid.data_handle());

  // The view is what the kernels take: a trivially copyable pair of the data and the manifest.
  const auto d = view.data_const_view();
  const auto m = view.const_manifest_view();
  EXPECT_EQ(d.codes.rows.extent(1), static_cast<int64_t>(len));
  EXPECT_EQ(m.layout, layout_t::packed_1b);
  EXPECT_EQ(m.metric, distance::DistanceType::L2Expanded);
  EXPECT_EQ(m.centroid_norm_sq, 2.5f);

  auto out = raft::make_device_vector<float, int64_t>(res, 4);
  read_bbq_view<<<1, 1, 0, raft::resource::get_cuda_stream(res).get()>>>(view, out.data_handle());
  std::array<float, 4> host{};
  raft::copy(host.data(), out.data_handle(), host.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  EXPECT_EQ(host[0], static_cast<float>(kRows));
  EXPECT_EQ(host[1], static_cast<float>(kDim));
  EXPECT_EQ(host[2], static_cast<float>(len));
  EXPECT_EQ(host[3], 2.5f);
}

// RaBitQ keeps two affine groups in one dataset: the low-bit sign codes the GPU reads and the wider
// extension codes. The data of such a dataset is built from the same component twice.
TEST(AffineDataset, TwoAffineGroupsConvertToViews)
{
  using codes_t = raft::device_matrix<uint8_t, int64_t>;
  using vec_t   = raft::device_vector<float, int64_t>;
  struct two_groups {
    affine_data<codes_t, vec_t> sign;
    affine_data<codes_t, vec_t> ex;
  };

  raft::resources res;
  auto make_group = [&](int64_t len) {
    return affine_data<codes_t, vec_t>{raft::make_device_matrix<uint8_t, int64_t>(res, kRows, len),
                                       raft::make_device_vector<float, int64_t>(res, kRows),
                                       raft::make_device_vector<float, int64_t>(res, kRows)};
  };
  two_groups data{make_group(kDim / 8), make_group(kDim)};

  const auto sign = std::as_const(data).sign.const_view();
  const auto ex   = data.ex.view().const_view();
  EXPECT_EQ(sign.rows.data_handle(), data.sign.rows.data_handle());
  EXPECT_EQ(ex.shift.data_handle(), data.ex.shift.data_handle());
  EXPECT_EQ(ex.rows.extent(1), kDim);
}

}  // namespace

}  // namespace cuvs
