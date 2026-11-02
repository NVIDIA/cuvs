/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/preprocessing/quantize/scalar.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/reduce.cuh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cub/device/device_merge_sort.cuh>

namespace cuvs::preprocessing::quantize::scalar::detail {

template <class T>
_RAFT_HOST_DEVICE bool fp_lt(const T& a, const T& b)
{
  return a < b;
}

template <>
_RAFT_HOST_DEVICE bool fp_lt(const half& a, const half& b)
{
  return static_cast<float>(a) < static_cast<float>(b);
}

template <class T>
struct fp_lt_op {
  _RAFT_HOST_DEVICE bool operator()(const T& a, const T& b) const { return fp_lt(a, b); }
};

template <typename T, typename QuantI, typename TempT = double>
struct quantize_op {
  const T min_;
  const T max_;
  const QuantI q_type_min_ = std::numeric_limits<QuantI>::min();
  const QuantI q_type_max_ = std::numeric_limits<QuantI>::max();
  const TempT scalar_;
  const TempT offset_;

  constexpr explicit quantize_op(T min, T max)
    : min_(min),
      max_(max),
      scalar_(static_cast<TempT>(max_) > static_cast<TempT>(min_)
                ? ((static_cast<TempT>(q_type_max_) - static_cast<TempT>(q_type_min_)) /
                   (static_cast<TempT>(max_) - static_cast<TempT>(min_)))
                : static_cast<TempT>(1)),
      offset_(static_cast<TempT>(q_type_min_) - static_cast<TempT>(min_) * scalar_)
  {
  }

  constexpr RAFT_INLINE_FUNCTION QuantI operator()(const T& x) const
  {
    if (!fp_lt(min_, x)) return q_type_min_;
    if (!fp_lt(x, max_)) return q_type_max_;
    return static_cast<QuantI>(lroundf(scalar_ * static_cast<TempT>(x) + offset_));
  }

  constexpr RAFT_INLINE_FUNCTION T operator()(const QuantI& x) const
  {
    return static_cast<T>((static_cast<TempT>(x) - offset_) / scalar_);
  }
};

template <typename T, typename IdxT = int64_t, typename accessor>
std::tuple<T, T> quantile_min_max(
  raft::resources const& res,
  raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, accessor> dataset,
  double quantile)
{
  // settings for quantile approximation
  constexpr size_t max_num_samples = 1000000;
  constexpr int seed               = 137;

  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();

  // select subsample
  raft::random::RngState rng(seed);
  size_t n_rows        = dataset.extent(0);
  size_t dim           = dataset.extent(1);
  size_t n_sample_rows = std::min<size_t>(std::ceil(max_num_samples / dim), n_rows);

  // select subsample rows (this returns device data for both device and host input)
  auto subset = raft::matrix::sample_rows(res, rng, dataset, (IdxT)n_sample_rows);

  // quantile / sort element-wise and pick for now
  auto subset_size = static_cast<int64_t>(n_sample_rows * dim);

  // Sort through CUB rather than thrust so that the sort workspace is queried and allocated
  // explicitly: thrust::sort takes its temporary storage from the thrust policy, which the
  // dry-run tracker would not see once the sort itself is skipped.
  size_t sort_ws_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceMergeSort::SortKeys(
    nullptr, sort_ws_bytes, subset.data_handle(), subset_size, fp_lt_op<T>{}, stream));
  auto sort_ws = raft::make_device_vector<char, int64_t>(res, sort_ws_bytes);
  if (!raft::resource::get_dry_run_flag(res)) {
    RAFT_CUDA_TRY(cub::DeviceMergeSort::SortKeys(sort_ws.data_handle(),
                                                 sort_ws_bytes,
                                                 subset.data_handle(),
                                                 subset_size,
                                                 fp_lt_op<T>{},
                                                 stream));
  }

  double half_quantile_pos = (0.5 + 0.5 * quantile) * subset_size;
  int pos_max              = std::ceil(half_quantile_pos) - 1;
  int pos_min              = subset_size - pos_max - 1;

  // The copies below do not run in dry-run mode, so these placeholders are what the caller
  // receives; they keep quantize_op's scale and offset finite.
  T minmax_h[2] = {T(0), T(1)};
  raft::copy(res,
             raft::make_host_scalar_view(&minmax_h[0]),
             raft::make_device_scalar_view(subset.data_handle() + pos_min));
  raft::copy(res,
             raft::make_host_scalar_view(&minmax_h[1]),
             raft::make_device_scalar_view(subset.data_handle() + pos_max));
  raft::resource::sync_stream(res);
  return {minmax_h[0], minmax_h[1]};
}

template <typename T>
cuvs::preprocessing::quantize::scalar::quantizer<T> train(
  raft::resources const& res,
  const cuvs::preprocessing::quantize::scalar::params params,
  raft::device_matrix_view<const T, int64_t> dataset)
{
  RAFT_EXPECTS(params.quantile > 0.0 && params.quantile <= 1.0,
               "quantile for scalar quantization needs to be within (0, 1] but is %f",
               params.quantile);

  auto [min, max] = detail::quantile_min_max(res, dataset, params.quantile);

  RAFT_LOG_DEBUG("quantizer train min=%lf max=%lf.", double(min), double(max));

  return cuvs::preprocessing::quantize::scalar::quantizer<T>{min, max};
}

template <typename T>
cuvs::preprocessing::quantize::scalar::quantizer<T> train(
  raft::resources const& res,
  const cuvs::preprocessing::quantize::scalar::params params,
  raft::host_matrix_view<const T, int64_t> dataset)
{
  RAFT_EXPECTS(params.quantile > 0.0 && params.quantile <= 1.0,
               "quantile for scalar quantization needs to be within (0, 1] but is %f",
               params.quantile);

  auto [min, max] = detail::quantile_min_max(res, dataset, params.quantile);

  RAFT_LOG_DEBUG("quantizer train min=%lf max=%lf.", double(min), double(max));

  return cuvs::preprocessing::quantize::scalar::quantizer<T>{min, max};
}

template <typename T, typename QuantI = int8_t>
void transform(raft::resources const& res,
               const cuvs::preprocessing::quantize::scalar::quantizer<T>& quantizer,
               raft::device_matrix_view<const T, int64_t> dataset,
               raft::device_matrix_view<QuantI, int64_t> out)
{
  raft::linalg::map(res,
                    out,
                    quantize_op<T, QuantI>(quantizer.min_, quantizer.max_),
                    raft::make_const_mdspan(dataset));
}

template <typename T, typename QuantI = int8_t>
void transform(raft::resources const& res,
               const cuvs::preprocessing::quantize::scalar::quantizer<T>& quantizer,
               raft::host_matrix_view<const T, int64_t> dataset,
               raft::host_matrix_view<QuantI, int64_t> out)
{
  auto main_op      = quantize_op<T, QuantI>(quantizer.min_, quantizer.max_);
  size_t n_elements = dataset.extent(0) * dataset.extent(1);

  // In dry-run mode the views may be unbacked or alias the shared probe buffer.
  if (raft::resource::get_dry_run_flag(res)) { return; }

#pragma omp parallel for
  for (size_t i = 0; i < n_elements; ++i) {
    out.data_handle()[i] = main_op(dataset.data_handle()[i]);
  }
}

template <typename T, typename QuantI = int8_t>
void inverse_transform(raft::resources const& res,
                       const cuvs::preprocessing::quantize::scalar::quantizer<T>& quantizer,
                       raft::device_matrix_view<const QuantI, int64_t> dataset,
                       raft::device_matrix_view<T, int64_t> out)
{
  raft::linalg::map(res,
                    out,
                    quantize_op<T, QuantI>(quantizer.min_, quantizer.max_),
                    raft::make_const_mdspan(dataset));
}

template <typename T, typename QuantI = int8_t>
void inverse_transform(raft::resources const& res,
                       const cuvs::preprocessing::quantize::scalar::quantizer<T>& quantizer,
                       raft::host_matrix_view<const QuantI, int64_t> dataset,
                       raft::host_matrix_view<T, int64_t> out)
{
  auto main_op      = quantize_op<T, QuantI>(quantizer.min_, quantizer.max_);
  size_t n_elements = dataset.extent(0) * dataset.extent(1);

  // In dry-run mode the views may be unbacked or alias the shared probe buffer.
  if (raft::resource::get_dry_run_flag(res)) { return; }

#pragma omp parallel for
  for (size_t i = 0; i < n_elements; ++i) {
    out.data_handle()[i] = main_op(dataset.data_handle()[i]);
  }
}

inline per_dim_quantizer train_per_dim(raft::resources const& res,
                                       raft::device_matrix_view<const float, int64_t> dataset,
                                       params params)
{
  RAFT_EXPECTS(params.quantile > 0.f && params.quantile <= 1.f, "Quantile must be within (0, 1]");
  auto const dim = dataset.extent(1);
  RAFT_EXPECTS(dataset.extent(0) > 0 && dim > 0, "Training requires nonempty rows and dimensions");
  auto offsets        = raft::make_device_vector<float, int64_t>(res, dim);
  auto maxima         = raft::make_device_vector<float, int64_t>(res, dim);
  auto max_half_range = raft::make_device_vector<float, int64_t>(res, 1);
  if (params.quantile == 1.f) {
    raft::linalg::reduce<raft::Apply::ALONG_COLUMNS>(res,
                                                     dataset,
                                                     offsets.view(),
                                                     std::numeric_limits<float>::infinity(),
                                                     false,
                                                     raft::identity_op{},
                                                     raft::min_op{});
    raft::linalg::reduce<raft::Apply::ALONG_COLUMNS>(res,
                                                     dataset,
                                                     maxima.view(),
                                                     -std::numeric_limits<float>::infinity(),
                                                     false,
                                                     raft::identity_op{},
                                                     raft::max_op{});
  } else {
    auto const rows        = dataset.extent(0);
    auto column            = raft::make_device_vector<float, int64_t>(res, rows);
    auto stream            = raft::resource::get_cuda_stream(res).get();
    size_t workspace_bytes = 0;
    RAFT_CUDA_TRY(cub::DeviceMergeSort::SortKeys(
      nullptr, workspace_bytes, column.data_handle(), rows, fp_lt_op<float>{}, stream));
    auto workspace    = raft::make_device_vector<char, int64_t>(res, workspace_bytes);
    auto const upper  = static_cast<int64_t>(std::ceil((0.5 + 0.5 * params.quantile) * rows)) - 1;
    auto const lower  = rows - upper - 1;
    auto const* input = dataset.data_handle();
    for (int64_t j = 0; j < dim; ++j) {
      raft::linalg::map_offset(
        res, column.view(), [input, dim, j] __device__(int64_t i) { return input[i * dim + j]; });
      if (!raft::resource::get_dry_run_flag(res)) {
        RAFT_CUDA_TRY(cub::DeviceMergeSort::SortKeys(workspace.data_handle(),
                                                     workspace_bytes,
                                                     column.data_handle(),
                                                     rows,
                                                     fp_lt_op<float>{},
                                                     stream));
      }
      raft::copy(res,
                 raft::make_device_scalar_view(offsets.data_handle() + j),
                 raft::make_device_scalar_view(column.data_handle() + lower));
      raft::copy(res,
                 raft::make_device_scalar_view(maxima.data_handle() + j),
                 raft::make_device_scalar_view(column.data_handle() + upper));
    }
  }
  auto* offset_ptr    = offsets.data_handle();
  auto const* max_ptr = maxima.data_handle();
  raft::linalg::map_offset(res, maxima.view(), [offset_ptr, max_ptr] __device__(int64_t j) {
    // Halve before adding/subtracting to avoid overflow for large finite inputs.
    float const lo = offset_ptr[j] * 0.5f;
    float const hi = max_ptr[j] * 0.5f;
    offset_ptr[j]  = lo + hi;
    return hi - lo;
  });
  raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
    res,
    raft::make_device_matrix_view<const float, int64_t>(maxima.data_handle(), 1, dim),
    max_half_range.view(),
    0.f,
    false,
    raft::identity_op{},
    raft::max_op{});
  float half_range = 0.f;
  raft::copy(res,
             raft::make_host_scalar_view(&half_range),
             raft::make_device_scalar_view(max_half_range.data_handle()));
  raft::resource::sync_stream(res);
  float scale =
    half_range > 0.f ? std::max(half_range / 127.f, std::numeric_limits<float>::min()) : 1.f;
  // Avoid upward division rounding overflowing the largest reconstructed magnitude.
  if (static_cast<double>(scale) * 127.0 > std::numeric_limits<float>::max()) {
    scale = std::nextafter(scale, 0.f);
  }
  return {std::move(offsets), scale};
}

template <typename InT, typename OutT>
void check_per_dim_shapes(per_dim_quantizer_const_view quantizer,
                          raft::device_matrix_view<const InT, int64_t> dataset,
                          raft::device_matrix_view<OutT, int64_t> out)
{
  RAFT_EXPECTS(
    std::isfinite(quantizer.scale) && quantizer.scale > 0.f && std::isfinite(1.f / quantizer.scale),
    "Scale and its reciprocal must be finite and positive");
  RAFT_EXPECTS(dataset.extent(1) > 0 && dataset.extent(1) == quantizer.offsets.extent(0),
               "Input width must match the nonempty offsets vector");
  RAFT_EXPECTS(dataset.extent(0) == out.extent(0) && dataset.extent(1) == out.extent(1),
               "Input and output shapes must match");
}

inline void transform(raft::resources const& res,
                      per_dim_quantizer_const_view quantizer,
                      raft::device_matrix_view<const float, int64_t> dataset,
                      raft::device_matrix_view<int8_t, int64_t> out)
{
  check_per_dim_shapes(quantizer, dataset, out);
  auto const* input   = dataset.data_handle();
  auto const* offsets = quantizer.offsets.data_handle();
  auto const dim      = dataset.extent(1);
  float const inv     = 1.f / quantizer.scale;
  raft::linalg::map_offset(res, out, [input, offsets, dim, inv] __device__(int64_t i) {
    float const code = rintf((input[i] - offsets[i % dim]) * inv);
    return static_cast<int8_t>(fminf(fmaxf(code, -127.f), 127.f));
  });
}

inline void inverse_transform(raft::resources const& res,
                              per_dim_quantizer_const_view quantizer,
                              raft::device_matrix_view<const int8_t, int64_t> dataset,
                              raft::device_matrix_view<float, int64_t> out)
{
  check_per_dim_shapes(quantizer, dataset, out);
  auto const* input   = dataset.data_handle();
  auto const* offsets = quantizer.offsets.data_handle();
  auto const dim      = dataset.extent(1);
  float const scale   = quantizer.scale;
  raft::linalg::map_offset(res, out, [input, offsets, dim, scale] __device__(int64_t i) {
    return fmaf(scale, static_cast<float>(input[i]), offsets[i % dim]);
  });
}

}  // namespace cuvs::preprocessing::quantize::scalar::detail
