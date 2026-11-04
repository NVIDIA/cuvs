/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../cluster/detail/kmeans_batch_loader.cuh"
#include "../../../neighbors/detail/ann_utils.cuh"

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
#include <optional>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <type_traits>
#include <vector>

#include <cub/device/device_merge_sort.cuh>
#include <cuda/std/limits>

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

template <typename T, typename Op>
RAFT_INLINE_FUNCTION double reconstruction_error(T x, int64_t j, Op op)
{
  double const value = static_cast<double>(x);
  double decoded;
  if constexpr (std::is_same_v<Op, typename quantizer_const_view<T>::shifts_type>) {
    using compute_t         = typename quantizer_const_view<T>::scale_type;
    compute_t const shift   = static_cast<compute_t>(op.shifts.data_handle()[j]);
    compute_t const inv     = compute_t{1} / op.scale;
    compute_t const rounded = rint((static_cast<compute_t>(x) - shift) * inv);
    int8_t const code = static_cast<int8_t>(fmin(fmax(rounded, compute_t{-127}), compute_t{127}));
    compute_t const reconstructed = fma(op.scale, static_cast<compute_t>(code), shift);
    compute_t const limit         = static_cast<compute_t>(cuda::std::numeric_limits<T>::max());
    decoded = static_cast<double>(static_cast<T>(fmin(fmax(reconstructed, -limit), limit)));
  } else {
    decoded = static_cast<double>(op(op(x)));
  }
  return isfinite(value) && isfinite(decoded) ? fabs(value - decoded)
                                              : std::numeric_limits<double>::infinity();
}

template <typename T>
auto round_error_up(double error) -> typename quantizer<T>::scale_type
{
  using bound_t = typename quantizer<T>::scale_type;
  // Match Sirius's conservative margin, then round upwards in the public result type.
  auto const bound = static_cast<bound_t>(error * (1.0 + 1e-6));
  return error > 0 ? std::nextafter(bound, std::numeric_limits<bound_t>::infinity()) : bound;
}

template <typename T, typename Op>
void max_reconstruction_error(raft::resources const& res,
                              raft::device_matrix_view<const T, int64_t> dataset,
                              Op op,
                              raft::device_vector_view<double, int64_t> errors,
                              raft::device_vector_view<double, int64_t> maximum,
                              bool inplace)
{
  raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
    res,
    dataset,
    errors,
    0.0,
    false,
    [op] __device__(T x, int64_t j) { return reconstruction_error(x, j, op); },
    [] __device__(double a, double b) { return hypot(a, b); });
  raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
    res,
    raft::make_device_matrix_view<const double, int64_t>(errors.data_handle(), 1, errors.size()),
    maximum,
    0.0,
    inplace,
    raft::identity_op{},
    raft::max_op{});
}

template <typename T, typename Op>
auto max_reconstruction_error(raft::resources const& res,
                              raft::device_matrix_view<const T, int64_t> dataset,
                              Op op) -> typename quantizer<T>::scale_type
{
  auto errors  = raft::make_device_vector<double, int64_t>(res, dataset.extent(0));
  auto maximum = raft::make_device_vector<double, int64_t>(res, 1);
  max_reconstruction_error(res, dataset, op, errors.view(), maximum.view(), false);
  double result = std::numeric_limits<double>::infinity();
  raft::copy(res,
             raft::make_host_scalar_view(&result),
             raft::make_device_scalar_view(maximum.data_handle()));
  raft::resource::sync_stream(res);
  return round_error_up<T>(result);
}

template <typename T, typename Op>
auto max_reconstruction_error(raft::resources const& res,
                              raft::host_matrix_view<const T, int64_t> dataset,
                              Op op) -> typename quantizer<T>::scale_type
{
  if (raft::resource::get_dry_run_flag(res)) {
    return std::numeric_limits<typename quantizer<T>::scale_type>::infinity();
  }
  double maximum = 0;
#pragma omp parallel for reduction(max : maximum)
  for (int64_t row = 0; row < dataset.extent(0); ++row) {
    double error = 0;
    for (int64_t j = 0; j < dataset.extent(1); ++j) {
      error = std::hypot(error, reconstruction_error(dataset(row, j), j, op));
    }
    maximum = std::max(maximum, error);
  }
  return round_error_up<T>(maximum);
}

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
typename quantizer<T>::shifts_type train(raft::resources const& res,
                                         raft::device_vector<T, int64_t>&& shifts,
                                         raft::device_vector<T, int64_t>&& maxima)
{
  using compute_t     = typename quantizer_const_view<T>::scale_type;
  auto const dim      = shifts.extent(0);
  auto half_ranges    = raft::make_device_vector<compute_t, int64_t>(res, dim);
  auto max_half_range = raft::make_device_vector<compute_t, int64_t>(res, 1);
  auto* shift_ptr     = shifts.data_handle();
  auto const* max_ptr = maxima.data_handle();
  raft::linalg::map_offset(res, half_ranges.view(), [shift_ptr, max_ptr] __device__(int64_t j) {
    // Halve before adding/subtracting to avoid overflow for large finite inputs.
    if (!isfinite(static_cast<compute_t>(shift_ptr[j])) ||
        !isfinite(static_cast<compute_t>(max_ptr[j])) || fp_lt(max_ptr[j], shift_ptr[j])) {
      return std::numeric_limits<compute_t>::infinity();
    }
    compute_t const lo = static_cast<compute_t>(shift_ptr[j]);
    compute_t const hi = static_cast<compute_t>(max_ptr[j]);
    shift_ptr[j]       = static_cast<T>(lo == hi ? lo : lo * compute_t{0.5} + hi * compute_t{0.5});
    compute_t const shift = static_cast<compute_t>(shift_ptr[j]);
    return fmax(hi - shift, shift - lo);
  });
  raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
    res,
    raft::make_device_matrix_view<const compute_t, int64_t>(half_ranges.data_handle(), 1, dim),
    max_half_range.view(),
    compute_t{0},
    false,
    raft::identity_op{},
    raft::max_op{});
  compute_t half_range = 0;
  raft::copy(res,
             raft::make_host_scalar_view(&half_range),
             raft::make_device_scalar_view(max_half_range.data_handle()));
  raft::resource::sync_stream(res);
  RAFT_EXPECTS(std::isfinite(half_range), "Bounds must be finite with min <= max");
  compute_t const scale =
    half_range > 0 ? std::max(half_range / compute_t{127}, std::numeric_limits<compute_t>::min())
                   : compute_t{1};
  return {std::move(shifts), scale};
}

template <typename T>
quantizer<T> train(raft::resources const& res,
                   raft::device_matrix_view<const T, int64_t> dataset,
                   params params)
{
  using compute_t = typename quantizer_const_view<T>::scale_type;
  RAFT_EXPECTS(params.quantile > 0.f && params.quantile <= 1.f, "Quantile must be within (0, 1]");
  auto const dim = dataset.extent(1);
  RAFT_EXPECTS(dataset.extent(0) > 0 && dim > 0, "Training requires nonempty rows and dimensions");
  auto shifts = raft::make_device_vector<T, int64_t>(res, dim);
  auto maxima = raft::make_device_vector<T, int64_t>(res, dim);
  if (params.quantile == 1.f) {
    raft::linalg::reduce<raft::Apply::ALONG_COLUMNS>(
      res,
      dataset,
      shifts.view(),
      static_cast<T>(std::numeric_limits<compute_t>::infinity()),
      false,
      raft::identity_op{},
      raft::min_op{});
    raft::linalg::reduce<raft::Apply::ALONG_COLUMNS>(
      res,
      dataset,
      maxima.view(),
      static_cast<T>(-std::numeric_limits<compute_t>::infinity()),
      false,
      raft::identity_op{},
      raft::max_op{});
  } else {
    auto const rows        = dataset.extent(0);
    auto column            = raft::make_device_vector<T, int64_t>(res, rows);
    auto stream            = raft::resource::get_cuda_stream(res).get();
    size_t workspace_bytes = 0;
    RAFT_CUDA_TRY(cub::DeviceMergeSort::SortKeys(
      nullptr, workspace_bytes, column.data_handle(), rows, fp_lt_op<T>{}, stream));
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
                                                     fp_lt_op<T>{},
                                                     stream));
      }
      raft::copy(res,
                 raft::make_device_scalar_view(shifts.data_handle() + j),
                 raft::make_device_scalar_view(column.data_handle() + lower));
      raft::copy(res,
                 raft::make_device_scalar_view(maxima.data_handle() + j),
                 raft::make_device_scalar_view(column.data_handle() + upper));
    }
  }
  quantizer<T> result;
  result.value     = train(res, std::move(shifts), std::move(maxima));
  result.max_error = max_reconstruction_error(
    res, dataset, std::get<typename quantizer_const_view<T>::shifts_type>(result.view().value));
  return result;
}

template <typename T, typename InT, typename OutT>
void check_per_dim_shapes(typename quantizer_const_view<T>::shifts_type quantizer,
                          raft::device_matrix_view<const InT, int64_t> dataset,
                          raft::device_matrix_view<OutT, int64_t> out)
{
  RAFT_EXPECTS(
    std::isfinite(quantizer.scale) && quantizer.scale > 0.f && std::isfinite(1.f / quantizer.scale),
    "Scale and its reciprocal must be finite and positive");
  RAFT_EXPECTS(dataset.extent(1) > 0 && dataset.extent(1) == quantizer.shifts.extent(0),
               "Input width must match the nonempty shifts vector");
  RAFT_EXPECTS(dataset.extent(0) == out.extent(0) && dataset.extent(1) == out.extent(1),
               "Input and output shapes must match");
}

template <typename T>
void transform(raft::resources const& res,
               typename quantizer_const_view<T>::shifts_type quantizer,
               raft::device_matrix_view<const T, int64_t> dataset,
               raft::device_matrix_view<int8_t, int64_t> out)
{
  using compute_t = typename quantizer_const_view<T>::scale_type;
  check_per_dim_shapes<T>(quantizer, dataset, out);
  auto const* input   = dataset.data_handle();
  auto const* shifts  = quantizer.shifts.data_handle();
  auto const dim      = dataset.extent(1);
  compute_t const inv = compute_t{1} / quantizer.scale;
  raft::linalg::map_offset(res, out, [input, shifts, dim, inv] __device__(int64_t i) {
    compute_t const code =
      rint((static_cast<compute_t>(input[i]) - static_cast<compute_t>(shifts[i % dim])) * inv);
    return static_cast<int8_t>(fmin(fmax(code, compute_t{-127}), compute_t{127}));
  });
}

template <typename T>
void inverse_transform(raft::resources const& res,
                       typename quantizer_const_view<T>::shifts_type quantizer,
                       raft::device_matrix_view<const int8_t, int64_t> dataset,
                       raft::device_matrix_view<T, int64_t> out)
{
  using compute_t = typename quantizer_const_view<T>::scale_type;
  check_per_dim_shapes<T>(quantizer, dataset, out);
  auto const* input     = dataset.data_handle();
  auto const* shifts    = quantizer.shifts.data_handle();
  auto const dim        = dataset.extent(1);
  compute_t const scale = quantizer.scale;
  raft::linalg::map_offset(res, out, [input, shifts, dim, scale] __device__(int64_t i) {
    compute_t const value =
      fma(scale, static_cast<compute_t>(input[i]), static_cast<compute_t>(shifts[i % dim]));
    compute_t const limit = static_cast<compute_t>(cuda::std::numeric_limits<T>::max());
    return static_cast<T>(fmin(fmax(value, -limit), limit));
  });
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

  if (params.range == range_mode::per_dimension) { return train(res, dataset, params); }

  auto [min, max] = detail::quantile_min_max(res, dataset, params.quantile);

  RAFT_LOG_DEBUG("quantizer train min=%lf max=%lf.", double(min), double(max));

  quantizer<T> result{min, max};
  result.max_error = max_reconstruction_error(res, dataset, quantize_op<T, int8_t>{min, max});
  return result;
}

template <typename T, typename QuantI = int8_t>
void transform(raft::resources const& res,
               quantizer_const_view<T> quantizer,
               raft::device_matrix_view<const T, int64_t> dataset,
               raft::device_matrix_view<QuantI, int64_t> out)
{
  auto view = quantizer;
  if (auto const* shifts =
        std::get_if<typename quantizer_const_view<T>::shifts_type>(&view.value)) {
    detail::transform(res, *shifts, dataset, out);
    return;
  }
  auto const& range = std::get<typename quantizer_const_view<T>::range_type>(view.value);
  raft::linalg::map(
    res, out, quantize_op<T, QuantI>(range.min_, range.max_), raft::make_const_mdspan(dataset));
}

template <typename T, typename QuantI = int8_t>
void transform(raft::resources const& res,
               quantizer_const_view<T> quantizer,
               raft::host_matrix_view<const T, int64_t> dataset,
               raft::host_matrix_view<QuantI, int64_t> out)
{
  RAFT_EXPECTS(
    std::holds_alternative<typename quantizer_const_view<T>::range_type>(quantizer.value),
    "Host quantization requires a global interval");
  auto const& range = std::get<typename quantizer_const_view<T>::range_type>(quantizer.value);
  auto main_op      = quantize_op<T, QuantI>(range.min_, range.max_);
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
                       quantizer_const_view<T> quantizer,
                       raft::device_matrix_view<const QuantI, int64_t> dataset,
                       raft::device_matrix_view<T, int64_t> out)
{
  auto view = quantizer;
  if (auto const* shifts =
        std::get_if<typename quantizer_const_view<T>::shifts_type>(&view.value)) {
    detail::inverse_transform(res, *shifts, dataset, out);
    return;
  }
  auto const& range = std::get<typename quantizer_const_view<T>::range_type>(view.value);
  raft::linalg::map(
    res, out, quantize_op<T, QuantI>(range.min_, range.max_), raft::make_const_mdspan(dataset));
}

template <typename T, typename QuantI = int8_t>
void inverse_transform(raft::resources const& res,
                       quantizer_const_view<T> quantizer,
                       raft::host_matrix_view<const QuantI, int64_t> dataset,
                       raft::host_matrix_view<T, int64_t> out)
{
  RAFT_EXPECTS(
    std::holds_alternative<typename quantizer_const_view<T>::range_type>(quantizer.value),
    "Host quantization requires a global interval");
  auto const& range = std::get<typename quantizer_const_view<T>::range_type>(quantizer.value);
  auto main_op      = quantize_op<T, QuantI>(range.min_, range.max_);
  size_t n_elements = dataset.extent(0) * dataset.extent(1);

  // In dry-run mode the views may be unbacked or alias the shared probe buffer.
  if (raft::resource::get_dry_run_flag(res)) { return; }

#pragma omp parallel for
  for (size_t i = 0; i < n_elements; ++i) {
    out.data_handle()[i] = main_op(dataset.data_handle()[i]);
  }
}

// The optional output lets dataset construction encode during the error-measurement pass.
// Standalone host training uses the same shared parameters and corpus-wide error reduction.
template <typename T>
quantizer<T> train_host(raft::resources const& res,
                        params params,
                        raft::host_matrix_view<const T, int64_t> dataset,
                        std::optional<raft::device_matrix_view<int8_t, int64_t>> out = std::nullopt)
{
  using compute_t = typename quantizer<T>::scale_type;
  auto const rows = dataset.extent(0);
  auto const dim  = dataset.extent(1);
  RAFT_EXPECTS(rows > 0 && dim > 0 && dataset.data_handle() != nullptr,
               "Training requires nonempty rows and dimensions");
  RAFT_EXPECTS(params.quantile > 0.f && params.quantile <= 1.f, "Quantile must be within (0, 1]");
  RAFT_EXPECTS(params.device_buffer_samples >= 0, "device_buffer_samples must be nonnegative");
  auto const batch_rows =
    params.device_buffer_samples == 0 ? rows : std::min(rows, params.device_buffer_samples);
  auto [copy_stream, prefetch] = cuvs::spatial::knn::detail::utils::get_prefetch_stream(res);
  cuvs::cluster::kmeans::detail::kmeans_batch_loader<T, int64_t, false> batches(
    res,
    dataset,
    batch_rows,
    copy_stream,
    raft::resource::get_large_workspace_resource_ref(res),
    prefetch);
  bool const dry_run  = raft::resource::get_dry_run_flag(res);
  auto for_each_batch = [&](auto&& consume, bool repeat) {
    if (dry_run) {
      // Account for the largest batch's workspace without submitting loader copies.
      consume(raft::make_device_matrix_view<const T, int64_t>(nullptr, batch_rows, dim),
              int64_t{0});
      return;
    }
    batches.start();
    for (std::size_t pos = 0; pos < batches.num_batches(); ++pos) {
      auto batch = batches.acquire(pos);
      if (pos + 1 < batches.num_batches() || repeat) {
        batches.prefetch((pos + 1) % batches.num_batches());
      }
      consume(raft::make_device_matrix_view<const T, int64_t>(batch.data(), batch.size(), dim),
              static_cast<int64_t>(batch.offset()));
      auto const next = pos + (prefetch ? 2 : 1);
      if (next < batches.num_batches() || repeat) {
        batches.recycle(batch, next % batches.num_batches());
      } else {
        batches.release(batch);
      }
    }
  };

  quantizer<T> result;
  if (params.range == range_mode::global) {
    auto [min, max] = quantile_min_max(res, dataset, params.quantile);
    result          = quantizer<T>{min, max};
  } else {
    auto minima = raft::make_device_vector<T, int64_t>(res, dim);
    auto maxima = raft::make_device_vector<T, int64_t>(res, dim);
    if (params.quantile == 1.f) {
      for_each_batch(
        [&](auto input, int64_t offset) {
          raft::linalg::reduce<raft::Apply::ALONG_COLUMNS>(
            res,
            input,
            minima.view(),
            static_cast<T>(std::numeric_limits<compute_t>::infinity()),
            offset != 0,
            raft::identity_op{},
            raft::min_op{});
          raft::linalg::reduce<raft::Apply::ALONG_COLUMNS>(
            res,
            input,
            maxima.view(),
            static_cast<T>(-std::numeric_limits<compute_t>::infinity()),
            offset != 0,
            raft::identity_op{},
            raft::max_op{});
        },
        true);
    } else {
      // Select over the complete column, not independently fitted batch quantiles.
      std::vector<T> mins(dim, T{0}), maxs(dim, T{0});
      if (!dry_run) {
        std::vector<T> column(rows);
        auto const upper =
          static_cast<int64_t>(std::ceil((0.5 + 0.5 * params.quantile) * rows)) - 1;
        auto const lower = rows - upper - 1;
        for (int64_t j = 0; j < dim; ++j) {
          for (int64_t i = 0; i < rows; ++i) {
            column[i] = dataset(i, j);
          }
          std::nth_element(column.begin(), column.begin() + upper, column.end(), fp_lt_op<T>{});
          maxs[j] = column[upper];
          std::nth_element(
            column.begin(), column.begin() + lower, column.begin() + upper + 1, fp_lt_op<T>{});
          mins[j] = column[lower];
        }
      }
      raft::copy(
        res, minima.view(), raft::make_host_vector_view<const T, int64_t>(mins.data(), dim));
      raft::copy(
        res, maxima.view(), raft::make_host_vector_view<const T, int64_t>(maxs.data(), dim));
      raft::resource::sync_stream(res);
    }
    result.value = train(res, std::move(minima), std::move(maxima));
  }

  auto errors  = raft::make_device_vector<double, int64_t>(res, batch_rows);
  auto maximum = raft::make_device_vector<double, int64_t>(res, 1);
  auto view    = result.view();
  for_each_batch(
    [&](auto input, int64_t offset) {
      auto batch_errors =
        raft::make_device_vector_view<double, int64_t>(errors.data_handle(), input.extent(0));
      if (auto const* shifts =
            std::get_if<typename quantizer_const_view<T>::shifts_type>(&view.value)) {
        max_reconstruction_error(res, input, *shifts, batch_errors, maximum.view(), offset != 0);
      } else {
        auto const& range = std::get<typename quantizer_const_view<T>::range_type>(view.value);
        max_reconstruction_error(res,
                                 input,
                                 quantize_op<T, int8_t>{range.min_, range.max_},
                                 batch_errors,
                                 maximum.view(),
                                 offset != 0);
      }
      if (out) {
        auto batch_codes = raft::make_device_matrix_view<int8_t, int64_t>(
          out->data_handle() + offset * dim, input.extent(0), dim);
        detail::transform(res, view, input, batch_codes);
      }
    },
    false);
  double error = std::numeric_limits<double>::infinity();
  raft::copy(
    res, raft::make_host_scalar_view(&error), raft::make_device_scalar_view(maximum.data_handle()));
  raft::resource::sync_stream(res);
  result.max_error = round_error_up<T>(error);
  return result;
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

  if (params.range == range_mode::per_dimension) { return train_host(res, params, dataset); }

  auto [min, max] = detail::quantile_min_max(res, dataset, params.quantile);

  RAFT_LOG_DEBUG("quantizer train min=%lf max=%lf.", double(min), double(max));

  quantizer<T> result{min, max};
  result.max_error = max_reconstruction_error(res, dataset, quantize_op<T, int8_t>{min, max});
  return result;
}

}  // namespace cuvs::preprocessing::quantize::scalar::detail
