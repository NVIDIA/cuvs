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
_RAFT_HOST_DEVICE inline bool fp_lt(const half& a, const half& b)
{
  return static_cast<float>(a) < static_cast<float>(b);
}

template <class T>
struct fp_lt_op {
  _RAFT_HOST_DEVICE bool operator()(const T& a, const T& b) const { return fp_lt(a, b); }
};

template <typename T>
typename quantizer<T>::global_type train(T min, T max)
{
  double const lo = static_cast<double>(min);
  double const hi = static_cast<double>(max);
  RAFT_EXPECTS(std::isfinite(lo) && std::isfinite(hi) && lo <= hi,
               "Bounds must be finite with min <= max");
  return {min, max};
}

template <typename T, typename QuantI>
struct quantize_op {
  double shift;
  double scale;

  explicit quantize_op(typename quantizer_view<const T>::global_type bounds)
  {
    auto const range = detail::train<T>(bounds.min, bounds.max);
    double const lo  = static_cast<double>(range.min);
    double const hi  = static_cast<double>(range.max);
    if (lo == hi) {
      shift = lo;
      scale = 0;
      return;
    }
    constexpr double q_min = std::numeric_limits<QuantI>::min();
    constexpr double q_max = std::numeric_limits<QuantI>::max();
    // Divide the endpoints separately if subtracting them would overflow.
    double const width = hi - lo;
    scale              = std::max(
      std::isfinite(width) ? width / (q_max - q_min) : hi / (q_max - q_min) - lo / (q_max - q_min),
      std::numeric_limits<double>::min());
    shift = lo * 0.5 + hi * 0.5 - (q_min + q_max) * 0.5 * scale;
  }

  constexpr RAFT_INLINE_FUNCTION QuantI operator()(const T& x) const
  {
    constexpr double q_min = std::numeric_limits<QuantI>::min();
    constexpr double q_max = std::numeric_limits<QuantI>::max();
    if (scale == 0) { return static_cast<QuantI>(static_cast<double>(x) <= shift ? q_min : q_max); }
    double const code = (static_cast<double>(x) - shift) / scale;
    return static_cast<QuantI>(lround(fmin(fmax(code, q_min), q_max)));
  }

  constexpr RAFT_INLINE_FUNCTION T operator()(const QuantI& x) const
  {
    double const value = fma(scale, static_cast<double>(x), shift);
    double const limit = static_cast<double>(cuda::std::numeric_limits<T>::max());
    return static_cast<T>(fmin(fmax(value, -limit), limit));
  }
};

template <typename T, typename QuantI>
struct per_dim_quantize_op {
  using compute_t = typename quantizer_view<const T>::scale_type;
  compute_t shift;
  compute_t scale;

  RAFT_INLINE_FUNCTION per_dim_quantize_op(T min, T max)
  {
    compute_t const lo = static_cast<compute_t>(min);
    compute_t const hi = static_cast<compute_t>(max);
    // Halve before adding to avoid overflow, and round the midpoint to the input type.
    shift = static_cast<compute_t>(
      static_cast<T>(lo == hi ? lo : lo * compute_t{0.5} + hi * compute_t{0.5}));
    compute_t const half_range = fmax(hi - shift, shift - lo);
    constexpr compute_t q_max  = std::numeric_limits<QuantI>::max();
    scale = half_range > 0 ? fmax(half_range / q_max, std::numeric_limits<compute_t>::min())
                           : compute_t{1};
  }

  RAFT_INLINE_FUNCTION QuantI operator()(const T& x) const
  {
    compute_t const code      = rint((static_cast<compute_t>(x) - shift) / scale);
    constexpr compute_t q_max = std::numeric_limits<QuantI>::max();
    constexpr compute_t q_min = std::max<compute_t>(std::numeric_limits<QuantI>::min(), -q_max);
    return static_cast<QuantI>(fmin(fmax(code, q_min), q_max));
  }

  RAFT_INLINE_FUNCTION T operator()(const QuantI& x) const
  {
    compute_t const value = fma(scale, static_cast<compute_t>(x), shift);
    compute_t const limit = static_cast<compute_t>(cuda::std::numeric_limits<T>::max());
    return static_cast<T>(fmin(fmax(value, -limit), limit));
  }
};

template <typename T, typename QuantI, typename Op>
RAFT_INLINE_FUNCTION double reconstruction_error(T x, int64_t j, Op op)
{
  double const value = static_cast<double>(x);
  double decoded;
  if constexpr (std::is_same_v<Op, typename quantizer_view<const T>::per_dim_type>) {
    auto const quantize =
      per_dim_quantize_op<T, QuantI>{op.mins.data_handle()[j], op.maxs.data_handle()[j]};
    decoded = static_cast<double>(quantize(quantize(x)));
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
  // Apply a conservative margin, then round upwards in the public result type.
  auto const bound = static_cast<bound_t>(error * (1.0 + 1e-6));
  return error > 0 ? std::nextafter(bound, std::numeric_limits<bound_t>::infinity()) : bound;
}

template <typename T, typename QuantI, typename Op>
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
    [op] __device__(T x, int64_t j) { return reconstruction_error<T, QuantI>(x, j, op); },
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
typename quantizer<T>::per_dim_type train(raft::resources const& res,
                                          raft::device_vector<T, int64_t>&& minima,
                                          raft::device_vector<T, int64_t>&& maxima)
{
  using compute_t     = typename quantizer_view<const T>::scale_type;
  auto const dim      = minima.extent(0);
  auto invalid        = raft::make_device_vector<int, int64_t>(res, 1);
  auto const* max_ptr = maxima.data_handle();
  raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
    res,
    raft::make_device_matrix_view<const T, int64_t>(minima.data_handle(), 1, dim),
    invalid.view(),
    0,
    false,
    [max_ptr] __device__(T min, int64_t j) {
      compute_t const lo = static_cast<compute_t>(min);
      compute_t const hi = static_cast<compute_t>(max_ptr[j]);
      return static_cast<int>(!isfinite(lo) || !isfinite(hi) || lo > hi);
    },
    raft::max_op{});
  int invalid_bounds = 0;
  raft::copy(res,
             raft::make_host_scalar_view(&invalid_bounds),
             raft::make_device_scalar_view(invalid.data_handle()));
  raft::resource::sync_stream(res);
  RAFT_EXPECTS(invalid_bounds == 0, "Bounds must be finite with min <= max");
  return {std::move(minima), std::move(maxima)};
}

// Encoding and decoding share the traversal; the input type selects the operator overload.
template <typename T, typename QuantI, typename InT, typename OutT>
void apply_quantizer(raft::resources const& res,
                     quantizer_view<const T> quantizer,
                     raft::device_matrix_view<const InT, int64_t> dataset,
                     raft::device_matrix_view<OutT, int64_t> out)
{
  if (auto const* bounds =
        std::get_if<typename quantizer_view<const T>::per_dim_type>(&quantizer.value)) {
    auto const dim = dataset.extent(1);
    RAFT_EXPECTS(dim > 0 && dim == bounds->mins.extent(0) && dim == bounds->maxs.extent(0),
                 "Input width must match the nonempty min and max vectors");
    RAFT_EXPECTS(dataset.extent(0) == out.extent(0) && dim == out.extent(1),
                 "Input and output shapes must match");
    auto const* input = dataset.data_handle();
    auto const* mins  = bounds->mins.data_handle();
    auto const* maxs  = bounds->maxs.data_handle();
    raft::linalg::map_offset(res, out, [input, mins, maxs, dim] __device__(int64_t i) {
      return per_dim_quantize_op<T, QuantI>{mins[i % dim], maxs[i % dim]}(input[i]);
    });
  } else {
    auto const& range = std::get<typename quantizer_view<const T>::global_type>(quantizer.value);
    raft::linalg::map(res, out, quantize_op<T, QuantI>{range}, dataset);
  }
}

template <typename T, typename QuantI, typename InT, typename OutT>
void apply_quantizer(raft::resources const& res,
                     quantizer_view<const T> quantizer,
                     raft::host_matrix_view<const InT, int64_t> dataset,
                     raft::host_matrix_view<OutT, int64_t> out)
{
  RAFT_EXPECTS(
    std::holds_alternative<typename quantizer_view<const T>::global_type>(quantizer.value),
    "Host quantization requires global parameters");
  auto const& range = std::get<typename quantizer_view<const T>::global_type>(quantizer.value);
  auto main_op      = quantize_op<T, QuantI>{range};
  size_t n_elements = dataset.extent(0) * dataset.extent(1);

  // In dry-run mode the views may be unbacked or alias the shared probe buffer.
  if (raft::resource::get_dry_run_flag(res)) { return; }

#pragma omp parallel for
  for (size_t i = 0; i < n_elements; ++i) {
    out.data_handle()[i] = main_op(dataset.data_handle()[i]);
  }
}

// The optional output lets dataset construction encode during the error-measurement pass.
// Host and device inputs share the same batch processing and corpus-wide error reduction.
template <typename T, typename QuantI = int8_t, typename Accessor>
quantizer<T> train(
  raft::resources const& res,
  params params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  std::optional<raft::device_matrix_view<QuantI, int64_t>> out = std::nullopt)
{
  using compute_t = typename quantizer<T>::scale_type;
  auto const rows = dataset.extent(0);
  auto const dim  = dataset.extent(1);
  RAFT_EXPECTS(rows > 0 && dim > 0 && dataset.data_handle() != nullptr,
               "Training requires nonempty rows and dimensions");
  RAFT_EXPECTS(params.quantile > 0.f && params.quantile <= 1.f, "Quantile must be within (0, 1]");
  constexpr bool data_on_device = raft::is_device_mdspan_v<decltype(dataset)>;
  if constexpr (!data_on_device) {
    RAFT_EXPECTS(params.device_buffer_samples >= 0, "device_buffer_samples must be nonnegative");
  }
  auto const batch_rows        = data_on_device || params.device_buffer_samples == 0
                                   ? rows
                                   : std::min(rows, params.device_buffer_samples);
  auto [copy_stream, prefetch] = cuvs::spatial::knn::detail::utils::get_prefetch_stream(res);
  cuvs::cluster::kmeans::detail::kmeans_batch_loader<T, int64_t, data_on_device> batches(
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
  if (params.range == quantization_mode::GLOBAL) {
    auto [min, max] = quantile_min_max(res, dataset, params.quantile);
    result.value    = detail::train<T>(min, max);
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
    } else if constexpr (data_on_device) {
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
                   raft::make_device_scalar_view(minima.data_handle() + j),
                   raft::make_device_scalar_view(column.data_handle() + lower));
        raft::copy(res,
                   raft::make_device_scalar_view(maxima.data_handle() + j),
                   raft::make_device_scalar_view(column.data_handle() + upper));
      }
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
    result.value = train<T>(res, std::move(minima), std::move(maxima));
  }

  auto errors                  = raft::make_device_vector<double, int64_t>(res, batch_rows);
  auto maximum                 = raft::make_device_vector<double, int64_t>(res, 1);
  quantizer_view<const T> view = result.view();
  for_each_batch(
    [&](auto input, int64_t offset) {
      auto batch_errors =
        raft::make_device_vector_view<double, int64_t>(errors.data_handle(), input.extent(0));
      if (auto const* bounds =
            std::get_if<typename quantizer_view<const T>::per_dim_type>(&view.value)) {
        max_reconstruction_error<T, QuantI>(
          res, input, *bounds, batch_errors, maximum.view(), offset != 0);
      } else {
        auto const& range = std::get<typename quantizer_view<const T>::global_type>(view.value);
        max_reconstruction_error<T, QuantI>(
          res, input, quantize_op<T, QuantI>{range}, batch_errors, maximum.view(), offset != 0);
      }
      if (out) {
        auto batch_codes = raft::make_device_matrix_view<QuantI, int64_t>(
          out->data_handle() + offset * dim, input.extent(0), dim);
        apply_quantizer<T, QuantI>(res, view, input, batch_codes);
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

}  // namespace cuvs::preprocessing::quantize::scalar::detail
