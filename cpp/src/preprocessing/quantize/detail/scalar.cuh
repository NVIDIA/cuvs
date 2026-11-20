/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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

// Bounds are validated before they are stored in a quantizer or used by a transform.
inline quantizer::global_type checked_bounds(double min, double max)
{
  RAFT_EXPECTS(std::isfinite(min) && std::isfinite(max) && min <= max,
               "Bounds must be finite with min <= max");
  return {min, max};
}

template <typename T, typename QuantI>
struct quantize_op {
  double shift;
  double scale;

  // Per-dimension transforms build this for each element on device, from validated bounds.
  RAFT_INLINE_FUNCTION quantize_op(double lo, double hi)
  {
    if (lo == hi) {
      shift = lo;
      scale = 0;
      return;
    }
    constexpr double q_min = std::numeric_limits<QuantI>::min();
    constexpr double q_max = std::numeric_limits<QuantI>::max();
    // Divide the endpoints separately if subtracting them would overflow.
    double const width = hi - lo;
    scale =
      fmax(isfinite(width) ? width / (q_max - q_min) : hi / (q_max - q_min) - lo / (q_max - q_min),
           std::numeric_limits<double>::min());
    shift = lo * 0.5 + hi * 0.5 - (q_min + q_max) * 0.5 * scale;
  }

  explicit quantize_op(quantizer_view<const double>::global_type bounds)
    : quantize_op(bounds.min, bounds.max)
  {
    checked_bounds(bounds.min, bounds.max);
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

template <typename T, typename QuantI, typename Op>
RAFT_INLINE_FUNCTION double reconstruction_error(T x, int64_t j, Op op)
{
  double const value = static_cast<double>(x);
  double decoded;
  if constexpr (std::is_same_v<Op, quantizer_view<const double>::per_dim_type>) {
    auto const quantize =
      quantize_op<T, QuantI>{op.mins.data_handle()[j], op.maxs.data_handle()[j]};
    decoded = static_cast<double>(quantize(quantize(x)));
  } else {
    decoded = static_cast<double>(op(op(x)));
  }
  return isfinite(value) && isfinite(decoded) ? fabs(value - decoded)
                                              : std::numeric_limits<double>::infinity();
}

inline double round_error_up(double error)
{
  // Apply a conservative margin, then round upwards.
  auto const bound = error * (1.0 + 1e-6);
  return error > 0 ? std::nextafter(bound, std::numeric_limits<double>::infinity()) : bound;
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

inline quantizer::per_dim_type checked_bounds(raft::resources const& res,
                                              raft::device_vector<double, int64_t>&& minima,
                                              raft::device_vector<double, int64_t>&& maxima)
{
  auto const dim      = minima.extent(0);
  auto invalid        = raft::make_device_vector<int, int64_t>(res, 1);
  auto const* max_ptr = maxima.data_handle();
  raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
    res,
    raft::make_device_matrix_view<const double, int64_t>(minima.data_handle(), 1, dim),
    invalid.view(),
    0,
    false,
    [max_ptr] __device__(double lo, int64_t j) {
      double const hi = max_ptr[j];
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
                     quantizer_view<const double> quantizer,
                     raft::device_matrix_view<const InT, int64_t> dataset,
                     raft::device_matrix_view<OutT, int64_t> out)
{
  if (auto const* bounds =
        std::get_if<quantizer_view<const double>::per_dim_type>(&quantizer.value)) {
    auto const dim = dataset.extent(1);
    RAFT_EXPECTS(dim > 0 && dim == bounds->mins.extent(0) && dim == bounds->maxs.extent(0),
                 "Input width must match the nonempty min and max vectors");
    RAFT_EXPECTS(dataset.extent(0) == out.extent(0) && dim == out.extent(1),
                 "Input and output shapes must match");
    auto const* input = dataset.data_handle();
    auto const* mins  = bounds->mins.data_handle();
    auto const* maxs  = bounds->maxs.data_handle();
    raft::linalg::map_offset(res, out, [input, mins, maxs, dim] __device__(int64_t i) {
      return quantize_op<T, QuantI>{mins[i % dim], maxs[i % dim]}(input[i]);
    });
  } else {
    auto const& range = std::get<quantizer_view<const double>::global_type>(quantizer.value);
    raft::linalg::map(res, out, quantize_op<T, QuantI>{range}, dataset);
  }
}

template <typename T, typename QuantI, typename InT, typename OutT>
void apply_quantizer(raft::resources const& res,
                     quantizer_view<const double> quantizer,
                     raft::host_matrix_view<const InT, int64_t> dataset,
                     raft::host_matrix_view<OutT, int64_t> out)
{
  RAFT_EXPECTS(std::holds_alternative<quantizer_view<const double>::global_type>(quantizer.value),
               "Host quantization requires global parameters");
  auto const& range = std::get<quantizer_view<const double>::global_type>(quantizer.value);
  auto main_op      = quantize_op<T, QuantI>{range};
  size_t n_elements = dataset.extent(0) * dataset.extent(1);

  // In dry-run mode the views may be unbacked or alias the shared probe buffer.
  if (raft::resource::get_dry_run_flag(res)) { return; }

#pragma omp parallel for
  for (size_t i = 0; i < n_elements; ++i) {
    out.data_handle()[i] = main_op(dataset.data_handle()[i]);
  }
}

// Host input is staged in batches of at most device_buffer_samples rows; device input is one batch.
template <typename T, typename Accessor>
int64_t staged_rows(
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  int64_t device_buffer_samples)
{
  auto const rows = dataset.extent(0);
  if constexpr (raft::is_device_mdspan_v<decltype(dataset)>) {
    return rows;
  } else {
    RAFT_EXPECTS(device_buffer_samples >= 0, "device_buffer_samples must be nonnegative");
    return device_buffer_samples == 0 ? rows : std::min(rows, device_buffer_samples);
  }
}

// One pass over the dataset: consume(batch, row_offset) receives each device batch in order.
template <typename T, typename Accessor, typename Consume>
void for_each_batch(
  raft::resources const& res,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  int64_t batch_rows,
  Consume&& consume)
{
  namespace utils              = cuvs::spatial::knn::detail::utils;
  auto const dim               = dataset.extent(1);
  auto [copy_stream, prefetch] = utils::get_prefetch_stream(res);
  utils::batch_load_iterator batches(res,
                                     dataset,
                                     batch_rows,
                                     copy_stream,
                                     raft::resource::get_large_workspace_resource_ref(res),
                                     prefetch);
  if (raft::resource::get_dry_run_flag(res)) {
    // Account for the largest batch's workspace without submitting copies.
    consume(raft::make_device_matrix_view<const T, int64_t>(nullptr, batch_rows, dim), int64_t{0});
    return;
  }
  batches.prefetch_next_batch();
  for (auto const& batch : batches) {
    consume(raft::make_device_matrix_view<const T, int64_t>(batch.data(), batch.size(), dim),
            static_cast<int64_t>(batch.offset()));
    batches.prefetch_next_batch();
  }
  // The last batch's kernels must finish before its staging buffer is released.
  if (batches.does_copy()) { raft::resource::sync_stream(res); }
}

// Measure the maximum reconstruction error over every row of a host or device dataset.
// Dataset construction also encodes into the optional output during the same pass, and may skip
// the measurement.
template <typename T, typename QuantI = int8_t, typename Accessor>
auto max_error(
  raft::resources const& res,
  quantizer_view<const double> view,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  int64_t device_buffer_samples,
  bool measure                                                 = true,
  std::optional<raft::device_matrix_view<QuantI, int64_t>> out = std::nullopt) -> double
{
  auto const dim = dataset.extent(1);
  RAFT_EXPECTS(dataset.extent(0) > 0 && dim > 0 && dataset.data_handle() != nullptr,
               "Dataset must have nonempty rows and dimensions");
  auto const* bounds = std::get_if<quantizer_view<const double>::per_dim_type>(&view.value);
  RAFT_EXPECTS(
    bounds == nullptr || (dim == bounds->mins.extent(0) && dim == bounds->maxs.extent(0)),
    "Input width must match the min and max vectors");
  auto const batch_rows = staged_rows(dataset, device_buffer_samples);
  auto errors           = raft::make_device_vector<double, int64_t>(res, measure ? batch_rows : 0);
  auto maximum          = raft::make_device_vector<double, int64_t>(res, 1);
  for_each_batch(res, dataset, batch_rows, [&](auto input, int64_t offset) {
    if (measure) {
      auto batch_errors =
        raft::make_device_vector_view<double, int64_t>(errors.data_handle(), input.extent(0));
      if (bounds != nullptr) {
        max_reconstruction_error<T, QuantI>(
          res, input, *bounds, batch_errors, maximum.view(), offset != 0);
      } else {
        auto const& range = std::get<quantizer_view<const double>::global_type>(view.value);
        max_reconstruction_error<T, QuantI>(
          res, input, quantize_op<T, QuantI>{range}, batch_errors, maximum.view(), offset != 0);
      }
    }
    if (out) {
      auto batch_codes = raft::make_device_matrix_view<QuantI, int64_t>(
        out->data_handle() + offset * dim, input.extent(0), dim);
      apply_quantizer<T, QuantI>(res, view, input, batch_codes);
    }
  });
  double error = std::numeric_limits<double>::infinity();
  if (measure) {
    raft::copy(res,
               raft::make_host_scalar_view(&error),
               raft::make_device_scalar_view(maximum.data_handle()));
    raft::resource::sync_stream(res);
  }
  return round_error_up(error);
}

// Host and device inputs share the same batch processing.
template <typename T, typename Accessor>
quantizer train(
  raft::resources const& res,
  params params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  auto const rows = dataset.extent(0);
  auto const dim  = dataset.extent(1);
  RAFT_EXPECTS(rows > 0 && dim > 0 && dataset.data_handle() != nullptr,
               "Training requires nonempty rows and dimensions");
  RAFT_EXPECTS(params.quantile > 0.f && params.quantile <= 1.f, "Quantile must be within (0, 1]");
  constexpr bool data_on_device = raft::is_device_mdspan_v<decltype(dataset)>;
  auto const batch_rows         = staged_rows(dataset, params.device_buffer_samples);

  quantizer result;
  if (params.range == quantization_mode::GLOBAL) {
    auto [min, max] = quantile_min_max(res, dataset, params.quantile);
    result.value    = checked_bounds(static_cast<double>(min), static_cast<double>(max));
  } else {
    auto minima = raft::make_device_vector<T, int64_t>(res, dim);
    auto maxima = raft::make_device_vector<T, int64_t>(res, dim);
    if (params.quantile == 1.f) {
      for_each_batch(res, dataset, batch_rows, [&](auto input, int64_t offset) {
        raft::linalg::reduce<raft::Apply::ALONG_COLUMNS>(
          res,
          input,
          minima.view(),
          static_cast<T>(std::numeric_limits<float>::infinity()),
          offset != 0,
          raft::identity_op{},
          raft::min_op{});
        raft::linalg::reduce<raft::Apply::ALONG_COLUMNS>(
          res,
          input,
          maxima.view(),
          static_cast<T>(-std::numeric_limits<float>::infinity()),
          offset != 0,
          raft::identity_op{},
          raft::max_op{});
      });
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
      if (!raft::resource::get_dry_run_flag(res)) {
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
    // Bounds are selected in the data type, which converts exactly to double.
    auto to_double = [&](raft::device_vector<T, int64_t> const& bounds) {
      auto converted = raft::make_device_vector<double, int64_t>(res, dim);
      raft::linalg::map(
        res, converted.view(), raft::cast_op<double>{}, raft::make_const_mdspan(bounds.view()));
      return converted;
    };
    result.value = checked_bounds(res, to_double(minima), to_double(maxima));
  }
  return result;
}

}  // namespace cuvs::preprocessing::quantize::scalar::detail
