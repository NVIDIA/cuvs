/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "distance_ops/all_ops.cuh"
#include "pairwise_matrix/dispatch.cuh"
#include <cuvs/distance/distance.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/util/cuda_dev_essentials.cuh>  // to_float

#include <type_traits>

namespace cuvs {
namespace distance {
namespace detail {

/**
 * @brief: A tag type for overload resolution based on DistanceType
 *
 * It is not possible to partially specialize function templates on a single
 * parameter. Instead, it is often easier to use a combination of conventional
 * method overloading and a parameter with a specific tag type. The following
 * type is used to help method overloading based on the DistanceType enum.
 */
template <DistanceType d>
using distance_tag = std::integral_constant<DistanceType, d>;

/**
 * @brief Implement pairwise_matrix for specific distance
 *
 * There are multiple overloads for this function, one for each distance type.
 * They are implemented below. The documentation of this function serves as
 * documentation for all functions. The following overloads are defined:
 *
 * - DistanceType::Canberra:
 * - DistanceType::CorrelationExpanded:
 * - DistanceType::CosineExpanded:
 * - DistanceType::HammingUnexpanded:
 * - DistanceType::HellingerExpanded:
 * - DistanceType::JensenShannon:
 * - DistanceType::KLDivergence:
 * - DistanceType::L1:
 * - DistanceType::L2Expanded:
 * - DistanceType::L2SqrtExpanded:
 * - DistanceType::L2Unexpanded:
 * - DistanceType::L2SqrtUnexpanded:
 * - DistanceType::Linf:
 * - DistanceType::LpUnexpanded:
 * - DistanceType::RusselRaoExpanded:
 *
 * @tparam DataT   Input data type
 * @tparam AccT    Accumulation data type
 * @tparam OutT    Output data type
 * @tparam FinOpT  Type of final operation
 * @tparam IdxT    Index type
 *
 * @param handle        RAFT resources handle
 * @param distance_type A tag type to indicate which distance is calculated.
 * @param x             First set of points
 * @param y             Second set of points
 * @param out           Output distance matrix
 * @param m             Number of points in x
 * @param n             Number of points in y
 * @param k             Dimensionality of points in x, y
 * @param workspace     Temporary workspace needed for computations
 * @param worksize      Number of bytes of the workspace
 * @param is_row_major  Whether the matrices are row-major or col-major
 * @param metric_arg    The `p` argument for Lp.
 */
template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::Canberra> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT* workspace,  // unused
                   size_t worksize,  // unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT metric_arg)  // unused
{
  ops::canberra_distance_op<DataT, AccT, IdxT> distance_op{};

  const OutT* x_norm = nullptr;
  const OutT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  pairwise_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::CorrelationExpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT* workspace,
                   size_t worksize,
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // unused
{
  ASSERT(!(worksize < 2 * (m + n) * sizeof(AccT)), "workspace size error");
  ASSERT(workspace != nullptr, "workspace is null");

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  AccT* x_norm    = workspace;
  AccT* y_norm    = workspace;
  AccT* sq_x_norm = workspace;
  AccT* sq_y_norm = workspace;
  // TODO: Column major case looks to have lower accuracy for X == Y,
  // perhaps the use of stridedSummationKernel could be causing this,
  // need to investigate and fix.
  if (x == y && is_row_major) {
    raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_device_matrix_view<const DataT, IdxT, raft::row_major>(x, std::max(m, n), k),
      raft::make_device_vector_view<AccT, IdxT>(x_norm, std::max(m, n)),
      (AccT)0,
      false,
      raft::identity_op(),
      raft::add_op());
    sq_x_norm += std::max(m, n);
    sq_y_norm = sq_x_norm;
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_device_matrix_view<const DataT, IdxT, raft::row_major>(x, std::max(m, n), k),
      raft::make_device_vector_view(sq_x_norm, std::max(m, n)));
  } else {
    y_norm += m;
    if (is_row_major) {
      raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IdxT, raft::row_major>(x, m, k),
        raft::make_device_vector_view<AccT, IdxT>(x_norm, m),
        (AccT)0,
        false,
        raft::identity_op(),
        raft::add_op());
      raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IdxT, raft::row_major>(y, n, k),
        raft::make_device_vector_view<AccT, IdxT>(y_norm, n),
        (AccT)0,
        false,
        raft::identity_op(),
        raft::add_op());
    } else {
      raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IdxT, raft::col_major>(x, m, k),
        raft::make_device_vector_view<AccT, IdxT>(x_norm, m),
        (AccT)0,
        false,
        raft::identity_op(),
        raft::add_op());
      raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IdxT, raft::col_major>(y, n, k),
        raft::make_device_vector_view<AccT, IdxT>(y_norm, n),
        (AccT)0,
        false,
        raft::identity_op(),
        raft::add_op());
    }

    sq_x_norm += (m + n);
    sq_y_norm = sq_x_norm + m;
    if (is_row_major) {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IdxT, raft::row_major>(x, m, k),
        raft::make_device_vector_view(sq_x_norm, m));
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IdxT, raft::row_major>(y, n, k),
        raft::make_device_vector_view(sq_y_norm, n));
    } else {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IdxT, raft::col_major>(x, m, k),
        raft::make_device_vector_view(sq_x_norm, m));
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IdxT, raft::col_major>(y, n, k),
        raft::make_device_vector_view(sq_y_norm, n));
    }
  }

  using OpT = ops::correlation_distance_op<DataT, AccT, IdxT>;
  OpT corr_op(is_row_major, sq_x_norm, sq_y_norm, m, n, k);
  pairwise_matrix_dispatch<decltype(corr_op), DataT, AccT, OutT, FinOpT, IdxT>(
    corr_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::CosineExpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT* workspace,
                   size_t worksize,
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // unused
{
  // raft distance support inputs as float/double and output as uint8_t/float/double.
  static_assert(!((sizeof(OutT) > 1) && (sizeof(AccT) != sizeof(OutT))),
                "OutT can be uint8_t, float, double,"
                "if sizeof(OutT) > 1 then sizeof(AccT) == sizeof(OutT).");

  ASSERT(!(worksize < (m + n) * sizeof(AccT)), "workspace size error");
  ASSERT(workspace != nullptr, "workspace is null");

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  OutT* x_norm = reinterpret_cast<OutT*>(workspace);
  OutT* y_norm = reinterpret_cast<OutT*>(workspace);
  // TODO: Column major case looks to have lower accuracy for X == Y,
  // perhaps the use of stridedSummationKernel could be causing this,
  // need to investigate and fix.
  if (x == y && is_row_major) {
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_device_matrix_view<const DataT, IdxT, raft::row_major>(x, std::max(m, n), k),
      raft::make_device_vector_view(x_norm, std::max(m, n)),
      raft::sqrt_op{});
  } else {
    y_norm += m;
    if (is_row_major) {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IdxT, raft::row_major>(x, m, k),
        raft::make_device_vector_view(x_norm, m),
        raft::sqrt_op{});
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IdxT, raft::row_major>(y, n, k),
        raft::make_device_vector_view(y_norm, n),
        raft::sqrt_op{});
    } else {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IdxT, raft::col_major>(x, m, k),
        raft::make_device_vector_view(x_norm, m),
        raft::sqrt_op{});
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IdxT, raft::col_major>(y, n, k),
        raft::make_device_vector_view(y_norm, n),
        raft::sqrt_op{});
    }
  }

  ops::cosine_distance_op<DataT, AccT, IdxT> distance_op{};
  pairwise_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::HammingUnexpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  ops::hamming_distance_op<DataT, AccT, IdxT> distance_op{k};

  const OutT* x_norm = nullptr;
  const OutT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  pairwise_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::InnerProduct> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  raft::linalg::gemm(handle,
                     out,
                     const_cast<DataT*>(x),
                     const_cast<DataT*>(y),
                     m,
                     n,
                     k,
                     !is_row_major,
                     !is_row_major,
                     is_row_major,
                     stream);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::HellingerExpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  // Check if arrays overlap
  const DataT* x_end  = x + m * k;
  const DataT* y_end  = y + n * k;
  bool arrays_overlap = (x < y_end) && (y < x_end);

  if (!arrays_overlap) {
    // Arrays don't overlap: sqrt each array independently
    raft::linalg::map(
      handle,
      raft::make_device_vector_view<DataT, IdxT>((DataT*)x, m * k),
      raft::sqrt_op{},
      raft::make_const_mdspan(raft::make_device_vector_view<const DataT, IdxT>(x, m * k)));
    raft::linalg::map(
      handle,
      raft::make_device_vector_view<DataT, IdxT>((DataT*)y, n * k),
      raft::sqrt_op{},
      raft::make_const_mdspan(raft::make_device_vector_view<const DataT, IdxT>(y, n * k)));
  } else {
    // Arrays overlap: sqrt the union of both arrays exactly once
    const DataT* start = (x < y) ? x : y;
    const DataT* end   = (x_end > y_end) ? x_end : y_end;
    IdxT union_size    = end - start;

    raft::linalg::map(
      handle,
      raft::make_device_vector_view<DataT, IdxT>((DataT*)start, union_size),
      raft::sqrt_op{},
      raft::make_const_mdspan(raft::make_device_vector_view<const DataT, IdxT>(start, union_size)));
  }

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  // Calculate Hellinger distance
  ops::hellinger_distance_op<DataT, AccT, IdxT> distance_op{};

  const OutT* x_norm = nullptr;
  const OutT* y_norm = nullptr;

  pairwise_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);

  // Restore arrays by squaring back
  if (!arrays_overlap) {
    // Arrays don't overlap: square each array independently
    raft::linalg::map(
      handle,
      raft::make_device_vector_view<DataT, IdxT>((DataT*)x, m * k),
      raft::sq_op{},
      raft::make_const_mdspan(raft::make_device_vector_view<const DataT, IdxT>(x, m * k)));
    raft::linalg::map(
      handle,
      raft::make_device_vector_view<DataT, IdxT>((DataT*)y, n * k),
      raft::sq_op{},
      raft::make_const_mdspan(raft::make_device_vector_view<const DataT, IdxT>(y, n * k)));
  } else {
    // Arrays overlap: square the union back
    const DataT* start = (x < y) ? x : y;
    const DataT* end   = (x_end > y_end) ? x_end : y_end;
    IdxT union_size    = end - start;

    raft::linalg::map(
      handle,
      raft::make_device_vector_view<DataT, IdxT>((DataT*)start, union_size),
      raft::sq_op{},
      raft::make_const_mdspan(raft::make_device_vector_view<const DataT, IdxT>(start, union_size)));
  }

  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::JensenShannon> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  ops::jensen_shannon_distance_op<DataT, AccT, IdxT> distance_op{};

  const OutT* x_norm = nullptr;
  const OutT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  pairwise_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::KLDivergence> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto unaryOp_lambda = [] __device__(DataT input) {
    auto input_       = raft::to_float(input);
    const bool x_zero = (input_ == 0);
    if constexpr (std::is_same_v<DataT, half>) {
      return __float2half((!x_zero) * raft::log(input_ + x_zero));
    } else {
      return (!x_zero) * raft::log(input_ + x_zero);
    }
  };

  auto unaryOp_lambda_reverse = [] __device__(DataT input) {
    // reverse previous log (x) back to x using (e ^ log(x))
    auto input_       = raft::to_float(input);
    const bool x_zero = (input_ == 0);
    if constexpr (std::is_same_v<DataT, half>) {
      return __float2half((!x_zero) * raft::exp(input_));
    } else {
      return (!x_zero) * raft::exp(input_);
    }
  };

  if (x != y) {
    raft::linalg::map(
      handle,
      raft::make_device_vector_view<DataT, IdxT>((DataT*)y, n * k),
      unaryOp_lambda,
      raft::make_const_mdspan(raft::make_device_vector_view<const DataT, IdxT>(y, n * k)));
  }

  const OutT* x_norm = nullptr;
  const OutT* y_norm = nullptr;

  // This op takes some shortcuts when x equals y. So its behavior changes based
  // on this.
  ops::kl_divergence_op<DataT, AccT, IdxT> distance_op{is_row_major, x == y};

  pairwise_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);

  if (x != y) {
    // Now reverse previous log (x) back to x using (e ^ log(x))
    raft::linalg::map(
      handle,
      raft::make_device_vector_view<DataT, IdxT>((DataT*)y, n * k),
      unaryOp_lambda_reverse,
      raft::make_const_mdspan(raft::make_device_vector_view<const DataT, IdxT>(y, n * k)));
  }
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::L1> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  ops::l1_distance_op<DataT, AccT, IdxT> distance_op{};

  const OutT* x_norm = nullptr;
  const OutT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  pairwise_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void distance_impl_l2_expanded(  // NOTE: different name
  bool perform_sqrt,             // dispatch on sqrt
  const DataT* x,
  const DataT* y,
  OutT* out,
  IdxT m,
  IdxT n,
  IdxT k,
  AccT* workspace,
  size_t worksize,
  FinOpT fin_op,
  cudaStream_t stream,
  bool is_row_major)
{
  // raft distance support inputs as float/double and output as uint8_t/float/double.
  static_assert(!((sizeof(OutT) > 1) && (sizeof(AccT) != sizeof(OutT))),
                "OutT can be uint8_t, float, double,"
                "if sizeof(OutT) > 1 then sizeof(AccT) == sizeof(OutT).");

  ASSERT(!(worksize < (m + n) * sizeof(AccT)), "workspace size error");
  ASSERT(workspace != nullptr, "workspace is null");

  // TODO: May we have a better method to avoid misalignment?
  uintptr_t offset = alignof(OutT) - (reinterpret_cast<uintptr_t>(workspace) % alignof(OutT));
  if (offset == alignof(OutT)) { offset = 0; }
  OutT* x_norm = reinterpret_cast<OutT*>(reinterpret_cast<char*>(workspace) + offset);

  offset       = (reinterpret_cast<uintptr_t>(x_norm) % alignof(OutT));
  OutT* y_norm = x_norm;
  // TODO: Column major case looks to have lower accuracy for X == Y,
  // perhaps the use of stridedSummationKernel could be causing this,
  // need to investigate and fix.
  if ((x == y) && is_row_major) {
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      x_norm, x, k, std::max(m, n), stream, raft::identity_op{});
  } else {
    y_norm += m;
    if (is_row_major) {
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
        x_norm, x, k, m, stream, raft::identity_op{});
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
        y_norm, y, k, n, stream, raft::identity_op{});
    } else {
      raft::linalg::rowNorm<raft::linalg::L2Norm, false>(
        x_norm, x, k, m, stream, raft::identity_op{});
      raft::linalg::rowNorm<raft::linalg::L2Norm, false>(
        y_norm, y, k, n, stream, raft::identity_op{});
    }
  }

  ops::l2_exp_distance_op<DataT, AccT, IdxT> distance_op{perform_sqrt};
  pairwise_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::L2Expanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT* workspace,
                   size_t worksize,
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  bool perform_sqrt   = false;
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  distance_impl_l2_expanded(
    perform_sqrt, x, y, out, m, n, k, workspace, worksize, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::L2SqrtExpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT* workspace,
                   size_t worksize,
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  bool perform_sqrt   = true;
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  distance_impl_l2_expanded(
    perform_sqrt, x, y, out, m, n, k, workspace, worksize, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::L2Unexpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  bool perform_sqrt = false;
  ops::l2_unexp_distance_op<DataT, AccT, IdxT> l2_op(perform_sqrt);

  // The unexpanded L2 does not require the norms of a and b to be calculated.
  const OutT* x_norm = nullptr;
  const OutT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  pairwise_matrix_dispatch<decltype(l2_op), DataT, AccT, OutT, FinOpT, IdxT>(
    l2_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::L2SqrtUnexpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  bool perform_sqrt = true;
  ops::l2_unexp_distance_op<DataT, AccT, IdxT> l2_op(perform_sqrt);

  // The unexpanded L2 does not require the norms of a and b to be calculated.
  const OutT* x_norm = nullptr;
  const OutT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  pairwise_matrix_dispatch<decltype(l2_op), DataT, AccT, OutT, FinOpT, IdxT>(
    l2_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::Linf> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  ops::l_inf_distance_op<DataT, AccT, IdxT> distance_op{};

  const OutT* x_norm = nullptr;
  const OutT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  pairwise_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::LpUnexpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT metric_arg)
{
  ops::lp_unexp_distance_op<DataT, AccT, IdxT> distance_op{metric_arg};

  const OutT* x_norm = nullptr;
  const OutT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  pairwise_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void distance_impl(raft::resources const& handle,
                   distance_tag<DistanceType::RusselRaoExpanded> distance_type,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   AccT*,   // workspace unused
                   size_t,  // worksize unused
                   FinOpT fin_op,
                   bool is_row_major,
                   DataT)  // metric_arg unused
{
  ops::russel_rao_distance_op<DataT, AccT, IdxT> distance_op{k};

  const OutT* x_norm = nullptr;
  const OutT* y_norm = nullptr;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  pairwise_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

/**
 * @brief Evaluate pairwise distances with the user epilogue lambda allowed
 * @tparam DistanceType which distance to evaluate
 * @tparam InType input argument type
 * @tparam AccType accumulation type
 * @tparam OutType output type
 * @tparam FinalLambda user-defined epilogue lambda
 * @tparam Index_ Index type
 *
 * @param x first set of points
 * @param y second set of points
 * @param out output distance matrix
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param fin_op the final gemm epilogue lambda
 * @param stream cuda stream
 * @param isRowMajor whether the matrices are row-major or col-major
 *
 * @note fin_op: This is a device lambda which is supposed to operate upon the
 * input which is AccType and returns the output in OutType. It's signature is
 * as follows:  <pre>OutType fin_op(AccType in, int g_idx);</pre>. If one needs
 * any other parameters, feel free to pass them via closure.
 */
template <cuvs::distance::DistanceType distanceType,
          typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_ = int>
void distance(raft::resources const& handle,
              const InType* x,
              const InType* y,
              OutType* out,
              Index_ m,
              Index_ n,
              Index_ k,
              void* workspace,
              size_t worksize,
              FinalLambda fin_op,
              bool isRowMajor    = true,
              OutType metric_arg = 2.0f)
{
  // raft distance support inputs as float/double and output as uint8_t/float/double.
  static_assert(!((sizeof(OutType) > 1) && (sizeof(AccType) != sizeof(OutType))),
                "OutType can be uint8_t, float, double,"
                "if sizeof(OutType) > 1 then sizeof(AccType) == sizeof(OutType).");

  distance_impl<InType, AccType, OutType, FinalLambda, Index_>(
    handle,
    distance_tag<distanceType>{},
    x,
    y,
    out,
    m,
    n,
    k,
    reinterpret_cast<AccType*>(workspace),
    worksize,
    fin_op,
    isRowMajor,
    metric_arg);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Evaluate pairwise distances for the simple use case
 * @tparam DistanceType which distance to evaluate
 * @tparam InType input argument type
 * @tparam AccType accumulation type
 * @tparam OutType output type
 * @tparam Index_ Index type
 * @param x first set of points
 * @param y second set of points
 * @param dist output distance matrix
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param stream cuda stream
 * @param isRowMajor whether the matrices are row-major or col-major
 */
template <cuvs::distance::DistanceType distanceType,
          typename InType,
          typename AccType,
          typename OutType,
          typename Index_ = int>
void distance(raft::resources const& handle,
              const InType* x,
              const InType* y,
              OutType* out,
              Index_ m,
              Index_ n,
              Index_ k,
              void* workspace,
              size_t worksize,
              bool isRowMajor    = true,
              OutType metric_arg = 2.0f)
{
  auto fin_op = raft::identity_op();

  distance<distanceType, InType, AccType, OutType, decltype(fin_op), Index_>(
    handle, x, y, out, m, n, k, workspace, worksize, fin_op, isRowMajor, metric_arg);
}

/**
 * @brief Return the exact workspace size to compute the distance
 * @tparam DistanceType which distance to evaluate
 * @tparam InType input argument type
 * @tparam AccType accumulation type
 * @tparam OutType output type
 * @tparam Index_ Index type
 * @param x first set of points
 * @param y second set of points
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 *
 * @note If the specified distanceType doesn't need the workspace at all, it
 * returns 0.
 */
template <cuvs::distance::DistanceType distanceType,
          typename InType,
          typename AccType,
          typename OutType,
          typename Index_ = int>
size_t getWorkspaceSize(const InType* x, const InType* y, Index_ m, Index_ n, Index_ k)
{
  size_t worksize             = 0;
  constexpr bool is_allocated = (distanceType <= cuvs::distance::DistanceType::CosineExpanded) ||
                                (distanceType == cuvs::distance::DistanceType::CorrelationExpanded);
  constexpr int numOfBuffers =
    (distanceType == cuvs::distance::DistanceType::CorrelationExpanded) ? 2 : 1;

  if (is_allocated) {
    // TODO : when X == Y allocate std::max(m, n) instead of m + n when column major input
    // accuracy issue is resolved until then we allocate as m + n.
    worksize += numOfBuffers * m * sizeof(AccType);
    worksize += numOfBuffers * n * sizeof(AccType);
  }

  return worksize;
}

};  // namespace detail
};  // namespace distance
};  // namespace cuvs
