/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::detail {
template <typename OutType,
          typename Func,
          typename... InTypes,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InTypes...>>
CUVS_EXPORT void map(const raft::resources& res, OutType out, Func f, InTypes... ins);

template <typename InType1,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1>>
CUVS_EXPORT void map(const raft::resources& res, InType1 in1, OutType out, Func f);

template <typename InType1,
          typename InType2,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1, InType2>>
CUVS_EXPORT void map(const raft::resources& res, InType1 in1, InType2 in2, OutType out, Func f);

template <typename InType1,
          typename InType2,
          typename InType3,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1, InType2, InType3>>
CUVS_EXPORT void map(
  const raft::resources& res, InType1 in1, InType2 in2, InType3 in3, OutType out, Func f);

template <typename OutType,
          typename Func,
          typename... InTypes,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InTypes...>>
void map_offset(const raft::resources& res, OutType out, Func f, InTypes... ins);

template <typename InType1,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1>>
void map_offset(const raft::resources& res, InType1 in1, OutType out, Func f);

template <typename InType1,
          typename InType2,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1, InType2>>
void map_offset(const raft::resources& res, InType1 in1, InType2 in2, OutType out, Func f);

template <typename InType1,
          typename InType2,
          typename InType3,
          typename OutType,
          typename Func,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType1, InType2, InType3>>
void map_offset(
  const raft::resources& res, InType1 in1, InType2 in2, InType3 in3, OutType out, Func f);
}  // namespace cuvs::detail

namespace cuvs::detail {
template <typename InT, typename OutT = InT, typename IdxType = int>
CUVS_EXPORT void add(OutT* out, const InT* in1, const InT* in2, IdxType len, cudaStream_t stream);

template <typename InType,
          typename OutType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
CUVS_EXPORT void add(raft::resources const& handle, InType in1, InType in2, OutType out);

template <typename InT, typename OutT = InT, typename IdxType = int>
void addScalar(OutT* out, const InT* in, const InT scalar, IdxType len, cudaStream_t stream);

template <typename InType,
          typename OutType,
          typename ScalarIdxType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void multiply_scalar(
  raft::resources const& handle,
  InType in,
  OutType out,
  raft::host_scalar_view<const typename InType::value_type, ScalarIdxType> scalar);

template <typename InType,
          typename OutType,
          typename ScalarIdxType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void power_scalar(
  raft::resources const& handle,
  InType in,
  OutType out,
  const raft::host_scalar_view<const typename InType::value_type, ScalarIdxType> scalar);
}  // namespace cuvs::detail

namespace cuvs::detail {
template <typename InType,
          typename Lambda,
          typename IdxType = int,
          typename OutType = InType,
          int TPB          = 256>
void unaryOp(OutType* out, const InType* in, IdxType len, Lambda op, cudaStream_t stream);
}

namespace cuvs::cluster::kmeans::detail {
template <typename LabelT, typename IndexT, typename IndicesIterator>
struct CopyClusterLabelOp {
  IndicesIterator indices;

  __device__ LabelT operator()(IndexT offset) const;
};
}  // namespace cuvs::cluster::kmeans::detail
