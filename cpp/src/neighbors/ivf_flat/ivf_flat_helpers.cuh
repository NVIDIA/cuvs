/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_flat.hpp>

#include "../detail/ann_utils.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/pow2_utils.cuh>
#include <variant>

namespace cuvs::neighbors::ivf_flat::helpers::codepacker {

namespace {
template <typename T>
__device__ void pack_1(const T* flat_code, T* block, uint32_t dim, uint32_t veclen, uint32_t offset)
{
  // The data is written in interleaved groups of `index::kGroupSize` vectors
  using interleaved_group = raft::Pow2<kIndexGroupSize>;

  // Interleave dimensions of the source vector while recording it.
  // NB: such `veclen` is selected, that `dim % veclen == 0`
  auto group_offset = interleaved_group::roundDown(offset);
  auto ingroup_id   = interleaved_group::mod(offset) * veclen;

  for (uint32_t l = 0; l < dim; l += veclen) {
    for (uint32_t j = 0; j < veclen; j++) {
      block[group_offset * dim + l * kIndexGroupSize + ingroup_id + j] = flat_code[l + j];
    }
  }
}

template <typename T>
__device__ void unpack_1(
  const T* block, T* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset)
{
  // The data is written in interleaved groups of `index::kGroupSize` vectors
  using interleaved_group = raft::Pow2<kIndexGroupSize>;

  // NB: such `veclen` is selected, that `dim % veclen == 0`
  auto group_offset = interleaved_group::roundDown(offset);
  auto ingroup_id   = interleaved_group::mod(offset) * veclen;

  for (uint32_t l = 0; l < dim; l += veclen) {
    for (uint32_t j = 0; j < veclen; j++) {
      flat_code[l + j] = block[group_offset * dim + l * kIndexGroupSize + ingroup_id + j];
    }
  }
}

template <typename T>
RAFT_KERNEL pack_interleaved_list_kernel(const T* codes,
                                         T* list_data,
                                         uint32_t n_rows,
                                         uint32_t dim,
                                         uint32_t veclen,
                                         std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  uint32_t tid          = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t dst_ix = std::holds_alternative<uint32_t>(offset_or_indices)
                            ? std::get<uint32_t>(offset_or_indices) + tid
                            : std::get<const uint32_t*>(offset_or_indices)[tid];
  if (tid < n_rows) { pack_1(codes + tid * dim, list_data, dim, veclen, dst_ix); }
}

template <typename T>
RAFT_KERNEL unpack_interleaved_list_kernel(
  const T* list_data,
  T* codes,
  uint32_t n_rows,
  uint32_t dim,
  uint32_t veclen,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  uint32_t tid          = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t src_ix = std::holds_alternative<uint32_t>(offset_or_indices)
                            ? std::get<uint32_t>(offset_or_indices) + tid
                            : std::get<const uint32_t*>(offset_or_indices)[tid];
  if (tid < n_rows) { unpack_1(list_data, codes + tid * dim, dim, veclen, src_ix); }
}

template <typename T, typename IdxT>
void pack_list_data(
  raft::resources const& res,
  raft::device_matrix_view<const T, uint32_t, raft::row_major> codes,
  uint32_t veclen,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  raft::device_mdspan<T, typename list_spec<uint32_t, T, IdxT>::list_extents, raft::row_major>
    list_data)
{
  uint32_t n_rows = codes.extent(0);
  uint32_t dim    = codes.extent(1);
  if (n_rows == 0 || dim == 0) return;
  static constexpr uint32_t kBlockSize = 256;
  dim3 blocks(raft::div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto stream = raft::resource::get_cuda_stream(res);
  pack_interleaved_list_kernel<<<blocks, threads, 0, stream.get()>>>(
    codes.data_handle(), list_data.data_handle(), n_rows, dim, veclen, offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T, typename IdxT>
void unpack_list_data(
  raft::resources const& res,
  raft::device_mdspan<const T, typename list_spec<uint32_t, T, IdxT>::list_extents, raft::row_major>
    list_data,
  uint32_t veclen,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  raft::device_matrix_view<T, uint32_t, raft::row_major> codes)
{
  uint32_t n_rows = codes.extent(0);
  uint32_t dim    = codes.extent(1);
  if (n_rows == 0 || dim == 0) return;
  static constexpr uint32_t kBlockSize = 256;
  dim3 blocks(raft::div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto stream = raft::resource::get_cuda_stream(res);
  unpack_interleaved_list_kernel<<<blocks, threads, 0, stream.get()>>>(
    list_data.data_handle(), codes.data_handle(), n_rows, dim, veclen, offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace

namespace detail {
template <typename T, typename IdxT>
void pack(
  raft::resources const& res,
  raft::device_matrix_view<const T, uint32_t, raft::row_major> codes,
  uint32_t veclen,
  uint32_t offset,
  raft::device_mdspan<T, typename list_spec<uint32_t, T, IdxT>::list_extents, raft::row_major>
    list_data)
{
  pack_list_data<T, IdxT>(res, codes, veclen, offset, list_data);
}

template <typename T, typename IdxT>
void unpack(
  raft::resources const& res,
  raft::device_mdspan<const T, typename list_spec<uint32_t, T, IdxT>::list_extents, raft::row_major>
    list_data,
  uint32_t veclen,
  uint32_t offset,
  raft::device_matrix_view<T, uint32_t, raft::row_major> codes)
{
  unpack_list_data<T, IdxT>(res, list_data, veclen, offset, codes);
}

template <typename T>
void pack_1(const T* flat_code, T* block, uint32_t dim, uint32_t veclen, uint32_t offset)
{
  // The data is written in interleaved groups of `index::kGroupSize` vectors
  using interleaved_group = raft::Pow2<kIndexGroupSize>;

  // Interleave dimensions of the source vector while recording it.
  // NB: such `veclen` is selected, that `dim % veclen == 0`
  auto group_offset = interleaved_group::roundDown(offset);
  auto ingroup_id   = interleaved_group::mod(offset) * veclen;

  for (uint32_t l = 0; l < dim; l += veclen) {
    for (uint32_t j = 0; j < veclen; j++) {
      block[group_offset * dim + l * kIndexGroupSize + ingroup_id + j] = flat_code[l + j];
    }
  }
}

template <typename T>
void unpack_1(const T* block, T* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset)
{
  // The data is written in interleaved groups of `index::kGroupSize` vectors
  using interleaved_group = raft::Pow2<kIndexGroupSize>;

  // NB: such `veclen` is selected, that `dim % veclen == 0`
  auto group_offset = interleaved_group::roundDown(offset);
  auto ingroup_id   = interleaved_group::mod(offset) * veclen;

  for (uint32_t l = 0; l < dim; l += veclen) {
    for (uint32_t j = 0; j < veclen; j++) {
      flat_code[l + j] = block[group_offset * dim + l * kIndexGroupSize + ingroup_id + j];
    }
  }
}

}  // namespace detail
}  // namespace cuvs::neighbors::ivf_flat::helpers::codepacker
