/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../neighbors_device_intrinsics.cuh"
#include "../compute_distance_standard-impl.cuh"
#include "../compute_distance_standard.hpp"
#include "../compute_distance_vpq-impl.cuh"
#include "../compute_distance_vpq.hpp"
#include "../device_memory_ops.hpp"

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto setup_workspace_standard_impl(
  const DescriptorT* that,
  void* smem_ptr,
  const typename DescriptorT::DATA_T* queries_ptr,
  uint32_t query_id) -> const DescriptorT*
{
  using DATA_T                    = typename DescriptorT::DATA_T;
  using LOAD_T                    = typename DescriptorT::LOAD_T;
  using QUERY_T                   = typename DescriptorT::QUERY_T;
  using word_type                 = uint32_t;
  constexpr auto kTeamSize        = DescriptorT::kTeamSize;
  constexpr auto kDatasetBlockDim = DescriptorT::kDatasetBlockDim;
  auto* r                         = reinterpret_cast<DescriptorT*>(smem_ptr);
  auto* buf                       = reinterpret_cast<QUERY_T*>(r + 1);
  if (r != that) {
    constexpr uint32_t kCount = sizeof(DescriptorT) / sizeof(word_type);
    using blob_type           = word_type[kCount];
    auto& src                 = reinterpret_cast<const blob_type&>(*that);
    auto& dst                 = reinterpret_cast<blob_type&>(*r);
    for (uint32_t i = threadIdx.x; i < kCount; i += blockDim.x) {
      dst[i] = src[i];
    }
    const auto smem_ptr_offset =
      reinterpret_cast<uint8_t*>(&(r->args.smem_ws_ptr)) - reinterpret_cast<uint8_t*>(r);
    if (threadIdx.x == uint32_t(smem_ptr_offset / sizeof(word_type))) {
      r->args.smem_ws_ptr = uint32_t(__cvta_generic_to_shared(buf));
    }
    __syncthreads();
  }

  uint32_t dim        = r->args.dim;
  auto buf_len        = raft::round_up_safe<uint32_t>(dim, kDatasetBlockDim);
  constexpr auto vlen = device::get_vlen<LOAD_T, DATA_T>();
  queries_ptr += dim * query_id;
  for (unsigned i = threadIdx.x; i < buf_len; i += blockDim.x) {
    unsigned j = device::swizzling<kDatasetBlockDim, vlen * kTeamSize>(i);
    if (i < dim) {
      buf[j] = cuvs::spatial::knn::detail::utils::mapping<QUERY_T>{}(queries_ptr[i]);
    } else {
      buf[j] = 0;
    }
  }
  return r;
}

template <auto Block, auto Stride, typename T>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto transpose(T x) -> T
{
  auto i = x % Block;
  auto j = x / Block;
  auto k = i % Stride;
  auto l = i / Stride;
  return j * Block + k * (Block / Stride) + l;
}

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto setup_workspace_vpq_impl(
  const DescriptorT* that,
  void* smem_ptr,
  const typename DescriptorT::DATA_T* queries_ptr,
  uint32_t query_id) -> const DescriptorT*
{
  using QUERY_T                      = typename DescriptorT::QUERY_T;
  using word_type                    = uint32_t;
  constexpr auto kDatasetBlockDim    = DescriptorT::kDatasetBlockDim;
  constexpr auto PQ_BITS             = DescriptorT::kPqBits;
  constexpr auto PQ_LEN              = DescriptorT::kPqLen;
  constexpr auto SmemDType           = DescriptorT::kSmemDType;
  using smem_val_config              = vpq_smem_value_config<PQ_LEN, SmemDType>;
  using smem_val_t                   = typename smem_val_config::smem_val_t;
  using smem_val_pack_t              = typename smem_val_config::smem_val_pack_t;
  using smem_val_pack_uint_t         = typename smem_val_config::smem_val_pack_uint_t;
  constexpr auto num_packed_elements = smem_val_config::num_packed_elements;

  auto* r = reinterpret_cast<DescriptorT*>(smem_ptr);

  if (r != that) {
    constexpr uint32_t kCount = sizeof(DescriptorT) / sizeof(word_type);
    using blob_type           = word_type[kCount];
    auto& src                 = reinterpret_cast<const blob_type&>(*that);
    auto& dst                 = reinterpret_cast<blob_type&>(*r);
    for (uint32_t i = threadIdx.x; i < kCount; i += blockDim.x) {
      dst[i] = src[i];
    }

    auto codebook_buf = uint32_t(__cvta_generic_to_shared(r + 1));
    const auto smem_ptr_offset =
      reinterpret_cast<uint8_t*>(&(r->args.smem_ws_ptr)) - reinterpret_cast<uint8_t*>(r);
    if (threadIdx.x == uint32_t(smem_ptr_offset / sizeof(word_type))) {
      r->args.smem_ws_ptr = codebook_buf;
    }
    __syncthreads();

    for (unsigned i = threadIdx.x * num_packed_elements; i < (1 << PQ_BITS) * PQ_LEN;
         i += blockDim.x * num_packed_elements) {
      constexpr auto num_elements_per_bank =
        num_packed_elements / (utils::size_of<smem_val_pack_uint_t>() / utils::size_of<uint32_t>());

      if constexpr (PQ_LEN >= num_elements_per_bank) {
        constexpr auto num_banks_per_subspace = PQ_LEN / num_elements_per_bank;
        const auto j                          = i / num_elements_per_bank;
        const auto smem_index =
          (j / num_banks_per_subspace) + (j % num_banks_per_subspace) * (1 << PQ_BITS);

        if constexpr (num_packed_elements == 2) {
          smem_val_pack_t buf;
          buf.x = r->pq_code_book_ptr()[i];
          buf.y = r->pq_code_book_ptr()[i + 1];
          device::sts(codebook_buf + smem_index * sizeof(smem_val_pack_t), buf);
        } else if constexpr (num_packed_elements == 4 || num_packed_elements == 8) {
          smem_val_pack_t buf;
#pragma unroll
          for (uint32_t k = 0; k < num_packed_elements; k++) {
            buf.data.x1[k] =
              static_cast<smem_val_t>(static_cast<float>(r->pq_code_book_ptr()[i + k]));
          }
          device::sts(codebook_buf + smem_index * sizeof(smem_val_pack_uint_t), buf.as_uint());
        }
      }
    }
  }

  uint32_t dim = r->args.dim;
  queries_ptr += dim * query_id;

  constexpr cuvs::spatial::knn::detail::utils::mapping<QUERY_T> mapping{};
  auto smem_query_ptr =
    reinterpret_cast<smem_val_t*>(reinterpret_cast<uint8_t*>(smem_ptr) + sizeof(DescriptorT) +
                                  DescriptorT::kSMemCodeBookSizeInBytes);
  for (unsigned i = threadIdx.x * num_packed_elements; i < dim;
       i += blockDim.x * num_packed_elements) {
    smem_val_pack_t buf;
    if constexpr (num_packed_elements == 2) {
      buf.x = 0;
      buf.y = 0;
      if (i < dim) { buf.x = mapping(queries_ptr[i]); }
      if (i + 1 < dim) { buf.y = mapping(queries_ptr[i + 1]); }
    } else if constexpr (num_packed_elements == 4 || num_packed_elements == 8) {
#pragma unroll
      for (uint32_t k = 0; k < num_packed_elements; k++) {
        buf.data.x1[k] = static_cast<smem_val_t>(0.0f);
        if (i + k < dim) {
          buf.data.x1[k] = static_cast<smem_val_t>(static_cast<float>(mapping(queries_ptr[i + k])));
        }
      }
    }
    if constexpr ((PQ_BITS == 8) && (PQ_LEN % num_packed_elements == 0)) {
      constexpr uint32_t vlen = 4;  // **** DO NOT CHANGE ****
      constexpr auto kStride  = vlen * PQ_LEN / num_packed_elements;
      reinterpret_cast<smem_val_pack_t*>(
        smem_query_ptr)[transpose<kDatasetBlockDim / num_packed_elements, kStride>(
        i / num_packed_elements)] = buf;
    } else {
      (reinterpret_cast<smem_val_pack_t*>(smem_query_ptr + i))[0] = buf;
    }
  }

  return r;
}

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT,
          cuvs::neighbors::cagra::internal_dtype SmemDType>
__device__ const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* setup_workspace_impl(
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* desc_ptr,
  void* smem,
  const DataT* queries,
  uint32_t query_id)
{
  if constexpr (PQ_BITS == 0 && PQ_LEN == 0 && std::is_same_v<CodebookT, void>) {
    using desc_t =
      standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, QueryT>;
    const desc_t* desc = static_cast<const desc_t*>(desc_ptr);

    const desc_t* result = setup_workspace_standard_impl<desc_t>(desc, smem, queries, query_id);
    return static_cast<const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*>(result);
  } else if constexpr (PQ_BITS > 0 && PQ_LEN > 0 && std::is_same_v<CodebookT, half> &&
                       std::is_same_v<QueryT, half>) {
    using desc_t       = cagra_q_dataset_descriptor_t<TeamSize,
                                                      DatasetBlockDim,
                                                      PQ_BITS,
                                                      PQ_LEN,
                                                      CodebookT,
                                                      DataT,
                                                      IndexT,
                                                      DistanceT,
                                                      QueryT,
                                                      SmemDType>;
    const desc_t* desc = static_cast<const desc_t*>(desc_ptr);

    const desc_t* result = setup_workspace_vpq_impl<desc_t>(desc, smem, queries, query_id);
    return static_cast<const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*>(result);
  } else {
    static_assert(
      sizeof(DataT) == 0,
      "setup_workspace_impl: unsupported PQ_BITS/PQ_LEN/CodebookT/QueryT for CAGRA JIT");
    return nullptr;
  }
}

}  // namespace cuvs::neighbors::cagra::detail
