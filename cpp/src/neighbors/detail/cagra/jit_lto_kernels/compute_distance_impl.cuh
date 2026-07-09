/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "extern_device_functions.cuh"

#include "../../neighbors_device_intrinsics.cuh"
#include "../compute_distance_standard-impl.cuh"
#include "../compute_distance_standard.hpp"
#include "../compute_distance_vpq-impl.cuh"
#include "../compute_distance_vpq.hpp"
#include "../device_memory_ops.hpp"

#include <raft/util/integer_utils.hpp>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <typename DescriptorT>
RAFT_DEVICE_INLINE_FUNCTION auto compute_distance_standard_worker_impl(
  const typename DescriptorT::DATA_T* __restrict__ dataset_ptr,
  uint32_t dim,
  uint32_t query_smem_ptr) -> typename DescriptorT::DISTANCE_T
{
  using DATA_T                    = typename DescriptorT::DATA_T;
  using DISTANCE_T                = typename DescriptorT::DISTANCE_T;
  using LOAD_T                    = typename DescriptorT::LOAD_T;
  using QUERY_T                   = typename DescriptorT::QUERY_T;
  constexpr auto kTeamSize        = DescriptorT::kTeamSize;
  constexpr auto kDatasetBlockDim = DescriptorT::kDatasetBlockDim;
  constexpr auto vlen             = device::get_vlen<LOAD_T, DATA_T>();
  constexpr auto reg_nelem =
    raft::div_rounding_up_unsafe<uint32_t>(kDatasetBlockDim, kTeamSize * vlen);

  DISTANCE_T r = 0;
  for (uint32_t elem_offset = (threadIdx.x % kTeamSize) * vlen; elem_offset < dim;
       elem_offset += kDatasetBlockDim) {
    DATA_T data[reg_nelem][vlen];
#pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) {
      const uint32_t k = e * (kTeamSize * vlen) + elem_offset;
      if (k >= dim) break;
      device::ldg_cg(reinterpret_cast<LOAD_T&>(data[e]),
                     reinterpret_cast<const LOAD_T*>(dataset_ptr + k));
    }
#pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) {
      const uint32_t k = e * (kTeamSize * vlen) + elem_offset;
      if (k >= dim) break;
#pragma unroll
      for (uint32_t v = 0; v < vlen; v++) {
        QUERY_T d;
        device::lds(
          d,
          query_smem_ptr +
            sizeof(QUERY_T) * device::swizzling<kDatasetBlockDim, vlen * kTeamSize>(k + v));
        r += dist_op<QUERY_T, DISTANCE_T>(
          d, cuvs::spatial::knn::detail::utils::mapping<QUERY_T>{}(data[e][v]));
      }
    }
  }
  return r;
}

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto compute_distance_standard_impl(
  const typename DescriptorT::args_t args, const typename DescriptorT::INDEX_T dataset_index) ->
  typename DescriptorT::DISTANCE_T
{
  auto distance = compute_distance_standard_worker_impl<DescriptorT>(
    DescriptorT::ptr(args) + (static_cast<std::uint64_t>(DescriptorT::ld(args)) * dataset_index),
    args.dim,
    args.smem_ws_ptr);

  distance =
    apply_normalization_standard<DescriptorT::kTeamSize,
                                 DescriptorT::kDatasetBlockDim,
                                 typename DescriptorT::DATA_T,
                                 typename DescriptorT::INDEX_T,
                                 typename DescriptorT::DISTANCE_T,
                                 typename DescriptorT::QUERY_T>(distance, args, dataset_index);

  return distance;
}

template <typename DescriptorT>
_RAFT_DEVICE RAFT_DEVICE_INLINE_FUNCTION auto compute_distance_vpq_worker_impl(
  const uint8_t* __restrict__ dataset_ptr,
  const typename DescriptorT::CODE_BOOK_T* __restrict__ vq_code_book_ptr,
  uint32_t dim,
  uint32_t pq_codebook_ptr) -> typename DescriptorT::DISTANCE_T
{
  using DISTANCE_T               = typename DescriptorT::DISTANCE_T;
  using QUERY_T                  = typename DescriptorT::QUERY_T;
  using CODE_BOOK_T              = typename DescriptorT::CODE_BOOK_T;
  constexpr auto TeamSize        = DescriptorT::kTeamSize;
  constexpr auto DatasetBlockDim = DescriptorT::kDatasetBlockDim;
  constexpr auto PQ_BITS         = DescriptorT::kPqBits;
  constexpr auto PQ_LEN          = DescriptorT::kPqLen;
  constexpr auto SmemDType       = DescriptorT::kSmemDType;
  using PQ_CODEBOOK_LOAD_T       = uint32_t;

  using smem_val_config                  = vpq_smem_value_config<PQ_LEN, SmemDType>;
  using smem_val_pack_t                  = typename smem_val_config::smem_val_pack_t;
  using smem_val_pack_uint_t             = typename smem_val_config::smem_val_pack_uint_t;
  constexpr uint32_t num_packed_elements = smem_val_config::num_packed_elements;

  const uint32_t query_ptr = pq_codebook_ptr + DescriptorT::kSMemCodeBookSizeInBytes;
  static_assert(PQ_BITS == 8, "Only pq_bits == 8 is supported at the moment.");
  constexpr uint32_t vlen = utils::size_of<PQ_CODEBOOK_LOAD_T>() / utils::size_of<uint8_t>();
  constexpr uint32_t nelem =
    raft::div_rounding_up_unsafe<uint32_t>(DatasetBlockDim / PQ_LEN, TeamSize * vlen);

  constexpr auto kTeamMask = DescriptorT::kTeamSize - 1;
  constexpr auto kTeamVLen = TeamSize * vlen;

  const auto n_subspace = raft::div_rounding_up_unsafe(dim, PQ_LEN);
  const auto laneId     = threadIdx.x & kTeamMask;
  DISTANCE_T norm       = 0;
  for (uint32_t elem_offset = 0; elem_offset * PQ_LEN < dim;
       elem_offset += DatasetBlockDim / PQ_LEN) {
    PQ_CODEBOOK_LOAD_T pq_codes[nelem];
#pragma unroll
    for (std::uint32_t e = 0; e < nelem; e++) {
      const std::uint32_t k = e * kTeamVLen + elem_offset + laneId * vlen;
      if (k >= n_subspace) break;
      device::ldg_cg(pq_codes[e], reinterpret_cast<const PQ_CODEBOOK_LOAD_T*>(dataset_ptr + 4 + k));
    }
    //
    if constexpr (PQ_LEN % 2 == 0) {
#pragma unroll
      for (std::uint32_t e = 0; e < nelem; e++) {
        const std::uint32_t k = e * kTeamVLen + elem_offset + laneId * vlen;
        if (k >= n_subspace) break;
        half2 vq_vals[PQ_LEN][vlen / 2];
#pragma unroll
        for (std::uint32_t m = 0; m < PQ_LEN; m++) {
          const uint32_t d = (vlen * m) + (PQ_LEN * k);
          if (d >= dim) break;
          device::ldg_ca(vq_vals[m], vq_code_book_ptr + d);
        }
        PQ_CODEBOOK_LOAD_T pq_code = pq_codes[e];
#pragma unroll
        for (std::uint32_t v = 0; v < vlen; v++) {
          if (PQ_LEN * (v + k) >= dim) break;
#pragma unroll
          for (std::uint32_t m = 0; m < PQ_LEN / num_packed_elements; m++) {
            constexpr uint32_t vq_val_pack_num_elements = 2;
            constexpr auto kQueryBlock                  = DatasetBlockDim / (vlen * PQ_LEN);
            std::uint32_t vq_half2_index =
              m * (num_packed_elements / vq_val_pack_num_elements) + (PQ_LEN / 2) * v;

            uint32_t query_val_index;
            if constexpr (num_packed_elements == 2) {
              query_val_index =
                vq_half2_index * kQueryBlock + elem_offset * (PQ_LEN / 2) + e * TeamSize + laneId;
            } else if constexpr (PQ_LEN == num_packed_elements) {
              query_val_index = elem_offset + v * (DatasetBlockDim / (num_packed_elements * vlen)) +
                                e * TeamSize + laneId;
            } else {
              const uint32_t query_vec_element_id =
                (elem_offset + e * vlen * TeamSize + v + laneId * vlen) * PQ_LEN /
                num_packed_elements;
              constexpr auto kStride = vlen * PQ_LEN / num_packed_elements;
              query_val_index =
                transpose<DatasetBlockDim / num_packed_elements, kStride>(query_vec_element_id);
            }

            if constexpr (num_packed_elements == 2) {
              smem_val_pack_t q2, c2;
              device::lds(q2, query_ptr + sizeof(smem_val_pack_t) * query_val_index);
              device::lds(c2,
                          pq_codebook_ptr +
                            sizeof(smem_val_pack_uint_t) * ((1 << PQ_BITS) * m + (pq_code & 0xff)));
              auto dist =
                q2 - c2 - reinterpret_cast<half2(&)[PQ_LEN * vlen / 2]>(vq_vals)[vq_half2_index];
              dist = dist * dist;
              norm += static_cast<DISTANCE_T>(dist.x + dist.y);
            } else if constexpr (num_packed_elements == 4 || num_packed_elements == 8) {
              smem_val_pack_t q_vec, c_vec;
              device::lds(q_vec.as_uint(),
                          query_ptr + sizeof(smem_val_pack_uint_t) * query_val_index);
              device::lds(c_vec.as_uint(),
                          pq_codebook_ptr +
                            sizeof(smem_val_pack_uint_t) * ((1 << PQ_BITS) * m + (pq_code & 0xff)));

              half2 q2, c2;
#pragma unroll
              for (uint32_t bi = 0; bi < num_packed_elements / 2; bi++) {
                q2 = q_vec.as_half2(bi);
                c2 = c_vec.as_half2(bi);
                auto dist =
                  q2 - c2 - reinterpret_cast<half2(&)[PQ_LEN * vlen / 2]>(vq_vals)[vq_half2_index];
                dist = dist * dist;
                norm += static_cast<DISTANCE_T>(dist.x + dist.y);
                vq_half2_index += 1;
              }
            }
          }
          pq_code >>= 8;
        }
      }
    } else {
#pragma unroll
      for (std::uint32_t e = 0; e < nelem; e++) {
        const std::uint32_t k = e * kTeamVLen + elem_offset + laneId * vlen;
        if (k >= n_subspace) break;
        CODE_BOOK_T vq_vals[PQ_LEN][vlen];
#pragma unroll
        for (std::uint32_t m = 0; m < PQ_LEN; m++) {
          const std::uint32_t d = (vlen * m) + (PQ_LEN * k);
          if (d >= dim) break;
          device::ldg_ca(vq_vals[m], vq_code_book_ptr + d);
        }
        std::uint32_t pq_code = pq_codes[e];
#pragma unroll
        for (std::uint32_t v = 0; v < vlen; v++) {
          if (PQ_LEN * (v + k) >= dim) break;
          CODE_BOOK_T pq_vals[PQ_LEN];
          device::lds(pq_vals, pq_codebook_ptr + sizeof(CODE_BOOK_T) * PQ_LEN * (pq_code & 0xff));
#pragma unroll
          for (std::uint32_t m = 0; m < PQ_LEN; m++) {
            const std::uint32_t d1 = m + (PQ_LEN * v);
            const std::uint32_t d  = d1 + (PQ_LEN * k);
            DISTANCE_T diff;
            device::lds(diff, query_ptr + sizeof(QUERY_T) * d);
            diff -= static_cast<DISTANCE_T>(pq_vals[m]);
            diff -=
              static_cast<DISTANCE_T>(reinterpret_cast<CODE_BOOK_T(&)[PQ_LEN * vlen]>(vq_vals)[d1]);
            norm += diff * diff;
          }
          pq_code >>= 8;
        }
      }
    }
  }
  return norm;
}

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto compute_distance_vpq_impl(
  const typename DescriptorT::args_t args, const typename DescriptorT::INDEX_T dataset_index) ->
  typename DescriptorT::DISTANCE_T
{
  const auto* dataset_ptr =
    DescriptorT::encoded_dataset_ptr(args) +
    (static_cast<std::uint64_t>(DescriptorT::encoded_dataset_dim(args)) * dataset_index);
  uint32_t vq_code;
  device::ldg_cg(vq_code, reinterpret_cast<const std::uint32_t*>(dataset_ptr));
  return compute_distance_vpq_worker_impl<DescriptorT>(
    dataset_ptr,
    DescriptorT::vq_code_book_ptr(args) + args.dim * vq_code,
    args.dim,
    args.smem_ws_ptr);
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
__device__ DistanceT compute_distance_impl(
  const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
  IndexT dataset_index)
{
  if constexpr (PQ_BITS == 0 && PQ_LEN == 0 && std::is_same_v<CodebookT, void>) {
    using desc_t =
      standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, QueryT>;
    return compute_distance_standard_impl<desc_t>(args, dataset_index);
  } else if constexpr (PQ_BITS > 0 && PQ_LEN > 0 && std::is_same_v<CodebookT, half> &&
                       std::is_same_v<QueryT, half>) {
    using desc_t = cagra_q_dataset_descriptor_t<TeamSize,
                                                DatasetBlockDim,
                                                PQ_BITS,
                                                PQ_LEN,
                                                CodebookT,
                                                DataT,
                                                IndexT,
                                                DistanceT,
                                                QueryT,
                                                SmemDType>;
    return compute_distance_vpq_impl<desc_t>(args, dataset_index);
  } else {
    static_assert(sizeof(TeamSize) == 0,
                  "compute_distance_impl: unsupported PQ_BITS/PQ_LEN/CodebookT/QueryT for CAGRA "
                  "JIT descriptor");
    return DistanceT{};
  }
}

}  // namespace cuvs::neighbors::cagra::detail
