/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_distance_vpq.hpp"
#include "packed_type.hpp"

#include <cuvs/distance/distance.hpp>
#include <raft/util/pow2_utils.cuh>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <uint32_t PQ_LEN, cuvs::neighbors::cagra::internal_dtype SmemDType, class Enable = void>
struct vpq_smem_value_config;

template <uint32_t PQ_LEN, cuvs::neighbors::cagra::internal_dtype SmemDType>
struct vpq_smem_value_config<
  PQ_LEN,
  SmemDType,
  std::enable_if_t<PQ_LEN == 2 || SmemDType == cuvs::neighbors::cagra::internal_dtype::F16>> {
  using smem_val_pack_t                         = half2;
  using smem_val_t                              = half;
  using smem_val_pack_uint_t                    = uint32_t;
  static constexpr uint32_t num_packed_elements = 2;
};

template <uint32_t PQ_LEN>
struct vpq_smem_value_config<PQ_LEN,
                             cuvs::neighbors::cagra::internal_dtype::E5M2,
                             std::enable_if_t<PQ_LEN == 4 || PQ_LEN == 8>> {
  using smem_val_pack_t                         = device::fp8xN<PQ_LEN, 5>;
  using smem_val_t                              = typename smem_val_pack_t::unit_t;
  using smem_val_pack_uint_t                    = typename smem_val_pack_t::uint_t;
  static constexpr uint32_t num_packed_elements = smem_val_pack_t::num_elements;
};

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
struct cagra_q_dataset_descriptor_t : public dataset_descriptor_base_t<DataT, IndexT, DistanceT> {
  using base_type   = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  using CODE_BOOK_T = CodebookT;
  using QUERY_T     = QueryT;
  using base_type::args;
  using base_type::extra_ptr3;
  using typename base_type::args_t;
  using typename base_type::DATA_T;
  using typename base_type::DISTANCE_T;
  using typename base_type::INDEX_T;
  using typename base_type::LOAD_T;
  constexpr static inline auto kTeamSize        = TeamSize;
  constexpr static inline auto kDatasetBlockDim = DatasetBlockDim;
  constexpr static inline auto kPqBits          = PQ_BITS;
  constexpr static inline auto kPqLen           = PQ_LEN;
  constexpr static inline auto kSmemDType       = SmemDType;

  static_assert(std::is_same_v<CODE_BOOK_T, half>, "Only CODE_BOOK_T = `half` is supported now");

  RAFT_INLINE_FUNCTION static constexpr auto encoded_dataset_ptr(args_t& args) noexcept
    -> const uint8_t*&
  {
    return (const uint8_t*&)args.extra_ptr1;
  }
  RAFT_INLINE_FUNCTION static constexpr auto vq_code_book_ptr(args_t& args) noexcept
    -> const CODE_BOOK_T*&
  {
    return (const CODE_BOOK_T*&)args.extra_ptr2;
  }
  RAFT_INLINE_FUNCTION constexpr auto pq_code_book_ptr() noexcept -> const CODE_BOOK_T*&
  {
    return (const CODE_BOOK_T*&)extra_ptr3;
  }
  RAFT_INLINE_FUNCTION static constexpr auto encoded_dataset_dim(args_t& args) noexcept -> uint32_t&
  {
    return args.extra_word1;
  }

  RAFT_INLINE_FUNCTION static constexpr auto encoded_dataset_ptr(const args_t& args) noexcept
    -> const uint8_t* const&
  {
    return (const uint8_t*&)args.extra_ptr1;
  }
  RAFT_INLINE_FUNCTION static constexpr auto vq_code_book_ptr(const args_t& args) noexcept
    -> const CODE_BOOK_T* const&
  {
    return (const CODE_BOOK_T*&)args.extra_ptr2;
  }
  RAFT_INLINE_FUNCTION constexpr auto pq_code_book_ptr() const noexcept -> const CODE_BOOK_T* const&
  {
    return (const CODE_BOOK_T*&)extra_ptr3;
  }
  RAFT_INLINE_FUNCTION static constexpr auto encoded_dataset_dim(const args_t& args) noexcept
    -> const uint32_t&
  {
    return args.extra_word1;
  }

  using smem_val_config = vpq_smem_value_config<PQ_LEN, SmemDType>;

  static constexpr std::uint32_t kSMemCodeBookSizeInBytes =
    (1 << PQ_BITS) * PQ_LEN * utils::size_of<typename smem_val_config::smem_val_pack_uint_t>() /
    smem_val_config::num_packed_elements;

  _RAFT_HOST_DEVICE cagra_q_dataset_descriptor_t(const std::uint8_t* encoded_dataset_ptr,
                                                 std::uint32_t encoded_dataset_dim,
                                                 const CODE_BOOK_T* vq_code_book_ptr,
                                                 const CODE_BOOK_T* pq_code_book_ptr,
                                                 IndexT size,
                                                 std::uint32_t dim)
    : base_type(size, dim, raft::Pow2<TeamSize>::Log2, get_smem_ws_size_in_bytes(dim))
  {
    cagra_q_dataset_descriptor_t::encoded_dataset_ptr(args) = encoded_dataset_ptr;
    cagra_q_dataset_descriptor_t::vq_code_book_ptr(args)    = vq_code_book_ptr;
    this->pq_code_book_ptr()                                = pq_code_book_ptr;
    cagra_q_dataset_descriptor_t::encoded_dataset_dim(args) = encoded_dataset_dim;
    static_assert(sizeof(*this) == sizeof(base_type));
    static_assert(alignof(cagra_q_dataset_descriptor_t) == alignof(base_type));
  }

  // Public static method to compute smem size without constructing descriptor
  RAFT_INLINE_FUNCTION constexpr static auto get_smem_ws_size_in_bytes(uint32_t dim) -> uint32_t
  {
    /* SMEM workspace layout:
      1. The descriptor itself
      2. Codebook (kSMemCodeBookSizeInBytes bytes)
      3. Queries (smem_query_buffer_length elems)
    */
    return sizeof(cagra_q_dataset_descriptor_t) + kSMemCodeBookSizeInBytes +
           raft::round_up_safe<uint32_t>(dim, DatasetBlockDim) *
             utils::size_of<typename smem_val_config::smem_val_pack_uint_t>() /
             smem_val_config::num_packed_elements;
  }

 private:
};

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PqBits,
          uint32_t PqLen,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          cuvs::neighbors::cagra::internal_dtype SmemDType>
RAFT_KERNEL __launch_bounds__(1, 1)
  vpq_dataset_descriptor_init_kernel(dataset_descriptor_base_t<DataT, IndexT, DistanceT>* out,
                                     const std::uint8_t* encoded_dataset_ptr,
                                     uint32_t encoded_dataset_dim,
                                     const CodebookT* vq_code_book_ptr,
                                     const CodebookT* pq_code_book_ptr,
                                     IndexT size,
                                     uint32_t dim)
{
  using desc_type = cagra_q_dataset_descriptor_t<TeamSize,
                                                 DatasetBlockDim,
                                                 PqBits,
                                                 PqLen,
                                                 CodebookT,
                                                 DataT,
                                                 IndexT,
                                                 DistanceT,
                                                 half,
                                                 SmemDType>;
  new (out) desc_type(
    encoded_dataset_ptr, encoded_dataset_dim, vq_code_book_ptr, pq_code_book_ptr, size, dim);
}

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PqBits,
          uint32_t PqLen,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          cuvs::neighbors::cagra::internal_dtype SmemDType>
dataset_descriptor_host<DataT, IndexT, DistanceT>
vpq_descriptor_spec<Metric,
                    TeamSize,
                    DatasetBlockDim,
                    PqBits,
                    PqLen,
                    CodebookT,
                    DataT,
                    IndexT,
                    DistanceT,
                    SmemDType>::init_(const cagra::search_params& params,
                                      const std::uint8_t* encoded_dataset_ptr,
                                      uint32_t encoded_dataset_dim,
                                      const CodebookT* vq_code_book_ptr,
                                      const CodebookT* pq_code_book_ptr,
                                      IndexT size,
                                      uint32_t dim)
{
  using desc_type = cagra_q_dataset_descriptor_t<TeamSize,
                                                 DatasetBlockDim,
                                                 PqBits,
                                                 PqLen,
                                                 CodebookT,
                                                 DataT,
                                                 IndexT,
                                                 DistanceT,
                                                 half,
                                                 SmemDType>;

  return host_type{
    desc_type{
      encoded_dataset_ptr, encoded_dataset_dim, vq_code_book_ptr, pq_code_book_ptr, size, dim},
    [=](dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dev_ptr,
        rmm::cuda_stream_view stream) {
      vpq_dataset_descriptor_init_kernel<Metric,
                                         TeamSize,
                                         DatasetBlockDim,
                                         PqBits,
                                         PqLen,
                                         CodebookT,
                                         DataT,
                                         IndexT,
                                         DistanceT,
                                         SmemDType><<<1, 1, 0, stream>>>(dev_ptr,
                                                                         encoded_dataset_ptr,
                                                                         encoded_dataset_dim,
                                                                         vq_code_book_ptr,
                                                                         pq_code_book_ptr,
                                                                         size,
                                                                         dim);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    },
    Metric,
    DatasetBlockDim,
    true,        // is_vpq
    PqBits,      // pq_bits
    PqLen,       // pq_len
    SmemDType};  // smem_dtype
}

}  // namespace cuvs::neighbors::cagra::detail
