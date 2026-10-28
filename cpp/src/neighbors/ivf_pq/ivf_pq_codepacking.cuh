/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/device_atomics.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

#include <variant>

namespace cuvs::neighbors::ivf_pq::detail {

/** A chunk of PQ-encoded vector managed by one CUDA thread. */
using pq_vec_t = raft::TxN_t<uint8_t, kIndexGroupVecLen>::io_t;

/**
 * This type mimics the `uint8_t&` for the indexing operator of `bitfield_view_t`.
 *
 * @tparam Bits number of bits comprising the value.
 */
template <uint32_t Bits, typename PtrT = uint8_t>
struct bitfield_ref_t {
  static_assert(Bits <= 8 && Bits > 0, "Bit code must fit one byte");
  constexpr static uint8_t kMask = static_cast<uint8_t>((1u << Bits) - 1u);
  PtrT* ptr;
  uint32_t offset;

  constexpr operator uint8_t()  // NOLINT
  {
    auto pair = static_cast<uint16_t>(ptr[0]);
    if (offset + Bits > 8) { pair |= static_cast<uint16_t>(ptr[1]) << 8; }
    return static_cast<uint8_t>((pair >> offset) & kMask);
  }

  constexpr auto operator=(uint8_t code) -> bitfield_ref_t&
  {
    if (offset + Bits > 8) {
      auto pair = static_cast<uint16_t>(ptr[0]);
      pair |= static_cast<uint16_t>(ptr[1]) << 8;
      pair &= ~(static_cast<uint16_t>(kMask) << offset);
      pair |= static_cast<uint16_t>(code) << offset;
      ptr[0] = static_cast<uint8_t>(raft::Pow2<256>::mod(pair));
      ptr[1] = static_cast<uint8_t>(raft::Pow2<256>::div(pair));
    } else {
      ptr[0] = (ptr[0] & ~(kMask << offset)) | (code << offset);
    }
    return *this;
  }
};

/**
 * View a byte array as an array of unsigned integers of custom small bit size.
 *
 * @tparam Bits number of bits comprising a single element of the array.
 */
template <uint32_t Bits, typename PtrT = uint8_t>
struct bitfield_view_t {
  static_assert(Bits <= 8 && Bits > 0, "Bit code must fit one byte");
  PtrT* raw;

  constexpr auto operator[](uint32_t i) -> bitfield_ref_t<Bits, PtrT>
  {
    uint32_t bit_offset = i * Bits;
    return bitfield_ref_t<Bits, PtrT>{raw + raft::Pow2<8>::div(bit_offset),
                                      raft::Pow2<8>::mod(bit_offset)};
  }
};

/**
 * Process a single vector in a list.
 *
 * @tparam PqBits
 * @tparam Action tells how to process a single vector (e.g. reconstruct or just unpack)
 *
 * @param[in] in_list_data the encoded cluster data.
 * @param[in] in_ix in-cluster index of the vector to be decoded (one-per-thread).
 * @param[in] out_ix the output index passed to the action
 * @param[in] pq_dim
 * @param action a callable action to be invoked on each PQ code (component of the encoding)
 *    type: void (uint8_t code, uint32_t out_ix, uint32_t j), where j = [0..pq_dim).
 */
template <uint32_t PqBits, typename Action>
__device__ void run_on_vector(
  raft::device_mdspan<const uint8_t,
                      list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                      raft::row_major> in_list_data,
  uint32_t in_ix,
  uint32_t out_ix,
  uint32_t pq_dim,
  Action action)
{
  using group_align         = raft::Pow2<kIndexGroupSize>;
  const uint32_t group_ix   = group_align::div(in_ix);
  const uint32_t ingroup_ix = group_align::mod(in_ix);

  pq_vec_t code_chunk;
  bitfield_view_t<PqBits> code_view{reinterpret_cast<uint8_t*>(&code_chunk)};
  constexpr uint32_t kChunkSize = (sizeof(pq_vec_t) * 8u) / PqBits;
  for (uint32_t j = 0, i = 0; j < pq_dim; i++) {
    // read the chunk
    code_chunk = *reinterpret_cast<const pq_vec_t*>(&in_list_data(group_ix, i, ingroup_ix, 0));
    // read the codes, one/pq_dim at a time
#pragma unroll
    for (uint32_t k = 0; k < kChunkSize && j < pq_dim; k++, j++) {
      // read a piece of the reconstructed vector
      action(code_view[k], out_ix, j);
    }
  }
}

/**
 * Process a single vector in a list.
 *
 * @tparam PqBits
 * @tparam SubWarpSize how many threads work on the same ix (only the first thread writes data).
 * @tparam IdxT type of the index passed to the action
 * @tparam Action tells how to process a single vector (e.g. encode or just pack)
 *
 * @param[in] out_list_data the encoded cluster data.
 * @param[in] out_ix in-cluster index of the vector to be processed (one-per-SubWarpSize threads).
 * @param[in] in_ix the input index passed to the action (one-per-SubWarpSize threads).
 * @param[in] pq_dim
 * @param action a callable action to be invoked on each PQ code (component of the encoding)
 *    type: (uint32_t in_ix, uint32_t j) -> uint8_t, where j = [0..pq_dim).
 */
template <uint32_t PqBits, uint32_t SubWarpSize, typename IdxT, typename Action>
__device__ void write_vector_interleaved(
  raft::device_mdspan<uint8_t,
                      list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                      raft::row_major> out_list_data,
  uint32_t out_ix,
  IdxT in_ix,
  uint32_t pq_dim,
  Action action)
{
  const uint32_t lane_id = raft::Pow2<SubWarpSize>::mod(threadIdx.x);

  using group_align         = raft::Pow2<kIndexGroupSize>;
  const uint32_t group_ix   = group_align::div(out_ix);
  const uint32_t ingroup_ix = group_align::mod(out_ix);

  pq_vec_t code_chunk;
  bitfield_view_t<PqBits> code_view{reinterpret_cast<uint8_t*>(&code_chunk)};
  constexpr uint32_t kChunkSize = (sizeof(pq_vec_t) * 8u) / PqBits;
  for (uint32_t j = 0, i = 0; j < pq_dim; i++) {
    // clear the chunk
    if (lane_id == 0) { code_chunk = pq_vec_t{}; }
    // write the codes, one/pq_dim at a time
#pragma unroll
    for (uint32_t k = 0; k < kChunkSize && j < pq_dim; k++, j++) {
      // write a single code
      uint8_t code = action(in_ix, j);
      if (lane_id == 0) { code_view[k] = code; }
    }
    // write the chunk to the list
    if (lane_id == 0) {
      *reinterpret_cast<pq_vec_t*>(&out_list_data(group_ix, i, ingroup_ix, 0)) = code_chunk;
    }
  }
}

/**
 * Process a single vector in a list.
 *
 * @tparam PqBits
 * @tparam SubWarpSize how many threads work on the same ix (only the first thread writes data).
 * @tparam IdxT type of the index passed to the action
 * @tparam Action tells how to process a single vector (e.g. encode or just pack)
 *
 * @param[in] out_list_data the encoded cluster data.
 * @param[in] out_ix in-cluster index of the vector to be processed (one-per-SubWarpSize threads).
 * @param[in] in_ix the input index passed to the action (one-per-SubWarpSize threads).
 * @param[in] pq_dim
 * @param action a callable action to be invoked on each PQ code (component of the encoding)
 *    type: (uint32_t in_ix, uint32_t j) -> uint8_t, where j = [0..pq_dim).
 */
/**
 * Process a single vector for flat (non-interleaved) storage.
 *
 * @tparam PqBits
 * @tparam SubWarpSize how many threads cooperatively compute each code
 * @tparam IdxT type of the index passed to the action
 * @tparam Action tells how to process a single vector (e.g. encode or just pack)
 *
 * @param[in] out_flat_codes flat PQ codes buffer [n_vectors, bytes_per_vector]
 * @param[in] out_ix output vector index
 * @param[in] in_ix the input index passed to the action
 * @param[in] pq_dim
 * @param[in] bytes_per_vector number of bytes per encoded vector = ceildiv(pq_dim * PqBits, 8)
 * @param action a callable action to be invoked on each PQ code (component of the encoding)
 *    type: (IdxT in_ix, uint32_t j) -> uint8_t, where j = [0..pq_dim).
 */
template <uint32_t PqBits, uint32_t SubWarpSize, typename IdxT, typename Action>
__device__ void write_vector_flat(uint8_t* out_flat_codes,
                                  uint32_t out_ix,
                                  IdxT in_ix,
                                  uint32_t pq_dim,
                                  uint32_t bytes_per_vector,
                                  Action action)
{
  const uint32_t lane_id = raft::Pow2<SubWarpSize>::mod(threadIdx.x);

  uint8_t* out_vec_ptr = out_flat_codes + out_ix * bytes_per_vector;

  bitfield_view_t<PqBits> code_view{out_vec_ptr};
  for (uint32_t j = 0; j < pq_dim; j++) {
    uint8_t code = action(in_ix, j);
    if (lane_id == 0) { code_view[j] = code; }
  }
}

/** Process the given indices or a block of a single list (cluster). */
template <uint32_t PqBits, typename Action>
__device__ void run_on_list(
  raft::device_mdspan<const uint8_t,
                      list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                      raft::row_major> in_list_data,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t len,
  uint32_t pq_dim,
  Action action)
{
  // Grid-stride loop: each vector is processed by exactly one thread.
  for (uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x; ix < len;
       ix += blockDim.x * gridDim.x) {
    const uint32_t src_ix = std::holds_alternative<uint32_t>(offset_or_indices)
                              ? std::get<uint32_t>(offset_or_indices) + ix
                              : std::get<const uint32_t*>(offset_or_indices)[ix];
    run_on_vector<PqBits>(in_list_data, src_ix, ix, pq_dim, action);
  }
}

/** Process the given indices or a block of a single list (cluster) - interleaved layout. */
template <uint32_t PqBits, uint32_t SubWarpSize, typename Action>
__device__ void write_list(
  raft::device_mdspan<uint8_t,
                      list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                      raft::row_major> out_list_data,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t len,
  uint32_t pq_dim,
  Action action)
{
  using subwarp_align = raft::Pow2<SubWarpSize>;
  // Grid-stride loop: each vector is processed by exactly one subwarp.
  uint32_t stride = subwarp_align::div(blockDim.x) * gridDim.x;
  uint32_t ix     = subwarp_align::div(threadIdx.x + blockDim.x * blockIdx.x);
  for (; ix < len; ix += stride) {
    const uint32_t dst_ix = std::holds_alternative<uint32_t>(offset_or_indices)
                              ? std::get<uint32_t>(offset_or_indices) + ix
                              : std::get<const uint32_t*>(offset_or_indices)[ix];
    write_vector_interleaved<PqBits, SubWarpSize>(out_list_data, dst_ix, ix, pq_dim, action);
  }
}

/** Process the given indices or a block of a single list (cluster) - flat layout. */
template <uint32_t PqBits, uint32_t SubWarpSize, typename Action>
__device__ void write_list_flat(uint8_t* out_flat_codes,
                                std::variant<uint32_t, const uint32_t*> offset_or_indices,
                                uint32_t len,
                                uint32_t pq_dim,
                                uint32_t bytes_per_vector,
                                Action action)
{
  using subwarp_align = raft::Pow2<SubWarpSize>;
  // Grid-stride loop: each vector is processed by exactly one subwarp.
  uint32_t stride = subwarp_align::div(blockDim.x) * gridDim.x;
  uint32_t ix     = subwarp_align::div(threadIdx.x + blockDim.x * blockIdx.x);
  for (; ix < len; ix += stride) {
    const uint32_t dst_ix = std::holds_alternative<uint32_t>(offset_or_indices)
                              ? std::get<uint32_t>(offset_or_indices) + ix
                              : std::get<const uint32_t*>(offset_or_indices)[ix];
    write_vector_flat<PqBits, SubWarpSize>(
      out_flat_codes, dst_ix, ix, pq_dim, bytes_per_vector, action);
  }
}

}  // namespace cuvs::neighbors::ivf_pq::detail
