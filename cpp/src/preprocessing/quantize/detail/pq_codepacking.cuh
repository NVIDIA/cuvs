/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_utils.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <type_traits>

namespace cuvs::preprocessing::quantize::pq::detail {

template <typename T>
constexpr bool is_const_ptr_v = std::is_const_v<std::remove_pointer_t<T>>;

/**
 * Bitfield reference for reading/writing from non-const pointers.
 * This type mimics `uint8_t&` or `uint16_t&` for the indexing operator.
 * Taken from src/neighbors/ivf_pq/ivf_pq_codepacking.cuh with the `bits` template
 * parameter becoming a member variable.
 * @param Bits number of bits comprising the value.
 */
template <typename PtrT = uint8_t*>
struct bitfield_ref_t {
  PtrT ptr;
  uint32_t offset;
  uint32_t bits;

  __host__ __device__ operator uint8_t() const  // NOLINT
  {
    const uint8_t mask = static_cast<uint8_t>((1u << bits) - 1u);
    uint32_t pair      = static_cast<uint32_t>(ptr[0]);
    if (offset + bits > 8) { pair |= static_cast<uint32_t>(ptr[1]) << 8; }
    return static_cast<uint8_t>((pair >> offset) & mask);
  }

  __host__ __device__ operator uint16_t() const  // NOLINT
  {
    const uint16_t mask = static_cast<uint16_t>((1u << bits) - 1u);
    uint32_t pair       = static_cast<uint32_t>(ptr[0]);
    if (offset + bits > 8) { pair |= static_cast<uint32_t>(ptr[1]) << 8; }
    if (offset + bits > 16) { pair |= static_cast<uint32_t>(ptr[2]) << 16; }
    return static_cast<uint16_t>((pair >> offset) & mask);
  }

  template <typename T = PtrT>
  __host__ __device__ auto operator=(uint8_t code)
    -> std::enable_if_t<!is_const_ptr_v<T>, bitfield_ref_t&>
  {
    const uint8_t mask = static_cast<uint8_t>((1u << bits) - 1u);

    if (offset + bits > 8) {
      auto pair = static_cast<uint16_t>(ptr[0]);
      pair |= static_cast<uint16_t>(ptr[1]) << 8;
      pair &= ~(static_cast<uint16_t>(mask) << offset);
      pair |= static_cast<uint16_t>(code) << offset;
      ptr[0] = static_cast<uint8_t>(pair & 0xFF);
      ptr[1] = static_cast<uint8_t>((pair >> 8) & 0xFF);
    } else {
      ptr[0] = (ptr[0] & ~(mask << offset)) | (code << offset);
    }
    return *this;
  }

  template <typename T = PtrT>
  __host__ __device__ auto operator=(uint16_t code)
    -> std::enable_if_t<!is_const_ptr_v<T>, bitfield_ref_t&>
  {
    const uint16_t mask = static_cast<uint16_t>((1u << bits) - 1u);

    // General case for multi-byte operations
    uint32_t pair = static_cast<uint32_t>(ptr[0]);
    pair |= static_cast<uint32_t>(ptr[1]) << 8;
    if (offset + bits > 16) { pair |= static_cast<uint32_t>(ptr[2]) << 16; }

    pair &= ~(static_cast<uint32_t>(mask) << offset);
    pair |= static_cast<uint32_t>(code) << offset;

    ptr[0] = static_cast<uint8_t>(pair & 0xFF);
    ptr[1] = static_cast<uint8_t>((pair >> 8) & 0xFF);
    if (offset + bits > 16) { ptr[2] = static_cast<uint8_t>((pair >> 16) & 0xFF); }

    return *this;
  }
};

/**
 * View a byte array as an array of unsigned integers of custom small bit size.
 * Taken from src/neighbors/ivf_pq/ivf_pq_codepacking.cuh with the `bits` template
 * parameter becoming a member variable.
 * @param Bits number of bits comprising the value.
 */
template <typename PtrT = uint8_t*>
struct bitfield_view_t {
  PtrT raw;
  uint32_t bits;

  template <typename T = PtrT>
  __host__ __device__ auto operator[](uint32_t i)
    -> std::enable_if_t<!is_const_ptr_v<T>, bitfield_ref_t<PtrT>>
  {
    uint32_t bit_offset = i * bits;
    return bitfield_ref_t<PtrT>{
      raw + raft::Pow2<8>::div(bit_offset), raft::Pow2<8>::mod(bit_offset), bits};
  }

  __host__ __device__ auto operator[](uint32_t i) const
    -> bitfield_ref_t<const std::remove_pointer_t<PtrT>*>
  {
    uint32_t bit_offset = i * bits;
    return bitfield_ref_t<const std::remove_pointer_t<PtrT>*>{
      raw + raft::Pow2<8>::div(bit_offset), raft::Pow2<8>::mod(bit_offset), bits};
  }
};
}  // namespace cuvs::preprocessing::quantize::pq::detail
