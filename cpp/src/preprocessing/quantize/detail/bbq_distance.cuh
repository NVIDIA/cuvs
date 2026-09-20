/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace CUVS_EXPORT cuvs {
namespace preprocessing::quantize::bbq {

#ifdef __CUDACC__

// --------------------------------------------------------------------------
// Layout geometry
// Encoded row length of a quantizer, and the bit planes its layout slices a row into.
// --------------------------------------------------------------------------

template <typename DataT, typename IdxT>
_RAFT_HOST_DEVICE constexpr uint32_t get_encoded_row_length(
  const quantizer_view<DataT, IdxT>& dataset)
{
  return get_encoded_row_length(dataset.dim(), dataset.layout);
}

/**
 * Bit planes a row is sliced into: the transposed layouts hold one plane per code bit, the packed
 * ones are a single dense plane. This is all the tiling and the cross-plane inner products need
 * to know about a layout.
 */
_RAFT_HOST_DEVICE constexpr int get_code_planes(const bbq_code_layout layout)
{
  switch (layout) {
    case bbq_code_layout::transposed_2b: return 2;
    case bbq_code_layout::transposed_4b: return 4;
    default: return 1;
  }
}

// --------------------------------------------------------------------------
// Code inner products (1x1)
// Raw byte-level dot products over one row pair, per code layout.
// --------------------------------------------------------------------------

/**
 * Cross-plane binary inner product over `Planes` bit planes, shifting each (i, j) plane pair by
 * i + j. `Planes == 1` is a plain binary product, so this covers packed_1b as well as the
 * transposed layouts. The 4-byte body needs both rows 4-byte aligned; the byte tail handles a
 * stripe whose length is not a multiple of 4.
 */
template <int Planes>
__device__ __forceinline__ uint32_t code_inner_product_transposed(const uint8_t* row_a,
                                                                  const uint8_t* row_b,
                                                                  size_t n_bytes,
                                                                  uint32_t result = 0)
{
  const size_t stripe = n_bytes / Planes;
  // Plane p starts at row + p * stripe and is read as uint32_t words, so a stripe that is not a
  // multiple of 4 misaligns every plane past the first -- the byte tail below only covers a ragged
  // stripe *length*, not a ragged stripe *offset*. Planes == 1 has no offset and so is exempt.
  assert(Planes == 1 || stripe % sizeof(uint32_t) == 0);
#pragma unroll
  for (int i = 0; i < Planes; ++i) {
#pragma unroll
    for (int j = 0; j < Planes; ++j) {
      const uint8_t* a = row_a + i * stripe;
      const uint8_t* b = row_b + j * stripe;
      uint32_t partial = 0;
      size_t k         = 0;
#pragma unroll 4
      for (; k + sizeof(uint32_t) <= stripe; k += sizeof(uint32_t)) {
        partial += __popc(*reinterpret_cast<const uint32_t*>(a + k) &
                          *reinterpret_cast<const uint32_t*>(b + k));
      }
      for (; k < stripe; ++k) {
        partial += __popc(static_cast<unsigned>(a[k] & b[k]));
      }
      result += partial << (i + j);
    }
  }
  return result;
}

/** One word of two packed_4b rows: the two nibble halves are two masked dp4a products. */
__device__ __forceinline__ uint32_t dp4a_packed_4b_word(uint32_t a, uint32_t b, uint32_t total)
{
  constexpr uint32_t nibble_mask = 0x0F0F0F0Fu;
  total                          = __dp4a(a & nibble_mask, b & nibble_mask, total);
  return __dp4a((a >> 4) & nibble_mask, (b >> 4) & nibble_mask, total);
}

/** Symmetric for packNibbles (Lucene int4DotProductBothPacked). */
__device__ __forceinline__ uint32_t code_inner_product_packed_4b(const uint8_t* row_a,
                                                                 const uint8_t* row_b,
                                                                 size_t n_bytes,
                                                                 uint32_t total = 0)
{
  size_t i = 0;
#pragma unroll 4
  for (; i + 4 <= n_bytes; i += 4) {
    total = dp4a_packed_4b_word(*reinterpret_cast<const uint32_t*>(row_a + i),
                                *reinterpret_cast<const uint32_t*>(row_b + i),
                                total);
  }
  for (; i < n_bytes; ++i) {
    const unsigned a = row_a[i];
    const unsigned b = row_b[i];
    total += (a & 0x0Fu) * (b & 0x0Fu);
    total += ((a >> 4) & 0x0Fu) * ((b >> 4) & 0x0Fu);
  }
  return total;
}

/** One-byte-per-code dot product, optionally masking unused high bits. */
__device__ __forceinline__ uint32_t code_inner_product_packed_8b(const uint8_t* row_a,
                                                                 const uint8_t* row_b,
                                                                 size_t n_bytes,
                                                                 uint32_t result   = 0,
                                                                 uint8_t code_mask = 0xFFu)
{
  const uint32_t word_mask = uint32_t{code_mask} * 0x01010101u;
  size_t i                 = 0;
#pragma unroll 4
  for (; i + 4 <= n_bytes; i += 4) {
    const auto a = *reinterpret_cast<const uint32_t*>(row_a + i) & word_mask;
    const auto b = *reinterpret_cast<const uint32_t*>(row_b + i) & word_mask;
    result       = __dp4a(a, b, result);
  }
  for (; i < n_bytes; ++i) {
    result +=
      static_cast<uint32_t>(row_a[i] & code_mask) * static_cast<uint32_t>(row_b[i] & code_mask);
  }
  return result;
}

/**
 * Integer inner product between two encoded rows.
 *
 * The uint32_t result bounds every BBQ layout to 66,050 dimensions: the worst case is
 * `packed_8b`, where `dim * 255 * 255` must not exceed UINT32_MAX.
 */
__device__ __forceinline__ uint32_t code_inner_product(const uint8_t* row_a,
                                                       const uint8_t* row_b,
                                                       const bbq_code_layout layout,
                                                       const size_t n_bytes,
                                                       uint32_t result = 0)
{
  switch (layout) {
    case bbq_code_layout::packed_1b:
      return code_inner_product_transposed<1>(row_a, row_b, n_bytes, result);
    case bbq_code_layout::transposed_2b:
      return code_inner_product_transposed<2>(row_a, row_b, n_bytes, result);
    case bbq_code_layout::packed_4b:
      return code_inner_product_packed_4b(row_a, row_b, n_bytes, result);
    case bbq_code_layout::transposed_4b:
      return code_inner_product_transposed<4>(row_a, row_b, n_bytes, result);
    case bbq_code_layout::packed_8b:
      return code_inner_product_packed_8b(row_a, row_b, n_bytes, result);
    case bbq_code_layout::packed_7b:
    default:
      return code_inner_product_packed_8b(
        row_a,
        row_b,
        n_bytes,
        result,
        static_cast<uint8_t>((uint32_t{1} << get_bit_width(layout)) - 1));
  }
}

// --------------------------------------------------------------------------
// Code promotion
// Widens a narrower layout to 4-bit width, so an asymmetric pair can meet in one format.
// --------------------------------------------------------------------------

// Promotes one native word of dense packed_1b codes (1 bit/value, 8 values/byte, MSB-first:
// the value at position 8*byte+i sits at bit (7-i)) into four 4-bit-width, packed_4b-style
// output words, packed into one uint4 for a single vectorized store.
//
// Branch-free SWAR: extract the 4 (2-bit) fields of each byte lane-wise across all 4 native
// bytes at once, spread each field 0-3 into a nibble value lane-wise, then transpose the 4
// resulting field-words into the 4 per-native-byte output words with chained __byte_perm pairs.
__device__ __forceinline__ uint4 packed_1b_to_4b(uint32_t native_word)
{
  const uint32_t spread0 = ((native_word >> 3) & 0x10101010u) | ((native_word >> 6) & 0x01010101u);
  const uint32_t spread1 = ((native_word >> 1) & 0x10101010u) | ((native_word >> 4) & 0x01010101u);
  const uint32_t spread2 = ((native_word << 1) & 0x10101010u) | ((native_word >> 2) & 0x01010101u);
  const uint32_t spread3 = ((native_word << 3) & 0x10101010u) | (native_word & 0x01010101u);
  const uint32_t t0      = __byte_perm(spread0, spread1, 0x5140);
  const uint32_t t1      = __byte_perm(spread0, spread1, 0x7362);
  const uint32_t t2      = __byte_perm(spread2, spread3, 0x5140);
  const uint32_t t3      = __byte_perm(spread2, spread3, 0x7362);
  return uint4{__byte_perm(t0, t2, 0x5410),
               __byte_perm(t0, t2, 0x7632),
               __byte_perm(t1, t3, 0x5410),
               __byte_perm(t1, t3, 0x7632)};
}

// --------------------------------------------------------------------------
// Cross-layout inner products (1x1)
// One document row against one query row encoded in a different layout.
// --------------------------------------------------------------------------

/**
 * Cross-plane binary inner product over two bit-sliced rows whose plane counts need not match,
 * shifting each (i, j) plane pair by i + j. The runtime-width counterpart of
 * code_inner_product_transposed, which the local-join kernels reach with both counts known at
 * compile time; a single plane is a plain binary product, so this covers packed_1b too.
 *
 * All three bit-sliced layouts store ceildiv(dim, 8) bytes per plane, which is what lets one
 * @p plane_bytes describe both operands.
 */
__device__ __forceinline__ uint32_t code_inner_product_planes(const uint8_t* row_document,
                                                              int document_planes,
                                                              const uint8_t* row_query,
                                                              int query_planes,
                                                              size_t plane_bytes,
                                                              uint32_t result = 0)
{
  // Planes past the first start at a multiple of plane_bytes and are read as uint32_t words.
  assert((document_planes == 1 && query_planes == 1) || plane_bytes % sizeof(uint32_t) == 0);
  for (int i = 0; i < document_planes; ++i) {
    for (int j = 0; j < query_planes; ++j) {
      const uint8_t* a = row_document + i * plane_bytes;
      const uint8_t* b = row_query + j * plane_bytes;
      uint32_t partial = 0;
      size_t k         = 0;
#pragma unroll 4
      for (; k + sizeof(uint32_t) <= plane_bytes; k += sizeof(uint32_t)) {
        partial += __popc(*reinterpret_cast<const uint32_t*>(a + k) &
                          *reinterpret_cast<const uint32_t*>(b + k));
      }
      for (; k < plane_bytes; ++k) {
        partial += __popc(static_cast<unsigned>(a[k] & b[k]));
      }
      result += partial << (i + j);
    }
  }
  return result;
}

/**
 * packed_1b document against a packed_4b query. The document is promoted to 4-bit width one
 * native word at a time -- 32 dimensions, which is exactly the four query words covering the same
 * range -- and the pair is multiplied as two packed_4b rows. The SIMT equivalent of what
 * stage_promoted_tile plus the u4 MMA do for this layout pair in the tensor-core local join, and
 * it pairs codes the same way, since packed_1b_to_4b emits packed_4b-style words.
 *
 * Requires dim % 32 == 0, so that the promoted document covers the query row exactly.
 */
__device__ __forceinline__ uint32_t code_inner_product_1b_x_packed_4b(const uint8_t* row_document,
                                                                      const uint8_t* row_query,
                                                                      size_t document_bytes,
                                                                      uint32_t result = 0)
{
  assert(document_bytes % sizeof(uint32_t) == 0);
  for (size_t i = 0; i + sizeof(uint32_t) <= document_bytes; i += sizeof(uint32_t)) {
    uint32_t promoted[4];
    reinterpret_cast<uint4&>(promoted) =
      packed_1b_to_4b(*reinterpret_cast<const uint32_t*>(row_document + i));
    const auto* query_words = reinterpret_cast<const uint32_t*>(row_query + 4 * i);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      result = dp4a_packed_4b_word(promoted[e], query_words[e], result);
    }
  }
  return result;
}

/**
 * Integer inner product between a document row and a query row, each read through its own
 * quantizer. The two layouts may differ: the supported (document, query) pairs are the ones
 * nn-descent's local join dispatches on, and passing the same view twice is the symmetric case.
 */
template <typename DataT, typename IdxT>
__device__ __forceinline__ uint32_t
code_inner_product(const quantizer_view<DataT, IdxT>& quantizer_document,
                   const quantizer_view<DataT, IdxT>& quantizer_query,
                   int64_t row_document,
                   int64_t row_query)
{
  const uint8_t* document = &quantizer_document.codes(row_document, 0);
  const uint8_t* query    = &quantizer_query.codes(row_query, 0);

  if (quantizer_document.layout == quantizer_query.layout) {
    return code_inner_product(
      document, query, quantizer_document.layout, get_encoded_row_length(quantizer_document));
  }
  if (quantizer_document.layout == bbq_code_layout::packed_1b &&
      quantizer_query.layout == bbq_code_layout::packed_4b) {
    return code_inner_product_1b_x_packed_4b(
      document, query, get_encoded_row_length(quantizer_document));
  }
  // Every remaining supported pair is bit-sliced on both sides: (packed_1b, transposed_2b),
  // (packed_1b, transposed_4b) and (transposed_2b, transposed_4b).
  assert(quantizer_document.layout == bbq_code_layout::packed_1b ||
         quantizer_document.layout == bbq_code_layout::transposed_2b);
  assert(quantizer_query.layout == bbq_code_layout::transposed_2b ||
         quantizer_query.layout == bbq_code_layout::transposed_4b);
  return code_inner_product_planes(document,
                                   get_code_planes(quantizer_document.layout),
                                   query,
                                   get_code_planes(quantizer_query.layout),
                                   (quantizer_document.dim() + 7) / 8);
}

// --------------------------------------------------------------------------
// Fused inner products (2x1)
// Two left rows against a shared right operand, for the local-join inner loop.
// --------------------------------------------------------------------------

/**
 * Two cross-plane inner products over statically known document and query plane counts.
 * Not asymmetric-specific: a symmetric pair is just document_planes == query_planes, which is how
 * the 1x1 and 2t x 2t self-joins are computed.
 */
template <int document_planes, int query_planes, size_t document_row_bytes, size_t query_row_bytes>
__device__ inline void code_inner_product_planes_2x1(const uint8_t* row_a0,
                                                     const uint8_t* row_a1,
                                                     const uint8_t* row_b,
                                                     uint32_t& total0,
                                                     uint32_t& total1)
{
  constexpr size_t document_plane_stride = document_row_bytes / document_planes;
  constexpr size_t query_plane_stride    = query_row_bytes / query_planes;
  // Both operands are stepped by their own plane stride and then read as uint32_t, so both
  // strides -- not just the query's -- must be 4-byte aligned.
  static_assert(query_plane_stride % sizeof(uint32_t) == 0);
  static_assert(document_plane_stride % sizeof(uint32_t) == 0);
#pragma unroll
  for (int p_query = 0; p_query < query_planes; ++p_query) {
#pragma unroll
    for (int p_document = 0; p_document < document_planes; ++p_document) {
      const uint8_t* a0 = row_a0 + p_document * document_plane_stride;
      const uint8_t* a1 = row_a1 + p_document * document_plane_stride;
      const uint8_t* b  = row_b + p_query * query_plane_stride;
      uint32_t partial0 = 0;
      uint32_t partial1 = 0;
#pragma unroll 4
      for (size_t i = 0; i < query_plane_stride; i += sizeof(uint32_t)) {
        const auto wa0 = *reinterpret_cast<const uint32_t*>(a0 + i);
        const auto wa1 = *reinterpret_cast<const uint32_t*>(a1 + i);
        const auto wb  = *reinterpret_cast<const uint32_t*>(b + i);
        partial0 += __popc(wa0 & wb);
        partial1 += __popc(wa1 & wb);
      }
      total0 += partial0 << (p_document + p_query);
      total1 += partial1 << (p_document + p_query);
    }
  }
}

template <size_t n_bytes>
__device__ inline void code_inner_product_packed_4b_symmetric_2x1(const uint8_t* row_a0,
                                                                  const uint8_t* row_a1,
                                                                  const uint8_t* row_b,
                                                                  uint32_t& total0,
                                                                  uint32_t& total1)
{
  static_assert(n_bytes % sizeof(uint32_t) == 0);
  constexpr uint32_t nibble_mask = 0x0F0F0F0Fu;
#pragma unroll 4
  for (size_t i = 0; i < n_bytes; i += sizeof(uint32_t)) {
    const auto a0     = *reinterpret_cast<const uint32_t*>(row_a0 + i);
    const auto a1     = *reinterpret_cast<const uint32_t*>(row_a1 + i);
    const auto b      = *reinterpret_cast<const uint32_t*>(row_b + i);
    const auto b_low  = b & nibble_mask;
    const auto b_high = (b >> 4) & nibble_mask;
    total0            = __dp4a(a0 & nibble_mask, b_low, total0);
    total0            = __dp4a((a0 >> 4) & nibble_mask, b_high, total0);
    total1            = __dp4a(a1 & nibble_mask, b_low, total1);
    total1            = __dp4a((a1 >> 4) & nibble_mask, b_high, total1);
  }
}

template <size_t n_bytes>
__device__ inline void code_inner_product_packed_8b_2x1(const uint8_t* row_a0,
                                                        const uint8_t* row_a1,
                                                        const uint8_t* row_b,
                                                        uint32_t& total0,
                                                        uint32_t& total1,
                                                        uint8_t code_mask = 0xFFu)
{
  static_assert(n_bytes % sizeof(uint32_t) == 0);
  const uint32_t word_mask = uint32_t{code_mask} * 0x01010101u;
#pragma unroll 4
  for (size_t i = 0; i < n_bytes; i += sizeof(uint32_t)) {
    const auto a0 = *reinterpret_cast<const uint32_t*>(row_a0 + i) & word_mask;
    const auto a1 = *reinterpret_cast<const uint32_t*>(row_a1 + i) & word_mask;
    const auto b  = *reinterpret_cast<const uint32_t*>(row_b + i) & word_mask;
    total0        = __dp4a(a0, b, total0);
    total1        = __dp4a(a1, b, total1);
  }
}

// Selects the SIMT inner product for a (document, query) layout pair: bit-sliced layouts go to the
// cross-plane popc, densely-packed ones to dp4a. dp4a needs both operands in the same packing, so
// those forms apply whenever the two layouts match. packed_7b and packed_8b reach this from
// GNND::local_join; packed_4b does not (symmetric packed_4b goes to the wmma kernel), but is kept
// as a SIMT reference point.
template <bbq_code_layout DocumentLayout,
          bbq_code_layout QueryLayout,
          int DocumentPlanes,
          int QueryPlanes,
          size_t DocumentRowBytes,
          size_t QueryRowBytes>
__device__ __forceinline__ void bbq_code_inner_product_2x1(const uint8_t* row_a0,
                                                           const uint8_t* row_a1,
                                                           const uint8_t* row_b,
                                                           uint32_t& total0,
                                                           uint32_t& total1)
{
  namespace bbq = cuvs::preprocessing::quantize::bbq;
  if constexpr (DocumentLayout == QueryLayout && DocumentLayout == bbq_code_layout::packed_4b) {
    bbq::code_inner_product_packed_4b_symmetric_2x1<DocumentRowBytes>(
      row_a0, row_a1, row_b, total0, total1);
  } else if constexpr (DocumentLayout == QueryLayout &&
                       (DocumentLayout == bbq_code_layout::packed_8b ||
                        DocumentLayout == bbq_code_layout::packed_7b)) {
    // packed_7b is packed_8b with the top bit masked off, matching code_inner_product's
    // (1 << bits) - 1 mask for the same two layouts.
    constexpr uint8_t code_mask = DocumentLayout == bbq_code_layout::packed_7b ? 0x7Fu : 0xFFu;
    bbq::code_inner_product_packed_8b_2x1<DocumentRowBytes>(
      row_a0, row_a1, row_b, total0, total1, code_mask);
  } else {
    bbq::
      code_inner_product_planes_2x1<DocumentPlanes, QueryPlanes, DocumentRowBytes, QueryRowBytes>(
        row_a0, row_a1, row_b, total0, total1);
  }
}

// --------------------------------------------------------------------------
// Dequantization
// Turns a raw code inner product into a final float distance.
// --------------------------------------------------------------------------

// Per-row dequantization terms needed by bbq_calculate_metric.
// row_norm is read directly from the quantizer view instead,
// since register pressure in the BBQ local-join kernels is already tight.
struct bbq_dequant_factors {
  float lower;
  float delta;
  float sum_delta;
  float corrections;
};

template <typename DataT, typename IdxT>
__device__ __forceinline__ bbq_dequant_factors
get_dequant_factors(const quantizer_view<DataT, IdxT>& quantizer, int64_t row)
{
  return bbq_dequant_factors{quantizer.lower_intervals(row),
                             quantizer.dequant_delta(row),
                             quantizer.dequant_sum_delta(row),
                             quantizer.additional_corrections(row)};
}

// Converts one raw BBQ dot product into a final (post-epilogue) float distance, given both
// operands' precomputed dequant factors. Evaluated exactly once per matrix cell
// dim/centroid_norm_sq/row_norm both come directly from quantizer_document/quantizer_query rather
// than being passed separately, since every caller reads them the same id-indexed way -- row_norm
// is only read for CosineExpanded (skipped entirely otherwise).
template <typename DataT, typename Index_t, typename DistEpilogue_t>
__device__ __forceinline__ float bbq_calculate_metric(
  uint32_t raw,
  const bbq_dequant_factors& doc_factors,
  const bbq_dequant_factors& query_factors,
  const quantizer_view<DataT, int64_t>& quantizer_document,
  const quantizer_view<DataT, int64_t>& quantizer_query,
  cuvs::distance::DistanceType metric,
  DistEpilogue_t dist_epilogue,
  Index_t document_id,
  Index_t query_id)
{
  constexpr bool can_postprocess_dist = std::is_same_v<DistEpilogue_t, raft::identity_op>;
  const float dim                     = static_cast<float>(quantizer_document.dim());

  const float centered = dim * doc_factors.lower * query_factors.lower +
                         query_factors.lower * doc_factors.sum_delta +
                         doc_factors.lower * query_factors.sum_delta +
                         doc_factors.delta * query_factors.delta * static_cast<float>(raw);
  const float corrections = doc_factors.corrections + query_factors.corrections;
  float d;
  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    const float raw_distance = corrections - 2.0f * centered;
    d                        = raw_distance < 0.0f ? 0.0f : raw_distance;
    if (!can_postprocess_dist && metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
      d = sqrtf(d);
    }
  } else if (metric == cuvs::distance::DistanceType::InnerProduct) {
    d = -(centered + corrections - quantizer_document.centroid_norm_sq);
  } else {  // CosineExpanded
    const float norm_product =
      quantizer_document.row_norm(document_id) * quantizer_query.row_norm(query_id);
    const float dot = centered + corrections - quantizer_document.centroid_norm_sq;
    d               = norm_product > 0.0f ? 1.0f - dot / sqrtf(norm_product) : 0.0f;
  }
  return dist_epilogue(d, document_id, query_id);
}

#endif  // __CUDACC__

}  // namespace preprocessing::quantize::bbq
}  // namespace CUVS_EXPORT cuvs
