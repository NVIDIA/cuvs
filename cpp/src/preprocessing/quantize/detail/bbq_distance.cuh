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
// Code inner products
// Raw byte-level dot products over one document row and one query row.
// --------------------------------------------------------------------------

/**
 * Cross-plane binary inner product over two bit-sliced rows whose plane counts need not match,
 * shifting each (i, j) plane pair by i + j. Equal counts are the symmetric case, and a single
 * plane is a plain binary product, so this covers every bit-sliced pair including packed_1b.
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
#pragma unroll
  for (int i = 0; i < document_planes; ++i) {
#pragma unroll
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
 * Integer inner product between a document row and a query row, each in its own layout. The query
 * layout picks the form, since each supported pair is fixed by it: a packed_4b query meets a
 * packed_4b document or a promoted packed_1b one, packed_7b/8b only themselves, and every other
 * query is bit-sliced against a bit-sliced document. The local join calls this with constant
 * layouts and row width, so once inlined the dispatch and the byte tails fold away.
 *
 * The uint32_t result bounds every BBQ layout to 66,050 dimensions: the worst case is
 * `packed_8b`, where `dim * 255 * 255` must not exceed UINT32_MAX.
 */
__device__ __forceinline__ uint32_t code_inner_product(const uint8_t* row_document,
                                                       const bbq_code_layout document_layout,
                                                       const uint8_t* row_query,
                                                       const bbq_code_layout query_layout,
                                                       const size_t document_row_bytes)
{
  switch (query_layout) {
    case bbq_code_layout::packed_4b:
      return document_layout == bbq_code_layout::packed_1b
               ? code_inner_product_1b_x_packed_4b(row_document, row_query, document_row_bytes)
               : code_inner_product_packed_4b(row_document, row_query, document_row_bytes);
    case bbq_code_layout::packed_7b:
    case bbq_code_layout::packed_8b:
      assert(document_layout == query_layout);
      return code_inner_product_packed_8b(
        row_document,
        row_query,
        document_row_bytes,
        0,
        static_cast<uint8_t>((uint32_t{1} << get_bit_width(query_layout)) - 1));
    default: {
      const int document_planes = get_code_planes(document_layout);
      return code_inner_product_planes(row_document,
                                       document_planes,
                                       row_query,
                                       get_code_planes(query_layout),
                                       document_row_bytes / document_planes);
    }
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
