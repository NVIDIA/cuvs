/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

/* Manually unrolled loop over a chunk of pq_dataset that fits into one VecT. */
template <typename OutT,
          typename LutT,
          typename VecT,
          bool CheckBounds,
          uint32_t PqBits,
          uint32_t BitsLeft = 0,
          uint32_t Ix       = 0>
__device__ __forceinline__ void compute_chunk(OutT& score /* NOLINT */,
                                              typename VecT::math_t& pq_code,
                                              const VecT& pq_codes,
                                              const LutT*& lut_head,
                                              const LutT*& lut_end)
{
  if constexpr (CheckBounds) {
    if (lut_head >= lut_end) { return; }
  }
  constexpr uint32_t kTotalBits = 8 * sizeof(typename VecT::math_t);
  constexpr uint32_t kPqShift   = 1u << PqBits;
  constexpr uint32_t kPqMask    = kPqShift - 1u;
  if constexpr (BitsLeft >= PqBits) {
    uint8_t code = pq_code & kPqMask;
    pq_code >>= PqBits;
    score += OutT(lut_head[code]);
    lut_head += kPqShift;
    return compute_chunk<OutT, LutT, VecT, CheckBounds, PqBits, BitsLeft - PqBits, Ix>(
      score, pq_code, pq_codes, lut_head, lut_end);
  } else if constexpr (Ix < VecT::Ratio) {
    uint8_t code                = pq_code;
    pq_code                     = pq_codes.val.data[Ix];
    constexpr uint32_t kRemBits = PqBits - BitsLeft;
    constexpr uint32_t kRemMask = (1u << kRemBits) - 1u;
    code |= (pq_code & kRemMask) << BitsLeft;
    pq_code >>= kRemBits;
    score += OutT(lut_head[code]);
    lut_head += kPqShift;
    return compute_chunk<OutT, LutT, VecT, CheckBounds, PqBits, kTotalBits - kRemBits, Ix + 1>(
      score, pq_code, pq_codes, lut_head, lut_end);
  }
}

/* Compute the similarity for one vector in the pq_dataset */
template <typename OutT, typename LutT, typename VecT, uint32_t PqBits>
__device__ auto compute_score_impl(uint32_t pq_dim,
                                   const typename VecT::io_t* pq_head,
                                   const LutT* lut_scores,
                                   OutT early_stop_limit) -> OutT
{
  constexpr uint32_t kChunkSize = sizeof(VecT) * 8u / PqBits;
  auto lut_head                 = lut_scores;
  auto lut_end                  = lut_scores + (pq_dim << PqBits);
  VecT pq_codes;
  OutT score{0};
  for (; pq_dim >= kChunkSize; pq_dim -= kChunkSize) {
    *pq_codes.vectorized_data() = *pq_head;
    pq_head += kIndexGroupSize;
    typename VecT::math_t pq_code = 0;
    compute_chunk<OutT, LutT, VecT, false, PqBits>(score, pq_code, pq_codes, lut_head, lut_end);
    // Early stop when it makes sense (otherwise early_stop_limit is kDummy/infinity).
    if (score >= early_stop_limit) { return score; }
  }
  if (pq_dim > 0) {
    *pq_codes.vectorized_data()   = *pq_head;
    typename VecT::math_t pq_code = 0;
    compute_chunk<OutT, LutT, VecT, true, PqBits>(score, pq_code, pq_codes, lut_head, lut_end);
  }
  return score;
}

}  // namespace cuvs::neighbors::ivf_pq::detail
