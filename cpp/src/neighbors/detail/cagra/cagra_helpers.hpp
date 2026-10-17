/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/common.hpp>

#include <raft/core/mdspan.hpp>
#include <raft/util/integer_utils.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cuvs::neighbors::cagra::helpers {

/** The batch size for the CAGRA optimize stage. */
constexpr static size_t kOptimizeBatchSize = 256 * 1024;

/** The query chunk size (search batch size) of the iterative CAGRA graph build. */
constexpr static size_t kIterativeBuildChunkSize = 8192;

/**
 * Calculate the device memory footprint of a VPQ-compressed (CAGRA-Q) dataset.
 *
 * The footprint is the sum of the VQ codebook, the PQ codebook and the encoded rows. Parameters
 * left at 0 are resolved with the same heuristics that `vpq_build` applies.
 *
 * @param[in] uncompressed_dataset_shape shape of the uncompressed dataset
 * @param[in] params VPQ compression parameters
 * @param[in] codebook_element_size size in bytes of a codebook element (2 for the f16 codebooks
 *            used by CAGRA-Q)
 *
 * @return compressed dataset size in bytes
 */
inline size_t vpq_dataset_size(raft::matrix_extent<int64_t> uncompressed_dataset_shape,
                               cuvs::neighbors::vpq_params params,
                               size_t codebook_element_size = 2)
{
  const size_t n_rows = uncompressed_dataset_shape.extent(0);
  const size_t dim    = uncompressed_dataset_shape.extent(1);

  // Mirror detail::fill_missing_params_heuristics for the fields that affect the footprint.
  const size_t pq_bits = params.pq_bits == 0 ? 8 : params.pq_bits;
  const size_t pq_dim =
    params.pq_dim == 0 ? raft::div_rounding_up_safe(dim, size_t{4}) : params.pq_dim;
  const size_t vq_n_centers =
    params.vq_n_centers == 0
      ? raft::round_up_safe<size_t>(static_cast<size_t>(std::sqrt(static_cast<double>(n_rows))), 8)
      : params.vq_n_centers;

  const size_t pq_len       = raft::div_rounding_up_safe(dim, pq_dim);
  const size_t pq_n_centers = size_t{1} << pq_bits;

  // Every encoded row starts with its inlined VQ label (vpq_build always requests inline labels)
  // and is followed by the bit-packed PQ codes.
  using label_type            = uint32_t;
  constexpr size_t kLabelBits = 8 * sizeof(label_type);
  const size_t encoded_row_length =
    sizeof(label_type) * (1 + raft::div_rounding_up_safe(pq_dim * pq_bits, kLabelBits));

  return vq_n_centers * dim * codebook_element_size       // vq_code_book
         + pq_n_centers * pq_len * codebook_element_size  // pq_code_book
         + n_rows * encoded_row_length;                   // encoded rows
}

}  // namespace cuvs::neighbors::cagra::helpers
