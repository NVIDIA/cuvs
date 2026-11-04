/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/mdspan_types.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

template <typename IdxT>
void encode_residuals(
  raft::device_matrix_view<float, IdxT> dataset_residual,
  raft::device_vector_view<uint32_t, IdxT> output_labels,
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> output_dataset,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
  codebook_gen codebook_kind,
  uint32_t pq_bits,
  uint32_t pq_book_size,
  cudaStream_t stream);

}  // namespace cuvs::neighbors::ivf_pq::detail
