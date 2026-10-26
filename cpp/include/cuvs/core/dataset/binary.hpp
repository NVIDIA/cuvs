/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/preprocessing/quantize/binary.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>

#include <cstdint>

namespace CUVS_EXPORT cuvs {

/**
 * @brief Non-owning view of the dictionary that binarizes a dataset: the per-dimension thresholds
 * of a binary quantizer. An empty threshold vector stands for the zero threshold.
 */
template <typename MathT>
struct binary_dictionary_view {
  raft::device_vector_view<const MathT, int64_t> threshold;
};

/**
 * @brief Non-owning view of a binary-quantized dataset.
 *
 * The data are bit-packed codes, `ceil(dim / 8)` bytes per row; bit `i` of a row is stored in bit
 * `i % 8` of byte `i / 8`, and the unused bits of the last byte are zero. The dictionary is the
 * quantizer that produced the codes, so the logical dimension `dim` (the number of bits) never has
 * to be guessed from the padded byte width.
 *
 * @tparam MathT the type of the source (unquantized) values
 * @tparam IdxT the type of the row indices
 */
template <typename MathT, typename IdxT>
struct device_binary_dataset_view {
  using value_type           = MathT;
  using index_type           = IdxT;
  using code_type            = uint8_t;
  using data_view_type       = raft::device_matrix_view<const code_type, int64_t, raft::row_major>;
  using dictionary_view_type = binary_dictionary_view<MathT>;

  /**
   * @param codes row-major codes [n_rows, ceil(dim / 8)]
   * @param dictionary the thresholds used to produce the codes; empty or of size `dim`
   * @param dim the number of bits in each vector
   */
  device_binary_dataset_view(data_view_type codes, dictionary_view_type dictionary, uint32_t dim)
    : codes_{codes}, dictionary_{dictionary}, dim_{dim}
  {
    RAFT_EXPECTS(dim > 0, "The dimension of a binary dataset must be positive");
    RAFT_EXPECTS(static_cast<int64_t>((uint64_t{dim} + 7) / 8) == codes.extent(1),
                 "The row length of the codes must be ceil(dim / 8) bytes");
    RAFT_EXPECTS(dictionary.threshold.extent(0) == 0 ||
                   dictionary.threshold.extent(0) == static_cast<int64_t>(dim),
                 "The threshold vector must be empty or have one element per dimension");
  }

  /** The number of vectors. */
  [[nodiscard]] auto n_rows() const noexcept -> index_type
  {
    return static_cast<index_type>(codes_.extent(0));
  }
  /** The number of bits in each vector. */
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return dim_; }
  /** The number of bytes in each row of the codes. */
  [[nodiscard]] auto encoded_row_length() const noexcept -> int64_t { return codes_.extent(1); }

  [[nodiscard]] auto data_const_view() const noexcept -> data_view_type { return codes_; }
  [[nodiscard]] auto const_dictionary_view() const noexcept -> dictionary_view_type
  {
    return dictionary_;
  }

 private:
  data_view_type codes_;
  dictionary_view_type dictionary_;
  uint32_t dim_;
};

/**
 * @brief View the codes produced by `preprocessing::quantize::binary::transform` together with the
 * quantizer that produced them.
 */
template <typename IdxT, typename MathT>
auto make_device_binary_dataset_view(
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> codes,
  const preprocessing::quantize::binary::quantizer<MathT>& quantizer,
  uint32_t dim) -> device_binary_dataset_view<MathT, IdxT>
{
  return device_binary_dataset_view<MathT, IdxT>{
    codes, binary_dictionary_view<MathT>{quantizer.threshold.view()}, dim};
}

}  // namespace CUVS_EXPORT cuvs
