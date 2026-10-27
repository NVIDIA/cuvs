/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <type_traits>

namespace cuvs::distance::detail {

/**
 * @brief Compress 2D boolean matrix to bitfield
 *
 * Utility kernel for masked_l2_nn.
 *
 * @tparam T
 *
 * @parameter[in]  in       An `m x n` boolean matrix. Row major.
 * @parameter[out] out      An `(m / bits_per_elem) x n` matrix with elements of
 *                          type T, where T is of size `bits_per_elem` bits.
 *                          Note: the division (`/`) is a ceilDiv.
 */
template <typename T = uint64_t, typename = std::enable_if_t<std::is_integral<T>::value>>
void compress_to_bits(raft::resources const& handle,
                      raft::device_matrix_view<const bool, int, raft::layout_c_contiguous> in,
                      raft::device_matrix_view<T, int, raft::layout_c_contiguous> out);

}  // namespace cuvs::distance::detail
