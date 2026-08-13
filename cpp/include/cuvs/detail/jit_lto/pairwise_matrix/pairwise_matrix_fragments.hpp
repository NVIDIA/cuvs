/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::distance::detail {

struct tag_layout_row {};
struct tag_layout_col {};

struct tag_fin_op_identity {};
struct tag_fin_op_rbf {};

struct tag_distance_canberra {};
struct tag_distance_correlation {};
struct tag_distance_cosine {};
struct tag_distance_hamming_unexpanded {};
struct tag_distance_hellinger_expanded {};
struct tag_distance_jensen_shannon {};
struct tag_distance_kl_divergence {};
struct tag_distance_l1 {};
struct tag_distance_l2_expanded {};
struct tag_distance_l2_unexpanded {};
struct tag_distance_l_inf {};
struct tag_distance_lp_unexpanded {};
struct tag_distance_russel_rao {};

template <typename DistanceTag,
          typename DataTag,
          typename AccTag,
          typename OutTag,
          typename IndexTag,
          typename FinOpTag,
          typename LayoutTag,
          int Veclen>
struct fragment_tag_pairwise_matrix {};

template <typename DistanceTag, typename DataTag, typename AccTag, typename IndexTag>
struct fragment_tag_compute_distance {};

template <typename DistanceTag,
          typename DataTag,
          typename AccTag,
          typename IndexTag,
          typename LayoutTag,
          int Veclen>
struct fragment_tag_compute_distance_epilog {};

}  // namespace cuvs::distance::detail
