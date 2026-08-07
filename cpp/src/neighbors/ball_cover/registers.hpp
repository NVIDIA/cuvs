/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "registers_types.cuh"
#include <cuvs/neighbors/ball_cover.hpp>

#include <raft/core/resources.hpp>

namespace cuvs::neighbors::ball_cover::detail {

template <typename value_idx, typename value_t>
void rbc_low_dim_pass_one(raft::resources const& handle,
                          const cuvs::neighbors::ball_cover::index<value_idx, value_t>& index,
                          const value_t* query,
                          const int64_t n_query_rows,
                          const int64_t k,
                          const value_idx* R_knn_inds,
                          const value_t* R_knn_dists,
                          value_idx* inds,
                          value_t* dists,
                          float weight,
                          int dims);

template <typename value_idx, typename value_t>
void rbc_low_dim_pass_two(raft::resources const& handle,
                          const cuvs::neighbors::ball_cover::index<value_idx, value_t>& index,
                          const value_t* query,
                          const int64_t n_query_rows,
                          const int64_t k,
                          const value_idx* R_knn_inds,
                          const value_t* R_knn_dists,
                          value_idx* inds,
                          value_t* dists,
                          float weight,
                          int dims);

template <typename value_idx, typename value_t, typename dist_func>
void rbc_eps_pass(raft::resources const& handle,
                  const cuvs::neighbors::ball_cover::index<value_idx, value_t>& index,
                  const value_t* query,
                  const int64_t n_query_rows,
                  value_t eps,
                  const value_t* R,
                  dist_func& dfunc,
                  bool* adj,
                  value_idx* vd);

template <typename value_idx, typename value_t, typename dist_func>
void rbc_eps_pass(raft::resources const& handle,
                  const cuvs::neighbors::ball_cover::index<value_idx, value_t>& index,
                  const value_t* query,
                  const int64_t n_query_rows,
                  value_t eps,
                  int64_t* max_k,
                  const value_t* R,
                  dist_func& dfunc,
                  value_idx* adj_ia,
                  value_idx* adj_ja,
                  value_idx* vd,
                  value_t* dists);

}  // namespace cuvs::neighbors::ball_cover::detail
