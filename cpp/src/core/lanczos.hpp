/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/solver/lanczos_types.hpp>

#include <optional>

namespace cuvs::detail {

template <typename Index, typename T, typename NnzIndex>
int lanczos_compute_eigenpairs(
  raft::resources const&,
  raft::sparse::solver::lanczos_solver_config<T> const&,
  raft::device_coo_matrix_view<T, Index, Index, NnzIndex>,
  std::optional<raft::device_vector_view<T, uint32_t, raft::row_major>>,
  raft::device_vector_view<T, uint32_t, raft::col_major>,
  raft::device_matrix_view<T, uint32_t, raft::col_major>);

}  // namespace cuvs::detail
