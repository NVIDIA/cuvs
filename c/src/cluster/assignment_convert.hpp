/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/cluster/assignment.h>
#include <cuvs/cluster/assignment.hpp>

#include <raft/core/error.hpp>

#include <optional>

namespace cuvs::core {

inline cuvs::cluster::ivf_assignment_params
convert_ivf_params(const cuvsIvfAssignmentParams &params) {
  cuvs::cluster::ivf_assignment_params out;
  out.n_lists = params.n_lists;
  out.kmeans_n_iters = params.kmeans_n_iters;
  out.n_probes = params.n_probes;
  return out;
}

/**
 * The C++ assignment params of the C ones; NULL means the default (exhaustive)
 * assignment, i.e. an empty optional. A prebuilt index is checked against the
 * kind and attached.
 */
inline std::optional<cuvs::cluster::assignment_params>
convert_assignment_params(const cuvsAssignmentParams *params) {
  if (params == nullptr) {
    return std::nullopt;
  }
  switch (params->kind) {
  case CUVS_ASSIGNMENT_IVF: {
    auto out = convert_ivf_params(params->ivf);
    if (params->index != nullptr) {
      RAFT_EXPECTS(
          params->index->kind == params->kind,
          "The assignment index was built for another assignment kind");
      RAFT_EXPECTS(params->index->addr != 0,
                   "The assignment index has not been built");
      out.index =
          reinterpret_cast<const cuvs::cluster::ivf_assignment_index *>(
              params->index->addr);
    }
    return cuvs::cluster::assignment_params{out};
  }
  default:
    RAFT_FAIL("Unsupported assignment kind: %d",
              static_cast<int>(params->kind));
  }
}

/** Frees what a built index holds and marks it empty. */
inline void release_assignment_index(cuvsAssignmentIndex &index) {
  if (index.addr == 0) {
    return;
  }
  switch (index.kind) {
  case CUVS_ASSIGNMENT_IVF:
    delete reinterpret_cast<cuvs::cluster::ivf_assignment_index *>(
        index.addr);
    break;
  default:
    break;
  }
  index.addr = 0;
}

/** Takes ownership of a built index. */
inline void
store_assignment_index(cuvsAssignmentIndex &index,
                       cuvs::cluster::ivf_assignment_index &&built) {
  release_assignment_index(index);
  index.kind = CUVS_ASSIGNMENT_IVF;
  index.addr = reinterpret_cast<uintptr_t>(
      new cuvs::cluster::ivf_assignment_index(std::move(built)));
}

} // namespace cuvs::core
