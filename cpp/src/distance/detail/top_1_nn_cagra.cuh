/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace cuvs::distance::detail {

/**
 * @brief A CAGRA index over a copy of the candidate set (`y`) of a top-1 NN problem.
 *
 * Used by `Top1nnBackend::Cagra`, the approximate counterpart of the exact top-1 backends. The
 * caller builds it (once per candidate set) and hands it to `top_1_nn` through
 * `Top1nnTuning::cagra.index`.
 *
 * `cagra::index` only stores a *view* of its dataset, and CAGRA needs padded rows, so the copy is
 * owned here to keep graph and vectors consistent. The index uses squared L2 distance.
 */
class top_1_nn_cagra_index {
 public:
  using index_type = cuvs::neighbors::cagra::device_padded_index<float, uint32_t>;

  top_1_nn_cagra_index() = default;

  /**
   * Build (or rebuild) the index over `n` row-major candidate vectors of `dim` columns.
   *
   * `index_params` and `search_params` are used as given, except for what this backend relies on:
   * the metric is set to L2Expanded (squared L2 distances), the dataset is attached on build, and
   * the graph degrees are clamped to `n - 1`.
   */
  void build(raft::resources const& handle,
             cuvs::neighbors::cagra::index_params const& index_params,
             cuvs::neighbors::cagra::search_params const& search_params,
             const float* candidates,
             int64_t n,
             int64_t dim)
  {
    const std::size_t max_degree = static_cast<std::size_t>(std::max<int64_t>(1, n - 1));

    cuvs::neighbors::cagra::index_params build_params = index_params;
    build_params.metric                               = cuvs::distance::DistanceType::L2Expanded;
    build_params.graph_degree = std::min(build_params.graph_degree, max_degree);
    build_params.intermediate_graph_degree =
      std::min(build_params.intermediate_graph_degree, max_degree);
    build_params.attach_dataset_on_build = true;

    search_params_ = search_params;

    index_.reset();  // drop the old index before replacing the dataset it views
    snapshot_.reset();
    aligned_snapshot_.reset();
    n_   = n;
    dim_ = dim;

    auto candidates_view =
      raft::make_device_matrix_view<const float, int64_t, raft::row_major>(candidates, n, dim);
    if (cuvs::neighbors::cagra_required_row_width<float>(static_cast<uint32_t>(dim)) ==
        static_cast<uint32_t>(dim)) {
      // Rows already have the stride CAGRA needs: make_device_padded_dataset refuses to copy in
      // this case, so take the snapshot ourselves and index a view of it.
      aligned_snapshot_.emplace(raft::make_device_matrix<float, int64_t>(handle, n, dim));
      raft::copy(aligned_snapshot_->data_handle(),
                 candidates,
                 static_cast<std::size_t>(n) * static_cast<std::size_t>(dim),
                 raft::resource::get_cuda_stream(handle));
      index_.emplace(cuvs::neighbors::cagra::build(
        handle,
        build_params,
        cuvs::neighbors::make_device_padded_dataset_view(handle, aligned_snapshot_->view())));
    } else {
      snapshot_ = cuvs::neighbors::make_device_padded_dataset(handle, candidates_view);
      index_.emplace(cuvs::neighbors::cagra::build(handle, build_params, snapshot_->view()));
    }
  }

  /**
   * Approximate top-1 search: for each of the `m` query rows, writes the id of the nearest
   * candidate to `neighbors` and its squared L2 distance to `distances`. The distance comes from
   * the candidate's vector, so it is exact for the candidate found; only the choice is approximate.
   * An id may be invalid (>= size()) if the search found nothing.
   */
  void search(raft::resources const& handle,
              const float* queries,
              int64_t m,
              uint32_t* neighbors,
              float* distances) const
  {
    RAFT_EXPECTS(index_.has_value(), "top_1_nn_cagra_index::search called before build");
    cuvs::neighbors::cagra::search(
      handle,
      search_params_,
      *index_,
      raft::make_device_matrix_view<const float, int64_t, raft::row_major>(queries, m, dim_),
      raft::make_device_matrix_view<uint32_t, int64_t, raft::row_major>(neighbors, m, 1),
      raft::make_device_matrix_view<float, int64_t, raft::row_major>(distances, m, 1));
  }

  bool built() const { return index_.has_value(); }
  int64_t size() const { return n_; }
  int64_t dim() const { return dim_; }

 private:
  // Owns the candidate snapshot the index views: either a padded copy, or (when the rows already
  // have the required stride) a plain device matrix.
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<float, int64_t>> snapshot_;
  std::optional<raft::device_matrix<float, int64_t>> aligned_snapshot_;
  cuvs::neighbors::cagra::search_params search_params_;
  std::optional<index_type> index_;
  int64_t n_   = 0;
  int64_t dim_ = 0;
};

}  // namespace cuvs::distance::detail
