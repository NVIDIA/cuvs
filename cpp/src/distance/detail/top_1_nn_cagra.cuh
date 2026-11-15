/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../top_1_nn.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>

namespace cuvs::distance::detail {

/**
 * @brief A CAGRA index over a copy of the candidate set (`y`) of a top-1 NN problem.
 *
 * Used by `Top1nnBackend::Cagra`, the approximate counterpart of the exact top-1 backends. The
 * caller builds it (once per candidate set) and hands it to `top_1_nn` through
 * `Top1nnTuning::cagra.index`.
 *
 * The index only views the candidates passed to `build`: they must stay alive and unchanged until
 * the next `build` or the destruction of this object (when their dimension does not meet CAGRA's
 * row alignment, `build` pads them into a copy it owns). The index uses squared L2 distance.
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
   * the graph degrees are clamped to `n - 1`. Candidates whose `dim` does not meet CAGRA's row
   * alignment (a multiple of 4 floats) are copied into a padded buffer owned by this object.
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
    // CAGRA requires the NN-descent graph degree to equal the intermediate graph degree (it warns
    // and realigns otherwise), so keep it in step with the clamped value.
    if (auto* nn_descent =
          std::get_if<cuvs::neighbors::cagra::graph_build_params::nn_descent_params>(
            &build_params.graph_build_params)) {
      nn_descent->graph_degree = build_params.intermediate_graph_degree;
    }

    search_params_ = search_params;

    index_.reset();  // drop the old index (and its view) before building the new one
    padded_copy_.reset();
    n_   = n;
    dim_ = dim;

    auto candidates_view =
      raft::make_device_matrix_view<const float, int64_t, raft::row_major>(candidates, n, dim);
    // CAGRA indexes rows with a padded stride. Aligned candidates are viewed in place; otherwise
    // they are copied into a padded buffer (zero padding leaves the L2 distances unchanged) that
    // lives as long as the index.
    if (cuvs::neighbors::cagra_required_row_width<float>(static_cast<uint32_t>(dim)) ==
        static_cast<uint32_t>(dim)) {
      index_.emplace(cuvs::neighbors::cagra::build(
        handle,
        build_params,
        cuvs::neighbors::make_device_padded_dataset_view(handle, candidates_view)));
    } else {
      padded_copy_ = cuvs::neighbors::make_device_padded_dataset(handle, candidates_view);
      index_.emplace(cuvs::neighbors::cagra::build(handle, build_params, padded_copy_->view()));
    }
  }

  /**
   * Approximate top-1 search: for each of the `m` query rows, writes the id of the nearest
   * candidate to `neighbors` and its squared L2 distance to `distances`. The distance comes from
   * the candidate's vector, so it is exact for the candidate found; only the choice is approximate.
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
  cuvs::neighbors::cagra::search_params search_params_;
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<float, int64_t>> padded_copy_;
  std::optional<index_type> index_;
  int64_t n_   = 0;
  int64_t dim_ = 0;
};

/**
 * Approximate top-1 via a CAGRA search of the prebuilt index over the candidates. `workspace`
 * holds the m uint32 neighbor ids returned by the search. The search writes the squared L2
 * distances straight into the Separate output; the epilog converts the ids to IdxT and applies
 * the sqrt when requested.
 */
template <typename DataT, typename IdxT, typename OutputT>
void top_1_nn_cagra(raft::resources const& handle,
                    OutputT output,
                    const DataT* x,
                    IdxT m,
                    void* workspace,
                    bool sqrt,
                    const top_1_nn_cagra_index& index)
{
  if constexpr (std::is_same_v<DataT, float> &&
                std::is_same_v<std::remove_cvref_t<OutputT>, Top1nnOutput<IdxT, float>>) {
    RAFT_EXPECTS(output.nearest_idx != nullptr && output.nearest_dist != nullptr,
                 "CAGRA top_1_nn requires index and distance outputs");
    auto* neighbors = static_cast<uint32_t*>(workspace);
    auto* dist      = output.nearest_dist;
    index.search(handle, x, static_cast<int64_t>(m), neighbors, dist);

    raft::linalg::map_offset(handle,
                             raft::make_device_vector_view<IdxT, IdxT>(output.nearest_idx, m),
                             [=] __device__(IdxT i) {
                               if (sqrt) { dist[i] = sqrtf(fmaxf(dist[i], 0.0f)); }
                               return static_cast<IdxT>(neighbors[i]);
                             });
  } else {
    RAFT_FAIL("The CAGRA top_1_nn backend requires float data and index/distance output");
  }
}

}  // namespace cuvs::distance::detail
