/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../core/nvtx.hpp"
#include "cagra_merge_scaffold.cuh"
#include "graph_core.cuh"

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/refine.hpp>

#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace cuvs::neighbors::cagra::detail {

/** Copy every attached input dataset into its row range in one contiguous device matrix. */
template <typename T, typename IdxT>
void copy_input_datasets(raft::resources const& handle,
                         std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> const& indices,
                         std::vector<int64_t> const& offsets,
                         int64_t dim,
                         T* destination)
{
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT>;
  using ds_idx_type   = typename cagra_index_t::dataset_index_type;

  for (std::size_t i = 0; i < indices.size(); ++i) {
    auto const* source_dataset =
      dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&indices[i]->data());
    auto source = source_dataset->view();
    raft::copy_matrix(destination + offsets[i] * dim,
                      static_cast<std::size_t>(dim),
                      source.data_handle(),
                      static_cast<std::size_t>(source_dataset->stride()),
                      static_cast<std::size_t>(dim),
                      static_cast<std::size_t>(source.extent(0)),
                      raft::resource::get_cuda_stream(handle));
  }
}

template <class T, class IdxT>
index<T, IdxT> merge_rebuild(raft::resources const& handle,
                             const cagra::index_params& params,
                             std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices,
                             const cuvs::neighbors::filtering::base_filter& row_filter)
{
  using cagra_index_t = cuvs::neighbors::cagra::index<T, IdxT>;
  using ds_idx_type   = typename cagra_index_t::dataset_index_type;

  std::size_t dim              = 0;
  std::size_t new_dataset_size = 0;
  int64_t stride               = -1;

  RAFT_EXPECTS(row_filter.get_filter_type() != cuvs::neighbors::filtering::FilterType::Bitmap,
               "Bitmap filter isn't supported inside cagra::merge");

  for (cagra_index_t* index : indices) {
    RAFT_EXPECTS(index != nullptr,
                 "Null pointer detected in 'indices'. Ensure all elements are valid before usage.");
    if (auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index->data());
        strided_dset != nullptr) {
      if (dim == 0) {
        dim    = index->dim();
        stride = strided_dset->stride();
      } else {
        RAFT_EXPECTS(dim == index->dim(), "Dimension of datasets in indices must be equal.");
      }
      new_dataset_size += index->size();
    } else if (dynamic_cast<const cuvs::neighbors::empty_dataset<int64_t>*>(&index->data()) !=
               nullptr) {
      RAFT_FAIL(
        "cagra::merge only supports an index to which the dataset is attached. Please check if the "
        "index was built with index_param.attach_dataset_on_build = true, or if a dataset was "
        "attached after the build.");
    } else {
      RAFT_FAIL("cagra::merge only supports an uncompressed dataset index");
    }
  }

  IdxT offset = 0;

  auto merge_dataset = [&](T* dst) {
    for (cagra_index_t* index : indices) {
      auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index->data());
      raft::copy_matrix(dst + offset * dim,
                        dim,
                        strided_dset->view().data_handle(),
                        static_cast<size_t>(stride),
                        dim,
                        static_cast<size_t>(strided_dset->n_rows()),
                        raft::resource::get_cuda_stream(handle));

      offset += IdxT(index->data().n_rows());
    }
  };

  try {
    auto updated_dataset =
      raft::make_device_matrix<T, int64_t>(handle, int64_t(new_dataset_size), int64_t(dim));

    merge_dataset(updated_dataset.data_handle());

    if (row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::Bitset) {
      auto actual_filter =
        dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(
          row_filter);
      auto filtered_row_count = actual_filter.view().count(handle);

      // Convert the filter to a CSR matrix (so that we can pass indices to raft::copy_rows)
      auto indices_csr = raft::make_device_csr_matrix<uint32_t, int64_t, int64_t, int64_t>(
        handle, 1, new_dataset_size);
      indices_csr.initialize_sparsity(filtered_row_count);

      actual_filter.view().to_csr(handle, indices_csr);

      // Get the indices array from the csr matrix. Note that this returns a raft::span object
      // and we need to pass as device_vector_view, which is a 1D mdspan (instead of a span)
      // so we need to translate here (and adjust to be const)
      auto indices      = indices_csr.structure_view().get_indices();
      auto indices_view = raft::make_device_vector_view<const int64_t, int64_t>(
        indices.data(), static_cast<int64_t>(indices.size()));

      auto filtered_dataset = raft::make_device_matrix<T, int64_t>(handle, filtered_row_count, dim);
      raft::matrix::copy_rows(handle,
                              raft::make_const_mdspan(updated_dataset.view()),
                              filtered_dataset.view(),
                              indices_view);

      auto merged_index =
        cagra::build(handle, params, raft::make_const_mdspan(filtered_dataset.view()));
      if (!merged_index.data().is_owning() && params.attach_dataset_on_build) {
        using matrix_t           = decltype(updated_dataset);
        using layout_t           = typename matrix_t::layout_type;
        using container_policy_t = typename matrix_t::container_policy_type;
        using owning_t           = owning_dataset<T, int64_t, layout_t, container_policy_t>;
        auto out_layout          = raft::make_strided_layout(filtered_dataset.view().extents(),
                                                    cuda::std::array<int64_t, 2>{stride, 1});

        merged_index.update_dataset(handle, owning_t{std::move(filtered_dataset), out_layout});
      }
      RAFT_LOG_DEBUG("cagra merge: using device memory for merged dataset");
      return merged_index;
    } else {
      auto merged_index =
        cagra::build(handle, params, raft::make_const_mdspan(updated_dataset.view()));
      if (!merged_index.data().is_owning() && params.attach_dataset_on_build) {
        using matrix_t           = decltype(updated_dataset);
        using layout_t           = typename matrix_t::layout_type;
        using container_policy_t = typename matrix_t::container_policy_type;
        using owning_t           = owning_dataset<T, int64_t, layout_t, container_policy_t>;
        auto out_layout          = raft::make_strided_layout(updated_dataset.view().extents(),
                                                    cuda::std::array<int64_t, 2>{stride, 1});

        merged_index.update_dataset(handle, owning_t{std::move(updated_dataset), out_layout});
      }
      RAFT_LOG_DEBUG("cagra merge: using device memory for merged dataset");
      return merged_index;
    }
  } catch (std::bad_alloc& e) {
    // We don't currently support the cpu memory fallback with filtered merge, since the
    // 'raft::matrix::copy_rows' only supports gpu memory
    RAFT_EXPECTS(row_filter.get_filter_type() == cuvs::neighbors::filtering::FilterType::None,
                 "Filtered merge isn't available on cpu memory");

    RAFT_LOG_DEBUG("cagra::merge: using host memory for merged dataset");

    auto updated_dataset =
      raft::make_host_matrix<T, std::int64_t>(std::int64_t(new_dataset_size), std::int64_t(dim));

    merge_dataset(updated_dataset.data_handle());

    auto merged_index =
      cagra::build(handle, params, raft::make_const_mdspan(updated_dataset.view()));
    if (!merged_index.data().is_owning() && params.attach_dataset_on_build) {
      using matrix_t           = decltype(updated_dataset);
      using layout_t           = typename matrix_t::layout_type;
      using container_policy_t = typename matrix_t::container_policy_type;
      using owning_t           = owning_dataset<T, int64_t, layout_t, container_policy_t>;
      auto out_layout          = raft::make_strided_layout(updated_dataset.view().extents(),
                                                  cuda::std::array<int64_t, 2>{stride, 1});
      merged_index.update_dataset(handle, owning_t{std::move(updated_dataset), out_layout});
    }
    return merged_index;
  }
}

struct fastener_preflight_result {
  bool eligible = false;
  int64_t rows  = 0;
  int64_t dim   = 0;
  std::vector<int64_t> offsets;
  std::string reason;
};

template <typename T, typename IdxT>
auto preflight_fastener(cagra::index_params const& params,
                        cagra::merge_params const& merge_params,
                        std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> const& indices,
                        cuvs::neighbors::filtering::base_filter const& row_filter)
  -> fastener_preflight_result
{
  fastener_preflight_result result;
  auto reject = [&](std::string reason) {
    result.reason = std::move(reason);
    return result;
  };

  if constexpr (!(std::is_same_v<T, float> || std::is_same_v<T, half> ||
                  std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) ||
                !std::is_same_v<IdxT, uint32_t>) {
    return reject("the scalar or graph index type is unsupported");
  }
  if (indices.size() < 2) { return reject("at least two input indices are required"); }
  if (row_filter.get_filter_type() != cuvs::neighbors::filtering::FilterType::None) {
    return reject("row filters are not supported");
  }
  if (params.compression.has_value()) { return reject("compressed output is not supported"); }
  if (params.metric != cuvs::distance::DistanceType::L2Expanded) {
    return reject("only L2Expanded is supported");
  }
  if (merge_params.levels == 0) { return reject("levels must be positive"); }
  if (merge_params.root_fanout < 1 || merge_params.root_fanout > merge_scaffold::MAX_FANOUT ||
      merge_params.lower_fanout < 1 || merge_params.lower_fanout > merge_scaffold::MAX_FANOUT) {
    return reject("root_fanout and lower_fanout must be between 1 and " +
                  std::to_string(merge_scaffold::MAX_FANOUT));
  }
  if (!(merge_params.leader_fraction > 0.0 && merge_params.leader_fraction <= 1.0)) {
    return reject("leader_fraction must be in (0, 1]");
  }
  if (merge_params.max_leaders == 0 || merge_params.max_leaders > merge_scaffold::MAX_LEADERS) {
    return reject("max_leaders must be between 1 and " +
                  std::to_string(merge_scaffold::MAX_LEADERS));
  }
  if (merge_params.max_leaders < std::max(merge_params.root_fanout, merge_params.lower_fanout)) {
    return reject("max_leaders must cover both configured fanouts");
  }
  if (merge_params.leaf_size == 0 || merge_params.leaf_size > merge_scaffold::MAX_LEAF_SIZE) {
    return reject("leaf_size must be between 1 and " +
                  std::to_string(merge_scaffold::MAX_LEAF_SIZE));
  }
  if (merge_params.leaf_degree == 0 ||
      merge_params.leaf_degree > static_cast<uint32_t>(merge_scaffold::MAX_LEAF_DEGREE)) {
    return reject("leaf_degree must be between 1 and " +
                  std::to_string(merge_scaffold::MAX_LEAF_DEGREE));
  }

  uint64_t const max_spill = std::numeric_limits<uint8_t>::max() / merge_params.leaf_degree;
  uint64_t spill           = merge_params.root_fanout;
  auto const candidate_width_limit =
    "root_fanout * lower_fanout^(levels - 1) * leaf_degree must not exceed " +
    std::to_string(std::numeric_limits<uint8_t>::max());
  if (spill > max_spill) { return reject(candidate_width_limit); }
  if (merge_params.lower_fanout > 1) {
    for (uint32_t level = 1; level < merge_params.levels; ++level) {
      if (spill > max_spill / merge_params.lower_fanout) { return reject(candidate_width_limit); }
      spill *= merge_params.lower_fanout;
    }
  }
  uint64_t const scaffold_degree = spill * merge_params.leaf_degree;

  using index_type          = cuvs::neighbors::cagra::index<T, IdxT>;
  using ds_idx_type         = typename index_type::dataset_index_type;
  uint64_t rows             = 0;
  uint64_t max_input_degree = 0;
  result.offsets.reserve(indices.size() + 1);
  result.offsets.push_back(0);

  for (auto const* index : indices) {
    if (index == nullptr) { return reject("all input index pointers must be non-null"); }
    auto const* dataset =
      dynamic_cast<cuvs::neighbors::strided_dataset<T, ds_idx_type> const*>(&index->data());
    if (dataset == nullptr || dataset->n_rows() != static_cast<ds_idx_type>(index->size())) {
      return reject("every input must have an attached, uncompressed dataset");
    }
    if (index->metric() != params.metric) {
      return reject("every input metric must match index_params.metric");
    }
    if (result.offsets.size() == 1) {
      result.dim = static_cast<int64_t>(index->dim());
    } else if (result.dim != static_cast<int64_t>(index->dim())) {
      return reject("all input dimensions must match");
    }
    auto graph = index->graph();
    if (graph.extent(0) <= 0 || graph.extent(1) <= 0 ||
        graph.extent(0) != static_cast<int64_t>(index->size())) {
      return reject("every input must have a nonempty device graph");
    }

    auto const input_rows = static_cast<uint64_t>(index->size());
    if (rows > std::numeric_limits<uint32_t>::max() - input_rows) {
      return reject("the combined row count must fit in uint32_t");
    }
    rows += input_rows;
    max_input_degree = std::max<uint64_t>(max_input_degree, static_cast<uint64_t>(graph.extent(1)));
    result.offsets.push_back(static_cast<int64_t>(rows));
  }

  if (result.dim <= 0 || result.dim > std::numeric_limits<int>::max()) {
    return reject("dataset dimension must be positive and fit cuBLAS int dimensions");
  }
  if (!merge_scaffold::leaf_gemm_supported<T>(result.dim, merge_params.leaf_size)) {
    if constexpr (std::is_integral_v<T>) {
      return reject("integer dataset dimension exceeds the INT32 leaf GEMM limit");
    } else {
      return reject("dataset dimension exceeds the leaf GEMM workspace limit");
    }
  }
  if (rows > std::numeric_limits<uint32_t>::max() / spill) {
    return reject("combined rows times the configured spill width must fit in uint32_t");
  }
  if (params.graph_degree == 0 || static_cast<uint64_t>(params.graph_degree) >= rows) {
    return reject("graph_degree must be positive and smaller than the combined row count");
  }
  if (static_cast<uint64_t>(params.graph_degree) > max_input_degree + scaffold_degree) {
    return reject("graph_degree exceeds the input graph plus scaffold capacity");
  }

  result.rows     = static_cast<int64_t>(rows);
  result.eligible = true;
  return result;
}

template <typename T, typename IdxT>
auto merge_fastener(raft::resources const& handle,
                    cagra::index_params const& params,
                    cagra::merge_params const& merge_params,
                    std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices,
                    fastener_preflight_result const& preflight) -> index<T, IdxT>
{
  auto dataset = [&] {
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> scope("cagra::merge/consolidate");
    auto combined = raft::make_device_matrix<T, int64_t>(handle, preflight.rows, preflight.dim);
    copy_input_datasets(handle, indices, preflight.offsets, preflight.dim, combined.data_handle());
    return combined;
  }();

  merge_scaffold::build_params scaffold_params;
  scaffold_params.levels          = merge_params.levels;
  scaffold_params.root_fanout     = merge_params.root_fanout;
  scaffold_params.lower_fanout    = merge_params.lower_fanout;
  scaffold_params.leader_fraction = merge_params.leader_fraction;
  scaffold_params.max_leaders     = merge_params.max_leaders;
  scaffold_params.leaf_size       = merge_params.leaf_size;
  scaffold_params.leaf_degree     = merge_params.leaf_degree;

  auto merged_graph = [&] {
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> scope("cagra::merge/scaffold");
    auto scaffold = merge_scaffold::build<T>(
      handle, raft::make_const_mdspan(dataset.view()), preflight.offsets, scaffold_params);
    return merge_scaffold::append_to_input_graphs<T, IdxT>(
      handle, indices, preflight.offsets, raft::make_const_mdspan(scaffold.view()));
  }();

  {
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> scope("cagra::merge/append/sort");
    cagra::detail::graph::sort_knn_graph_device_inplace(
      handle, params.metric, raft::make_const_mdspan(dataset.view()), merged_graph.view());
    merged_graph = merge_scaffold::cap_sorted_graph(
      handle, raft::make_const_mdspan(merged_graph.view()), params.graph_degree);
  }

  auto optimized_graph =
    raft::make_device_matrix<uint32_t, int64_t>(handle, preflight.rows, params.graph_degree);
  {
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> scope("cagra::merge/optimize");
    cagra::detail::graph::optimize(
      handle, merged_graph.view(), optimized_graph.view(), params.guarantee_connectivity);
  }

  index<T, IdxT> merged_index(handle, params.metric);
  merged_index.update_graph(handle, std::move(optimized_graph));
  if (params.attach_dataset_on_build) {
    // CAGRA search assumes 16-byte row alignment (vectorized loads). Consolidate remains dense for
    // the scaffold; attach through make_aligned_dataset so unaligned dims are padded.
    merged_index.update_dataset(handle, make_aligned_dataset(handle, std::move(dataset), 16));
  } else {
    using ds_idx_type = typename index<T, IdxT>::dataset_index_type;
    merged_index.update_dataset(
      handle, std::make_unique<cuvs::neighbors::empty_dataset<ds_idx_type>>(preflight.dim));
  }

  return merged_index;
}

template <class T, class IdxT>
index<T, IdxT> merge(raft::resources const& handle,
                     cagra::index_params const& params,
                     std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices,
                     cagra::merge_params const& merge_params,
                     cuvs::neighbors::filtering::base_filter const& row_filter)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> merge_scope(
    "cagra::merge(algo=%d,parts=%zu)", static_cast<int>(merge_params.algo), indices.size());

  RAFT_EXPECTS(merge_params.algo == cagra::merge_algo::AUTO ||
                 merge_params.algo == cagra::merge_algo::FASTENER ||
                 merge_params.algo == cagra::merge_algo::REBUILD,
               "Unknown cagra::merge algorithm");
  if (merge_params.algo == cagra::merge_algo::REBUILD) {
    return merge_rebuild(handle, params, indices, row_filter);
  }

  auto preflight = preflight_fastener<T, IdxT>(params, merge_params, indices, row_filter);
  if (!preflight.eligible) {
    if (merge_params.algo == cagra::merge_algo::AUTO) {
      return merge_rebuild(handle, params, indices, row_filter);
    }
    RAFT_FAIL("FASTENER cagra::merge is unsupported: %s", preflight.reason.c_str());
  }

  return merge_fastener(handle, params, merge_params, indices, preflight);
}

template <class T, class IdxT>
index<T, IdxT> merge(raft::resources const& handle,
                     cagra::index_params const& params,
                     std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices,
                     cuvs::neighbors::filtering::base_filter const& row_filter)
{
  return merge(handle, params, indices, cagra::merge_params{}, row_filter);
}

}  // namespace cuvs::neighbors::cagra::detail
