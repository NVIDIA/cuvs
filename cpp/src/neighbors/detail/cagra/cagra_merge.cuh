/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

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

#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::cagra::detail {

template <class T, class IdxT>
index<T, IdxT> merge(raft::resources const& handle,
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
  RAFT_EXPECTS(row_filter.get_filter_type() != cuvs::neighbors::filtering::FilterType::Bloom,
               "Bloom filter isn't supported inside cagra::merge");

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

}  // namespace cuvs::neighbors::cagra::detail
