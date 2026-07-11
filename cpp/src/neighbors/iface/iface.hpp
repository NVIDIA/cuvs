/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <fstream>
#include <raft/core/copy.cuh>
#include <raft/core/device_resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <fstream>
#include <mutex>

#include <raft/core/device_mdspan.hpp>

namespace cuvs::neighbors {

using namespace raft;

namespace iface_detail {
/**
 * @brief True when \p mds is already CAGRA row-padded on device (device or managed memory).
 */
template <typename T, typename Accessor>
bool dataset_mdspan_uses_padded_device_view(
  raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> mds)
{
  using value_type = T;
  uint32_t const required_stride =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(mds.extent(1)));
  uint32_t const src_stride =
    mds.stride(0) > 0 ? static_cast<uint32_t>(mds.stride(0)) : static_cast<uint32_t>(mds.extent(1));
  cudaPointerAttributes a{};
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&a, mds.data_handle()));
  bool const device_src = (a.type == cudaMemoryTypeDevice) || (a.type == cudaMemoryTypeManaged);
  return device_src && (src_stride == required_stride);
}

/** Graph build via padded device view, not mdspan host build. */
template <typename AnnIndexType, typename T, typename IdxT, typename Accessor>
void cagra_build_from_device_dataset(
  raft::resources const& h,
  cagra::index_params const& cagra_params,
  raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> m,
  cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface)
{
  uint32_t const stride =
    m.stride(0) > 0 ? static_cast<uint32_t>(m.stride(0)) : static_cast<uint32_t>(m.extent(1));
  auto dview = raft::make_device_strided_matrix_view<const T, int64_t, row_major>(
    m.data_handle(), m.extent(0), m.extent(1), stride);
  if constexpr (std::is_same_v<AnnIndexType, cagra::device_padded_index<T, IdxT>>) {
    auto padded = cuvs::neighbors::make_device_padded_dataset_view(h, dview);
    auto index  = cuvs::neighbors::cagra::build(h, cagra_params, padded);
    index.update_dataset(h, padded);
    interface.index_.emplace(std::move(index));
  } else {
    auto standard = cuvs::neighbors::make_device_standard_dataset_view(dview);
    auto index    = cuvs::neighbors::cagra::build(h, cagra_params, standard);
    index.update_dataset(h, standard);
    interface.index_.emplace(std::move(index));
  }
  interface.cagra_owned_standard_dataset_.reset();
  interface.cagra_owned_padded_dataset_.reset();
}
}  // namespace iface_detail

// TODO: Refactor this function signature to use the Dataset API instead of raft::mdspan.
//       Currently takes a raw mdspan; should accept a dataset_view<...> so callers pass typed
//       views.
template <typename AnnIndexType, typename T, typename IdxT, typename Accessor>
void build(const raft::resources& handle,
           cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
           const cuvs::neighbors::index_params* index_params,
           raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> index_dataset)
{
  std::lock_guard lock(*interface.mutex_);

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    auto idx = cuvs::neighbors::ivf_flat::build(
      handle, *static_cast<const ivf_flat::index_params*>(index_params), index_dataset);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    auto idx = cuvs::neighbors::ivf_pq::build(
      handle, *static_cast<const ivf_pq::index_params*>(index_params), index_dataset);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::device_padded_index<T, IdxT>>::value ||
                       std::is_same<AnnIndexType, cagra::device_standard_index<T, IdxT>>::value) {
    const auto& cagra_params = *static_cast<const cagra::index_params*>(index_params);
    if (raft::get_device_for_address(index_dataset.data_handle()) != -1) {
      iface_detail::cagra_build_from_device_dataset<AnnIndexType>(
        handle, cagra_params, index_dataset, interface);
    } else {
      // Explicitly form a host_matrix_view so the call always resolves to the host build
      // regardless of the mdspan Accessor type (both branches compile for all Accessors;
      // at runtime this else branch is only reached when data_handle() is host memory).
      // Wrap in host_padded_dataset_view directly (graph build is CPU-side; CUDA alignment
      // is not required here — the device dataset is padded separately below).
      auto host_view = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
        index_dataset.data_handle(), index_dataset.extent(0), index_dataset.extent(1));
      if constexpr (std::is_same<AnnIndexType, cagra::device_padded_index<T, IdxT>>::value) {
        cuvs::neighbors::host_padded_dataset_view<T, int64_t> host_padded(
          host_view, static_cast<uint32_t>(host_view.extent(1)));
        auto host_idx   = cuvs::neighbors::cagra::build(handle, cagra_params, host_padded);
        auto padded_r   = cuvs::neighbors::make_device_padded_dataset(handle, index_dataset);
        auto device_idx = cuvs::neighbors::cagra::attach_device_dataset_on_host_index(
          handle, host_idx, padded_r->as_dataset_view());
        interface.cagra_owned_padded_dataset_ = std::move(padded_r);
        interface.cagra_owned_standard_dataset_.reset();
        interface.index_.emplace(std::move(device_idx));
      } else {
        auto host_standard = cuvs::neighbors::make_host_standard_dataset_view(host_view);
        auto host_idx      = cuvs::neighbors::cagra::build(handle, cagra_params, host_standard);
        auto standard_r    = cuvs::neighbors::make_device_standard_dataset(
          handle,
          index_dataset,
          static_cast<uint32_t>(index_dataset.extent(1)),
          static_cast<uint32_t>(index_dataset.stride(0)));
        auto device_idx = cuvs::neighbors::cagra::attach_device_dataset_on_host_index(
          handle, host_idx, standard_r->as_dataset_view());
        interface.cagra_owned_standard_dataset_ = std::move(standard_r);
        interface.cagra_owned_padded_dataset_.reset();
        interface.index_.emplace(std::move(device_idx));
      }
    }
  }
  resource::sync_stream(handle);
}

template <typename AnnIndexType, typename T, typename IdxT, typename Accessor1, typename Accessor2>
void extend(
  const raft::resources& handle,
  cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
  raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor1> new_vectors,
  std::optional<raft::mdspan<const IdxT, vector_extent<int64_t>, layout_c_contiguous, Accessor2>>
    new_indices)
{
  std::lock_guard lock(*interface.mutex_);

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    auto idx =
      cuvs::neighbors::ivf_flat::extend(handle, new_vectors, new_indices, interface.index_.value());
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    auto idx =
      cuvs::neighbors::ivf_pq::extend(handle, new_vectors, new_indices, interface.index_.value());
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::device_padded_index<T, IdxT>>::value) {
    RAFT_FAIL("CAGRA does not implement the extend method");
  }
  resource::sync_stream(handle);
}

template <typename AnnIndexType, typename T, typename IdxT, typename searchIdxT>
void search(const raft::resources& handle,
            const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
            const cuvs::neighbors::search_params* search_params,
            raft::device_matrix_view<const T, int64_t, row_major> queries,
            raft::device_matrix_view<searchIdxT, int64_t, row_major> neighbors,
            raft::device_matrix_view<float, int64_t, row_major> distances)
{
  // std::lock_guard(*interface.mutex_);
  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, int64_t>>::value) {
    cuvs::neighbors::ivf_flat::search(
      handle,
      *reinterpret_cast<const ivf_flat::search_params*>(search_params),
      interface.index_.value(),
      queries,
      neighbors,
      distances);
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<int64_t>>::value) {
    cuvs::neighbors::ivf_pq::search(handle,
                                    *reinterpret_cast<const ivf_pq::search_params*>(search_params),
                                    interface.index_.value(),
                                    queries,
                                    neighbors,
                                    distances);
  } else if constexpr (std::is_same<AnnIndexType, cagra::device_padded_index<T, uint32_t>>::value ||
                       std::is_same<AnnIndexType,
                                    cagra::device_standard_index<T, uint32_t>>::value) {
    cuvs::neighbors::cagra::search(handle,
                                   *reinterpret_cast<const cagra::search_params*>(search_params),
                                   interface.index_.value(),
                                   queries,
                                   neighbors,
                                   distances);
  }
  // resource::sync_stream(handle);
}

// for MG ANN only
template <typename AnnIndexType, typename T, typename IdxT, typename searchIdxT>
void search(const raft::resources& handle,
            const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
            const cuvs::neighbors::search_params* search_params,
            raft::host_matrix_view<const T, int64_t, row_major> h_queries,
            raft::device_matrix_view<searchIdxT, int64_t, row_major> d_neighbors,
            raft::device_matrix_view<float, int64_t, row_major> d_distances)
{
  // std::lock_guard(*interface.mutex_);

  int64_t n_rows = h_queries.extent(0);
  int64_t n_dims = h_queries.extent(1);
  auto d_queries = raft::make_device_matrix<T, int64_t, row_major>(handle, n_rows, n_dims);
  raft::copy(handle, d_queries.view(), h_queries);
  auto d_query_view = raft::make_const_mdspan(d_queries.view());

  search(handle, interface, search_params, d_query_view, d_neighbors, d_distances);
}

template <typename AnnIndexType, typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
               std::ostream& os)
{
  std::lock_guard lock(*interface.mutex_);

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    ivf_flat::serialize(handle, os, interface.index_.value());
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    ivf_pq::serialize(handle, os, interface.index_.value());
  } else if constexpr (std::is_same<AnnIndexType, cagra::device_padded_index<T, IdxT>>::value ||
                       std::is_same<AnnIndexType, cagra::device_standard_index<T, IdxT>>::value) {
    cagra::serialize(handle, os, interface.index_.value(), true);
  }

  resource::sync_stream(handle);
}

template <typename AnnIndexType, typename T, typename IdxT>
void deserialize(const raft::resources& handle,
                 cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
                 std::istream& is)
{
  std::lock_guard lock(*interface.mutex_);

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    ivf_flat::index<T, IdxT> idx(handle);
    ivf_flat::deserialize(handle, is, &idx);
    resource::sync_stream(handle);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    ivf_pq::index<IdxT> idx(handle);
    ivf_pq::deserialize(handle, is, &idx);
    resource::sync_stream(handle);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::device_padded_index<T, IdxT>>::value) {
    cagra::device_padded_index<T, IdxT> idx(handle);
    std::unique_ptr<cuvs::neighbors::device_padded_dataset<T, int64_t>> out_dataset;
    cagra::deserialize(handle, is, &idx, &out_dataset);
    interface.cagra_owned_padded_dataset_.reset();
    interface.cagra_owned_standard_dataset_.reset();
    if (out_dataset) { interface.cagra_owned_padded_dataset_ = std::move(out_dataset); }
    resource::sync_stream(handle);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::device_standard_index<T, IdxT>>::value) {
    cagra::device_standard_index<T, IdxT> idx(handle);
    std::unique_ptr<cuvs::neighbors::device_standard_dataset<T, int64_t>> out_dataset;
    cagra::deserialize(handle, is, &idx, &out_dataset);
    interface.cagra_owned_padded_dataset_.reset();
    interface.cagra_owned_standard_dataset_.reset();
    if (out_dataset) { interface.cagra_owned_standard_dataset_ = std::move(out_dataset); }
    resource::sync_stream(handle);
    interface.index_.emplace(std::move(idx));
  }
}

template <typename AnnIndexType, typename T, typename IdxT>
void deserialize(const raft::resources& handle,
                 cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
                 const std::string& filename)
{
  std::lock_guard lock(*interface.mutex_);

  std::ifstream is(filename, std::ios::in | std::ios::binary);
  if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    ivf_flat::index<T, IdxT> idx(handle);
    ivf_flat::deserialize(handle, is, &idx);
    resource::sync_stream(handle);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    ivf_pq::index<IdxT> idx(handle);
    ivf_pq::deserialize(handle, is, &idx);
    resource::sync_stream(handle);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::device_padded_index<T, IdxT>>::value) {
    cagra::device_padded_index<T, IdxT> idx(handle);
    std::unique_ptr<cuvs::neighbors::device_padded_dataset<T, int64_t>> out_dataset;
    cagra::deserialize(handle, is, &idx, &out_dataset);
    interface.cagra_owned_padded_dataset_.reset();
    interface.cagra_owned_standard_dataset_.reset();
    if (out_dataset) { interface.cagra_owned_padded_dataset_ = std::move(out_dataset); }
    resource::sync_stream(handle);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::device_standard_index<T, IdxT>>::value) {
    cagra::device_standard_index<T, IdxT> idx(handle);
    std::unique_ptr<cuvs::neighbors::device_standard_dataset<T, int64_t>> out_dataset;
    cagra::deserialize(handle, is, &idx, &out_dataset);
    interface.cagra_owned_padded_dataset_.reset();
    interface.cagra_owned_standard_dataset_.reset();
    if (out_dataset) { interface.cagra_owned_standard_dataset_ = std::move(out_dataset); }
    resource::sync_stream(handle);
    interface.index_.emplace(std::move(idx));
  }

  is.close();
}

};  // namespace cuvs::neighbors
