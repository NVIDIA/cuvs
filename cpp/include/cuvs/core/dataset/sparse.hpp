/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/type.hpp>
#include <cuvs/core/export.hpp>

#include <raft/core/coo_matrix.hpp>
#include <raft/core/csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>

#include <cstdint>
#include <type_traits>

namespace CUVS_EXPORT cuvs {

/**
 * Sparse dataset specification implemented via the raft sparse matrix hierarchy. `SparseLayoutT`
 * selects the representation (`csr_layout`, `coo_layout`).
 *
 * NOTE: only the memory space of the container policy is used here; raft's sparse types bind the
 * policy themselves, separately for the values, the offsets and the indices.
 */
template <typename SparseLayoutT, typename ContainerPolicy>
struct sparse_spec {
  template <typename T, typename IdxT>
  struct apply {
    using value_type = std::remove_cv_t<T>;
    using index_type = std::remove_cv_t<IdxT>;

    static constexpr raft::memory_type mem_type = ContainerPolicy::mem_type;
    static_assert(mem_type == raft::memory_type::host || mem_type == raft::memory_type::device,
                  "raft's sparse hierarchy only provides host and device containers");

    using data_type = std::conditional_t<
      mem_type == raft::memory_type::device,
      typename SparseLayoutT::
        template matrix_type<value_type, index_type, true, raft::device_container_policy>,
      typename SparseLayoutT::
        template matrix_type<value_type, index_type, false, raft::host_container_policy>>;
    using data_view_type = typename data_type::view_type;
    // NOTE: raft's sparse hierarchy provides no read-only view yet (see the TODO in
    // raft/core/sparse_types.hpp); until then, even reading the structure requires a mutable
    // matrix.
    //
    // Consequently a read-only `dataset_view<const T, ...>` over a sparse spec is read-only in name
    // only: the two view types coincide, so writing through it still compiles. That resolves itself
    // once raft grows a const sparse view.
    using const_data_view_type = data_view_type;

    [[nodiscard]] static auto get_data_view(data_type& data) noexcept -> data_view_type
    {
      return data.view();
    }

    [[nodiscard]] static auto get_data_const_view(const data_type& data) noexcept
      -> const_data_view_type
    {
      return const_cast<data_type&>(data).view();
    }
    // NOTE: there is no read-only view to convert to (see above), so the view passes through.
    [[nodiscard]] static auto get_data_const_view(data_view_type data) noexcept
      -> const_data_view_type
    {
      return data;
    }
    // NOTE: raft's sparse hierarchy exposes the structure through a mutable accessor only, so
    // reading the shape of either the owning matrix or its view takes a cast.
    [[nodiscard]] static auto get_structure_view(const auto& data) noexcept
    {
      return const_cast<std::remove_cvref_t<decltype(data)>&>(data).structure_view();
    }
    // NOTE: the shape accessors take the data in either form - the storage an owning dataset holds,
    // or the view a `dataset_view` holds. A plain spec defines no manifest either, so it ignores
    // whatever the dataset puts in that slot instead.
    [[nodiscard]] static auto get_n_rows(const auto& data) noexcept -> index_type
    {
      return static_cast<index_type>(get_structure_view(data).get_n_rows());
    }
    [[nodiscard]] static auto get_dim(const auto& data, const auto&) noexcept -> uint32_t
    {
      return static_cast<uint32_t>(get_structure_view(data).get_n_cols());
    }
  };
};

/**
 * CSR representation: one compressed row per vector of the dataset.
 *
 * NOTE: the row offsets are 64-bit for the same reason the dense extents are: the number of
 * non-zeros is not bounded by the index type. The column indices are feature ids, so they are
 * bounded by the dataset dimension rather than by its size.
 */
struct csr_layout {
  template <typename T, typename IdxT, bool IsDevice, template <typename> typename ContainerPolicy>
  using matrix_type = raft::csr_matrix<T, int64_t, uint32_t, uint64_t, IsDevice, ContainerPolicy>;
};

/**
 * COO representation: one (row, column, value) triple per non-zero.
 *
 * NOTE: unlike CSR, the row component is an index into the dataset rather than an offset into the
 * non-zeros, hence the dataset index type.
 */
struct coo_layout {
  template <typename T, typename IdxT, bool IsDevice, template <typename> typename ContainerPolicy>
  using matrix_type = raft::coo_matrix<T, IdxT, uint32_t, uint64_t, IsDevice, ContainerPolicy>;
};

template <typename T, typename IdxT, typename ContainerPolicy>
using csr_dataset = dataset<T, IdxT, sparse_spec<csr_layout, ContainerPolicy>>;
template <typename T, typename IdxT, typename ContainerPolicy>
using coo_dataset = dataset<T, IdxT, sparse_spec<coo_layout, ContainerPolicy>>;
template <typename T, typename IdxT>
using device_csr_dataset =
  csr_dataset<T, IdxT, raft::device_accessor<raft::device_container_policy<T>>>;
template <typename T, typename IdxT>
using device_coo_dataset =
  coo_dataset<T, IdxT, raft::device_accessor<raft::device_container_policy<T>>>;
template <typename T, typename IdxT>
using host_csr_dataset = csr_dataset<T, IdxT, raft::host_accessor<raft::host_container_policy<T>>>;
template <typename T, typename IdxT>
using host_coo_dataset = coo_dataset<T, IdxT, raft::host_accessor<raft::host_container_policy<T>>>;

}  // namespace CUVS_EXPORT cuvs
