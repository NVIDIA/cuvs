/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>   // get_device_for_address, copy_matrix
#include <raft/util/integer_utils.hpp>  // rounding up

#include <cuvs/core/bitmap.hpp>
#include <cuvs/core/bitset.hpp>
#include <cuvs/core/export.hpp>
#include <raft/core/detail/macros.hpp>

#include <cuda_fp16.h>

#include <concepts>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#ifdef __cpp_lib_bitops
#include <bit>
#endif

namespace CUVS_EXPORT cuvs {
namespace neighbors {
/**
 * @addtogroup cagra_cpp_index_params
 * @{
 */

/* Graph build algo used in cagra and all_neighbors */
enum GRAPH_BUILD_ALGO { BRUTE_FORCE = 0, IVF_PQ = 1, NN_DESCENT = 2, ACE = 3 };

/** Parameters for VPQ compression. */
struct vpq_params {
  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible values: [4, 5, 6, 7, 8].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the better the search
   * performance, but the lower the recall.
   */
  uint32_t pq_bits = 8;
  /**
   * The dimensionality of the vector after compression by PQ.
   * When zero, an optimal value is selected using a heuristic.
   *
   * TODO: at the moment `dim` must be a multiple `pq_dim`.
   */
  uint32_t pq_dim = 0;
  /**
   * Vector Quantization (VQ) codebook size - number of "coarse cluster centers".
   * When zero, an optimal value is selected using a heuristic.
   */
  uint32_t vq_n_centers = 0;
  /** The number of iterations searching for kmeans centers (both VQ & PQ phases). */
  uint32_t kmeans_n_iters = 25;
  /**
   * The fraction of data to use during iterative kmeans building (VQ phase).
   * When zero, an optimal value is selected using a heuristic.
   * @deprecated Prefer using `max_train_points_per_vq_cluster` instead.
   */
  double vq_kmeans_trainset_fraction = 0;
  /**
   * The fraction of data to use during iterative kmeans building (PQ phase).
   * When zero, an optimal value is selected using a heuristic.
   * @deprecated Prefer using `max_train_points_per_pq_code` instead.
   */
  double pq_kmeans_trainset_fraction = 0;
  /**
   * Type of k-means algorithm for PQ training.
   * Balanced k-means tends to be faster than regular k-means for PQ training, for
   * problem sets where the number of points per cluster are approximately equal.
   * Regular k-means may be better for skewed cluster distributions.
   */
  cuvs::cluster::kmeans::kmeans_type pq_kmeans_type =
    cuvs::cluster::kmeans::kmeans_type::KMeansBalanced;
  /**
   * The max number of data points to use per PQ code during PQ codebook training. Using more data
   * points per PQ code may increase the quality of PQ codebook but may also increase the build
   * time. We will use `pq_n_centers * max_train_points_per_pq_code` training
   * points to train each PQ codebook.
   */
  uint32_t max_train_points_per_pq_code = 256;
  /**
   * The max number of data points to use per VQ cluster during training.
   */
  uint32_t max_train_points_per_vq_cluster = 1024;
};

/** @} */  // end group cagra_cpp_index_params

/**
 * @defgroup neighbors_index Approximate Nearest Neighbors Types
 * @{
 */

/** The base for approximate KNN index structures. */
struct index {};

/** The base for KNN index parameters. */
struct index_params {
  /** Distance type. */
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
  /** The argument used by some distance metrics. */
  float metric_arg = 2.0f;
};

struct search_params {};

/**
 * @brief Strategy for merging indices.
 *
 * This enum is declared separately to avoid namespace pollution when including common.hpp.
 * It provides a generic merge strategy that can be used across different index types.
 */
enum class MergeStrategy {
  /** Merge indices physically by combining their data structures */
  MERGE_STRATEGY_PHYSICAL = 0,
  /** Merge indices logically by creating a composite wrapper */
  MERGE_STRATEGY_LOGICAL = 1
};

/** @} */  // end group neighbors_index

/**
 * @brief Tags selecting dataset representation for `dataset` / `dataset_view`.
 *
 * Each container defines nested `owning_storage` then `view_storage` (aliases into `detail::*`
 * storage types shared by device/host). Accessibility (device vs host) is selected by the
 * `Accessor` template parameter on `dataset` / `dataset_view`, not by duplicating containers.
 * Layout kinds: empty, padded, standard, VPQ. `dataset` / `dataset_view` only express ownership
 * vs view.
 */

template <typename ContainerType, typename DataT, typename IdxT, typename Accessor>
struct dataset;

template <typename ContainerType, typename DataT, typename IdxT, typename Accessor>
struct dataset_view;

namespace detail {

// Default owning/view accessors for public dataset aliases.
template <typename T>
using device_owning_accessor = raft::device_accessor<raft::device_container_policy<T>>;

template <typename T>
using host_owning_accessor = raft::host_accessor<raft::host_container_policy<T>>;

template <typename T>
using device_view_accessor = raft::device_accessor<cuda::std::default_accessor<const T>>;

template <typename T>
using host_view_accessor = raft::host_accessor<cuda::std::default_accessor<const T>>;

/** View accessor paired with an owning dataset accessor (same residency). */
template <typename DataT, typename Accessor>
using dataset_view_accessor_for_owning = std::conditional_t<Accessor::is_device_accessible,
                                                            device_view_accessor<DataT>,
                                                            host_view_accessor<DataT>>;

/** Owning accessor paired with a view accessor (same residency). */
template <typename DataT, typename Accessor>
using dataset_owning_accessor_for_view = std::conditional_t<Accessor::is_device_accessible,
                                                            device_owning_accessor<DataT>,
                                                            host_owning_accessor<DataT>>;

template <typename DataT, typename IdxT, typename Accessor>
using dense_owning_matrix = std::conditional_t<Accessor::is_device_accessible,
                                               raft::device_matrix<DataT, IdxT, raft::row_major>,
                                               raft::host_matrix<DataT, IdxT, raft::row_major>>;

template <typename DataT, typename IdxT, typename Accessor>
using dense_view_matrix =
  std::conditional_t<Accessor::is_device_accessible,
                     raft::device_matrix_view<const DataT, IdxT, raft::row_major>,
                     raft::host_matrix_view<const DataT, IdxT, raft::row_major>>;

template <typename MathT, typename IdxT, typename Accessor>
using vpq_vq_book_matrix = std::conditional_t<Accessor::is_device_accessible,
                                              raft::device_matrix<MathT, uint32_t, raft::row_major>,
                                              raft::host_matrix<MathT, uint32_t, raft::row_major>>;

template <typename IdxT, typename Accessor>
using vpq_data_matrix = std::conditional_t<Accessor::is_device_accessible,
                                           raft::device_matrix<uint8_t, IdxT, raft::row_major>,
                                           raft::host_matrix<uint8_t, IdxT, raft::row_major>>;

// -----------------------------------------------------------------------------
// empty
// -----------------------------------------------------------------------------

template <typename IdxT>
struct empty_dataset_storage {
  uint32_t suggested_dim{};
  empty_dataset_storage() noexcept = default;
  explicit empty_dataset_storage(uint32_t dim) noexcept : suggested_dim(dim) {}
  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return 0; }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return suggested_dim; }
};

template <typename IdxT>
using empty_dataset_owning_storage = empty_dataset_storage<IdxT>;

template <typename IdxT>
using empty_dataset_view_storage = empty_dataset_storage<IdxT>;

// -----------------------------------------------------------------------------
// dense row-major (logical dim may differ from row pitch; shared by padded & standard)
// -----------------------------------------------------------------------------

/**
 * Dense row-major owning storage shared by padded and standard dataset containers.
 *
 * Template parameters:
 * - MatrixT: owning matrix type that stores the payload (host/device matrix).
 * - ViewT: non-owning row-major view type returned by `view()`.
 * - DataT: scalar element type of the dataset payload.
 * - IdxT: index type used for row counts (`n_rows()` return type).
 */
template <typename MatrixT, typename ViewT, typename DataT, typename IdxT>
struct dense_row_major_dataset_owning_storage {
  MatrixT data_;
  uint32_t logical_dim_;

  dense_row_major_dataset_owning_storage(MatrixT&& data, uint32_t logical_dim) noexcept
    : data_{std::move(data)}, logical_dim_{logical_dim}
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return data_.extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return logical_dim_; }
  [[nodiscard]] auto stride() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(data_.extent(1));
  }
  [[nodiscard]] auto view() const noexcept -> ViewT { return data_.view(); }
  [[nodiscard]] auto data_handle() noexcept -> DataT* { return data_.data_handle(); }
  [[nodiscard]] auto data_handle() const noexcept -> const DataT* { return data_.data_handle(); }
};

template <typename ViewT, typename DataT, typename IdxT>
struct dense_row_major_dataset_view_storage {
  ViewT data_;
  uint32_t logical_dim_;

  dense_row_major_dataset_view_storage() noexcept = default;

  explicit dense_row_major_dataset_view_storage(ViewT v) noexcept
    : data_(v), logical_dim_(static_cast<uint32_t>(v.extent(1)))
  {
  }

  dense_row_major_dataset_view_storage(ViewT v, uint32_t logical_dim) noexcept
    : data_(v), logical_dim_(logical_dim)
  {
  }

  dense_row_major_dataset_view_storage(dense_row_major_dataset_view_storage const& other) noexcept
    : data_(other.data_), logical_dim_(other.logical_dim_)
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return data_.extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return logical_dim_; }
  [[nodiscard]] auto stride() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(data_.stride(0) > 0 ? data_.stride(0) : data_.extent(1));
  }
  [[nodiscard]] auto view() const noexcept -> ViewT { return data_; }
};

template <typename MatrixT, typename ViewT, typename DataT, typename IdxT>
using padded_dataset_owning_storage =
  dense_row_major_dataset_owning_storage<MatrixT, ViewT, DataT, IdxT>;

template <typename ViewT, typename DataT, typename IdxT>
using padded_dataset_view_storage = dense_row_major_dataset_view_storage<ViewT, DataT, IdxT>;

template <typename MatrixT, typename ViewT, typename DataT, typename IdxT>
using standard_dataset_owning_storage =
  dense_row_major_dataset_owning_storage<MatrixT, ViewT, DataT, IdxT>;

template <typename ViewT, typename DataT, typename IdxT>
using standard_dataset_view_storage = dense_row_major_dataset_view_storage<ViewT, DataT, IdxT>;

// -----------------------------------------------------------------------------
// VPQ compressed
// -----------------------------------------------------------------------------

/**
 * Owning storage for VPQ-compressed datasets.
 *
 * Template parameters:
 * - VqBookMatrixT: owning matrix type for the VQ codebook.
 * - PqBookMatrixT: owning matrix type for the PQ codebook.
 * - DataMatrixT: owning matrix type for encoded row data (uint8 codes).
 * - MathT: floating-point type used by VQ/PQ codebooks.
 * - IdxT: index type used for row counts (`n_rows()` return type).
 */
template <typename VqBookMatrixT,
          typename PqBookMatrixT,
          typename DataMatrixT,
          typename MathT,
          typename IdxT>
struct vpq_dataset_owning_storage {
  /** Floating-point type used for VQ/PQ codebooks (rows are still uint8 codes). */
  using math_type = MathT;

  VqBookMatrixT vq_code_book;
  PqBookMatrixT pq_code_book;
  DataMatrixT data;

  vpq_dataset_owning_storage(VqBookMatrixT&& vq_code_book,
                             PqBookMatrixT&& pq_code_book,
                             DataMatrixT&& data) noexcept
    : vq_code_book{std::move(vq_code_book)},
      pq_code_book{std::move(pq_code_book)},
      data{std::move(data)}
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return data.extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return vq_code_book.extent(1); }

  [[nodiscard]] constexpr inline auto encoded_row_length() const noexcept -> uint32_t
  {
    return data.extent(1);
  }
  [[nodiscard]] constexpr inline auto vq_n_centers() const noexcept -> uint32_t
  {
    return vq_code_book.extent(0);
  }
  [[nodiscard]] constexpr inline auto pq_bits() const noexcept -> uint32_t
  {
    auto pq_width = pq_n_centers();
#ifdef __cpp_lib_bitops
    return std::countr_zero(pq_width);
#else
    uint32_t pq_bits = 0;
    while (pq_width > 1) {
      pq_bits++;
      pq_width >>= 1;
    }
    return pq_bits;
#endif
  }
  [[nodiscard]] constexpr inline auto pq_dim() const noexcept -> uint32_t
  {
    return raft::div_rounding_up_unsafe(dim(), pq_len());
  }
  [[nodiscard]] constexpr inline auto pq_len() const noexcept -> uint32_t
  {
    return pq_code_book.extent(1);
  }
  [[nodiscard]] constexpr inline auto pq_n_centers() const noexcept -> uint32_t
  {
    return pq_code_book.extent(0);
  }
};

template <typename ContainerType, typename DataT, typename IdxT, typename Accessor>
struct vpq_dataset_view_storage {
  using owning_dataset_type =
    dataset<ContainerType, DataT, IdxT, dataset_owning_accessor_for_view<DataT, Accessor>>;

  owning_dataset_type const* dataset_{nullptr};

  vpq_dataset_view_storage() = default;

  explicit vpq_dataset_view_storage(owning_dataset_type const* ptr) : dataset_(ptr)
  {
    RAFT_EXPECTS(ptr != nullptr, "vpq_dataset_view: null dataset pointer");
  }

  [[nodiscard]] auto n_rows() const noexcept
  {
    using idx_type = decltype(std::declval<owning_dataset_type const&>().n_rows());
    return dataset_ != nullptr ? dataset_->n_rows() : idx_type{0};
  }
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return dataset_ != nullptr ? dataset_->dim() : uint32_t{0};
  }
  [[nodiscard]] owning_dataset_type const& dset() const noexcept { return *dataset_; }
};

}  // namespace detail

// -----------------------------------------------------------------------------
// empty
// -----------------------------------------------------------------------------

struct empty_dataset_container {
  template <typename IdxT, typename Accessor>
  using owning_storage = detail::empty_dataset_owning_storage<IdxT>;
  template <typename IdxT, typename Accessor>
  using view_storage = detail::empty_dataset_view_storage<IdxT>;
};

// -----------------------------------------------------------------------------
// padded (row-major with logical dim vs stride)
// -----------------------------------------------------------------------------

struct padded_dataset_container {
  template <typename DataT, typename IdxT, typename Accessor>
  using owning_storage =
    detail::padded_dataset_owning_storage<detail::dense_owning_matrix<DataT, IdxT, Accessor>,
                                          detail::dense_view_matrix<DataT, IdxT, Accessor>,
                                          DataT,
                                          IdxT>;
  template <typename DataT, typename IdxT, typename Accessor>
  using view_storage = detail::
    padded_dataset_view_storage<detail::dense_view_matrix<DataT, IdxT, Accessor>, DataT, IdxT>;
};

// -----------------------------------------------------------------------------
// standard (row-major with arbitrary stride; no CAGRA alignment requirement)
// -----------------------------------------------------------------------------

struct standard_dataset_container {
  template <typename DataT, typename IdxT, typename Accessor>
  using owning_storage =
    detail::standard_dataset_owning_storage<detail::dense_owning_matrix<DataT, IdxT, Accessor>,
                                            detail::dense_view_matrix<DataT, IdxT, Accessor>,
                                            DataT,
                                            IdxT>;
  template <typename DataT, typename IdxT, typename Accessor>
  using view_storage = detail::
    standard_dataset_view_storage<detail::dense_view_matrix<DataT, IdxT, Accessor>, DataT, IdxT>;
};

// -----------------------------------------------------------------------------
// VPQ compressed
// -----------------------------------------------------------------------------

struct vpq_dataset_container {
  template <typename MathT, typename IdxT, typename Accessor>
  using owning_storage =
    detail::vpq_dataset_owning_storage<detail::vpq_vq_book_matrix<MathT, IdxT, Accessor>,
                                       detail::vpq_vq_book_matrix<MathT, IdxT, Accessor>,
                                       detail::vpq_data_matrix<IdxT, Accessor>,
                                       MathT,
                                       IdxT>;
  template <typename MathT, typename IdxT, typename Accessor>
  using view_storage =
    detail::vpq_dataset_view_storage<vpq_dataset_container, MathT, IdxT, Accessor>;
};

template <typename ContainerType, typename DataT, typename IdxT, typename Accessor>
struct dataset {
  static_assert(!std::is_same_v<ContainerType, ContainerType>,
                "dataset: unsupported ContainerType / type-parameter combination");
};

template <typename ContainerType, typename DataT, typename IdxT, typename Accessor>
struct dataset_view {
  static_assert(!std::is_same_v<ContainerType, ContainerType>,
                "dataset_view: unsupported ContainerType / type-parameter combination");
};

// -----------------------------------------------------------------------------
// empty
// -----------------------------------------------------------------------------

template <typename IdxT, typename Accessor>
struct dataset<empty_dataset_container, void, IdxT, Accessor>
  : empty_dataset_container::template owning_storage<IdxT, Accessor> {
  using container_type      = empty_dataset_container;
  using owning_storage_type = typename container_type::template owning_storage<IdxT, Accessor>;
  using owning_storage_type::owning_storage_type;

  [[nodiscard]] auto as_dataset_view() const noexcept
    -> dataset_view<empty_dataset_container,
                    void,
                    IdxT,
                    detail::dataset_view_accessor_for_owning<char, Accessor>>
  {
    return dataset_view<empty_dataset_container,
                        void,
                        IdxT,
                        detail::dataset_view_accessor_for_owning<char, Accessor>>{this->dim()};
  }
};

template <typename IdxT, typename Accessor>
struct dataset_view<empty_dataset_container, void, IdxT, Accessor>
  : empty_dataset_container::template view_storage<IdxT, Accessor> {
  using container_type    = empty_dataset_container;
  using view_storage_type = typename container_type::template view_storage<IdxT, Accessor>;
  using view_storage_type::view_storage_type;
};

// -----------------------------------------------------------------------------
// standard (row-major with arbitrary stride)
// -----------------------------------------------------------------------------

template <typename DataT, typename IdxT, typename Accessor>
struct dataset<standard_dataset_container, DataT, IdxT, Accessor>
  : standard_dataset_container::template owning_storage<DataT, IdxT, Accessor> {
  using container_type = standard_dataset_container;
  using owning_storage_type =
    typename container_type::template owning_storage<DataT, IdxT, Accessor>;
  using owning_storage_type::owning_storage_type;

  [[nodiscard]] auto as_dataset_view() const noexcept
    -> dataset_view<standard_dataset_container,
                    DataT,
                    IdxT,
                    detail::dataset_view_accessor_for_owning<DataT, Accessor>>
  {
    return dataset_view<standard_dataset_container,
                        DataT,
                        IdxT,
                        detail::dataset_view_accessor_for_owning<DataT, Accessor>>(this->view(),
                                                                                   this->dim());
  }
};

template <typename DataT, typename IdxT, typename Accessor>
struct dataset_view<standard_dataset_container, DataT, IdxT, Accessor>
  : standard_dataset_container::template view_storage<DataT, IdxT, Accessor> {
  using container_type    = standard_dataset_container;
  using view_storage_type = typename container_type::template view_storage<DataT, IdxT, Accessor>;
  using view_storage_type::view_storage_type;
};

// -----------------------------------------------------------------------------
// padded (row-major with logical dim vs stride)
// -----------------------------------------------------------------------------

template <typename DataT, typename IdxT, typename Accessor>
struct dataset<padded_dataset_container, DataT, IdxT, Accessor>
  : padded_dataset_container::template owning_storage<DataT, IdxT, Accessor> {
  using container_type = padded_dataset_container;
  using owning_storage_type =
    typename container_type::template owning_storage<DataT, IdxT, Accessor>;
  using owning_storage_type::owning_storage_type;

  [[nodiscard]] auto as_dataset_view() const noexcept
    -> dataset_view<padded_dataset_container,
                    DataT,
                    IdxT,
                    detail::dataset_view_accessor_for_owning<DataT, Accessor>>
  {
    return dataset_view<padded_dataset_container,
                        DataT,
                        IdxT,
                        detail::dataset_view_accessor_for_owning<DataT, Accessor>>(this->view(),
                                                                                   this->dim());
  }
};

template <typename DataT, typename IdxT, typename Accessor>
struct dataset_view<padded_dataset_container, DataT, IdxT, Accessor>
  : padded_dataset_container::template view_storage<DataT, IdxT, Accessor> {
  using container_type    = padded_dataset_container;
  using view_storage_type = typename container_type::template view_storage<DataT, IdxT, Accessor>;
  using view_storage_type::view_storage_type;
};

// -----------------------------------------------------------------------------
// VPQ compressed (view holds non-owning pointer to owning dataset)
// -----------------------------------------------------------------------------

template <typename DataT, typename IdxT, typename Accessor>
struct dataset<vpq_dataset_container, DataT, IdxT, Accessor>
  : vpq_dataset_container::template owning_storage<DataT, IdxT, Accessor> {
  using container_type = vpq_dataset_container;
  using owning_storage_type =
    typename container_type::template owning_storage<DataT, IdxT, Accessor>;
  using owning_storage_type::owning_storage_type;

  [[nodiscard]] auto as_dataset_view() const
    -> dataset_view<vpq_dataset_container,
                    DataT,
                    IdxT,
                    detail::dataset_view_accessor_for_owning<DataT, Accessor>>
  {
    return dataset_view<vpq_dataset_container,
                        DataT,
                        IdxT,
                        detail::dataset_view_accessor_for_owning<DataT, Accessor>>{this};
  }
};

template <typename DataT, typename IdxT, typename Accessor>
struct dataset_view<vpq_dataset_container, DataT, IdxT, Accessor>
  : vpq_dataset_container::template view_storage<DataT, IdxT, Accessor> {
  using container_type    = vpq_dataset_container;
  using view_storage_type = typename container_type::template view_storage<DataT, IdxT, Accessor>;
  using view_storage_type::view_storage_type;
};

/**
 * @brief Aliases for concrete `dataset` / `dataset_view` layouts.
 */
template <typename IdxT>
using device_empty_dataset =
  dataset<empty_dataset_container, void, IdxT, detail::device_owning_accessor<char>>;

template <typename IdxT>
using device_empty_dataset_view =
  dataset_view<empty_dataset_container, void, IdxT, detail::device_view_accessor<char>>;

template <typename IdxT>
using host_empty_dataset =
  dataset<empty_dataset_container, void, IdxT, detail::host_owning_accessor<char>>;

template <typename IdxT>
using host_empty_dataset_view =
  dataset_view<empty_dataset_container, void, IdxT, detail::host_view_accessor<char>>;

template <typename DataT, typename IdxT>
using device_padded_dataset =
  dataset<padded_dataset_container, DataT, IdxT, detail::device_owning_accessor<DataT>>;

template <typename DataT, typename IdxT>
using device_padded_dataset_view =
  dataset_view<padded_dataset_container, DataT, IdxT, detail::device_view_accessor<DataT>>;

template <typename DataT, typename IdxT>
using host_padded_dataset =
  dataset<padded_dataset_container, DataT, IdxT, detail::host_owning_accessor<DataT>>;

template <typename DataT, typename IdxT>
using host_padded_dataset_view =
  dataset_view<padded_dataset_container, DataT, IdxT, detail::host_view_accessor<DataT>>;

template <typename DataT, typename IdxT>
using device_standard_dataset =
  dataset<standard_dataset_container, DataT, IdxT, detail::device_owning_accessor<DataT>>;

template <typename DataT, typename IdxT>
using device_standard_dataset_view =
  dataset_view<standard_dataset_container, DataT, IdxT, detail::device_view_accessor<DataT>>;

template <typename DataT, typename IdxT>
using host_standard_dataset =
  dataset<standard_dataset_container, DataT, IdxT, detail::host_owning_accessor<DataT>>;

template <typename DataT, typename IdxT>
using host_standard_dataset_view =
  dataset_view<standard_dataset_container, DataT, IdxT, detail::host_view_accessor<DataT>>;

template <typename DataT, typename IdxT>
using device_vpq_dataset =
  dataset<vpq_dataset_container, DataT, IdxT, detail::device_owning_accessor<DataT>>;

template <typename DataT, typename IdxT>
using device_vpq_dataset_view =
  dataset_view<vpq_dataset_container, DataT, IdxT, detail::device_view_accessor<DataT>>;

template <typename DataT, typename IdxT>
using host_vpq_dataset =
  dataset<vpq_dataset_container, DataT, IdxT, detail::host_owning_accessor<DataT>>;

template <typename DataT, typename IdxT>
using host_vpq_dataset_view =
  dataset_view<vpq_dataset_container, DataT, IdxT, detail::host_view_accessor<DataT>>;

// Maps a dataset view type to its owning (allocating) dataset counterpart.
// Used by serialize/deserialize to type the out_dataset output parameter;
// adding a new dataset type only requires adding a new specialization here.
template <typename DatasetViewT>
struct owning_dataset_for_view;

template <typename DataT, typename IdxT>
struct owning_dataset_for_view<device_padded_dataset_view<DataT, IdxT>> {
  using type = device_padded_dataset<DataT, IdxT>;
};

template <typename DataT, typename IdxT>
struct owning_dataset_for_view<device_standard_dataset_view<DataT, IdxT>> {
  using type = device_standard_dataset<DataT, IdxT>;
};

template <typename DataT, typename IdxT>
struct owning_dataset_for_view<device_vpq_dataset_view<DataT, IdxT>> {
  using type = device_vpq_dataset<DataT, IdxT>;
};

template <typename DatasetViewT>
using owning_dataset_for_view_t = typename owning_dataset_for_view<DatasetViewT>::type;

template <typename DatasetT>
struct is_padded_dataset : std::false_type {};

template <typename DataT, typename IdxT, typename Accessor>
struct is_padded_dataset<dataset<padded_dataset_container, DataT, IdxT, Accessor>>
  : std::true_type {};

template <typename DataT, typename IdxT, typename Accessor>
struct is_padded_dataset<dataset_view<padded_dataset_container, DataT, IdxT, Accessor>>
  : std::true_type {};

template <typename DatasetT>
inline constexpr bool is_padded_dataset_v = is_padded_dataset<DatasetT>::value;

template <typename DatasetT>
struct is_standard_dataset : std::false_type {};

template <typename DataT, typename IdxT, typename Accessor>
struct is_standard_dataset<dataset<standard_dataset_container, DataT, IdxT, Accessor>>
  : std::true_type {};

template <typename DataT, typename IdxT, typename Accessor>
struct is_standard_dataset<dataset_view<standard_dataset_container, DataT, IdxT, Accessor>>
  : std::true_type {};

template <typename DatasetT>
inline constexpr bool is_standard_dataset_v = is_standard_dataset<DatasetT>::value;

template <typename DatasetT>
struct is_vpq_dataset : std::false_type {};

template <typename DataT, typename IdxT, typename Accessor>
struct is_vpq_dataset<dataset<vpq_dataset_container, DataT, IdxT, Accessor>> : std::true_type {};

template <typename DatasetT>
inline constexpr bool is_vpq_dataset_v = is_vpq_dataset<DatasetT>::value;

// -----------------------------------------------------------------------------
// Dataset view compile-time classification (replaces runtime std::variant dispatch).
// -----------------------------------------------------------------------------

/** Any non-owning dataset view exposing row count and logical dimension. */
template <typename V, typename IdxT = int64_t>
concept cagra_dataset_view = requires(V const& v) {
  { v.n_rows() } -> std::convertible_to<IdxT>;
  { v.dim() } -> std::convertible_to<uint32_t>;
};

enum class dataset_view_kind {
  // TODO(removal): Remove `unknown` once all deprecated host_matrix_view / device_matrix_view /
  // mdspan overloads are deleted. It exists solely so that overload resolution on the deprecated
  // build(host_matrix_view) / build(device_matrix_view) shims does not cause a hard error when
  // the compiler evaluates is_host/device_dataset_view_v for a plain mdspan type.
  unknown,
  empty,
  padded,
  standard,
  vpq_f16,
  vpq_f32,
};

/** Primary template returns `unknown` so traits safely return `false` for non-dataset-view types.
 */
template <typename V>
struct dataset_view_kind_of {
  static constexpr dataset_view_kind value = dataset_view_kind::unknown;
};

template <typename IdxT, typename Accessor>
struct dataset_view_kind_of<dataset_view<empty_dataset_container, void, IdxT, Accessor>> {
  static constexpr dataset_view_kind value = dataset_view_kind::empty;
};

template <typename DataT, typename IdxT, typename Accessor>
struct dataset_view_kind_of<dataset_view<padded_dataset_container, DataT, IdxT, Accessor>> {
  static constexpr dataset_view_kind value = dataset_view_kind::padded;
};

template <typename DataT, typename IdxT, typename Accessor>
struct dataset_view_kind_of<dataset_view<standard_dataset_container, DataT, IdxT, Accessor>> {
  static constexpr dataset_view_kind value = dataset_view_kind::standard;
};

template <typename MathT, typename IdxT, typename Accessor>
struct dataset_view_kind_of<dataset_view<vpq_dataset_container, MathT, IdxT, Accessor>> {
  static_assert(std::is_same_v<MathT, half> || std::is_same_v<MathT, float>,
                "VPQ dataset_view_kind_of expects MathT to be half or float");
  static constexpr dataset_view_kind value =
    std::is_same_v<MathT, half> ? dataset_view_kind::vpq_f16 : dataset_view_kind::vpq_f32;
};

template <typename V>
using dataset_view_type_t = std::remove_cvref_t<V>;

/** True when the dataset view accessor is device-accessible. */
template <typename V>
struct dataset_view_is_device_accessible : std::false_type {};

template <typename ContainerType, typename DataT, typename IdxT, typename Accessor>
struct dataset_view_is_device_accessible<dataset_view<ContainerType, DataT, IdxT, Accessor>>
  : std::bool_constant<Accessor::is_device_accessible> {};

template <typename V>
inline constexpr bool dataset_view_is_device_accessible_v =
  dataset_view_is_device_accessible<dataset_view_type_t<V>>::value;

template <typename V>
inline constexpr dataset_view_kind dataset_view_kind_v =
  dataset_view_kind_of<dataset_view_type_t<V>>::value;

template <typename V>
inline constexpr bool is_device_empty_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::empty && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_empty_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::empty && !dataset_view_is_device_accessible_v<V>;

/** True for any empty dataset view (device or host). */
template <typename V>
inline constexpr bool is_empty_dataset_view_v =
  is_device_empty_dataset_view_v<V> || is_host_empty_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_padded_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::padded && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_padded_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::padded && !dataset_view_is_device_accessible_v<V>;

/** True for either `device_padded_dataset_view` or `host_padded_dataset_view`. */
template <typename V>
inline constexpr bool is_padded_dataset_view_v =
  is_device_padded_dataset_view_v<V> || is_host_padded_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_standard_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::standard && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_standard_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::standard && !dataset_view_is_device_accessible_v<V>;

/** True for either `device_standard_dataset_view` or `host_standard_dataset_view`. */
template <typename V>
inline constexpr bool is_standard_dataset_view_v =
  is_device_standard_dataset_view_v<V> || is_host_standard_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_vpq_f16_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::vpq_f16 && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_vpq_f16_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::vpq_f16 && !dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_vpq_f16_dataset_view_v =
  is_device_vpq_f16_dataset_view_v<V> || is_host_vpq_f16_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_vpq_f32_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::vpq_f32 && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_vpq_f32_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::vpq_f32 && !dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_vpq_f32_dataset_view_v =
  is_device_vpq_f32_dataset_view_v<V> || is_host_vpq_f32_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_vpq_dataset_view_v =
  is_device_vpq_f16_dataset_view_v<V> || is_device_vpq_f32_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_host_vpq_dataset_view_v =
  is_host_vpq_f16_dataset_view_v<V> || is_host_vpq_f32_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_vpq_dataset_view_v =
  is_device_vpq_dataset_view_v<V> || is_host_vpq_dataset_view_v<V>;

/** True for any device-resident dataset view. */
template <typename V>
inline constexpr bool is_device_dataset_view_v =
  dataset_view_kind_v<V> != dataset_view_kind::unknown && dataset_view_is_device_accessible_v<V>;

/** True for any host-resident dataset view. */
template <typename V>
inline constexpr bool is_host_dataset_view_v =
  dataset_view_kind_v<V> != dataset_view_kind::unknown && !dataset_view_is_device_accessible_v<V>;

/**
 * True when a host view `H` and device view `D` represent the same storage kind and differ
 * only in residency (host vs. device). Used to constrain `attach_device_dataset_on_host_index`.
 */
template <typename HostViewT, typename DeviceViewT>
inline constexpr bool compatible_host_device_dataset_views_v =
  is_host_dataset_view_v<HostViewT> && is_device_dataset_view_v<DeviceViewT> &&
  (dataset_view_kind_v<HostViewT> == dataset_view_kind_v<DeviceViewT>);

/**
 * Generic accessor retargeting while preserving the dataset tag/layout and value/index types:
 * `dataset<Tag, DataT, IdxT, OldAccessor>      -> dataset<Tag, DataT, IdxT, NewAccessor>`
 * `dataset_view<Tag, DataT, IdxT, OldAccessor> -> dataset_view<Tag, DataT, IdxT, NewAccessor>`
 */
template <typename DatasetLikeT, typename NewAccessor>
struct with_accessor;

template <typename ContainerType,
          typename DataT,
          typename IdxT,
          typename OldAccessor,
          typename NewAccessor>
struct with_accessor<dataset<ContainerType, DataT, IdxT, OldAccessor>, NewAccessor> {
  using type = dataset<ContainerType, DataT, IdxT, NewAccessor>;
};

template <typename ContainerType,
          typename DataT,
          typename IdxT,
          typename OldAccessor,
          typename NewAccessor>
struct with_accessor<dataset_view<ContainerType, DataT, IdxT, OldAccessor>, NewAccessor> {
  using type = dataset_view<ContainerType, DataT, IdxT, NewAccessor>;
};

template <typename DatasetLikeT, typename NewAccessor>
using with_accessor_t =
  typename with_accessor<dataset_view_type_t<DatasetLikeT>, NewAccessor>::type;

/** Map any host accessor to its device counterpart (same payload policy). */
template <typename Accessor>
struct to_device_accessor {
  using type = Accessor;
};

template <typename T>
struct to_device_accessor<detail::host_view_accessor<T>> {
  using type = detail::device_view_accessor<T>;
};

template <typename T>
struct to_device_accessor<detail::host_owning_accessor<T>> {
  using type = detail::device_owning_accessor<T>;
};

template <typename Accessor>
using to_device_accessor_t = typename to_device_accessor<Accessor>::type;

/** Maps a host dataset view type to its device-resident counterpart. */
template <typename HostViewT>
struct device_counterpart;

template <typename ContainerType, typename DataT, typename IdxT, typename Accessor>
struct device_counterpart<dataset_view<ContainerType, DataT, IdxT, Accessor>> {
  using type = with_accessor_t<dataset_view<ContainerType, DataT, IdxT, Accessor>,
                               to_device_accessor_t<Accessor>>;
};

template <typename HostViewT>
using device_counterpart_t = typename device_counterpart<dataset_view_type_t<HostViewT>>::type;

/** True for device padded or standard views accepted by dense graph build (VPQ excluded). */
template <typename V>
inline constexpr bool is_dense_row_major_device_dataset_view_v =
  is_device_padded_dataset_view_v<V> || is_device_standard_dataset_view_v<V>;

/** True for host or device padded/standard views (iterative graph build; VPQ excluded). */
template <typename V>
inline constexpr bool is_dense_row_major_dataset_view_v =
  is_padded_dataset_view_v<V> || is_standard_dataset_view_v<V>;

/** Element type `T` for `cagra::build(res, params, dataset_view)` (deduced, not a template arg). */
template <typename V, typename = void>
struct cagra_view_element_type;

template <typename DataT, typename IdxT>
struct cagra_view_element_type<device_padded_dataset_view<DataT, IdxT>> {
  using type = DataT;
};

template <typename DataT, typename IdxT>
struct cagra_view_element_type<host_padded_dataset_view<DataT, IdxT>> {
  using type = DataT;
};

template <typename DataT, typename IdxT>
struct cagra_view_element_type<device_standard_dataset_view<DataT, IdxT>> {
  using type = DataT;
};

template <typename DataT, typename IdxT>
struct cagra_view_element_type<host_standard_dataset_view<DataT, IdxT>> {
  using type = DataT;
};

template <typename MathT, typename IdxT>
struct cagra_view_element_type<device_vpq_dataset_view<MathT, IdxT>> {
  using type = MathT;
};

template <typename V>
using cagra_view_element_type_t = typename cagra_view_element_type<dataset_view_type_t<V>>::type;

// -----------------------------------------------------------------------------
// CAGRA row width in elements (same for make_device_padded_dataset* and index layout checks).
// -----------------------------------------------------------------------------

/**
 * @brief Required row width in elements for CAGRA: minimum leading dimension (LDA) per row for the
 *        default per-row byte alignment (16 bytes, combined with `sizeof` element type), given
 *        `logical_columns` feature columns.
 */
[[nodiscard]] inline uint32_t cagra_required_row_width(uint32_t logical_columns,
                                                       std::size_t sizeof_value,
                                                       uint32_t align_bytes = 16)
{
  return static_cast<uint32_t>(
    raft::round_up_safe<std::size_t>(static_cast<std::size_t>(logical_columns) * sizeof_value,
                                     std::lcm(align_bytes, static_cast<uint32_t>(sizeof_value))) /
    sizeof_value);
}

template <typename ValueT>
[[nodiscard]] inline uint32_t cagra_required_row_width(uint32_t logical_columns,
                                                       uint32_t align_bytes = 16)
{
  return cagra_required_row_width(logical_columns, sizeof(ValueT), align_bytes);
}

/** Actual row width in elements (leading dimension) of a 2D row-major matrix view. */
template <typename T, typename I, typename L>
[[nodiscard]] inline uint32_t matrix_actual_row_width(raft::device_matrix_view<T, I, L> m)
{
  return m.stride(0) > 0 ? static_cast<uint32_t>(m.stride(0)) : static_cast<uint32_t>(m.extent(1));
}

template <typename T, typename I, typename L>
[[nodiscard]] inline uint32_t matrix_actual_row_width(raft::host_matrix_view<T, I, L> m)
{
  return m.stride(0) > 0 ? static_cast<uint32_t>(m.stride(0)) : static_cast<uint32_t>(m.extent(1));
}

/**
 * @brief True if the matrix's row width in elements matches `cagra_required_row_width` for
 *        `m.extent(1)` and element type `T` (CAGRA row layout is satisfied for this view).
 */
template <typename T, typename I, typename L>
[[nodiscard]] inline bool matrix_row_width_matches_cagra_required(
  raft::device_matrix_view<T, I, L> m, uint32_t align_bytes = 16)
{
  using value_type = std::remove_const_t<T>;
  const uint32_t need =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(m.extent(1)), align_bytes);
  return matrix_actual_row_width(m) == need;
}

template <typename T, typename I, typename L>
[[nodiscard]] inline bool matrix_row_width_matches_cagra_required(raft::host_matrix_view<T, I, L> m,
                                                                  uint32_t align_bytes = 16)
{
  using value_type = std::remove_const_t<T>;
  const uint32_t need =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(m.extent(1)), align_bytes);
  return matrix_actual_row_width(m) == need;
}

namespace detail {

template <typename SrcT>
[[nodiscard]] inline uint32_t mdspan_row_stride_elements(SrcT const& src)
{
  return src.stride(0) > 0 ? static_cast<uint32_t>(src.stride(0))
                           : static_cast<uint32_t>(src.extent(1));
}

template <typename ValueT, typename SrcT>
[[nodiscard]] inline ValueT* expect_device_accessible_data_handle(SrcT const& src,
                                                                  char const* error_msg)
{
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  auto* device_ptr = reinterpret_cast<ValueT*>(ptr_attrs.devicePointer);
  RAFT_EXPECTS(device_ptr != nullptr, error_msg);
  return device_ptr;
}

template <typename ValueT, typename IndexT, typename ViewT, typename SrcT>
[[nodiscard]] inline ViewT make_device_dense_row_major_view_from_src(SrcT const& src,
                                                                     uint32_t logical_dim)
{
  auto* device_ptr = expect_device_accessible_data_handle<ValueT>(
    src, "make_device_*_dataset_view: source must be device-accessible.");
  auto v = raft::make_device_matrix_view(
    device_ptr, src.extent(0), static_cast<IndexT>(mdspan_row_stride_elements(src)));
  return ViewT(v, logical_dim);
}

template <typename ValueT, typename IndexT, typename ViewT, typename SrcT>
[[nodiscard]] inline ViewT make_host_dense_row_major_view_from_src(SrcT const& src,
                                                                   uint32_t logical_dim)
{
  RAFT_EXPECTS(raft::get_device_for_address(src.data_handle()) == -1,
               "make_host_*_dataset_view: source must be host-accessible.");
  auto v = raft::make_host_matrix_view(const_cast<ValueT*>(src.data_handle()),
                                       src.extent(0),
                                       static_cast<IndexT>(mdspan_row_stride_elements(src)));
  return ViewT(v, logical_dim);
}

template <typename DatasetT, typename ValueT, typename IndexT, typename SrcT>
auto make_device_dense_row_major_dataset_from_src(raft::resources const& res,
                                                  SrcT const& src,
                                                  uint32_t logical_dim,
                                                  uint32_t target_stride,
                                                  char const* view_factory_name)
  -> std::unique_ptr<DatasetT>
{
  uint32_t const src_stride = mdspan_row_stride_elements(src);
  RAFT_EXPECTS(logical_dim <= target_stride,
               "logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(logical_dim),
               static_cast<unsigned>(target_stride));
  RAFT_EXPECTS(static_cast<uint32_t>(src.extent(1)) <= target_stride,
               "Source row length must not exceed required stride.");
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  bool const device_src =
    (ptr_attrs.type == cudaMemoryTypeDevice) || (ptr_attrs.type == cudaMemoryTypeManaged);
  if (device_src && src_stride == target_stride) {
    RAFT_EXPECTS(false,
                 "source is device and stride is already correct. "
                 "Use %s() to get a view instead.",
                 view_factory_name);
  }
  auto out_array = raft::make_device_matrix<ValueT, IndexT>(res, src.extent(0), target_stride);
  RAFT_CUDA_TRY(cudaMemsetAsync(out_array.data_handle(),
                                0,
                                out_array.size() * sizeof(ValueT),
                                raft::resource::get_cuda_stream(res)));
  raft::copy_matrix(out_array.data_handle(),
                    target_stride,
                    src.data_handle(),
                    src_stride,
                    logical_dim,
                    src.extent(0),
                    raft::resource::get_cuda_stream(res));
  return std::make_unique<DatasetT>(std::move(out_array), logical_dim);
}

template <typename DatasetT, typename ValueT, typename IndexT, typename SrcT>
auto make_host_dense_row_major_dataset_from_src(raft::resources const& res,
                                                SrcT const& src,
                                                uint32_t logical_dim,
                                                uint32_t target_stride,
                                                char const* device_factory_name,
                                                char const* view_factory_name)
  -> std::unique_ptr<DatasetT>
{
  uint32_t const src_stride = mdspan_row_stride_elements(src);
  RAFT_EXPECTS(raft::get_device_for_address(src.data_handle()) == -1,
               "source must be host-accessible. Use %s() for device sources.",
               device_factory_name);
  RAFT_EXPECTS(logical_dim <= target_stride,
               "logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(logical_dim),
               static_cast<unsigned>(target_stride));
  if (src_stride == target_stride) {
    RAFT_EXPECTS(false,
                 "source stride is already correct. Use %s() to get a view instead.",
                 view_factory_name);
  }
  RAFT_EXPECTS(static_cast<uint32_t>(src.extent(1)) <= target_stride,
               "Source row length must not exceed required stride.");
  auto out_array = raft::make_host_matrix<ValueT, IndexT>(src.extent(0), target_stride);
  std::memset(out_array.data_handle(), 0, out_array.size() * sizeof(ValueT));
  raft::copy_matrix(out_array.data_handle(),
                    target_stride,
                    src.data_handle(),
                    src_stride,
                    logical_dim,
                    src.extent(0),
                    raft::resource::get_cuda_stream(res));
  return std::make_unique<DatasetT>(std::move(out_array), logical_dim);
}

}  // namespace detail

template <typename SrcT>
auto make_device_padded_dataset_view(const raft::resources& res,
                                     SrcT const& src,
                                     uint32_t align_bytes = 16)
  -> device_padded_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  uint32_t required_stride =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(src.extent(1)), align_bytes);
  RAFT_EXPECTS(
    detail::mdspan_row_stride_elements(src) == required_stride,
    "make_device_padded_dataset_view: stride is incorrect (required stride for alignment). "
    "Use make_device_padded_dataset() to get an owning padded copy.");
  return detail::make_device_dense_row_major_view_from_src<
    value_type,
    index_type,
    device_padded_dataset_view<value_type, index_type>>(src, static_cast<uint32_t>(src.extent(1)));
}

template <typename SrcT>
auto make_device_padded_dataset(const raft::resources& res,
                                SrcT const& src,
                                uint32_t align_bytes = 16)
  -> std::unique_ptr<device_padded_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type               = typename SrcT::value_type;
  using index_type               = typename SrcT::index_type;
  uint32_t const logical_dim     = static_cast<uint32_t>(src.extent(1));
  uint32_t const required_stride = cagra_required_row_width<value_type>(logical_dim, align_bytes);
  return detail::make_device_dense_row_major_dataset_from_src<
    device_padded_dataset<value_type, index_type>,
    value_type,
    index_type>(res, src, logical_dim, required_stride, "make_device_padded_dataset_view");
}

template <typename SrcT>
auto make_host_padded_dataset_view(SrcT const& src, uint32_t align_bytes = 16)
  -> host_padded_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  uint32_t required_stride =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(src.extent(1)), align_bytes);
  RAFT_EXPECTS(
    detail::mdspan_row_stride_elements(src) == required_stride,
    "make_host_padded_dataset_view: stride is incorrect (required stride for alignment). "
    "Use make_host_padded_dataset() to get an owning padded copy.");
  return detail::make_host_dense_row_major_view_from_src<
    value_type,
    index_type,
    host_padded_dataset_view<value_type, index_type>>(src, static_cast<uint32_t>(src.extent(1)));
}

template <typename SrcT>
auto make_host_padded_dataset(const raft::resources& res,
                              SrcT const& src,
                              uint32_t align_bytes = 16)
  -> std::unique_ptr<host_padded_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type               = typename SrcT::value_type;
  using index_type               = typename SrcT::index_type;
  uint32_t const logical_dim     = static_cast<uint32_t>(src.extent(1));
  uint32_t const required_stride = cagra_required_row_width<value_type>(logical_dim, align_bytes);
  return detail::make_host_dense_row_major_dataset_from_src<
    host_padded_dataset<value_type, index_type>,
    value_type,
    index_type>(res,
                src,
                logical_dim,
                required_stride,
                "make_device_padded_dataset",
                "make_host_padded_dataset_view");
}

template <typename SrcT>
auto make_device_standard_dataset_view(SrcT const& src)
  -> device_standard_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  return detail::make_device_dense_row_major_view_from_src<
    value_type,
    index_type,
    device_standard_dataset_view<value_type, index_type>>(src,
                                                          static_cast<uint32_t>(src.extent(1)));
}

/**
 * @brief Create an owning device standard dataset with explicit row layout.
 *
 * Internal use only: the sole call site today is
 * `cuvs::neighbors::detail::deserialize_standard()` in `dataset_serialize.hpp`, which must pass
 * wire-format `(logical_dim, stride)` because the deserialized host buffer is tight `[n_rows x
 * dim]` while the on-disk stride may be larger. Do not call from user code; prefer
 * `make_device_standard_dataset_view()` when wrapping existing correctly-strided storage.
 *
 * Potential future call sites if an owning copy with explicit stride is needed:
 * - C API dataset upload (mirroring `make_device_padded_dataset` in `c/src/neighbors/cagra.cpp`)
 * - `tiered_index` / composite index paths that materialize standard-layout device storage
 * - Multigpu (MG) index build or merge when rehydrating a strided dataset from host fragments
 */
template <typename SrcT>
auto make_device_standard_dataset(const raft::resources& res,
                                  SrcT const& src,
                                  uint32_t logical_dim,
                                  uint32_t target_stride)
  -> std::unique_ptr<device_standard_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  return detail::make_device_dense_row_major_dataset_from_src<
    device_standard_dataset<value_type, index_type>,
    value_type,
    index_type>(res, src, logical_dim, target_stride, "make_device_standard_dataset_view");
}

template <typename SrcT>
auto make_host_standard_dataset_view(SrcT const& src)
  -> host_standard_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  return detail::make_host_dense_row_major_view_from_src<
    value_type,
    index_type,
    host_standard_dataset_view<value_type, index_type>>(src, static_cast<uint32_t>(src.extent(1)));
}

namespace filtering {

/**
 * @defgroup neighbors_filtering Filtering for ANN Types
 * @{
 */

enum class FilterType { None, Bitmap, Bitset, UDF };

struct base_filter {
  ~base_filter()                             = default;
  virtual FilterType get_filter_type() const = 0;
};

/* A filter that filters nothing. This is the default behavior. */
struct none_sample_filter : public base_filter {
  /** \cond */
  constexpr __forceinline__ _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the current inverted list index
    const uint32_t cluster_ix,
    // the index of the current sample inside the current inverted list
    const uint32_t sample_ix) const;

  constexpr __forceinline__ _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the index of the current sample
    const uint32_t sample_ix) const;
  /** \endcond */
  FilterType get_filter_type() const override { return FilterType::None; }
};

/**
 * @brief Filter used to convert the cluster index and sample index
 * of an IVF search into a sample index. This can be used as an
 * intermediate filter.
 *
 * @tparam index_t Indexing type
 * @tparam filter_t
 */
template <typename index_t, typename filter_t>
struct ivf_to_sample_filter : public base_filter {
  const index_t* const* inds_ptrs_;
  const filter_t next_filter_;

  _RAFT_HOST_DEVICE ivf_to_sample_filter(const index_t* const* inds_ptrs,
                                         const filter_t next_filter);

  /** \cond */
  /** If the original filter takes three arguments, then don't modify the arguments.
   * If the original filter takes two arguments, then we are using `inds_ptr_` to obtain the sample
   * index.
   */
  inline _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the current inverted list index
    const uint32_t cluster_ix,
    // the index of the current sample inside the current inverted list
    const uint32_t sample_ix) const;

  FilterType get_filter_type() const override { return next_filter_.get_filter_type(); }
  /** \endcond */
};

/**
 * @brief Filter an index with a bitmap
 *
 * @tparam bitmap_t Data type of the bitmap
 * @tparam index_t Indexing type
 */
template <typename bitmap_t, typename index_t>
struct bitmap_filter : public base_filter {
  using view_t = cuvs::core::bitmap_view<bitmap_t, index_t>;

  // View of the bitset to use as a filter
  const view_t bitmap_view_;

  bitmap_filter(const view_t bitmap_for_filtering);
  /** \cond */
  inline _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the index of the current sample
    const uint32_t sample_ix) const;
  /** \endcond */

  FilterType get_filter_type() const override { return FilterType::Bitmap; }

  view_t view() const { return bitmap_view_; }

  template <typename csr_matrix_t>
  void to_csr(raft::resources const& handle, csr_matrix_t& csr);
};

/**
 * @brief Filter an index with a bitset
 *
 * @tparam bitset_t Data type of the bitset
 * @tparam index_t Indexing type
 */
template <typename bitset_t, typename index_t>
struct bitset_filter : public base_filter {
  using view_t = cuvs::core::bitset_view<bitset_t, index_t>;

  // View of the bitset to use as a filter
  const view_t bitset_view_;

  /** \cond */
  _RAFT_HOST_DEVICE bitset_filter(const view_t bitset_for_filtering);
  constexpr __forceinline__ _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the index of the current sample
    const uint32_t sample_ix) const;
  /** \endcond */

  FilterType get_filter_type() const override { return FilterType::Bitset; }

  view_t view() const { return bitset_view_; }

  template <typename csr_matrix_t>
  void to_csr(raft::resources const& handle, csr_matrix_t& csr);
};

/**
 * @brief JIT-LTO user-defined filter predicate.
 *
 * The source must define a device function named by @c function_name with signature:
 *
 * @code{.cpp}
 * __device__ bool cuvs_filter_udf(uint32_t query_id, source_index_t source_id, void* filter_data);
 * @endcode
 *
 * Return @c true to allow a source vector to appear in the results and @c false to reject it.
 * @c filter_data is passed through unchanged and must point to device-accessible memory when the
 * UDF dereferences it. CAGRA currently provides @c source_index_t as @c uint32_t in the generated
 * JIT fragment.
 */
struct udf_filter : public base_filter {
  /** CUDA C++ source containing the device predicate. */
  std::string source;
  /** Opaque device-accessible pointer passed to the predicate. */
  void* filter_data = nullptr;
  /** Estimated fraction of rows rejected by the predicate, or negative if unknown. */
  float filtering_rate = -1.0f;
  /** Device function name to call from the generated CAGRA sample filter. */
  std::string function_name = "cuvs_filter_udf";

  udf_filter() = default;

  explicit udf_filter(std::string source,
                      void* filter_data         = nullptr,
                      float filtering_rate      = -1.0f,
                      std::string function_name = "cuvs_filter_udf")
    : source(std::move(source)),
      filter_data(filter_data),
      filtering_rate(filtering_rate),
      function_name(std::move(function_name))
  {
  }

  FilterType get_filter_type() const override { return FilterType::UDF; }
};

/** @} */  // end group neighbors_filtering

/**
 * If the filtering depends on the index of a sample, then the following
 * filter template can be used:
 *
 * template <typename IdxT>
 * struct index_ivf_sample_filter {
 *   using index_type = IdxT;
 *
 *   const index_type* const* inds_ptr = nullptr;
 *
 *   index_ivf_sample_filter() {}
 *   index_ivf_sample_filter(const index_type* const* _inds_ptr)
 *       : inds_ptr{_inds_ptr} {}
 *   index_ivf_sample_filter(const index_ivf_sample_filter&) = default;
 *   index_ivf_sample_filter(index_ivf_sample_filter&&) = default;
 *   index_ivf_sample_filter& operator=(const index_ivf_sample_filter&) = default;
 *   index_ivf_sample_filter& operator=(index_ivf_sample_filter&&) = default;
 *
 *   inline _RAFT_HOST_DEVICE bool operator()(
 *       const uint32_t query_ix,
 *       const uint32_t cluster_ix,
 *       const uint32_t sample_ix) const {
 *     index_type database_idx = inds_ptr[cluster_ix][sample_ix];
 *
 *     // return true or false, depending on the database_idx
 *     return true;
 *   }
 * };
 *
 * Initialize it as:
 *   using filter_type = index_ivf_sample_filter<idx_t>;
 *   filter_type filter(cuvs_ivfpq_index.inds_ptrs().data_handle());
 *
 * Use it as:
 *   cuvs::neighbors::ivf_pq::search_with_filtering<data_t, idx_t, filter_type>(
 *     ...regular parameters here...,
 *     filter
 *   );
 *
 * Another example would be the following filter that greenlights samples according
 * to a contiguous bit mask vector.
 *
 * template <typename IdxT>
 * struct bitmask_ivf_sample_filter {
 *   using index_type = IdxT;
 *
 *   const index_type* const* inds_ptr = nullptr;
 *   const uint64_t* const bit_mask_ptr = nullptr;
 *   const int64_t bit_mask_stride_64 = 0;
 *
 *   bitmask_ivf_sample_filter() {}
 *   bitmask_ivf_sample_filter(
 *       const index_type* const* _inds_ptr,
 *       const uint64_t* const _bit_mask_ptr,
 *       const int64_t _bit_mask_stride_64)
 *       : inds_ptr{_inds_ptr},
 *         bit_mask_ptr{_bit_mask_ptr},
 *         bit_mask_stride_64{_bit_mask_stride_64} {}
 *   bitmask_ivf_sample_filter(const bitmask_ivf_sample_filter&) = default;
 *   bitmask_ivf_sample_filter(bitmask_ivf_sample_filter&&) = default;
 *   bitmask_ivf_sample_filter& operator=(const bitmask_ivf_sample_filter&) = default;
 *   bitmask_ivf_sample_filter& operator=(bitmask_ivf_sample_filter&&) = default;
 *
 *   inline _RAFT_HOST_DEVICE bool operator()(
 *       const uint32_t query_ix,
 *       const uint32_t cluster_ix,
 *       const uint32_t sample_ix) const {
 *     const index_type database_idx = inds_ptr[cluster_ix][sample_ix];
 *     const uint64_t bit_mask_element =
 *         bit_mask_ptr[query_ix * bit_mask_stride_64 + database_idx / 64];
 *     const uint64_t masked_bool =
 *         bit_mask_element & (1ULL << (uint64_t)(database_idx % 64));
 *     const bool is_bit_set = (masked_bool != 0);
 *
 *     return is_bit_set;
 *   }
 * };
 */
}  // namespace filtering

namespace ivf {

/**
 * Default value filled in the `indices` array.
 * One may encounter it trying to access a record within a list that is outside of the
 * `size` bound or whenever the list is allocated but not filled-in yet.
 */
template <typename IdxT>
constexpr static IdxT kInvalidRecord =
  (std::is_signed_v<IdxT> ? IdxT{0} : std::numeric_limits<IdxT>::max()) - 1;

/**
 * Abstract base class for IVF list data.
 * This allows polymorphic access to list data regardless of the underlying layout.
 *
 * @tparam ValueT The data element type (e.g., uint8_t for PQ codes, float for raw vectors)
 * @tparam IdxT The index type for source indices
 * @tparam SizeT The size type
 *
 * TODO: Make this struct internal (tracking issue: https://github.com/rapidsai/cuvs/issues/1726)
 */
template <typename ValueT, typename IdxT, typename SizeT = uint32_t>
struct list_base {
  using value_type = ValueT;
  using index_type = IdxT;
  using size_type  = SizeT;

  virtual ~list_base() = default;

  /** Get the raw data pointer. */
  virtual value_type* data_ptr() noexcept             = 0;
  virtual const value_type* data_ptr() const noexcept = 0;

  /** Get the indices pointer. */
  virtual index_type* indices_ptr() noexcept             = 0;
  virtual const index_type* indices_ptr() const noexcept = 0;

  /** Get the current size (number of records). */
  virtual size_type get_size() const noexcept = 0;

  /** Set the current size (number of records). */
  virtual void set_size(size_type new_size) noexcept = 0;

  /** Get the total size of the data array in bytes. */
  virtual size_t data_byte_size() const noexcept = 0;

  /** Get the capacity (number of indices that can be stored). */
  virtual size_type indices_capacity() const noexcept = 0;
};

/** The data for a single IVF list. */
template <template <typename, typename...> typename SpecT,
          typename SizeT,
          typename... SpecExtraArgs>
struct list : public list_base<typename SpecT<SizeT, SpecExtraArgs...>::value_type,
                               typename SpecT<SizeT, SpecExtraArgs...>::index_type,
                               SizeT> {
  using size_type    = SizeT;
  using spec_type    = SpecT<size_type, SpecExtraArgs...>;
  using value_type   = typename spec_type::value_type;
  using index_type   = typename spec_type::index_type;
  using list_extents = typename spec_type::list_extents;

  /** Possibly encoded data; it's layout is defined by `SpecT`. */
  raft::device_mdarray<value_type, list_extents, raft::row_major> data;
  /** Source indices. */
  raft::device_mdarray<index_type, raft::extent_1d<size_type>, raft::row_major> indices;
  /** The actual size of the content. */
  std::atomic<size_type> size;

  /** Allocate a new list capable of holding at least `n_rows` data records and indices. */
  list(raft::resources const& res, const spec_type& spec, size_type n_rows);

  value_type* data_ptr() noexcept override { return data.data_handle(); }
  const value_type* data_ptr() const noexcept override { return data.data_handle(); }

  index_type* indices_ptr() noexcept override { return indices.data_handle(); }
  const index_type* indices_ptr() const noexcept override { return indices.data_handle(); }

  size_type get_size() const noexcept override { return size.load(); }
  void set_size(size_type new_size) noexcept override { size.store(new_size); }

  size_t data_byte_size() const noexcept override { return data.size() * sizeof(value_type); }
  size_type indices_capacity() const noexcept override { return indices.extent(0); }
};

template <typename ListT, class T = void>
struct enable_if_valid_list {};

template <class T,
          template <typename, typename...> typename SpecT,
          typename SizeT,
          typename... SpecExtraArgs>
struct enable_if_valid_list<list<SpecT, SizeT, SpecExtraArgs...>, T> {
  using type = T;
};

/**
 * Designed after `std::enable_if_t`, this trait is helpful in the instance resolution;
 * plug this in the return type of a function that has an instance of `ivf::list` as
 * a template parameter.
 */
template <typename ListT, class T = void>
using enable_if_valid_list_t = typename enable_if_valid_list<ListT, T>::type;

/**
 * Resize a list by the given id, so that it can contain the given number of records;
 * copy the data if necessary.
 *
 * @note This is an internal function that requires the concrete list type.
 *       For IVF-PQ indexes, prefer using the helper functions in
 *       `cuvs::neighbors::ivf_pq::helpers::resize_list` which handle type casting internally.
 */
template <typename ListT>
CUVS_EXPORT void resize_list(raft::resources const& res,
                             std::shared_ptr<ListT>& orig_list,  // NOLINT
                             const typename ListT::spec_type& spec,
                             typename ListT::size_type new_used_size,
                             typename ListT::size_type old_used_size);

/**
 * Serialize a list to an output stream.
 *
 * @note This function requires the concrete list type (not the base class) because:
 *       1. It needs access to the spec_type to determine the data layout for serialization
 *       2. The serialized format depends on the spec's make_list_extents() method
 *       When calling from code that only has a base class pointer, use std::static_pointer_cast
 *       to obtain the typed pointer first.
 */
template <typename ListT>
enable_if_valid_list_t<ListT> serialize_list(
  const raft::resources& handle,
  std::ostream& os,
  const ListT& ld,
  const typename ListT::spec_type& store_spec,
  std::optional<typename ListT::size_type> size_override = std::nullopt);

template <typename ListT>
enable_if_valid_list_t<ListT> serialize_list(
  const raft::resources& handle,
  std::ostream& os,
  const std::shared_ptr<ListT>& ld,
  const typename ListT::spec_type& store_spec,
  std::optional<typename ListT::size_type> size_override = std::nullopt);

template <typename ListT>
enable_if_valid_list_t<ListT> deserialize_list(const raft::resources& handle,
                                               std::istream& is,
                                               std::shared_ptr<ListT>& ld,
                                               const typename ListT::spec_type& store_spec,
                                               const typename ListT::spec_type& device_spec);
}  // namespace ivf

using namespace raft;

template <typename AnnIndexType, typename T, typename IdxT>
struct iface {
  iface() : cagra_owned_dataset_(nullptr), mutex_(std::make_shared<std::mutex>()) {}

  const IdxT size() const { return index_.value().size(); }

  std::optional<AnnIndexType> index_;
  /** Used by CAGRA when built from host: holds device copy so index dataset view stays valid. */
  std::optional<raft::device_matrix<T, int64_t, raft::row_major>> cagra_build_dataset_;
  /** Used by CAGRA when deserializing an index that contains a dataset; keeps it alive for the
   * view. */
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<T, int64_t>> cagra_owned_dataset_;
  std::shared_ptr<std::mutex> mutex_;
};

template <typename AnnIndexType, typename T, typename IdxT, typename Accessor>
void build(const raft::resources& handle,
           cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
           const cuvs::neighbors::index_params* index_params,
           raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> index_dataset);

template <typename AnnIndexType, typename T, typename IdxT, typename Accessor1, typename Accessor2>
void extend(
  const raft::resources& handle,
  cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
  raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor1> new_vectors,
  std::optional<raft::mdspan<const IdxT, vector_extent<int64_t>, layout_c_contiguous, Accessor2>>
    new_indices);

template <typename AnnIndexType, typename T, typename IdxT, typename searchIdxT>
void search(const raft::resources& handle,
            const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
            const cuvs::neighbors::search_params* search_params,
            raft::device_matrix_view<const T, int64_t, row_major> h_queries,
            raft::device_matrix_view<searchIdxT, int64_t, row_major> d_neighbors,
            raft::device_matrix_view<float, int64_t, row_major> d_distances);

template <typename AnnIndexType, typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
               std::ostream& os);

template <typename AnnIndexType, typename T, typename IdxT>
void deserialize(const raft::resources& handle,
                 cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
                 std::istream& is);

template <typename AnnIndexType, typename T, typename IdxT>
void deserialize(const raft::resources& handle,
                 cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
                 const std::string& filename);

/// \defgroup mg_cpp_index_params ANN MG index build parameters

/** Distribution mode */
/// \ingroup mg_cpp_index_params
enum distribution_mode {
  /** Index is replicated on each device, favors throughput */
  REPLICATED,
  /** Index is split on several devices, favors scaling */
  SHARDED
};

/// \defgroup mg_cpp_search_params ANN MG search parameters

/** Search mode when using a replicated index */
/// \ingroup mg_cpp_search_params
enum replicated_search_mode {
  /** Search queries are split to maintain equal load on GPUs */
  LOAD_BALANCER,
  /** Each search query is processed by a single GPU in a round-robin fashion */
  ROUND_ROBIN
};

/** Merge mode when using a sharded index */
/// \ingroup mg_cpp_search_params
enum sharded_merge_mode {
  /** Search batches are merged on the root rank */
  MERGE_ON_ROOT_RANK,
  /** Search batches are merged in a tree reduction fashion */
  TREE_MERGE
};

/** Build parameters */
/// \ingroup mg_cpp_index_params
template <typename Upstream>
struct mg_index_params : public Upstream {
  mg_index_params() : mode(SHARDED) {}

  mg_index_params(const Upstream& sp) : Upstream(sp), mode(SHARDED) {}

  /** Distribution mode */
  cuvs::neighbors::distribution_mode mode = SHARDED;
};

/** Search parameters */
/// \ingroup mg_cpp_search_params
template <typename Upstream>
struct mg_search_params : public Upstream {
  mg_search_params() : search_mode(LOAD_BALANCER), merge_mode(TREE_MERGE) {}

  mg_search_params(const Upstream& sp)
    : Upstream(sp), search_mode(LOAD_BALANCER), merge_mode(TREE_MERGE)
  {
  }

  /** Replicated search mode */
  cuvs::neighbors::replicated_search_mode search_mode = LOAD_BALANCER;
  /** Sharded merge mode */
  cuvs::neighbors::sharded_merge_mode merge_mode = TREE_MERGE;
  /** Number of rows per batch */
  int64_t n_rows_per_batch = 1 << 20;
};

template <typename AnnIndexType, typename T, typename IdxT>
struct mg_index {
  mg_index(const raft::resources& clique, distribution_mode mode);
  mg_index(const raft::resources& clique, const std::string& filename);

  mg_index(const mg_index&)                    = delete;
  mg_index(mg_index&&)                         = default;
  auto operator=(const mg_index&) -> mg_index& = delete;
  auto operator=(mg_index&&) -> mg_index&      = default;

  distribution_mode mode_;
  int num_ranks_;
  std::vector<iface<AnnIndexType, T, IdxT>> ann_interfaces_;

  // for load balancing mechanism
  std::shared_ptr<std::atomic<int64_t>> round_robin_counter_;
};

}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
