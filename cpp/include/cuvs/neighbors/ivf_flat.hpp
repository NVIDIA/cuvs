/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.hpp"
#include <cstdint>
#include <cuvs/core/export.hpp>
#include <cuvs/neighbors/common.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <sstream>
#include <string>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace ivf_flat {
/**
 * @defgroup ivf_flat_cpp_index_params IVF-Flat index build parameters
 * @{
 */

/** Size of the interleaved group (see `index::data` description). */
constexpr static uint32_t kIndexGroupSize = 32;

struct index_params : cuvs::neighbors::index_params {
  /** The number of inverted lists (clusters) */
  uint32_t n_lists = 1024;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters = 20;
  /** The fraction of data to use during iterative kmeans building. */
  double kmeans_trainset_fraction = 0.5;
  /**
   * By default (adaptive_centers = false), the cluster centers are trained in `ivf_flat::build`,
   * and never modified in `ivf_flat::extend`. As a result, you may need to retrain the index
   * from scratch after invoking (`ivf_flat::extend`) a few times with new data, the distribution of
   * which is no longer representative of the original training set.
   *
   * The alternative behavior (adaptive_centers = true) is to update the cluster centers for new
   * data when it is added. In this case, `index.centers()` are always exactly the centroids of the
   * data in the corresponding clusters. The drawback of this behavior is that the centroids depend
   * on the order of adding new data (through the classification of the added data); that is,
   * `index.centers()` "drift" together with the changing distribution of the newly added data.
   */
  bool adaptive_centers = false;
  /**
   * By default, the algorithm allocates more space than necessary for individual clusters
   * (`list_data`). This allows to amortize the cost of memory allocation and reduce the number of
   * data copies during repeated calls to `extend` (extending the database).
   *
   * The alternative is the conservative allocation behavior; when enabled, the algorithm always
   * allocates the minimum amount of memory required to store the given number of records. Set this
   * flag to `true` if you prefer to use as little GPU memory for the database as possible.
   */
  bool conservative_memory_allocation = false;
  /**
   * Whether to add the dataset content to the index, i.e.:
   *
   *  - `true` means the index is filled with the dataset vectors and ready to search after calling
   * `build`.
   *  - `false` means `build` only trains the underlying model (e.g. quantizer or clustering), but
   * the index is left empty; you'd need to call `extend` on the index afterwards to populate it.
   */
  bool add_data_on_build = true;
};
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_search_params IVF-Flat index search parameters
 * @{
 */
struct search_params : cuvs::neighbors::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
  /** Custom metric UDF code. */
  std::optional<std::string> metric_udf = std::nullopt;
};

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

template <typename SizeT, typename ValueT, typename IdxT>
struct list_spec {
  using value_type   = ValueT;
  using list_extents = raft::matrix_extent<SizeT>;
  using index_type   = IdxT;

  SizeT align_max;
  SizeT align_min;
  uint32_t dim;

  constexpr list_spec(uint32_t dim, bool conservative_memory_allocation)
    : dim(dim),
      align_min(kIndexGroupSize),
      align_max(conservative_memory_allocation ? kIndexGroupSize : 1024)
  {
  }

  // Allow casting between different size-types (for safer size and offset calculations)
  template <typename OtherSizeT>
  constexpr explicit list_spec(const list_spec<OtherSizeT, ValueT, IdxT>& other_spec)
    : dim{other_spec.dim}, align_min{other_spec.align_min}, align_max{other_spec.align_max}
  {
  }

  /** Determine the extents of an array enough to hold a given amount of data. */
  constexpr auto make_list_extents(SizeT n_rows) const -> list_extents
  {
    return raft::make_extents<SizeT>(n_rows, dim);
  }
};

template <typename ValueT, typename IdxT, typename SizeT = uint32_t>
using list_data = ivf::list<list_spec, SizeT, ValueT, IdxT>;

/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_index IVF-Flat index
 * @{
 */
/**
 * @brief IVF-flat index.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 */
template <typename T, typename IdxT>
struct index : cuvs::neighbors::index {
  using index_params_type  = ivf_flat::index_params;
  using search_params_type = ivf_flat::search_params;
  using index_type         = IdxT;
  using value_type         = T;
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  index(const index&)            = delete;
  index(index&&)                 = default;
  index& operator=(const index&) = delete;
  index& operator=(index&&)      = default;
  ~index()                       = default;

  /**
   * @brief Construct an empty index.
   *
   * Constructs an empty index. This index will either need to be trained with `build`
   * or loaded from a saved copy with `deserialize`
   */
  index(raft::resources const& res);

  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::resources const& res, const index_params& params, uint32_t dim);
  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::resources const& res,
        cuvs::distance::DistanceType metric,
        uint32_t n_lists,
        bool adaptive_centers,
        bool conservative_memory_allocation,
        uint32_t dim);

  /**
   * Vectorized load/store size in elements, determines the size of interleaved data chunks.
   */
  uint32_t veclen() const noexcept;

  /** Distance metric used for clustering. */
  cuvs::distance::DistanceType metric() const noexcept;

  /** Whether `centers()` change upon extending the index (ivf_flat::extend). */
  bool adaptive_centers() const noexcept;

  /**
   * Inverted list data [size, dim].
   *
   * The data consists of the dataset rows, grouped by their labels (into clusters/lists).
   * Within each list (cluster), the data is grouped into blocks of `kIndexGroupSize` interleaved
   * vectors. Note, the total index length is slightly larger than the source dataset length,
   * because each cluster is padded by `kIndexGroupSize` elements.
   *
   * Interleaving pattern:
   * within groups of `kIndexGroupSize` rows, the data is interleaved with the block size equal to
   * `veclen * sizeof(T)`. That is, a chunk of `veclen` consecutive components of one row is
   * followed by a chunk of the same size of the next row, and so on.
   *
   * __Example__: veclen = 2, dim = 6, kIndexGroupSize = 32, list_size = 31
   *
   *     x[ 0, 0], x[ 0, 1], x[ 1, 0], x[ 1, 1], ... x[14, 0], x[14, 1], x[15, 0], x[15, 1],
   *     x[16, 0], x[16, 1], x[17, 0], x[17, 1], ... x[30, 0], x[30, 1],    -    ,    -    ,
   *     x[ 0, 2], x[ 0, 3], x[ 1, 2], x[ 1, 3], ... x[14, 2], x[14, 3], x[15, 2], x[15, 3],
   *     x[16, 2], x[16, 3], x[17, 2], x[17, 3], ... x[30, 2], x[30, 3],    -    ,    -    ,
   *     x[ 0, 4], x[ 0, 5], x[ 1, 4], x[ 1, 5], ... x[14, 4], x[14, 5], x[15, 4], x[15, 5],
   *     x[16, 4], x[16, 5], x[17, 4], x[17, 5], ... x[30, 4], x[30, 5],    -    ,    -    ,
   *
   */
  /** Sizes of the lists (clusters) [n_lists]
   * NB: This may differ from the actual list size if the shared lists have been extended by another
   * index
   */
  raft::device_vector_view<uint32_t, uint32_t> list_sizes() noexcept;
  raft::device_vector_view<const uint32_t, uint32_t> list_sizes() const noexcept;

  /** k-means cluster centers corresponding to the lists [n_lists, dim] */
  raft::device_matrix_view<float, uint32_t, raft::row_major> centers() noexcept;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers() const noexcept;

  /**
   * (Optional) Precomputed norms of the `centers` w.r.t. the chosen distance metric [n_lists].
   *
   * NB: this may be empty if the index is empty or if the metric does not require the center norms
   * calculation.
   */
  std::optional<raft::device_vector_view<float, uint32_t>> center_norms() noexcept;
  std::optional<raft::device_vector_view<const float, uint32_t>> center_norms() const noexcept;

  /**
   * Accumulated list sizes, sorted in descending order [n_lists + 1].
   * The last value contains the total length of the index.
   * The value at index zero is always zero.
   *
   * That is, the content of this span is as if the `list_sizes` was sorted and then accumulated.
   *
   * This span is used during search to estimate the maximum size of the workspace.
   */
  auto accum_sorted_sizes() noexcept -> raft::host_vector_view<IdxT, uint32_t>;
  [[nodiscard]] auto accum_sorted_sizes() const noexcept
    -> raft::host_vector_view<const IdxT, uint32_t>;

  /** Total length of the index. */
  IdxT size() const noexcept;

  /** Dimensionality of the data. */
  uint32_t dim() const noexcept;

  /** Number of clusters/inverted lists. */
  uint32_t n_lists() const noexcept;
  raft::device_vector_view<T*, uint32_t> data_ptrs() noexcept;
  raft::device_vector_view<T* const, uint32_t> data_ptrs() const noexcept;

  /** Pointers to the inverted lists (clusters) indices  [n_lists]. */
  raft::device_vector_view<IdxT*, uint32_t> inds_ptrs() noexcept;
  raft::device_vector_view<IdxT* const, uint32_t> inds_ptrs() const noexcept;

  /**
   * Whether to use conservative memory allocation when extending the list (cluster) data
   * (see index_params.conservative_memory_allocation).
   */
  bool conservative_memory_allocation() const noexcept;

  void allocate_center_norms(raft::resources const& res);

  /** Lists' data and indices. */
  std::vector<std::shared_ptr<list_data<T, IdxT>>>& lists() noexcept;
  const std::vector<std::shared_ptr<list_data<T, IdxT>>>& lists() const noexcept;

  void check_consistency();

 private:
  /**
   * TODO: in theory, we can lift this to the template parameter and keep it at hardware maximum
   * possible value by padding the `dim` of the data https://github.com/rapidsai/raft/issues/711
   */
  uint32_t veclen_;
  cuvs::distance::DistanceType metric_;
  bool adaptive_centers_;
  bool conservative_memory_allocation_;
  std::vector<std::shared_ptr<list_data<T, IdxT>>> lists_;
  raft::device_vector<uint32_t, uint32_t> list_sizes_;
  raft::device_matrix<float, uint32_t, raft::row_major> centers_;
  std::optional<raft::device_vector<float, uint32_t>> center_norms_;

  // Computed members
  raft::device_vector<T*, uint32_t> data_ptrs_;
  raft::device_vector<IdxT*, uint32_t> inds_ptrs_;
  raft::host_vector<IdxT, uint32_t> accum_sorted_sizes_;

  static auto calculate_veclen(uint32_t dim) -> uint32_t
  {
    // TODO: consider padding the dimensions and fixing veclen to its maximum possible value as a
    // template parameter (https://github.com/rapidsai/raft/issues/711)

    // NOTE: keep this consistent with the select_interleaved_scan_kernel logic
    // in detail/ivf_flat_interleaved_scan-inl.cuh.
    uint32_t veclen = std::max<uint32_t>(1, 16 / sizeof(T));
    if (dim % veclen != 0) { veclen = 1; }
    return veclen;
  }
};
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_index_build IVF-Flat index build
 * @{
 */
/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<float, int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_flat::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_flat::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_flat::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<float, int64_t>& idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<half, int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_flat::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_flat::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_flat::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<half, int64_t>& idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<int8_t, int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_flat::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_flat::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_flat::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_flat::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_flat::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_flat::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<float, int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   ivf_flat::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_flat::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_flat::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<float, int64_t>& idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<half, int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   ivf_flat::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_flat::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_flat::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<half, int64_t>& idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<int8_t, int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   ivf_flat::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_flat::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_flat::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   ivf_flat::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_flat::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_flat::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx);
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_index_extend IVF-Flat index extend
 * @{
 */

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_flat::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_flat::index<float, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<float, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_flat::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_flat::index<float, int64_t>* idx);

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_flat::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_flat::index<half, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<half, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_flat::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_flat::index<half, int64_t>* idx);

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, dataset, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_flat::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<int8_t, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_flat::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* idx);

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, dataset, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_flat::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_flat::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* idx);

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_flat::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_flat::index<float, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<float, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_flat::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_flat::index<float, int64_t>* idx);

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_flat::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_flat::index<half, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<half, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_flat::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_flat::index<half, int64_t>* idx);

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, dataset, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_flat::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<int8_t, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_flat::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* idx);

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, dataset, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_flat::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_flat::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* idx);
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_index_search IVF-Flat index search
 * @{
 */

/**
 * @brief Search ANN using the constructed index.
 * This function JIT compiles the kernel for the very first usage, after which it maintains an
 * in-memory and disk-based cache of the compiled kernels. We recommend running a warmup search
 * before the actual searches to avoid the first-time JIT compilation overhead.
 *
 * See the [ivf_flat::build](#ivf_flat::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`:
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_flat::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_flat::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_flat::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_flat::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] index ivf-flat constructed index
 * @param[in] queries raft::device_matrix_view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors raft::device_matrix_view to the indices of the neighbors in the source
 * dataset [n_queries, k]
 * @param[out] distances raft::device_matrix_view to the distances to the selected neighbors
 * [n_queries, k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_flat::search_params& params,
            const cuvs::neighbors::ivf_flat::index<float, int64_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 * This function JIT compiles the kernel for the very first usage, after which it maintains an
 * in-memory and disk-based cache of the compiled kernels. We recommend running a warmup search
 * before the actual searches to avoid the first-time JIT compilation overhead.
 *
 * See the [ivf_flat::build](#ivf_flat::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`:
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_flat::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_flat::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_flat::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_flat::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] index ivf-flat constructed index
 * @param[in] queries raft::device_matrix_view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors raft::device_matrix_view to the indices of the neighbors in the source
 * dataset [n_queries, k]
 * @param[out] distances raft::device_matrix_view to the distances to the selected neighbors
 * [n_queries, k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_flat::search_params& params,
            const cuvs::neighbors::ivf_flat::index<half, int64_t>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});
/**
 * @brief Search ANN using the constructed index.
 * This function JIT compiles the kernel for the very first usage, after which it maintains an
 * in-memory and disk-based cache of the compiled kernels. We recommend running a warmup search
 * before the actual searches to avoid the first-time JIT compilation overhead.
 *
 * See the [ivf_flat::build](#ivf_flat::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`:
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_flat::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_flat::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_flat::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_flat::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] index ivf-flat constructed index
 * @param[in] queries raft::device_matrix_view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors raft::device_matrix_view to the indices of the neighbors in the source
 * dataset [n_queries, k]
 * @param[out] distances raft::device_matrix_view to the distances to the selected neighbors
 * [n_queries, k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_flat::search_params& params,
            const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 * This function JIT compiles the kernel for the very first usage, after which it maintains an
 * in-memory and disk-based cache of the compiled kernels. We recommend running a warmup search
 * before the actual searches to avoid the first-time JIT compilation overhead.
 *
 * See the [ivf_flat::build](#ivf_flat::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`:
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_flat::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_flat::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_flat::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_flat::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] index ivf-flat constructed index
 * @param[in] queries raft::device_matrix_view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors raft::device_matrix_view to the indices of the neighbors in the source
 * dataset [n_queries, k]
 * @param[out] distances raft::device_matrix_view to the distances to the selected neighbors
 * [n_queries, k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_flat::search_params& params,
            const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_serialize IVF-Flat index serialize
 * @{
 */

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = ivf_flat::build(...);`
 * cuvs::neighbors::ivf_flat::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::ivf_flat::index<float, int64_t>& index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using T    = float; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with `ivf_flat::index<T, IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::ivf_flat::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_flat::index<float, int64_t>* index);

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = ivf_flat::build(...);`
 * cuvs::neighbors::ivf_flat::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::ivf_flat::index<float, int64_t>& index);

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * using T    = float; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with `ivf_flat::index<T, IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::ivf_flat::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::ivf_flat::index<float, int64_t>* index);

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = ivf_flat::build(...);`
 * cuvs::neighbors::ivf_flat::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::ivf_flat::index<half, int64_t>& index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using T    = half; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with `ivf_flat::index<T, IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::ivf_flat::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_flat::index<half, int64_t>* index);

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = ivf_flat::build(...);`
 * cuvs::neighbors::ivf_flat::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::ivf_flat::index<half, int64_t>& index);

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * using T    = half; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with `ivf_flat::index<T, IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::ivf_flat::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::ivf_flat::index<half, int64_t>* index);

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = ivf_flat::build(...);`
 * cuvs::neighbors::ivf_flat::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using T    = float; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with `ivf_flat::index<T, IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::ivf_flat::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* index);

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = ivf_flat::build(...);`
 * cuvs::neighbors::ivf_flat::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& index);

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * using T    = float; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with `ivf_flat::index<T, IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::ivf_flat::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* index);

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = ivf_flat::build(...);`
 * cuvs::neighbors::ivf_flat::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using T    = float; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with ivf_flat::index<T, IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::ivf_flat::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* index);

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = ivf_flat::build(...);`
 * cuvs::neighbors::ivf_flat::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& index);

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * using T    = float; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with `ivf_flat::index<T, IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::ivf_flat::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* index);

/**
 * @}
 */

/// \defgroup mg_cpp_index_build ANN MG index build

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-Flat MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<ivf_flat::index_params>& index_params,
           raft::host_matrix_view<const float, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<ivf_flat::index<float, int64_t>, float, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-Flat MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<ivf_flat::index_params>& index_params,
           raft::host_matrix_view<const half, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<ivf_flat::index<half, int64_t>, half, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-Flat MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<ivf_flat::index_params>& index_params,
           raft::host_matrix_view<const int8_t, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<ivf_flat::index<int8_t, int64_t>, int8_t, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-Flat MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<ivf_flat::index_params>& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<ivf_flat::index<uint8_t, int64_t>, uint8_t, int64_t>;

/// \defgroup mg_cpp_index_extend ANN MG index extend

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * cuvs::neighbors::ivf_flat::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<ivf_flat::index<float, int64_t>, float, int64_t>& index,
            raft::host_matrix_view<const float, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * cuvs::neighbors::ivf_flat::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<ivf_flat::index<half, int64_t>, half, int64_t>& index,
            raft::host_matrix_view<const half, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * cuvs::neighbors::ivf_flat::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<ivf_flat::index<int8_t, int64_t>, int8_t, int64_t>& index,
            raft::host_matrix_view<const int8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * cuvs::neighbors::ivf_flat::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<ivf_flat::index<uint8_t, int64_t>, uint8_t, int64_t>& index,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \defgroup mg_cpp_index_search ANN MG index search

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<ivf_flat::search_params> search_params;
 * cuvs::neighbors::ivf_flat::search(clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 *
 */
void search(const raft::resources& clique,
            const cuvs::neighbors::mg_index<ivf_flat::index<float, int64_t>, float, int64_t>& index,
            const cuvs::neighbors::mg_search_params<ivf_flat::search_params>& search_params,
            raft::host_matrix_view<const float, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<ivf_flat::search_params> search_params;
 * cuvs::neighbors::ivf_flat::search(clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 *
 */
void search(const raft::resources& clique,
            const cuvs::neighbors::mg_index<ivf_flat::index<half, int64_t>, half, int64_t>& index,
            const cuvs::neighbors::mg_search_params<ivf_flat::search_params>& search_params,
            raft::host_matrix_view<const half, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<ivf_flat::search_params> search_params;
 * cuvs::neighbors::ivf_flat::search(clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 *
 */
void search(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<ivf_flat::index<int8_t, int64_t>, int8_t, int64_t>& index,
  const cuvs::neighbors::mg_search_params<ivf_flat::search_params>& search_params,
  raft::host_matrix_view<const int8_t, int64_t, row_major> queries,
  raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<ivf_flat::search_params> search_params;
 * cuvs::neighbors::ivf_flat::search(clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 *
 */
void search(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<ivf_flat::index<uint8_t, int64_t>, uint8_t, int64_t>& index,
  const cuvs::neighbors::mg_search_params<ivf_flat::search_params>& search_params,
  raft::host_matrix_view<const uint8_t, int64_t, row_major> queries,
  raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances);

/// \defgroup mg_cpp_serialize ANN MG index serialization

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::ivf_flat::serialize(clique, index, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<ivf_flat::index<float, int64_t>, float, int64_t>& index,
  const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::ivf_flat::serialize(clique, index, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<ivf_flat::index<half, int64_t>, half, int64_t>& index,
  const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::ivf_flat::serialize(clique, index, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<ivf_flat::index<int8_t, int64_t>, int8_t, int64_t>& index,
  const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::ivf_flat::serialize(clique, index, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<ivf_flat::index<uint8_t, int64_t>, uint8_t, int64_t>& index,
  const std::string& filename);

/// \ingroup mg_cpp_deserialize
/**
 * @brief Deserializes an IVF-Flat multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::ivf_flat::serialize(clique, index, filename);
 * auto new_index = cuvs::neighbors::ivf_flat::deserialize<float, int64_t>(clique, filename);
 *
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] filename path to the file to be deserialized
 *
 */
template <typename T, typename IdxT>
auto deserialize(const raft::resources& clique, const std::string& filename)
  -> cuvs::neighbors::mg_index<ivf_flat::index<T, IdxT>, T, IdxT>;

/// \defgroup mg_cpp_distribute ANN MG local index distribution

/// \ingroup mg_cpp_distribute
/**
 * @brief Replicates a locally built and serialized IVF-Flat index to all GPUs to form a distributed
 * multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::ivf_flat::index_params index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, index_dataset);
 * const std::string filename = "local_index.cuvs";
 * cuvs::neighbors::ivf_flat::serialize(clique, filename, index);
 * auto new_index = cuvs::neighbors::ivf_flat::distribute<float, int64_t>(clique, filename);
 *
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] filename path to the file to be deserialized : a local index
 *
 */
template <typename T, typename IdxT>
auto distribute(const raft::resources& clique, const std::string& filename)
  -> cuvs::neighbors::mg_index<ivf_flat::index<T, IdxT>, T, IdxT>;

namespace helpers {

/**
 * @defgroup ivf_flat_helpers Helper functions for IVF Flat
 * @{
 */

namespace codepacker {

/**
 * Write flat codes into an existing list by the given offset.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data  = index.lists()[label]->data.view();
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<float>(res, n_vec, index.dim());
 *   ... prepare n_vecs to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position
 *   ivf_flat::helpers::codepacker::pack(
 *       res, make_const_mdspan(codes.view()), index.veclen(), 42, list_data);
 * @endcode
 *
 * @param[in] res
 * @param[in] codes flat codes [n_vec, dim]
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[inout] list_data block to write into
 */
void pack(raft::resources const& res,
          raft::device_matrix_view<const float, uint32_t, raft::row_major> codes,
          uint32_t veclen,
          uint32_t offset,
          raft::device_mdspan<float,
                              typename list_spec<uint32_t, float, int64_t>::list_extents,
                              raft::row_major> list_data);

/**
 * Write flat codes into an existing list by the given offset.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data  = index.lists()[label]->data.view();
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<half>(res, n_vec, index.dim());
 *   ... prepare n_vecs to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position
 *   ivf_flat::helpers::codepacker::pack(
 *       res, make_const_mdspan(codes.view()), index.veclen(), 42, list_data);
 * @endcode
 *
 * @param[in] res
 * @param[in] codes flat codes [n_vec, dim]
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[inout] list_data block to write into
 */
void pack(raft::resources const& res,
          raft::device_matrix_view<const half, uint32_t, raft::row_major> codes,
          uint32_t veclen,
          uint32_t offset,
          raft::device_mdspan<half,
                              typename list_spec<uint32_t, half, int64_t>::list_extents,
                              raft::row_major> list_data);

/**
 * Write flat codes into an existing list by the given offset.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data  = index.lists()[label]->data.view();
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<int8_t>(res, n_vec, index.dim());
 *   ... prepare n_vecs to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position
 *   ivf_flat::helpers::codepacker::pack(
 *       res, make_const_mdspan(codes.view()), index.veclen(), 42, list_data);
 * @endcode
 *
 * @param[in] res
 * @param[in] codes flat codes [n_vec, dim]
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[inout] list_data block to write into
 */
void pack(raft::resources const& res,
          raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> codes,
          uint32_t veclen,
          uint32_t offset,
          raft::device_mdspan<int8_t,
                              typename list_spec<uint32_t, int8_t, int64_t>::list_extents,
                              raft::row_major> list_data);

/**
 * Write flat codes into an existing list by the given offset.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data  = index.lists()[label]->data.view();
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<uint8_t>(res, n_vec, index.dim());
 *   ... prepare n_vecs to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position
 *   ivf_flat::helpers::codepacker::pack(
 *       res, make_const_mdspan(codes.view()), index.veclen(), 42, list_data);
 * @endcode
 *
 * @param[in] res
 * @param[in] codes flat codes [n_vec, dim]
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[inout] list_data block to write into
 */
void pack(raft::resources const& res,
          raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
          uint32_t veclen,
          uint32_t offset,
          raft::device_mdspan<uint8_t,
                              typename list_spec<uint32_t, uint8_t, int64_t>::list_extents,
                              raft::row_major> list_data);

/**
 * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`.
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_take = 4;
 *   auto codes = raft::make_device_matrix<float>(res, n_take, index.dim());
 *   uint32_t offset = 0;
 *   // unpack n_take elements from the list
 *   ivf_fat::helpers::codepacker::unpack(res, list_data, index.veclen(), offset, codes.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] list_data block to read from
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset
 *   How many records in the list to skip.
 * @param[inout] codes
 *   the destination buffer [n_take, index.dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be <= the list size.
 */
void unpack(raft::resources const& res,
            raft::device_mdspan<const float,
                                typename list_spec<uint32_t, float, int64_t>::list_extents,
                                raft::row_major> list_data,
            uint32_t veclen,
            uint32_t offset,
            raft::device_matrix_view<float, uint32_t, raft::row_major> codes);

/**
 * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`.
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_take = 4;
 *   auto codes = raft::make_device_matrix<half>(res, n_take, index.dim());
 *   uint32_t offset = 0;
 *   // unpack n_take elements from the list
 *   ivf_fat::helpers::codepacker::unpack(res, list_data, index.veclen(), offset, codes.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] list_data block to read from
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset
 *   How many records in the list to skip.
 * @param[inout] codes
 *   the destination buffer [n_take, index.dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be <= the list size.
 */
void unpack(raft::resources const& res,
            raft::device_mdspan<const half,
                                typename list_spec<uint32_t, half, int64_t>::list_extents,
                                raft::row_major> list_data,
            uint32_t veclen,
            uint32_t offset,
            raft::device_matrix_view<half, uint32_t, raft::row_major> codes);
/**
 * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`.
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_take = 4;
 *   auto codes = raft::make_device_matrix<int8_t>(res, n_take, index.dim());
 *   uint32_t offset = 0;
 *   // unpack n_take elements from the list
 *   ivf_fat::helpers::codepacker::unpack(res, list_data, index.veclen(), offset, codes.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] list_data block to read from
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset
 *   How many records in the list to skip.
 * @param[inout] codes
 *   the destination buffer [n_take, index.dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be <= the list size.
 */
void unpack(raft::resources const& res,
            raft::device_mdspan<const int8_t,
                                typename list_spec<uint32_t, int8_t, int64_t>::list_extents,
                                raft::row_major> list_data,
            uint32_t veclen,
            uint32_t offset,
            raft::device_matrix_view<int8_t, uint32_t, raft::row_major> codes);

/**
 * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`.
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_take = 4;
 *   auto codes = raft::make_device_matrix<uint8_t>(res, n_take, index.dim());
 *   uint32_t offset = 0;
 *   // unpack n_take elements from the list
 *   ivf_fat::helpers::codepacker::unpack(res, list_data, index.veclen(), offset, codes.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] list_data block to read from
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset
 *   How many records in the list to skip.
 * @param[inout] codes
 *   the destination buffer [n_take, index.dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be <= the list size.
 */
void unpack(raft::resources const& res,
            raft::device_mdspan<const uint8_t,
                                typename list_spec<uint32_t, uint8_t, int64_t>::list_extents,
                                raft::row_major> list_data,
            uint32_t veclen,
            uint32_t offset,
            raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes);

/**
 * Write one flat code into a block by the given offset. The offset indicates the id of the record
 * in the list. This function interleaves the code and is intended to later copy the interleaved
 * codes over to the IVF list on device. NB: no memory allocation happens here; the block must fit
 * the record (offset + 1).
 *
 * @param[in] flat_code input flat code
 * @param[out] block block of memory to write interleaved codes to
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 */
void pack_1(const float* flat_code, float* block, uint32_t dim, uint32_t veclen, uint32_t offset);

/**
 * Write one flat code into a block by the given offset. The offset indicates the id of the record
 * in the list. This function interleaves the code and is intended to later copy the interleaved
 * codes over to the IVF list on device. NB: no memory allocation happens here; the block must fit
 * the record (offset + 1).
 *
 * @param[in] flat_code input flat code
 * @param[out] block block of memory to write interleaved codes to
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 */
void pack_1(const half* flat_code, half* block, uint32_t dim, uint32_t veclen, uint32_t offset);

/**
 * Write one flat code into a block by the given offset. The offset indicates the id of the record
 * in the list. This function interleaves the code and is intended to later copy the interleaved
 * codes over to the IVF list on device. NB: no memory allocation happens here; the block must fit
 * the record (offset + 1).
 *
 * @param[in] flat_code input flat code
 * @param[out] block block of memory to write interleaved codes to
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 */
void pack_1(const int8_t* flat_code, int8_t* block, uint32_t dim, uint32_t veclen, uint32_t offset);

/**
 * Write one flat code into a block by the given offset. The offset indicates the id of the record
 * in the list. This function interleaves the code and is intended to later copy the interleaved
 * codes over to the IVF list on device. NB: no memory allocation happens here; the block must fit
 * the record (offset + 1).
 *
 * @param[in] flat_code input flat code
 * @param[out] block block of memory to write interleaved codes to
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 */
void pack_1(
  const uint8_t* flat_code, uint8_t* block, uint32_t dim, uint32_t veclen, uint32_t offset);

/**
 * Unpack 1 record of a single list (cluster) in the index to fetch the flat code. The offset
 * indicates the id of the record. This function fetches one flat code from an interleaved code.
 *
 * @param[in] block interleaved block. The block can be thought of as the whole inverted list in
 * interleaved format.
 * @param[out] flat_code output flat code
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset fetch the flat code by the given offset
 */
void unpack_1(const float* block, float* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset);

/**
 * Unpack 1 record of a single list (cluster) in the index to fetch the flat code. The offset
 * indicates the id of the record. This function fetches one flat code from an interleaved code.
 *
 * @param[in] block interleaved block. The block can be thought of as the whole inverted list in
 * interleaved format.
 * @param[out] flat_code output flat code
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset fetch the flat code by the given offset
 */
void unpack_1(const half* block, half* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset);

/**
 * Unpack 1 record of a single list (cluster) in the index to fetch the flat code. The offset
 * indicates the id of the record. This function fetches one flat code from an interleaved code.
 *
 * @param[in] block interleaved block. The block can be thought of as the whole inverted list in
 * interleaved format.
 * @param[out] flat_code output flat code
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset fetch the flat code by the given offset
 */
void unpack_1(
  const int8_t* block, int8_t* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset);

/**
 * Unpack 1 record of a single list (cluster) in the index to fetch the flat code. The offset
 * indicates the id of the record. This function fetches one flat code from an interleaved code.
 *
 * @param[in] block interleaved block. The block can be thought of as the whole inverted list in
 * interleaved format.
 * @param[out] flat_code output flat code
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset fetch the flat code by the given offset
 */
void unpack_1(
  const uint8_t* block, uint8_t* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset);

}  // namespace codepacker

/**
 * @brief Public helper API to reset the data and indices ptrs, and the list sizes. Useful for
 * externally modifying the index without going through the build stage. The data and indices of the
 * IVF lists will be lost.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // initialize an empty index
 *   ivf_flat::index<float, int64_t> index(res, index_params, D);
 *   // reset the index's state and list sizes
 *   ivf_flat::helpers::reset_index(res, &index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-Flat index
 */
void reset_index(const raft::resources& res, index<float, int64_t>* index);

/**
 * @brief Public helper API to reset the data and indices ptrs, and the list sizes. Useful for
 * externally modifying the index without going through the build stage. The data and indices of the
 * IVF lists will be lost.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // initialize an empty index
 *   ivf_flat::index<half, int64_t> index(res, index_params, D);
 *   // reset the index's state and list sizes
 *   ivf_flat::helpers::reset_index(res, &index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-Flat index
 */
void reset_index(const raft::resources& res, index<half, int64_t>* index);

/**
 * @brief Public helper API to reset the data and indices ptrs, and the list sizes. Useful for
 * externally modifying the index without going through the build stage. The data and indices of the
 * IVF lists will be lost.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // initialize an empty index
 *   ivf_flat::index<int8_t, int64_t> index(res, index_params, D);
 *   // reset the index's state and list sizes
 *   ivf_flat::helpers::reset_index(res, &index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-Flat index
 */
void reset_index(const raft::resources& res, index<int8_t, int64_t>* index);

/**
 * @brief Public helper API to reset the data and indices ptrs, and the list sizes. Useful for
 * externally modifying the index without going through the build stage. The data and indices of the
 * IVF lists will be lost.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // initialize an empty index
 *   ivf_flat::index<uint8_t, int64_t> index(res, index_params, D);
 *   // reset the index's state and list sizes
 *   ivf_flat::helpers::reset_index(res, &index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-Flat index
 */
void reset_index(const raft::resources& res, index<uint8_t, int64_t>* index);

/**
 * @brief Helper exposing the re-computation of list sizes and related arrays if IVF lists have been
 * modified externally.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // initialize an empty index
 *   ivf_flat::index<uint8_t, int64_t> index(res, index_params, D);
 *   ivf_flat::helpers::reset_index(res, &index);
 *   // resize the first IVF list to hold 5 records
 *   auto spec = list_spec<uint32_t, float, int64_t>{
 *     index.dim(), index.conservative_memory_allocation()};
 *   uint32_t new_size = 5;
 *   ivf::resize_list(res, index.lists()[0], spec, new_size, 0);
 *   raft::update_device(index.list_sizes(), &new_size, 1, stream);
 *   // recompute the internal state of the index
 *   ivf_flat::helpers::recompute_internal_state(res, index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-Flat index
 */
void recompute_internal_state(const raft::resources& res, index<float, int64_t>* index);

/**
 * @brief Helper exposing the re-computation of list sizes and related arrays if IVF lists have been
 * modified externally.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // initialize an empty index
 *   ivf_flat::index<uint8_t, int64_t> index(res, index_params, D);
 *   ivf_flat::helpers::reset_index(res, &index);
 *   // resize the first IVF list to hold 5 records
 *   auto spec = list_spec<uint32_t, half, int64_t>{
 *     index.dim(), index.conservative_memory_allocation()};
 *   uint32_t new_size = 5;
 *   ivf::resize_list(res, index.lists()[0], spec, new_size, 0);
 *   raft::update_device(index.list_sizes(), &new_size, 1, stream);
 *   // recompute the internal state of the index
 *   ivf_flat::helpers::recompute_internal_state(res, index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-Flat index
 */
void recompute_internal_state(const raft::resources& res, index<half, int64_t>* index);

/**
 * @brief Helper exposing the re-computation of list sizes and related arrays if IVF lists have been
 * modified externally.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // initialize an empty index
 *   ivf_flat::index<uint8_t, int64_t> index(res, index_params, D);
 *   ivf_flat::helpers::reset_index(res, &index);
 *   // resize the first IVF list to hold 5 records
 *   auto spec = list_spec<uint32_t, int8_t, int64_t>{
 *     index.dim(), index.conservative_memory_allocation()};
 *   uint32_t new_size = 5;
 *   ivf::resize_list(res, index.lists()[0], spec, new_size, 0);
 *   raft::update_device(index.list_sizes(), &new_size, 1, stream);
 *   // recompute the internal state of the index
 *   ivf_flat::helpers::recompute_internal_state(res, index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-Flat index
 */
void recompute_internal_state(const raft::resources& res, index<int8_t, int64_t>* index);

/**
 * @brief Helper exposing the re-computation of list sizes and related arrays if IVF lists have been
 * modified externally.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // initialize an empty index
 *   ivf_flat::index<uint8_t, int64_t> index(res, index_params, D);
 *   ivf_flat::helpers::reset_index(res, &index);
 *   // resize the first IVF list to hold 5 records
 *   auto spec = list_spec<uint32_t, uint8_t, int64_t>{
 *     index.dim(), index.conservative_memory_allocation()};
 *   uint32_t new_size = 5;
 *   ivf::resize_list(res, index.lists()[0], spec, new_size, 0);
 *   raft::update_device(index.list_sizes(), &new_size, 1, stream);
 *   // recompute the internal state of the index
 *   ivf_flat::helpers::recompute_internal_state(res, index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-Flat index
 */
void recompute_internal_state(const raft::resources& res, index<uint8_t, int64_t>* index);
/**
 * @}
 */

}  // namespace helpers

namespace experimental::udf {

// ============================================================================
// Real C++ implementations for compile-time validation. These allow users to
// verify that their custom UDFs will compile correctly and be usable in our kernels.
// ============================================================================

/**
 * @brief Wrapper for vector elements that provides both packed and unpacked access.
 *
 * For float: trivial wrapper around scalar values
 * For int8/uint8 with Veclen > 1: wraps packed bytes in a 32-bit word
 *
 * @tparam T Data type (float, int8_t, uint8_t). Fp16 vector elements are not supported for UDFs
 *             at this time (see `metric_interface` static_assert when `cuda_fp16.h` is available).
 * @tparam AccT Storage/accumulator type (float, __half, int32_t, uint32_t)
 * @tparam Veclen Vector length (1, 2, 4, 8, 16)
 */
template <typename T, typename AccT, int Veclen>
struct point {
  using element_type          = T;
  using storage_type          = AccT;
  static constexpr int veclen = Veclen;

  storage_type data_;

  point() = default;
  __device__ __host__ explicit point(storage_type d) : data_(d) {}

  /*
  @brief Returns the raw storage type of the point. This is only relevant for int8/uint8 with Veclen
  > 1. When working with packed data, one may use CUDA intrinsics to access faster vectorized
  operations.

  @return The raw storage type of the point.
  */
  __device__ __forceinline__ storage_type raw() const { return data_; }
  __device__ __forceinline__ storage_type& raw() { return data_; }

  /*
  @brief Returns the number of elements in the point. Prefer to use this along with operator[] when
  unsure about CUDA intrinsics.

  @return The number of elements in the point.
  */
  __device__ __host__ static constexpr int size()
  {
    if constexpr ((std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) && Veclen > 1) {
      return 4;
    } else {
      return 1;
    }
  }

  __device__ __host__ static constexpr bool is_packed()
  {
    return (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) && Veclen > 1;
  }

  __device__ __forceinline__ T operator[](int i) const
  {
    if constexpr (std::is_same_v<T, int8_t> && Veclen > 1) {
      return static_cast<int8_t>((data_ >> (i * 8)) & 0xFF);
    } else if constexpr (std::is_same_v<T, uint8_t> && Veclen > 1) {
      return static_cast<uint8_t>((data_ >> (i * 8)) & 0xFF);
    } else {
      (void)i;
      return static_cast<T>(data_);
    }
  }
};

/**
 * @brief Base interface for custom distance metrics.
 */
template <typename T, typename AccT, int Veclen = 1>
struct metric_interface {
  using point_type = point<T, AccT, Veclen>;

#if CUVS_IVF_FLAT_UDF_HAS_CUDA_FP16
  static_assert(
    !(std::is_same_v<std::remove_cv_t<T>, __half> || std::is_same_v<std::remove_cv_t<T>, half>),
    "IVF-Flat custom metric UDF does not support fp16 (__half / half) at this time; do not set "
    "search_params.metric_udf for fp16 indices.");
#endif

  virtual __device__ void operator()(AccT& acc, point_type x, point_type y) = 0;
  virtual ~metric_interface()                                               = default;
};

// ============================================================
// Helper Operations - Deduce Veclen from point type!
// ============================================================

/** @brief Squared difference: (x - y)² */
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT squared_diff(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr (std::is_same_v<T, uint8_t> && V > 1) {
    auto diff = __vabsdiffu4(x.raw(), y.raw());
    return __dp4a(diff, diff, AccT{0});
  } else if constexpr (std::is_same_v<T, int8_t> && V > 1) {
    auto diff = __vabsdiffs4(x.raw(), y.raw());
    return __dp4a(diff, diff, uint32_t{0});
  } else {
    auto diff = x.raw() - y.raw();
    return diff * diff;
  }
}

/** @brief Absolute difference: |x - y| */
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT abs_diff(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr (std::is_same_v<T, uint8_t> && V > 1) {
    auto diff = __vabsdiffu4(x.raw(), y.raw());
    return diff;
  } else if constexpr (std::is_same_v<T, int8_t> && V > 1) {
    auto diff = __vabsdiffs4(x.raw(), y.raw());
    return diff;
  } else {
    auto a = x.raw();
    auto b = y.raw();
    return (a > b) ? (a - b) : (b - a);
  }
}

/** @brief Dot product: x · y */
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT dot_product(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr ((std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) && V > 1) {
    return __dp4a(x.raw(), y.raw(), AccT{0});
  } else {
    return x.raw() * y.raw();
  }
}

/** @brief Element-wise product: x * y */
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT product(point<T, AccT, V> x, point<T, AccT, V> y)
{
  return dot_product(x, y);
}

/** @brief Element-wise sum: x + y */
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT sum(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr ((std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) && V > 1) {
    AccT result = 0;
    for (int i = 0; i < x.size(); ++i) {
      result += static_cast<AccT>(x[i]) + static_cast<AccT>(y[i]);
    }
    return result;
  } else {
    return x.raw() + y.raw();
  }
}

/** @brief Maximum element: max(x, y) */
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT max_elem(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr ((std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) && V > 1) {
    AccT result = 0;
    for (int i = 0; i < x.size(); ++i) {
      auto xi  = static_cast<AccT>(x[i]);
      auto yi  = static_cast<AccT>(y[i]);
      auto val = (xi > yi) ? xi : yi;
      if (val > result) result = val;
    }
    return result;
  } else {
    auto a = x.raw();
    auto b = y.raw();
    return (a > b) ? a : b;
  }
}

// ============================================================================
// String versions for JIT compilation
// ============================================================================

constexpr std::string_view point_code = R"(
template <typename T, typename AccT, int Veclen>
struct point {
  using element_type = T;
  using storage_type = AccT;
  static constexpr int veclen = Veclen;

  storage_type data_;

  point() = default;
  __device__ __host__ explicit point(storage_type d) : data_(d) {}

  __device__ __forceinline__ storage_type raw() const { return data_; }
  __device__ __forceinline__ storage_type& raw() { return data_; }

  __device__ __host__ static constexpr int size()
  {
    if constexpr ((std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) && Veclen > 1) {
      return 4;
    } else {
      return 1;
    }
  }

  __device__ __host__ static constexpr bool is_packed()
  {
    return (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) && Veclen > 1;
  }

  __device__ __forceinline__ T operator[](int i) const
  {
    if constexpr (std::is_same_v<T, int8_t> && Veclen > 1) {
      return static_cast<int8_t>((data_ >> (i * 8)) & 0xFF);
    } else if constexpr (std::is_same_v<T, uint8_t> && Veclen > 1) {
      return static_cast<uint8_t>((data_ >> (i * 8)) & 0xFF);
    } else {
      (void)i;
      return static_cast<T>(data_);
    }
  }
};
)";

constexpr std::string_view metric_interface_code = R"(
template <typename T, typename AccT, int Veclen = 1>
struct metric_interface {
  using point_type = point<T, AccT, Veclen>;

  virtual __device__ void operator()(AccT& acc, point_type x, point_type y) = 0;
  virtual ~metric_interface() = default;
};
)";

constexpr std::string_view squared_diff_code = R"(
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT squared_diff(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr (std::is_same_v<T, uint8_t> && V > 1) {
#if CUVS_UDF_JIT_SCALAR_PACKED_BYTE_OPS
    AccT acc{0};
    for (int i = 0; i < 4; ++i) {
      unsigned xi = (x.raw() >> (i * 8)) & 0xFFu;
      unsigned yi = (y.raw() >> (i * 8)) & 0xFFu;
      AccT d        = static_cast<AccT>(xi) - static_cast<AccT>(yi);
      acc += d * d;
    }
    return acc;
#else
    auto diff = __vabsdiffu4(x.raw(), y.raw());
    return __dp4a(diff, diff, AccT{0});
#endif
  } else if constexpr (std::is_same_v<T, int8_t> && V > 1) {
#if CUVS_UDF_JIT_SCALAR_PACKED_BYTE_OPS
    AccT acc{0};
    for (int i = 0; i < 4; ++i) {
      int xi = static_cast<int>(static_cast<int8_t>((x.raw() >> (i * 8)) & 0xFFu));
      int yi = static_cast<int>(static_cast<int8_t>((y.raw() >> (i * 8)) & 0xFFu));
      AccT d = static_cast<AccT>(xi - yi);
      acc += d * d;
    }
    return acc;
#else
    auto diff = __vabsdiffs4(x.raw(), y.raw());
    return __dp4a(diff, diff, uint32_t{0});
#endif
  } else {
    auto diff = x.raw() - y.raw();
    return diff * diff;
  }
}
)";

constexpr std::string_view abs_diff_code = R"(
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT abs_diff(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr (std::is_same_v<T, uint8_t> && V > 1) {
#if CUVS_UDF_JIT_SCALAR_PACKED_BYTE_OPS
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
      unsigned xi = (x.raw() >> (i * 8)) & 0xFFu;
      unsigned yi = (y.raw() >> (i * 8)) & 0xFFu;
      unsigned d  = (xi > yi) ? (xi - yi) : (yi - xi);
      r |= (d & 0xFFu) << (i * 8);
    }
    return r;
#else
    auto diff = __vabsdiffu4(x.raw(), y.raw());
    return diff;
#endif
  } else if constexpr (std::is_same_v<T, int8_t> && V > 1) {
#if CUVS_UDF_JIT_SCALAR_PACKED_BYTE_OPS
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
      int xi     = static_cast<int8_t>((x.raw() >> (i * 8)) & 0xFFu);
      int yi     = static_cast<int8_t>((y.raw() >> (i * 8)) & 0xFFu);
      unsigned d = static_cast<unsigned>(xi > yi ? xi - yi : yi - xi);
      r |= (d & 0xFFu) << (i * 8);
    }
    return r;
#else
    auto diff = __vabsdiffs4(x.raw(), y.raw());
    return diff;
#endif
  } else {
    auto a = x.raw();
    auto b = y.raw();
    return (a > b) ? (a - b) : (b - a);
  }
}
)";

constexpr std::string_view dot_product_code = R"(
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT dot_product(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr ((std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) && V > 1) {
#if CUVS_UDF_JIT_SCALAR_PACKED_BYTE_OPS
    AccT acc = AccT{0};
    for (int i = 0; i < 4; ++i) {
      int xi = static_cast<int8_t>((x.raw() >> (i * 8)) & 0xFFu);
      int yi = static_cast<int8_t>((y.raw() >> (i * 8)) & 0xFFu);
      acc += static_cast<AccT>(xi) * static_cast<AccT>(yi);
    }
    return acc;
#else
    return __dp4a(x.raw(), y.raw(), AccT{0});
#endif
  } else {
    return x.raw() * y.raw();
  }
}
)";

constexpr std::string_view product_code = R"(
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT product(point<T, AccT, V> x, point<T, AccT, V> y)
{
  return dot_product(x, y);
}
)";

constexpr std::string_view sum_code = R"(
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT sum(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr ((std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) && V > 1) {
    AccT result = 0;
    for (int i = 0; i < x.size(); ++i) {
      result += static_cast<AccT>(x[i]) + static_cast<AccT>(y[i]);
    }
    return result;
  } else {
    return x.raw() + y.raw();
  }
}
)";

constexpr std::string_view max_elem_code = R"(
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT max_elem(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr ((std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) && V > 1) {
    AccT result = 0;
    for (int i = 0; i < x.size(); ++i) {
      auto xi  = static_cast<AccT>(x[i]);
      auto yi  = static_cast<AccT>(y[i]);
      auto val = (xi > yi) ? xi : yi;
      if (val > result) result = val;
    }
    return result;
  } else {
    auto a = x.raw();
    auto b = y.raw();
    return (a > b) ? a : b;
  }
}
)";

/**
 * @brief Preamble code for JIT compilation.
 *
 * nvrtc doesn't have access to standard library headers, so we define
 * the necessary types and utilities inline.
 */
constexpr std::string_view jit_preamble_code = R"(
/* Fixed-width integer types for nvrtc */
using int8_t = signed char;
using uint8_t = unsigned char;
using int32_t = int;
using uint32_t = unsigned int;
using int64_t = long long;
using uint64_t = unsigned long long;

/* std::is_same_v implementation for nvrtc */
namespace std {
template <typename T, typename U> struct is_same { static constexpr bool value = false; };
template <typename T> struct is_same<T, T> { static constexpr bool value = true; };
template <typename T, typename U> inline constexpr bool is_same_v = is_same<T, U>::value;
}

/* NVRTC sets __CUDACC_VER_*; CUDA 12.2 and earlier lack __dp4a in this UDF path. */
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && (__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ < 3)
#define CUVS_UDF_JIT_SCALAR_PACKED_BYTE_OPS 1
#else
#define CUVS_UDF_JIT_SCALAR_PACKED_BYTE_OPS 0
#endif
)";

/**
 * NVRTC/LTO fragment appended to custom metric UDF source. The UDF program is compiled alone, so it
 * must supply the `compute_dist` definition that device_functions.cuh only declares.
 */
inline std::string instantiate_udf(char const* data_type, char const* acc_type, int veclen)
{
  std::ostringstream oss;
  oss << "\nnamespace cuvs::neighbors::ivf_flat::detail {\n"
      << "template <typename AccT>\n"
      << "__device__ void compute_dist(AccT& acc, AccT x, AccT y) {\n"
      << "  compute_dist_udf_impl<" << data_type << ", " << acc_type << ", " << veclen
      << ">(acc, x, y);\n"
      << "}\n"
      << "template __device__ void compute_dist<" << acc_type << ">(" << acc_type << "&, "
      << acc_type << ", " << acc_type << ");\n"
      << "}\n";
  return oss.str();
}

/**
 * @brief Define a custom distance metric with compile-time validation.
 *
 * This macro creates:
 * 1. A struct that inherits from metric_interface (compile-time validation)
 * 2. A function NAME_udf() that returns a metric_source for JIT compilation
 * 3. A BODY that needs to be compiled by nvrtc and must be valid CUDA device code
 *
 * NOTE: The metric function only supports ascending order top-k selection without norms or post
 * processing.
 *
 * @param NAME The name of your metric (becomes struct name and function prefix)
 * @param BODY The body of operator()(AccT& acc, point_type x, point_type y)
 *
 * Available in BODY:
 *   acc    - Accumulated distance (AccT&, modify in-place)
 *   x, y   - Vector elements (point<T, AccT, Veclen>)
 *   T      - Data type (float, __half, int8_t, uint8_t)
 *   AccT   - Accumulator type
 *   Veclen - Vector length (compile-time constant)
 *
 * x and y provide:
 *   x.raw()     - Raw packed storage (for power users)
 *   x[i]        - Unpacked element access
 *   x.size()    - Number of elements (4 for packed int8, 1 for float)
 *   x.is_packed() - Whether data is packed (constexpr)
 *
 * Helper functions (Veclen deduced automatically!):
 *   squared_diff(x, y) - (x-y)² optimized for all types
 *   abs_diff(x, y)     - |x-y| optimized for all types
 *   dot_product(x, y)  - x·y optimized for all types
 *   product(x, y)      - element-wise product
 *
 * Example:
 *   CUVS_METRIC(my_l2, {
 *       acc += squared_diff(x, y);  // Just works for all types!
 *   })
 *
 *   CUVS_METRIC(my_chebyshev, {
 *       auto d = abs_diff(x, y);
 *       acc    = (d > acc) ? d : acc;
 *   })
 */
#define CUVS_METRIC(NAME, BODY)                                                                   \
  template <typename T, typename AccT, int Veclen>                                                \
  struct NAME : cuvs::neighbors::ivf_flat::experimental::udf::metric_interface<T, AccT, Veclen> { \
    using point_type = cuvs::neighbors::ivf_flat::experimental::udf::point<T, AccT, Veclen>;      \
    __device__ void operator()(AccT& acc, point_type x, point_type y) override { BODY }           \
  };                                                                                              \
                                                                                                  \
  inline std::string NAME##_udf()                                                                 \
  {                                                                                               \
    using namespace cuvs::neighbors::ivf_flat::experimental::udf;                                 \
    std::string result;                                                                           \
    result += jit_preamble_code;                                                                  \
    result += point_code;                                                                         \
    result += squared_diff_code;                                                                  \
    result += abs_diff_code;                                                                      \
    result += dot_product_code;                                                                   \
    result += product_code;                                                                       \
    result += sum_code;                                                                           \
    result += max_elem_code;                                                                      \
    result += metric_interface_code;                                                              \
    result += R"(                                                                        \
template <typename T, typename AccT, int Veclen>                                       \
struct )" #NAME R"( : metric_interface<T, AccT, Veclen> {                              \
  using point_type = point<T, AccT, Veclen>;                                           \
  __device__ void operator()(AccT& acc, point_type x, point_type y) override           \
)" #BODY R"(                                                                           \
};                                                                                     \
                                                                                       \
namespace cuvs::neighbors::ivf_flat::detail {                                        \
template <typename T, typename AccT, int Veclen>                                       \
__device__ __forceinline__ void compute_dist_udf_impl(AccT& acc, AccT x, AccT y)       \
{                                                                                      \
  ::)" #NAME R"(<T, AccT, Veclen> metric;                                              \
  metric(acc, ::point<T, AccT, Veclen>(x), ::point<T, AccT, Veclen>(y));               \
}                                                                                      \
}                                                                                    \
)";                                                                                               \
    return result;                                                                                \
  }

}  // namespace experimental::udf

}  // namespace ivf_flat
}  // namespace neighbors

}  // namespace CUVS_EXPORT cuvs
