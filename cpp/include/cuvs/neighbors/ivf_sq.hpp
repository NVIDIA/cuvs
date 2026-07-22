/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/util/integer_utils.hpp>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace ivf_sq {

/**
 * @defgroup ivf_sq_cpp_index_params IVF-SQ index build parameters
 * @{
 */

constexpr static uint32_t kIndexGroupSize = 32;

/**
 * @brief IVF-SQ index build parameters.
 *
 * IVF-SQ currently uses 8-bit scalar quantization, storing one `uint8_t` code per vector
 * dimension.
 */
struct index_params : cuvs::neighbors::index_params {
  /** The number of inverted lists (clusters) */
  uint32_t n_lists = 1024;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters = 20;
  /** The number of data vectors (per cluster) to use during iterative kmeans building. */
  uint32_t max_train_points_per_cluster = 256;
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

static_assert(std::is_aggregate_v<index_params>);

/**
 * @}
 */

/**
 * @defgroup ivf_sq_cpp_search_params IVF-SQ index search parameters
 * @{
 */

struct search_params : cuvs::neighbors::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
};

static_assert(std::is_aggregate_v<search_params>);

/**
 * @}
 */

/**
 * @defgroup ivf_sq_cpp_list_spec IVF-SQ list storage spec
 * @{
 */

template <typename SizeT, typename CodeT, typename IdxT>
struct CUVS_EXPORT list_spec {
  static_assert(std::is_same_v<CodeT, uint8_t>, "IVF-SQ code type CodeT must be uint8_t");

  using value_type   = CodeT;
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

  template <typename OtherSizeT>
  constexpr explicit list_spec(const list_spec<OtherSizeT, CodeT, IdxT>& other_spec)
    : dim{other_spec.dim}, align_min{other_spec.align_min}, align_max{other_spec.align_max}
  {
  }

  static constexpr uint32_t kVecLen = 16;

  constexpr auto make_list_extents(SizeT n_rows) const -> list_extents
  {
    uint32_t padded = raft::round_up_safe<uint32_t>(dim, kVecLen);
    return raft::make_extents<SizeT>(raft::round_up_safe<SizeT>(n_rows, SizeT{kIndexGroupSize}),
                                     padded);
  }
};

template <typename CodeT, typename IdxT, typename SizeT = uint32_t>
using list_data = ivf::list<list_spec, SizeT, CodeT, IdxT>;

/**
 * @}
 */

/**
 * @defgroup ivf_sq_cpp_index IVF-SQ index
 * @{
 */

/**
 * @brief IVF-SQ index.
 *
 * In the IVF-SQ index, a database vector is first assigned to the nearest cluster center
 * using an inverted file (IVF) structure, and then compressed using scalar quantization (SQ).
 *
 * Scalar quantization independently maps each dimension of the per-cluster residual (the input
 * vector minus its assigned centroid) to a fixed-width integer code. For 8-bit quantization
 * (`uint8_t`), each residual component is linearly mapped to an integer in [0, 255] using learned
 * per-dimension minimum (`sq_vmin`) and step-size (`sq_delta`) values.
 *
 * For a vector component `x_i`, centroid component `centroid_i`, residual minimum `vmin_i`, and
 * quantization step `delta_i`, the stored code is:
 *
 * $$code_i = clamp(round((x_i - centroid_i - vmin_i) / delta_i), 0, 255)$$
 *
 * The corresponding reconstructed component is:
 *
 * $$x_i \approx centroid_i + vmin_i + code_i \cdot delta_i$$
 *
 * This provides a compact representation (1 byte per dimension) while preserving the relative
 * distances between vectors with high fidelity, offering a good trade-off between index size,
 * search speed, and recall compared to flat (uncompressed) and product-quantized (PQ)
 * representations.
 *
 * Note: `CodeT` is the storage type for scalar-quantized residual codes in the inverted lists, not
 * the input dataset type. The public build and search APIs accept `float` and `half` input vectors,
 * then store the quantized residual components as `CodeT` inside the index. Currently, IVF-SQ
 * supports only `uint8_t` codes, so use `CodeT = uint8_t`. Each code uses the full 8-bit range
 * [0, 255].
 *
 * @tparam CodeT SQ code storage type. Must be `uint8_t` in the current implementation.
 *
 */
template <typename CodeT>
struct CUVS_EXPORT index : cuvs::neighbors::index {
  static_assert(std::is_same_v<CodeT, uint8_t>, "IVF-SQ code type CodeT must be uint8_t for now.");

  using index_params_type  = ivf_sq::index_params;
  using search_params_type = ivf_sq::search_params;
  using code_type          = CodeT;

  static constexpr uint32_t sq_bits = sizeof(CodeT) * 8;

 public:
  index(const index&)            = delete;
  index(index&&)                 = default;
  index& operator=(const index&) = delete;
  index& operator=(index&&)      = default;
  ~index()                       = default;

  index(raft::resources const& res);
  index(raft::resources const& res, const index_params& params, uint32_t dim);
  index(raft::resources const& res,
        cuvs::distance::DistanceType metric,
        uint32_t n_lists,
        uint32_t dim,
        bool conservative_memory_allocation);

  cuvs::distance::DistanceType metric() const noexcept;
  int64_t size() const noexcept;
  uint32_t dim() const noexcept;
  uint32_t n_lists() const noexcept;
  bool conservative_memory_allocation() const noexcept;

  raft::device_vector_view<uint32_t, uint32_t> list_sizes() noexcept;
  raft::device_vector_view<const uint32_t, uint32_t> list_sizes() const noexcept;

  raft::device_matrix_view<float, uint32_t, raft::row_major> centers() noexcept;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers() const noexcept;

  std::optional<raft::device_vector_view<float, uint32_t>> center_norms() noexcept;
  std::optional<raft::device_vector_view<const float, uint32_t>> center_norms() const noexcept;
  void allocate_center_norms(raft::resources const& res);

  raft::device_vector_view<float, uint32_t> sq_vmin() noexcept;
  raft::device_vector_view<const float, uint32_t> sq_vmin() const noexcept;

  raft::device_vector_view<float, uint32_t> sq_delta() noexcept;
  raft::device_vector_view<const float, uint32_t> sq_delta() const noexcept;

  raft::host_vector_view<int64_t, uint32_t> accum_sorted_sizes() noexcept;
  [[nodiscard]] raft::host_vector_view<const int64_t, uint32_t> accum_sorted_sizes() const noexcept;

  raft::device_vector_view<CodeT*, uint32_t> data_ptrs() noexcept;
  raft::device_vector_view<CodeT* const, uint32_t> data_ptrs() const noexcept;

  raft::device_vector_view<int64_t*, uint32_t> inds_ptrs() noexcept;
  raft::device_vector_view<int64_t* const, uint32_t> inds_ptrs() const noexcept;

  std::vector<std::shared_ptr<list_data<CodeT, int64_t>>>& lists() noexcept;
  const std::vector<std::shared_ptr<list_data<CodeT, int64_t>>>& lists() const noexcept;

  void check_consistency();

 private:
  cuvs::distance::DistanceType metric_;
  bool conservative_memory_allocation_;

  std::vector<std::shared_ptr<list_data<CodeT, int64_t>>> lists_;
  raft::device_vector<uint32_t, uint32_t> list_sizes_;
  raft::device_matrix<float, uint32_t, raft::row_major> centers_;
  std::optional<raft::device_vector<float, uint32_t>> center_norms_;
  raft::device_vector<float, uint32_t> sq_vmin_;
  raft::device_vector<float, uint32_t> sq_delta_;

  raft::device_vector<CodeT*, uint32_t> data_ptrs_;
  raft::device_vector<int64_t*, uint32_t> inds_ptrs_;
  raft::host_vector<int64_t, uint32_t> accum_sorted_sizes_;
};

/**
 * @}
 */

/**
 * @defgroup ivf_sq_cpp_index_build IVF-SQ index build
 * @{
 */

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2SqrtExpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_sq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_sq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-sq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_sq::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_sq::index<uint8_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_sq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_sq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-sq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_sq::index_params& index_params,
           raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_sq::index<uint8_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_sq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_sq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-sq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_sq::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_sq::index<uint8_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_sq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_sq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-sq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_sq::index_params& index_params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_sq::index<uint8_t>;

/**
 * @}
 */

/**
 * @defgroup ivf_sq_cpp_index_extend IVF-SQ index extend
 * @{
 */

/**
 * @brief Extend the index with the new data in-place.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_sq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_sq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const int64_t, int64_t>> no_op = std::nullopt;
 *   ivf_sq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to ivf_sq::index
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_sq::index<uint8_t>* idx);

/**
 * @brief Extend the index with the new data in-place.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_sq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_sq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const int64_t, int64_t>> no_op = std::nullopt;
 *   ivf_sq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to ivf_sq::index
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_sq::index<uint8_t>* idx);

/**
 * @brief Extend the index with the new data in-place.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_sq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_sq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const int64_t, int64_t>> no_op = std::nullopt;
 *   ivf_sq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to ivf_sq::index
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_sq::index<uint8_t>* idx);

/**
 * @brief Extend the index with the new data in-place.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_sq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_sq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const int64_t, int64_t>> no_op = std::nullopt;
 *   ivf_sq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to ivf_sq::index
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_sq::index<uint8_t>* idx);

/**
 * @}
 */

/**
 * @defgroup ivf_sq_cpp_index_search IVF-SQ index search
 * @{
 */

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_sq::build](#ivf_sq::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 * The exact size of the temporary buffer depends on multiple factors and is an implementation
 * detail. However, you can safely specify a small initial size for the memory pool, so that only a
 * few allocations happen to grow it during the first invocations of the `search`.
 *
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default search parameters
 *   ivf_sq::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_sq::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_sq::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_sq::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 * @endcode
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] index ivf-sq constructed index
 * @param[in] queries raft::device_matrix_view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors raft::device_matrix_view to the indices of the neighbors in the source
 * dataset [n_queries, k]
 * @param[out] distances raft::device_matrix_view to the distances to the selected neighbors
 * [n_queries, k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_sq::search_params& params,
            const cuvs::neighbors::ivf_sq::index<uint8_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index with half-precision queries.
 *
 * See the [ivf_sq::build](#ivf_sq::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 *
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default search parameters
 *   ivf_sq::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_sq::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_sq::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_sq::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 * @endcode
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] index ivf-sq constructed index
 * @param[in] queries raft::device_matrix_view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors raft::device_matrix_view to the indices of the neighbors in the source
 * dataset [n_queries, k]
 * @param[out] distances raft::device_matrix_view to the distances to the selected neighbors
 * [n_queries, k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_sq::search_params& params,
            const cuvs::neighbors::ivf_sq::index<uint8_t>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @}
 */

/**
 * @defgroup ivf_sq_cpp_index_serialize IVF-SQ index serialize
 * @{
 */

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_sq.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = ivf_sq::build(...);`
 * cuvs::neighbors::ivf_sq::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-SQ index
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::ivf_sq::index<uint8_t>& index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_sq.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an empty index with `ivf_sq::index<uint8_t> index(handle);`
 * cuvs::neighbors::ivf_sq::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index IVF-SQ index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_sq::index<uint8_t>* index);

/**
 * @}
 */

}  // namespace ivf_sq
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
