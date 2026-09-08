/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>

#include <cuvs/core/cuda_fp16.hpp>
#include <cuvs/core/export.hpp>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace brute_force {

struct index_params : cuvs::neighbors::index_params {};

struct search_params : cuvs::neighbors::search_params {};

/**
 * @defgroup bruteforce_cpp_index Bruteforce index
 * @{
 */
/**
 * @brief Brute Force index.
 *
 * The index stores the dataset and norms for the dataset in device memory.
 *
 * @tparam T data element type
 */
template <typename T, typename DistT = T>
struct CUVS_EXPORT index : cuvs::neighbors::index {
  using index_params_type  = brute_force::index_params;
  using search_params_type = brute_force::search_params;
  using index_type         = int64_t;
  using value_type         = T;

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
  index(raft::resources const& handle);

  /** Construct a brute force index from dataset
   *
   * Constructs a brute force index from a dataset. This lets us precompute norms for
   * the dataset, providing a speed benefit over doing this at query time.
   * This index will copy the host dataset onto the device, and take ownership of any
   * precaculated norms.
   */
  index(raft::resources const& res,
        raft::host_matrix_view<const T, int64_t, raft::row_major> dataset_view,
        std::optional<raft::device_vector<DistT, int64_t>>&& norms,
        cuvs::distance::DistanceType metric,
        DistT metric_arg = 0.0);

  /** Construct a brute force index and take ownership of a row-major device dataset. */
  index(raft::resources const& res,
        raft::device_matrix<T, int64_t, raft::row_major>&& dataset,
        std::optional<raft::device_vector<DistT, int64_t>>&& norms,
        cuvs::distance::DistanceType metric,
        DistT metric_arg = 0.0);

  /** Construct a brute force index from dataset
   *
   * Constructs a brute force index from a dataset. This lets us precompute norms for
   * the dataset, providing a speed benefit over doing this at query time.
   * This index will store a non-owning reference to the dataset, but will move
   * any norms supplied.
   */
  index(raft::resources const& res,
        raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
        std::optional<raft::device_vector<DistT, int64_t>>&& norms,
        cuvs::distance::DistanceType metric,
        DistT metric_arg = 0.0);

  /** Construct a brute force index from dataset
   *
   * This class stores a non-owning reference to the dataset and norms.
   * Having precomputed norms gives us a performance advantage at query time.
   */
  index(raft::resources const& res,
        raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
        std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view,
        cuvs::distance::DistanceType metric,
        DistT metric_arg = 0.0);

  /** Construct a brute force index from dataset
   *
   * Constructs a brute force index from a dataset. This lets us precompute norms for
   * the dataset, providing a speed benefit over doing this at query time.
   * This index will store a non-owning reference to the dataset, but will move
   * any norms supplied.
   */
  index(raft::resources const& res,
        raft::device_matrix_view<const T, int64_t, raft::col_major> dataset_view,
        std::optional<raft::device_vector<DistT, int64_t>>&& norms,
        cuvs::distance::DistanceType metric,
        DistT metric_arg = 0.0);

  /** Construct a brute force index from dataset
   *
   * This class stores a non-owning reference to the dataset and norms, with
   * the dataset being supplied on device in a col_major format
   */
  index(raft::resources const& res,
        raft::device_matrix_view<const T, int64_t, raft::col_major> dataset_view,
        std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view,
        cuvs::distance::DistanceType metric,
        DistT metric_arg = 0.0);

  /**
   * Replace the dataset with a new dataset.
   */
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t, raft::row_major> dataset);

  /**
   * Replace the dataset with a new dataset.
   *
   * We create a copy of the dataset on the device. The index manages the lifetime of this copy.
   */
  void update_dataset(raft::resources const& res,
                      raft::host_matrix_view<const T, int64_t, raft::row_major> dataset);

  /** Distance metric used for retrieval */
  cuvs::distance::DistanceType metric() const noexcept { return metric_; }

  /** Metric argument */
  DistT metric_arg() const noexcept { return metric_arg_; }

  /** Total length of the index (number of vectors). */
  size_t size() const noexcept { return dataset_view_.extent(0); }

  /** Dimensionality of the data. */
  size_t dim() const noexcept { return dataset_view_.extent(1); }

  /** Dataset [size, dim] */
  raft::device_matrix_view<const T, int64_t, raft::row_major> dataset() const noexcept
  {
    return dataset_view_;
  }

  /** Dataset norms */
  raft::device_vector_view<const DistT, int64_t, raft::row_major> norms() const
  {
    return norms_view_.value();
  }

  /** Whether ot not this index has dataset norms */
  inline bool has_norms() const noexcept { return norms_view_.has_value(); }

 private:
  cuvs::distance::DistanceType metric_;
  raft::device_matrix<T, int64_t, raft::row_major> dataset_;
  std::optional<raft::device_vector<DistT, int64_t>> norms_;
  std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view_;
  raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view_;
  DistT metric_arg_;
};

/**
 * @}
 */

/**
 * @defgroup bruteforce_cpp_index_build Bruteforce index build
 * @{
 */
/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // create and fill the index from a [N, D] dataset
 *   brute_force::index_params index_params;
 *   auto index = brute_force::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params parameters such as the distance metric to use
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed brute-force index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::brute_force::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::brute_force::index<float, float>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * @param[in] handle
 * @param[in] index_params parameters such as the distance metric to use
 * @param[in] dataset a host pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed brute-force index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::brute_force::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::brute_force::index<float, float>;

[[deprecated]] auto build(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded,
  float metric_arg                    = 0) -> cuvs::neighbors::brute_force::index<float, float>;
/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // create and fill the index from a [N, D] dataset
 *   brute_force::index_params index_params;
 *   auto index = brute_force::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params parameters such as the distance metric to use
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed brute force index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::brute_force::index_params& index_params,
           raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::brute_force::index<half, float>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * @param[in] handle
 * @param[in] index_params parameters such as the distance metric to use
 * @param[in] dataset a host pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed brute-force index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::brute_force::index_params& index_params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::brute_force::index<half, float>;

[[deprecated]] auto build(
  raft::resources const& handle,
  raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded,
  float metric_arg                    = 0) -> cuvs::neighbors::brute_force::index<half, float>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   brute_force::index_params index_params;
 *   auto index = brute_force::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params parameters such as the distance metric to use
 * @param[in] dataset a device pointer to a col-major matrix [n_rows, dim]
 *
 * @return the constructed brute force index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::brute_force::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::col_major> dataset)
  -> cuvs::neighbors::brute_force::index<float, float>;

[[deprecated]] auto build(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::col_major> dataset,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded,
  float metric_arg                    = 0) -> cuvs::neighbors::brute_force::index<float, float>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   brute_force::index_params index_params;
 *   auto index = brute_force::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params parameters such as the distance metric to use
 * @param[in] dataset a device pointer to a col-major matrix [n_rows, dim]
 *
 * @return the constructed brute force index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::brute_force::index_params& index_params,
           raft::device_matrix_view<const half, int64_t, raft::col_major> dataset)
  -> cuvs::neighbors::brute_force::index<half, float>;

[[deprecated]] auto build(
  raft::resources const& handle,
  raft::device_matrix_view<const half, int64_t, raft::col_major> dataset,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded,
  float metric_arg                    = 0) -> cuvs::neighbors::brute_force::index<half, float>;
/**
 * @}
 */

/**
 * @defgroup bruteforce_cpp_index_search Bruteforce index search
 * @{
 */
/**
 * @brief Search ANN using the constructed index.
 *
 * See the [brute_force::build](#brute_force::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *
 *   // use default index parameters
 *   brute_force::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   brute_force::index_params index_params;
 *   auto index = brute_force::build(handle, index_params, dataset);
 *   // use default search parameters
 *   brute_force::search_params search_params;
 *   // create a bitset to filter the search
 *   auto removed_indices = raft::make_device_vector<int64_t, int64_t>(res, n_removed_indices);
 *   raft::core::bitset<std::uint32_t, int64_t> removed_indices_bitset(
 *     res, removed_indices.view(), dataset.extent(0));
 *   // search K nearest neighbours according to a bitset
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   auto filter    = filtering::bitset_filter(removed_indices_bitset.view());
 *   brute_force::search(res, search_params, index, queries, neighbors, distances, filter);
 * @endcode
 *
 * @param[in] handle
 * @param[in] params parameters configuring the search
 * @param[in] index brute-force constructed index
 * @param[in] queries a device pointer to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device pointer to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device pointer to the distances to the selected neighbors [n_queries, k]
 * @param[in] sample_filter An optional device filter that restricts which dataset elements should
 * be considered for each query.
 *
 * - Supports two types of filters:
 *   1. **Bitset Filter**: A shared filter where each bit corresponds to a dataset element.
 *      All queries share the same filter, with a logical shape of `[1, index->size()]`.
 *   2. **Bitmap Filter**: A per-query filter with a logical shape of `[n_queries, index->size()]`,
 *      where each bit indicates whether a specific dataset element should be considered for a
 *      particular query. (1 for inclusion, 0 for exclusion).
 *
 * - The default value is `none_sample_filter`, which applies no filtering.
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::brute_force::search_params& params,
            const cuvs::neighbors::brute_force::index<float, float>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

[[deprecated]] void search(raft::resources const& handle,
                           const cuvs::neighbors::brute_force::index<float, float>& index,
                           raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
                           raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                           raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                           const cuvs::neighbors::filtering::base_filter& sample_filter =
                             cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [brute_force::build](#brute_force::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *
 *   // use default index parameters
 *   brute_force::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   brute_force::index_params index_params;
 *   auto index = brute_force::build(handle, index_params, dataset);
 *   // use default search parameters
 *   brute_force::search_params search_params;
 *   // create a bitset to filter the search
 *   auto removed_indices = raft::make_device_vector<int64_t, int64_t>(res, n_removed_indices);
 *   raft::core::bitset<std::uint32_t, int64_t> removed_indices_bitset(
 *     res, removed_indices.view(), dataset.extent(0));
 *   // search K nearest neighbours according to a bitset
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<half>(res, n_queries, k);
 *   auto filter    = filtering::bitset_filter(removed_indices_bitset.view());
 *   brute_force::search(res, search_params, index, queries, neighbors, distances, filter);
 * @endcode
 *
 * @param[in] handle
 * @param[in] params parameters configuring the search
 * @param[in] index ivf-flat constructed index
 * @param[in] queries a device pointer to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device pointer to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device pointer to the distances to the selected neighbors [n_queries, k]
 * @param[in] sample_filter An optional device filter that restricts which dataset elements should
 * be considered for each query.
 *
 * - Supports two types of filters:
 *   1. **Bitset Filter**: A shared filter where each bit corresponds to a dataset element.
 *      All queries share the same filter, with a logical shape of `[1, index->size()]`.
 *   2. **Bitmap Filter**: A per-query filter with a logical shape of `[n_queries, index->size()]`,
 *      where each bit indicates whether a specific dataset element should be considered for a
 *      particular query. (1 for inclusion, 0 for exclusion).
 *
 * - The default value is `none_sample_filter`, which applies no filtering.
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::brute_force::search_params& params,
            const cuvs::neighbors::brute_force::index<half, float>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

[[deprecated]] void search(raft::resources const& handle,
                           const cuvs::neighbors::brute_force::index<half, float>& index,
                           raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
                           raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                           raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                           const cuvs::neighbors::filtering::base_filter& sample_filter =
                             cuvs::neighbors::filtering::none_sample_filter{});
/**
 * @brief Search ANN using the constructed index.
 *
 * See the [brute_force::build](#brute_force::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *
 *   // use default index parameters
 *   brute_force::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   brute_force::index_params index_params;
 *   auto index = brute_force::build(handle, index_params, dataset);
 *   // use default search parameters
 *   brute_force::search_params search_params;
 *   // create a bitset to filter the search
 *   auto removed_indices = raft::make_device_vector<int64_t, int64_t>(res, n_removed_indices);
 *   raft::core::bitset<std::uint32_t, int64_t> removed_indices_bitset(
 *     res, removed_indices.view(), dataset.extent(0));
 *   // search K nearest neighbours according to a bitset
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   auto filter    = filtering::bitset_filter(removed_indices_bitset.view());
 *   brute_force::search(res, search_params, index, queries, neighbors, distances, filter);
 * @endcode
 *
 * @param[in] handle
 * @param[in] params parameters configuring the search
 * @param[in] index bruteforce constructed index
 * @param[in] queries a device pointer to a col-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device pointer to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device pointer to the distances to the selected neighbors [n_queries, k]
 * @param[in] sample_filter An optional device filter that restricts which dataset elements should
 * be considered for each query.
 *
 * - Supports two types of filters:
 *   1. **Bitset Filter**: A shared filter where each bit corresponds to a dataset element.
 *      All queries share the same filter, with a logical shape of `[1, index->size()]`.
 *   2. **Bitmap Filter**: A per-query filter with a logical shape of `[n_queries, index->size()]`,
 *      where each bit indicates whether a specific dataset element should be considered for a
 *      particular query. (1 for inclusion, 0 for exclusion).
 *
 * - The default value is `none_sample_filter`, which applies no filtering.
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::brute_force::search_params& params,
            const cuvs::neighbors::brute_force::index<float, float>& index,
            raft::device_matrix_view<const float, int64_t, raft::col_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

[[deprecated]] void search(raft::resources const& handle,
                           const cuvs::neighbors::brute_force::index<float, float>& index,
                           raft::device_matrix_view<const float, int64_t, raft::col_major> queries,
                           raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                           raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                           const cuvs::neighbors::filtering::base_filter& sample_filter =
                             cuvs::neighbors::filtering::none_sample_filter{});
/**
 * @brief Search ANN using the constructed index.
 *
 * See the [brute_force::build](#brute_force::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *
 *   // use default index parameters
 *   brute_force::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   brute_force::index_params index_params;
 *   auto index = brute_force::build(handle, index_params, dataset);
 *   // use default search parameters
 *   brute_force::search_params search_params;
 *   // create a bitset to filter the search
 *   auto removed_indices = raft::make_device_vector<int64_t, int64_t>(res, n_removed_indices);
 *   raft::core::bitset<std::uint32_t, int64_t> removed_indices_bitset(
 *     res, removed_indices.view(), dataset.extent(0));
 *   // search K nearest neighbours according to a bitset
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<half>(res, n_queries, k);
 *   auto filter    = filtering::bitset_filter(removed_indices_bitset.view());
 *   brute_force::search(res, search_params, index, queries, neighbors, distances, filter);
 * @endcode
 *
 * @param[in] handle
 * @param[in] params parameters configuring the search
 * @param[in] index bruteforce constructed index
 * @param[in] queries a device pointer to a col-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device pointer to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device pointer to the distances to the selected neighbors [n_queries, k]
 * @param[in] sample_filter An optional device filter that restricts which dataset elements should
 * be considered for each query.
 *
 * - Supports two types of filters:
 *   1. **Bitset Filter**: A shared filter where each bit corresponds to a dataset element.
 *      All queries share the same filter, with a logical shape of `[1, index->size()]`.
 *   2. **Bitmap Filter**: A per-query filter with a logical shape of `[n_queries, index->size()]`,
 *      where each bit indicates whether a specific dataset element should be considered for a
 *      particular query. (1 for inclusion, 0 for exclusion).
 *
 * - The default value is `none_sample_filter`, which applies no filtering.
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::brute_force::search_params& params,
            const cuvs::neighbors::brute_force::index<half, float>& index,
            raft::device_matrix_view<const half, int64_t, raft::col_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

[[deprecated]] void search(raft::resources const& handle,
                           const cuvs::neighbors::brute_force::index<half, float>& index,
                           raft::device_matrix_view<const half, int64_t, raft::col_major> queries,
                           raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                           raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                           const cuvs::neighbors::filtering::base_filter& sample_filter =
                             cuvs::neighbors::filtering::none_sample_filter{});
/**
 * @}
 */

/**
 * @defgroup sparse_bruteforce_cpp_index Sparse Brute Force index
 * @{
 */
/**
 * @brief Sparse Brute Force index.
 *
 * @tparam T Data element type
 * @tparam IdxT Index element type
 */
template <typename T, typename IdxT>
struct sparse_index {
 public:
  sparse_index(const sparse_index&)            = delete;
  sparse_index(sparse_index&&)                 = default;
  sparse_index& operator=(const sparse_index&) = delete;
  sparse_index& operator=(sparse_index&&)      = default;
  ~sparse_index()                              = default;

  /** Construct a sparse brute force sparse_index from dataset */
  sparse_index(raft::resources const& res,
               raft::device_csr_matrix_view<const T, IdxT, IdxT, IdxT> dataset,
               cuvs::distance::DistanceType metric,
               T metric_arg);

  /** Distance metric used for retrieval */
  cuvs::distance::DistanceType metric() const noexcept { return metric_; }

  /** Metric argument */
  T metric_arg() const noexcept { return metric_arg_; }

  raft::device_csr_matrix_view<const T, IdxT, IdxT, IdxT> dataset() const noexcept
  {
    return dataset_;
  }

 private:
  raft::device_csr_matrix_view<const T, IdxT, IdxT, IdxT> dataset_;
  cuvs::distance::DistanceType metric_;
  T metric_arg_;
};
/**
 * @}
 */

/**
 * @defgroup sparse_bruteforce_cpp_index_build Sparse Brute Force index build
 * @{
 */

/*
 * @brief Build the Sparse index from the dataset
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // create and fill the index from a CSR dataset
 *   auto index = brute_force::build(handle, dataset, metric);
 * @endcode
 *
 * @param[in] handle
 * @param[in] dataset A sparse CSR matrix in device memory to search against
 * @param[in] metric cuvs::distance::DistanceType
 * @param[in] metric_arg metric argument
 *
 * @return the constructed Sparse brute-force index
 */
auto build(raft::resources const& handle,
           raft::device_csr_matrix_view<const float, int, int, int> dataset,
           cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded,
           float metric_arg = 0) -> cuvs::neighbors::brute_force::sparse_index<float, int>;
/**
 * @}
 */

/**
 * @defgroup sparse_bruteforce_cpp_index_search Sparse Brute Force index search
 * @{
 */
struct sparse_search_params {
  int batch_size_index = 2 << 14;
  int batch_size_query = 2 << 14;
};

/*
 * @brief Search the sparse bruteforce index for nearest neighbors
 *
 * @param[in] handle
 * @param[in] params parameters configuring the search
 * @param[in] index Sparse brute-force constructed index
 * @param[in] queries a sparse CSR matrix on the device to query
 * @param[out] neighbors a device pointer to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device pointer to the distances to the selected neighbors [n_queries, k]
 */
void search(raft::resources const& handle,
            const sparse_search_params& params,
            const sparse_index<float, int>& index,
            raft::device_csr_matrix_view<const float, int, int, int> dataset,
            raft::device_matrix_view<int, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);
/**
 * @}
 */

/**
 * @defgroup bruteforce_cpp_index_serialize Bruteforce index serialize functions
 * @{
 */
/**
 * Save the index to file.
 * The serialization format can be subject to changes, therefore loading
 * an index saved with a previous version of cuvs is not guaranteed
 * to work.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/brute_force.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = brute_force::build(...);`
 * cuvs::neighbors::brute_force::serialize(handle, filename, index);
 * @endcode
 *
 * @tparam T data element type
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index brute force index
 * @param[in] include_dataset whether to include the dataset in the serialized
 * output
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::brute_force::index<half, float>& index,
               bool include_dataset = true);
/**
 * Save the index to file.
 * The serialization format can be subject to changes, therefore loading
 * an index saved with a previous version of cuvs is not guaranteed
 * to work.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/brute_force.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = brute_force::build(...);`
 * cuvs::neighbors::brute_force::serialize(handle, filename, index);
 * @endcode
 *
 * @tparam T data element type
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index brute force index
 * @param[in] include_dataset whether to include the dataset in the serialized
 * output
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::brute_force::index<float, float>& index,
               bool include_dataset = true);

/**
 * Write the index to an output stream
 * The serialization format can be subject to changes, therefore loading
 * an index saved with a previous version of cuvs is not guaranteed
 * to work.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/brute_force.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = cuvs::neighbors::brute_force::build(...);`
 * cuvs::neighbors::brute_force::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index brute force index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::brute_force::index<half, float>& index,
               bool include_dataset = true);

/**
 * Write the index to an output stream
 * The serialization format can be subject to changes, therefore loading
 * an index saved with a previous version of cuvs is not guaranteed
 * to work.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/brute_force.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = cuvs::neighbors::brute_force::build(...);`
 * cuvs::neighbors::brute_force::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index brute force index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::brute_force::index<float, float>& index,
               bool include_dataset = true);

/**
 * Load index from file.
 * The serialization format can be subject to changes, therefore loading
 * an index saved with a previous version of cuvs is not guaranteed
 * to work.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/brute_force.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using T    = half; // data element type
 * brute_force::index<T, float> index(handle);
 * cuvs::neighbors::brute_force::deserialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index brute force index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::brute_force::index<half, float>* index);
/**
 * Load index from file.
 * The serialization format can be subject to changes, therefore loading
 * an index saved with a previous version of cuvs is not guaranteed
 * to work.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/brute_force.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using T    = float; // data element type
 * brute_force::index<T, float> index(handle);
 * cuvs::neighbors::brute_force::deserialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index brute force index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::brute_force::index<float, float>* index);
/**
 * Load index from input stream
 * The serialization format can be subject to changes, therefore loading
 * an index saved with a previous version of cuvs is not guaranteed
 * to work.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/brute_force.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * using T    = half; // data element type
 * brute_force::index<T, float> index(handle);
 * cuvs::neighbors::brute_force::deserialize(handle, is, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[out] index brute force index
 *
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::brute_force::index<half, float>* index);
/**
 * Load index from input stream
 * The serialization format can be subject to changes, therefore loading
 * an index saved with a previous version of cuvs is not guaranteed
 * to work.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/brute_force.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * using T    = float; // data element type
 * brute_force::index<T, float> index(handle);
 * cuvs::neighbors::brute_force::deserialize(handle, is, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[out] index brute force index
 *
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::brute_force::index<float, float>* index);
/**
 * @}
 */

}  // namespace brute_force
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
/** Specialized parameters utilizing brute force to build knn graph */
namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace graph_build_params {
struct brute_force_params {
  cuvs::neighbors::brute_force::index_params build_params;
  cuvs::neighbors::brute_force::search_params search_params;
};
}  // namespace graph_build_params
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
