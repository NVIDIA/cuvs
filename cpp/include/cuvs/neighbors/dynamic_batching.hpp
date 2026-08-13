/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/neighbors/common.hpp>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace dynamic_batching {

namespace detail {
template <typename T, typename IdxT>
class batch_runner;
}

/**
 * @defgroup dynamic_batching_cpp_index_params Dynamic Batching index parameters
 * @{
 */
struct index_params : cuvs::neighbors::index_params {
  /** The number of neighbors to search is fixed at construction time. */
  int64_t k;
  /** Maximum size of the batch to submit to the upstream index. */
  int64_t max_batch_size = 100;
  /**
   * The number of independent request queues.
   *
   * Each queue is associated with a unique CUDA stream and IO device buffers. If the number of
   * concurrent requests is high, using multiple queues allows to fill-in data and prepare the batch
   * while the other queue is busy. Moreover, the queues are submitted concurrently; this allows to
   * better utilize the GPU by hiding the kernel launch latencies, which helps to improve the
   * throughput.
   */
  size_t n_queues = 3;
  /**
   * By default (`conservative_dispatch = false`) the first CPU thread to commit a query to a batch
   * dispatches the upstream search function as soon as possible (before the batch is full). In that
   * case, it does not know the final batch size at the time of calling the upstream search and thus
   * runs the upstream search with the maximum batch size every time, even if only one valid query
   * is present in the batch. This reduces the latency at the cost of wasted GPU resources.
   *
   * The alternative behavaior (`conservative_dispatch = true`) is more conservative: the dispatcher
   * thread starts the kernel that gathers input queries, but waits till the batch is full or the
   * waiting time is exceeded. Only then it acquires the actual batch size and launches the upstream
   * search. As a result, less GPU resources are wasted at the cost of exposing upstream search
   * latency.
   *
   * *Rule of Thumb*:
   *    for a large `max_batch_size` set `conservative_dispatch = true`, otherwise keep it disabled.
   */
  bool conservative_dispatch = false;
};
/** @} */

/**
 * @defgroup dynamic_batching_cpp_search_params Dynamic Batching search parameters
 * @{
 */
struct search_params : cuvs::neighbors::search_params {
  /**
   * How long a request can stay in the queue (milliseconds).
   * Note, this only affects the dispatch time and does not reflect full request latency;
   * the latter depends on the upstream search parameters and the batch size.
   */
  double dispatch_timeout_ms = 1.0;
};
/** @} */

/**
 * @defgroup dynamic_batching_cpp_index Dynamic Batching index type
 * @{
 */

/**
 * @brief Lightweight dynamic batching index wrapper
 *
 * @tparam T data type
 * @tparam IdxT index type
 *
 * One lightweight dynamic batching index manages a single index and a single search parameter set.
 * This structure should be shared among multiple users via copy semantics: access to the
 * underlying implementation is managed via a shared pointer, and concurrent search among the
 * participants is thread-safe.
 *
 * __Usage example__
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // When creating a dynamic batching index, k parameter has to be passed explicitly.
 *   // The first empty braces default-initialize the parent `neighbors::index_params` (unused).
 *   dynamic_batching::index_params dynb_index_params{{}, k};
 *   // Construct the index by wrapping the upstream index and search parameters.
 *   dynamic_batching::index<float, uint32_t> index{
 *       res, dynb_index_params, upstream_index, upstream_search_params
 *   };
 *   // Use default search parameters
 *   dynamic_batching::search_params search_params;
 *   // Search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   dynamic_batching::search(
 *       res, search_params, index, queries, neighbors.view(), distances.view()
 *   );
 * @endcode
 *
 *
 * __Priority queues__
 *
 * The dynamic batching index has a limited support for prioritizing individual requests.
 * There's only one pool of queues in the batcher and no functionality to prioritize one bach over
 * the other. The `search_params::dispatch_timeout_ms` parameters passed in each request are
 * aggregated internally and the batch is dispatched no later than any of the timeouts is exceeded.
 * In this logic, a high-priority request can never be processed earlier than any lower-priority
 * requests submitted earlier.
 *
 * However, dynamic batching indexes are lightweight and do not contain any global or static state.
 * This means it's easy to combine multiple batchers.
 * As an example, you can construct one batching index per priority class:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // Large batch size (128), couple queues (2),
 *   //   enabled conservative dispatch - all for better throughput
 *   dynamic_batching::index_params low_priority_params{{}, k, 128, 2, true};
 *   // Small batch size (16), more queues (4),
 *   //   disabled conservative dispatch - to minimize latency with reasonable throughput
 *   dynamic_batching::index_params high_priority_params{{}, k, 16, 4, false};
 *   // Construct the indexes by wrapping the upstream index and search parameters.
 *   dynamic_batching::index<float, uint32_t> low_priority_index{
 *       res, low_priority_params, upstream_index, upstream_search_params
 *   };
 *   dynamic_batching::index<float, uint32_t> high_priority_index{
 *       res, high_priority_params, upstream_index, upstream_search_params
 *   };
 *   // Define a combined search function with priority selection
 *   double high_priority_threshold_ms = 0.1;
 *   auto search_function =
 *      [low_priority_index, high_priority_index, high_priority_threshold_ms](
 *        raft::resources const &res,
 *        dynamic_batching::search_params search_params,
 *        raft::device_matrix_view<const float, int64_t> queries,
 *        raft::device_matrix_view<uint32_t, int64_t> neighbors,
 *        raft::device_matrix_view<float, int64_t> distances) {
 *      dynamic_batching::search(
 *          res,
 *          search_params,
 *          search_params.dispatch_timeout_ms < high_priority_threshold_ms
 *            ? high_priority_index : low_priority_index,
 *          queries,
 *          neighbors,
 *          distances
 *      );
 *   };
 * @endcode
 */
template <typename T, typename IdxT>
struct index : cuvs::neighbors::index {
  std::shared_ptr<detail::batch_runner<T, IdxT>> runner;

  /**
   * @brief Construct a dynamic batching index by wrapping the upstream index.
   *
   * @tparam Upstream the upstream index type
   *
   * @param[in] res raft resources
   * @param[in] params dynamic batching parameters
   * @param[in] upstream_index the original index to perform the search
   *     (the reference must be alive for the lifetime of the dynamic batching index)
   * @param[in] upstream_params the original index search parameters for all queries in a batch
   *     (the parameters are captured by value for the lifetime of the dynamic batching index)
   * @param[in] sample_filter
   *     filtering function, if any, must be the same for all requests in a batch
   *     (the pointer must be alive for the lifetime of the dynamic batching index)
   */
  template <typename Upstream>
  index(const raft::resources& res,
        const cuvs::neighbors::dynamic_batching::index_params& params,
        const Upstream& upstream_index,
        const typename Upstream::search_params_type& upstream_params,
        const cuvs::neighbors::filtering::base_filter* sample_filter = nullptr);
};
/** @} */

/**
 *
 * @defgroup dynamic_batching_cpp_search Dynamic Batching search
 *
 * @{
 */

/**
 * @brief Search ANN using a dynamic batching index.
 *
 * The search parameters of the upstream index and the optional filtering function are configured at
 * the dynamic batching index construction time.
 *
 * Like with many other indexes, the dynamic batching search has the stream-ordered semantics: the
 * host function may return the control before the results are ready. Synchronize with the main CUDA
 * stream in the given resource object to wait for arrival of the search results.
 *
 * Dynamic batching search is thread-safe: call the search function with copies of the same index in
 * multiple threads to increase the occupancy of the batches.
 *
 * @param[in] res
 * @param[in] params query-specific batching parameters, such as the maximum waiting time
 * @param[in] index a dynamic batching index
 * @param[in] queries a device matrix view to a row-major matrix
 *               [n_queries, dim]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 *               [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors
 *               [n_queries, k]
 *
 */
void search(raft::resources const& res,
            cuvs::neighbors::dynamic_batching::search_params const& params,
            dynamic_batching::index<float, uint32_t> const& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/** @copydoc search */
void search(raft::resources const& res,
            cuvs::neighbors::dynamic_batching::search_params const& params,
            dynamic_batching::index<half, uint32_t> const& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/** @copydoc search */
void search(raft::resources const& res,
            cuvs::neighbors::dynamic_batching::search_params const& params,
            dynamic_batching::index<int8_t, uint32_t> const& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/** @copydoc search */
void search(raft::resources const& res,
            cuvs::neighbors::dynamic_batching::search_params const& params,
            dynamic_batching::index<uint8_t, uint32_t> const& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/** @copydoc search */
void search(raft::resources const& res,
            cuvs::neighbors::dynamic_batching::search_params const& params,
            dynamic_batching::index<float, int64_t> const& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/** @copydoc search */
void search(raft::resources const& res,
            cuvs::neighbors::dynamic_batching::search_params const& params,
            dynamic_batching::index<half, int64_t> const& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/** @copydoc search */
void search(raft::resources const& res,
            cuvs::neighbors::dynamic_batching::search_params const& params,
            dynamic_batching::index<int8_t, int64_t> const& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/** @copydoc search */
void search(raft::resources const& res,
            cuvs::neighbors::dynamic_batching::search_params const& params,
            dynamic_batching::index<uint8_t, int64_t> const& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/** @} */

}  // namespace dynamic_batching
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
