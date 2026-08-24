/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/bitset.hpp>
#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <cuvs/util/file_io.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>

#include <fcntl.h>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace graph_build_params {
using iterative_search_params = cuvs::neighbors::search_params;

/** Specialized parameters for ACE (Augmented Core Extraction) graph build */
struct ace_params {
  /**
   * Number of partitions for ACE (Augmented Core Extraction) partitioned build.
   *
   * When set to 0 (default), the number of partitions is automatically derived
   * based on available host and GPU memory to maximize partition size while
   * ensuring the build fits in memory.
   *
   * Small values might improve recall but potentially degrade performance and
   * increase memory usage. Partitions should not be too small to prevent issues
   * in KNN graph construction. The partition size is on average 2 * (n_rows / npartitions) * dim *
   * sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance in the
   * partition sizes (up to 3x in our tests).
   *
   * If the specified number of partitions results in partitions that exceed
   * available memory, the value will be automatically increased to fit memory
   * constraints and a warning will be issued.
   */
  size_t npartitions = 0;
  /**
   * The index quality for the ACE build.
   *
   * Bigger values increase the index quality. At some point, increasing this will no longer improve
   * the quality.
   */
  size_t ef_construction = 120;
  /**
   * Directory to store ACE build artifacts (e.g., KNN graph, optimized graph).
   *
   * Used when `use_disk` is true or when the graph does not fit in host and GPU
   * memory. This should be the fastest disk in the system and hold enough space
   * for twice the dataset, final graph, and label mapping.
   */
  std::string build_dir = "/tmp/ace_build";
  /**
   * Whether to use disk-based storage for ACE build.
   *
   * When true, enables disk-based operations for memory-efficient graph construction.
   */
  bool use_disk = false;

  /**
   * Maximum host memory to use for ACE build in GiB.
   *
   * When set to 0 (default), uses available host memory.
   * When set to a positive value, limits host memory usage to the specified amount.
   * Useful for testing or when running alongside other memory-intensive processes.
   */
  double max_host_memory_gb = 0;
  /**
   * Maximum GPU memory to use for ACE build in GiB.
   *
   * When set to 0 (default), uses available GPU memory.
   * When set to a positive value, limits GPU memory usage to the specified amount.
   * Useful for testing or when running alongside other memory-intensive processes.
   */
  double max_gpu_memory_gb = 0;

  ace_params() = default;
};

}  // namespace graph_build_params
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace cagra {
// For re-exporting into cagra namespace
namespace graph_build_params = cuvs::neighbors::graph_build_params;
namespace detail {
struct fd_transfer;
}
/**
 * @defgroup cagra_cpp_index_params CAGRA index build parameters
 * @{
 */

using graph_build_params_t = std::variant<std::monostate,
                                          graph_build_params::ivf_pq_params,
                                          graph_build_params::nn_descent_params,
                                          graph_build_params::ace_params,
                                          graph_build_params::iterative_search_params>;

/**
 * @brief A strategy for selecting the graph build parameters based on similar HNSW index
 * parameters.
 *
 * Define how `cagra::index_params::from_hnsw_params` should construct a graph to construct a graph
 * that is to be converted to (used by) a CPU HNSW index.
 */
enum class hnsw_heuristic_type : uint32_t {
  /**
   * Create a graph that is very similar to an HNSW graph in
   * terms of the number of nodes and search performance. Since HNSW produces a variable-degree
   * graph (2M being the max graph degree) and CAGRA produces a fixed-degree graph, there's always a
   * difference in the performance of the two.
   *
   * This function attempts to produce such a graph that the QPS and recall of the two graphs being
   * searched by HNSW are close for any search parameter combination. The CAGRA-produced graph tends
   * to have a "longer tail" on the low recall side (that is being slightly faster and less
   * precise).
   *
   */
  SIMILAR_SEARCH_PERFORMANCE = 0,
  /**
   * Create a graph that has the same binary size as an HNSW graph with the given parameters
   * (`graph_degree = 2 * M`) while trying to match the search performance as closely as possible.
   *
   * The reference HNSW index and the corresponding from-CAGRA generated HNSW index will NOT produce
   * the same recalls and QPS for the same parameter `ef`. The graphs are different internally. For
   * the same `ef`, the from-CAGRA index likely has a slightly higher recall and slightly lower QPS.
   * However, the Recall-QPS curves should be similar (i.e. the points are just shifted along the
   * curve).
   */
  SAME_GRAPH_FOOTPRINT = 1
};

struct index_params : cuvs::neighbors::index_params {
  /** Degree of input graph for pruning. */
  size_t intermediate_graph_degree = 128;
  /** Degree of output graph. */
  size_t graph_degree = 64;
  /** Parameters for graph building.
   *
   * Set ivf_pq_params, nn_descent_params, ace_params, or iterative_search_params to select the
   * graph build algorithm and control their parameters. The default (std::monostate) is to use a
   * heuristic to decide the algorithm and its parameters.
   *
   * @code{.cpp}
   * cagra::index_params params;
   * // 1. Choose IVF-PQ algorithm
   * params.graph_build_params = cagra::graph_build_params::ivf_pq_params(dataset.extent,
   * params.metric);
   *
   * // 2. Choose NN Descent algorithm for kNN graph construction
   * params.graph_build_params =
   * cagra::graph_build_params::nn_descent_params(params.intermediate_graph_degree);
   *
   * // 3. Choose ACE algorithm for graph construction
   * params.graph_build_params = cagra::graph_build_params::ace_params();
   *
   * // 4. Choose iterative graph building using CAGRA's search() and optimize()  [Experimental]
   * params.graph_build_params =
   * cagra::graph_build_params::iterative_search_params();
   * @endcode
   */
  graph_build_params_t graph_build_params;
  /**
   * Whether to use MST optimization to guarantee graph connectivity.
   */
  bool guarantee_connectivity = false;

  /**
   * Whether to attach the dataset to the index after graph construction, i.e.:
   *
   *  - `true` (default) means `build` attaches the input dataset as a **non-owning view** to the
   * index. The caller is responsible for keeping the underlying dataset storage alive for as long
   * as the index is used. A device-backed index is ready to search immediately; a host-backed index
   * retains the dataset for operations such as serialization but is not searchable.
   *  - `false` means `build` only builds the graph and the caller is expected to attach a dataset
   * separately via `cuvs::neighbors::cagra::update_dataset` before searching.
   *
   * Unlike the legacy behavior, no copy of the dataset is made: the index always stores a view.
   * Setting `attach_dataset_on_build = false` is useful when the caller needs to apply specific
   * memory placement or transformation (e.g. moving to managed memory) before attaching.
   *
   * Host indexes are not directly searchable. Call the type-changing `update_dataset` with a
   * user-provided device-padded dataset view to obtain a search-ready `device_padded_index`.
   * Disk-based ACE builds manage file-backed dataset state separately and ignore this flag.
   *
   * @code{.cpp}
   *   auto dataset = cuvs::neighbors::make_device_padded_dataset(res, host_matrix.view());
   *   cagra::index_params index_params;
   *   // Build graph only — caller attaches dataset later.
   *   index_params.attach_dataset_on_build = false;
   *   auto index = cagra::build(res, index_params, dataset->as_dataset_view());
   *   // ASSERT(index.size() == 0);  // no dataset yet
   *   // Attach with a view (storage owned by `dataset`).
   *   index = cagra::update_dataset(res, std::move(index), dataset->as_dataset_view());
   *   cagra::search(res, search_params, index, queries, neighbors, distances);
   * @endcode
   */
  bool attach_dataset_on_build = true;

  /**
   * @brief Select the graph build algorithm and its parameters for a dataset.
   *
   * This is the main CAGRA build heuristic: it chooses between NN-descent and IVF-PQ based on the
   * dataset size and tunes their parameters based on the target intermediate graph degree and the
   * requested build quality. It returns the `graph_build_params` variant only; the caller is
   * responsible for setting `graph_degree` / `intermediate_graph_degree`.
   *
   * @param dataset The shape of the input dataset
   * @param intermediate_graph_degree The intermediate (kNN) graph degree the build should target.
   *  Note: the intermediate graph degree must be not smaller than the output graph degree; a good
   *        practice is to have it 1.5x to 2x of the desired graph_degree and a multiple of 32.
   * @param metric The distance metric to search
   * @param build_quality Higher values increase the build quality (and cost) up to a point.
   *        Any value is valid, but values below 20 are the most practical (default = 7).
   */
  static graph_build_params_t graph_build_heuristic(
    raft::matrix_extent<int64_t> dataset,
    size_t intermediate_graph_degree,
    cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded,
    size_t build_quality                = 7);

  /**
   * @brief Create CAGRA index parameters heuristically tuned for a dataset.
   *
   * Returns default CAGRA `index_params` with `graph_build_params` selected by
   * `graph_build_heuristic` for the given dataset.
   *
   * @param dataset The shape of the input dataset
   * @param graph_degree Degree of the output graph.
   * @param metric The distance metric to search
   * @param build_quality Higher values increase the build quality (and cost) up to a point.
   *        Any value is valid, but values below 20 are the most practical (default = 7).
   *
   * Usage example:
   * @code{.cpp}
   *   using namespace cuvs::neighbors;
   *   raft::resources res;
   *   auto dataset = raft::make_device_matrix<float, int64_t>(res, N, D);
   *   auto cagra_params = cagra::index_params::from_dataset(dataset.extents());
   *   auto cagra_index = cagra::build(res, cagra_params, dataset);
   * @endcode
   */
  static cagra::index_params from_dataset(
    raft::matrix_extent<int64_t> dataset,
    size_t graph_degree                 = 64,
    cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded,
    size_t build_quality                = 7);

  /**
   * @brief Create a CAGRA index parameters compatible with HNSW index
   *
   * @param dataset The shape of the input dataset
   * @param M HNSW index parameter M
   * @param ef_construction HNSW index parameter ef_construction
   * @param heuristic The heuristic to use for selecting the graph build parameters
   * @param metric The distance metric to search
   *
   * * IMPORTANT NOTE *
   *
   * The reference HNSW index and the corresponding from-CAGRA generated HNSW index will NOT produce
   * exactly the same recalls and QPS for the same parameter `ef`. The graphs are different
   * internally. Depending on the selected heuristics, the CAGRA-produced graph's QPS-Recall curve
   * may be shifted along the curve right or left. See the heuristics descriptions for more details.
   *
   * Usage example:
   * @code{.cpp}
   *   using namespace cuvs::neighbors;
   *   raft::resources res;
   *   auto dataset = raft::make_device_matrix<float, int64_t>(res, N, D);
   *   auto cagra_params = cagra::index_params::from_hnsw_params(dataset.extents(), M, efc);
   *   auto cagra_index = cagra::build(res, cagra_params, dataset);
   *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, cagra_index);
   * @endcode
   */
  static cagra::index_params from_hnsw_params(
    raft::matrix_extent<int64_t> dataset,
    int M,
    int ef_construction,
    hnsw_heuristic_type heuristic       = hnsw_heuristic_type::SIMILAR_SEARCH_PERFORMANCE,
    cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
};

/**
 * @}
 */

/**
 * @defgroup cagra_cpp_search_params CAGRA index search parameters
 * @{
 */

enum class search_algo {
  /** For large batch sizes. */
  SINGLE_CTA = 0,
  /** For small batch sizes. */
  MULTI_CTA    = 1,
  MULTI_KERNEL = 2,
  AUTO         = 100
};

enum class hash_mode { HASH = 0, SMALL = 1, AUTO = 100 };

enum class internal_dtype { F16 = 0, E5M2 = 1 };

struct search_params : cuvs::neighbors::search_params {
  /** Maximum number of queries to search at the same time (batch size). Auto select when 0.*/
  size_t max_queries = 0;

  /** Number of intermediate search results retained during the search.
   *
   *  This is the main knob to adjust trade off between accuracy and search speed.
   *  Higher values improve the search accuracy.
   */
  size_t itopk_size = 64;

  /** Upper limit of search iterations. Auto select when 0.*/
  size_t max_iterations = 0;

  // In the following we list additional search parameters for fine tuning.
  // Reasonable default values are automatically chosen.

  /** Which search implementation to use. */
  search_algo algo = search_algo::AUTO;

  /** Number of threads used to calculate a single distance. 4, 8, 16, or 32. */
  size_t team_size = 0;

  /** Number of graph nodes to select as the starting point for the search in each iteration. aka
   * search width?*/
  size_t search_width = 1;
  /** Lower limit of search iterations. */
  size_t min_iterations = 0;

  /** Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0. */
  size_t thread_block_size = 0;
  /** Hashmap type. Auto selection when AUTO. */
  hash_mode hashmap_mode = hash_mode::AUTO;
  /** Lower limit of hashmap bit length. More than 8. */
  size_t hashmap_min_bitlen = 0;
  /** Upper limit of hashmap fill rate. More than 0.1, less than 0.9.*/
  float hashmap_max_fill_rate = 0.5;

  /** Number of iterations of initial random seed node selection. 1 or more. */
  uint32_t num_random_samplings = 1;
  /** Bit mask used for initial random seed node selection. */
  uint64_t rand_xor_mask = 0x128394;

  /** Whether to use the persistent version of the kernel (only SINGLE_CTA is supported a.t.m.) */
  bool persistent = false;
  /** Persistent kernel: time in seconds before the kernel stops if no requests received. */
  float persistent_lifetime = 2;
  /**
   * Set the fraction of maximum grid size used by persistent kernel.
   * Value 1.0 means the kernel grid size is maximum possible for the selected device.
   * The value must be greater than 0.0 and not greater than 1.0.
   *
   * One may need to run other kernels alongside this persistent kernel. This parameter can
   * be used to reduce the grid size of the persistent kernel to leave a few SMs idle.
   * Note: running any other work on GPU alongside with the persistent kernel makes the setup
   * fragile.
   *   - Running another kernel in another thread usually works, but no progress guaranteed
   *   - Any CUDA allocations block the context (this issue may be obscured by using pools)
   *   - Memory copies to not-pinned host memory may block the context
   *
   * Even when we know there are no other kernels working at the same time, setting
   * kDeviceUsage to 1.0 surprisingly sometimes hurts performance. Proceed with care.
   * If you suspect this is an issue, you can reduce this number to ~0.9 without a significant
   * impact on the throughput.
   */
  float persistent_device_usage = 1.0;

  /**
   * A parameter indicating the rate of nodes to be filtered-out, when filtering is used.
   * The value must be equal to or greater than 0.0 and less than 1.0. Default value is
   * negative, in which case the filtering rate is automatically calculated when possible.
   * For `filtering::udf_filter`, CAGRA uses `udf_filter::filtering_rate` when this value is
   * negative. If both values are negative, CAGRA assumes 0.0 because a UDF's selectivity cannot be
   * inferred from the source string.
   */
  float filtering_rate = -1.0;

  /** Data type of the query vector and codebook table on shared memory. Currently, only VPQ
   * supports FP8. **/
  internal_dtype smem_dtype = internal_dtype::F16;
};

/**
 * @}
 */

/**
 * @defgroup cagra_cpp_extend_params CAGRA index extend parameters
 * @{
 */

struct extend_params {
  /** The additional dataset is divided into chunks and added to the graph. This is the knob to
   * adjust the tradeoff between the recall and operation throughput. Large chunk sizes can result
   * in high throughput, but use more working memory (O(max_chunk_size*degree^2)). This can also
   * degrade recall because no edges are added between the nodes in the same chunk. Auto select when
   * 0. */
  uint32_t max_chunk_size = 0;
};
/**
 * @}
 */

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

/**
 * @defgroup cagra_cpp_index CAGRA index type
 * @{
 */

/**
 * @brief CAGRA index.
 *
 * The index stores the dataset and a kNN graph in device memory.
 *
 * @tparam T data element type
 * @tparam IdxT the data type used to store the neighbor indices in the  search graph.
 *              It must be large enough to represent values up to dataset.extent(0).
 * @tparam DatasetViewT concrete non-owning dataset view type stored by the index
 *
 */
template <typename T,
          typename IdxT,
          ann_dataset_view DatasetViewT = device_padded_dataset_view<T, int64_t>>
struct CUVS_EXPORT index : cuvs::neighbors::index {
  using index_params_type  = cagra::index_params;
  using search_params_type = cagra::search_params;
  using index_type         = IdxT;
  using value_type         = T;
  using dataset_index_type = int64_t;
  using graph_index_type   = uint32_t;

  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> cuvs::distance::DistanceType
  {
    return metric_;
  }

  /** Total length of the index (number of vectors). */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
  {
    if (dataset_fd_.has_value() || graph_fd_.has_value()) { return n_rows_; }
    auto data_rows = dataset_.n_rows();
    return data_rows > 0 ? data_rows : graph_view_.extent(0);
  }

  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t
  {
    return dataset_fd_.has_value() ? dim_ : dataset_.dim();
  }
  /** Graph degree */
  [[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t
  {
    return graph_fd_.has_value() ? graph_degree_ : graph_view_.extent(1);
  }

  /** Number of rows represented by the graph. */
  [[nodiscard]] constexpr inline auto graph_size() const noexcept -> IdxT
  {
    return graph_fd_.has_value() ? static_cast<IdxT>(n_rows_)
                                 : static_cast<IdxT>(graph_view_.extent(0));
  }

  /** Non-owning dataset binding stored by the index. */
  [[nodiscard]] inline auto dataset() const noexcept -> DatasetViewT const& { return dataset_; }

  /** neighborhood graph [size, graph-degree] */
  [[nodiscard]] inline auto graph() const noexcept
    -> raft::device_matrix_view<const graph_index_type, int64_t, raft::row_major>
  {
    return graph_view_;
  }

  /** Mapping from internal graph node indices to the original user-provided indices. */
  [[nodiscard]] inline auto source_indices() const noexcept
    -> std::optional<raft::device_vector_view<const index_type, int64_t>>
  {
    return source_indices_.has_value()
             ? std::optional<raft::device_vector_view<const index_type, int64_t>>(
                 source_indices_->view())
             : std::nullopt;
  }

  /** Get the dataset file descriptor (for disk-backed index) */
  [[nodiscard]] inline auto dataset_fd() const noexcept
    -> const std::optional<cuvs::util::file_descriptor>&
  {
    return dataset_fd_;
  }

  /** Get the graph file descriptor (for disk-backed index) */
  [[nodiscard]] inline auto graph_fd() const noexcept
    -> const std::optional<cuvs::util::file_descriptor>&
  {
    return graph_fd_;
  }

  /** Get the mapping file descriptor (for disk-backed index) */
  [[nodiscard]] inline auto mapping_fd() const noexcept
    -> const std::optional<cuvs::util::file_descriptor>&
  {
    return mapping_fd_;
  }

  /** Dataset norms for cosine distance [size] */
  [[nodiscard]] inline auto dataset_norms() const noexcept
    -> std::optional<raft::device_vector_view<const float, int64_t>>
  {
    if (dataset_norms_.has_value()) { return raft::make_const_mdspan(dataset_norms_->view()); }
    return std::nullopt;
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  /** \cond */
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;
  /** \endcond */

  /** Construct a graph-only index with a zero-row dataset view placeholder. */
  explicit index(raft::resources const& res,
                 cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
    requires(cuvs::neighbors::ann_dataset_view<DatasetViewT, int64_t>)
    : cuvs::neighbors::index(),
      metric_(metric),
      graph_(raft::make_device_matrix<graph_index_type, int64_t>(res, 0, 0)),
      dataset_{},
      dataset_norms_(std::nullopt)
  {
  }

  /** Construct an index from a `dataset_view` and knn_graph.
   *
   * Stores a shallow copy of the dataset view. The index stores a **non-owning** view; the caller
   * must keep the underlying host or device storage alive for the index lifetime.
   *
   * Example — **non-owning** `make_device_padded_dataset_view` (wraps an existing device matrix;
   * that matrix must outlive the index):
   * @code{.cpp}
   *   raft::device_matrix_view<const float, int64_t, raft::row_major> dataset = ...;
   *   auto view = cuvs::neighbors::make_device_padded_dataset_view(res, dataset);
   *   auto graph = raft::make_device_matrix_view<const uint32_t, int64_t>(...);
   *   cuvs::neighbors::cagra::device_padded_index<float> idx(res, metric, view,
   *                                                       raft::make_const_mdspan(graph));
   * @endcode
   *
   * Example — **owning** `make_device_padded_dataset` returns owning storage (`std::unique_ptr`).
   * You must
   * **keep that object alive** (e.g. hold the `unique_ptr` in a variable or member) for as long as
   * the index uses the dataset; the index does not take ownership of the buffer.
   * @code{.cpp}
   *   auto padded_owner = cuvs::neighbors::make_device_padded_dataset(res, dataset_mdspan);
   *   auto view         = padded_owner->as_dataset_view();
   *   cuvs::neighbors::cagra::device_padded_index<float> idx(res, metric, view,
   *                                                       raft::make_const_mdspan(graph));
   *   // `padded_owner` must outlive `idx` (do not let it go out of scope while `idx` is used).
   * @endcode
   */
  template <typename graph_accessor>
  index(raft::resources const& res,
        cuvs::distance::DistanceType metric,
        DatasetViewT const& dataset,
        raft::mdspan<const graph_index_type,
                     raft::matrix_extent<int64_t>,
                     raft::row_major,
                     graph_accessor> knn_graph)
    : cuvs::neighbors::index(),
      metric_(metric),
      graph_(raft::make_device_matrix<graph_index_type, int64_t>(res, 0, 0)),
      dataset_(dataset),
      dataset_norms_(std::nullopt)
  {
    RAFT_EXPECTS(dataset.n_rows() == static_cast<int64_t>(knn_graph.extent(0)),
                 "Dataset and knn_graph must have equal number of rows");
    update_graph(res, knn_graph);

    if constexpr (cuvs::neighbors::is_device_dataset_view_v<DatasetViewT>) {
      if (metric_ == cuvs::distance::DistanceType::CosineExpanded && dataset.n_rows() > 0) {
        compute_dataset_norms_(res);
      }
    }

    raft::resource::sync_stream(res);
  }

  /* Construct an index with a new dataset type by moving the old index and passing in a new
   * dataset*/
  template <ann_dataset_view SrcDatasetViewT>
  index(raft::resources const& res, index<T, IdxT, SrcDatasetViewT>&& other, DatasetViewT dataset)
    : metric_(other.metric_),
      graph_(std::move(other.graph_)),
      graph_view_(other.graph_view_),
      dataset_(dataset),
      source_indices_(std::move(other.source_indices_)),
      dataset_norms_(std::nullopt),
      graph_fd_(std::move(other.graph_fd_)),
      mapping_fd_(std::move(other.mapping_fd_)),
      n_rows_(other.n_rows_),
      dim_(other.dim_),
      graph_degree_(other.graph_degree_)
  {
    if constexpr (is_device_dataset_view_v<DatasetViewT>) {
      if (metric() == cuvs::distance::DistanceType::CosineExpanded) {
        if (dataset_.n_rows() > 0) { compute_dataset_norms_(res); }
      }
    }
  }

  /**
   * Replace the graph with a new graph.
   *
   * Since the new graph is a device array, we store a reference to that, and it is
   * the caller's responsibility to ensure that knn_graph stays alive as long as the index.
   */
  void update_graph(
    raft::resources const& res,
    raft::device_matrix_view<const graph_index_type, int64_t, raft::row_major> knn_graph)
  {
    graph_view_ = knn_graph;
  }

  /**
   * Replace the graph by taking ownership of an existing device matrix.
   */
  void update_graph(raft::resources const&,
                    raft::device_matrix<graph_index_type, int64_t, raft::row_major>&& knn_graph)
  {
    graph_      = std::move(knn_graph);
    graph_view_ = graph_.view();
  }

  /**
   * Replace the graph with a new graph.
   *
   * We create a copy of the graph on the device. The index manages the lifetime of this copy.
   */
  void update_graph(
    raft::resources const& res,
    raft::host_matrix_view<const graph_index_type, int64_t, raft::row_major> knn_graph)
  {
    RAFT_LOG_DEBUG("Copying CAGRA knn graph from host to device");

    if ((graph_.extent(0) != knn_graph.extent(0)) || (graph_.extent(1) != knn_graph.extent(1))) {
      // clear existing memory before allocating to prevent OOM errors on large graphs
      if (graph_.size()) {
        graph_ = raft::make_device_matrix<graph_index_type, int64_t>(res, 0, 0);
      }
      graph_ = raft::make_device_matrix<graph_index_type, int64_t>(
        res, knn_graph.extent(0), knn_graph.extent(1));
    }
    raft::copy(graph_.data_handle(),
               knn_graph.data_handle(),
               knn_graph.size(),
               raft::resource::get_cuda_stream(res));
    graph_view_ = graph_.view();
  }

  /**
   * Replace the source indices with a new source indices taking the ownership of the passed vector.
   */
  void update_source_indices(raft::device_vector<index_type, int64_t>&& source_indices)
  {
    RAFT_EXPECTS(source_indices.extent(0) == size(),
                 "Source indices must have the same number of rows as the index");
    source_indices_.emplace(std::move(source_indices));
  }

  /**
   * Copy the provided source indices into the index.
   */
  template <typename Accessor>
  void update_source_indices(
    raft::resources const& res,
    raft::mdspan<const index_type, raft::vector_extent<int64_t>, raft::row_major, Accessor>
      source_indices)
  {
    RAFT_EXPECTS(source_indices.extent(0) == size(),
                 "Source indices must have the same number of rows as the index");
    // Reset the array if it's not compatible to avoid using more memory than necessary.
    // NB: this likely is never triggered because we check the invariant above (but it doesn't
    // hurt).
    if (source_indices_.has_value()) {
      if (source_indices_->extent(0) != source_indices.extent(0)) { source_indices_.reset(); }
    }
    // Allocate the new array if needed.
    if (!source_indices_.has_value()) {
      source_indices_.emplace(
        raft::make_device_vector<index_type, int64_t>(res, source_indices.extent(0)));
    }
    // Copy the data.
    raft::copy(source_indices_->data_handle(),
               source_indices.data_handle(),
               source_indices.extent(0),
               raft::resource::get_cuda_stream(res));
  }

  /**
   * Update the dataset from a disk file using a file descriptor.
   *
   * This method configures the index to use a disk-based dataset.
   * The dataset file should contain a numpy header followed by vectors in row-major format.
   * The number of rows and dimensionality are read from the numpy header.
   *
   * @param[in] res raft resources
   * @param[in] fd File descriptor (will be moved into the index for lifetime management)
   */
  void update_dataset(raft::resources const& res, cuvs::util::file_descriptor&& fd)
  {
    RAFT_EXPECTS(fd.is_valid(), "Invalid file descriptor provided for dataset");

    auto stream = fd.make_istream();
    if (lseek(fd.get(), 0, SEEK_SET) == -1) {
      RAFT_FAIL("Failed to seek to beginning of dataset file");
    }
    auto header = raft::numpy_serializer::read_header(stream);
    RAFT_EXPECTS(header.shape.size() == 2,
                 "Dataset file should be 2D, got %zu dimensions",
                 header.shape.size());

    n_rows_ = header.shape[0];
    dim_    = header.shape[1];

    RAFT_LOG_DEBUG("ACE: Dataset has shape [%zu, %zu]", n_rows_, dim_);

    // Re-open the file descriptor in read-only mode for subsequent operations
    dataset_fd_.emplace(std::move(fd));

    if constexpr (cuvs::neighbors::is_device_padded_dataset_view_v<DatasetViewT>) {
      auto v = raft::make_device_matrix_view<const T, int64_t>(
        static_cast<const T*>(nullptr), int64_t{0}, dim_);
      dataset_ = DatasetViewT(v, dim_);
    } else if constexpr (cuvs::neighbors::is_device_standard_dataset_view_v<DatasetViewT>) {
      auto v = raft::make_device_matrix_view<const T, int64_t>(
        static_cast<const T*>(nullptr), int64_t{0}, dim_);
      dataset_ = DatasetViewT(v);
    } else if constexpr (cuvs::neighbors::is_host_padded_dataset_view_v<DatasetViewT>) {
      auto v = raft::make_host_matrix_view<const T, int64_t>(
        static_cast<const T*>(nullptr), int64_t{0}, dim_);
      dataset_ = DatasetViewT(v, dim_);
    } else if constexpr (cuvs::neighbors::is_host_standard_dataset_view_v<DatasetViewT>) {
      auto v = raft::make_host_matrix_view<const T, int64_t>(
        static_cast<const T*>(nullptr), int64_t{0}, dim_);
      dataset_ = DatasetViewT(v);
    } else if constexpr (cuvs::neighbors::is_empty_dataset_view_v<DatasetViewT>) {
      dataset_ = DatasetViewT{dim_};
    } else {
      RAFT_FAIL("update_dataset(fd): unsupported DatasetViewT for disk-backed dataset");
    }
    dataset_norms_.reset();
  }

  /**
   * Update the graph from a disk file using a file descriptor.
   *
   * This method configures the index to use a disk-based graph.
   * The graph file should contain a numpy header followed by neighbor indices in row-major format.
   * The number of rows and graph degree are read from the numpy header.
   *
   * @param[in] res raft resources
   * @param[in] fd File descriptor (will be moved into the index for lifetime management)
   */
  void update_graph(raft::resources const& res, cuvs::util::file_descriptor&& fd)
  {
    RAFT_EXPECTS(fd.is_valid(), "Invalid file descriptor provided for graph");

    auto stream = fd.make_istream();
    if (lseek(fd.get(), 0, SEEK_SET) == -1) {
      RAFT_FAIL("Failed to seek to beginning of graph file");
    }
    auto header = raft::numpy_serializer::read_header(stream);
    RAFT_EXPECTS(
      header.shape.size() == 2, "Graph file should be 2D, got %zu dimensions", header.shape.size());

    if (dataset_fd_.has_value() && n_rows_ != 0) {
      RAFT_EXPECTS(n_rows_ == header.shape[0],
                   "Graph size (%zu) must match dataset size (%zu)",
                   header.shape[0],
                   n_rows_);
    }

    n_rows_       = header.shape[0];
    graph_degree_ = header.shape[1];

    RAFT_LOG_DEBUG("ACE: Graph has shape [%zu, %zu]", n_rows_, graph_degree_);

    // Re-open the file descriptor in read-only mode for subsequent operations
    graph_fd_.emplace(std::move(fd));

    graph_      = raft::make_device_matrix<IdxT, int64_t>(res, 0, 0);
    graph_view_ = graph_.view();
  }

  /**
   * Update the dataset mapping from a disk file using a file descriptor.
   *
   * This method configures the index to use a disk-based dataset mapping.
   * The mapping file should contain a numpy header followed by index mappings.
   *
   * @param[in] res raft resources
   * @param[in] fd File descriptor (will be moved into the index for lifetime management)
   */
  void update_mapping(raft::resources const& res, cuvs::util::file_descriptor&& fd)
  {
    RAFT_EXPECTS(fd.is_valid(), "Invalid file descriptor provided for mapping");

    // Read header from file using ifstream
    auto stream = fd.make_istream();
    if (lseek(fd.get(), 0, SEEK_SET) == -1) {
      RAFT_FAIL("Failed to seek to beginning of mapping file");
    }
    auto header = raft::numpy_serializer::read_header(stream);
    RAFT_EXPECTS(header.shape.size() == 1,
                 "Mapping file should be 1D, got %zu dimensions",
                 header.shape.size());

    if (dataset_fd_.has_value() && n_rows_ != 0) {
      RAFT_EXPECTS(header.shape[0] == n_rows_,
                   "Mapping size (%zu) must match dataset size (%zu)",
                   header.shape[0],
                   n_rows_);
    }

    RAFT_LOG_DEBUG("ACE: Mapping has %zu elements", header.shape[0]);

    mapping_fd_.emplace(std::move(fd));
  }

 private:
  template <typename, typename, ann_dataset_view>
  friend struct index;

  friend struct detail::fd_transfer;

  [[nodiscard]] inline auto steal_dataset_fd_() noexcept
    -> std::optional<cuvs::util::file_descriptor>
  {
    return std::exchange(dataset_fd_, std::nullopt);
  }

  [[nodiscard]] inline auto steal_graph_fd_() noexcept -> std::optional<cuvs::util::file_descriptor>
  {
    return std::exchange(graph_fd_, std::nullopt);
  }

  [[nodiscard]] inline auto steal_mapping_fd_() noexcept
    -> std::optional<cuvs::util::file_descriptor>
  {
    return std::exchange(mapping_fd_, std::nullopt);
  }

  cuvs::distance::DistanceType metric_;
  raft::device_matrix<graph_index_type, int64_t, raft::row_major> graph_;
  raft::device_matrix_view<const graph_index_type, int64_t, raft::row_major> graph_view_;
  DatasetViewT dataset_;
  // Mapping from internal graph node indices to the original user-provided indices.
  std::optional<raft::device_vector<IdxT, int64_t>> source_indices_;
  // only float distances supported at the moment
  std::optional<raft::device_vector<float, int64_t>> dataset_norms_;
  // File descriptors for disk-backed index components (ACE disk mode)
  std::optional<cuvs::util::file_descriptor> dataset_fd_;
  std::optional<cuvs::util::file_descriptor> graph_fd_;
  std::optional<cuvs::util::file_descriptor> mapping_fd_;

  CUVS_EXPORT void compute_dataset_norms_(raft::resources const& res);
  size_t n_rows_       = 0;
  size_t dim_          = 0;
  size_t graph_degree_ = 0;
};

/** CAGRA index with the usual padded device dataset view (graph build output type). */
template <typename T, typename IdxT = uint32_t>
using device_padded_index = index<T, IdxT, cuvs::neighbors::device_padded_dataset_view<T, int64_t>>;

/** CAGRA index with a host-resident padded dataset view (returned by host build path). */
template <typename T, typename IdxT = uint32_t>
using host_padded_index = index<T, IdxT, cuvs::neighbors::host_padded_dataset_view<T, int64_t>>;

/** CAGRA index with a device-resident standard (arbitrary stride) dataset view. */
template <typename T, typename IdxT = uint32_t>
using device_standard_index =
  index<T, IdxT, cuvs::neighbors::device_standard_dataset_view<T, int64_t>>;

/** CAGRA index with a host-resident standard dataset view. */
template <typename T, typename IdxT = uint32_t>
using host_standard_index = index<T, IdxT, cuvs::neighbors::host_standard_dataset_view<T, int64_t>>;

/** CAGRA index with a device-resident VPQ dataset (f16 codebook vectors). */
template <typename T, typename IdxT = uint32_t>
using vpq_f16_index = index<T, IdxT, cuvs::neighbors::device_vpq_dataset_view<half, int64_t>>;

/** CAGRA index with a device-resident VPQ dataset (f32 codebook vectors). */
template <typename T, typename IdxT = uint32_t>
using vpq_f32_index = index<T, IdxT, cuvs::neighbors::device_vpq_dataset_view<float, int64_t>>;

/** Index type returned by `cagra::build(res, params, dataset_view)`. */
template <typename DatasetViewT>
using cagra_index_t = index<cuvs::neighbors::cagra_view_element_type_t<DatasetViewT>,
                            uint32_t,
                            cuvs::neighbors::dataset_view_type_t<DatasetViewT>>;

/**
 * @}
 */

/**
 * @defgroup cagra_cpp_index_build CAGRA index build functions
 * @{
 */

/**
 * @brief Build the index from a `dataset_view` (device padded/standard or host padded/standard).
 *
 * VPQ-compressed device views are rejected: dense graph construction requires uncompressed data.
 * Use a separate VPQ index workflow after building the graph from an uncompressed dataset.
 *
 * When `index_params.attach_dataset_on_build = true` (the default), a dense `dataset` view is
 * stored in the returned index as a **non-owning view** — no copy is made. The caller must keep the
 * underlying storage alive for the lifetime of the index. An index backed by a device view is then
 * ready to search immediately.
 *
 * When `index_params.attach_dataset_on_build = false`, only the search graph is built and the
 * returned index holds no dataset.
 *
 * An index backed by a host view cannot be searched. Call the type-changing `update_dataset` with
 * a user-provided device-padded dataset view to obtain a search-ready `device_padded_index`.
 *
 * Note: disk-based ACE builds (`ace_params::use_disk = true`) always set a file-descriptor
 * dataset internally (also host-typed); `attach_dataset_on_build` is ignored there too.
 */
// Concrete non-template overloads for all supported build dataset view types.
// This keeps the public header explicit and stable while implementation remains shared internally.
/**
 * @brief Build from a device padded dataset view (`float`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset device padded dataset view [n_rows, dim]
 * @return built `device_padded_index<float, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::device_padded_dataset_view<float, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::device_padded_index<float, uint32_t>;

/**
 * @brief Build from a device standard dataset view (`float`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset device standard dataset view [n_rows, dim]
 * @return built `device_standard_index<float, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::device_standard_dataset_view<float, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::device_standard_index<float, uint32_t>;

/**
 * @brief Build from a host padded dataset view (`float`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset host padded dataset view [n_rows, dim]
 * @return built `host_padded_index<float, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::host_padded_dataset_view<float, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::host_padded_index<float, uint32_t>;

/**
 * @brief Build from a host standard dataset view (`float`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset host standard dataset view [n_rows, dim]
 * @return built `host_standard_index<float, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::host_standard_dataset_view<float, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::host_standard_index<float, uint32_t>;

/**
 * @brief Build from a device padded dataset view (`half`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset device padded dataset view [n_rows, dim]
 * @return built `device_padded_index<half, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::device_padded_dataset_view<half, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::device_padded_index<half, uint32_t>;

/**
 * @brief Build from a device standard dataset view (`half`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset device standard dataset view [n_rows, dim]
 * @return built `device_standard_index<half, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::device_standard_dataset_view<half, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::device_standard_index<half, uint32_t>;

/**
 * @brief Build from a host padded dataset view (`half`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset host padded dataset view [n_rows, dim]
 * @return built `host_padded_index<half, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::host_padded_dataset_view<half, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::host_padded_index<half, uint32_t>;

/**
 * @brief Build from a host standard dataset view (`half`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset host standard dataset view [n_rows, dim]
 * @return built `host_standard_index<half, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::host_standard_dataset_view<half, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::host_standard_index<half, uint32_t>;

/**
 * @brief Build from a device padded dataset view (`int8_t`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset device padded dataset view [n_rows, dim]
 * @return built `device_padded_index<int8_t, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::device_padded_dataset_view<int8_t, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::device_padded_index<int8_t, uint32_t>;

/**
 * @brief Build from a device standard dataset view (`int8_t`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset device standard dataset view [n_rows, dim]
 * @return built `device_standard_index<int8_t, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::device_standard_dataset_view<int8_t, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::device_standard_index<int8_t, uint32_t>;

/**
 * @brief Build from a host padded dataset view (`int8_t`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset host padded dataset view [n_rows, dim]
 * @return built `host_padded_index<int8_t, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::host_padded_dataset_view<int8_t, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::host_padded_index<int8_t, uint32_t>;

/**
 * @brief Build from a host standard dataset view (`int8_t`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset host standard dataset view [n_rows, dim]
 * @return built `host_standard_index<int8_t, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::host_standard_dataset_view<int8_t, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::host_standard_index<int8_t, uint32_t>;

/**
 * @brief Build from a device padded dataset view (`uint8_t`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset device padded dataset view [n_rows, dim]
 * @return built `device_padded_index<uint8_t, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::device_padded_dataset_view<uint8_t, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::device_padded_index<uint8_t, uint32_t>;

/**
 * @brief Build from a device standard dataset view (`uint8_t`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset device standard dataset view [n_rows, dim]
 * @return built `device_standard_index<uint8_t, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::device_standard_dataset_view<uint8_t, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::device_standard_index<uint8_t, uint32_t>;

/**
 * @brief Build from a host padded dataset view (`uint8_t`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset host padded dataset view [n_rows, dim]
 * @return built `host_padded_index<uint8_t, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::host_padded_dataset_view<uint8_t, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::host_padded_index<uint8_t, uint32_t>;

/**
 * @brief Build from a host standard dataset view (`uint8_t`).
 * @param[in] res raft resources
 * @param[in] params CAGRA index build parameters
 * @param[in] dataset host standard dataset view [n_rows, dim]
 * @return built `host_standard_index<uint8_t, uint32_t>`
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           cuvs::neighbors::host_standard_dataset_view<uint8_t, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::host_standard_index<uint8_t, uint32_t>;

/**
 * @}
 */

/**
 * @defgroup cagra_cpp_index_extend CAGRA extend functions
 * @{
 */

// Concrete non-template overloads for all supported index types.
// Previously a single template <T, IdxT, DatasetViewT> covered all index types; it has been
// replaced with explicit overloads to maintain a stable non-template ABI. When a new index
// type is added (e.g. a future host_padded_index extend), add a corresponding overload here.
// Index types for which extend is not meaningful (e.g. VPQ — read-only compressed codes)
// are intentionally omitted.

/** @brief Add new vectors to a CAGRA index
 *
 * Note: `extend` does not concatenate datasets. The caller owns the final dataset and must
 * pre-populate a single padded device matrix of size `(n_old + n_new) x dim` (or overallocation
 * with a view whose logical `n_rows` is `n_old + n_new`):
 *   - rows `[0, new_start_row)` hold the original vectors attached to `idx`
 *   - rows `[new_start_row, n_rows)` hold the additional vectors
 * `new_start_row` must equal `idx.size()` today. The library only extends the graph and rebinds
 * the index to `extended_dataset`. Keep that view alive for the index lifetime.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // Build `extended` = old || new on device, padded for CAGRA.
 *   auto extended = make_device_padded_dataset(res, concatenated_view);
 *   auto extended_view = extended->as_dataset_view();
 *
 *   cagra::extend_params params;
 *   int64_t new_start_row = static_cast<int64_t>(index.size());
 *   cagra::extend(res, params, extended_view, new_start_row, index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] extended_dataset caller-owned device-padded view already containing old || new rows
 * @param[in] new_start_row row index where the additional vectors begin (must equal `idx.size()`)
 * @param[in,out] idx CAGRA index; graph is extended and dataset view is rebound
 */
void extend(raft::resources const& handle,
            const cagra::extend_params& params,
            cuvs::neighbors::device_padded_dataset_view<float, int64_t> extended_dataset,
            int64_t new_start_row,
            cuvs::neighbors::cagra::device_padded_index<float, uint32_t>& idx);

/** @brief Add new vectors to a CAGRA index. See the float overload for the full contract. */
void extend(raft::resources const& handle,
            const cagra::extend_params& params,
            cuvs::neighbors::device_padded_dataset_view<half, int64_t> extended_dataset,
            int64_t new_start_row,
            cuvs::neighbors::cagra::device_padded_index<half, uint32_t>& idx);

/** @brief Add new vectors to a CAGRA index. See the float overload for the full contract. */
void extend(raft::resources const& handle,
            const cagra::extend_params& params,
            cuvs::neighbors::device_padded_dataset_view<int8_t, int64_t> extended_dataset,
            int64_t new_start_row,
            cuvs::neighbors::cagra::device_padded_index<int8_t, uint32_t>& idx);

/** @brief Add new vectors to a CAGRA index. See the float overload for the full contract. */
void extend(raft::resources const& handle,
            const cagra::extend_params& params,
            cuvs::neighbors::device_padded_dataset_view<uint8_t, int64_t> extended_dataset,
            int64_t new_start_row,
            cuvs::neighbors::cagra::device_padded_index<uint8_t, uint32_t>& idx);

/**
 * @}
 */

/**
 * @defgroup cagra_cpp_index_search CAGRA search functions
 * @{
 */

// Concrete non-template overloads for all supported index types.
// Explicit overloads (not a single template) keep a stable link ABI for the C API.
// When a new index type is added, add matching overloads here and in cagra_search_inst.cu.in.

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built device_padded_index with uint32_t neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_padded_index<float, uint32_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built device_padded_index with uint32_t neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_padded_index<half, uint32_t>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built device_padded_index with uint32_t neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_padded_index<int8_t, uint32_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built device_padded_index with uint32_t neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_padded_index<uint8_t, uint32_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built device_padded_index with int64_t neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_padded_index<float, uint32_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built device_padded_index with int64_t neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_padded_index<half, uint32_t>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built device_padded_index with int64_t neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_padded_index<int8_t, uint32_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built device_padded_index with int64_t neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_padded_index<uint8_t, uint32_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

// device_standard_index overloads (uint32_t neighbor indices)
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_standard_index<float, uint32_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_standard_index<half, uint32_t>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_standard_index<int8_t, uint32_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_standard_index<uint8_t, uint32_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

// device_standard_index overloads (int64_t neighbor indices)
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_standard_index<float, uint32_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_standard_index<half, uint32_t>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_standard_index<int8_t, uint32_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::device_standard_index<uint8_t, uint32_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

// vpq_f16_index overloads (uint32_t neighbor indices)
/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f16_index (CAGRA-Q, VPQ f16-compressed dataset) with uint32_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f16_index<float, uint32_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f16_index (CAGRA-Q, VPQ f16-compressed dataset) with uint32_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f16_index<half, uint32_t>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f16_index (CAGRA-Q, VPQ f16-compressed dataset) with uint32_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f16_index<int8_t, uint32_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f16_index (CAGRA-Q, VPQ f16-compressed dataset) with uint32_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f16_index<uint8_t, uint32_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

// vpq_f16_index overloads (int64_t neighbor indices)
/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f16_index (CAGRA-Q, VPQ f16-compressed dataset) with int64_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f16_index<float, uint32_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f16_index (CAGRA-Q, VPQ f16-compressed dataset) with int64_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f16_index<half, uint32_t>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f16_index (CAGRA-Q, VPQ f16-compressed dataset) with int64_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f16_index<int8_t, uint32_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f16_index (CAGRA-Q, VPQ f16-compressed dataset) with int64_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f16_index<uint8_t, uint32_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

// vpq_f32_index overloads (uint32_t neighbor indices)
/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f32_index (CAGRA-Q, VPQ f32-compressed dataset) with uint32_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 *
 * @note FP32 VPQ search is declared for ABI stability but fails at runtime until implemented.
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f32_index<float, uint32_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f32_index (CAGRA-Q, VPQ f32-compressed dataset) with uint32_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 *
 * @note FP32 VPQ search is declared for ABI stability but fails at runtime until implemented.
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f32_index<half, uint32_t>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f32_index (CAGRA-Q, VPQ f32-compressed dataset) with uint32_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 *
 * @note FP32 VPQ search is declared for ABI stability but fails at runtime until implemented.
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f32_index<int8_t, uint32_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f32_index (CAGRA-Q, VPQ f32-compressed dataset) with uint32_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 *
 * @note FP32 VPQ search is declared for ABI stability but fails at runtime until implemented.
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f32_index<uint8_t, uint32_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

// vpq_f32_index overloads (int64_t neighbor indices)
/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f32_index (CAGRA-Q, VPQ f32-compressed dataset) with int64_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 *
 * @note FP32 VPQ search is declared for ABI stability but fails at runtime until implemented.
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f32_index<float, uint32_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f32_index (CAGRA-Q, VPQ f32-compressed dataset) with int64_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 *
 * @note FP32 VPQ search is declared for ABI stability but fails at runtime until implemented.
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f32_index<half, uint32_t>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f32_index (CAGRA-Q, VPQ f32-compressed dataset) with int64_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 *
 * @note FP32 VPQ search is declared for ABI stability but fails at runtime until implemented.
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f32_index<int8_t, uint32_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index pre-built vpq_f32_index (CAGRA-Q, VPQ f32-compressed dataset) with int64_t
 * neighbor indices
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index.dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering).
 *
 * @note FP32 VPQ search is declared for ABI stability but fails at runtime until implemented.
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::vpq_f32_index<uint8_t, uint32_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

// TODO: Create an abstraction for multi-partition indices.
// Reference issue: https://github.com/NVIDIA/cuvs/issues/2281
/**
 * @brief Search multiple CAGRA index partitions concurrently and return the global top-k per
 * query.
 *
 * For each query row in @p queries, the kernel searches all partitions in parallel into an
 * internal intermediate buffer, applies per-partition distance post-processing, runs a batched
 * top-k merge across partitions, and writes the final outputs. The call returns when all work
 * has been submitted to the stream associated with @p res (not necessarily completed); call
 * @c raft::resource::sync_stream on @p res to wait for completion.
 *
 * @note Calling this API with a single partition (@p indices of size 1) still exercises the
 * multi-partition implementation rather than the single-index search overloads above, and the
 * behaviors are not guaranteed to be equivalent.
 *
 * @note All index partitions must use the same distance metric and graph degree; partition sizes
 * may differ. Compressed (VPQ) datasets are not currently supported in multi-partition search, so
 * partitions must be built on in-memory strided datasets.
 *
 * @param[in]  res            raft resources
 * @param[in]  params         search parameters (shared across partitions)
 * @param[in]  indices        CAGRA index objects, one per partition
 * @param[in]  queries        queries matrix, shape [n_queries, dim]; searched against every
 *                            partition
 * @param[out] partition_ids  which partition each neighbor came from, shape [n_queries, k]
 * @param[out] neighbors      ordinal in the corresponding partition's dataset, shape
 *                            [n_queries, k]
 * @param[out] distances      post-processed distance for each (query, neighbor), shape
 *                            [n_queries, k]
 */
void search(
  raft::resources const& res,
  cuvs::neighbors::cagra::search_params const& params,
  const std::vector<const cuvs::neighbors::cagra::index<float, uint32_t>*>& indices,
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> partition_ids,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  const std::vector<cuvs::core::bitset_view<std::uint32_t, int64_t>>& partition_bitsets = {});

/**
 * @brief Search multiple CAGRA index partitions concurrently and return the global top-k per
 * query.
 *
 * For each query row in @p queries, the kernel searches all partitions in parallel into an
 * internal intermediate buffer, applies per-partition distance post-processing, runs a batched
 * top-k merge across partitions, and writes the final outputs. The call returns when all work
 * has been submitted to the stream associated with @p res (not necessarily completed); call
 * @c raft::resource::sync_stream on @p res to wait for completion.
 *
 * @note Calling this API with a single partition (@p indices of size 1) still exercises the
 * multi-partition implementation rather than the single-index search overloads above, and the
 * behaviors are not guaranteed to be equivalent.
 *
 * @note All index partitions must use the same distance metric and graph degree; partition sizes
 * may differ. Compressed (VPQ) datasets are not currently supported in multi-partition search, so
 * partitions must be built on in-memory strided datasets.
 *
 * @param[in]  res            raft resources
 * @param[in]  params         search parameters (shared across partitions)
 * @param[in]  indices        CAGRA index objects, one per partition
 * @param[in]  queries        queries matrix, shape [n_queries, dim]; searched against every
 *                            partition
 * @param[out] partition_ids  which partition each neighbor came from, shape [n_queries, k]
 * @param[out] neighbors      ordinal in the corresponding partition's dataset, shape
 *                            [n_queries, k]
 * @param[out] distances      post-processed distance for each (query, neighbor), shape
 *                            [n_queries, k]
 */
void search(
  raft::resources const& res,
  cuvs::neighbors::cagra::search_params const& params,
  const std::vector<const cuvs::neighbors::cagra::index<float, uint32_t>*>& indices,
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> partition_ids,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  const std::vector<cuvs::core::bitset_view<std::uint32_t, int64_t>>& partition_bitsets = {});

/**
 * @brief Search multiple CAGRA index partitions concurrently and return the global top-k per
 * query.
 *
 * For each query row in @p queries, the kernel searches all partitions in parallel into an
 * internal intermediate buffer, applies per-partition distance post-processing, runs a batched
 * top-k merge across partitions, and writes the final outputs. The call returns when all work
 * has been submitted to the stream associated with @p res (not necessarily completed); call
 * @c raft::resource::sync_stream on @p res to wait for completion.
 *
 * @note Calling this API with a single partition (@p indices of size 1) still exercises the
 * multi-partition implementation rather than the single-index search overloads above, and the
 * behaviors are not guaranteed to be equivalent.
 *
 * @note All index partitions must use the same distance metric and graph degree; partition sizes
 * may differ. Compressed (VPQ) datasets are not currently supported in multi-partition search, so
 * partitions must be built on in-memory strided datasets.
 *
 * @param[in]  res            raft resources
 * @param[in]  params         search parameters (shared across partitions)
 * @param[in]  indices        CAGRA index objects, one per partition
 * @param[in]  queries        queries matrix, shape [n_queries, dim]; searched against every
 *                            partition
 * @param[out] partition_ids  which partition each neighbor came from, shape [n_queries, k]
 * @param[out] neighbors      ordinal in the corresponding partition's dataset, shape
 *                            [n_queries, k]
 * @param[out] distances      post-processed distance for each (query, neighbor), shape
 *                            [n_queries, k]
 */
void search(
  raft::resources const& res,
  cuvs::neighbors::cagra::search_params const& params,
  const std::vector<const cuvs::neighbors::cagra::index<half, uint32_t>*>& indices,
  raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> partition_ids,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  const std::vector<cuvs::core::bitset_view<std::uint32_t, int64_t>>& partition_bitsets = {});

/**
 * @brief Search multiple CAGRA index partitions concurrently and return the global top-k per
 * query.
 *
 * For each query row in @p queries, the kernel searches all partitions in parallel into an
 * internal intermediate buffer, applies per-partition distance post-processing, runs a batched
 * top-k merge across partitions, and writes the final outputs. The call returns when all work
 * has been submitted to the stream associated with @p res (not necessarily completed); call
 * @c raft::resource::sync_stream on @p res to wait for completion.
 *
 * @note Calling this API with a single partition (@p indices of size 1) still exercises the
 * multi-partition implementation rather than the single-index search overloads above, and the
 * behaviors are not guaranteed to be equivalent.
 *
 * @note All index partitions must use the same distance metric and graph degree; partition sizes
 * may differ. Compressed (VPQ) datasets are not currently supported in multi-partition search, so
 * partitions must be built on in-memory strided datasets.
 *
 * @param[in]  res            raft resources
 * @param[in]  params         search parameters (shared across partitions)
 * @param[in]  indices        CAGRA index objects, one per partition
 * @param[in]  queries        queries matrix, shape [n_queries, dim]; searched against every
 *                            partition
 * @param[out] partition_ids  which partition each neighbor came from, shape [n_queries, k]
 * @param[out] neighbors      ordinal in the corresponding partition's dataset, shape
 *                            [n_queries, k]
 * @param[out] distances      post-processed distance for each (query, neighbor), shape
 *                            [n_queries, k]
 */
void search(
  raft::resources const& res,
  cuvs::neighbors::cagra::search_params const& params,
  const std::vector<const cuvs::neighbors::cagra::index<half, uint32_t>*>& indices,
  raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> partition_ids,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  const std::vector<cuvs::core::bitset_view<std::uint32_t, int64_t>>& partition_bitsets = {});

/**
 * @brief Search multiple CAGRA index partitions concurrently and return the global top-k per
 * query.
 *
 * For each query row in @p queries, the kernel searches all partitions in parallel into an
 * internal intermediate buffer, applies per-partition distance post-processing, runs a batched
 * top-k merge across partitions, and writes the final outputs. The call returns when all work
 * has been submitted to the stream associated with @p res (not necessarily completed); call
 * @c raft::resource::sync_stream on @p res to wait for completion.
 *
 * @note Calling this API with a single partition (@p indices of size 1) still exercises the
 * multi-partition implementation rather than the single-index search overloads above, and the
 * behaviors are not guaranteed to be equivalent.
 *
 * @note All index partitions must use the same distance metric and graph degree; partition sizes
 * may differ. Compressed (VPQ) datasets are not currently supported in multi-partition search, so
 * partitions must be built on in-memory strided datasets.
 *
 * @param[in]  res            raft resources
 * @param[in]  params         search parameters (shared across partitions)
 * @param[in]  indices        CAGRA index objects, one per partition
 * @param[in]  queries        queries matrix, shape [n_queries, dim]; searched against every
 *                            partition
 * @param[out] partition_ids  which partition each neighbor came from, shape [n_queries, k]
 * @param[out] neighbors      ordinal in the corresponding partition's dataset, shape
 *                            [n_queries, k]
 * @param[out] distances      post-processed distance for each (query, neighbor), shape
 *                            [n_queries, k]
 */
void search(
  raft::resources const& res,
  cuvs::neighbors::cagra::search_params const& params,
  const std::vector<const cuvs::neighbors::cagra::index<int8_t, uint32_t>*>& indices,
  raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> partition_ids,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  const std::vector<cuvs::core::bitset_view<std::uint32_t, int64_t>>& partition_bitsets = {});

/**
 * @brief Search multiple CAGRA index partitions concurrently and return the global top-k per
 * query.
 *
 * For each query row in @p queries, the kernel searches all partitions in parallel into an
 * internal intermediate buffer, applies per-partition distance post-processing, runs a batched
 * top-k merge across partitions, and writes the final outputs. The call returns when all work
 * has been submitted to the stream associated with @p res (not necessarily completed); call
 * @c raft::resource::sync_stream on @p res to wait for completion.
 *
 * @note Calling this API with a single partition (@p indices of size 1) still exercises the
 * multi-partition implementation rather than the single-index search overloads above, and the
 * behaviors are not guaranteed to be equivalent.
 *
 * @note All index partitions must use the same distance metric and graph degree; partition sizes
 * may differ. Compressed (VPQ) datasets are not currently supported in multi-partition search, so
 * partitions must be built on in-memory strided datasets.
 *
 * @param[in]  res            raft resources
 * @param[in]  params         search parameters (shared across partitions)
 * @param[in]  indices        CAGRA index objects, one per partition
 * @param[in]  queries        queries matrix, shape [n_queries, dim]; searched against every
 *                            partition
 * @param[out] partition_ids  which partition each neighbor came from, shape [n_queries, k]
 * @param[out] neighbors      ordinal in the corresponding partition's dataset, shape
 *                            [n_queries, k]
 * @param[out] distances      post-processed distance for each (query, neighbor), shape
 *                            [n_queries, k]
 */
void search(
  raft::resources const& res,
  cuvs::neighbors::cagra::search_params const& params,
  const std::vector<const cuvs::neighbors::cagra::index<int8_t, uint32_t>*>& indices,
  raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> partition_ids,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  const std::vector<cuvs::core::bitset_view<std::uint32_t, int64_t>>& partition_bitsets = {});

/**
 * @brief Search multiple CAGRA index partitions concurrently and return the global top-k per
 * query.
 *
 * For each query row in @p queries, the kernel searches all partitions in parallel into an
 * internal intermediate buffer, applies per-partition distance post-processing, runs a batched
 * top-k merge across partitions, and writes the final outputs. The call returns when all work
 * has been submitted to the stream associated with @p res (not necessarily completed); call
 * @c raft::resource::sync_stream on @p res to wait for completion.
 *
 * @note Calling this API with a single partition (@p indices of size 1) still exercises the
 * multi-partition implementation rather than the single-index search overloads above, and the
 * behaviors are not guaranteed to be equivalent.
 *
 * @note All index partitions must use the same distance metric and graph degree; partition sizes
 * may differ. Compressed (VPQ) datasets are not currently supported in multi-partition search, so
 * partitions must be built on in-memory strided datasets.
 *
 * @param[in]  res            raft resources
 * @param[in]  params         search parameters (shared across partitions)
 * @param[in]  indices        CAGRA index objects, one per partition
 * @param[in]  queries        queries matrix, shape [n_queries, dim]; searched against every
 *                            partition
 * @param[out] partition_ids  which partition each neighbor came from, shape [n_queries, k]
 * @param[out] neighbors      ordinal in the corresponding partition's dataset, shape
 *                            [n_queries, k]
 * @param[out] distances      post-processed distance for each (query, neighbor), shape
 *                            [n_queries, k]
 */
void search(
  raft::resources const& res,
  cuvs::neighbors::cagra::search_params const& params,
  const std::vector<const cuvs::neighbors::cagra::index<uint8_t, uint32_t>*>& indices,
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> partition_ids,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  const std::vector<cuvs::core::bitset_view<std::uint32_t, int64_t>>& partition_bitsets = {});

/**
 * @brief Search multiple CAGRA index partitions concurrently and return the global top-k per
 * query.
 *
 * For each query row in @p queries, the kernel searches all partitions in parallel into an
 * internal intermediate buffer, applies per-partition distance post-processing, runs a batched
 * top-k merge across partitions, and writes the final outputs. The call returns when all work
 * has been submitted to the stream associated with @p res (not necessarily completed); call
 * @c raft::resource::sync_stream on @p res to wait for completion.
 *
 * @note Calling this API with a single partition (@p indices of size 1) still exercises the
 * multi-partition implementation rather than the single-index search overloads above, and the
 * behaviors are not guaranteed to be equivalent.
 *
 * @note All index partitions must use the same distance metric and graph degree; partition sizes
 * may differ. Compressed (VPQ) datasets are not currently supported in multi-partition search, so
 * partitions must be built on in-memory strided datasets.
 *
 * @param[in]  res            raft resources
 * @param[in]  params         search parameters (shared across partitions)
 * @param[in]  indices        CAGRA index objects, one per partition
 * @param[in]  queries        queries matrix, shape [n_queries, dim]; searched against every
 *                            partition
 * @param[out] partition_ids  which partition each neighbor came from, shape [n_queries, k]
 * @param[out] neighbors      ordinal in the corresponding partition's dataset, shape
 *                            [n_queries, k]
 * @param[out] distances      post-processed distance for each (query, neighbor), shape
 *                            [n_queries, k]
 */
void search(
  raft::resources const& res,
  cuvs::neighbors::cagra::search_params const& params,
  const std::vector<const cuvs::neighbors::cagra::index<uint8_t, uint32_t>*>& indices,
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> partition_ids,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  const std::vector<cuvs::core::bitset_view<std::uint32_t, int64_t>>& partition_bitsets = {});

/**
 * @}
 */

/**
 * @defgroup cagra_cpp_serialize CAGRA serialize functions
 * @{
 */

/** Dense dataset storage kind recorded in a serialized CAGRA index. */
enum class serialized_dataset_kind : std::uint32_t {
  /** The serialized index does not contain a dataset payload. */
  none = 0,
  /** Device-resident dataset using CAGRA's padded row layout. */
  device_padded = 1,
  /** Device-resident dataset using its standard row layout. */
  device_standard = 2,
  /** Host-resident dataset using CAGRA's padded row layout. */
  host_padded = 3,
  /** Host-resident dataset using its standard row layout. */
  host_standard = 4,
};

/** Current experimental CAGRA serialization format version. */
inline constexpr int cagra_serialization_version = 6;

// Serialize and deserialize are overloaded for device/host and padded/standard dense indexes.
// They use the same strided dataset payload; the serialized dataset kind selects the matching
// owning dataset type during deserialization. To support a new dataset kind (e.g. vpq_f16_index),
// add matching overloads here and a corresponding deserialize_<kind> in
// detail/dataset_serialize.hpp (dense views use serialize_cagra_dense_dataset).

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::device_padded_index<float>& index,
               bool include_dataset = true);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");

 * cuvs::neighbors::cagra::device_padded_index<float> index;
 * cuvs::neighbors::cagra::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index the cagra index
 * @param[out] out_dataset if non-null, on success may be set to an owned deserialized dataset
 *            when the file includes dataset data; may be left unchanged otherwise. Optional; pass
 *            nullptr to ignore.
 */
void deserialize(
  raft::resources const& handle,
  const std::string& filename,
  cuvs::neighbors::cagra::device_padded_index<float>* index,
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<float, int64_t>>* out_dataset = nullptr);

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::device_padded_index<float>& index,
               bool include_dataset = true);

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * cuvs::neighbors::cagra::device_padded_index<float> index;
 * cuvs::neighbors::cagra::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[out] index the cagra index
 * @param[out] out_dataset if non-null, on success may be set to an owned deserialized dataset
 *            when the stream includes dataset data; may be left unchanged otherwise. Optional; pass
 *            nullptr to ignore.
 */
void deserialize(
  raft::resources const& handle,
  std::istream& is,
  cuvs::neighbors::cagra::device_padded_index<float>* index,
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<float, int64_t>>* out_dataset = nullptr);
/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::device_padded_index<half>& index,
               bool include_dataset = true);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");

 * cuvs::neighbors::cagra::device_padded_index<half> index;
 * cuvs::neighbors::cagra::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index the cagra index
 * @param[out] out_dataset if non-null, on success may be set to an owned deserialized dataset
 *            when the file includes dataset data; may be left unchanged otherwise. Optional; pass
 *            nullptr to ignore.
 */
void deserialize(
  raft::resources const& handle,
  const std::string& filename,
  cuvs::neighbors::cagra::device_padded_index<half>* index,
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<half, int64_t>>* out_dataset = nullptr);

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::device_padded_index<half>& index,
               bool include_dataset = true);

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * cuvs::neighbors::cagra::device_padded_index<half> index;
 * cuvs::neighbors::cagra::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[out] index the cagra index
 * @param[out] out_dataset if non-null, on success may be set to an owned deserialized dataset
 *            when the stream includes dataset data; may be left unchanged otherwise. Optional; pass
 *            nullptr to ignore.
 */
void deserialize(
  raft::resources const& handle,
  std::istream& is,
  cuvs::neighbors::cagra::device_padded_index<half>* index,
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<half, int64_t>>* out_dataset = nullptr);

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::device_padded_index<int8_t>& index,
               bool include_dataset = true);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");

 * cuvs::neighbors::cagra::device_padded_index<int8_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index the cagra index
 * @param[out] out_dataset if non-null, on success may be set to an owned deserialized dataset
 *            when the file includes dataset data; may be left unchanged otherwise. Optional; pass
 *            nullptr to ignore.
 */
void deserialize(
  raft::resources const& handle,
  const std::string& filename,
  cuvs::neighbors::cagra::device_padded_index<int8_t>* index,
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<int8_t, int64_t>>* out_dataset = nullptr);

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::device_padded_index<int8_t>& index,
               bool include_dataset = true);

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * cuvs::neighbors::cagra::device_padded_index<int8_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[out] index the cagra index
 * @param[out] out_dataset if non-null, on success may be set to an owned deserialized dataset
 *            when the stream includes dataset data; may be left unchanged otherwise. Optional; pass
 *            nullptr to ignore.
 */
void deserialize(
  raft::resources const& handle,
  std::istream& is,
  cuvs::neighbors::cagra::device_padded_index<int8_t>* index,
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<int8_t, int64_t>>* out_dataset = nullptr);

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::device_padded_index<uint8_t>& index,
               bool include_dataset = true);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");

 * cuvs::neighbors::cagra::device_padded_index<uint8_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index the cagra index
 * @param[out] out_dataset if non-null, on success may be set to an owned deserialized dataset
 *            when the file includes dataset data; may be left unchanged otherwise. Optional; pass
 *            nullptr to ignore.
 */
void deserialize(
  raft::resources const& handle,
  const std::string& filename,
  cuvs::neighbors::cagra::device_padded_index<uint8_t>* index,
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<uint8_t, int64_t>>* out_dataset = nullptr);

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::device_padded_index<uint8_t>& index,
               bool include_dataset = true);

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * cuvs::neighbors::cagra::device_padded_index<uint8_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[out] index the cagra index
 * @param[out] out_dataset if non-null, on success may be set to an owned deserialized dataset
 *            when the stream includes dataset data; may be left unchanged otherwise. Optional; pass
 *            nullptr to ignore.
 */
void deserialize(
  raft::resources const& handle,
  std::istream& is,
  cuvs::neighbors::cagra::device_padded_index<uint8_t>* index,
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<uint8_t, int64_t>>* out_dataset = nullptr);

void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::device_standard_index<float>& index,
               bool include_dataset = true);

void deserialize(
  raft::resources const& handle,
  const std::string& filename,
  cuvs::neighbors::cagra::device_standard_index<float>* index,
  std::unique_ptr<cuvs::neighbors::device_standard_dataset<float, int64_t>>* out_dataset = nullptr);

void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::device_standard_index<float>& index,
               bool include_dataset = true);

void deserialize(
  raft::resources const& handle,
  std::istream& is,
  cuvs::neighbors::cagra::device_standard_index<float>* index,
  std::unique_ptr<cuvs::neighbors::device_standard_dataset<float, int64_t>>* out_dataset = nullptr);

void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::device_standard_index<half>& index,
               bool include_dataset = true);

void deserialize(
  raft::resources const& handle,
  const std::string& filename,
  cuvs::neighbors::cagra::device_standard_index<half>* index,
  std::unique_ptr<cuvs::neighbors::device_standard_dataset<half, int64_t>>* out_dataset = nullptr);

void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::device_standard_index<half>& index,
               bool include_dataset = true);

void deserialize(
  raft::resources const& handle,
  std::istream& is,
  cuvs::neighbors::cagra::device_standard_index<half>* index,
  std::unique_ptr<cuvs::neighbors::device_standard_dataset<half, int64_t>>* out_dataset = nullptr);

void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::device_standard_index<int8_t>& index,
               bool include_dataset = true);

void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::cagra::device_standard_index<int8_t>* index,
                 std::unique_ptr<cuvs::neighbors::device_standard_dataset<int8_t, int64_t>>*
                   out_dataset = nullptr);

void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::device_standard_index<int8_t>& index,
               bool include_dataset = true);

void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::cagra::device_standard_index<int8_t>* index,
                 std::unique_ptr<cuvs::neighbors::device_standard_dataset<int8_t, int64_t>>*
                   out_dataset = nullptr);

void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::device_standard_index<uint8_t>& index,
               bool include_dataset = true);

void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::cagra::device_standard_index<uint8_t>* index,
                 std::unique_ptr<cuvs::neighbors::device_standard_dataset<uint8_t, int64_t>>*
                   out_dataset = nullptr);

void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::device_standard_index<uint8_t>& index,
               bool include_dataset = true);

void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::cagra::device_standard_index<uint8_t>* index,
                 std::unique_ptr<cuvs::neighbors::device_standard_dataset<uint8_t, int64_t>>*
                   out_dataset = nullptr);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::host_padded_index<float>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::host_padded_index<float>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::host_standard_index<float>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::host_standard_index<float>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::host_padded_index<half>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::host_padded_index<half>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::host_standard_index<half>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::host_standard_index<half>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::host_padded_index<int8_t>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::host_padded_index<int8_t>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::host_standard_index<int8_t>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::host_standard_index<int8_t>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::host_padded_index<uint8_t>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::host_padded_index<uint8_t>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::host_standard_index<uint8_t>& index,
               bool include_dataset = true);

/** @copydoc serialize */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::host_standard_index<uint8_t>& index,
               bool include_dataset = true);

/** @copydoc deserialize */
void deserialize(
  raft::resources const& handle,
  const std::string& filename,
  cuvs::neighbors::cagra::host_padded_index<float>* index,
  std::unique_ptr<cuvs::neighbors::host_padded_dataset<float, int64_t>>* out_dataset = nullptr);

/** @copydoc deserialize */
void deserialize(
  raft::resources const& handle,
  const std::string& filename,
  cuvs::neighbors::cagra::host_standard_index<float>* index,
  std::unique_ptr<cuvs::neighbors::host_standard_dataset<float, int64_t>>* out_dataset = nullptr);

/** @copydoc deserialize */
void deserialize(
  raft::resources const& handle,
  const std::string& filename,
  cuvs::neighbors::cagra::host_padded_index<half>* index,
  std::unique_ptr<cuvs::neighbors::host_padded_dataset<half, int64_t>>* out_dataset = nullptr);

/** @copydoc deserialize */
void deserialize(
  raft::resources const& handle,
  const std::string& filename,
  cuvs::neighbors::cagra::host_standard_index<half>* index,
  std::unique_ptr<cuvs::neighbors::host_standard_dataset<half, int64_t>>* out_dataset = nullptr);

/** @copydoc deserialize */
void deserialize(
  raft::resources const& handle,
  const std::string& filename,
  cuvs::neighbors::cagra::host_padded_index<int8_t>* index,
  std::unique_ptr<cuvs::neighbors::host_padded_dataset<int8_t, int64_t>>* out_dataset = nullptr);

/** @copydoc deserialize */
void deserialize(
  raft::resources const& handle,
  const std::string& filename,
  cuvs::neighbors::cagra::host_standard_index<int8_t>* index,
  std::unique_ptr<cuvs::neighbors::host_standard_dataset<int8_t, int64_t>>* out_dataset = nullptr);

/** @copydoc deserialize */
void deserialize(
  raft::resources const& handle,
  const std::string& filename,
  cuvs::neighbors::cagra::host_padded_index<uint8_t>* index,
  std::unique_ptr<cuvs::neighbors::host_padded_dataset<uint8_t, int64_t>>* out_dataset = nullptr);

/** @copydoc deserialize */
void deserialize(
  raft::resources const& handle,
  const std::string& filename,
  cuvs::neighbors::cagra::host_standard_index<uint8_t>* index,
  std::unique_ptr<cuvs::neighbors::host_standard_dataset<uint8_t, int64_t>>* out_dataset = nullptr);

/**
 * Write the CAGRA built index as a base layer HNSW index to an output stream
 * NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
 *       as the serialization format is not compatible with the original hnswlib.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize_to_hnswlib(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 * @param[in] dataset [optional] host array that stores the dataset, required if the index
 *            does not contain the dataset.
 *
 */
void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::device_padded_index<float>& index,
  std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * Save a CAGRA build index in hnswlib base-layer-only serialized format
 * NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
 *       as the serialization format is not compatible with the original hnswlib.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize_to_hnswlib(handle, filename, index);
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 * @param[in] dataset [optional] host array that stores the dataset, required if the index
 *            does not contain the dataset.
 *
 */
void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::device_padded_index<float>& index,
  std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * Write the CAGRA built index as a base layer HNSW index to an output stream
 * NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
 *       as the serialization format is not compatible with the original hnswlib.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize_to_hnswlib(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 * @param[in] dataset [optional] host array that stores the dataset, required if the index
 *            does not contain the dataset.
 *
 */
void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::device_padded_index<half>& index,
  std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * Save a CAGRA build index in hnswlib base-layer-only serialized format
 * NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
 *       as the serialization format is not compatible with the original hnswlib.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize_to_hnswlib(handle, filename, index);
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 * @param[in] dataset [optional] host array that stores the dataset, required if the index
 *            does not contain the dataset.
 *
 */
void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::device_padded_index<half>& index,
  std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * Write the CAGRA built index as a base layer HNSW index to an output stream
 * NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
 *       as the serialization format is not compatible with the original hnswlib.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize_to_hnswlib(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 * @param[in] dataset [optional] host array that stores the dataset, required if the index
 *            does not contain the dataset.
 *
 */
void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::device_padded_index<int8_t>& index,
  std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * Save a CAGRA build index in hnswlib base-layer-only serialized format
 * NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
 *       as the serialization format is not compatible with the original hnswlib.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize_to_hnswlib(handle, filename, index);
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 * @param[in] dataset [optional] host array that stores the dataset, required if the index
 *            does not contain the dataset.
 *
 */
void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::device_padded_index<int8_t>& index,
  std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * Write the CAGRA built index as a base layer HNSW index to an output stream
 * NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
 *       as the serialization format is not compatible with the original hnswlib.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize_to_hnswlib(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 * @param[in] dataset [optional] host array that stores the dataset, required if the index
 *            does not contain the dataset.
 *
 */
void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::device_padded_index<uint8_t>& index,
  std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * Save a CAGRA build index in hnswlib base-layer-only serialized format
 * NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
 *       as the serialization format is not compatible with the original hnswlib.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize_to_hnswlib(handle, filename, index);
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 * @param[in] dataset [optional] host array that stores the dataset, required if the index
 *            does not contain the dataset.
 *
 */
void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::device_padded_index<uint8_t>& index,
  std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::device_standard_index<float>& index,
  std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::device_standard_index<float>& index,
  std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::device_standard_index<half>& index,
  std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::device_standard_index<half>& index,
  std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::device_standard_index<int8_t>& index,
  std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::device_standard_index<int8_t>& index,
  std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::device_standard_index<uint8_t>& index,
  std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::device_standard_index<uint8_t>& index,
  std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * Write the CAGRA built index as a base layer HNSW index to an output stream.
 * Requires `dataset` — host builds do not store vectors in the index.
 */
void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::host_padded_index<float>& index,
  std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::host_padded_index<float>& index,
  std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::host_padded_index<half>& index,
  std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::host_padded_index<half>& index,
  std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::host_padded_index<int8_t>& index,
  std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::host_padded_index<int8_t>& index,
  std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::host_padded_index<uint8_t>& index,
  std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::host_padded_index<uint8_t>& index,
  std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * Write the CAGRA built index as a base layer HNSW index to an output stream.
 * Requires `dataset` — host builds do not store vectors in the index.
 */
void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::host_standard_index<float>& index,
  std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::host_standard_index<float>& index,
  std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::host_standard_index<half>& index,
  std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::host_standard_index<half>& index,
  std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::host_standard_index<int8_t>& index,
  std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::host_standard_index<int8_t>& index,
  std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  std::ostream& os,
  const cuvs::neighbors::cagra::host_standard_index<uint8_t>& index,
  std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

void serialize_to_hnswlib(
  raft::resources const& handle,
  const std::string& filename,
  const cuvs::neighbors::cagra::host_standard_index<uint8_t>& index,
  std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * @}
 */

/**
 * @defgroup cagra_cpp_index_merge CAGRA index build functions
 * @{
 */

enum class merge_algo {
  /** Select Fastener when its complete preflight succeeds, otherwise preserve rebuild behavior. */
  AUTO,
  /** Require Fastener and reject unsupported configurations. */
  FASTENER,
  /** Concatenate datasets and rebuild the graph. */
  REBUILD,
};

/** C++ controls for physical CAGRA index merge. */
struct merge_params {
  merge_algo algo        = merge_algo::AUTO;
  uint32_t levels        = 2;
  uint32_t root_fanout   = 2;
  uint32_t lower_fanout  = 3;
  double leader_fraction = 0.02;
  uint32_t max_leaders   = 1024;
  uint32_t leaf_size     = 256;
  uint32_t leaf_degree   = 4;
};

/** @brief Merge multiple physical CAGRA indices into one.
 *
 * The overload without `merge_params` uses `merge_algo::AUTO`. AUTO runs Fastener only after a
 * non-mutating preflight validates every input and option; otherwise it calls the existing rebuild
 * implementation. REBUILD always preserves every input dataset.
 *
 * Fastener supports unfiltered, uncompressed `float`, `half`, `int8_t`, and `uint8_t`
 * indices using L2Expanded and `uint32_t` graph IDs. Fastener uses `root_fanout` at the first
 * split and `lower_fanout` at every later split. Fanouts from 1 through 32, leader fractions in
 * (0, 1], leader caps through 8192, leaf sizes from 1 through 256, and leaf degrees from 1
 * through 8 are supported. The configured spill width times `leaf_degree` must not exceed 255.
 * `index_params::graph_degree` is the final output degree.
 *
 * Fastener copies the input datasets into `merged_dataset` and never mutates the input indices.
 * The caller owns `merged_dataset` and must keep it alive for the lifetime of the returned index,
 * which holds only a view of it.
 *
 * @note This API only supports physical merge (`merge_strategy = MERGE_STRATEGY_PHYSICAL`).
 * All input indices must use the same `DatasetViewT` (dense padded or standard device views),
 * and must share one row stride.
 *
 * @param[in] res RAFT resources used for the merge.
 * @param[in] params Parameters for the returned CAGRA index.
 * @param[in] indices CAGRA indices to merge.
 * @param[in] merged_dataset Caller-owned storage for the consolidated dataset. Must have one row
 * per retained vector after applying `row_filter`, and the same dimension and stride as the input
 * datasets.
 * @param[in] row_filter Optional row filter. Any filter selects rebuild in AUTO and is rejected by
 * explicit FASTENER.
 * @return The merged physical CAGRA index.
 */
template <typename T, typename IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
auto merge(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*>& indices,
           DatasetViewT merged_dataset,
           const cuvs::neighbors::filtering::base_filter& row_filter =
             cuvs::neighbors::filtering::none_sample_filter{})
  -> cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>;

/** @copydoc merge
 * @param[in] merge_params Parameters for the merge, including the algorithm selection.
 */
template <typename T, typename IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
auto merge(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*>& indices,
           DatasetViewT merged_dataset,
           const cuvs::neighbors::cagra::merge_params& merge_params,
           const cuvs::neighbors::filtering::base_filter& row_filter =
             cuvs::neighbors::filtering::none_sample_filter{})
  -> cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>;

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
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed CAGRA MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<cagra::index_params>& index_params,
           cuvs::neighbors::host_standard_dataset_view<float, int64_t> const& index_dataset)
  -> cuvs::neighbors::mg_index<cagra::device_standard_index<float, uint32_t>, float, uint32_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed CAGRA MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<cagra::index_params>& index_params,
           cuvs::neighbors::host_standard_dataset_view<half, int64_t> const& index_dataset)
  -> cuvs::neighbors::mg_index<cagra::device_standard_index<half, uint32_t>, half, uint32_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed CAGRA MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<cagra::index_params>& index_params,
           cuvs::neighbors::host_standard_dataset_view<int8_t, int64_t> const& index_dataset)
  -> cuvs::neighbors::mg_index<cagra::device_standard_index<int8_t, uint32_t>, int8_t, uint32_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed CAGRA MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<cagra::index_params>& index_params,
           cuvs::neighbors::host_standard_dataset_view<uint8_t, int64_t> const& index_dataset)
  -> cuvs::neighbors::mg_index<cagra::device_standard_index<uint8_t, uint32_t>, uint8_t, uint32_t>;

auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<cagra::index_params>& index_params,
           cuvs::neighbors::host_padded_dataset_view<float, int64_t> const& index_dataset)
  -> cuvs::neighbors::mg_index<cagra::device_padded_index<float, uint32_t>, float, uint32_t>;

auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<cagra::index_params>& index_params,
           cuvs::neighbors::host_padded_dataset_view<half, int64_t> const& index_dataset)
  -> cuvs::neighbors::mg_index<cagra::device_padded_index<half, uint32_t>, half, uint32_t>;

auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<cagra::index_params>& index_params,
           cuvs::neighbors::host_padded_dataset_view<int8_t, int64_t> const& index_dataset)
  -> cuvs::neighbors::mg_index<cagra::device_padded_index<int8_t, uint32_t>, int8_t, uint32_t>;

auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<cagra::index_params>& index_params,
           cuvs::neighbors::host_padded_dataset_view<uint8_t, int64_t> const& index_dataset)
  -> cuvs::neighbors::mg_index<cagra::device_padded_index<uint8_t, uint32_t>, uint8_t, uint32_t>;

/**
 * @brief Consume a standard MG CAGRA index and attach a padded dataset for search.
 *
 * This moves each rank-local CAGRA graph into the returned padded MG index.
 */
auto update_dataset(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_standard_index<float, uint32_t>, float, uint32_t>&& idx,
  cuvs::neighbors::device_padded_dataset_view<float, int64_t> const& padded_dataset)
  -> cuvs::neighbors::mg_index<cagra::device_padded_index<float, uint32_t>, float, uint32_t>;

auto update_dataset(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_standard_index<half, uint32_t>, half, uint32_t>&& idx,
  cuvs::neighbors::device_padded_dataset_view<half, int64_t> const& padded_dataset)
  -> cuvs::neighbors::mg_index<cagra::device_padded_index<half, uint32_t>, half, uint32_t>;

auto update_dataset(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_standard_index<int8_t, uint32_t>, int8_t, uint32_t>&& idx,
  cuvs::neighbors::device_padded_dataset_view<int8_t, int64_t> const& padded_dataset)
  -> cuvs::neighbors::mg_index<cagra::device_padded_index<int8_t, uint32_t>, int8_t, uint32_t>;

auto update_dataset(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_standard_index<uint8_t, uint32_t>, uint8_t, uint32_t>&&
    idx,
  cuvs::neighbors::device_padded_dataset_view<uint8_t, int64_t> const& padded_dataset)
  -> cuvs::neighbors::mg_index<cagra::device_padded_index<uint8_t, uint32_t>, uint8_t, uint32_t>;

/**
 * @brief Update an existing padded MG CAGRA index with a padded dataset of the same layout.
 */
void update_dataset(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_padded_index<float, uint32_t>, float, uint32_t>& idx,
  cuvs::neighbors::device_padded_dataset_view<float, int64_t> const& padded_dataset);

void update_dataset(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_padded_index<half, uint32_t>, half, uint32_t>& idx,
  cuvs::neighbors::device_padded_dataset_view<half, int64_t> const& padded_dataset);

void update_dataset(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_padded_index<int8_t, uint32_t>, int8_t, uint32_t>& idx,
  cuvs::neighbors::device_padded_dataset_view<int8_t, int64_t> const& padded_dataset);

void update_dataset(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_padded_index<uint8_t, uint32_t>, uint8_t, uint32_t>& idx,
  cuvs::neighbors::device_padded_dataset_view<uint8_t, int64_t> const& padded_dataset);

/// \defgroup mg_cpp_index_extend ANN MG index extend

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * cuvs::neighbors::cagra::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors host dataset view [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_padded_index<float, uint32_t>, float, uint32_t>& index,
  cuvs::neighbors::host_padded_dataset_view<float, int64_t> new_vectors,
  std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * cuvs::neighbors::cagra::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors host dataset view [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_standard_index<float, uint32_t>, float, uint32_t>& index,
  cuvs::neighbors::host_standard_dataset_view<float, int64_t> new_vectors,
  std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * cuvs::neighbors::cagra::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors host dataset view [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_padded_index<half, uint32_t>, half, uint32_t>& index,
  cuvs::neighbors::host_padded_dataset_view<half, int64_t> new_vectors,
  std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/** @copydoc extend */
void extend(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_standard_index<half, uint32_t>, half, uint32_t>& index,
  cuvs::neighbors::host_standard_dataset_view<half, int64_t> new_vectors,
  std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * cuvs::neighbors::cagra::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors host dataset view [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_padded_index<int8_t, uint32_t>, int8_t, uint32_t>& index,
  cuvs::neighbors::host_padded_dataset_view<int8_t, int64_t> new_vectors,
  std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/** @copydoc extend */
void extend(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_standard_index<int8_t, uint32_t>, int8_t, uint32_t>&
    index,
  cuvs::neighbors::host_standard_dataset_view<int8_t, int64_t> new_vectors,
  std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * cuvs::neighbors::cagra::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors host dataset view [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_padded_index<uint8_t, uint32_t>, uint8_t, uint32_t>&
    index,
  cuvs::neighbors::host_padded_dataset_view<uint8_t, int64_t> new_vectors,
  std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/** @copydoc extend */
void extend(
  const raft::resources& clique,
  cuvs::neighbors::mg_index<cagra::device_standard_index<uint8_t, uint32_t>, uint8_t, uint32_t>&
    index,
  cuvs::neighbors::host_standard_dataset_view<uint8_t, int64_t> new_vectors,
  std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/// \defgroup mg_cpp_index_search ANN MG index search

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<cagra::search_params> search_params;
 * cuvs::neighbors::cagra::search(clique, index, search_params, queries, neighbors,
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
  const cuvs::neighbors::mg_index<cagra::device_padded_index<float, uint32_t>, float, uint32_t>&
    index,
  const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
  raft::host_matrix_view<const float, int64_t, row_major> queries,
  raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances);

/** @copydoc search */
void search(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<cagra::device_standard_index<float, uint32_t>, float, uint32_t>&
    index,
  const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
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
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<cagra::search_params> search_params;
 * cuvs::neighbors::cagra::search(clique, index, search_params, queries, neighbors,
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
  const cuvs::neighbors::mg_index<cagra::device_padded_index<half, uint32_t>, half, uint32_t>&
    index,
  const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
  raft::host_matrix_view<const half, int64_t, row_major> queries,
  raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances);

/** @copydoc search */
void search(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<cagra::device_standard_index<half, uint32_t>, half, uint32_t>&
    index,
  const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
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
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<cagra::search_params> search_params;
 * cuvs::neighbors::cagra::search(clique, index, search_params, queries, neighbors,
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
  const cuvs::neighbors::mg_index<cagra::device_padded_index<int8_t, uint32_t>, int8_t, uint32_t>&
    index,
  const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
  raft::host_matrix_view<const int8_t, int64_t, row_major> queries,
  raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances);

/** @copydoc search */
void search(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<cagra::device_standard_index<int8_t, uint32_t>, int8_t, uint32_t>&
    index,
  const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
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
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<cagra::search_params> search_params;
 * cuvs::neighbors::cagra::search(clique, index, search_params, queries, neighbors,
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
  const cuvs::neighbors::mg_index<cagra::device_padded_index<uint8_t, uint32_t>, uint8_t, uint32_t>&
    index,
  const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
  raft::host_matrix_view<const uint8_t, int64_t, row_major> queries,
  raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances);

/** @copydoc search */
void search(const raft::resources& clique,
            const cuvs::neighbors::
              mg_index<cagra::device_standard_index<uint8_t, uint32_t>, uint8_t, uint32_t>& index,
            const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<cagra::search_params> search_params;
 * cuvs::neighbors::cagra::search(clique, index, search_params, queries, neighbors,
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
  const cuvs::neighbors::mg_index<cagra::device_padded_index<float, uint32_t>, float, uint32_t>&
    index,
  const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
  raft::host_matrix_view<const float, int64_t, row_major> queries,
  raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances);

/** @copydoc search */
void search(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<cagra::device_standard_index<float, uint32_t>, float, uint32_t>&
    index,
  const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
  raft::host_matrix_view<const float, int64_t, row_major> queries,
  raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<cagra::search_params> search_params;
 * cuvs::neighbors::cagra::search(clique, index, search_params, queries, neighbors,
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
  const cuvs::neighbors::mg_index<cagra::device_padded_index<half, uint32_t>, half, uint32_t>&
    index,
  const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
  raft::host_matrix_view<const half, int64_t, row_major> queries,
  raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances);

/** @copydoc search */
void search(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<cagra::device_standard_index<half, uint32_t>, half, uint32_t>&
    index,
  const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
  raft::host_matrix_view<const half, int64_t, row_major> queries,
  raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<cagra::search_params> search_params;
 * cuvs::neighbors::cagra::search(clique, index, search_params, queries, neighbors,
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
  const cuvs::neighbors::mg_index<cagra::device_padded_index<int8_t, uint32_t>, int8_t, uint32_t>&
    index,
  const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
  raft::host_matrix_view<const int8_t, int64_t, row_major> queries,
  raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances);

/** @copydoc search */
void search(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<cagra::device_standard_index<int8_t, uint32_t>, int8_t, uint32_t>&
    index,
  const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
  raft::host_matrix_view<const int8_t, int64_t, row_major> queries,
  raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<cagra::search_params> search_params;
 * cuvs::neighbors::cagra::search(clique, index, search_params, queries, neighbors,
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
  const cuvs::neighbors::mg_index<cagra::device_padded_index<uint8_t, uint32_t>, uint8_t, uint32_t>&
    index,
  const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
  raft::host_matrix_view<const uint8_t, int64_t, row_major> queries,
  raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances);

/** @copydoc search */
void search(const raft::resources& clique,
            const cuvs::neighbors::
              mg_index<cagra::device_standard_index<uint8_t, uint32_t>, uint8_t, uint32_t>& index,
            const cuvs::neighbors::mg_search_params<cagra::search_params>& search_params,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> queries,
            raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances);

/// \defgroup mg_cpp_serialize ANN MG index serialization

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::cagra::serialize(clique, index, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<cagra::device_padded_index<float, uint32_t>, float, uint32_t>&
    index,
  const std::string& filename);

/** @copydoc serialize */
void serialize(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<cagra::device_standard_index<float, uint32_t>, float, uint32_t>&
    index,
  const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::cagra::serialize(clique, index, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<cagra::device_padded_index<half, uint32_t>, half, uint32_t>&
    index,
  const std::string& filename);

/** @copydoc serialize */
void serialize(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<cagra::device_standard_index<half, uint32_t>, half, uint32_t>&
    index,
  const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::cagra::serialize(clique, index, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<cagra::device_padded_index<int8_t, uint32_t>, int8_t, uint32_t>&
    index,
  const std::string& filename);

/** @copydoc serialize */
void serialize(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<cagra::device_standard_index<int8_t, uint32_t>, int8_t, uint32_t>&
    index,
  const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::cagra::serialize(clique, index, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<cagra::device_padded_index<uint8_t, uint32_t>, uint8_t, uint32_t>&
    index,
  const std::string& filename);

/** @copydoc serialize */
void serialize(const raft::resources& clique,
               const cuvs::neighbors::mg_index<cagra::device_standard_index<uint8_t, uint32_t>,
                                               uint8_t,
                                               uint32_t>& index,
               const std::string& filename);

/// \defgroup mg_cpp_deserialize ANN MG index deserialization

/// \ingroup mg_cpp_deserialize
/**
 * @brief Deserializes a CAGRA multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::cagra::serialize(clique, index, filename);
 * cuvs::neighbors::mg_index<cagra::device_standard_index<float, uint32_t>, float, uint32_t>
 *   new_index(clique, REPLICATED);
 * cuvs::neighbors::cagra::deserialize<float, uint32_t>(clique, filename, &new_index);
 *
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] filename path to the file to be deserialized
 *
 */
template <typename T, typename IdxT>
void deserialize(const raft::resources& clique,
                 const std::string& filename,
                 cuvs::neighbors::mg_index<cagra::device_standard_index<T, IdxT>, T, IdxT>* index);

template <typename T, typename IdxT>
void deserialize(const raft::resources& clique,
                 const std::string& filename,
                 cuvs::neighbors::mg_index<cagra::device_padded_index<T, IdxT>, T, IdxT>* index);

/// \defgroup mg_cpp_distribute ANN MG local index distribution

/// \ingroup mg_cpp_distribute
/**
 * @brief Replicates a locally built and serialized CAGRA index to all GPUs to form a distributed
 * multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::cagra::index_params index_params;
 * auto index = cuvs::neighbors::cagra::build(clique, index_params, index_dataset);
 * const std::string filename = "local_index.cuvs";
 * cuvs::neighbors::cagra::serialize(clique, filename, index);
 * cuvs::neighbors::mg_index<cagra::device_standard_index<float, uint32_t>, float, uint32_t>
 *   distributed_index(clique, REPLICATED);
 * cuvs::neighbors::cagra::distribute<float, uint32_t>(clique, filename, &distributed_index);
 *
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] filename path to the file to be deserialized : a local index
 *
 */
template <typename T, typename IdxT>
void distribute(const raft::resources& clique,
                const std::string& filename,
                cuvs::neighbors::mg_index<cagra::device_standard_index<T, IdxT>, T, IdxT>* index);

template <typename T, typename IdxT>
void distribute(const raft::resources& clique,
                const std::string& filename,
                cuvs::neighbors::mg_index<cagra::device_padded_index<T, IdxT>, T, IdxT>* index);

/**
 * @brief Build a kNN graph using IVF-PQ.
 *
 * The kNN graph is the first building block for CAGRA index.
 *
 * The output is a dense matrix that stores the neighbor indices for each point in the dataset.
 * Each point has the same number of neighbors.
 *
 * See [cagra::build](#cagra::build) for an alternative method.
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - InnerProduct
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // raft::host_matrix_view<const float, int64_t, raft::row_major> dataset;
 *   auto metric       = cuvs::distance::DistanceType::L2Expanded;
 *   auto build_params = cagra::graph_build_params::ivf_pq_params(dataset.extents(), metric);
 *   auto knn_graph = raft::make_host_matrix<uint32_t, int64_t>(dataset.extent(0), 128);
 *   // create knn graph
 *   cagra::build_knn_graph(res, dataset, knn_graph.view(), build_params);
 *   auto optimized_graph = raft::make_host_matrix<uint32_t, int64_t>(dataset.extent(0), 64);
 *   cagra::helpers::optimize(res, knn_graph.view(), optimized_graph.view());
 *   // Construct an index from dataset and optimized knn_graph
 *   auto dataset_view = make_host_standard_dataset_view(dataset);
 *   auto index = cagra::host_standard_index<float, uint32_t>(
 *     res, metric, dataset_view, raft::make_const_mdspan(optimized_graph.view()));
 * @endcode
 *
 * @param[in] res raft resources
 * @param[in] dataset a matrix view (host or device) to a row-major matrix [n_rows, dim]
 * @param[out] knn_graph a host matrix view to store the output knn graph [n_rows, graph_degree]
 * @param[in] build_params ivf-pq parameters for graph build
 */
void build_knn_graph(raft::resources const& res,
                     raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
                     raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
                     cuvs::neighbors::cagra::graph_build_params::ivf_pq_params build_params);

/**
 * @brief Build a kNN graph using IVF-PQ.
 *
 * The kNN graph is the first building block for CAGRA index.
 *
 * The output is a dense matrix that stores the neighbor indices for each point in the dataset.
 * Each point has the same number of neighbors.
 *
 * See [cagra::build](#cagra::build) for an alternative method.
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - InnerProduct
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // raft::host_matrix_view<const half, int64_t, raft::row_major> dataset;
 *   auto metric       = cuvs::distance::DistanceType::L2Expanded;
 *   auto build_params = cagra::graph_build_params::ivf_pq_params(dataset.extents(), metric);
 *   auto knn_graph = raft::make_host_matrix<uint32_t, int64_t>(dataset.extent(0), 128);
 *   // create knn graph
 *   cagra::build_knn_graph(res, dataset, knn_graph.view(), build_params);
 *   auto optimized_graph = raft::make_host_matrix<uint32_t, int64_t>(dataset.extent(0), 64);
 *   cagra::helpers::optimize(res, knn_graph.view(), optimized_graph.view());
 *   // Construct an index from dataset and optimized knn_graph
 *   auto dataset_view = make_host_standard_dataset_view(dataset);
 *   auto index = cagra::host_standard_index<half, uint32_t>(
 *     res, metric, dataset_view, raft::make_const_mdspan(optimized_graph.view()));
 * @endcode
 *
 * @param[in] res raft resources
 * @param[in] dataset a matrix view (host or device) to a row-major matrix [n_rows, dim]
 * @param[out] knn_graph a host matrix view to store the output knn graph [n_rows, graph_degree]
 * @param[in] build_params ivf-pq parameters for graph build
 */
void build_knn_graph(raft::resources const& res,
                     raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
                     raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
                     cuvs::neighbors::cagra::graph_build_params::ivf_pq_params build_params);

/**
 * @brief Build a kNN graph using IVF-PQ.
 *
 * The kNN graph is the first building block for CAGRA index.
 *
 * The output is a dense matrix that stores the neighbor indices for each point in the dataset.
 * Each point has the same number of neighbors.
 *
 * See [cagra::build](#cagra::build) for an alternative method.
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - InnerProduct
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset;
 *   auto metric       = cuvs::distance::DistanceType::L2Expanded;
 *   auto build_params = cagra::graph_build_params::ivf_pq_params(dataset.extents(), metric);
 *   auto knn_graph = raft::make_host_matrix<uint32_t, int64_t>(dataset.extent(0), 128);
 *   // create knn graph
 *   cagra::build_knn_graph(res, dataset, knn_graph.view(), build_params);
 *   auto optimized_graph = raft::make_host_matrix<uint32_t, int64_t>(dataset.extent(0), 64);
 *   cagra::helpers::optimize(res, knn_graph.view(), optimized_graph.view());
 *   // Construct an index from dataset and optimized knn_graph
 *   auto dataset_view = make_host_standard_dataset_view(dataset);
 *   auto index = cagra::host_standard_index<int8_t, uint32_t>(
 *     res, metric, dataset_view, raft::make_const_mdspan(optimized_graph.view()));
 * @endcode
 *
 * @param[in] res raft resources
 * @param[in] dataset a matrix view (host or device) to a row-major matrix [n_rows, dim]
 * @param[out] knn_graph a host matrix view to store the output knn graph [n_rows, graph_degree]
 * @param[in] build_params ivf-pq parameters for graph build
 */
void build_knn_graph(raft::resources const& res,
                     raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
                     raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
                     cuvs::neighbors::cagra::graph_build_params::ivf_pq_params build_params);

/**
 * @brief Build a kNN graph using IVF-PQ.
 *
 * The kNN graph is the first building block for CAGRA index.
 *
 * The output is a dense matrix that stores the neighbor indices for each point in the dataset.
 * Each point has the same number of neighbors.
 *
 * See [cagra::build](#cagra::build) for an alternative method.
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - InnerProduct
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset;
 *   auto metric       = cuvs::distance::DistanceType::L2Expanded;
 *   auto build_params = cagra::graph_build_params::ivf_pq_params(dataset.extents(), metric);
 *   auto knn_graph = raft::make_host_matrix<uint32_t, int64_t>(dataset.extent(0), 128);
 *   // create knn graph
 *   cagra::build_knn_graph(res, dataset, knn_graph.view(), build_params);
 *   auto optimized_graph = raft::make_host_matrix<uint32_t, int64_t>(dataset.extent(0), 64);
 *   cagra::helpers::optimize(res, knn_graph.view(), optimized_graph.view());
 *   // Construct an index from dataset and optimized knn_graph
 *   auto dataset_view = make_host_standard_dataset_view(dataset);
 *   auto index = cagra::host_standard_index<uint8_t, uint32_t>(
 *     res, metric, dataset_view, raft::make_const_mdspan(optimized_graph.view()));
 * @endcode
 *
 * @param[in] res raft resources
 * @param[in] dataset a matrix view (host or device) to a row-major matrix [n_rows, dim]
 * @param[out] knn_graph a host matrix view to store the output knn graph [n_rows, graph_degree]
 * @param[in] build_params ivf-pq parameters for graph build
 */
void build_knn_graph(raft::resources const& res,
                     raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
                     raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
                     cuvs::neighbors::cagra::graph_build_params::ivf_pq_params build_params);

namespace detail {

/**
 * @brief Internal helper to transfer ACE disk FDs between CAGRA indexes.
 *
 * @internal
 */
struct fd_transfer {
  template <typename T,
            typename IdxT,
            cuvs::neighbors::ann_dataset_view SrcDatasetViewT,
            cuvs::neighbors::ann_dataset_view DstDatasetViewT>
  static inline void steal_disk_fds_to(raft::resources const& res,
                                       index<T, IdxT, SrcDatasetViewT>& src,
                                       index<T, IdxT, DstDatasetViewT>& dst)
  {
    if (src.dataset_fd().has_value()) {
      dst.update_dataset(res, std::move(*src.steal_dataset_fd_()));
    }
    if (src.graph_fd().has_value()) { dst.update_graph(res, std::move(*src.steal_graph_fd_())); }
    if (src.mapping_fd().has_value()) {
      dst.update_mapping(res, std::move(*src.steal_mapping_fd_()));
    }
  }
};

}  // namespace detail

/**
 * @brief Convert a standard-device index into a padded-device index and attach padded dataset.
 *
 * CAGRA search requires padded device layout. This helper copies graph/source-indices from
 * `standard_idx` into a new `device_padded_index` and attaches `padded_dataset`.
 *
 * @param[in] res             RAFT resources
 * @param[in] standard_idx    index returned by `build` with a standard device dataset view
 * @param[in] padded_dataset  device padded dataset view (caller owns underlying memory)
 * @return device padded index with graph and dataset ready for search
 */
template <typename T, typename IdxT>
auto convert_standard_to_padded_index(
  raft::resources const& res,
  index<T, IdxT, cuvs::neighbors::device_standard_dataset_view<T, int64_t>> const& standard_idx,
  cuvs::neighbors::device_padded_dataset_view<T, int64_t> const& padded_dataset)
  -> device_padded_index<T, IdxT>
{
  RAFT_EXPECTS(padded_dataset.n_rows() == standard_idx.size(),
               "Padded dataset row count must match the index size");

  using GraphIndexType =
    typename index<T, IdxT, cuvs::neighbors::device_standard_dataset_view<T, int64_t>>::
      graph_index_type;
  auto graph_host = raft::make_host_matrix<GraphIndexType, int64_t>(standard_idx.graph().extent(0),
                                                                    standard_idx.graph().extent(1));
  if (standard_idx.graph().size() > 0) {
    raft::copy(graph_host.data_handle(),
               standard_idx.graph().data_handle(),
               standard_idx.graph().size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
  }
  device_padded_index<T, IdxT> out(
    res, standard_idx.metric(), padded_dataset, raft::make_const_mdspan(graph_host.view()));
  if (standard_idx.source_indices().has_value()) {
    out.update_source_indices(res, standard_idx.source_indices().value());
  }
  return out;
}

auto update_dataset(
  raft::resources const& res,
  index<float, uint32_t, host_standard_dataset_view<float, int64_t>>&& cagra_index,
  device_padded_dataset_view<float, int64_t> dataset)
  -> index<float, uint32_t, device_padded_dataset_view<float, int64_t>>;
auto update_dataset(raft::resources const& res,
                    index<half, uint32_t, host_standard_dataset_view<half, int64_t>>&& cagra_index,
                    device_padded_dataset_view<half, int64_t> dataset)
  -> index<half, uint32_t, device_padded_dataset_view<half, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<int8_t, uint32_t, host_standard_dataset_view<int8_t, int64_t>>&& cagra_index,
  device_padded_dataset_view<int8_t, int64_t> dataset)
  -> index<int8_t, uint32_t, device_padded_dataset_view<int8_t, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<uint8_t, uint32_t, host_standard_dataset_view<uint8_t, int64_t>>&& cagra_index,
  device_padded_dataset_view<uint8_t, int64_t> dataset)
  -> index<uint8_t, uint32_t, device_padded_dataset_view<uint8_t, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<float, uint32_t, host_standard_dataset_view<float, int64_t>>&& cagra_index,
  device_standard_dataset_view<float, int64_t> dataset)
  -> index<float, uint32_t, device_standard_dataset_view<float, int64_t>>;
auto update_dataset(raft::resources const& res,
                    index<half, uint32_t, host_standard_dataset_view<half, int64_t>>&& cagra_index,
                    device_standard_dataset_view<half, int64_t> dataset)
  -> index<half, uint32_t, device_standard_dataset_view<half, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<int8_t, uint32_t, host_standard_dataset_view<int8_t, int64_t>>&& cagra_index,
  device_standard_dataset_view<int8_t, int64_t> dataset)
  -> index<int8_t, uint32_t, device_standard_dataset_view<int8_t, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<uint8_t, uint32_t, host_standard_dataset_view<uint8_t, int64_t>>&& cagra_index,
  device_standard_dataset_view<uint8_t, int64_t> dataset)
  -> index<uint8_t, uint32_t, device_standard_dataset_view<uint8_t, int64_t>>;
auto update_dataset(raft::resources const& res,
                    index<float, uint32_t, host_padded_dataset_view<float, int64_t>>&& cagra_index,
                    device_padded_dataset_view<float, int64_t> dataset)
  -> index<float, uint32_t, device_padded_dataset_view<float, int64_t>>;
auto update_dataset(raft::resources const& res,
                    index<half, uint32_t, host_padded_dataset_view<half, int64_t>>&& cagra_index,
                    device_padded_dataset_view<half, int64_t> dataset)
  -> index<half, uint32_t, device_padded_dataset_view<half, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<int8_t, uint32_t, host_padded_dataset_view<int8_t, int64_t>>&& cagra_index,
  device_padded_dataset_view<int8_t, int64_t> dataset)
  -> index<int8_t, uint32_t, device_padded_dataset_view<int8_t, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<uint8_t, uint32_t, host_padded_dataset_view<uint8_t, int64_t>>&& cagra_index,
  device_padded_dataset_view<uint8_t, int64_t> dataset)
  -> index<uint8_t, uint32_t, device_padded_dataset_view<uint8_t, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<float, uint32_t, device_standard_dataset_view<float, int64_t>>&& cagra_index,
  device_padded_dataset_view<float, int64_t> dataset)
  -> index<float, uint32_t, device_padded_dataset_view<float, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<half, uint32_t, device_standard_dataset_view<half, int64_t>>&& cagra_index,
  device_padded_dataset_view<half, int64_t> dataset)
  -> index<half, uint32_t, device_padded_dataset_view<half, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<int8_t, uint32_t, device_standard_dataset_view<int8_t, int64_t>>&& cagra_index,
  device_padded_dataset_view<int8_t, int64_t> dataset)
  -> index<int8_t, uint32_t, device_padded_dataset_view<int8_t, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<uint8_t, uint32_t, device_standard_dataset_view<uint8_t, int64_t>>&& cagra_index,
  device_padded_dataset_view<uint8_t, int64_t> dataset)
  -> index<uint8_t, uint32_t, device_padded_dataset_view<uint8_t, int64_t>>;

auto update_dataset(
  raft::resources const& res,
  index<float, uint32_t, device_standard_dataset_view<float, int64_t>>&& cagra_index,
  device_standard_dataset_view<float, int64_t> dataset)
  -> index<float, uint32_t, device_standard_dataset_view<float, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<half, uint32_t, device_standard_dataset_view<half, int64_t>>&& cagra_index,
  device_standard_dataset_view<half, int64_t> dataset)
  -> index<half, uint32_t, device_standard_dataset_view<half, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<int8_t, uint32_t, device_standard_dataset_view<int8_t, int64_t>>&& cagra_index,
  device_standard_dataset_view<int8_t, int64_t> dataset)
  -> index<int8_t, uint32_t, device_standard_dataset_view<int8_t, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<uint8_t, uint32_t, device_standard_dataset_view<uint8_t, int64_t>>&& cagra_index,
  device_standard_dataset_view<uint8_t, int64_t> dataset)
  -> index<uint8_t, uint32_t, device_standard_dataset_view<uint8_t, int64_t>>;

auto update_dataset(
  raft::resources const& res,
  index<float, uint32_t, device_padded_dataset_view<float, int64_t>>&& cagra_index,
  device_padded_dataset_view<float, int64_t> dataset)
  -> index<float, uint32_t, device_padded_dataset_view<float, int64_t>>;
auto update_dataset(raft::resources const& res,
                    index<half, uint32_t, device_padded_dataset_view<half, int64_t>>&& cagra_index,
                    device_padded_dataset_view<half, int64_t> dataset)
  -> index<half, uint32_t, device_padded_dataset_view<half, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<int8_t, uint32_t, device_padded_dataset_view<int8_t, int64_t>>&& cagra_index,
  device_padded_dataset_view<int8_t, int64_t> dataset)
  -> index<int8_t, uint32_t, device_padded_dataset_view<int8_t, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<uint8_t, uint32_t, device_padded_dataset_view<uint8_t, int64_t>>&& cagra_index,
  device_padded_dataset_view<uint8_t, int64_t> dataset)
  -> index<uint8_t, uint32_t, device_padded_dataset_view<uint8_t, int64_t>>;

auto update_dataset(
  raft::resources const& res,
  index<float, uint32_t, device_padded_dataset_view<float, int64_t>>&& cagra_index,
  device_vpq_dataset_view<half, int64_t> dataset)
  -> index<float, uint32_t, device_vpq_dataset_view<half, int64_t>>;
auto update_dataset(raft::resources const& res,
                    index<half, uint32_t, device_padded_dataset_view<half, int64_t>>&& cagra_index,
                    device_vpq_dataset_view<half, int64_t> dataset)
  -> index<half, uint32_t, device_vpq_dataset_view<half, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<int8_t, uint32_t, device_padded_dataset_view<int8_t, int64_t>>&& cagra_index,
  device_vpq_dataset_view<half, int64_t> dataset)
  -> index<int8_t, uint32_t, device_vpq_dataset_view<half, int64_t>>;
auto update_dataset(
  raft::resources const& res,
  index<uint8_t, uint32_t, device_padded_dataset_view<uint8_t, int64_t>>&& cagra_index,
  device_vpq_dataset_view<half, int64_t> dataset)
  -> index<uint8_t, uint32_t, device_vpq_dataset_view<half, int64_t>>;

}  // namespace cagra
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace cagra {
namespace helpers {

/** Calculates the workspace for graph optimization
 *
 * @param[in] n_rows number of rows in the dataset (or number of points in the graph)
 * @param[in] graph_degree degree of the output graph
 * @param[in] intermediate_graph_degree degree of the input graph for the optimization process
 * @param[in] index_size
 * @param[in] mst_optimize whether to use MST optimization
 * @return tuple of [host_size, device_size, host_fixed_size, device_fixed_size] memory sizes in
 * bytes
 */
std::tuple<size_t, size_t, size_t, size_t> optimize_workspace_size(size_t n_rows,
                                                                   size_t graph_degree,
                                                                   size_t intermediate_degree,
                                                                   size_t index_size,
                                                                   bool mst_optimize = false);

/**
 * Calculate memory usage of CAGRA build.
 *
 * @param[in] res raft resource
 * @param[in] dataset shape of the dataset
 * @param[in] dtype element type of the dataset
 *            (e.g. `CUDA_R_32F`, `CUDA_R_16F`, `CUDA_R_8I`, `CUDA_R_8U`)
 * @param[in] cparams CAGRA index building parameters
 *
 * @return pair of [host_size, device_size] memory sizes in bytes
 */
std::pair<size_t, size_t> cagra_build_mem_usage(raft::resources const& res,
                                                raft::matrix_extent<int64_t> dataset,
                                                cudaDataType_t dtype,
                                                cuvs::neighbors::cagra::index_params cparams);

/**
 * @brief Optimize a KNN graph into a CAGRA graph.
 *
 * This function optimizes a k-NN graph to create a CAGRA graph.
 * The input/output graphs must be on host memory.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   auto h_knn = raft::make_host_matrix<uint32_t, int64_t>(N, K_in);
 *   // Fill h_knn with KNN graph
 *   auto h_out = raft::make_host_matrix<uint32_t, int64_t>(N, K_out);
 *   cuvs::neighbors::cagra::helpers::optimize(res, h_knn.view(), h_out.view());
 * @endcode
 *
 * @param[in] handle RAFT resources
 * @param[in] knn_graph Input KNN graph on host [n_rows, k_in]
 * @param[out] new_graph Output CAGRA graph on host [n_rows, k_out]
 */
void optimize(raft::resources const& handle,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> new_graph);

}  // namespace helpers
}  // namespace cagra
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
