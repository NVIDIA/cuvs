/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.hpp"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <cuvs/util/file_io.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resource/stream_view.hpp>
#include <raft/core/serialize.hpp>

#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuvs/core/export.hpp>
#include <optional>
#include <string>
#include <variant>

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
/**
 * @defgroup cagra_cpp_index_params CAGRA index build parameters
 * @{
 */

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
  /**
   * Specify compression parameters if compression is desired. If set, overrides the
   * attach_dataset_on_build (and the compressed dataset is always added to the index).
   */
  std::optional<cuvs::neighbors::vpq_params> compression = std::nullopt;

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
  std::variant<std::monostate,
               graph_build_params::ivf_pq_params,
               graph_build_params::nn_descent_params,
               graph_build_params::ace_params,
               graph_build_params::iterative_search_params>
    graph_build_params;
  /**
   * Whether to use MST optimization to guarantee graph connectivity.
   */
  bool guarantee_connectivity = false;

  /**
   * Whether to add the dataset content to the index, i.e.:
   *
   *  - `true` means the index is filled with the dataset vectors and ready to search after calling
   * `build` provided there is enough memory available.
   *  - `false` means `build` only builds the graph and the user is expected to
   * update the dataset using cuvs::neighbors::cagra::update_dataset.
   *
   * Regardless of the value of `attach_dataset_on_build`, the search graph is created using all
   * the vectors in the dataset.  Setting `attach_dataset_on_build = false` can be useful if
   * the user needs to build only the search graph but does not intend to search it using CAGRA
   * (e.g. search using another graph search algorithm), or if specific memory placement options
   * need to be applied on the dataset before it is attached to the index using `update_dataset`.
   * API.
   * @code{.cpp}
   *   auto dataset = raft::make_device_matrix<float, int64_t>(res, n_rows, n_cols);
   *   // use default index_parameters
   *   cagra::index_params index_params;
   *   // update index_params to only build the CAGRA graph
   *   index_params.attach_dataset_on_build = false;
   *   auto index = cagra::build(res, index_params, dataset.view());
   *   // assert that the dataset is not attached to the index
   *   ASSERT(index.dataset().extent(0) == 0);
   *   // update dataset
   *   index.update_dataset(res, dataset.view());
   *   // The index is now ready for search
   *   cagra::search(res, search_params, index, queries, neighbors, distances);
   * @endcode
   */
  bool attach_dataset_on_build = true;

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
 *
 */
template <typename T, typename IdxT>
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
    auto data_rows = dataset_->n_rows();
    if (dataset_fd_.has_value()) { return n_rows_; }
    return data_rows > 0 ? data_rows : graph_view_.extent(0);
  }

  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t
  {
    return dataset_fd_.has_value() ? dim_ : dataset_->dim();
  }
  /** Graph degree */
  [[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t
  {
    return dataset_fd_.has_value() ? graph_degree_ : graph_view_.extent(1);
  }

  [[nodiscard]] inline auto dataset() const noexcept
    -> raft::device_matrix_view<const T, int64_t, raft::layout_stride>
  {
    auto p = dynamic_cast<strided_dataset<T, int64_t>*>(dataset_.get());
    if (p != nullptr) { return p->view(); }
    auto d = dataset_->dim();
    return raft::make_device_strided_matrix_view<const T, int64_t>(nullptr, 0, d, d);
  }

  /** Dataset [size, dim] */
  [[nodiscard]] inline auto data() const noexcept -> const cuvs::neighbors::dataset<int64_t>&
  {
    return *dataset_;
  }

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

  /** Construct an empty index. */
  index(raft::resources const& res,
        cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
    : cuvs::neighbors::index(),
      metric_(metric),
      graph_(raft::make_device_matrix<graph_index_type, int64_t>(res, 0, 0)),
      dataset_(new cuvs::neighbors::empty_dataset<int64_t>(0)),
      dataset_norms_(std::nullopt)
  {
  }

  /** Construct an index from dataset and knn_graph arrays
   *
   * If the dataset and graph is already in GPU memory, then the index is just a thin wrapper around
   * these that stores a non-owning a reference to the arrays.
   *
   * The constructor also accepts host arrays. In that case they are copied to the device, and the
   * device arrays will be owned by the index.
   *
   * In case the dasates rows are not 16 bytes aligned, then we create a padded copy in device
   * memory to ensure alignment for vectorized load.
   *
   * Usage examples:
   *
   * - Cagra index is normally created by the cagra::build
   * @code{.cpp}
   *   using namespace cuvs::neighbors;
   *   auto dataset = raft::make_host_matrix<float, int64_t>(n_rows, n_cols);
   *   load_dataset(dataset.view());
   *   // use default index parameters
   *   cagra::index_params index_params;
   *   // create and fill the index from a [N, D] dataset
   *   auto index = cagra::build(res, index_params, dataset);
   *   // use default search parameters
   *   cagra::search_params search_params;
   *   // search K nearest neighbours
   *   auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);
   *   auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
   *   cagra::search(res, search_params, index, queries, neighbors.view(), distances.view());
   * @endcode
   *   In the above example, we have passed a host dataset to build. The returned index will own a
   * device copy of the dataset and the knn_graph. In contrast, if we pass the dataset as a
   * device_mdspan to build, then it will only store a reference to it.
   *
   * - Constructing index using existing knn-graph
   * @code{.cpp}
   *   using namespace cuvs::neighbors;
   *
   *   auto dataset = raft::make_device_matrix<float, int64_t>(res, n_rows, n_cols);
   *   auto knn_graph = raft::make_device_matrix<uint32_t, int64_t>(res, n_rows, graph_degree);
   *
   *   // custom loading and graph creation
   *   // load_dataset(dataset.view());
   *   // create_knn_graph(knn_graph.view());
   *
   *   // Wrap the existing device arrays into an index structure
   *   cagra::index<T, IdxT> index(res, metric, raft::make_const_mdspan(dataset.view()),
   *                               raft::make_const_mdspan(knn_graph.view()));
   *
   *   // Both knn_graph and dataset objects have to be in scope while the index is used because
   *   // the index only stores a reference to these.
   *   cagra::search(res, search_params, index, queries, neighbors, distances);
   * @endcode
   */
  template <typename data_accessor, typename graph_accessor>
  index(raft::resources const& res,
        cuvs::distance::DistanceType metric,
        raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, data_accessor> dataset,
        raft::mdspan<const graph_index_type,
                     raft::matrix_extent<int64_t>,
                     raft::row_major,
                     graph_accessor> knn_graph)
    : cuvs::neighbors::index(),
      metric_(metric),
      graph_(raft::make_device_matrix<graph_index_type, int64_t>(res, 0, 0)),
      dataset_(make_aligned_dataset(res, dataset, 16)),
      dataset_norms_(std::nullopt)
  {
    RAFT_EXPECTS(dataset.extent(0) == knn_graph.extent(0),
                 "Dataset and knn_graph must have equal number of rows");
    update_graph(res, knn_graph);

    if (metric_ == cuvs::distance::DistanceType::CosineExpanded) {
      auto p = dynamic_cast<strided_dataset<T, int64_t>*>(dataset_.get());
      if (p) {
        auto dataset_view = p->view();
        if (dataset_view.extent(0) > 0) { compute_dataset_norms_(res); }
      }
    }

    raft::resource::sync_stream(res);
  }

  /**
   * Replace the dataset with a new dataset.
   *
   * If the new dataset rows are aligned on 16 bytes, then only a reference is stored to the
   * dataset. It is the caller's responsibility to ensure that dataset stays alive as long as the
   * index. It is expected that the same set of vectors are used for update_dataset and index build.
   *
   * Note: This will clear any precomputed dataset norms.
   */
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)
  {
    dataset_ = make_aligned_dataset(res, dataset, 16);
    dataset_norms_.reset();

    if (metric() == cuvs::distance::DistanceType::CosineExpanded) {
      if (dataset.extent(0) > 0) { compute_dataset_norms_(res); }
    }
  }

  /** Set the dataset reference explicitly to a device matrix view with padding. */
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t, raft::layout_stride> dataset)
  {
    dataset_ = make_aligned_dataset(res, dataset, 16);
    dataset_norms_.reset();

    if (metric() == cuvs::distance::DistanceType::CosineExpanded) {
      if (dataset.extent(0) > 0) { compute_dataset_norms_(res); }
    }
  }

  /**
   * Replace the dataset with a new dataset.
   *
   * We create a copy of the dataset on the device. The index manages the lifetime of this copy. It
   * is expected that the same set of vectors are used for update_dataset and index build.
   *
   * Note: This will clear any precomputed dataset norms.
   */
  void update_dataset(raft::resources const& res,
                      raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
  {
    dataset_ = make_aligned_dataset(res, dataset, 16);
    dataset_norms_.reset();
    if (metric() == cuvs::distance::DistanceType::CosineExpanded) {
      if (dataset.extent(0) > 0) { compute_dataset_norms_(res); }
    }
  }

  /**
   * Replace the dataset with a new dataset. It is expected that the same set of vectors are used
   * for update_dataset and index build.
   *
   * Note: This will clear any precomputed dataset norms.
   */
  template <typename DatasetT>
  auto update_dataset(raft::resources const& res, DatasetT&& dataset)
    -> std::enable_if_t<std::is_base_of_v<cuvs::neighbors::dataset<dataset_index_type>, DatasetT>>
  {
    dataset_ = std::make_unique<DatasetT>(std::move(dataset));
    dataset_norms_.reset();
    if (metric() == cuvs::distance::DistanceType::CosineExpanded) {
      auto p = dynamic_cast<strided_dataset<T, int64_t>*>(dataset_.get());
      if (p) {
        auto dataset_view = p->view();
        if (dataset_view.extent(0) > 0) { compute_dataset_norms_(res); }
      }
    }
  }

  template <typename DatasetT>
  auto update_dataset(raft::resources const& res, std::unique_ptr<DatasetT>&& dataset)
    -> std::enable_if_t<std::is_base_of_v<neighbors::dataset<dataset_index_type>, DatasetT>>
  {
    dataset_ = std::move(dataset);
    dataset_norms_.reset();
    if (metric() == cuvs::distance::DistanceType::CosineExpanded) {
      auto dataset_view = this->dataset();
      if (dataset_view.extent(0) > 0) { compute_dataset_norms_(res); }
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

    dataset_ = std::make_unique<cuvs::neighbors::empty_dataset<int64_t>>(0);
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
  cuvs::distance::DistanceType metric_;
  raft::device_matrix<graph_index_type, int64_t, raft::row_major> graph_;
  raft::device_matrix_view<const graph_index_type, int64_t, raft::row_major> graph_view_;
  std::unique_ptr<neighbors::dataset<dataset_index_type>> dataset_;
  // Mapping from internal graph node indices to the original user-provided indices.
  std::optional<raft::device_vector<IdxT, int64_t>> source_indices_;
  // only float distances supported at the moment
  std::optional<raft::device_vector<float, int64_t>> dataset_norms_;

  // File descriptors for disk-backed index components (ACE disk mode)
  std::optional<cuvs::util::file_descriptor> dataset_fd_;
  std::optional<cuvs::util::file_descriptor> graph_fd_;
  std::optional<cuvs::util::file_descriptor> mapping_fd_;

  void compute_dataset_norms_(raft::resources const& res);
  size_t n_rows_       = 0;
  size_t dim_          = 0;
  size_t graph_degree_ = 0;
};

/**
 * @}
 */

/**
 * @defgroup cagra_cpp_index_build CAGRA index build functions
 * @{
 */

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * The following distance metrics are supported:
 * - L2
 * - InnerProduct (currently only supported with IVF-PQ as the build algorithm)
 * - CosineExpanded
 * - L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors.view(), distances.view());
 * @endcode
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (device) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<float, uint32_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * The following distance metrics are supported:
 * - L2
 * - InnerProduct (currently only supported with IVF-PQ as the build algorithm)
 * - CosineExpanded
 * - L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors.view(), distances.view());
 * @endcode
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (host) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<float, uint32_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * The following distance metrics are supported:
 * - L2
 * - InnerProduct (currently only supported with IVF-PQ as the build algorithm)
 * - CosineExpanded (dataset norms are computed as float regardless of input data type)
 * - L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors.view(), distances.view());
 * @endcode
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (device) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<half, uint32_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * The following distance metrics are supported:
 * - L2
 * - CosineExpanded (dataset norms are computed as float regardless of input data type)
 * - L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors.view(), distances.view());
 * @endcode
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (host) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<half, uint32_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * The following distance metrics are supported:
 * - L2
 * - CosineExpanded (dataset norms are computed as float regardless of input data type)
 * - L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)
 * - BitwiseHamming (currently only supported with NN-Descent and Iterative Search as the build
 * algorithm, and only for int8_t and uint8_t data types)
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors.view(), distances.view());
 * @endcode
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (device) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<int8_t, uint32_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * The following distance metrics are supported:
 * - L2
 * - InnerProduct (currently only supported with IVF-PQ as the build algorithm)
 * - CosineExpanded (dataset norms are computed as float regardless of input data type)
 * - L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)
 * - BitwiseHamming (currently only supported with NN-Descent and Iterative Search as the build
 * algorithm, and only for int8_t and uint8_t data types)
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors.view(), distances.view());
 * @endcode
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (host) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<int8_t, uint32_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * The following distance metrics are supported:
 * - L2
 * - InnerProduct (currently only supported with IVF-PQ as the build algorithm)
 * - CosineExpanded (dataset norms are computed as float regardless of input data type)
 * - L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)
 * - BitwiseHamming (currently only supported with NN-Descent and Iterative Search as the build
 * algorithm, and only for int8_t and uint8_t data types)
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors.view(), distances.view());
 * @endcode
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (device) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<uint8_t, uint32_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * The following distance metrics are supported:
 * - L2
 * - InnerProduct (currently only supported with IVF-PQ as the build algorithm)
 * - CosineExpanded (dataset norms are computed as float regardless of input data type)
 * - L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)
 * - BitwiseHamming (currently only supported with NN-Descent and Iterative Search as the build
 * algorithm, and only for int8_t and uint8_t data types)
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors.view(), distances.view());
 * @endcode
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (host) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<uint8_t, uint32_t>;
/**
 * @}
 */

/**
 * @defgroup cagra_cpp_index_extend CAGRA extend functions
 * @{
 */

/** @brief Add new vectors to a CAGRA index
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   auto additional_dataset = raft::make_device_matrix<float, int64_t>(handle,add_size,dim);
 *   // set_additional_dataset(additional_dataset.view());
 *
 *   cagra::extend_params params;
 *   cagra::extend(res, params, raft::make_const_mdspan(additional_dataset.view()), index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] additional_dataset additional dataset on device memory
 * @param[in,out] idx CAGRA index
 * @param[out] new_dataset_buffer_view memory buffer view for the dataset including the additional
 * part. The data will be copied from the current index in this function. The num rows must be the
 * sum of the original and additional datasets, cols must be the dimension of the dataset, and the
 * stride must be the same as the original index dataset. This view will be stored in the output
 * index. It is the caller's responsibility to ensure that dataset stays alive as long as the index.
 * This option is useful when users want to manage the memory space for the dataset themselves.
 * @param[out] new_graph_buffer_view memory buffer view for the graph including the additional part.
 * The data will be copied from the current index in this function. The num rows must be the sum of
 * the original and additional datasets and cols must be the graph degree. This view will be stored
 * in the output index. It is the caller's responsibility to ensure that dataset stays alive as long
 * as the index. This option is useful when users want to manage the memory space for the graph
 * themselves.
 */
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::device_matrix_view<const float, int64_t, raft::row_major> additional_dataset,
  cuvs::neighbors::cagra::index<float, uint32_t>& idx,
  std::optional<raft::device_matrix_view<float, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                        = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);

/** @brief Add new vectors to a CAGRA index
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   auto additional_dataset = raft::make_host_matrix<float, int64_t>(handle,add_size,dim);
 *   // set_additional_dataset(additional_dataset.view());
 *
 *   cagra::extend_params params;
 *   cagra::extend(res, params, raft::make_const_mdspan(additional_dataset.view()), index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] additional_dataset additional dataset on host memory
 * @param[in,out] idx CAGRA index
 * @param[out] new_dataset_buffer_view memory buffer view for the dataset including the additional
 * part. The data will be copied from the current index in this function. The num rows must be the
 * sum of the original and additional datasets, cols must be the dimension of the dataset, and the
 * stride must be the same as the original index dataset. This view will be stored in the output
 * index. It is the caller's responsibility to ensure that dataset stays alive as long as the index.
 * This option is useful when users want to manage the memory space for the dataset themselves.
 * @param[out] new_graph_buffer_view memory buffer view for the graph including the additional part.
 * The data will be copied from the current index in this function. The num rows must be the sum of
 * the original and additional datasets and cols must be the graph degree. This view will be stored
 * in the output index. It is the caller's responsibility to ensure that dataset stays alive as long
 * as the index. This option is useful when users want to manage the memory space for the graph
 * themselves.
 */
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::host_matrix_view<const float, int64_t, raft::row_major> additional_dataset,
  cuvs::neighbors::cagra::index<float, uint32_t>& idx,
  std::optional<raft::device_matrix_view<float, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                        = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);

/** @brief Add new vectors to a CAGRA index
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   auto additional_dataset = raft::make_device_matrix<half, int64_t>(handle,add_size,dim);
 *   // set_additional_dataset(additional_dataset.view());
 *
 *   cagra::extend_params params;
 *   cagra::extend(res, params, raft::make_const_mdspan(additional_dataset.view()), index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] additional_dataset additional dataset on device memory
 * @param[in,out] idx CAGRA index
 * @param[out] new_dataset_buffer_view memory buffer view for the dataset including the additional
 * part. The data will be copied from the current index in this function. The num rows must be the
 * sum of the original and additional datasets, cols must be the dimension of the dataset, and the
 * stride must be the same as the original index dataset. This view will be stored in the output
 * index. It is the caller's responsibility to ensure that dataset stays alive as long as the index.
 * This option is useful when users want to manage the memory space for the dataset themselves.
 * @param[out] new_graph_buffer_view memory buffer view for the graph including the additional part.
 * The data will be copied from the current index in this function. The num rows must be the sum of
 * the original and additional datasets and cols must be the graph degree. This view will be stored
 * in the output index. It is the caller's responsibility to ensure that dataset stays alive as long
 * as the index. This option is useful when users want to manage the memory space for the graph
 * themselves.
 */
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::device_matrix_view<const half, int64_t, raft::row_major> additional_dataset,
  cuvs::neighbors::cagra::index<half, uint32_t>& idx,
  std::optional<raft::device_matrix_view<half, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                        = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);

/** @brief Add new vectors to a CAGRA index
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   auto additional_dataset = raft::make_host_matrix<half, int64_t>(handle,add_size,dim);
 *   // set_additional_dataset(additional_dataset.view());
 *
 *   cagra::extend_params params;
 *   cagra::extend(res, params, raft::make_const_mdspan(additional_dataset.view()), index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] additional_dataset additional dataset on host memory
 * @param[in,out] idx CAGRA index
 * @param[out] new_dataset_buffer_view memory buffer view for the dataset including the additional
 * part. The data will be copied from the current index in this function. The num rows must be the
 * sum of the original and additional datasets, cols must be the dimension of the dataset, and the
 * stride must be the same as the original index dataset. This view will be stored in the output
 * index. It is the caller's responsibility to ensure that dataset stays alive as long as the index.
 * This option is useful when users want to manage the memory space for the dataset themselves.
 * @param[out] new_graph_buffer_view memory buffer view for the graph including the additional part.
 * The data will be copied from the current index in this function. The num rows must be the sum of
 * the original and additional datasets and cols must be the graph degree. This view will be stored
 * in the output index. It is the caller's responsibility to ensure that dataset stays alive as long
 * as the index. This option is useful when users want to manage the memory space for the graph
 * themselves.
 */
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::host_matrix_view<const half, int64_t, raft::row_major> additional_dataset,
  cuvs::neighbors::cagra::index<half, uint32_t>& idx,
  std::optional<raft::device_matrix_view<half, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                        = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);

/** @brief Add new vectors to a CAGRA index
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   auto additional_dataset = raft::make_device_matrix<int8_t, int64_t>(handle,add_size,dim);
 *   // set_additional_dataset(additional_dataset.view());
 *
 *   cagra::extend_params params;
 *   cagra::extend(res, params, raft::make_const_mdspan(additional_dataset.view()), index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] additional_dataset additional dataset on device memory
 * @param[in,out] idx CAGRA index
 * @param[out] new_dataset_buffer_view memory buffer view for the dataset including the additional
 * part. The data will be copied from the current index in this function. The num rows must be the
 * sum of the original and additional datasets, cols must be the dimension of the dataset, and the
 * stride must be the same as the original index dataset. This view will be stored in the output
 * index. It is the caller's responsibility to ensure that dataset stays alive as long as the index.
 * This option is useful when users want to manage the memory space for the dataset themselves.
 * @param[out] new_graph_buffer_view memory buffer view for the graph including the additional part.
 * The data will be copied from the current index in this function. The num rows must be the sum of
 * the original and additional datasets and cols must be the graph degree. This view will be stored
 * in the output index. It is the caller's responsibility to ensure that dataset stays alive as long
 * as the index. This option is useful when users want to manage the memory space for the graph
 * themselves.
 */
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::device_matrix_view<const int8_t, int64_t, raft::row_major> additional_dataset,
  cuvs::neighbors::cagra::index<int8_t, uint32_t>& idx,
  std::optional<raft::device_matrix_view<int8_t, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                        = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);

/** @brief Add new vectors to a CAGRA index
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   auto additional_dataset = raft::make_host_matrix<int8_t, int64_t>(handle,add_size,dim);
 *   // set_additional_dataset(additional_dataset.view());
 *
 *   cagra::extend_params params;
 *   cagra::extend(res, params, raft::make_const_mdspan(additional_dataset.view()), index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] additional_dataset additional dataset on host memory
 * @param[in,out] idx CAGRA index
 * @param[out] new_dataset_buffer_view memory buffer view for the dataset including the additional
 * part. The data will be copied from the current index in this function. The num rows must be the
 * sum of the original and additional datasets, cols must be the dimension of the dataset, and the
 * stride must be the same as the original index dataset. This view will be stored in the output
 * index. It is the caller's responsibility to ensure that dataset stays alive as long as the index.
 * This option is useful when users want to manage the memory space for the dataset themselves.
 * @param[out] new_graph_buffer_view memory buffer view for the graph including the additional part.
 * The data will be copied from the current index in this function. The num rows must be the sum of
 * the original and additional datasets and cols must be the graph degree. This view will be stored
 * in the output index. It is the caller's responsibility to ensure that dataset stays alive as long
 * as the index. This option is useful when users want to manage the memory space for the graph
 * themselves.
 */
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::host_matrix_view<const int8_t, int64_t, raft::row_major> additional_dataset,
  cuvs::neighbors::cagra::index<int8_t, uint32_t>& idx,
  std::optional<raft::device_matrix_view<int8_t, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                        = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);

/** @brief Add new vectors to a CAGRA index
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   auto additional_dataset = raft::make_host_matrix<uint8_t, int64_t>(handle,add_size,dim);
 *   // set_additional_dataset(additional_dataset.view());
 *
 *   cagra::extend_params params;
 *   cagra::extend(res, params, raft::make_const_mdspan(additional_dataset.view()), index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] additional_dataset additional dataset on host memory
 * @param[in,out] idx CAGRA index
 * @param[out] new_dataset_buffer_view memory buffer view for the dataset including the additional
 * part. The data will be copied from the current index in this function. The num rows must be the
 * sum of the original and additional datasets, cols must be the dimension of the dataset, and the
 * stride must be the same as the original index dataset. This view will be stored in the output
 * index. It is the caller's responsibility to ensure that dataset stays alive as long as the index.
 * This option is useful when users want to manage the memory space for the dataset themselves.
 * @param[out] new_graph_buffer_view memory buffer view for the graph including the additional part.
 * The data will be copied from the current index in this function. The num rows must be the sum of
 * the original and additional datasets and cols must be the graph degree. This view will be stored
 * in the output index. It is the caller's responsibility to ensure that dataset stays alive as long
 * as the index. This option is useful when users want to manage the memory space for the graph
 * themselves.
 */
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> additional_dataset,
  cuvs::neighbors::cagra::index<uint8_t, uint32_t>& idx,
  std::optional<raft::device_matrix_view<uint8_t, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                        = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);

/** @brief Add new vectors to a CAGRA index
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   auto additional_dataset = raft::make_host_matrix<uint8_t, int64_t>(handle,add_size,dim);
 *   // set_additional_dataset(additional_dataset.view());
 *
 *   cagra::extend_params params;
 *   cagra::extend(res, params, raft::make_const_mdspan(additional_dataset.view()), index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] additional_dataset additional dataset on host memory
 * @param[in,out] idx CAGRA index
 * @param[out] new_dataset_buffer_view memory buffer view for the dataset including the additional
 * part. The data will be copied from the current index in this function. The num rows must be the
 * sum of the original and additional datasets, cols must be the dimension of the dataset, and the
 * stride must be the same as the original index dataset. This view will be stored in the output
 * index. It is the caller's responsibility to ensure that dataset stays alive as long as the index.
 * This option is useful when users want to manage the memory space for the dataset themselves.
 * @param[out] new_graph_buffer_view memory buffer view for the graph including the additional part.
 * The data will be copied from the current index in this function. The num rows must be the sum of
 * the original and additional datasets and cols must be the graph degree. This view will be stored
 * in the output index. It is the caller's responsibility to ensure that dataset stays alive as long
 * as the index. This option is useful when users want to manage the memory space for the graph
 * themselves.
 */
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> additional_dataset,
  cuvs::neighbors::cagra::index<uint8_t, uint32_t>& idx,
  std::optional<raft::device_matrix_view<uint8_t, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                        = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);
/**
 * @}
 */

/**
 * @defgroup cagra_cpp_index_search CAGRA search functions
 * @{
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index cagra index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */

void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::index<float, uint32_t>& index,
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
 * @param[in] index cagra index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::index<half, uint32_t>& index,
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
 * @param[in] index cagra index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index,
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
 * @param[in] index cagra index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
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
 * @param[in] index cagra index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */

void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::index<float, uint32_t>& index,
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
 * @param[in] index cagra index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::index<half, uint32_t>& index,
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
 * @param[in] index cagra index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index,
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
 * @param[in] index cagra index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @}
 */

/**
 * @defgroup cagra_cpp_serialize CAGRA serialize functions
 * @{
 */

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
               const cuvs::neighbors::cagra::index<float, uint32_t>& index,
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

 * cuvs::neighbors::cagra::index<float, uint32_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index the cagra index
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::cagra::index<float, uint32_t>* index);

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
               const cuvs::neighbors::cagra::index<float, uint32_t>& index,
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
 * cuvs::neighbors::cagra::index<float, uint32_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[out] index the cagra index
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::cagra::index<float, uint32_t>* index);
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
               const cuvs::neighbors::cagra::index<half, uint32_t>& index,
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

 * cuvs::neighbors::cagra::index<half, uint32_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index the cagra index
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::cagra::index<half, uint32_t>* index);

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
               const cuvs::neighbors::cagra::index<half, uint32_t>& index,
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
 * cuvs::neighbors::cagra::index<half, uint32_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[out] index the cagra index
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::cagra::index<half, uint32_t>* index);

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
               const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index,
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

 * cuvs::neighbors::cagra::index<int8_t, uint32_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index the cagra index
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::cagra::index<int8_t, uint32_t>* index);

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
               const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index,
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
 * cuvs::neighbors::cagra::index<int8_t, uint32_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[out] index the cagra index
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::cagra::index<int8_t, uint32_t>* index);

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
               const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
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

 * cuvs::neighbors::cagra::index<uint8_t, uint32_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index the cagra index
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::cagra::index<uint8_t, uint32_t>* index);

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
               const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
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
 * cuvs::neighbors::cagra::index<uint8_t, uint32_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[out] index the cagra index
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::cagra::index<uint8_t, uint32_t>* index);

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
  const cuvs::neighbors::cagra::index<float, uint32_t>& index,
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
  const cuvs::neighbors::cagra::index<float, uint32_t>& index,
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
  const cuvs::neighbors::cagra::index<half, uint32_t>& index,
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
  const cuvs::neighbors::cagra::index<half, uint32_t>& index,
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
  const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index,
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
  const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index,
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
  const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
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
  const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
  std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * @}
 */

/**
 * @defgroup cagra_cpp_index_merge CAGRA index build functions
 * @{
 */

/** @brief Merge multiple CAGRA indices into a single index.
 *
 * This function merges multiple CAGRA indices into one, combining both the datasets and graph
 * structures.
 *
 * @note: When device memory is sufficient, the dataset attached to the returned index is allocated
 * in device memory by default; otherwise, host memory is used automatically.
 *
 * @note: This API only supports physical merge (`merge_strategy = MERGE_STRATEGY_PHYSICAL`), and
 * attempting a logical merge here will throw an error.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   auto dataset0 = raft::make_host_matrix<float, int64_t>(handle, size0, dim);
 *   auto dataset1 = raft::make_host_matrix<float, int64_t>(handle, size1, dim);
 *
 *   auto index0 = cagra::build(res, index_params, dataset0);
 *   auto index1 = cagra::build(res, index_params, dataset1);
 *
 *   std::vector<cagra::index<float, uint32_t>*> indices{&index0, &index1};
 *
 *   auto merged_index = cagra::merge(res, index_params, indices);
 * @endcode
 *
 * @param[in] res RAFT resources used for the merge operation.
 * @param[in] params Parameters that control the merging process.
 * @param[in] indices A vector of pointers to the CAGRA indices to merge. All indices must:
 *                    - Have attached datasets with the same dimension.
 * @param[in] row_filter an optional device filter function object that greenlights rows
 *    to include in the merged index  (none_sample_filter for no filtering)
 * @return A new CAGRA index containing the merged indices, graph, and dataset.
 */
auto merge(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           std::vector<cuvs::neighbors::cagra::index<float, uint32_t>*>& indices,
           const cuvs::neighbors::filtering::base_filter& row_filter =
             cuvs::neighbors::filtering::none_sample_filter{})
  -> cuvs::neighbors::cagra::index<float, uint32_t>;

/** @copydoc merge */
auto merge(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           std::vector<cuvs::neighbors::cagra::index<half, uint32_t>*>& indices,
           const cuvs::neighbors::filtering::base_filter& row_filter =
             cuvs::neighbors::filtering::none_sample_filter{})
  -> cuvs::neighbors::cagra::index<half, uint32_t>;

/** @copydoc merge */
auto merge(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           std::vector<cuvs::neighbors::cagra::index<int8_t, uint32_t>*>& indices,
           const cuvs::neighbors::filtering::base_filter& row_filter =
             cuvs::neighbors::filtering::none_sample_filter{})
  -> cuvs::neighbors::cagra::index<int8_t, uint32_t>;

/** @copydoc merge */
auto merge(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           std::vector<cuvs::neighbors::cagra::index<uint8_t, uint32_t>*>& indices,
           const cuvs::neighbors::filtering::base_filter& row_filter =
             cuvs::neighbors::filtering::none_sample_filter{})
  -> cuvs::neighbors::cagra::index<uint8_t, uint32_t>;
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
           raft::host_matrix_view<const float, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<cagra::index<float, uint32_t>, float, uint32_t>;

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
           raft::host_matrix_view<const half, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<cagra::index<half, uint32_t>, half, uint32_t>;

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
           raft::host_matrix_view<const int8_t, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<cagra::index<int8_t, uint32_t>, int8_t, uint32_t>;

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
           raft::host_matrix_view<const uint8_t, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<cagra::index<uint8_t, uint32_t>, uint8_t, uint32_t>;

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
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<cagra::index<float, uint32_t>, float, uint32_t>& index,
            raft::host_matrix_view<const float, int64_t, row_major> new_vectors,
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
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<cagra::index<half, uint32_t>, half, uint32_t>& index,
            raft::host_matrix_view<const half, int64_t, row_major> new_vectors,
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
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<cagra::index<int8_t, uint32_t>, int8_t, uint32_t>& index,
            raft::host_matrix_view<const int8_t, int64_t, row_major> new_vectors,
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
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<cagra::index<uint8_t, uint32_t>, uint8_t, uint32_t>& index,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> new_vectors,
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
void search(const raft::resources& clique,
            const cuvs::neighbors::mg_index<cagra::index<float, uint32_t>, float, uint32_t>& index,
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
void search(const raft::resources& clique,
            const cuvs::neighbors::mg_index<cagra::index<half, uint32_t>, half, uint32_t>& index,
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
  const cuvs::neighbors::mg_index<cagra::index<int8_t, uint32_t>, int8_t, uint32_t>& index,
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
  const cuvs::neighbors::mg_index<cagra::index<uint8_t, uint32_t>, uint8_t, uint32_t>& index,
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
void search(const raft::resources& clique,
            const cuvs::neighbors::mg_index<cagra::index<float, uint32_t>, float, uint32_t>& index,
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
void search(const raft::resources& clique,
            const cuvs::neighbors::mg_index<cagra::index<half, uint32_t>, half, uint32_t>& index,
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
  const cuvs::neighbors::mg_index<cagra::index<int8_t, uint32_t>, int8_t, uint32_t>& index,
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
  const cuvs::neighbors::mg_index<cagra::index<uint8_t, uint32_t>, uint8_t, uint32_t>& index,
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
  const cuvs::neighbors::mg_index<cagra::index<float, uint32_t>, float, uint32_t>& index,
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
void serialize(const raft::resources& clique,
               const cuvs::neighbors::mg_index<cagra::index<half, uint32_t>, half, uint32_t>& index,
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
  const cuvs::neighbors::mg_index<cagra::index<int8_t, uint32_t>, int8_t, uint32_t>& index,
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
  const cuvs::neighbors::mg_index<cagra::index<uint8_t, uint32_t>, uint8_t, uint32_t>& index,
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
 * auto new_index = cuvs::neighbors::cagra::deserialize<float, uint32_t>(clique, filename);
 *
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] filename path to the file to be deserialized
 *
 */
template <typename T, typename IdxT>
auto deserialize(const raft::resources& clique, const std::string& filename)
  -> cuvs::neighbors::mg_index<cagra::index<T, IdxT>, T, IdxT>;

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
 * auto new_index = cuvs::neighbors::cagra::distribute<float, uint32_t>(clique, filename);
 *
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] filename path to the file to be deserialized : a local index
 *
 */
template <typename T, typename IdxT>
auto distribute(const raft::resources& clique, const std::string& filename)
  -> cuvs::neighbors::mg_index<cagra::index<T, IdxT>, T, IdxT>;

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
 *   // use default index parameters based on shape of the dataset
 *   ivf_pq::index_params build_params = ivf_pq::index_params::from_dataset(dataset);
 *   ivf_pq::search_params search_params;
 *   auto knn_graph      = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 128);
 *   // create knn graph
 *   cagra::build_knn_graph(res, dataset, knn_graph.view(), 2, build_params, search_params);
 *   auto optimized_gaph = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 64);
 *   cagra::optimize(res, dataset, knn_graph.view(), optimized_graph.view());
 *   // Construct an index from dataset and optimized knn_graph
 *   auto index = cagra::index<T, IdxT>(res, build_params.metric(), dataset,
 *                                      optimized_graph.view());
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
 *   // use default index parameters based on shape of the dataset
 *   ivf_pq::index_params build_params = ivf_pq::index_params::from_dataset(dataset);
 *   ivf_pq::search_params search_params;
 *   auto knn_graph      = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 128);
 *   // create knn graph
 *   cagra::build_knn_graph(res, dataset, knn_graph.view(), 2, build_params, search_params);
 *   auto optimized_gaph = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 64);
 *   cagra::optimize(res, dataset, knn_graph.view(), optimized_graph.view());
 *   // Construct an index from dataset and optimized knn_graph
 *   auto index = cagra::index<T, IdxT>(res, build_params.metric(), dataset,
 *                                      optimized_graph.view());
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
 *   // use default index parameters based on shape of the dataset
 *   ivf_pq::index_params build_params = ivf_pq::index_params::from_dataset(dataset);
 *   ivf_pq::search_params search_params;
 *   auto knn_graph      = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 128);
 *   // create knn graph
 *   cagra::build_knn_graph(res, dataset, knn_graph.view(), 2, build_params, search_params);
 *   auto optimized_gaph = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 64);
 *   cagra::optimize(res, dataset, knn_graph.view(), optimized_graph.view());
 *   // Construct an index from dataset and optimized knn_graph
 *   auto index = cagra::index<T, IdxT>(res, build_params.metric(), dataset,
 *                                      optimized_graph.view());
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
 *   // use default index parameters based on shape of the dataset
 *   ivf_pq::index_params build_params = ivf_pq::index_params::from_dataset(dataset);
 *   ivf_pq::search_params search_params;
 *   auto knn_graph      = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 128);
 *   // create knn graph
 *   cagra::build_knn_graph(res, dataset, knn_graph.view(), 2, build_params, search_params);
 *   auto optimized_gaph = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 64);
 *   cagra::optimize(res, dataset, knn_graph.view(), optimized_graph.view());
 *   // Construct an index from dataset and optimized knn_graph
 *   auto index = cagra::index<T, IdxT>(res, build_params.metric(), dataset,
 *                                      optimized_graph.view());
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
