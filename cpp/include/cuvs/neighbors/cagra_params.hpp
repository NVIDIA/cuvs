/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * CAGRA index and search parameter types. Split out of cagra.hpp so that other headers (e.g.
 * cluster/kmeans.hpp) can hold CAGRA parameters without including the full CAGRA index API.
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <raft/core/mdspan_types.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace cagra {

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

}  // namespace cagra
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace graph_build_params {
/**
 * Parameters for the iterative CAGRA graph build algorithm.
 *
 * Inherits from cagra::search_params so that all search tuning knobs
 * (search_width, max_iterations, itopk_size, etc.) are available for
 * controlling the search-and-optimize loop during graph construction.
 * The defaults are tuned for the build loop (e.g. search_width=1,
 * max_iterations=8) and may differ from the regular search defaults.
 *
 */
struct iterative_search_params : cuvs::neighbors::cagra::search_params {
  iterative_search_params()
  {
    this->search_width   = 1;
    this->max_iterations = 8;
    // itopk_size controls the search during the *growing* iterations of the build loop.
    // 0 (default) means auto-select per iteration (max(graph_degree + 32, 128)); a nonzero
    // value overrides it for the growing iterations. The final iteration always uses a fixed
    // itopk tied to the output topk, regardless of this value.
    this->itopk_size = 0;
  }
};

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
   * for twice the dataset, final graph, and label mapping. The directory may
   * already exist, but ACE's named artifacts must not already exist. Simultaneous
   * builds must use different directories. On failure, ACE removes only artifacts
   * it created and never deletes unrelated directory contents.
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
   *   auto index = cagra::build(res, index_params, dataset->view());
   *   // ASSERT(index.size() == 0);  // no dataset yet
   *   // Attach with a view (storage owned by `dataset`).
   *   index = cagra::update_dataset(res, std::move(index), dataset->view());
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

}  // namespace cagra
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
