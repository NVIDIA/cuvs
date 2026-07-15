/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/ivf_pq.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup cagra_c_index_params C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Enum to denote which ANN algorithm is used to build CAGRA graph
 *
 */
enum cuvsCagraGraphBuildAlgo {
  /* Select build algorithm automatically */
  AUTO_SELECT = 0,
  /* Use IVF-PQ to build all-neighbors knn graph */
  IVF_PQ = 1,
  /* Experimental, use NN-Descent to build all-neighbors knn graph */
  NN_DESCENT = 2,
  /* Experimental, use iterative cagra search and optimize to build the knn graph */
  ITERATIVE_CAGRA_SEARCH = 3,
  /**
   * Experimental, use ACE (Augmented Core Extraction) to build the graph. ACE partitions the
   * dataset into core and augmented partitions and builds a sub-index for each partition. This
   * enables building indices for datasets too large to fit in GPU or host memory.
   * See cuvsAceParams for more details about the ACE algorithm and its parameters.
   */
  ACE = 4
};

/**
 * @brief A strategy for selecting the graph build parameters based on similar HNSW index
 * parameters.
 *
 * Define how cuvsCagraIndexParamsFromHnswParams should construct a graph to construct a graph
 * that is to be converted to (used by) a CPU HNSW index.
 */
enum cuvsCagraHnswHeuristicType {
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
   CUVS_CAGRA_HEURISTIC_SIMILAR_SEARCH_PERFORMANCE = 0,
  /**
   * Create a graph that has the same binary size as an HNSW graph with the given parameters
   * (graph_degree = 2 * M) while trying to match the search performance as closely as possible.
   *
   * The reference HNSW index and the corresponding from-CAGRA generated HNSW index will NOT produce
   * the same recalls and QPS for the same parameter ef. The graphs are different internally. For
   * the same ef, the from-CAGRA index likely has a slightly higher recall and slightly lower QPS.
   * However, the Recall-QPS curves should be similar (i.e. the points are just shifted along the
   * curve).
   */
   CUVS_CAGRA_HEURISTIC_SAME_GRAPH_FOOTPRINT = 1
};

/** Parameters for VPQ compression. */
struct cuvsCagraCompressionParams {
  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible values: [4, 5, 6, 7, 8].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the better the search
   * performance, but the lower the recall.
   */
  uint32_t pq_bits;
  /**
   * The dimensionality of the vector after compression by PQ.
   * When zero, an optimal value is selected using a heuristic.
   *
   * TODO: at the moment `dim` must be a multiple `pq_dim`.
   */
  uint32_t pq_dim;
  /**
   * Vector Quantization (VQ) codebook size - number of "coarse cluster centers".
   * When zero, an optimal value is selected using a heuristic.
   */
  uint32_t vq_n_centers;
  /** The number of iterations searching for kmeans centers (both VQ & PQ phases). */
  uint32_t kmeans_n_iters;
  /**
   * The fraction of data to use during iterative kmeans building (VQ phase).
   * When zero, an optimal value is selected using a heuristic.
   */
  double vq_kmeans_trainset_fraction;
  /**
   * The fraction of data to use during iterative kmeans building (PQ phase).
   * When zero, an optimal value is selected using a heuristic.
   */
  double pq_kmeans_trainset_fraction;
};

typedef struct cuvsCagraCompressionParams* cuvsCagraCompressionParams_t;

struct cuvsIvfPqParams {
  cuvsIvfPqIndexParams_t ivf_pq_build_params;
  cuvsIvfPqSearchParams_t ivf_pq_search_params;
  float refinement_rate;
};

typedef struct cuvsIvfPqParams* cuvsIvfPqParams_t;

/**
 * Parameters for ACE (Augmented Core Extraction) graph build.
 * ACE enables building indexes for datasets too large to fit in GPU memory by:
 * 1. Partitioning the dataset in core (closest) and augmented (second-closest)
 * partitions using balanced k-means.
 * 2. Building sub-indexes for each partition independently
 * 3. Concatenating sub-graphs into a final unified index
 */
struct cuvsAceParams {
  /**
   * Number of partitions for ACE (Augmented Core Extraction) partitioned build.
   *
   * When set to 0 (default), the number of partitions is automatically derived
   * based on available host and GPU memory to maximize partition size while
   * ensuring the build fits in memory.
   *
   * Small values might improve recall but potentially degrade performance and
   * increase memory usage. Partitions should not be too small to prevent issues
   * in KNN graph construction. The partition size is on average 2 * (n_rows /
   * npartitions) * dim * sizeof(T). 2 is because of the core and augmented
   * vectors. Please account for imbalance in the partition sizes (up to 3x in
   * our tests).
   *
   * If the specified number of partitions results in partitions that exceed
   * available memory, the value will be automatically increased to fit memory
   * constraints and a warning will be issued.
   */
  size_t npartitions;
  /**
   * The index quality for the ACE build.
   *
   * Bigger values increase the index quality. At some point, increasing this will no longer
   * improve the quality.
   */
  size_t ef_construction;
  /**
   * Directory to store ACE build artifacts (e.g., KNN graph, optimized graph).
   *
   * Used when `use_disk` is true or when the graph does not fit in host and GPU
   * memory. This should be the fastest disk in the system and hold enough space
   * for twice the dataset, final graph, and label mapping.
   */
  const char* build_dir;
  /**
   * Whether to use disk-based storage for ACE build.
   *
   * When true, enables disk-based operations for memory-efficient graph construction.
   */
  bool use_disk;
  /**
   * Maximum host memory to use for ACE build in GiB.
   *
   * When set to 0 (default), uses available host memory.
   * When set to a positive value, limits host memory usage to the specified amount.
   * Useful for testing or when running alongside other memory-intensive processes.
   */
  double max_host_memory_gb;
  /**
   * Maximum GPU memory to use for ACE build in GiB.
   *
   * When set to 0 (default), uses available GPU memory.
   * When set to a positive value, limits GPU memory usage to the specified amount.
   * Useful for testing or when running alongside other memory-intensive processes.
   */
  double max_gpu_memory_gb;
};

typedef struct cuvsAceParams* cuvsAceParams_t;

/**
 * @brief Supplemental parameters to build CAGRA Index
 *
 */
struct cuvsCagraIndexParams {
  /** Distance type. */
  cuvsDistanceType metric;
  /** Degree of input graph for pruning. */
  size_t intermediate_graph_degree;
  /** Degree of output graph. */
  size_t graph_degree;
  /** ANN algorithm to build knn graph. */
  enum cuvsCagraGraphBuildAlgo build_algo;
  /** Number of Iterations to run if building with NN_DESCENT */
  size_t nn_descent_niter;
  /**
   * Optional: specify compression parameters if compression is desired.
   *
   * NOTE: this is experimental new API, consider it unsafe.
   */
  cuvsCagraCompressionParams_t compression;
  /**
   * Optional: specify graph build params based on build_algo
   * - IVF_PQ: cuvsIvfPqParams_t
   * - ACE: cuvsAceParams_t
   * - Others: nullptr
   */
  void* graph_build_params;
};

typedef struct cuvsCagraIndexParams* cuvsCagraIndexParams_t;

/**
 * @brief Allocate CAGRA Index params, and populate with default values
 *
 * @param[in] params cuvsCagraIndexParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* params);

/**
 * @brief De-allocate CAGRA Index params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params);

/**
 * @brief Allocate CAGRA Compression params, and populate with default values
 *
 * @param[in] params cuvsCagraCompressionParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraCompressionParamsCreate(cuvsCagraCompressionParams_t* params);

/**
 * @brief De-allocate CAGRA Compression params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraCompressionParamsDestroy(cuvsCagraCompressionParams_t params);

/**
 * @brief Allocate ACE params, and populate with default values
 *
 * @param[in] params cuvsAceParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsAceParamsCreate(cuvsAceParams_t* params);

/**
 * @brief De-allocate ACE params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsAceParamsDestroy(cuvsAceParams_t params);

/**
 * @brief Create CAGRA index parameters similar to an HNSW index
 *
 * This factory function creates CAGRA parameters that yield a graph compatible with
 * an HNSW graph with the given parameters.
 *
 * @param[out] params The CAGRA index params to populate
 * @param[in] n_rows Number of rows in the dataset
 * @param[in] dim Number of dimensions in the dataset
 * @param[in] M HNSW index parameter M
 * @param[in] ef_construction HNSW index parameter ef_construction
 * @param[in] heuristic Strategy for parameter selection
 * @param[in] metric Distance metric to use
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraIndexParamsFromHnswParams(cuvsCagraIndexParams_t params,
                                                int64_t n_rows,
                                                int64_t dim,
                                                int M,
                                                int ef_construction,
                                                enum cuvsCagraHnswHeuristicType heuristic,
                                                cuvsDistanceType metric);

/**
 * @}
 */

/**
 * @defgroup cagra_c_extend_params C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */
/**
 * @brief Supplemental parameters to extend CAGRA Index
 *
 */
struct cuvsCagraExtendParams {
  /** The additional dataset is divided into chunks and added to the graph. This is the knob to
   * adjust the tradeoff between the recall and operation throughput. Large chunk sizes can result
   * in high throughput, but use more working memory (O(max_chunk_size*degree^2)). This can also
   * degrade recall because no edges are added between the nodes in the same chunk. Auto select when
   * 0. */
  uint32_t max_chunk_size;
};

typedef struct cuvsCagraExtendParams* cuvsCagraExtendParams_t;

/**
 * @brief Allocate CAGRA Extend params, and populate with default values
 *
 * @param[in] params cuvsCagraExtendParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraExtendParamsCreate(cuvsCagraExtendParams_t* params);

/**
 * @brief De-allocate CAGRA Extend params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraExtendParamsDestroy(cuvsCagraExtendParams_t params);

/**
 * @}
 */

/**
 * @defgroup cagra_c_search_params C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Enum to denote algorithm used to search CAGRA Index
 *
 */
enum cuvsCagraSearchAlgo {
  /** For large batch sizes. */
  SINGLE_CTA = 0,
  /** For small batch sizes. */
  MULTI_CTA    = 1,
  MULTI_KERNEL = 2,
  AUTO         = 100
};

/**
 * @brief Enum to denote Hash Mode used while searching CAGRA index
 *
 */
enum cuvsCagraHashMode { HASH = 0, SMALL = 1, AUTO_HASH = 100 };

/**
 * @brief Supplemental parameters to search CAGRA index
 *
 */
struct cuvsCagraSearchParams {
  /** Maximum number of queries to search at the same time (batch size). Auto select when 0.*/
  size_t max_queries;

  /** Number of intermediate search results retained during the search.
   *
   *  This is the main knob to adjust trade off between accuracy and search speed.
   *  Higher values improve the search accuracy.
   */
  size_t itopk_size;

  /** Upper limit of search iterations. Auto select when 0.*/
  size_t max_iterations;

  // In the following we list additional search parameters for fine tuning.
  // Reasonable default values are automatically chosen.

  /** Which search implementation to use. */
  enum cuvsCagraSearchAlgo algo;

  /** Number of threads used to calculate a single distance. 4, 8, 16, or 32. */
  size_t team_size;

  /** Number of graph nodes to select as the starting point for the search in each iteration. aka
   * search width?*/
  size_t search_width;
  /** Lower limit of search iterations. */
  size_t min_iterations;

  /** Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0. */
  size_t thread_block_size;
  /** Hashmap type. Auto selection when AUTO. */
  enum cuvsCagraHashMode hashmap_mode;
  /** Lower limit of hashmap bit length. More than 8. */
  size_t hashmap_min_bitlen;
  /** Upper limit of hashmap fill rate. More than 0.1, less than 0.9.*/
  float hashmap_max_fill_rate;

  /** Number of iterations of initial random seed node selection. 1 or more. */
  uint32_t num_random_samplings;
  /** Bit mask used for initial random seed node selection. */
  uint64_t rand_xor_mask;

  /** Whether to use the persistent version of the kernel (only SINGLE_CTA is supported a.t.m.) */
  bool persistent;
  /** Persistent kernel: time in seconds before the kernel stops if no requests received. */
  float persistent_lifetime;
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
  float persistent_device_usage;
};

typedef struct cuvsCagraSearchParams* cuvsCagraSearchParams_t;

/**
 * @brief Allocate CAGRA search params, and populate with default values
 *
 * @param[in] params cuvsCagraSearchParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* params);

/**
 * @brief De-allocate CAGRA search params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t params);

/**
 * @}
 */

/**
 * @defgroup cagra_c_index C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Struct to hold address of cuvs::neighbors::cagra::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;

} cuvsCagraIndex;

typedef cuvsCagraIndex* cuvsCagraIndex_t;

/**
 * @brief Allocate CAGRA index
 *
 * @param[in] index cuvsCagraIndex_t to allocate
 * @return cagraError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t* index);

/**
 * @brief De-allocate CAGRA index
 *
 * @param[in] index cuvsCagraIndex_t to de-allocate
 */
CUVS_EXPORT cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t index);

/**
 * @brief Get dimension of the CAGRA index
 *
 * @param[in] index CAGRA index
 * @param[out] dim return dimension of the index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraIndexGetDims(cuvsCagraIndex_t index, int64_t* dim);

/**
 * @brief Get size of the CAGRA index
 *
 * @param[in] index CAGRA index
 * @param[out] size return number of vectors in the index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraIndexGetSize(cuvsCagraIndex_t index, int64_t* size);

/**
 * @brief Get graph degree of the CAGRA index
 *
 * @param[in] index CAGRA index
 * @param[out] graph_degree return graph degree
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraIndexGetGraphDegree(cuvsCagraIndex_t index, int64_t* graph_degree);

/**
 * @brief Returns a view of the CAGRA dataset
 *
 * This function returns a non-owning view of the CAGRA dataset.
 * The output will be referencing device memory that is directly used
 * in CAGRA, without copying the dataset at all. This means that the
 * output is only valid as long as the CAGRA index is alive, and once
 * cuvsCagraIndexDestroy is called on the cagra index - the returned
 * dataset view will be invalid.
 *
 * Note that the DLManagedTensor dataset returned will have an associated
 * 'deleter' function that must be called when the dataset is no longer
 * needed. This will free up host memory that stores the shape of the
 * dataset view.
 *
 * @param[in] index CAGRA index
 * @param[out] dataset the dataset used in cagra
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraIndexGetDataset(cuvsCagraIndex_t index, DLManagedTensor* dataset);

/**
 * @brief Returns a view of the CAGRA graph
 *
 * This function returns a non-owning view of the CAGRA graph.
 * The output will be referencing device memory that is directly used
 * in CAGRA, without copying the graph at all. This means that the
 * output is only valid as long as the CAGRA index is alive, and once
 * cuvsCagraIndexDestroy is called on the cagra index - the returned
 * graph view will be invalid.
 *
 * Note that the DLManagedTensor graph returned will have an associated
 * 'deleter' function that must be called when the graph is no longer
 * needed. This will free up host memory that stores the metadata for the
 * graph view.
 *
 * @param[in] index CAGRA index
 * @param[out] graph the output knn graph.
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraIndexGetGraph(cuvsCagraIndex_t index, DLManagedTensor* graph);

/**
 * @}
 */

/**
 * @defgroup cagra_c_index_build C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Build a CAGRA index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
 *        3. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *        4. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create default index params
 * cuvsCagraIndexParams_t params;
 * cuvsError_t params_create_status = cuvsCagraIndexParamsCreate(&params);
 *
 * // Create CAGRA index
 * cuvsCagraIndex_t index;
 * cuvsError_t index_create_status = cuvsCagraIndexCreate(&index);
 *
 * // Build the CAGRA Index
 * cuvsError_t build_status = cuvsCagraBuild(res, params, &dataset, index);
 *
 * // de-allocate `params`, `index` and `res`
 * cuvsError_t params_destroy_status = cuvsCagraIndexParamsDestroy(params);
 * cuvsError_t index_destroy_status = cuvsCagraIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsCagraIndexParams_t used to build CAGRA index
 * @param[in] dataset DLManagedTensor* training dataset
 * @param[inout] index cuvsCagraIndex_t Newly built CAGRA index. This index needs to be already
 *                                      created with cuvsCagraIndexCreate.
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraBuild(cuvsResources_t res,
                           cuvsCagraIndexParams_t params,
                           DLManagedTensor* dataset,
                           cuvsCagraIndex_t index);

/**
 * @}
 */

/**
 * @defgroup cagra_c_extend_params C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Extend a CAGRA index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
 *        3. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *        4. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsCagraExtendParams_t used to extend CAGRA index
 * @param[in] additional_dataset DLManagedTensor* additional dataset
 * @param[in,out] index cuvsCagraIndex_t CAGRA index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsCagraExtend(cuvsResources_t res,
                            cuvsCagraExtendParams_t params,
                            DLManagedTensor* additional_dataset,
                            cuvsCagraIndex_t index);

/**
 * @}
 */

/**
 * @defgroup cagra_c_index_search C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */
/**
 * @brief Search a CAGRA index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`.
 *        It is also important to note that the CAGRA Index must have been built
 *        with the same type of `queries`, such that `index.dtype.code ==
 * queries.dl_tensor.dtype.code` Types for input are:
 *        1. `queries`:
 *          a. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *          b. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
 *          c. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *          d. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *        2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32`
 *                     or `kDLDataType.code == kDLInt`  and `kDLDataType.bits = 64`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 * DLManagedTensor queries;
 * DLManagedTensor neighbors;
 *
 * // Create default search params
 * cuvsCagraSearchParams_t params;
 * cuvsError_t params_create_status = cuvsCagraSearchParamsCreate(&params);
 *
 * // Search the `index` built using `cuvsCagraBuild`
 * cuvsError_t search_status = cuvsCagraSearch(res, params, index, &queries, &neighbors,
 * &distances);
 *
 * // de-allocate `params` and `res`
 * cuvsError_t params_destroy_status = cuvsCagraSearchParamsDestroy(params);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsCagraSearchParams_t used to search CAGRA index
 * @param[in] index cuvsCagraIndex which has been returned by `cuvsCagraBuild`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 * @param[in] filter cuvsFilter input filter that can be used
              to filter queries and neighbors based on the given bitset.
 */
CUVS_EXPORT cuvsError_t cuvsCagraSearch(cuvsResources_t res,
                            cuvsCagraSearchParams_t params,
                            cuvsCagraIndex_t index,
                            DLManagedTensor* queries,
                            DLManagedTensor* neighbors,
                            DLManagedTensor* distances,
                            cuvsFilter filter);

/**
 * @}
 */

/**
 * @defgroup cagra_c_index_serialize CAGRA C-API serialize functions
 * @{
 */
/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.c}
 * #include <cuvs/neighbors/cagra.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // create an index with `cuvsCagraBuild`
 * cuvsCagraSerialize(res, "/path/to/index", index, true);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 *
 */
CUVS_EXPORT cuvsError_t cuvsCagraSerialize(cuvsResources_t res,
                               const char* filename,
                               cuvsCagraIndex_t index,
                               bool include_dataset);

/**
 * Save the CAGRA index to file in hnswlib format.
 * NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
 *       as the serialization format is not compatible with the original hnswlib.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // create an index with `cuvsCagraBuild`
 * cuvsCagraSerializeHnswlib(res, "/path/to/index", index);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 *
 */
CUVS_EXPORT cuvsError_t cuvsCagraSerializeToHnswlib(cuvsResources_t res,
                                        const char* filename,
                                        cuvsCagraIndex_t index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the name of the file that stores the index
 * @param[inout] index cuvsCagraIndex_t CAGRA index loaded from disk. This index needs to be already
 *                                      created with cuvsCagraIndexCreate.
 */
CUVS_EXPORT cuvsError_t cuvsCagraDeserialize(cuvsResources_t res, const char* filename, cuvsCagraIndex_t index);

/**
 * Load index from a dataset and graph
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] metric cuvsDistanceType to use in the index
 * @param[in] graph the knn graph to use, shape (size, graph_degree)
 * @param[in] dataset the dataset to use, shape (size, dim)
 * @param[inout] index cuvsCagraIndex_t CAGRA index populated with the graph and dataset.
 *                                      This index needs to be already created with
 *                                      cuvsCagraIndexCreate.
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Create CAGRA index
 * cuvsCagraIndex_t index;
 * cuvsError_t index_create_status = cuvsCagraIndexCreate(&index);
 *
 * // Assume a populated `DLManagedTensor` type here for the graph and dataset
 * DLManagedTensor dataset;
 * DLManagedTensor graph;
 *
 * cuvsDistanceType metric = L2Expanded;
 *
 * // Build the CAGRA Index from the graph/dataset
 * cuvsError_t status = cuvsCagraIndexFromArgs(res, metric, &graph, &dataset, index);
 *
 * @endcode
 */
CUVS_EXPORT cuvsError_t cuvsCagraIndexFromArgs(cuvsResources_t res,
                                   cuvsDistanceType metric,
                                   DLManagedTensor* graph,
                                   DLManagedTensor* dataset,
                                   cuvsCagraIndex_t index);

/**
 * @}
 */

/**
 * @defgroup cagra_c_index_merge CAGRA C-API merge functions
 * @{
 */

/**
 * @brief Merge multiple CAGRA indices into a single CAGRA index.
 *
 * All input indices must have been built with the same data type (`index.dtype`) and
 * have the same dimensionality (`index.dims`). The merged index uses the output
 * parameters specified in `cuvsCagraIndexParams`.
 *
 * Input indices must have:
 *  - `index.dtype.code` and `index.dtype.bits` matching across all indices.
 *  - Supported data types for indices:
 *      a. `kDLFloat` with `bits = 32`
 *      b. `kDLFloat` with `bits = 16`
 *      c. `kDLInt` with `bits = 8`
 *      d. `kDLUInt` with `bits = 8`
 *
 * The resulting output index will have the same data type as the input indices.
 *
 * Example:
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
 *
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * cuvsCagraIndex_t index1, index2, merged_index;
 * cuvsCagraIndexCreate(&index1);
 * cuvsCagraIndexCreate(&index2);
 * cuvsCagraIndexCreate(&merged_index);
 *
 * // Assume index1 and index2 have been built using cuvsCagraBuild
 *
 * cuvsCagraIndexParams_t merge_params;
 * cuvsError_t params_create_status = cuvsCagraIndexParamsCreate(&merge_params);
 *
 * cuvsError_t merge_status = cuvsCagraMerge(res, merge_params, (cuvsCagraIndex_t[]){index1,
 * index2}, 2, merged_index);
 *
 * // Use merged_index for search operations
 *
 * cuvsError_t params_destroy_status = cuvsCagraIndexParamsDestroy(merge_params);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsCagraIndexParams_t parameters controlling merge behavior
 * @param[in] indices Array of input cuvsCagraIndex_t handles to merge
 * @param[in] num_indices Number of input indices
 * @param[in] filter Filter that can be used to filter out vectors from the merged index
 * @param[out] output_index Output handle that will store the merged index.
 *                          Must be initialized using `cuvsCagraIndexCreate` before use.
 */
CUVS_EXPORT cuvsError_t cuvsCagraMerge(cuvsResources_t res,
                           cuvsCagraIndexParams_t params,
                           cuvsCagraIndex_t* indices,
                           size_t num_indices,
                           cuvsFilter filter,
                           cuvsCagraIndex_t output_index);

/**
 * @}
 */
#ifdef __cplusplus
}
#endif
