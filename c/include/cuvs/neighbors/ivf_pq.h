/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup ivf_pq_c_index_params IVF-PQ index build parameters
 * @{
 */
/**
 * @brief A type for specifying how PQ codebooks are created
 *
 */
enum cuvsIvfPqCodebookGen {
  CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE = 0,
  CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER  = 1,
};

/**
 * @brief A type for specifying the memory layout of IVF-PQ list data
 *
 */
enum cuvsIvfPqListLayout {
  CUVS_IVF_PQ_LIST_LAYOUT_FLAT        = 0,
  CUVS_IVF_PQ_LIST_LAYOUT_INTERLEAVED = 1,
};

/**
 * @brief Supplemental parameters to build IVF-PQ Index
 *
 */
struct cuvsIvfPqIndexParams {
  /** Distance type. */
  cuvsDistanceType metric;
  /** The argument used by some distance metrics. */
  float metric_arg;
  /**
   * Whether to add the dataset content to the index, i.e.:
   *
   *  - `true` means the index is filled with the dataset vectors and ready to search after calling
   * `build`.
   *  - `false` means `build` only trains the underlying model (e.g. quantizer or clustering), but
   * the index is left empty; you'd need to call `extend` on the index afterwards to populate it.
   */
  bool add_data_on_build;
  /**
   * The number of inverted lists (clusters)
   *
   * Hint: the number of vectors per cluster (`n_rows/n_lists`) should be approximately 1,000 to
   * 10,000.
   */
  uint32_t n_lists;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters;
  /** The fraction of data to use during iterative kmeans building. */
  double kmeans_trainset_fraction;
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
   * The dimensionality of the vector after compression by PQ. When zero, an optimal value is
   * selected using a heuristic.
   *
   * NB: `pq_dim * pq_bits` must be a multiple of 8.
   *
   * Hint: a smaller 'pq_dim' results in a smaller index size and better search performance, but
   * lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number, but multiple of 8 are
   * desirable for good performance. If 'pq_bits' is not 8, 'pq_dim' should be a multiple of 8.
   * For good performance, it is desirable that 'pq_dim' is a multiple of 32. Ideally, 'pq_dim'
   * should be also a divisor of the dataset dim.
   */
  uint32_t pq_dim;
  /** How PQ codebooks are created. */
  enum cuvsIvfPqCodebookGen codebook_kind;
  /**
   * Apply a random rotation matrix on the input data and queries even if `dim % pq_dim == 0`.
   *
   * Note: if `dim` is not multiple of `pq_dim`, a random rotation is always applied to the input
   * data and queries to transform the working space from `dim` to `rot_dim`, which may be slightly
   * larger than the original space and and is a multiple of `pq_dim` (`rot_dim % pq_dim == 0`).
   * However, this transform is not necessary when `dim` is multiple of `pq_dim`
   *   (`dim == rot_dim`, hence no need in adding "extra" data columns / features).
   *
   * By default, if `dim == rot_dim`, the rotation transform is initialized with the identity
   * matrix. When `force_random_rotation == true`, a random orthogonal transform matrix is generated
   * regardless of the values of `dim` and `pq_dim`.
   */
  bool force_random_rotation;
  /**
   * By default, the algorithm allocates more space than necessary for individual clusters
   * (`list_data`). This allows to amortize the cost of memory allocation and reduce the number of
   * data copies during repeated calls to `extend` (extending the database).
   *
   * The alternative is the conservative allocation behavior; when enabled, the algorithm always
   * allocates the minimum amount of memory required to store the given number of records. Set this
   * flag to `true` if you prefer to use as little GPU memory for the database as possible.
   */
  bool conservative_memory_allocation;

  /**
   * The max number of data points to use per PQ code during PQ codebook training. Using more data
   * points per PQ code may increase the quality of PQ codebook but may also increase the build
   * time. The parameter is applied to both PQ codebook generation methods, i.e., PER_SUBSPACE and
   * PER_CLUSTER. In both cases, we will use `pq_book_size * max_train_points_per_pq_code` training
   * points to train each codebook.
   */
  uint32_t max_train_points_per_pq_code;
  /**
   * Memory layout of the IVF-PQ list data.
   *
   * - CUVS_IVF_PQ_LIST_LAYOUT_FLAT: Codes are stored contiguously, one vector's codes after another.
   * - CUVS_IVF_PQ_LIST_LAYOUT_INTERLEAVED: Codes are interleaved for optimized search performance.
   *   This is the default and recommended for search workloads.
   */
  enum cuvsIvfPqListLayout codes_layout;
};

typedef struct cuvsIvfPqIndexParams* cuvsIvfPqIndexParams_t;

/**
 * @brief Allocate IVF-PQ Index params, and populate with default values
 *
 * @param[in] index_params cuvsIvfPqIndexParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexParamsCreate(cuvsIvfPqIndexParams_t* index_params);

/**
 * @brief De-allocate IVF-PQ Index params
 *
 * @param[in] index_params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexParamsDestroy(cuvsIvfPqIndexParams_t index_params);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_c_search_params IVF-PQ index search parameters
 * @{
 */
/**
 * @brief Supplemental parameters to search IVF-PQ index
 *
 */
struct cuvsIvfPqSearchParams {
  /** The number of clusters to search. */
  uint32_t n_probes;
  /**
   * Data type of look up table to be created dynamically at search time.
   *
   * Possible values: [CUDA_R_32F, CUDA_R_16F, CUDA_R_8U]
   *
   * The use of low-precision types reduces the amount of shared memory required at search time, so
   * fast shared memory kernels can be used even for datasets with large dimansionality. Note that
   * the recall is slightly degraded when low-precision type is selected.
   */
  cudaDataType_t lut_dtype;
  /**
   * Storage data type for distance/similarity computed at search time.
   *
   * Possible values: [CUDA_R_16F, CUDA_R_32F]
   *
   * If the performance limiter at search time is device memory access, selecting FP16 will improve
   * performance slightly.
   */
  cudaDataType_t internal_distance_dtype;
  /**
   * The data type to use as the GEMM element type when searching the clusters to probe.
   *
   * Possible values: [CUDA_R_8I, CUDA_R_16F, CUDA_R_32F].
   *
   * - Legacy default: CUDA_R_32F (float)
   * - Recommended for performance: CUDA_R_16F (half)
   * - Experimental/low-precision: CUDA_R_8I (int8_t)
   *    (WARNING: int8_t variant degrades recall unless data is normalized and low-dimensional)
   */
  cudaDataType_t coarse_search_dtype;
  /**
   * Set the internal batch size to improve GPU utilization at the cost of larger memory footprint.
   */
  uint32_t max_internal_batch_size;
  /**
   * Preferred fraction of SM's unified memory / L1 cache to be used as shared memory.
   *
   * Possible values: [0.0 - 1.0] as a fraction of the `sharedMemPerMultiprocessor`.
   *
   * One wants to increase the carveout to make sure a good GPU occupancy for the main search
   * kernel, but not to keep it too high to leave some memory to be used as L1 cache. Note, this
   * value is interpreted only as a hint. Moreover, a GPU usually allows only a fixed set of cache
   * configurations, so the provided value is rounded up to the nearest configuration. Refer to the
   * NVIDIA tuning guide for the target GPU architecture.
   *
   * Note, this is a low-level tuning parameter that can have drastic negative effects on the search
   * performance if tweaked incorrectly.
   */
  double preferred_shmem_carveout;
};

typedef struct cuvsIvfPqSearchParams* cuvsIvfPqSearchParams_t;

/**
 * @brief Allocate IVF-PQ search params, and populate with default values
 *
 * @param[in] params cuvsIvfPqSearchParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqSearchParamsCreate(cuvsIvfPqSearchParams_t* params);

/**
 * @brief De-allocate IVF-PQ search params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqSearchParamsDestroy(cuvsIvfPqSearchParams_t params);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_c_index IVF-PQ index
 * @{
 */
/**
 * @brief Struct to hold address of cuvs::neighbors::ivf_pq::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsIvfPqIndex;

typedef cuvsIvfPqIndex* cuvsIvfPqIndex_t;

/**
 * @brief Allocate IVF-PQ index
 *
 * @param[in] index cuvsIvfPqIndex_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexCreate(cuvsIvfPqIndex_t* index);

/**
 * @brief De-allocate IVF-PQ index
 *
 * @param[in] index cuvsIvfPqIndex_t to de-allocate
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexDestroy(cuvsIvfPqIndex_t index);

/** Get the number of clusters/inverted lists */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexGetNLists(cuvsIvfPqIndex_t index, int64_t* n_lists);

/** Get the dimensionality */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexGetDim(cuvsIvfPqIndex_t index, int64_t* dim);

/** Get the size of the index */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexGetSize(cuvsIvfPqIndex_t index, int64_t* size);

/** Get the dimensionality of an encoded vector after compression by PQ. */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexGetPqDim(cuvsIvfPqIndex_t index, int64_t* pq_dim);

/** Get the bit length of an encoded vector element after compression by PQ.*/
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexGetPqBits(cuvsIvfPqIndex_t index, int64_t* pq_bits);

/** Get the Dimensionality of a subspace, i.e. the number of vector
 * components mapped to a subspace */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexGetPqLen(cuvsIvfPqIndex_t index, int64_t* pq_len);

/**
 * @brief Get the cluster centers corresponding to the lists in the original space
 *
 * @param[in] index cuvsIvfPqIndex_t Built Ivf-Pq index
 * @param[out] centers Output tensor that will be populated with a non-owning view of the data
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexGetCenters(cuvsIvfPqIndex_t index, DLManagedTensor* centers);

/**
 * @brief Get the padded cluster centers [n_lists, dim_ext]
 *   where dim_ext = round_up(dim + 1, 8)
 *
 * This returns the full padded centers as a contiguous array, suitable for
 * use with cuvsIvfPqBuildPrecomputed.
 *
 * @param[in] index cuvsIvfPqIndex_t Built Ivf-Pq index
 * @param[out] centers Output tensor that will be populated with a non-owning view of the data
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexGetCentersPadded(cuvsIvfPqIndex_t index, DLManagedTensor* centers);

/**
 * @brief Get the PQ cluster centers
 *
 *   - CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE: [pq_dim , pq_len, pq_book_size]
 *   - CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER:  [n_lists, pq_len, pq_book_size]
 *
 * @param[in] index cuvsIvfPqIndex_t Built Ivf-Pq index
 * @param[out] pq_centers Output tensor that will be populated with a non-owning view of the data
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexGetPqCenters(cuvsIvfPqIndex_t index, DLManagedTensor* pq_centers);

/**
 * @brief Get the rotated cluster centers [n_lists, rot_dim]
 *   where rot_dim = pq_len * pq_dim
 *
 * @param[in] index cuvsIvfPqIndex_t Built Ivf-Pq index
 * @param[out] centers_rot Output tensor that will be populated with a non-owning view of the data
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexGetCentersRot(cuvsIvfPqIndex_t index, DLManagedTensor* centers_rot);

/**
 * @brief Get the rotation matrix [rot_dim, dim]
 *   Transform matrix (original space -> rotated padded space)
 *
 * @param[in] index cuvsIvfPqIndex_t Built Ivf-Pq index
 * @param[out] rotation_matrix Output tensor that will be populated with a non-owning view of the
 * data
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexGetRotationMatrix(cuvsIvfPqIndex_t index,
                                            DLManagedTensor* rotation_matrix);

/**
 * @brief Get the sizes of each list
 *
 * @param[in] index cuvsIvfPqIndex_t Built Ivf-Pq index
 * @param[out] list_sizes Output tensor that will be populated with a non-owning view of the data
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexGetListSizes(cuvsIvfPqIndex_t index, DLManagedTensor* list_sizes);

/**
 * @brief Unpack `n_rows` consecutive PQ encoded vectors of a single list (cluster) in the
 * compressed index starting at given `offset`, not expanded to one code per byte. Each code in the
 * output buffer occupies ceildiv(index.pq_dim() * index.pq_bits(), 8) bytes.
 *
 * @param[in] res raft resource
 * @param[in] index cuvsIvfPqIndex_t Built Ivf-Pq index
 * @param[out] out_codes
 *   the destination buffer [n_rows, ceildiv(index.pq_dim() * index.pq_bits(), 8)].
 *   The length `n_rows` defines how many records to unpack,
 *   offset + n_rows must be smaller than or equal to the list size.
 *   This DLManagedTensor must already point to allocated device memory
 * @param[in] label
 *   The id of the list (cluster) to decode.
 * @param[in] offset
 *   How many records in the list to skip.
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexUnpackContiguousListData(cuvsResources_t res,
                                                   cuvsIvfPqIndex_t index,
                                                   DLManagedTensor* out_codes,
                                                   uint32_t label,
                                                   uint32_t offset);
/**
 * @brief Get the indices of each vector in a ivf-pq list
 *
 * @param[in] index cuvsIvfPqIndex_t Built Ivf-Pq index
 * @param[in] label
 *   The id of the list (cluster) to decode.
 * @param[out] out_labels
 *   output tensor that will be populated with a non-owning view of the data
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqIndexGetListIndices(cuvsIvfPqIndex_t index,
                                         uint32_t label,
                                         DLManagedTensor* out_labels);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_c_index_build IVF-PQ index build
 * @{
 */
/**
 * @brief Build a IVF-PQ index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
 *        3. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *        4. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/ivf_pq.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create default index params
 * cuvsIvfPqIndexParams_t index_params;
 * cuvsError_t params_create_status = cuvsIvfPqIndexParamsCreate(&index_params);
 *
 * // Create IVF-PQ index
 * cuvsIvfPqIndex_t index;
 * cuvsError_t index_create_status = cuvsIvfPqIndexCreate(&index);
 *
 * // Build the IVF-PQ Index
 * cuvsError_t build_status = cuvsIvfPqBuild(res, index_params, &dataset, index);
 *
 * // de-allocate `index_params`, `index` and `res`
 * cuvsError_t params_destroy_status = cuvsIvfPqIndexParamsDestroy(index_params);
 * cuvsError_t index_destroy_status = cuvsIvfPqIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsIvfPqIndexParams_t used to build IVF-PQ index
 * @param[in] dataset DLManagedTensor* training dataset
 * @param[out] index cuvsIvfPqIndex_t Newly built IVF-PQ index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqBuild(cuvsResources_t res,
                           cuvsIvfPqIndexParams_t params,
                           DLManagedTensor* dataset,
                           cuvsIvfPqIndex_t index);

/**
 * @brief Build a view-type IVF-PQ index from device memory precomputed centroids and codebook.
 *
 * This function creates a non-owning index that stores a reference to the provided device data.
 * All parameters must be provided with correct extents. The caller is responsible for ensuring
 * the lifetime of the input data exceeds the lifetime of the returned index.
 *
 * The index_params must be consistent with the provided matrices. Specifically:
 * - index_params.codebook_kind determines the expected shape of pq_centers
 * - index_params.metric will be stored in the index
 * - index_params.conservative_memory_allocation will be stored in the index
 * The function will verify consistency between index_params, dim, and the matrix extents.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsIvfPqIndexParams_t used to configure the index (must be consistent with
 * matrices)
 * @param[in] dim dimensionality of the input data
 * @param[in] pq_centers PQ codebook on device memory with required shape:
 *   - codebook_kind CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE: [pq_dim, pq_len, pq_book_size]
 *   - codebook_kind CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER:  [n_lists, pq_len, pq_book_size]
 * @param[in] centers Cluster centers in the original space [n_lists, dim_ext]
 *   where dim_ext = round_up(dim + 1, 8)
 * @param[in] centers_rot Rotated cluster centers [n_lists, rot_dim]
 *   where rot_dim = pq_len * pq_dim
 * @param[in] rotation_matrix Transform matrix (original space -> rotated padded space) [rot_dim,
 * dim]
 * @param[out] index cuvsIvfPqIndex_t Newly built view-type IVF-PQ index
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqBuildPrecomputed(cuvsResources_t res,
                                      cuvsIvfPqIndexParams_t params,
                                      uint32_t dim,
                                      DLManagedTensor* pq_centers,
                                      DLManagedTensor* centers,
                                      DLManagedTensor* centers_rot,
                                      DLManagedTensor* rotation_matrix,
                                      cuvsIvfPqIndex_t index);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_c_index_search IVF-PQ index search
 * @{
 */
/**
 * @brief Search a IVF-PQ index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`.
 *        It is also important to note that the IVF-PQ Index must have been built
 *        with the same type of `queries`, such that `index.dtype.code ==
 * queries.dl_tensor.dtype.code` Types for input are:
 *        1. `queries`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *            or `kDLDataType.bits = 16`
 *        2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/ivf_pq.h>
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
 * cuvsIvfPqSearchParams_t search_params;
 * cuvsError_t params_create_status = cuvsIvfPqSearchParamsCreate(&search_params);
 *
 * // Search the `index` built using `cuvsIvfPqBuild`
 * cuvsError_t search_status = cuvsIvfPqSearch(res, search_params, index, &queries, &neighbors,
 * &distances);
 *
 * // de-allocate `search_params` and `res`
 * cuvsError_t params_destroy_status = cuvsIvfPqSearchParamsDestroy(search_params);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] search_params cuvsIvfPqSearchParams_t used to search IVF-PQ index
 * @param[in] index cuvsIvfPqIndex which has been returned by `cuvsIvfPqBuild`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqSearch(cuvsResources_t res,
                            cuvsIvfPqSearchParams_t search_params,
                            cuvsIvfPqIndex_t index,
                            DLManagedTensor* queries,
                            DLManagedTensor* neighbors,
                            DLManagedTensor* distances);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_c_index_serialize IVF-PQ C-API serialize functions
 * @{
 */
/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <cuvs/neighbors/ivf_pq.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // create an index with `cuvsIvfPqBuild`
 * cuvsIvfPqSerialize(res, "/path/to/index", index, true);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-PQ index
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqSerialize(cuvsResources_t res, const char* filename, cuvsIvfPqIndex_t index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index IVF-PQ index loaded disk
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqDeserialize(cuvsResources_t res, const char* filename, cuvsIvfPqIndex_t index);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_c_index_extend IVF-PQ index extend
 * @{
 */
/**
 * @brief Extend the index with the new data.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] new_vectors DLManagedTensor* the new vectors to add to the index
 * @param[in] new_indices DLManagedTensor* vector of new indices for the new vectors
 * @param[inout] index IVF-PQ index to be extended
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqExtend(cuvsResources_t res,
                            DLManagedTensor* new_vectors,
                            DLManagedTensor* new_indices,
                            cuvsIvfPqIndex_t index);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_c_index_transform IVF-PQ index transform
 * @{
 */
/**
 * @brief Transform the input data by applying pq-encoding
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index IVF-PQ index
 * @param[in] input_dataset DLManagedTensor* vectors to transform
 * @param[out] output_labels DLManagedTensor* Vector of cluster labels for each vector in the input
 * @param[out] output_dataset DLManagedTensor* input vectors after pq-encoding
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsIvfPqTransform(cuvsResources_t res,
                               cuvsIvfPqIndex_t index,
                               DLManagedTensor* input_dataset,
                               DLManagedTensor* output_labels,
                               DLManagedTensor* output_dataset);
/**
 * @}
 */
#ifdef __cplusplus
}
#endif
