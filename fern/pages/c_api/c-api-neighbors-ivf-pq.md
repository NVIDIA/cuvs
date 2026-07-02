---
slug: api-reference/c-api-neighbors-ivf-pq
---

# IVF PQ

_Source header: `cuvs/neighbors/ivf_pq.h`_

## IVF-PQ index build parameters

<a id="cuvsivfpqcodebookgen"></a>
### cuvsIvfPqCodebookGen

A type for specifying how PQ codebooks are created

```c
enum cuvsIvfPqCodebookGen {
  CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE = 0,
  CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER = 1
};
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE` | `0` |
| `CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER` | `1` |

<a id="cuvsivfpqlistlayout"></a>
### cuvsIvfPqListLayout

A type for specifying the memory layout of IVF-PQ list data

```c
enum cuvsIvfPqListLayout {
  CUVS_IVF_PQ_LIST_LAYOUT_FLAT = 0,
  CUVS_IVF_PQ_LIST_LAYOUT_INTERLEAVED = 1
};
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_IVF_PQ_LIST_LAYOUT_FLAT` | `0` |
| `CUVS_IVF_PQ_LIST_LAYOUT_INTERLEAVED` | `1` |

<a id="cuvsivfpqindexparams"></a>
### cuvsIvfPqIndexParams

Supplemental parameters to build IVF-PQ Index

```c
struct cuvsIvfPqIndexParams {
  cuvsDistanceType metric;
  float metric_arg;
  bool add_data_on_build;
  uint32_t n_lists;
  uint32_t kmeans_n_iters;
  double kmeans_trainset_fraction;
  uint32_t pq_bits;
  uint32_t pq_dim;
  enum cuvsIvfPqCodebookGen codebook_kind;
  bool force_random_rotation;
  bool conservative_memory_allocation;
  uint32_t max_train_points_per_pq_code;
  enum cuvsIvfPqListLayout codes_layout;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | Distance type. |
| `metric_arg` | `float` | The argument used by some distance metrics. |
| `add_data_on_build` | `bool` | Whether to add the dataset content to the index, i.e.:<br /><br />- `true` means the index is filled with the dataset vectors and ready to search after calling `build`.<br />- `false` means `build` only trains the underlying model (e.g. quantizer or clustering), but the index is left empty; you'd need to call `extend` on the index afterwards to populate it. |
| `n_lists` | `uint32_t` | The number of inverted lists (clusters)<br /><br />Hint: the number of vectors per cluster (`n_rows/n_lists`) should be approximately 1,000 to 10,000. |
| `kmeans_n_iters` | `uint32_t` | The number of iterations searching for kmeans centers (index building). |
| `kmeans_trainset_fraction` | `double` | The fraction of data to use during iterative kmeans building. |
| `pq_bits` | `uint32_t` | The bit length of the vector element after compression by PQ.<br /><br />Possible values: [4, 5, 6, 7, 8].<br /><br />Hint: the smaller the 'pq_bits', the smaller the index size and the better the search performance, but the lower the recall. |
| `pq_dim` | `uint32_t` | The dimensionality of the vector after compression by PQ. When zero, an optimal value is selected using a heuristic.<br /><br />NB: `pq_dim * pq_bits` must be a multiple of 8.<br /><br />Hint: a smaller 'pq_dim' results in a smaller index size and better search performance, but lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number, but multiple of 8 are desirable for good performance. If 'pq_bits' is not 8, 'pq_dim' should be a multiple of 8. For good performance, it is desirable that 'pq_dim' is a multiple of 32. Ideally, 'pq_dim' should be also a divisor of the dataset dim. |
| `codebook_kind` | [`enum cuvsIvfPqCodebookGen`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqcodebookgen) | How PQ codebooks are created. |
| `force_random_rotation` | `bool` | Apply a random rotation matrix on the input data and queries even if `dim % pq_dim == 0`.<br /><br />Note: if `dim` is not multiple of `pq_dim`, a random rotation is always applied to the input data and queries to transform the working space from `dim` to `rot_dim`, which may be slightly larger than the original space and and is a multiple of `pq_dim` (`rot_dim % pq_dim == 0`). However, this transform is not necessary when `dim` is multiple of `pq_dim` (`dim == rot_dim`, hence no need in adding "extra" data columns / features).<br /><br />By default, if `dim == rot_dim`, the rotation transform is initialized with the identity matrix. When `force_random_rotation == true`, a random orthogonal transform matrix is generated regardless of the values of `dim` and `pq_dim`. |
| `conservative_memory_allocation` | `bool` | By default, the algorithm allocates more space than necessary for individual clusters (`list_data`). This allows to amortize the cost of memory allocation and reduce the number of data copies during repeated calls to `extend` (extending the database).<br /><br />The alternative is the conservative allocation behavior; when enabled, the algorithm always allocates the minimum amount of memory required to store the given number of records. Set this flag to `true` if you prefer to use as little GPU memory for the database as possible. |
| `max_train_points_per_pq_code` | `uint32_t` | The max number of data points to use per PQ code during PQ codebook training. Using more data points per PQ code may increase the quality of PQ codebook but may also increase the build time. The parameter is applied to both PQ codebook generation methods, i.e., PER_SUBSPACE and PER_CLUSTER. In both cases, we will use `pq_book_size * max_train_points_per_pq_code` training points to train each codebook. |
| `codes_layout` | [`enum cuvsIvfPqListLayout`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqlistlayout) | Memory layout of the IVF-PQ list data.<br /><br />- CUVS_IVF_PQ_LIST_LAYOUT_FLAT: Codes are stored contiguously, one vector's codes after another.<br />- CUVS_IVF_PQ_LIST_LAYOUT_INTERLEAVED: Codes are interleaved for optimized search performance. This is the default and recommended for search workloads. |

<a id="cuvsivfpqindexparamscreate"></a>
### cuvsIvfPqIndexParamsCreate

Allocate IVF-PQ Index params, and populate with default values

```c
cuvsError_t cuvsIvfPqIndexParamsCreate(cuvsIvfPqIndexParams_t* index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsIvfPqIndexParams_t*`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindexparams) | cuvsIvfPqIndexParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexparamsdestroy"></a>
### cuvsIvfPqIndexParamsDestroy

De-allocate IVF-PQ Index params

```c
cuvsError_t cuvsIvfPqIndexParamsDestroy(cuvsIvfPqIndexParams_t index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsIvfPqIndexParams_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindexparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-PQ index search parameters

<a id="cuvsivfpqsearchparams"></a>
### cuvsIvfPqSearchParams

Supplemental parameters to search IVF-PQ index

```c
struct cuvsIvfPqSearchParams {
  uint32_t n_probes;
  cudaDataType_t lut_dtype;
  cudaDataType_t internal_distance_dtype;
  cudaDataType_t coarse_search_dtype;
  uint32_t max_internal_batch_size;
  double preferred_shmem_carveout;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_probes` | `uint32_t` | The number of clusters to search. |
| `lut_dtype` | `cudaDataType_t` | Data type of look up table to be created dynamically at search time.<br /><br />Possible values: [CUDA_R_32F, CUDA_R_16F, CUDA_R_8U]<br /><br />The use of low-precision types reduces the amount of shared memory required at search time, so fast shared memory kernels can be used even for datasets with large dimansionality. Note that the recall is slightly degraded when low-precision type is selected. |
| `internal_distance_dtype` | `cudaDataType_t` | Storage data type for distance/similarity computed at search time.<br /><br />Possible values: [CUDA_R_16F, CUDA_R_32F]<br /><br />If the performance limiter at search time is device memory access, selecting FP16 will improve performance slightly. |
| `coarse_search_dtype` | `cudaDataType_t` | The data type to use as the GEMM element type when searching the clusters to probe.<br /><br />Possible values: [CUDA_R_8I, CUDA_R_16F, CUDA_R_32F].<br /><br />- Legacy default: CUDA_R_32F (float)<br />- Recommended for performance: CUDA_R_16F (half)<br />- Experimental/low-precision: CUDA_R_8I (int8_t) (WARNING: int8_t variant degrades recall unless data is normalized and low-dimensional) |
| `max_internal_batch_size` | `uint32_t` | Set the internal batch size to improve GPU utilization at the cost of larger memory footprint. |
| `preferred_shmem_carveout` | `double` | Preferred fraction of SM's unified memory / L1 cache to be used as shared memory.<br /><br />Possible values: [0.0 - 1.0] as a fraction of the `sharedMemPerMultiprocessor`.<br /><br />One wants to increase the carveout to make sure a good GPU occupancy for the main search kernel, but not to keep it too high to leave some memory to be used as L1 cache. Note, this value is interpreted only as a hint. Moreover, a GPU usually allows only a fixed set of cache configurations, so the provided value is rounded up to the nearest configuration. Refer to the NVIDIA tuning guide for the target GPU architecture.<br /><br />Note, this is a low-level tuning parameter that can have drastic negative effects on the search performance if tweaked incorrectly. |

<a id="cuvsivfpqsearchparamscreate"></a>
### cuvsIvfPqSearchParamsCreate

Allocate IVF-PQ search params, and populate with default values

```c
cuvsError_t cuvsIvfPqSearchParamsCreate(cuvsIvfPqSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsIvfPqSearchParams_t*`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqsearchparams) | cuvsIvfPqSearchParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqsearchparamsdestroy"></a>
### cuvsIvfPqSearchParamsDestroy

De-allocate IVF-PQ search params

```c
cuvsError_t cuvsIvfPqSearchParamsDestroy(cuvsIvfPqSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsIvfPqSearchParams_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqsearchparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-PQ index

<a id="cuvsivfpqindex"></a>
### cuvsIvfPqIndex

Struct to hold address of cuvs::neighbors::ivf_pq::index and its active trained dtype

```c
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsIvfPqIndex;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |

<a id="cuvsivfpqindexcreate"></a>
### cuvsIvfPqIndexCreate

Allocate IVF-PQ index

```c
cuvsError_t cuvsIvfPqIndexCreate(cuvsIvfPqIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfPqIndex_t*`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | cuvsIvfPqIndex_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexdestroy"></a>
### cuvsIvfPqIndexDestroy

De-allocate IVF-PQ index

```c
cuvsError_t cuvsIvfPqIndexDestroy(cuvsIvfPqIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | cuvsIvfPqIndex_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexgetnlists"></a>
### cuvsIvfPqIndexGetNLists

Get the number of clusters/inverted lists

```c
cuvsError_t cuvsIvfPqIndexGetNLists(cuvsIvfPqIndex_t index, int64_t* n_lists);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) |  |
| `n_lists` |  | `int64_t*` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexgetdim"></a>
### cuvsIvfPqIndexGetDim

Get the dimensionality

```c
cuvsError_t cuvsIvfPqIndexGetDim(cuvsIvfPqIndex_t index, int64_t* dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) |  |
| `dim` |  | `int64_t*` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexgetsize"></a>
### cuvsIvfPqIndexGetSize

Get the size of the index

```c
cuvsError_t cuvsIvfPqIndexGetSize(cuvsIvfPqIndex_t index, int64_t* size);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) |  |
| `size` |  | `int64_t*` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexgetpqdim"></a>
### cuvsIvfPqIndexGetPqDim

Get the dimensionality of an encoded vector after compression by PQ.

```c
cuvsError_t cuvsIvfPqIndexGetPqDim(cuvsIvfPqIndex_t index, int64_t* pq_dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) |  |
| `pq_dim` |  | `int64_t*` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexgetpqbits"></a>
### cuvsIvfPqIndexGetPqBits

Get the bit length of an encoded vector element after compression by PQ.

```c
cuvsError_t cuvsIvfPqIndexGetPqBits(cuvsIvfPqIndex_t index, int64_t* pq_bits);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) |  |
| `pq_bits` |  | `int64_t*` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexgetpqlen"></a>
### cuvsIvfPqIndexGetPqLen

Get the Dimensionality of a subspace, i.e. the number of vector

```c
cuvsError_t cuvsIvfPqIndexGetPqLen(cuvsIvfPqIndex_t index, int64_t* pq_len);
```

components mapped to a subspace

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) |  |
| `pq_len` |  | `int64_t*` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexgetcenters"></a>
### cuvsIvfPqIndexGetCenters

Get the cluster centers corresponding to the lists in the original space

```c
cuvsError_t cuvsIvfPqIndexGetCenters(cuvsIvfPqIndex_t index, DLManagedTensor* centers);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `centers` | out | `DLManagedTensor*` | Output tensor that will be populated with a non-owning view of the data |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexgetcenterspadded"></a>
### cuvsIvfPqIndexGetCentersPadded

Get the padded cluster centers [n_lists, dim_ext] where dim_ext = round_up(dim + 1, 8)

```c
cuvsError_t cuvsIvfPqIndexGetCentersPadded(cuvsIvfPqIndex_t index, DLManagedTensor* centers);
```

This returns the full padded centers as a contiguous array, suitable for use with cuvsIvfPqBuildPrecomputed.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `centers` | out | `DLManagedTensor*` | Output tensor that will be populated with a non-owning view of the data |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexgetpqcenters"></a>
### cuvsIvfPqIndexGetPqCenters

Get the PQ cluster centers

```c
cuvsError_t cuvsIvfPqIndexGetPqCenters(cuvsIvfPqIndex_t index, DLManagedTensor* pq_centers);
```

- CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE: [pq_dim , pq_len, pq_book_size]
- CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER:  [n_lists, pq_len, pq_book_size]

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `pq_centers` | out | `DLManagedTensor*` | Output tensor that will be populated with a non-owning view of the data |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexgetcentersrot"></a>
### cuvsIvfPqIndexGetCentersRot

Get the rotated cluster centers [n_lists, rot_dim] where rot_dim = pq_len * pq_dim

```c
cuvsError_t cuvsIvfPqIndexGetCentersRot(cuvsIvfPqIndex_t index, DLManagedTensor* centers_rot);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `centers_rot` | out | `DLManagedTensor*` | Output tensor that will be populated with a non-owning view of the data |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexgetrotationmatrix"></a>
### cuvsIvfPqIndexGetRotationMatrix

Get the rotation matrix [rot_dim, dim] Transform matrix (original space -&gt; rotated padded space)

```c
cuvsError_t cuvsIvfPqIndexGetRotationMatrix(cuvsIvfPqIndex_t index,
DLManagedTensor* rotation_matrix);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `rotation_matrix` | out | `DLManagedTensor*` | Output tensor that will be populated with a non-owning view of the data |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexgetlistsizes"></a>
### cuvsIvfPqIndexGetListSizes

Get the sizes of each list

```c
cuvsError_t cuvsIvfPqIndexGetListSizes(cuvsIvfPqIndex_t index, DLManagedTensor* list_sizes);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `list_sizes` | out | `DLManagedTensor*` | Output tensor that will be populated with a non-owning view of the data |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexunpackcontiguouslistdata"></a>
### cuvsIvfPqIndexUnpackContiguousListData

Unpack `n_rows` consecutive PQ encoded vectors of a single list (cluster) in the compressed index starting at given `offset`, not expanded to one code per byte. Each code in the output buffer occupies ceildiv(index.pq_dim() * index.pq_bits(), 8) bytes.

```c
cuvsError_t cuvsIvfPqIndexUnpackContiguousListData(cuvsResources_t res,
cuvsIvfPqIndex_t index,
DLManagedTensor* out_codes,
uint32_t label,
uint32_t offset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | raft resource |
| `index` | in | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `out_codes` | out | `DLManagedTensor*` | the destination buffer [n_rows, ceildiv(index.pq_dim() * index.pq_bits(), 8)]. The length `n_rows` defines how many records to unpack, offset + n_rows must be smaller than or equal to the list size. This DLManagedTensor must already point to allocated device memory |
| `label` | in | `uint32_t` | The id of the list (cluster) to decode. |
| `offset` | in | `uint32_t` | How many records in the list to skip. |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqindexgetlistindices"></a>
### cuvsIvfPqIndexGetListIndices

Get the indices of each vector in a ivf-pq list

```c
cuvsError_t cuvsIvfPqIndexGetListIndices(cuvsIvfPqIndex_t index,
uint32_t label,
DLManagedTensor* out_labels);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `label` | in | `uint32_t` | The id of the list (cluster) to decode. |
| `out_labels` | out | `DLManagedTensor*` | output tensor that will be populated with a non-owning view of the data |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-PQ index build

<a id="cuvsivfpqbuild"></a>
### cuvsIvfPqBuild

Build a IVF-PQ index with a `DLManagedTensor` which has underlying `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`, or `kDLCPU`. Also, acceptable underlying types are:

1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
3. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
4. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`

```c
cuvsError_t cuvsIvfPqBuild(cuvsResources_t res,
cuvsIvfPqIndexParams_t params,
DLManagedTensor* dataset,
cuvsIvfPqIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsIvfPqIndexParams_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindexparams) | cuvsIvfPqIndexParams_t used to build IVF-PQ index |
| `dataset` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `index` | out | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | cuvsIvfPqIndex_t Newly built IVF-PQ index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqbuildprecomputed"></a>
### cuvsIvfPqBuildPrecomputed

Build a view-type IVF-PQ index from device memory precomputed centroids and codebook.

```c
cuvsError_t cuvsIvfPqBuildPrecomputed(cuvsResources_t res,
cuvsIvfPqIndexParams_t params,
uint32_t dim,
DLManagedTensor* pq_centers,
DLManagedTensor* centers,
DLManagedTensor* centers_rot,
DLManagedTensor* rotation_matrix,
cuvsIvfPqIndex_t index);
```

This function creates a non-owning index that stores a reference to the provided device data. All parameters must be provided with correct extents. The caller is responsible for ensuring the lifetime of the input data exceeds the lifetime of the returned index.

The index_params must be consistent with the provided matrices. Specifically:

- index_params.codebook_kind determines the expected shape of pq_centers
- index_params.metric will be stored in the index
- index_params.conservative_memory_allocation will be stored in the index The function will verify consistency between index_params, dim, and the matrix extents.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsIvfPqIndexParams_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindexparams) | cuvsIvfPqIndexParams_t used to configure the index (must be consistent with matrices) |
| `dim` | in | `uint32_t` | dimensionality of the input data |
| `pq_centers` | in | `DLManagedTensor*` | PQ codebook on device memory with required shape:<br />- codebook_kind CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE: [pq_dim, pq_len, pq_book_size]<br />- codebook_kind CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER:  [n_lists, pq_len, pq_book_size] |
| `centers` | in | `DLManagedTensor*` | Cluster centers in the original space [n_lists, dim_ext] where dim_ext = round_up(dim + 1, 8) |
| `centers_rot` | in | `DLManagedTensor*` | Rotated cluster centers [n_lists, rot_dim] where rot_dim = pq_len * pq_dim |
| `rotation_matrix` | in | `DLManagedTensor*` | Transform matrix (original space -&gt; rotated padded space) [rot_dim, dim] |
| `index` | out | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | cuvsIvfPqIndex_t Newly built view-type IVF-PQ index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-PQ index search

<a id="cuvsivfpqsearch"></a>
### cuvsIvfPqSearch

Search a IVF-PQ index with a `DLManagedTensor` which has underlying `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`. It is also important to note that the IVF-PQ Index must have been built with the same type of `queries`, such that `index.dtype.code == queries.dl_tensor.dtype.code` Types for input are:

1. `queries`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` or `kDLDataType.bits = 16`
2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32`
3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`

```c
cuvsError_t cuvsIvfPqSearch(cuvsResources_t res,
cuvsIvfPqSearchParams_t search_params,
cuvsIvfPqIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* neighbors,
DLManagedTensor* distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `search_params` | in | [`cuvsIvfPqSearchParams_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqsearchparams) | cuvsIvfPqSearchParams_t used to search IVF-PQ index |
| `index` | in | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | cuvsIvfPqIndex which has been returned by `cuvsIvfPqBuild` |
| `queries` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset to search |
| `neighbors` | out | `DLManagedTensor*` | DLManagedTensor* output `k` neighbors for queries |
| `distances` | out | `DLManagedTensor*` | DLManagedTensor* output `k` distances for queries |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-PQ C-API serialize functions

<a id="cuvsivfpqserialize"></a>
### cuvsIvfPqSerialize

Save the index to file.

```c
cuvsError_t cuvsIvfPqSerialize(cuvsResources_t res, const char* filename, cuvsIvfPqIndex_t index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the file name for saving the index |
| `index` | in | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | IVF-PQ index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsivfpqdeserialize"></a>
### cuvsIvfPqDeserialize

Load index from file.

```c
cuvsError_t cuvsIvfPqDeserialize(cuvsResources_t res, const char* filename, cuvsIvfPqIndex_t index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the name of the file that stores the index |
| `index` | out | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | IVF-PQ index loaded disk |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-PQ index extend

<a id="cuvsivfpqextend"></a>
### cuvsIvfPqExtend

Extend the index with the new data.

```c
cuvsError_t cuvsIvfPqExtend(cuvsResources_t res,
DLManagedTensor* new_vectors,
DLManagedTensor* new_indices,
cuvsIvfPqIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `new_vectors` | in | `DLManagedTensor*` | DLManagedTensor* the new vectors to add to the index |
| `new_indices` | in | `DLManagedTensor*` | DLManagedTensor* vector of new indices for the new vectors |
| `index` | inout | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | IVF-PQ index to be extended |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## IVF-PQ index transform

<a id="cuvsivfpqtransform"></a>
### cuvsIvfPqTransform

Transform the input data by applying pq-encoding

```c
cuvsError_t cuvsIvfPqTransform(cuvsResources_t res,
cuvsIvfPqIndex_t index,
DLManagedTensor* input_dataset,
DLManagedTensor* output_labels,
DLManagedTensor* output_dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `index` | in | [`cuvsIvfPqIndex_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindex) | IVF-PQ index |
| `input_dataset` | in | `DLManagedTensor*` | DLManagedTensor* vectors to transform |
| `output_labels` | out | `DLManagedTensor*` | DLManagedTensor* Vector of cluster labels for each vector in the input |
| `output_dataset` | out | `DLManagedTensor*` | DLManagedTensor* input vectors after pq-encoding |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
