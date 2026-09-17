---
slug: api-reference/c-api-core-dataset
---

# Dataset

_Source header: `cuvs/core/dataset.h`_

## Types

<a id="cuvsdatasetlayout-t"></a>
### cuvsDatasetLayout_t

Generic dataset layout kind for C API dataset handles.

```c
typedef enum {
  CUVS_DATASET_LAYOUT_STANDARD = 0,
  CUVS_DATASET_LAYOUT_PADDED = 1,
  CUVS_DATASET_LAYOUT_PQ = 2
} cuvsDatasetLayout_t;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_DATASET_LAYOUT_STANDARD` | `0` |
| `CUVS_DATASET_LAYOUT_PADDED` | `1` |
| `CUVS_DATASET_LAYOUT_PQ` | `2` |

<a id="cuvsdatasetmemtype-t"></a>
### cuvsDatasetMemType_t

Memory space holding a C API dataset handle's data.

```c
typedef enum {
  CUVS_DATASET_MEM_TYPE_HOST = 0,
  CUVS_DATASET_MEM_TYPE_DEVICE = 1
} cuvsDatasetMemType_t;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_DATASET_MEM_TYPE_HOST` | `0` |
| `CUVS_DATASET_MEM_TYPE_DEVICE` | `1` |

<a id="destroy-addr"></a>
### destroy_addr

Dataset handle representing owning storage or a non-owning view.

`addr` points to C++ dataset storage or view metadata managed by the C API. `mem_type` identifies the memory space, `layout` identifies the data layout, and `is_owning` indicates whether the handle owns its backing data.

```c
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
  cuvsDatasetMemType_t mem_type;
  cuvsDatasetLayout_t layout;
  bool is_owning;
} cuvsDataset;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |
| `mem_type` | [`cuvsDatasetMemType_t`](/api-reference/c-api-core-dataset#cuvsdatasetmemtype-t) |  |
| `layout` | [`cuvsDatasetLayout_t`](/api-reference/c-api-core-dataset#cuvsdatasetlayout-t) |  |
| `is_owning` | `bool` |  |

<a id="cuvscagracompressionparams"></a>
### cuvsCagraCompressionParams

Parameters for PQ dataset compression.

The `cuvsCagraCompressionParams` tag is retained for source and ABI compatibility and is planned for removal in the 27.02 ABI-breaking release. Use `cuvsPQDatasetParams` in new code.

```c
typedef struct cuvsCagraCompressionParams {
  uint32_t pq_bits;
  uint32_t pq_dim;
  uint32_t vq_n_centers;
  uint32_t kmeans_n_iters;
  double vq_kmeans_trainset_fraction;
  double pq_kmeans_trainset_fraction;
} cuvsPQDatasetParams;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `pq_bits` | `uint32_t` | The bit length of the vector element after compression by PQ.<br /><br />Possible values: [4, 5, 6, 7, 8].<br /><br />Hint: the smaller the `pq_bits`, the smaller the index size and the better the search performance, but the lower the recall. |
| `pq_dim` | `uint32_t` | The dimensionality of the vector after compression by PQ. When zero, an optimal value is selected using a heuristic.<br /><br />TODO: at the moment `dim` must be a multiple `pq_dim`. |
| `vq_n_centers` | `uint32_t` | Vector Quantization (VQ) codebook size - number of "coarse cluster centers". When zero, an optimal value is selected using a heuristic. |
| `kmeans_n_iters` | `uint32_t` | The number of iterations searching for kmeans centers (both VQ & PQ phases). |
| `vq_kmeans_trainset_fraction` | `double` | The fraction of data to use during iterative kmeans building (VQ phase). When zero, an optimal value is selected using a heuristic. |
| `pq_kmeans_trainset_fraction` | `double` | The fraction of data to use during iterative kmeans building (PQ phase). When zero, an optimal value is selected using a heuristic. |

<a id="cuvscagracompressionparams-t"></a>
### cuvsCagraCompressionParams_t

Compatibility name for PQ dataset parameters; planned for removal in the 27.02 ABI-breaking release.

```c
typedef struct cuvsCagraCompressionParams* cuvsCagraCompressionParams_t;
```
