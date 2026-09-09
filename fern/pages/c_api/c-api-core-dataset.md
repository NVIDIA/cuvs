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
  CUVS_DATASET_LAYOUT_PADDED = 1
} cuvsDatasetLayout_t;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_DATASET_LAYOUT_STANDARD` | `0` |
| `CUVS_DATASET_LAYOUT_PADDED` | `1` |

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
