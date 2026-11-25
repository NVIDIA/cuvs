---
slug: api-reference/c-api-cluster-assignment
---

# Assignment

_Source header: `cuvs/cluster/assignment.h`_

## Nearest-centroid assignment backends

<a id="cuvsassignmentkind"></a>
### cuvsAssignmentKind

The backend that assigns vectors to their nearest centroid.

```c
enum cuvsAssignmentKind {
  CUVS_ASSIGNMENT_IVF = 0
};
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_ASSIGNMENT_IVF` | `0` |

<a id="cuvsivfassignmentparams"></a>
### cuvsIvfAssignmentParams

Parameters of the IVF assignment (see `CUVS_ASSIGNMENT_IVF`).

```c
struct cuvsIvfAssignmentParams {
  uint32_t n_lists;
  uint32_t kmeans_n_iters;
  uint32_t n_probes;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_lists` | `uint32_t` | Number of lists; 0 means round(sqrt(n_centroids)). Only used when building an index. |
| `kmeans_n_iters` | `uint32_t` | Balanced k-means iterations used to group the centroids. Only used when building an index. |
| `n_probes` | `uint32_t` | Lists probed per vector (clamped to n_lists); 0 selects 16. |

<a id="cuvsassignmentindex"></a>
### cuvsAssignmentIndex

Index of an assignment backend, built once from a fixed set of centroids and reused across calls. Opaque; `kind` records which backend built it.

```c
typedef struct {
  uintptr_t addr;
} cuvsAssignmentIndex;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |

<a id="cuvsassignmentindexcreate"></a>
### cuvsAssignmentIndexCreate

Allocate an assignment index (empty until built).

```c
cuvsError_t cuvsAssignmentIndexCreate(cuvsAssignmentIndex_t *index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsAssignmentIndex_t*`](/api-reference/c-api-cluster-assignment#cuvsassignmentindex) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsassignmentindexdestroy"></a>
### cuvsAssignmentIndexDestroy

De-allocate an assignment index.

```c
cuvsError_t cuvsAssignmentIndexDestroy(cuvsAssignmentIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsAssignmentIndex_t`](/api-reference/c-api-cluster-assignment#cuvsassignmentindex) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsassignmentparams"></a>
### cuvsAssignmentParams

Selects the assignment backend and holds its parameters.

Allocate with `cuvsAssignmentParamsCreate`, which selects the IVF backend with default parameters. Operations that take an optional `cuvsAssignmentParams_t` use their default (exhaustive) assignment when it is NULL. Only the member of `kind` is read.

```c
struct cuvsAssignmentParams {
  cuvsAssignmentIndex_t index;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `index` | [`cuvsAssignmentIndex_t`](/api-reference/c-api-cluster-assignment#cuvsassignmentindex) | Optional prebuilt index, not owned (must outlive the call); NULL builds one per call and discards it. When set, it must have been built for `kind`, the index-building parameters (e.g. `ivf.n_lists`) are ignored, and its centroid count, dimension and metric must match the call. |

<a id="cuvsassignmentparams-cuvsivfassignmentparams"></a>
### cuvsAssignmentParams::cuvsIvfAssignmentParams

Parameters of `CUVS_ASSIGNMENT_IVF`.

```c
struct cuvsIvfAssignmentParams ivf;
```

<a id="cuvsassignmentparamscreate"></a>
### cuvsAssignmentParamsCreate

Allocate assignment params (IVF backend, default parameters, no index).

```c
cuvsError_t
cuvsAssignmentParamsCreate(cuvsAssignmentParams_t *params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` |  | [`cuvsAssignmentParams_t*`](/api-reference/c-api-cluster-assignment#cuvsassignmentparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsassignmentparamsdestroy"></a>
### cuvsAssignmentParamsDestroy

De-allocate assignment params.

```c
cuvsError_t
cuvsAssignmentParamsDestroy(cuvsAssignmentParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` |  | [`cuvsAssignmentParams_t`](/api-reference/c-api-cluster-assignment#cuvsassignmentparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvsassignmentindexbuild"></a>
### cuvsAssignmentIndexBuild

Build an assignment index from centroids.

```c
cuvsError_t cuvsAssignmentIndexBuild(cuvsResources_t res,
cuvsAssignmentParams_t params,
DLManagedTensor *centroids,
cuvsDistanceType metric,
cuvsAssignmentIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsAssignmentParams_t`](/api-reference/c-api-cluster-assignment#cuvsassignmentparams) | backend and its index-building parameters; `params->index` is ignored |
| `centroids` | in | `DLManagedTensor*` | DLManagedTensor* float32 device matrix [n_centroids, dim] |
| `metric` | in | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | L2Expanded, L2SqrtExpanded, CosineExpanded or InnerProduct |
| `index` | out | [`cuvsAssignmentIndex_t`](/api-reference/c-api-cluster-assignment#cuvsassignmentindex) | index to build, allocated with `cuvsAssignmentIndexCreate`; a previously built index is replaced |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
