---
slug: api-reference/c-api-cluster-mg-kmeans
---

# Multi-GPU K-Means

_Source header: `cuvs/cluster/mg_kmeans.h`_

## Multi-GPU k-means clustering APIs

<a id="cuvsmultigpukmeansfit"></a>
### cuvsMultiGpuKMeansFit

Find clusters with single-node multi-GPU k-means using host data.

```c
CUVS_EXPORT cuvsError_t cuvsMultiGpuKMeansFit(cuvsResources_t res,
cuvsKMeansParams_v2_t params,
DLManagedTensor* X,
DLManagedTensor* sample_weight,
DLManagedTensor* centroids,
double* inertia,
int* n_iter);
```

X, sample_weight, and centroids must be host-accessible, row-major, C-contiguous DLPack tensors. X and centroids must have dtype float32 or float64, and sample_weight must match X when provided.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsMultiGpuResources_t opaque C handle created by cuvsMultiGpuResourcesCreate or cuvsMultiGpuResourcesCreateWithDeviceIds. |
| `params` | in | [`cuvsKMeansParams_v2_t`](/api-reference/c-api-cluster-kmeans#cuvskmeansparams-v2) | Parameters for KMeans model. |
| `X` | in | `DLManagedTensor*` | Host training instances to cluster. [dim = n_samples x n_features] |
| `sample_weight` | in | `DLManagedTensor*` | Optional host weights for each observation in X. [len = n_samples] |
| `centroids` | inout | `DLManagedTensor*` | Host centroids. When init is Array, used as the initial cluster centers. The final generated centroids are copied back to this tensor. [dim = n_clusters x n_features] |
| `inertia` | out | `double*` | Sum of squared distances of samples to their closest cluster center. |
| `n_iter` | out | `int*` | Number of iterations run. |

**Returns**

[`CUVS_EXPORT cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
