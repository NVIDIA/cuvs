---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-cagraindexparamsfactory
---

# CagraIndexParamsFactory

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class CagraIndexParamsFactory
```

A centralized place for producing `CagraIndexParams` from the cuvs-lucene input parameter
classes based on the chosen strategy.

For the `HEURISTIC` strategy the build heuristics are delegated to cuVS.

## Public Members

### create

```java
public static CagraIndexParams create( GPUSearchParams gpuSearchParams, long rows, long dimension)
```

Creates an instance of `CagraIndexParams` for the GPU-native CAGRA index based on the
chosen strategy in the `GPUSearchParams`.

**Parameters**

| Name | Description |
| --- | --- |
| `gpuSearchParams` | the input parameters for the build and search on the GPU API |
| `rows` | number of vectors in the data set |
| `dimension` | the dimension of the vectors in the data set |

**Returns**

an instance of `CagraIndexParams`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CagraIndexParamsFactory.java:29`_

### create

```java
public static CagraIndexParams create( AcceleratedHNSWParams acceleratedHNSWParams, long rows, long dimension)
```

Creates an instance of `CagraIndexParams` for the accelerated-HNSW index based on the
chosen strategy in the `AcceleratedHNSWParams`.

**Parameters**

| Name | Description |
| --- | --- |
| `acceleratedHNSWParams` | the input parameters for the build on the GPU API |
| `rows` | number of vectors in the data set |
| `dimension` | the dimension of the vectors in the data set |

**Returns**

an instance of `CagraIndexParams`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CagraIndexParamsFactory.java:73`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/CagraIndexParamsFactory.java:17`_
