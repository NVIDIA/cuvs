---
slug: api-reference/java-api-com-nvidia-cuvs-hnswmaterializeparams
---

# HnswMaterializeParams

_Java package: `com.nvidia.cuvs`_

```java
public class HnswMaterializeParams
```

Parameters for materializing a layered HNSW artifact into a standard hnswlib
index file on disk.

## Public Members

### getDatasetPath

```java
public String getDatasetPath()
```

Gets the local dataset path holding the original-ID-ordered vectors used to
build the artifact.

**Returns**

the dataset path, or null if not set

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswMaterializeParams.java:30`_

### getMaxHostMemoryGb

```java
public double getMaxHostMemoryGb()
```

Gets the upper bound on host memory (in GiB) used for the base-topology
reorder buffer. When `&lt;= 0`, the whole base topology is reordered in a
single in-memory pass.

**Returns**

the max host memory in GiB (0 means a single in-memory pass)

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswMaterializeParams.java:41`_

### getNumThreads

```java
public int getNumThreads()
```

Gets the number of host threads to use. When 0, the maximum number of
threads is used.

**Returns**

the number of threads

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswMaterializeParams.java:51`_

### Builder

```java
public Builder()
```

Constructs this Builder.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswMaterializeParams.java:78`_

### withDatasetPath

```java
public Builder withDatasetPath(String datasetPath)
```

Sets the local dataset path holding the original-ID-ordered vectors used to
build the artifact. Supported formats match layered deserialization:
`.npy` and ANN benchmark `*.bin` files with a
`[uint32 rows, uint32 cols]` header (`.fbin`, `.f16bin`,
`.u8bin`, `.i8bin`).

**Parameters**

| Name | Description |
| --- | --- |
| `datasetPath` | the local dataset path |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswMaterializeParams.java:90`_

### withMaxHostMemoryGb

```java
public Builder withMaxHostMemoryGb(double maxHostMemoryGb)
```

Sets the upper bound on host memory (in GiB) used for the base-topology
reorder buffer.

When `&lt;= 0` (default), the whole base topology is reordered in a single
in-memory pass (no temporary files). When set, the base topology is reordered
through bucketed temporary files so that peak host memory stays close to this
budget.

**Parameters**

| Name | Description |
| --- | --- |
| `maxHostMemoryGb` | the max host memory in GiB |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswMaterializeParams.java:107`_

### withNumThreads

```java
public Builder withNumThreads(int numThreads)
```

Sets the number of host threads to use. When 0 (default), the maximum number
of threads is used.

**Parameters**

| Name | Description |
| --- | --- |
| `numThreads` | the number of threads |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswMaterializeParams.java:119`_

### build

```java
public HnswMaterializeParams build()
```

Builds an instance of `HnswMaterializeParams`.

**Returns**

an instance of `HnswMaterializeParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswMaterializeParams.java:129`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswMaterializeParams.java:13`_
