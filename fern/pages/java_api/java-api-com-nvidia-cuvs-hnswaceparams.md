---
slug: api-reference/java-api-com-nvidia-cuvs-hnswaceparams
---

# HnswAceParams

_Java package: `com.nvidia.cuvs`_

```java
public class HnswAceParams
```

Parameters for ACE (Augmented Core Extraction) graph build for HNSW.
ACE enables building indexes for datasets too large to fit in GPU memory by:
1. Partitioning the dataset in core and augmented partitions using balanced k-means
2. Building sub-indexes for each partition independently
3. Concatenating sub-graphs into a final unified index

## Public Members

### getNpartitions

```java
public long getNpartitions()
```

Gets the number of partitions for ACE partitioned build.

**Returns**

the number of partitions

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswAceParams.java:37`_

### getBuildDir

```java
public String getBuildDir()
```

Gets the directory to store ACE build artifacts.

**Returns**

the build directory path

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswAceParams.java:46`_

### isUseDisk

```java
public boolean isUseDisk()
```

Gets whether disk-based storage is enabled for ACE build.

**Returns**

true if disk mode is enabled

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswAceParams.java:55`_

### getMaxHostMemoryGb

```java
public double getMaxHostMemoryGb()
```

Gets the maximum host memory limit in GiB.

**Returns**

the max host memory limit (0 means use available memory)

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswAceParams.java:64`_

### getMaxGpuMemoryGb

```java
public double getMaxGpuMemoryGb()
```

Gets the maximum GPU memory limit in GiB.

**Returns**

the max GPU memory limit (0 means use available memory)

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswAceParams.java:73`_

### Builder

```java
public Builder()
```

Constructs this Builder.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswAceParams.java:106`_

### withNpartitions

```java
public Builder withNpartitions(long npartitions)
```

Sets the number of partitions for ACE partitioned build.

When set to 0 (default), the number of partitions is automatically derived
based on available host and GPU memory to maximize partition size while
ensuring the build fits in memory.

Small values might improve recall but potentially degrade performance.
The partition size is on average 2 * (n_rows / npartitions) * dim *
sizeof(T). 2 is because of the core and augmented vectors. Please account
for imbalance in the partition sizes (up to 3x in our tests).

If the specified number of partitions results in partitions that exceed
available memory, the value will be automatically increased to fit memory
constraints and a warning will be issued.

**Parameters**

| Name | Description |
| --- | --- |
| `npartitions` | the number of partitions |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswAceParams.java:127`_

### withBuildDir

```java
public Builder withBuildDir(String buildDir)
```

Sets the directory to store ACE build artifacts.
Used when useDisk is true or when the graph does not fit in memory.

**Parameters**

| Name | Description |
| --- | --- |
| `buildDir` | the build directory path |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswAceParams.java:139`_

### withUseDisk

```java
public Builder withUseDisk(boolean useDisk)
```

Sets whether to use disk-based storage for ACE build.
When true, enables disk-based operations for memory-efficient graph construction.

**Parameters**

| Name | Description |
| --- | --- |
| `useDisk` | true to enable disk mode |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswAceParams.java:151`_

### withMaxHostMemoryGb

```java
public Builder withMaxHostMemoryGb(double maxHostMemoryGb)
```

Sets the maximum host memory to use for ACE build in GiB.

When set to 0 (default), uses available host memory.
Useful for testing or when running alongside other memory-intensive processes.

**Parameters**

| Name | Description |
| --- | --- |
| `maxHostMemoryGb` | the max host memory in GiB |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswAceParams.java:165`_

### withMaxGpuMemoryGb

```java
public Builder withMaxGpuMemoryGb(double maxGpuMemoryGb)
```

Sets the maximum GPU memory to use for ACE build in GiB.

When set to 0 (default), uses available GPU memory.
Useful for testing or when running alongside other memory-intensive processes.

**Parameters**

| Name | Description |
| --- | --- |
| `maxGpuMemoryGb` | the max GPU memory in GiB |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswAceParams.java:179`_

### build

```java
public HnswAceParams build()
```

Builds an instance of `HnswAceParams`.

**Returns**

an instance of `HnswAceParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswAceParams.java:189`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswAceParams.java:16`_
