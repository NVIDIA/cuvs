---
slug: api-reference/java-api-com-nvidia-cuvs-cuvsaceparams
---

# CuVSAceParams

_Java package: `com.nvidia.cuvs`_

```java
public class CuVSAceParams
```

Parameters for ACE (Augmented Core Extraction) graph build algorithm.
ACE enables building indexes for datasets too large to fit in GPU memory by:
1. Partitioning the dataset in core (closest) and augmented (second-closest)
partitions using balanced k-means.
2. Building sub-indexes for each partition independently
3. Concatenating sub-graphs into a final unified index

## Public Members

### getNpartitions

```java
public long getNpartitions()
```

Gets the number of partitions.

**Returns**

the number of partitions

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSAceParams.java:92`_

### getEfConstruction

```java
public long getEfConstruction()
```

Gets the `ef_construction` parameter.

**Returns**

the `ef_construction` parameter

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSAceParams.java:101`_

### getBuildDir

```java
public String getBuildDir()
```

Gets the build directory path.

**Returns**

the build directory path

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSAceParams.java:110`_

### isUseDisk

```java
public boolean isUseDisk()
```

Gets whether disk-based mode is enabled.

**Returns**

true if disk-based mode is enabled

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSAceParams.java:119`_

### getMaxHostMemoryGb

```java
public double getMaxHostMemoryGb()
```

Gets the maximum host memory limit in GiB.

**Returns**

the max host memory limit (0 means use available memory)

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSAceParams.java:128`_

### getMaxGpuMemoryGb

```java
public double getMaxGpuMemoryGb()
```

Gets the maximum GPU memory limit in GiB.

**Returns**

the max GPU memory limit (0 means use available memory)

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSAceParams.java:137`_

### withNpartitions

```java
public Builder withNpartitions(long npartitions)
```

Sets the number of partitions.

**Parameters**

| Name | Description |
| --- | --- |
| `npartitions` | the number of partitions |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSAceParams.java:189`_

### withEfConstruction

```java
public Builder withEfConstruction(long efConstruction)
```

Sets the ef_construction parameter.

**Parameters**

| Name | Description |
| --- | --- |
| `efConstruction` | the ef_construction parameter |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSAceParams.java:200`_

### withBuildDir

```java
public Builder withBuildDir(String buildDir)
```

Sets the build directory path.

**Parameters**

| Name | Description |
| --- | --- |
| `buildDir` | the build directory path |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSAceParams.java:211`_

### withUseDisk

```java
public Builder withUseDisk(boolean useDisk)
```

Sets whether to use disk-based mode.

**Parameters**

| Name | Description |
| --- | --- |
| `useDisk` | whether to use disk-based mode |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSAceParams.java:222`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSAceParams.java:236`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSAceParams.java:250`_

### build

```java
public CuVSAceParams build()
```

Builds an instance of `CuVSAceParams`.

**Returns**

an instance of `CuVSAceParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSAceParams.java:260`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSAceParams.java:17`_
