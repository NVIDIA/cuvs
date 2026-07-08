---
slug: api-reference/java-api-com-nvidia-cuvs-hnswindexparams
---

# HnswIndexParams

_Java package: `com.nvidia.cuvs`_

```java
public class HnswIndexParams
```

Supplemental parameters to build HNSW index.

## Public Members

### NONE

```java
NONE(0), /** * Full hierarchy is built using the CPU */ CPU(1), /** * Full hierarchy is built using the GPU */ GPU(2)
```

Flat hierarchy, search is base-layer only

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:40`_

### CPU

```java
CPU(1), /** * Full hierarchy is built using the GPU */ GPU(2)
```

Full hierarchy is built using the CPU

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:45`_

### GPU

```java
GPU(2)
```

Full hierarchy is built using the GPU

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:50`_

### getHierarchy

```java
public CuvsHnswHierarchy getHierarchy()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:100`_

### getEfConstruction

```java
public int getEfConstruction()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:108`_

### getNumThreads

```java
public int getNumThreads()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:116`_

### getVectorDimension

```java
public int getVectorDimension()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:124`_

### getM

```java
public long getM()
```

Gets the HNSW M parameter: number of bi-directional links per node
used to derive the internal GPU graph degree.

**Returns**

the M parameter

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:134`_

### getMetric

```java
public CuvsDistanceType getMetric()
```

Gets the distance metric type.

**Returns**

the metric type

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:143`_

### getAceParams

```java
public HnswAceParams getAceParams()
```

Gets optional ACE parameters for partitioned or disk-backed GPU graph building.

**Returns**

the ACE parameters, or null if not set

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:152`_

### Builder

```java
public Builder()
```

Constructs this Builder with an instance of Arena.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:191`_

### withHierarchy

```java
public Builder withHierarchy(CuvsHnswHierarchy hierarchy)
```

Sets the hierarchy for HNSW index when converting from CAGRA index.

NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only
index.

**Parameters**

| Name | Description |
| --- | --- |
| `hierarchy` | the hierarchy for HNSW index when converting from CAGRA index |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:203`_

### withEfConstruction

```java
public Builder withEfConstruction(int efConstruction)
```

Sets the size of the candidate list during index construction.

**Parameters**

| Name | Description |
| --- | --- |
| `efConstruction` | the size of the candidate list during index construction |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:214`_

### withNumThreads

```java
public Builder withNumThreads(int numThreads)
```

Sets the number of host threads used during construction when hierarchy is
`CPU` or `GPU`. When the value is 0, the number of threads is automatically
determined to the maximum number of threads available. The default is 2.

**Parameters**

| Name | Description |
| --- | --- |
| `numThreads` | the number of threads |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:227`_

### withVectorDimension

```java
public Builder withVectorDimension(int vectorDimension)
```

Sets the vector dimension

**Parameters**

| Name | Description |
| --- | --- |
| `vectorDimension` | the vector dimension |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:238`_

### withM

```java
public Builder withM(long m)
```

Sets the HNSW M parameter: number of bi-directional links per node
used to derive the internal GPU graph degree.

**Parameters**

| Name | Description |
| --- | --- |
| `m` | the M parameter |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:250`_

### withMetric

```java
public Builder withMetric(CuvsDistanceType metric)
```

Sets the distance metric type.

**Parameters**

| Name | Description |
| --- | --- |
| `metric` | the metric type |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:261`_

### withAceParams

```java
public Builder withAceParams(HnswAceParams aceParams)
```

Sets optional ACE parameters for partitioned or disk-backed GPU graph building.

**Parameters**

| Name | Description |
| --- | --- |
| `aceParams` | the ACE parameters |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:272`_

### build

```java
public HnswIndexParams build()
```

Builds an instance of `HnswIndexParams`.

**Returns**

an instance of `HnswIndexParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:282`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:12`_
