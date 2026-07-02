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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:38`_

### CPU

```java
CPU(1), /** * Full hierarchy is built using the GPU */ GPU(2)
```

Full hierarchy is built using the CPU

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:43`_

### GPU

```java
GPU(2)
```

Full hierarchy is built using the GPU

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:48`_

### getHierarchy

```java
public CuvsHnswHierarchy getHierarchy()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:98`_

### getEfConstruction

```java
public int getEfConstruction()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:106`_

### getNumThreads

```java
public int getNumThreads()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:114`_

### getVectorDimension

```java
public int getVectorDimension()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:122`_

### getM

```java
public long getM()
```

Gets the HNSW M parameter: number of bi-directional links per node
(used when building with ACE). graph_degree = m * 2,
intermediate_graph_degree = m * 3.

**Returns**

the M parameter

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:133`_

### getMetric

```java
public CuvsDistanceType getMetric()
```

Gets the distance metric type.

**Returns**

the metric type

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:142`_

### getAceParams

```java
public HnswAceParams getAceParams()
```

Gets the ACE parameters for building HNSW index using ACE algorithm.

**Returns**

the ACE parameters, or null if not set

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:151`_

### Builder

```java
public Builder()
```

Constructs this Builder with an instance of Arena.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:190`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:202`_

### withEfConstruction

```java
public Builder withEfConstruction(int efConstruction)
```

Sets the size of the candidate list during hierarchy construction when
hierarchy is `CPU`.

**Parameters**

| Name | Description |
| --- | --- |
| `efConstruction` | the size of the candidate list during hierarchy construction when hierarchy is `CPU` |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:215`_

### withNumThreads

```java
public Builder withNumThreads(int numThreads)
```

Sets the number of host threads to use to construct hierarchy when hierarchy
is `CPU`.

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
(used when building with ACE). graph_degree = m * 2,
intermediate_graph_degree = m * 3.

**Parameters**

| Name | Description |
| --- | --- |
| `m` | the M parameter |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:251`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:262`_

### withAceParams

```java
public Builder withAceParams(HnswAceParams aceParams)
```

Sets the ACE parameters for building HNSW index using ACE algorithm.

**Parameters**

| Name | Description |
| --- | --- |
| `aceParams` | the ACE parameters |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:273`_

### build

```java
public HnswIndexParams build()
```

Builds an instance of `HnswIndexParams`.

**Returns**

an instance of `HnswIndexParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:283`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswIndexParams.java:12`_
