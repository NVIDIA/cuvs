---
slug: api-reference/java-api-com-nvidia-cuvs-vamanaindexparams
---

# VamanaIndexParams

_Java package: `com.nvidia.cuvs`_

```java
public class VamanaIndexParams
```

Supplemental parameters to build a Vamana index.

The defaults match the native `cuvs::neighbors::vamana::index_params`
defaults.

## Public and Protected Members

### supportedGraphDegrees

```java
public static int[] supportedGraphDegrees()
```

Returns the graph degrees the native Vamana builder supports.

**Returns**

a copy of the supported graph degrees, in ascending order

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:30`_

### L2Expanded

```java
L2Expanded(0), /** * Euclidean, the square root of
```

Squared L2.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:45`_

### L2SqrtExpanded

```java
L2SqrtExpanded(1)
```

Euclidean, the square root of `#L2Expanded`.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:50`_

### EMPTY

```java
EMPTY(0), /** * Start from a CAGRA graph built with NN-descent (kNN degree twice the * graph degree, pruned to the graph degree), with each neighbor list * sorted by distance. The medoid is the vector nearest the dataset mean, * and the insertion passes use full batches from the start. One pass at a * smaller visited size, such as 128 for graph degree 64, can match the * quality of two passes over an empty graph in less time than one. */ NN_DESCENT(1)
```

Start from an empty graph. This is the default.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:69`_

### NN_DESCENT

```java
NN_DESCENT(1)
```

Start from a CAGRA graph built with NN-descent (kNN degree twice the
graph degree, pruned to the graph degree), with each neighbor list
sorted by distance. The medoid is the vector nearest the dataset mean,
and the insertion passes use full batches from the start. One pass at a
smaller visited size, such as 128 for graph degree 64, can match the
quality of two passes over an empty graph in less time than one.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:79`_

### getGraphDegree

```java
public int getGraphDegree()
```

Gets the maximum degree of the output graph, the R parameter in the Vamana
literature.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:129`_

### getVisitedSize

```java
public int getVisitedSize()
```

Gets the maximum number of visited nodes per search, the L parameter in the
Vamana literature.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:137`_

### getVamanaIters

```java
public float getVamanaIters()
```

Gets the number of Vamana vector insertion iterations.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:144`_

### getAlpha

```java
public float getAlpha()
```

Gets the alpha pruning parameter.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:151`_

### getMaxFraction

```java
public float getMaxFraction()
```

Gets the maximum fraction of the dataset inserted per batch.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:158`_

### getBatchBase

```java
public float getBatchBase()
```

Gets the growth rate base for batch sizes.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:165`_

### getQueueSize

```java
public int getQueueSize()
```

Gets the candidate queue size.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:172`_

### getReverseBatchSize

```java
public int getReverseBatchSize()
```

Gets the maximum batch size of reverse edge processing.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:179`_

### getMetric

```java
public CuvsDistanceType getMetric()
```

Gets the distance metric.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:186`_

### getGraphInit

```java
public GraphInitAlgo getGraphInit()
```

Gets how the graph is initialized before the insertion passes.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:193`_

### withGraphDegree

```java
public Builder withGraphDegree(int graphDegree)
```

Sets the maximum degree of the output graph.

**Parameters**

| Name | Description |
| --- | --- |
| `graphDegree` | the graph degree, one of `VamanaIndexParams#supportedGraphDegrees()` |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:247`_

### withVisitedSize

```java
public Builder withVisitedSize(int visitedSize)
```

Sets the maximum number of visited nodes per search.

The native builder requires this to be greater than the graph degree.

**Parameters**

| Name | Description |
| --- | --- |
| `visitedSize` | the visited size |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:260`_

### withVamanaIters

```java
public Builder withVamanaIters(float vamanaIters)
```

Sets the number of Vamana vector insertion iterations.

**Parameters**

| Name | Description |
| --- | --- |
| `vamanaIters` | the iteration count |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:271`_

### withAlpha

```java
public Builder withAlpha(float alpha)
```

Sets the alpha pruning parameter.

**Parameters**

| Name | Description |
| --- | --- |
| `alpha` | the alpha value |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:282`_

### withMaxFraction

```java
public Builder withMaxFraction(float maxFraction)
```

Sets the maximum fraction of the dataset inserted per batch. A larger
batch decreases graph quality but improves build speed.

**Parameters**

| Name | Description |
| --- | --- |
| `maxFraction` | the maximum fraction |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:294`_

### withBatchBase

```java
public Builder withBatchBase(float batchBase)
```

Sets the growth rate base for batch sizes.

**Parameters**

| Name | Description |
| --- | --- |
| `batchBase` | the batch base |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:305`_

### withQueueSize

```java
public Builder withQueueSize(int queueSize)
```

Sets the candidate queue size. The native builder expects a value of the
form `(2^x) - 1`.

**Parameters**

| Name | Description |
| --- | --- |
| `queueSize` | the queue size |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:317`_

### withReverseBatchSize

```java
public Builder withReverseBatchSize(int reverseBatchSize)
```

Sets the maximum batch size of reverse edge processing, which bounds the
memory footprint of that stage.

**Parameters**

| Name | Description |
| --- | --- |
| `reverseBatchSize` | the reverse batch size |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:329`_

### withMetric

```java
public Builder withMetric(CuvsDistanceType metric)
```

Sets the distance metric.

**Parameters**

| Name | Description |
| --- | --- |
| `metric` | the distance metric |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:340`_

### withGraphInit

```java
public Builder withGraphInit(GraphInitAlgo graphInit)
```

Sets how the graph is initialized before the insertion passes. The
default is `GraphInitAlgo#EMPTY`.

**Parameters**

| Name | Description |
| --- | --- |
| `graphInit` | the graph initialization |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:352`_

### build

```java
public VamanaIndexParams build()
```

Builds an instance of `VamanaIndexParams`.

**Returns**

an instance of `VamanaIndexParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:362`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/VamanaIndexParams.java:18`_
