---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-acceleratedhnswparams
---

# AcceleratedHNSWParams

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class AcceleratedHNSWParams
```

## Public Members

### getWriterThreads

```java
public int getWriterThreads()
```

Get the native cuVS writer threads parameter.

**Returns**

cuVS writer threads parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:154`_

### getGraphThreads

```java
public int getGraphThreads()
```

Get the maximum threads per HNSW graph materialization or serialization operation. The count
includes the calling thread; shared helper capacity may reduce actual concurrency.

**Returns**

HNSW graph processing threads parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:164`_

### getGraphCopyMemoryBudgetBytes

```java
public long getGraphCopyMemoryBudgetBytes()
```

Get the configured temporary host-copy budget for parallel device-graph materialization.

**Returns**

graph-copy memory budget in bytes

### getIntermediateGraphDegree

```java
public int getIntermediateGraphDegree()
```

Get the intermediate graph degree

**Returns**

the graph degree parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:173`_

### getGraphdegree

```java
public int getGraphdegree()
```

Get the graph degree

**Returns**

the graph degree parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:182`_

### getHnswLayers

```java
public int getHnswLayers()
```

Get the number of HNSW layers

**Returns**

the number of HNSW layers

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:191`_

### getMaxConn

```java
public int getMaxConn()
```

Get the max connection parameter

**Returns**

the max connection parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:200`_

### getBeamWidth

```java
public int getBeamWidth()
```

Get the beam width parameter

**Returns**

the beam width parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:209`_

### getCagraGraphBuildAlgo

```java
public CagraGraphBuildAlgo getCagraGraphBuildAlgo()
```

Get the CAGRA graph build algorithm

**Returns**

the CAGRA graph build algorithm

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:218`_

### getCuVSIvfPqParams

```java
public CuVSIvfPqParams getCuVSIvfPqParams()
```

Get the instance of `CuVSIvfPqParams`

**Returns**

the instance of `CuVSIvfPqParams`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:227`_

### getNumMergeWorkers

```java
public int getNumMergeWorkers()
```

Get the number of merge workers set to be used in the fallback mechanism

**Returns**

the number of merge workers

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:236`_

### getMergeExec

```java
public ExecutorService getMergeExec()
```

Get the instance of the `ExecutorService` to be used in the fallback mechanism

**Returns**

the instance of the `ExecutorService`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:245`_

### getStrategy

```java
public Strategy getStrategy()
```

Get the chosen strategy:

When HEURISTIC [Default] is chosen, the CAGRA build algorithm and its indexing parameters are automatically chosen based on the size of the data set
When CUSTOM is chosen, the build algorithm and its parameters (either defaults or overridden values with the use of With* methods) is used internally

**Returns**

get the chosen `Strategy`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:257`_

### getCuvsDistanceType

```java
public CuvsDistanceType getCuvsDistanceType()
```

Get the cuvs distance type

**Returns**

the distance type

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:266`_

### getNNDescentNumIterations

```java
public int getNNDescentNumIterations()
```

get the number of Iterations to run if building with NN_DESCENT

**Returns**

the number of iterations for NN_DESCENT

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:275`_

### getHnswHeuristicType

```java
public HnswHeuristicType getHnswHeuristicType()
```

Get the heuristic cuVS applies when deriving the CAGRA build parameters from maxConn and
beamWidth. Only consulted under the `Strategy#HEURISTIC` strategy.

**Returns**

the `HnswHeuristicType` to hand to cuVS

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:285`_

### withWriterThreads

```java
public Builder withWriterThreads(int writerThreads)
```

Set the number of native cuVS writer threads while building the index.
Valid range - Minimum: \{@value MIN_WRITER_THREADS\}, Maximum: \{@value MAX_WRITER_THREADS\}
Default value - \{@value DEFAULT_WRITER_THREADS\}

**Parameters**

| Name | Description |
| --- | --- |
| `writerThreads` |  |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:353`_

### withGraphThreads

```java
public Builder withGraphThreads(int graphThreads)
```

Set the maximum threads per HNSW graph materialization or serialization operation. The count
includes the calling thread. Valid range - Minimum: \{@value MIN_GRAPH_THREADS\}, Maximum:
\{@value MAX_GRAPH_THREADS\}. Default value - \{@value DEFAULT_GRAPH_THREADS\}.

**Parameters**

| Name | Description |
| --- | --- |
| `graphThreads` | maximum graph-processing threads per operation |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:366`_

### withGraphCopyMemoryBudgetBytes

```java
public Builder withGraphCopyMemoryBudgetBytes(long graphCopyMemoryBudgetBytes)
```

Set the per-operation ceiling for the raw temporary host adjacency copy used by parallel
device-graph materialization. Concurrent copies in the same class loader share reservations;
applications that require one ceiling across codecs should configure the same value for each
codec. This setting does not cap the heap-backed Lucene graph and is not a guarantee of physical
memory availability. A value of `0` preserves the serial fallback without making a temporary
device-to-host graph copy. The default is 42 GiB (`42L << 30` bytes).

**Parameters**

| Name | Description |
| --- | --- |
| `graphCopyMemoryBudgetBytes` | graph-copy memory budget in bytes |

**Returns**

instance of `Builder`

### withIntermediateGraphDegree

```java
public Builder withIntermediateGraphDegree(int intermediateGraphDegree)
```

Set the intermediate graph degree to use while building CAGRA index
Valid range - Minimum: \{@value MIN_INT_GRAPH_DEG\}, Maximum: \{@value MAX_INT_GRAPH_DEG\}
Default value - \{@value DEFAULT_INT_GRAPH_DEGREE\}

**Parameters**

| Name | Description |
| --- | --- |
| `intermediateGraphDegree` |  |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:379`_

### withGraphDegree

```java
public Builder withGraphDegree(int graphDegree)
```

Set the graph degree to use while building CAGRA index
Valid range - Minimum: \{@value MIN_GRAPH_DEG\}, Maximum: \{@value MAX_GRAPH_DEG\}
Default value - \{@value DEFAULT_GRAPH_DEGREE\}

**Parameters**

| Name | Description |
| --- | --- |
| `graphDegree` |  |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:392`_

### withHNSWLayer

```java
public Builder withHNSWLayer(int hnswLayers)
```

Set the number of HNSW layers to construct while building the HNSW index
Valid range - Minimum: \{@value MIN_HNSW_LAYERS\}, Maximum: \{@value MAX_HNSW_LAYERS\}
Default value - \{@value DEFAULT_HNSW_LAYERS\}

**Parameters**

| Name | Description |
| --- | --- |
| `hnswLayers` | the number of HNSW layers |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:405`_

### withMaxConn

```java
public Builder withMaxConn(int maxConn)
```

Set the max connections parameter while building HNSW index with fallback mechanism
Valid range - Minimum: \{@value MIN_MAX_CONN\}, Maximum: \{@value MAX_MAX_CONN\}
Default value - \{@value DEFAULT_MAX_CONN\}

**Parameters**

| Name | Description |
| --- | --- |
| `maxConn` | the max connections parameter |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:418`_

### withBeamWidth

```java
public Builder withBeamWidth(int beamWidth)
```

Set the beam width parameter while building HNSW index with fallback mechanism
Valid range - Minimum: \{@value MIN_BEAM_WIDTH\}, Maximum: \{@value MAX_BEAM_WIDTH\}
Default value - \{@value DEFAULT_BEAM_WIDTH\}

**Parameters**

| Name | Description |
| --- | --- |
| `beamWidth` | the beam width parameter |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:431`_

### withCagraGraphBuildAlgo

```java
public Builder withCagraGraphBuildAlgo(CagraGraphBuildAlgo cagraGraphBuildAlgo)
```

Set the CAGRA graph build algorithm to use
Default value - NN_DESCENT

**Parameters**

| Name | Description |
| --- | --- |
| `cagraGraphBuildAlgo` |  |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:443`_

### withCuVSIvfPqParams

```java
public Builder withCuVSIvfPqParams(CuVSIvfPqParams cuVSIvfPqParams)
```

Set the instance of `CuVSIvfPqParams`

**Parameters**

| Name | Description |
| --- | --- |
| `cuVSIvfPqParams` |  |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:454`_

### withNumMergeWorkers

```java
public Builder withNumMergeWorkers(int numMergeWorkers)
```

Set the number of merge workers to be used with the fallback mechanism
Default value - \{@value DEFAULT_NUM_MERGE_WORKERS\}

**Parameters**

| Name | Description |
| --- | --- |
| `numMergeWorkers` | number of merge workers to set |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:466`_

### withMergeExecutorService

```java
public Builder withMergeExecutorService(ExecutorService mergeExec)
```

Set the merge executor service to be used in the fallback mechanism
Default value an instance with one thread

**Parameters**

| Name | Description |
| --- | --- |
| `mergeExec` | an instance of `ExecutorService` |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:478`_

### withStrategy

```java
public Builder withStrategy(Strategy strategy)
```

Set the chosen strategy:

When HEURISTIC [Default] is chosen, the CAGRA build algorithm and its indexing parameters are automatically chosen based on the size of the data set
When CUSTOM is chosen, the build algorithm and its parameters (either defaults or overridden values with the use of With* methods) is used internally

Valid options - HEURISTIC, CUSTOM
Default value - HEURISTIC

**Parameters**

| Name | Description |
| --- | --- |
| `strategy` | , the strategy to choose |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:495`_

### withCuvsDistanceType

```java
public Builder withCuvsDistanceType(CuvsDistanceType cuvsDistanceType)
```

Set the CuvsDistanceType

**Parameters**

| Name | Description |
| --- | --- |
| `cuvsDistanceType` | the CuvsDistanceType to set |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:506`_

### withNNDescentNumIterations

```java
public Builder withNNDescentNumIterations(int nnDescentNumIterations)
```

Set the number of Iterations to run if building with NN_DESCENT

Valid range - Minimum: \{@value MIN_NN_DESCENT_NUM_ITERATIONS\}, Maximum: \{@value MAX_NN_DESCENT_NUM_ITERATIONS\}
Default value - \{@value DEFAULT_NN_DESCENT_NUM_ITERATIONS\}

**Parameters**

| Name | Description |
| --- | --- |
| `nnDescentNumIterations` | number of merge workers to set |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:520`_

### withHnswHeuristicType

```java
public Builder withHnswHeuristicType(HnswHeuristicType hnswHeuristicType)
```

Set the heuristic cuVS applies when deriving the CAGRA build parameters from maxConn and
beamWidth. Only consulted under the `Strategy#HEURISTIC` strategy.

Default value - SAME_GRAPH_FOOTPRINT, which targets a CAGRA graph of the same on-disk size as
the equivalent HNSW graph (graph degree = 2 * maxConn).

**Parameters**

| Name | Description |
| --- | --- |
| `hnswHeuristicType` | the `HnswHeuristicType` to hand to cuVS |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:535`_

### build

```java
public AcceleratedHNSWParams build()
```

Create an instance of `AcceleratedHNSWParams`

**Returns**

instance of `AcceleratedHNSWParams`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:582`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:17`_
