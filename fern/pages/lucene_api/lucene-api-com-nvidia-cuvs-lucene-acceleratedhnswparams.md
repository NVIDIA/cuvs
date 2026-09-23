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

Get the cuVS writer threads parameter

**Returns**

cuVS writer threads parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:146`_

### getIntermediateGraphDegree

```java
public int getIntermediateGraphDegree()
```

Get the intermediate graph degree

**Returns**

the graph degree parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:155`_

### getGraphdegree

```java
public int getGraphdegree()
```

Get the graph degree

**Returns**

the graph degree parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:164`_

### getHnswLayers

```java
public int getHnswLayers()
```

Get the number of HNSW layers

**Returns**

the number of HNSW layers

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:173`_

### getMaxConn

```java
public int getMaxConn()
```

Get the max connection parameter

**Returns**

the max connection parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:182`_

### getBeamWidth

```java
public int getBeamWidth()
```

Get the beam width parameter

**Returns**

the beam width parameter

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:191`_

### getCagraGraphBuildAlgo

```java
public CagraGraphBuildAlgo getCagraGraphBuildAlgo()
```

Get the CAGRA graph build algorithm

**Returns**

the CAGRA graph build algorithm

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:200`_

### getCuVSIvfPqParams

```java
public CuVSIvfPqParams getCuVSIvfPqParams()
```

Get the instance of `CuVSIvfPqParams`

**Returns**

the instance of `CuVSIvfPqParams`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:209`_

### getNumMergeWorkers

```java
public int getNumMergeWorkers()
```

Get the number of merge workers set to be used in the fallback mechanism

**Returns**

the number of merge workers

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:218`_

### getMergeExec

```java
public ExecutorService getMergeExec()
```

Get the instance of the `ExecutorService` to be used in the fallback mechanism

**Returns**

the instance of the `ExecutorService`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:227`_

### getStrategy

```java
public Strategy getStrategy()
```

Get the chosen strategy:

When HEURISTIC [Default] is chosen, the CAGRA build algorithm and its indexing parameters are automatically chosen based on the size of the data set
When CUSTOM is chosen, the build algorithm and its parameters (either defaults or overridden values with the use of With* methods) is used internally

**Returns**

get the chosen `Strategy`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:239`_

### getCuvsDistanceType

```java
public CuvsDistanceType getCuvsDistanceType()
```

Get the cuvs distance type

**Returns**

the distance type

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:248`_

### getNNDescentNumIterations

```java
public int getNNDescentNumIterations()
```

get the number of Iterations to run if building with NN_DESCENT

**Returns**

the number of iterations for NN_DESCENT

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:257`_

### getHnswHeuristicType

```java
public HnswHeuristicType getHnswHeuristicType()
```

Get the heuristic cuVS applies when deriving the CAGRA build parameters from maxConn and
beamWidth. Only consulted under the `Strategy#HEURISTIC` strategy.

**Returns**

the `HnswHeuristicType` to hand to cuVS

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:267`_

### withWriterThreads

```java
public Builder withWriterThreads(int writerThreads)
```

Set the number of cuVS writer threads while building the index
Valid range - Minimum: \{@value MIN_WRITER_THREADS\}, Maximum: \{@value MAX_WRITER_THREADS\}
Default value - \{@value DEFAULT_WRITER_THREADS\}

**Parameters**

| Name | Description |
| --- | --- |
| `writerThreads` |  |

**Returns**

instance of `Builder`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:332`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:345`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:358`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:371`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:384`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:397`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:409`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:420`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:432`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:444`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:461`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:472`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:486`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:501`_

### build

```java
public AcceleratedHNSWParams build()
```

Create an instance of `AcceleratedHNSWParams`

**Returns**

instance of `AcceleratedHNSWParams`

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:546`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWParams.java:17`_
