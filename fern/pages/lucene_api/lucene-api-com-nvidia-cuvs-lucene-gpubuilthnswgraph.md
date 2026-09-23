---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-gpubuilthnswgraph
---

# GPUBuiltHnswGraph

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class GPUBuiltHnswGraph extends HnswGraph
```

This class holds the in-memory representation of the HNSW graph

## Public Members

### GPUBuiltHnswGraph

```java
public GPUBuiltHnswGraph( int size, int dimensions, List<int[]> layerNodes, List<CuVSMatrix> layerAdjacencies)
```

Multi-layer constructor that supports arbitrary number of layers.

**Parameters**

| Name | Description |
| --- | --- |
| `size` | the size of the dataset |
| `dimensions` | the vector dimension |
| `layerNodes` | the nodes on the layer |
| `layerAdjacencies` | adjacency list |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:55`_

### GPUBuiltHnswGraph

```java
public GPUBuiltHnswGraph( int size, int dimensions, List<int[]> layerNodes, List<CuVSMatrix> layerAdjacencies, int numThreads) throws IOException
```

Builds a graph while materializing adjacency rows with the requested number of threads.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:61`_

### fitsParallelGraphCopyBudget

```java
static boolean fitsParallelGraphCopyBudget(long rows, long columns)
```

Returns whether an INT32 adjacency can be copied without exceeding the native-host budget.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:169`_

### getNodesOnLevel

```java
public NodesIterator getNodesOnLevel(int level)
```

Get all nodes on a given level as node 0th ordinals.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:253`_

### getNeighbors

```java
public NeighborArray getNeighbors(int level, int node)
```

Get the neighbors for the node and the level it resides.

**Parameters**

| Name | Description |
| --- | --- |
| `level` | the level |
| `node` | the node |

**Returns**

an instance of NeighborArray

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:271`_

### seek

```java
@Override public void seek(int level, int target)
```

Move the pointer to exactly the given level's target.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:296`_

### nextNeighbor

```java
@Override public int nextNeighbor()
```

Iterates over the neighbor list.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:306`_

### entryNode

```java
@Override public int entryNode()
```

Returns graph's entry point on the top level.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:337`_

### maxConn

```java
@Override public int maxConn()
```

returns M, the maximum number of connections for a node.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:356`_

### neighborCount

```java
@Override public int neighborCount()
```

Returns the neighbor count.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:371`_

### size

```java
public int size()
```

Returns the number of nodes in the graph.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:446`_

### numLevels

```java
public int numLevels()
```

Returns the number of levels in the HNSW graph.

**Returns**

the number of levels

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:455`_

### dimensions

```java
public int dimensions()
```

Gets the vector dimension.

**Returns**

the vector dimension

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:464`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:29`_
