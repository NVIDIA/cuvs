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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:44`_

### getNodesOnLevel

```java
public NodesIterator getNodesOnLevel(int level)
```

Get all nodes on a given level as node 0th ordinals.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:161`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:179`_

### seek

```java
@Override public void seek(int level, int target)
```

Move the pointer to exactly the given level's target.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:206`_

### nextNeighbor

```java
@Override public int nextNeighbor()
```

Iterates over the neighbor list.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:216`_

### entryNode

```java
@Override public int entryNode()
```

Returns graph's entry point on the top level.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:247`_

### maxConn

```java
@Override public int maxConn()
```

returns M, the maximum number of connections for a node.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:266`_

### neighborCount

```java
@Override public int neighborCount()
```

Returns the neighbor count.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:281`_

### size

```java
public int size()
```

Returns the number of nodes in the graph.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:356`_

### numLevels

```java
public int numLevels()
```

Returns the number of levels in the HNSW graph.

**Returns**

the number of levels

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:365`_

### dimensions

```java
public int dimensions()
```

Gets the vector dimension.

**Returns**

the vector dimension

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:374`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/GPUBuiltHnswGraph.java:23`_
