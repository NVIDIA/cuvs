---
slug: api-reference/lucene-api-com-nvidia-cuvs-lucene-acceleratedhnswutils
---

# AcceleratedHNSWUtils

_Java package: `com.nvidia.cuvs.lucene`_

```java
public class AcceleratedHNSWUtils
```

## Public Members

### createSingleVectorHnswGraph

```java
public static GPUBuiltHnswGraph createSingleVectorHnswGraph(int size, int dimensions) throws Throwable
```

Creates a dummy HNSW graph for a single vector.
The graph will have 1 level with 1 node and no neighbors.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:55`_

### createMultiLayerHnswGraph

```java
public static GPUBuiltHnswGraph createMultiLayerHnswGraph( FieldInfo fieldInfo, int size, int dimensions, CuVSMatrix adjacencyListMatrix, List<?> vectors, int hnswLayers, CagraIndexParams params, QuantizationType quantization) throws Throwable
```

Creates up to `hnswLayers` total layers. Layer 0 uses the full CAGRA graph. Each upper
layer samples `max(2, floor(previousLayerSize / M))` nodes and is built with graph degree
at most `M`. The value `M` is the ceiling of half the layer-0 graph degree.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:73`_

### createMultiLayerHnswGraph

```java
static GPUBuiltHnswGraph createMultiLayerHnswGraph( int dimensions, CuVSMatrix adjacencyListMatrix, CuVSMatrix vectorDataset, int hnswLayers, CagraIndexParams params, QuantizationType quantization, int maxConn) throws Throwable
```

Creates a multi-layer HNSW graph from a native matrix without copying the complete dataset to
the Java heap. The list view copies only rows selected for an upper layer.

The upper layers are built with graph degree at most maxConn, the most neighbors HNSW holds
per node above layer 0. Rows wider than that on an upper layer are read fine, but a CPU merge
copies them into arrays of maxConn + 1 slots and fails with "No growth is allowed".

**Parameters**

| Name | Description |
| --- | --- |
| `maxConn` | the HNSW maxConn the segment is written with |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:230`_

### writeGraph

```java
public static int[][] writeGraph(GPUBuiltHnswGraph graph, IndexOutput vectorIndex) throws IOException
```

Returns a 2D array of offsets (information written while writing the meta info)

**Parameters**

| Name | Description |
| --- | --- |
| `graph` | instance of GPUBuiltHnswGraph |
| `vectorIndex` | instance of IndexOutput |

**Returns**

a 2D array of offsets

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | I/O Exceptions |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:360`_

### writeMeta

```java
@Deprecated public static void writeMeta( IndexOutput vectorIndex, IndexOutput meta, FieldInfo field, long vectorIndexOffset, long vectorIndexLength, int count, HnswGraph graph, int[][] graphLevelNodeOffsets) throws IOException
```

Writes the meta information for the index. Deprecated: it records an M that can be smaller
than the maxConn of a CPU writer merging the segment, which makes that merge fail on Lucene
10.4 and later. Use the overload that also takes maxConn.

**Parameters**

| Name | Description |
| --- | --- |
| `vectorIndex` | instance of IndexOutput |
| `meta` | instance of IndexOutput |
| `field` | instance of FieldInfo |
| `vectorIndexOffset` | vector index offset |
| `vectorIndexLength` | vector index length |
| `count` | the count of vectors |
| `graph` | instance of HnswGraph |
| `graphLevelNodeOffsets` | graph level node offsets |

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | I/O Exceptions |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:428`_

### writeMeta

```java
public static void writeMeta( IndexOutput vectorIndex, IndexOutput meta, FieldInfo field, long vectorIndexOffset, long vectorIndexLength, int count, HnswGraph graph, int[][] graphLevelNodeOffsets, int maxConn) throws IOException
```

Writes the meta information for the index.

**Parameters**

| Name | Description |
| --- | --- |
| `vectorIndex` | instance of IndexOutput |
| `meta` | instance of IndexOutput |
| `field` | instance of FieldInfo |
| `vectorIndexOffset` | vector index offset |
| `vectorIndexLength` | vector index length |
| `count` | the count of vectors |
| `graph` | instance of HnswGraph |
| `graphLevelNodeOffsets` | graph level node offsets |
| `maxConn` | the configured maxConn, the smallest M to record |

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | I/O Exceptions |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:465`_

### printInfoStream

```java
public static void printInfoStream(InfoStream infoStream, String component, String msg)
```

A utility method to print info/debugging messages using InfoStream.

**Parameters**

| Name | Description |
| --- | --- |
| `msg` | the debugging message to print |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:552`_

### writeEmpty

```java
@Deprecated public static void writeEmpty(FieldInfo fieldInfo, IndexOutput op) throws IOException
```

Writes an empty meta information for the field. Deprecated: it records M = 0; use the
overload that also takes maxConn.

**Parameters**

| Name | Description |
| --- | --- |
| `fieldInfo` | instance of FieldInfo |

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | I/O Exceptions |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:566`_

### writeEmpty

```java
public static void writeEmpty(FieldInfo fieldInfo, IndexOutput op, int maxConn) throws IOException
```

Writes an empty meta information for the field.

**Parameters**

| Name | Description |
| --- | --- |
| `fieldInfo` | instance of FieldInfo |
| `maxConn` | the configured maxConn, recorded as the field's M |

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | I/O Exceptions |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:578`_

### quantizeFloatVectorsToBinary

```java
public static List<byte[]> quantizeFloatVectorsToBinary(List<float[]> floatVectors)
```

Quantizes FLOAT32 vectors to binary (1 bit per dimension, packed into bytes).
Binary quantization: each dimension is compared to a centroid (mean of all values for that dimension).
If value &gt; centroid, bit = 1, else bit = 0.
Bits are packed: 8 dimensions per byte.

**Parameters**

| Name | Description |
| --- | --- |
| `floatVectors` | A list of float vectors |

**Returns**

A list of byte binary representation for the input vectors

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:592`_

### quantizeFloatVectorsToScalar

```java
public static List<byte[]> quantizeFloatVectorsToScalar(List<float[]> floatVectors)
```

Scalar quantization.

**Parameters**

| Name | Description |
| --- | --- |
| `floatVectors` | A list of float vectors |

**Returns**

A list of byte scalar representation for the input vectors

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:634`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:32`_
