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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:61`_

### createMultiLayerHnswGraph

```java
public static GPUBuiltHnswGraph createMultiLayerHnswGraph( FieldInfo fieldInfo, int size, int dimensions, CuVSMatrix adjacencyListMatrix, List<?> vectors, int hnswLayers, CagraIndexParams params, QuantizationType quantization) throws Throwable
```

Creates up to `hnswLayers` total layers. Layer 0 uses the full CAGRA graph. Each upper
layer samples `max(2, floor(previousLayerSize / M))` nodes, where `M` is \{@code
ceil(layer-0 graph degree / 2)\}.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:86`_

### createMultiLayerHnswGraph

```java
static GPUBuiltHnswGraph createMultiLayerHnswGraph( FieldInfo fieldInfo, int dimensions, CuVSMatrix adjacencyListMatrix, CuVSMatrix vectorDataset, int hnswLayers, CagraIndexParams params, QuantizationType quantization) throws Throwable
```

Creates a multi-layer HNSW graph from a native matrix without copying the complete dataset to
the Java heap. The list view copies only rows selected for an upper layer.

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:203`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:337`_

### writeMeta

```java
public static void writeMeta( IndexOutput vectorIndex, IndexOutput meta, FieldInfo field, long vectorIndexOffset, long vectorIndexLength, int count, HnswGraph graph, int[][] graphLevelNodeOffsets) throws IOException
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

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | I/O Exceptions |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:501`_

### printInfoStream

```java
public static void printInfoStream(InfoStream infoStream, String component, String msg)
```

A utility method to print info/debugging messages using InfoStream.

**Parameters**

| Name | Description |
| --- | --- |
| `msg` | the debugging message to print |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:583`_

### writeEmpty

```java
public static void writeEmpty(FieldInfo fieldInfo, IndexOutput op) throws IOException
```

Writes an empty meta information for the field.

**Parameters**

| Name | Description |
| --- | --- |
| `fieldInfo` | instance of FieldInfo |

**Throws**

| Type | Description |
| --- | --- |
| `IOException` | I/O Exceptions |

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:595`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:608`_

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

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:650`_

_Source: `java/cuvs-lucene/src/main/java/com/nvidia/cuvs/lucene/AcceleratedHNSWUtils.java:38`_
