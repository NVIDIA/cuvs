---
slug: api-reference/java-api-com-nvidia-cuvs-cagraindex
---

# CagraIndex

_Java package: `com.nvidia.cuvs`_

```java
public interface CagraIndex extends AutoCloseable
```

`CagraIndex` encapsulates a CAGRA index, along with methods to interact
with it.

CAGRA is a graph-based nearest neighbors algorithm that was built from the
ground up for GPU acceleration. CAGRA demonstrates state-of-the art index
build and query performance for both small and large-batch sized search. Know
more about this algorithm
here

## Public Members

### setDelegate

```java
public final void setDelegate(AutoCloseable delegate, long handleAddress)
```

Internal wiring hook used by the Java wrapper implementation.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:34`_

### isPresent

```java
public final boolean isPresent()
```

Returns true when this view has a native handle.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:42`_

### nativeHandleAddress

```java
public final long nativeHandleAddress()
```

Internal accessor for native handle address.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:49`_

### setDelegate

```java
public final void setDelegate(AutoCloseable delegate)
```

Internal wiring hook used by the Java wrapper implementation.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:84`_

### setDelegate

```java
public final void setDelegate(AutoCloseable delegate, long handleAddress)
```

Internal wiring hook used by the Java wrapper implementation.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:91`_

### isPresent

```java
public final boolean isPresent()
```

Returns true when this handle owns native dataset storage.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:99`_

### nativeHandleAddress

```java
public final long nativeHandleAddress()
```

Internal accessor for native handle address.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:106`_

### close

```java
@Override void close() throws Exception
```

Invokes the native destroy_cagra_index to de-allocate the CAGRA index. Also attempts to close
any dataset whose ownership transferred to this index during construction.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:137`_

### search

```java
SearchResults search(CagraQuery query) throws Throwable
```

Invokes the native search_cagra_index via the Panama API for searching a
CAGRA index.

**Parameters**

| Name | Description |
| --- | --- |
| `query` | an instance of `CagraQuery` holding the query vectors and other parameters |

**Returns**

an instance of `SearchResults` containing the results

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:148`_

### makePaddedDataset

```java
PaddedDataset makePaddedDataset(CuVSMatrix dataset) throws Throwable
```

Create an owning padded dataset by allocating padded storage and copying
`dataset`. Prefer this when the source matrix is not already padded to CAGRA's
required row stride (e.g. unaligned dimensions).

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:155`_

### makePaddedDatasetView

```java
PaddedDatasetView makePaddedDatasetView(CuVSMatrix dataset) throws Throwable
```

Create a caller-owned padded dataset view handle from a matrix that is already
padded to CAGRA's required row stride. For unpadded matrices use
`#makePaddedDataset(CuVSMatrix)`.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:162`_

### makeStandardDatasetView

```java
StandardDatasetView makeStandardDatasetView(CuVSMatrix dataset) throws Throwable
```

Create a caller-owned standard dataset view handle from a matrix.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:165`_

### updateDataset

```java
void updateDataset(PaddedDatasetView datasetView) throws Throwable
```

Update this index with a caller-provided padded device dataset view and leave it
search-ready in padded-device layout. The caller retains ownership of the underlying
padded storage and must keep it alive while this index uses it.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:172`_

### updateDataset

```java
void updateDataset(PaddedDataset dataset) throws Throwable
```

Update this index with a caller-owned padded device dataset. The dataset must remain alive
while this index uses it.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:178`_

### getGraph

```java
CuVSDeviceMatrix getGraph()
```

Returns the CAGRA graph

**Returns**

a `CuVSDeviceMatrix` encapsulating the native int (uint32_t) array used to represent the cagra graph

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:185`_

### getGraphDegree

```java
long getGraphDegree()
```

Returns the degree of the built CAGRA graph (its number of edges per node), which may be
smaller than the requested `graph_degree` when the dataset is small enough that the
build truncated it.

**Returns**

the built graph degree (`graph().extent(1)`)

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:194`_

### size

```java
long size()
```

Returns the number of vectors in this index.

**Returns**

the number of rows of the indexed dataset

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:201`_

### serialize

```java
void serialize(OutputStream outputStream) throws Throwable
```

A method to persist a CAGRA index using an instance of `OutputStream`
for writing index bytes.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes into |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:210`_

### serialize

```java
void serialize(OutputStream outputStream, int bufferLength) throws Throwable
```

A method to persist a CAGRA index using an instance of `OutputStream`
for writing index bytes.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes into |
| `bufferLength` | the length of buffer to use for writing bytes. Default value is 1024 |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:221`_

### serialize

```java
default void serialize(OutputStream outputStream, Path tempFile) throws Throwable
```

A method to persist a CAGRA index using an instance of `OutputStream`
for writing index bytes.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes into |
| `tempFile` | an intermediate `Path` where CAGRA index is written temporarily |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:232`_

### serialize

```java
void serialize(OutputStream outputStream, Path tempFile, int bufferLength) throws Throwable
```

A method to persist a CAGRA index using an instance of `OutputStream`
and path to the intermediate temporary file.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes to |
| `tempFile` | an intermediate `Path` where CAGRA index is written temporarily |
| `bufferLength` | the length of buffer to use for writing bytes. Default value is 1024 |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:247`_

### serializeToHNSW

```java
void serializeToHNSW(OutputStream outputStream) throws Throwable
```

A method to create and persist HNSW index from CAGRA index using an instance
of `OutputStream` and path to the intermediate temporary file.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes to |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:256`_

### serializeToHNSW

```java
void serializeToHNSW(OutputStream outputStream, int bufferLength) throws Throwable
```

A method to create and persist HNSW index from CAGRA index using an instance
of `OutputStream` and path to the intermediate temporary file.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes to |
| `bufferLength` | the length of buffer to use for writing bytes. Default value is 1024 |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:267`_

### serializeToHNSW

```java
default void serializeToHNSW(OutputStream outputStream, Path tempFile) throws Throwable
```

A method to create and persist HNSW index from CAGRA index using an instance
of `OutputStream` and path to the intermediate temporary file.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes to |
| `tempFile` | an intermediate `Path` where CAGRA index is written temporarily |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:278`_

### serializeToHNSW

```java
void serializeToHNSW(OutputStream outputStream, Path tempFile, int bufferLength) throws Throwable
```

A method to create and persist HNSW index from CAGRA index using an instance
of `OutputStream` and path to the intermediate temporary file.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes to |
| `tempFile` | an intermediate `Path` where CAGRA index is written temporarily |
| `bufferLength` | the length of buffer to use for writing bytes. Default value is 1024 |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:293`_

### getCuVSResources

```java
CuVSResources getCuVSResources()
```

Gets an instance of `CuVSResources`

**Returns**

an instance of `CuVSResources`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:300`_

### newBuilder

```java
static Builder newBuilder(CuVSResources cuvsResources)
```

Creates a new Builder with an instance of `CuVSResources`. Pick what the index is created
from with one of the `from*` methods of `Builder`, then call `build()` on the
builder it returns.

**Parameters**

| Name | Description |
| --- | --- |
| `cuvsResources` | an instance of `CuVSResources` |

**Throws**

| Type | Description |
| --- | --- |
| `UnsupportedOperationException` | if cuVS is not available, for example without a GPU or without the native library |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:311`_

### merge

```java
static CagraIndex merge(CagraIndex[] indexes) throws Throwable
```

Merges multiple CAGRA indexes into a single index using default merge parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to merge |

**Returns**

A new merged CAGRA index

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the merge operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:323`_

### merge

```java
static CagraIndex merge(CagraIndex[] indexes, CagraIndexParams mergeParams) throws Throwable
```

Merges multiple CAGRA indexes into a single index with the specified merge parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to merge |
| `mergeParams` | Parameters to control the merge operation, or null to use defaults |

**Returns**

A new merged CAGRA index

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the merge operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:335`_

### merge

```java
static CagraIndex merge(CagraIndex[] indexes, CagraIndexParams mergeParams, BitSet rowFilter) throws Throwable
```

Merges multiple CAGRA indexes into a single index, keeping only the rows selected by
`rowFilter`.

The merge concatenates the input datasets in the order the indexes are given, so bit
`i` of the filter refers to row `i` of that concatenation: bits `0` to
`indexes[0].size() - 1` address the first index, the bits that follow address the second,
and so on. A set bit keeps the row; a clear bit drops it. The rows that survive keep
their relative order and are packed together, so the merged index has one row per set bit.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to merge |
| `mergeParams` | Parameters to control the merge operation, or null to use defaults |
| `rowFilter` | The rows to keep, or null to keep all of them. A BitSet shorter than the total row count is valid: the rows beyond its logical length are treated as clear (dropped). A bit set at a position at or beyond the total row count throws `IllegalArgumentException`. |

**Returns**

A new merged CAGRA index

**Throws**

| Type | Description |
| --- | --- |
| `IllegalArgumentException` | if `rowFilter` has a bit set beyond the last row, or if it is non-null but keeps no rows at all |
| `Throwable` | if an error occurs during the merge operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:359`_

### isPaddedDataset

```java
static boolean isPaddedDataset(CuVSMatrix dataset)
```

Reports whether the rows of `dataset` already sit at the row stride CAGRA requires, which
is the row length in bytes rounded up to a 16 byte boundary.

Use it to pick between the two padded dataset factories: a matrix that is already padded has
to go through `#makePaddedDatasetView(CuVSMatrix)`, because cuVS rejects a request to
copy it into padded storage it already occupies, and one that is not has to go through
`#makePaddedDataset(CuVSMatrix)`.

**Parameters**

| Name | Description |
| --- | --- |
| `dataset` | the matrix to inspect |

**Returns**

true when the rows are already padded the way CAGRA requires

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:387`_

### fromDataset

```java
default FromDatasetBuilder fromDataset(CuVSMatrix dataset)
```

Builds the CAGRA graph from dense vectors.

The index takes ownership of `dataset` when `FromDatasetBuilder#build()`
returns, and closes it when the index is closed. If `build()` throws, the caller still
owns it.

**Parameters**

| Name | Description |
| --- | --- |
| `dataset` | the vectors to index |

**Returns**

a builder for the optional inputs

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:427`_

### fromDataset

```java
default FromDatasetBuilder fromDataset(float[][] vectors)
```

Builds the CAGRA graph from dense vectors held in a Java array.
`FromDatasetBuilder#build()` checks them and copies them into a matrix that the index
owns, so changes made to the array before then reach the index.

**Parameters**

| Name | Description |
| --- | --- |
| `vectors` | the vectors to index, one row per vector, all of the same length |

**Returns**

a builder for the optional inputs

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:439`_

### fromBbq

```java
default FromBbqBuilder fromBbq(BbqQuantizer quantizer)
```

Builds the CAGRA graph from one BBQ encoding of the vectors, used on both sides of every
distance. Only nn-descent graph construction is supported.

The index stores views over the quantizer's matrices rather than copying them, so they
must stay open for as long as the index is in use. Unless a dense dataset is attached with
`FromBbqBuilder#withDenseDataset(CuVSMatrix)`, the index can't be searched until
`CagraIndex#updateDataset(PaddedDatasetView)` or
`CagraIndex#updateDataset(PaddedDataset)` attaches one.

**Parameters**

| Name | Description |
| --- | --- |
| `quantizer` | the encoded vectors |

**Returns**

a builder for the optional inputs

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:456`_

### fromBbq

```java
default FromBbqBuilder fromBbq(BbqQuantizer quantizer, BbqQuantizer other)
```

Builds the CAGRA graph from two BBQ encodings of the same vectors at different precisions.
They can be given in either order: cuVS uses the lower-precision one for the stored side of
every distance and the other one for the query side. The supported pairs of layouts are
`PACKED_1B` with `PACKED_4B`, `TRANSPOSED_2B` or `TRANSPOSED_4B`, and
`TRANSPOSED_2B` with `TRANSPOSED_4B`.

Otherwise this behaves like `#fromBbq(BbqQuantizer)`, and both quantizers must stay
open for as long as the index is in use.

**Parameters**

| Name | Description |
| --- | --- |
| `quantizer` | one encoding of the vectors |
| `other` | the other encoding, with a different layout |

**Returns**

a builder for the optional inputs

**Throws**

| Type | Description |
| --- | --- |
| `IllegalArgumentException` | if both quantizers use the same layout |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:475`_

### fromGraph

```java
default FromGraphBuilder fromGraph( CagraIndexParams.CuvsDistanceType metric, CuVSMatrix graph, CuVSMatrix dataset)
```

Creates an index around a graph built earlier and the dataset it was built from. No graph is
built.

The index takes ownership of `dataset` when `FromGraphBuilder#build()`
returns. It never owns `graph`: a graph in host memory is copied, but one in device
memory is used in place, so it must stay open for as long as the index is in use. The graph
returned by `CagraIndex#getGraph()` is a view into its index's own device memory, so
passing it directly ties the new index to that one, which must then stay open too. Pass a
copy made with `CuVSMatrix#toHost()` to keep the two independent.

**Parameters**

| Name | Description |
| --- | --- |
| `metric` | the distance the graph was built for |
| `graph` | the graph, one row of neighbor indices per vector |
| `dataset` | the vectors the graph was built from, in device memory |

**Returns**

a builder for the optional inputs

**Throws**

| Type | Description |
| --- | --- |
| `IllegalArgumentException` | if `dataset` is not in device memory |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:496`_

### fromSerialized

```java
default FromSerializedBuilder fromSerialized(InputStream inputStream)
```

Loads an index written by `CagraIndex#serialize(OutputStream)`, with its graph and
dataset. `FromSerializedBuilder#build()` reads `inputStream` to its end, but
closing it is left to the caller. The index owns the loaded dataset unless
`FromSerializedBuilder#withOutputDataset(DeserializeDataset)` hands it to the caller.

**Parameters**

| Name | Description |
| --- | --- |
| `inputStream` | the serialized index |

**Returns**

a builder for the optional inputs

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:510`_

### from

```java
@Deprecated(since = "26.12", forRemoval = true) Builder from(InputStream inputStream)
```

**Deprecated.** Use `#fromSerialized(InputStream)`.

Sets an instance of InputStream typically used when index deserialization is
needed. Unlike `#fromSerialized(InputStream)`, `#build()` closes it.

**Parameters**

| Name | Description |
| --- | --- |
| `inputStream` | an instance of `InputStream` |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:522`_

### from

```java
@Deprecated(since = "26.12", forRemoval = true) Builder from(InputStream inputStream, DeserializeDataset outDataset)
```

**Deprecated.** Use `#fromSerialized(InputStream)` and `FromSerializedBuilder#withOutputDataset(DeserializeDataset)`.

Sets an input stream and an empty caller-owned output handle for explicit dataset
deserialization. The concrete output type must match the dataset layout stored in the
serialized index. Keep `outDataset` alive while the built index is in use. Unlike
`#fromSerialized(InputStream)`, `#build()` closes `inputStream`.

**Parameters**

| Name | Description |
| --- | --- |
| `inputStream` | an instance of `InputStream` |
| `outDataset` | an empty `PaddedDataset` or `StandardDataset` |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:537`_

### from

```java
@Deprecated(since = "26.12", forRemoval = true) Builder from(CuVSMatrix graph)
```

**Deprecated.** Use `#fromGraph(CagraIndexParams.CuvsDistanceType, CuVSMatrix, CuVSMatrix)`, which takes the metric directly.

Sets a CAGRA graph instance to re-create an index from a
previously built graph.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:547`_

### withDataset

```java
@Deprecated(since = "26.12", forRemoval = true) Builder withDataset(float[][] vectors)
```

**Deprecated.** Use `#fromDataset(float[][])`.

Sets the dataset vectors for building the `CagraIndex`.

**Parameters**

| Name | Description |
| --- | --- |
| `vectors` | a two-dimensional float array |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:557`_

### withDataset

```java
@Deprecated(since = "26.12", forRemoval = true) Builder withDataset(CuVSMatrix dataset)
```

**Deprecated.** Use `#fromDataset(CuVSMatrix)`. To create an index from a graph or from BBQ quantizers, pass the dataset to `#fromGraph(CagraIndexParams.CuvsDistanceType, CuVSMatrix, CuVSMatrix)` or `FromBbqBuilder#withDenseDataset(CuVSMatrix)`.

Sets the dataset for building the `CagraIndex`.

The caller retains ownership until a build that uses this dataset returns successfully.
The returned index then owns the dataset, and the caller must leave it open until the index
is closed. If the build fails or uses another configured input source, ownership remains
with the caller.

**Parameters**

| Name | Description |
| --- | --- |
| `dataset` | a `CuVSMatrix` object containing the vectors |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:574`_

### withBbqDataset

```java
@Deprecated(since = "26.12", forRemoval = true) Builder withBbqDataset(BbqQuantizer... quantizers)
```

**Deprecated.** Use `#fromBbq(BbqQuantizer)` or `#fromBbq(BbqQuantizer, BbqQuantizer)`.

Builds the graph from one or two encoded BBQ representations. An optional dense dataset
supplied with `#withDataset(CuVSMatrix)` is attached before search; otherwise call
`CagraIndex#updateDataset(PaddedDatasetView)` or
`CagraIndex#updateDataset(PaddedDataset)` before searching.

The index stores views over the quantizer tensors rather than copying them, so they must
stay open for as long as the index is in use. A dense dataset passed to
`#withDataset(CuVSMatrix)` is owned by the index, as it is for a non-BBQ build, and is
closed with it.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:591`_

### withIndexParams

```java
@Deprecated(since = "26.12", forRemoval = true) Builder withIndexParams(CagraIndexParams cagraIndexParameters)
```

**Deprecated.** Use `FromDatasetBuilder#withIndexParams(CagraIndexParams)` or `FromBbqBuilder#withIndexParams(CagraIndexParams)`. An index created from a graph takes its metric as an argument of `#fromGraph(CagraIndexParams.CuvsDistanceType, CuVSMatrix, CuVSMatrix)`.

Registers an instance of configured `CagraIndexParams` with this
Builder.

**Parameters**

| Name | Description |
| --- | --- |
| `cagraIndexParameters` | An instance of CagraIndexParams. |

**Returns**

An instance of this Builder.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:605`_

### build

```java
@Deprecated(since = "26.12", forRemoval = true) CagraIndex build() throws Throwable
```

**Deprecated.** Call `build()` on the builder returned by one of the `from*` methods.

Builds and returns an instance of CagraIndex. With a dataset alone, `#withIndexParams`
is optional, and without it the index is built with the defaults of
`CagraIndexParams.Builder`. With BBQ quantizers it is optional too, and without it the
index is built with those defaults and the quantizers' metric. With a graph it is required,
because the index takes its metric from it. With a stream to load it is not allowed,
because the loaded index has its own.

**Returns**

an instance of CagraIndex

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:620`_

### withIndexParams

```java
FromDatasetBuilder withIndexParams(CagraIndexParams indexParams)
```

Sets the build parameters. Without this call the index is built with the defaults of
`CagraIndexParams.Builder`.

**Parameters**

| Name | Description |
| --- | --- |
| `indexParams` | the build parameters |

**Returns**

this builder

**Throws**

| Type | Description |
| --- | --- |
| `IllegalStateException` | if called more than once |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:643`_

### build

```java
CagraIndex build() throws Throwable
```

Builds the index. A builder creates one index, so this can be called only once,
even if it fails. To try again, start a new builder with
`CagraIndex#newBuilder(CuVSResources)`.

**Returns**

the new index

**Throws**

| Type | Description |
| --- | --- |
| `IllegalStateException` | if `build()` was already called, whether or not it succeeded |
| `IllegalArgumentException` | if the array given to `Builder#fromDataset(float[][])` is empty or its rows differ in length |
| `Throwable` | if the build fails |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:657`_

### withDenseDataset

```java
FromBbqBuilder withDenseDataset(CuVSMatrix dataset)
```

Attaches full-precision vectors once the graph is built, so that the index can be searched
straight away. The index takes ownership of `dataset` when `#build()` returns,
and closes it when the index is closed. If `build()` throws, the caller still owns it.

**Parameters**

| Name | Description |
| --- | --- |
| `dataset` | the full-precision vectors, in the same order as the encoded ones |

**Returns**

this builder

**Throws**

| Type | Description |
| --- | --- |
| `IllegalStateException` | if called more than once |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:675`_

### withIndexParams

```java
FromBbqBuilder withIndexParams(CagraIndexParams indexParams)
```

Sets the build parameters. Their graph build algorithm must be
`CagraIndexParams.CagraGraphBuildAlgo#NN_DESCENT`, or
`CagraIndexParams.CagraGraphBuildAlgo#AUTO_SELECT`, which picks nn-descent, and their
metric must be the one the quantizers were encoded for. Without this call the index is built
with the defaults of `CagraIndexParams.Builder` and the quantizers' metric.

**Parameters**

| Name | Description |
| --- | --- |
| `indexParams` | the build parameters |

**Returns**

this builder

**Throws**

| Type | Description |
| --- | --- |
| `IllegalArgumentException` | if the parameters ask for another graph build algorithm, or for a metric other than the quantizers' |
| `IllegalStateException` | if called more than once |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:690`_

### build

```java
CagraIndex build() throws Throwable
```

Builds the index. A builder creates one index, so this can be called only once,
even if it fails. To try again, start a new builder with
`CagraIndex#newBuilder(CuVSResources)`.

**Returns**

the new index

**Throws**

| Type | Description |
| --- | --- |
| `IllegalStateException` | if `build()` was already called, whether or not it succeeded |
| `Throwable` | if the build fails |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:702`_

### build

```java
CagraIndex build() throws Throwable
```

Creates the index. A builder creates one index, so this can be called only once,
even if it fails. To try again, start a new builder with
`CagraIndex#newBuilder(CuVSResources)`.

**Returns**

the new index

**Throws**

| Type | Description |
| --- | --- |
| `IllegalStateException` | if `build()` was already called, whether or not it succeeded |
| `Throwable` | if the index can't be created |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:721`_

### withOutputDataset

```java
FromSerializedBuilder withOutputDataset(DeserializeDataset outDataset)
```

Hands the loaded dataset to the caller instead of the index. `outDataset` must be
empty, and its type must match the layout of the dataset stored in the serialized index: a
`PaddedDataset` or a `StandardDataset`. The index uses the dataset in place, so
keep `outDataset` open for as long as the index is in use.

**Parameters**

| Name | Description |
| --- | --- |
| `outDataset` | an empty handle that receives the dataset |

**Returns**

this builder

**Throws**

| Type | Description |
| --- | --- |
| `IllegalStateException` | if called more than once |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:739`_

### build

```java
CagraIndex build() throws Throwable
```

Loads the index. A builder creates one index, so this can be called only once,
even if it fails. To try again, start a new builder with
`CagraIndex#newBuilder(CuVSResources)`.

**Returns**

the loaded index

**Throws**

| Type | Description |
| --- | --- |
| `IllegalStateException` | if `build()` was already called, whether or not it succeeded |
| `Throwable` | if the index can't be loaded |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:751`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:26`_
