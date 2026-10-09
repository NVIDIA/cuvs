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

Creates a new Builder with an instance of `CuVSResources`.

**Parameters**

| Name | Description |
| --- | --- |
| `cuvsResources` | an instance of `CuVSResources` |

**Throws**

| Type | Description |
| --- | --- |
| `UnsupportedOperationException` | if the provider does not cuvs |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:308`_

### merge

```java
static CagraIndex merge(CagraIndex[] indexes, PaddedDataset mergedDataset, long[] offsets) throws Throwable
```

Merges multiple CAGRA indexes into a single index using default merge parameters.

The caller is responsible for concatenating every input index's dataset (in \{@code
indexes\} order) into a single caller-owned padded dataset before calling this, and for
providing `offsets`: entry `i` is the row at which `indexes[i]`'s rows
start in `mergedDataset`, and the last entry (`offsets[indexes.length]`) must
equal `mergedDataset`'s total row count. For example, with no filtering, \{@code
offsets\} is simply the cumulative row counts of `indexes` in order. This mirrors the
caller-owned-buffer contract used by `#updateDataset(PaddedDataset)`. Keep \{@code
mergedDataset\} alive for as long as the returned index remains in use.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to merge |
| `mergedDataset` | Caller-owned padded dataset holding the concatenation of every input index's rows, in `indexes` order |
| `offsets` | Per-index starting row within `mergedDataset`. Array of \{@code indexes.length + 1\} entries; the last entry must equal `mergedDataset`'s row count |

**Returns**

A new merged CAGRA index

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the merge operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:334`_

### merge

```java
static CagraIndex merge( CagraIndex[] indexes, PaddedDataset mergedDataset, long[] offsets, CagraIndexParams mergeParams) throws Throwable
```

Merges multiple CAGRA indexes into a single index with the specified merge parameters.

See PaddedDataset, long[]) for the `mergedDataset`/
`offsets` contract.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to merge |
| `mergedDataset` | Caller-owned padded dataset holding the concatenation of every input index's rows, in `indexes` order |
| `offsets` | Per-index starting row within `mergedDataset`. Array of \{@code indexes.length + 1\} entries; the last entry must equal `mergedDataset`'s row count |
| `mergeParams` | Parameters to control the merge operation, or null to use defaults |

**Returns**

A new merged CAGRA index

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the merge operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:355`_

### merge

```java
static CagraIndex merge( CagraIndex[] indexes, PaddedDataset mergedDataset, long[] offsets, BitSet filter, CagraIndexParams mergeParams) throws Throwable
```

Merges multiple CAGRA indexes into a single index with the specified merge parameters, from a
`mergedDataset` that was already filtered by `filter` (e.g. via
BitSet)).

See PaddedDataset, long[]) for the `mergedDataset`/
`offsets` contract. When `filter` is non-null, `offsets` must be the ones
returned by BitSet) called with the same \{@code
filter\}, not the plain cumulative row counts.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to merge |
| `mergedDataset` | Caller-owned padded dataset holding the concatenation of every input index's surviving rows, in `indexes` order |
| `offsets` | Per-index starting row within `mergedDataset`. Array of \{@code indexes.length + 1\} entries; the last entry must equal `mergedDataset`'s row count |
| `filter` | Bitset selecting which rows (over the concatenation of every index's rows, in `indexes` order) survive into `mergedDataset`; a set bit keeps the row. Must be the same filter used to build `mergedDataset`. Pass null for an unfiltered merge |
| `mergeParams` | Parameters to control the merge operation, or null to use defaults |

**Returns**

A new merged CAGRA index

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the merge operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:388`_

### merge

```java
static CagraIndex merge(CagraIndex[] indexes, PaddedDatasetView mergedDataset, long[] offsets) throws Throwable
```

Merges multiple CAGRA indexes into a single index using default merge parameters, from a
caller-owned padded dataset view over a buffer that is already padded to CAGRA's required
row stride.

See PaddedDataset, long[]) for the `mergedDataset`/
`offsets` contract.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to merge |
| `mergedDataset` | Caller-owned padded dataset view holding the concatenation of every input index's rows, in `indexes` order |
| `offsets` | Per-index starting row within `mergedDataset`. Array of \{@code indexes.length + 1\} entries; the last entry must equal `mergedDataset`'s row count |

**Returns**

A new merged CAGRA index

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the merge operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:422`_

### merge

```java
static CagraIndex merge( CagraIndex[] indexes, PaddedDatasetView mergedDataset, long[] offsets, CagraIndexParams mergeParams) throws Throwable
```

Merges multiple CAGRA indexes into a single index with the specified merge parameters, from a
caller-owned padded dataset view over a buffer that is already padded to CAGRA's required
row stride.

See PaddedDataset, long[]) for the `mergedDataset`/
`offsets` contract.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to merge |
| `mergedDataset` | Caller-owned padded dataset view holding the concatenation of every input index's rows, in `indexes` order |
| `offsets` | Per-index starting row within `mergedDataset`. Array of \{@code indexes.length + 1\} entries; the last entry must equal `mergedDataset`'s row count |
| `mergeParams` | Parameters to control the merge operation, or null to use defaults |

**Returns**

A new merged CAGRA index

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the merge operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:445`_

### merge

```java
static CagraIndex merge( CagraIndex[] indexes, PaddedDatasetView mergedDataset, long[] offsets, BitSet filter, CagraIndexParams mergeParams) throws Throwable
```

Merges multiple CAGRA indexes into a single index with the specified merge parameters, from a
caller-owned padded dataset view over a buffer that is already padded to CAGRA's required row
stride and that was already filtered by `filter` (e.g. via
BitSet)).

See PaddedDataset, long[], BitSet, CagraIndexParams) for the
`filter`/`offsets` contract.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to merge |
| `mergedDataset` | Caller-owned padded dataset view holding the concatenation of every input index's surviving rows, in `indexes` order |
| `offsets` | Per-index starting row within `mergedDataset`. Array of \{@code indexes.length + 1\} entries; the last entry must equal `mergedDataset`'s row count |
| `filter` | Bitset selecting which rows (over the concatenation of every index's rows, in `indexes` order) survive into `mergedDataset`; a set bit keeps the row. Must be the same filter used to build `mergedDataset`. Pass null for an unfiltered merge |
| `mergeParams` | Parameters to control the merge operation, or null to use defaults |

**Returns**

A new merged CAGRA index

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the merge operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:477`_

### mergedDatasetOffsets

```java
static long[] mergedDatasetOffsets(CagraIndex[] indexes, BitSet filter) throws Throwable
```

Computes per-index write offsets for a bitset-filtered merged dataset buffer.

For an unfiltered merge the offsets are simply the cumulative row counts of \{@code
indexes\} in order, and this method is not needed. For a bitset-filtered merge, the number of
surviving rows per index cannot be derived any other way; call this before
BitSet) (or before building a
filtered `mergedDataset` by hand) to get the matching offsets.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes that will be merged |
| `filter` | Bitset selecting which rows (over the concatenation of every index's rows, in `indexes` order) survive; a set bit keeps the row |

**Returns**

Array of `indexes.length + 1` entries; entry `i` is the row at which `indexes[i]`'s surviving rows start in the merged buffer, and the last entry is the merged buffer's total row count

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:511`_

### concatenateDatasets

```java
static PaddedDataset concatenateDatasets(CagraIndex[] indexes) throws Throwable
```

Concatenates every input index's dataset (unfiltered, in `indexes` order) into a
freshly allocated, owning padded dataset, for use as `mergedDataset` in
PaddedDataset, long[]).

This is an optional convenience for building the unfiltered `mergedDataset`; callers
that already have their own concatenated buffer are not required to use it. The matching
`offsets` are simply each index's cumulative row count. Keep the returned dataset alive
for as long as any index built from it remains in use.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to concatenate, in the order they will be passed to PaddedDataset, long[]) |

**Returns**

A newly allocated owning padded dataset containing every index's rows, concatenated in `indexes` order

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:533`_

### concatenateAndFilterDatasets

```java
static PaddedDataset concatenateAndFilterDatasets(CagraIndex[] indexes, BitSet filter) throws Throwable
```

Concatenates every input index's dataset (in `indexes` order), retaining only the rows
selected by `filter`, into a freshly allocated, owning padded dataset, for use as
`mergedDataset` in \{@link #merge(CagraIndex[], PaddedDataset, long[], BitSet,
CagraIndexParams)\}.

This is an optional convenience for building the filtered `mergedDataset`; callers
that already have their own filtered, concatenated buffer are not required to use it. Call
BitSet) with the same `filter` to get the
matching `offsets`. Keep the returned dataset alive for as long as any index built from
it remains in use.

**Parameters**

| Name | Description |
| --- | --- |
| `indexes` | Array of CAGRA indexes to concatenate, in the order they will be passed to PaddedDataset, long[], BitSet, CagraIndexParams) |
| `filter` | Bitset selecting which rows (over the concatenation of every index's rows, in `indexes` order) survive into the output; a set bit keeps the row |

**Returns**

A newly allocated owning padded dataset containing every index's surviving rows, concatenated in `indexes` order

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:558`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:599`_

### from

```java
Builder from(InputStream inputStream)
```

Sets an instance of InputStream typically used when index deserialization is
needed.

**Parameters**

| Name | Description |
| --- | --- |
| `inputStream` | an instance of `InputStream` |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:616`_

### from

```java
Builder from(InputStream inputStream, DeserializeDataset outDataset)
```

Sets an input stream and an empty caller-owned output handle for explicit dataset
deserialization. The concrete output type must match the dataset layout stored in the
serialized index. Keep `outDataset` alive while the built index is in use.

**Parameters**

| Name | Description |
| --- | --- |
| `inputStream` | an instance of `InputStream` |
| `outDataset` | an empty `PaddedDataset` or `StandardDataset` |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:627`_

### from

```java
Builder from(CuVSMatrix graph)
```

Sets a CAGRA graph instance to re-create an index from a
previously built graph.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:633`_

### withDataset

```java
Builder withDataset(float[][] vectors)
```

Sets the dataset vectors for building the `CagraIndex`.

**Parameters**

| Name | Description |
| --- | --- |
| `vectors` | a two-dimensional float array |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:641`_

### withDataset

```java
Builder withDataset(CuVSMatrix dataset)
```

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:654`_

### withBbqDataset

```java
Builder withBbqDataset(BbqQuantizer... quantizers)
```

Builds the graph from one or two encoded BBQ representations. An optional dense dataset
supplied with `#withDataset(CuVSMatrix)` is attached before search; otherwise call
`CagraIndex#updateDataset(PaddedDatasetView)` or
`CagraIndex#updateDataset(PaddedDataset)` before searching.

The index stores views over the quantizer tensors rather than copying them, so they must
stay open for as long as the index is in use. A dense dataset passed to
`#withDataset(CuVSMatrix)` is owned by the index, as it is for a non-BBQ build, and is
closed with it.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:667`_

### withIndexParams

```java
Builder withIndexParams(CagraIndexParams cagraIndexParameters)
```

Registers an instance of configured `CagraIndexParams` with this
Builder.

**Parameters**

| Name | Description |
| --- | --- |
| `cagraIndexParameters` | An instance of CagraIndexParams. |

**Returns**

An instance of this Builder.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:676`_

### build

```java
CagraIndex build() throws Throwable
```

Builds and returns an instance of CagraIndex.

**Returns**

an instance of CagraIndex

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:683`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraIndex.java:26`_
