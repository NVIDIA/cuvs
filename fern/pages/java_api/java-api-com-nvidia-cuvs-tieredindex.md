---
slug: api-reference/java-api-com-nvidia-cuvs-tieredindex
---

# TieredIndex

_Java package: `com.nvidia.cuvs`_

```java
public interface TieredIndex extends AutoCloseable
```

`TieredIndex` encapsulates a Tiered index, along with methods to
interact with it.

## Public Members

### close

```java
@Override void close() throws Exception
```

Destroys the underlying native TieredIndex object and releases associated
resources.

**Throws**

| Type | Description |
| --- | --- |
| `Exception` | if an error occurs during index destruction |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:22`_

### search

```java
SearchResults search(TieredIndexQuery query) throws Throwable
```

Searches the index with the specified query and search parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `query` | An instance of `TieredIndexQuery` describing the queries and search parameters |

**Returns**

An instance of `SearchResults` containing the k-nearest neighbors and their distances for each query

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the search operation |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:34`_

### getIndexType

```java
TieredIndexType getIndexType()
```

Returns the algorithm type backing this TieredIndex.

**Returns**

The `TieredIndexType` indicating the underlying algorithm (e.g., CAGRA)

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:42`_

### getCuVSResources

```java
CuVSResources getCuVSResources()
```

Returns the resources handle associated with this TieredIndex.

**Returns**

The `CuVSResources` instance used by this index

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:49`_

### newBuilder

```java
static Builder newBuilder(CuVSResources cuvsResources)
```

Creates a new Builder with an instance of `CuVSResources`.

**Parameters**

| Name | Description |
| --- | --- |
| `cuvsResources` | An instance of `CuVSResources` |

**Returns**

A new `Builder` instance for constructing a TieredIndex

**Throws**

| Type | Description |
| --- | --- |
| `NullPointerException` | if cuvsResources is null |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:58`_

### extend

```java
ExtendBuilder extend()
```

Returns an ExtendBuilder to add new data to the existing index.

**Returns**

An `ExtendBuilder` instance for extending the index

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:68`_

### from

```java
Builder from(InputStream inputStream)
```

**Parameters**

| Name | Description |
| --- | --- |
| `inputStream` | The input stream containing serialized index data |

**Returns**

This Builder instance for method chaining

**Throws**

| Type | Description |
| --- | --- |
| `UnsupportedOperationException` | as deserialization is not yet supported |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:82`_

### withDataset

```java
Builder withDataset(float[][] vectors)
```

Sets the dataset vectors for building the TieredIndex.

**Parameters**

| Name | Description |
| --- | --- |
| `vectors` | A two-dimensional float array containing the dataset vectors [n_vectors, dimensions] |

**Returns**

This Builder instance for method chaining

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:91`_

### withDataset

```java
Builder withDataset(CuVSMatrix dataset)
```

Sets the dataset for building the TieredIndex.

**Parameters**

| Name | Description |
| --- | --- |
| `dataset` | A `CuVSMatrix` instance containing the vectors |

**Returns**

This Builder instance for method chaining

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:99`_

### withIndexParams

```java
Builder withIndexParams(TieredIndexParams params)
```

Registers TieredIndex parameters with this Builder.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | An instance of `TieredIndexParams` containing the index configuration |

**Returns**

This Builder instance for method chaining

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:108`_

### withIndexType

```java
Builder withIndexType(TieredIndexType indexType)
```

Sets the index type for the TieredIndex.

**Parameters**

| Name | Description |
| --- | --- |
| `indexType` | The `TieredIndexType` to use (currently only CAGRA is supported) |

**Returns**

This Builder instance for method chaining

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:117`_

### build

```java
TieredIndex build() throws Throwable
```

Builds and returns an instance of TieredIndex with the configured
parameters.

**Returns**

A new `TieredIndex` instance

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during index construction |
| `IllegalArgumentException` | if both vectors and dataset are provided, or if required parameters are missing |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:129`_

### withDataset

```java
ExtendBuilder withDataset(float[][] vectors)
```

Sets the vectors to add to the existing index.

**Parameters**

| Name | Description |
| --- | --- |
| `vectors` | A two-dimensional float array containing the new vectors to add [n_new_vectors, dimensions] |

**Returns**

This ExtendBuilder instance for method chaining

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:152`_

### withDataset

```java
ExtendBuilder withDataset(CuVSMatrix dataset)
```

Sets the dataset to add to the existing index.

**Parameters**

| Name | Description |
| --- | --- |
| `dataset` | A `CuVSMatrix` instance containing the new vectors to add |

**Returns**

This ExtendBuilder instance for method chaining

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:161`_

### execute

```java
void execute() throws Throwable
```

Executes the extend operation, adding the specified data to the index.

**Throws**

| Type | Description |
| --- | --- |
| `Throwable` | if an error occurs during the extend operation |
| `IllegalArgumentException` | if both vectors and dataset are provided, or if no data is provided |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:171`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndex.java:15`_
