---
slug: api-reference/java-api-com-nvidia-cuvs-bruteforceindex
---

# BruteForceIndex

_Java package: `com.nvidia.cuvs`_

```java
public interface BruteForceIndex extends AutoCloseable
```

`BruteForceIndex` encapsulates a BRUTEFORCE index, along with methods
to interact with it.

## Public Members

### close

```java
@Override void close() throws Exception
```

Invokes the native destroy_brute_force_index function to de-allocate
BRUTEFORCE index

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndex.java:25`_

### search

```java
SearchResults search(BruteForceQuery cuvsQuery) throws Throwable
```

Invokes the native search_brute_force_index via the Panama API for searching
a BRUTEFORCE index.

**Parameters**

| Name | Description |
| --- | --- |
| `cuvsQuery` | an instance of `BruteForceQuery` holding the query vectors and other parameters |

**Returns**

an instance of `SearchResults` containing the results

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndex.java:36`_

### serialize

```java
void serialize(OutputStream outputStream) throws Throwable
```

A method to persist a BRUTEFORCE index using an instance of
`OutputStream` for writing index bytes.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes into |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndex.java:45`_

### serialize

```java
void serialize(OutputStream outputStream, Path tempFile) throws Throwable
```

A method to persist a BRUTEFORCE index using an instance of
`OutputStream` and path to the intermediate temporary file.

**Parameters**

| Name | Description |
| --- | --- |
| `outputStream` | an instance of `OutputStream` to write the index bytes to |
| `tempFile` | an intermediate `Path` where BRUTEFORCE index is written temporarily |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndex.java:56`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndex.java:64`_

### withIndexParams

```java
Builder withIndexParams(BruteForceIndexParams bruteForceIndexParams)
```

Registers an instance of configured `BruteForceIndexParams` with this
Builder.

**Parameters**

| Name | Description |
| --- | --- |
| `bruteForceIndexParams` | An instance of BruteForceIndexParams |

**Returns**

An instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndex.java:81`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndex.java:90`_

### withDataset

```java
Builder withDataset(float[][] vectors)
```

Sets the dataset vectors for building the `BruteForceIndex`.

**Parameters**

| Name | Description |
| --- | --- |
| `vectors` | a two-dimensional float array |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndex.java:98`_

### withDataset

```java
Builder withDataset(CuVSMatrix dataset)
```

Sets the dataset for building the `BruteForceIndex`.

**Parameters**

| Name | Description |
| --- | --- |
| `dataset` | a `CuVSMatrix` object containing the vectors |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndex.java:106`_

### build

```java
BruteForceIndex build() throws Throwable
```

Builds and returns an instance of `BruteForceIndex`.

**Returns**

an instance of `BruteForceIndex`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndex.java:113`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndex.java:20`_
