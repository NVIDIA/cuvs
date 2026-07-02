---
slug: api-reference/java-api-com-nvidia-cuvs-tieredindexquery
---

# TieredIndexQuery

_Java package: `com.nvidia.cuvs`_

```java
public class TieredIndexQuery
```

TieredIndexQuery holds the search parameters and query vectors to be used
while invoking search. Currently only supports CAGRA index type.

Thread Safety: Each TieredIndexQuery instance should use its own
CuVSResources object that is not shared with other threads. Sharing CuVSResources
between threads can lead to memory allocation errors or JVM crashes.

## Public Members

### getIndexType

```java
public TieredIndexType getIndexType()
```

Gets the index type for this query.

**Returns**

the TieredIndexType

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:57`_

### getCagraSearchParameters

```java
public CagraSearchParams getCagraSearchParameters()
```

Gets the instance of CagraSearchParams initially set.

**Returns**

an instance CagraSearchParams

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:66`_

### getQueryVectors

```java
public float[][] getQueryVectors()
```

Gets the query vector 2D float array.

**Returns**

2D float array

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:75`_

### getMapping

```java
public List<Integer> getMapping()
```

Gets the passed map instance.

**Returns**

a map of ID mappings

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:84`_

### getTopK

```java
public int getTopK()
```

Gets the topK value.

**Returns**

the topK value

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:93`_

### getPrefilter

```java
public BitSet getPrefilter()
```

Gets the prefilter BitSet.

**Returns**

a BitSet object representing the prefilter

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:102`_

### getNumDocs

```java
public long getNumDocs()
```

Gets the number of documents in this index, as used for prefilter.

**Returns**

number of documents as an integer

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:111`_

### getResources

```java
public CuVSResources getResources()
```

Gets the CuVSResources instance for this query.

**Returns**

the CuVSResources instance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:120`_

### newBuilder

```java
public static Builder newBuilder(CuVSResources resources)
```

Creates a new Builder instance.

**Parameters**

| Name | Description |
| --- | --- |
| `resources` | the CuVSResources instance to use for this query |

**Returns**

a new Builder instance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:145`_

### Builder

```java
public Builder(CuVSResources resources)
```

Constructor that requires CuVSResources.

Important: The provided CuVSResources instance should not be
shared with other threads. Each thread performing searches should create its own
CuVSResources instance to avoid memory allocation conflicts and potential JVM crashes.

**Parameters**

| Name | Description |
| --- | --- |
| `resources` | the CuVSResources instance to use for this query (must not be shared between threads) |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:171`_

### withIndexType

```java
public Builder withIndexType(TieredIndexType indexType)
```

Sets the index type for this query.

**Parameters**

| Name | Description |
| --- | --- |
| `indexType` | the index type |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:181`_

### withSearchParams

```java
public Builder withSearchParams(CagraSearchParams cagraSearchParams)
```

Sets the instance of configured CagraSearchParams to be passed for search.

**Parameters**

| Name | Description |
| --- | --- |
| `cagraSearchParams` | an instance of the configured CagraSearchParams to be used for this query |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:193`_

### withQueryVectors

```java
public Builder withQueryVectors(float[][] queryVectors)
```

Registers the query vectors to be passed in the search call.

**Parameters**

| Name | Description |
| --- | --- |
| `queryVectors` | 2D float query vector array |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:204`_

### withMapping

```java
public Builder withMapping(List<Integer> mapping)
```

Sets the instance of mapping to be used for ID mapping.

**Parameters**

| Name | Description |
| --- | --- |
| `mapping` | the ID mapping instance |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:215`_

### withTopK

```java
public Builder withTopK(int topK)
```

Registers the topK value.

**Parameters**

| Name | Description |
| --- | --- |
| `topK` | the topK value used to retrieve the topK results |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:226`_

### withPrefilter

```java
public Builder withPrefilter(BitSet prefilter, int numDocs)
```

Sets a BitSet to use as prefilter while searching.

**Parameters**

| Name | Description |
| --- | --- |
| `prefilter` | the BitSet to use as prefilter |
| `numDocs` | Total number of dataset vectors; used to align the prefilter correctly |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:239`_

### build

```java
public TieredIndexQuery build()
```

Builds an instance of TieredIndexQuery.

**Returns**

an instance of TieredIndexQuery

**Throws**

| Type | Description |
| --- | --- |
| `IllegalStateException` | if required parameters are missing |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:251`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexQuery.java:23`_
