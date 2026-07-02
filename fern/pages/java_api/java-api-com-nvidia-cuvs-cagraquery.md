---
slug: api-reference/java-api-com-nvidia-cuvs-cagraquery
---

# CagraQuery

_Java package: `com.nvidia.cuvs`_

```java
public class CagraQuery
```

CagraQuery holds the CagraSearchParams and the query vectors to be used while
invoking search.

Thread Safety: Each CagraQuery instance should use its own
CuVSResources object that is not shared with other threads. Sharing CuVSResources
between threads can lead to memory allocation errors or JVM crashes.

## Public Members

### CagraQuery

```java
private CagraQuery( CagraSearchParams cagraSearchParameters, CuVSMatrix queryVectors, LongToIntFunction mapping, int topK, BitSet prefilter, int numDocs, CuVSResources resources)
```

Constructs an instance of `CagraQuery` using cagraSearchParameters,
preFilter, queryVectors, mapping, and topK.

**Parameters**

| Name | Description |
| --- | --- |
| `cagraSearchParameters` | an instance of `CagraSearchParams` holding the search parameters |
| `queryVectors` | 2D float query vector array |
| `mapping` | a function mapping ordinals (neighbor IDs) to custom user IDs |
| `topK` | the top k results to return |
| `prefilter` | A single BitSet to use as filter while searching the CAGRA index |
| `numDocs` | Total number of dataset vectors; used to align the prefilter correctly |
| `resources` | CuVSResources instance to use for this query |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:43`_

### getCagraSearchParameters

```java
public CagraSearchParams getCagraSearchParameters()
```

Gets the instance of CagraSearchParams initially set.

**Returns**

an instance CagraSearchParams

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:66`_

### getQueryVectors

```java
public CuVSMatrix getQueryVectors()
```

Gets the query vector matrix.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:73`_

### getMapping

```java
public LongToIntFunction getMapping()
```

Gets the function mapping ordinals (neighbor IDs) to custom user IDs

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:80`_

### getTopK

```java
public int getTopK()
```

Gets the topK value.

**Returns**

the topK value

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:89`_

### getPrefilter

```java
public BitSet getPrefilter()
```

Gets the prefilter BitSet.

**Returns**

a BitSet object representing the prefilter

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:98`_

### getNumDocs

```java
public int getNumDocs()
```

Gets the number of documents in this index, as used for prefilter

**Returns**

number of documents as an integer

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:107`_

### getResources

```java
public CuVSResources getResources()
```

Gets the CuVSResources instance for this query.

**Returns**

the CuVSResources instance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:116`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:155`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:166`_

### withQueryVectors

```java
public Builder withQueryVectors(CuVSMatrix queryVectors)
```

Registers the query vectors to be passed in the search call.

**Parameters**

| Name | Description |
| --- | --- |
| `queryVectors` | 2D query vector array |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:177`_

### withMapping

```java
public Builder withMapping(LongToIntFunction mapping)
```

Sets the function used to map ordinals (neighbor IDs) to custom user IDs

**Parameters**

| Name | Description |
| --- | --- |
| `mapping` | a function mapping ordinals (neighbor IDs) to custom user IDs |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:188`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:199`_

### withPrefilter

```java
public Builder withPrefilter(BitSet prefilter, int numDocs)
```

Sets a global prefilter for all queries in this `CagraQuery`.
The `prefilter` array must contain exactly one `BitSet`,
which is applied to all queries. A bit value of `1` includes the
corresponding dataset vector; `0` excludes it.

**Parameters**

| Name | Description |
| --- | --- |
| `prefilter` | an array with the global filter BitSet |
| `numDocs` | total number of vectors in the dataset (for alignment) |

**Returns**

this `Builder` instance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:214`_

### build

```java
public CagraQuery build()
```

Builds an instance of CuVSQuery.

**Returns**

an instance of CuVSQuery

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:225`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraQuery.java:21`_
