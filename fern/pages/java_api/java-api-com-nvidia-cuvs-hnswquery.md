---
slug: api-reference/java-api-com-nvidia-cuvs-hnswquery
---

# HnswQuery

_Java package: `com.nvidia.cuvs`_

```java
public class HnswQuery
```

HnswQuery holds the query vectors to be used while invoking search on the
HNSW index.

Thread Safety: Each HnswQuery instance should use its own
CuVSResources object that is not shared with other threads. Sharing CuVSResources
between threads can lead to memory allocation errors or JVM crashes.

## Public Members

### HnswQuery

```java
private HnswQuery( HnswSearchParams hnswSearchParams, float[][] queryVectors, LongToIntFunction mapping, int topK, CuVSResources resources)
```

Constructs an instance of `HnswQuery` using queryVectors, mapping, and
topK.

**Parameters**

| Name | Description |
| --- | --- |
| `hnswSearchParams` | the search parameters to use |
| `queryVectors` | 2D float query vector array |
| `mapping` | a function mapping ordinals (neighbor IDs) to custom user IDs |
| `topK` | the top k results to return |
| `resources` | CuVSResources instance to use for this query |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswQuery.java:38`_

### getHnswSearchParams

```java
public HnswSearchParams getHnswSearchParams()
```

Gets the instance of HnswSearchParams.

**Returns**

the instance of `HnswSearchParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswQuery.java:56`_

### getQueryVectors

```java
public float[][] getQueryVectors()
```

Gets the query vector 2D float array.

**Returns**

2D float array

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswQuery.java:65`_

### getMapping

```java
public LongToIntFunction getMapping()
```

Gets the function mapping ordinals (neighbor IDs) to custom user IDs

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswQuery.java:72`_

### getTopK

```java
public int getTopK()
```

Gets the topK value.

**Returns**

an integer

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswQuery.java:81`_

### getResources

```java
public CuVSResources getResources()
```

Gets the CuVSResources instance for this query.

**Returns**

the CuVSResources instance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswQuery.java:90`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswQuery.java:125`_

### withSearchParams

```java
public Builder withSearchParams(HnswSearchParams hnswSearchParams)
```

Sets the instance of configured HnswSearchParams to be passed for search.

**Parameters**

| Name | Description |
| --- | --- |
| `hnswSearchParams` | an instance of the configured HnswSearchParams to be used for this query |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswQuery.java:136`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswQuery.java:147`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswQuery.java:158`_

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

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswQuery.java:169`_

### build

```java
public HnswQuery build()
```

Builds an instance of `HnswQuery`

**Returns**

an instance of `HnswQuery`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswQuery.java:179`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswQuery.java:21`_
