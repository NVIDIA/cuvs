---
slug: api-reference/java-api-com-nvidia-cuvs-cagrasearchparams
---

# CagraSearchParams

_Java package: `com.nvidia.cuvs`_

```java
public class CagraSearchParams
```

CagraSearchParams encapsulates the logic for configuring and holding search
parameters.

## Public Members

### SINGLE_CTA

```java
SINGLE_CTA(0), /** * for small batch sizes */ MULTI_CTA(1), /** * MULTI_KERNEL */ MULTI_KERNEL(2), /** * AUTO */ AUTO(100)
```

for large batch sizes

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:35`_

### MULTI_CTA

```java
MULTI_CTA(1), /** * MULTI_KERNEL */ MULTI_KERNEL(2), /** * AUTO */ AUTO(100)
```

for small batch sizes

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:39`_

### MULTI_KERNEL

```java
MULTI_KERNEL(2), /** * AUTO */ AUTO(100)
```

MULTI_KERNEL

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:43`_

### AUTO

```java
AUTO(100)
```

AUTO

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:47`_

### HASH

```java
HASH(0), /** * SMALL */ SMALL(1), /** * AUTO_HASH */ AUTO_HASH(100)
```

HASH

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:66`_

### SMALL

```java
SMALL(1), /** * AUTO_HASH */ AUTO_HASH(100)
```

SMALL

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:70`_

### AUTO_HASH

```java
AUTO_HASH(100)
```

AUTO_HASH

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:74`_

### CagraSearchParams

```java
private CagraSearchParams( int maxQueries, int iTopKSize, int maxIterations, SearchAlgo searchAlgo, int teamSize, int searchWidth, int minIterations, int threadBlockSize, HashMapMode hashmapMode, int hashmapMinBitlen, float hashmapMaxFillRate, int numRandomSamplings, long randXORMask)
```

Constructs an instance of CagraSearchParams with passed search parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `maxQueries` | the maximum number of queries to search at the same time (batch size) |
| `iTopKSize` | the number of intermediate search results retained during the search |
| `maxIterations` | the upper limit of search iterations |
| `searchAlgo` | the search implementation is configured |
| `teamSize` | the number of threads used to calculate a single distance |
| `searchWidth` | the number of graph nodes to select as the starting point for the search in each iteration |
| `minIterations` | the lower limit of search iterations |
| `threadBlockSize` | the thread block size |
| `hashmapMode` | the hash map type configured |
| `hashmapMinBitlen` | the lower limit of hash map bit length |
| `hashmapMaxFillRate` | the upper limit of hash map fill rate |
| `numRandomSamplings` | the number of iterations of initial random seed node selection |
| `randXORMask` | the bit mask used for initial random seed node selection |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:109`_

### getMaxQueries

```java
public int getMaxQueries()
```

Gets the maximum number of queries to search at the same time (batch size).

**Returns**

the maximum number of queries

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:143`_

### getITopKSize

```java
public int getITopKSize()
```

Gets the number of intermediate search results retained during the search.

**Returns**

the number of intermediate search results

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:152`_

### getMaxIterations

```java
public int getMaxIterations()
```

Gets the upper limit of search iterations.

**Returns**

the upper limit value

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:161`_

### getTeamSize

```java
public int getTeamSize()
```

Gets the number of threads used to calculate a single distance.

**Returns**

the number of threads configured

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:170`_

### getSearchWidth

```java
public int getSearchWidth()
```

Gets the number of graph nodes to select as the starting point for the search
in each iteration.

**Returns**

the number of graph nodes

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:180`_

### getMinIterations

```java
public int getMinIterations()
```

Gets the lower limit of search iterations.

**Returns**

the lower limit value

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:189`_

### getThreadBlockSize

```java
public int getThreadBlockSize()
```

Gets the thread block size.

**Returns**

the thread block size

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:198`_

### getHashmapMinBitlen

```java
public int getHashmapMinBitlen()
```

Gets the lower limit of hash map bit length.

**Returns**

the lower limit value

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:207`_

### getNumRandomSamplings

```java
public int getNumRandomSamplings()
```

Gets the number of iterations of initial random seed node selection.

**Returns**

the number of iterations

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:216`_

### getHashMapMaxFillRate

```java
public float getHashMapMaxFillRate()
```

Gets the upper limit of hash map fill rate.

**Returns**

the upper limit of hash map fill rate

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:225`_

### getRandXORMask

```java
public long getRandXORMask()
```

Gets the bit mask used for initial random seed node selection.

**Returns**

the bit mask value

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:234`_

### getCagraSearchAlgo

```java
public SearchAlgo getCagraSearchAlgo()
```

Gets which search implementation is configured.

**Returns**

the configured `SearchAlgo`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:243`_

### getHashMapMode

```java
public HashMapMode getHashMapMode()
```

Gets the hash map mode configured.

**Returns**

the configured `HashMapMode`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:252`_

### Builder

```java
public Builder()
```

Default constructor.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:309`_

### withMaxQueries

```java
public Builder withMaxQueries(int maxQueries)
```

Sets the maximum number of queries to search at the same time (batch size).
Auto select when 0.

**Parameters**

| Name | Description |
| --- | --- |
| `maxQueries` | the maximum number of queries |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:318`_

### withItopkSize

```java
public Builder withItopkSize(int iTopKSize)
```

Sets the number of intermediate search results retained during the search.
This is the main knob to adjust trade off between accuracy and search speed.
Higher values improve the search accuracy.

**Parameters**

| Name | Description |
| --- | --- |
| `iTopKSize` | the number of intermediate search results |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:331`_

### withMaxIterations

```java
public Builder withMaxIterations(int maxIterations)
```

Sets the upper limit of search iterations. Auto select when 0.

**Parameters**

| Name | Description |
| --- | --- |
| `maxIterations` | the upper limit of search iterations |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:342`_

### withAlgo

```java
public Builder withAlgo(SearchAlgo cuvsCagraSearchAlgo)
```

Sets which search implementation to use.

**Parameters**

| Name | Description |
| --- | --- |
| `cuvsCagraSearchAlgo` | the `SearchAlgo` to use |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:353`_

### withTeamSize

```java
public Builder withTeamSize(int teamSize)
```

Sets the number of threads used to calculate a single distance. 4, 8, 16, or
32.

**Parameters**

| Name | Description |
| --- | --- |
| `teamSize` | the number of threads used to calculate a single distance |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:365`_

### withSearchWidth

```java
public Builder withSearchWidth(int searchWidth)
```

Sets the number of graph nodes to select as the starting point for the search
in each iteration.

**Parameters**

| Name | Description |
| --- | --- |
| `searchWidth` | the number of graph nodes to select |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:377`_

### withMinIterations

```java
public Builder withMinIterations(int minIterations)
```

Sets the lower limit of search iterations.

**Parameters**

| Name | Description |
| --- | --- |
| `minIterations` | the lower limit of search iterations |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:388`_

### withThreadBlockSize

```java
public Builder withThreadBlockSize(int threadBlockSize)
```

Sets the thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when
0.

**Parameters**

| Name | Description |
| --- | --- |
| `threadBlockSize` | the thread block size |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:400`_

### withHashMapMode

```java
public Builder withHashMapMode(HashMapMode hashMapMode)
```

Sets the hash map type. Auto selection when AUTO.

**Parameters**

| Name | Description |
| --- | --- |
| `hashMapMode` | the `HashMapMode` |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:411`_

### withHashMapMinBitlen

```java
public Builder withHashMapMinBitlen(int hashMapMinBitlen)
```

Sets the lower limit of hash map bit length. More than 8.

**Parameters**

| Name | Description |
| --- | --- |
| `hashMapMinBitlen` | the lower limit of hash map bit length |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:422`_

### withHashMapMaxFillRate

```java
public Builder withHashMapMaxFillRate(float hashMapMaxFillRate)
```

Sets the upper limit of hash map fill rate. More than 0.1, less than 0.9.

**Parameters**

| Name | Description |
| --- | --- |
| `hashMapMaxFillRate` | the upper limit of hash map fill rate |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:433`_

### withNumRandomSamplings

```java
public Builder withNumRandomSamplings(int numRandomSamplings)
```

Sets the number of iterations of initial random seed node selection. 1 or
more.

**Parameters**

| Name | Description |
| --- | --- |
| `numRandomSamplings` | the number of iterations of initial random seed node selection |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:446`_

### withRandXorMask

```java
public Builder withRandXorMask(long randXORMask)
```

Sets the bit mask used for initial random seed node selection.

**Parameters**

| Name | Description |
| --- | --- |
| `randXORMask` | the bit mask used for initial random seed node selection |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:457`_

### build

```java
public CagraSearchParams build()
```

Builds an instance of `CagraSearchParams` with passed search
parameters.

**Returns**

an instance of CagraSearchParams

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:468`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraSearchParams.java:13`_
