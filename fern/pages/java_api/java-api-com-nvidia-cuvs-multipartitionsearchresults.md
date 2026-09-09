---
slug: api-reference/java-api-com-nvidia-cuvs-multipartitionsearchresults
---

# MultiPartitionSearchResults

_Java package: `com.nvidia.cuvs`_

```java
public class MultiPartitionSearchResults
```

Holds the decoded results of a multi-partition GPU search.

Each entry `i` in [0, `#count`) identifies:

which input partition the result came from (`#getPartitionIndex(int)`)
the local vector ordinal within that partition (`#getOrdinal(int)`)
the raw CAGRA distance (`#getDistance(int)`)


The caller is responsible for mapping ordinals to its own global identifiers.

## Public Members

### count

```java
public int count()
```

Number of valid results (may be less than k if fewer candidates exist).

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/MultiPartitionSearchResults.java:36`_

### getPartitionIndex

```java
public int getPartitionIndex(int i)
```

Index into the original partition list for result `i`.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/MultiPartitionSearchResults.java:41`_

### getOrdinal

```java
public int getOrdinal(int i)
```

Local vector ordinal within the partition for result `i`.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/MultiPartitionSearchResults.java:46`_

### getDistance

```java
public float getDistance(int i)
```

Post-processed distance for result `i` (scaled + metric-transformed).

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/MultiPartitionSearchResults.java:51`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/MultiPartitionSearchResults.java:21`_
