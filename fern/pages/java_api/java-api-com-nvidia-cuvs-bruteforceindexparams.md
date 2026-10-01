---
slug: api-reference/java-api-com-nvidia-cuvs-bruteforceindexparams
---

# BruteForceIndexParams

_Java package: `com.nvidia.cuvs`_

```java
public class BruteForceIndexParams
```

Supplemental parameters to build BRUTEFORCE index.

## Public Members

### getNumWriterThreads

```java
public int getNumWriterThreads()
```

Gets the number of threads used to build the index.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndexParams.java:27`_

### withNumWriterThreads

```java
public Builder withNumWriterThreads(int numWriterThreads)
```

Sets the number of writer threads to use for indexing.

**Parameters**

| Name | Description |
| --- | --- |
| `numWriterThreads` | number of writer threads to use |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndexParams.java:44`_

### build

```java
public BruteForceIndexParams build()
```

Builds an instance of `BruteForceIndexParams`.

**Returns**

an instance of `BruteForceIndexParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndexParams.java:54`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BruteForceIndexParams.java:12`_
