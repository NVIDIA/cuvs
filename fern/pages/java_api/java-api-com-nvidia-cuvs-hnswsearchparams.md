---
slug: api-reference/java-api-com-nvidia-cuvs-hnswsearchparams
---

# HnswSearchParams

_Java package: `com.nvidia.cuvs`_

```java
public record HnswSearchParams(int ef, int numThreads)
```

HnswSearchParams encapsulates the logic for configuring and holding search
parameters for HNSW index.

**Parameters**

| Name | Description |
| --- | --- |
| `ef` | the ef value |
| `numThreads` | the number of threads |

## Public Members

### Builder

```java
public Builder()
```

Constructs this Builder with an instance of Arena.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswSearchParams.java:39`_

### withEF

```java
public Builder withEF(int ef)
```

Sets the ef value

**Parameters**

| Name | Description |
| --- | --- |
| `ef` | the ef value |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswSearchParams.java:47`_

### withNumThreads

```java
public Builder withNumThreads(int numThreads)
```

Sets the number of threads

**Parameters**

| Name | Description |
| --- | --- |
| `numThreads` | the number of threads |

**Returns**

an instance of this Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswSearchParams.java:58`_

### build

```java
public HnswSearchParams build()
```

Builds an instance of `HnswSearchParams` with passed search parameters.

**Returns**

an instance of HnswSearchParams

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswSearchParams.java:68`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/HnswSearchParams.java:15`_
