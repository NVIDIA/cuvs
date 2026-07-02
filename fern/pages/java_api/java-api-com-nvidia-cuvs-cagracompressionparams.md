---
slug: api-reference/java-api-com-nvidia-cuvs-cagracompressionparams
---

# CagraCompressionParams

_Java package: `com.nvidia.cuvs`_

```java
public class CagraCompressionParams
```

Supplemental compression parameters to build CAGRA Index.

## Public Members

### CagraCompressionParams

```java
private CagraCompressionParams( int pqBits, int pqDim, int vqNCenters, int kmeansNIters, double vqKmeansTrainsetFraction, double pqKmeansTrainsetFraction)
```

Constructs an instance of CagraCompressionParams with passed search
parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `pqBits` | the bit length of the vector element after compression by PQ |
| `pqDim` | the dimensionality of the vector after compression by PQ |
| `vqNCenters` | the vector quantization (VQ) codebook size - number of “coarse cluster centers” |
| `kmeansNIters` | the number of iterations searching for kmeans centers (both VQ and PQ phases) |
| `vqKmeansTrainsetFraction` | the fraction of data to use during iterative kmeans building (VQ phase) |
| `pqKmeansTrainsetFraction` | the fraction of data to use during iterative kmeans building (PQ phase) |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:37`_

### getPqBits

```java
public int getPqBits()
```

Gets the bit length of the vector element after compression by PQ.

**Returns**

the bit length of the vector element after compression by PQ.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:57`_

### getPqDim

```java
public int getPqDim()
```

Gets the dimensionality of the vector after compression by PQ.

**Returns**

the dimensionality of the vector after compression by PQ.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:66`_

### getVqNCenters

```java
public int getVqNCenters()
```

Gets the vector quantization (VQ) codebook size - number of “coarse cluster
centers”.

**Returns**

the vector quantization (VQ) codebook size - number of “coarse cluster centers”.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:77`_

### getKmeansNIters

```java
public int getKmeansNIters()
```

Gets the number of iterations searching for kmeans centers (both VQ and PQ
phases).

**Returns**

the number of iterations searching for kmeans centers (both VQ and PQ phases).

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:88`_

### getVqKmeansTrainsetFraction

```java
public double getVqKmeansTrainsetFraction()
```

Gets the fraction of data to use during iterative kmeans building (VQ phase).

**Returns**

the fraction of data to use during iterative kmeans building (VQ phase).

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:98`_

### getPqKmeansTrainsetFraction

```java
public double getPqKmeansTrainsetFraction()
```

Gets the fraction of data to use during iterative kmeans building (PQ phase).

**Returns**

the fraction of data to use during iterative kmeans building (PQ phase).

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:108`_

### withPqBits

```java
public Builder withPqBits(int pqBits)
```

Sets the bit length of the vector element after compression by PQ.

Possible values: [4, 5, 6, 7, 8]. Hint: the smaller the ‘pq_bits’, the
smaller the index size and the better the search performance, but the lower
the recall.

**Parameters**

| Name | Description |
| --- | --- |
| `pqBits` |  |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:153`_

### withPqDim

```java
public Builder withPqDim(int pqDim)
```

Sets the dimensionality of the vector after compression by PQ.

When zero, an optimal value is selected using a heuristic.

**Parameters**

| Name | Description |
| --- | --- |
| `pqDim` |  |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:166`_

### withVqNCenters

```java
public Builder withVqNCenters(int vqNCenters)
```

Sets the vector quantization (VQ) codebook size - number of “coarse cluster
centers”.

When zero, an optimal value is selected using a heuristic.

**Parameters**

| Name | Description |
| --- | --- |
| `vqNCenters` |  |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:180`_

### withKmeansNIters

```java
public Builder withKmeansNIters(int kmeansNIters)
```

Sets the number of iterations searching for kmeans centers (both VQ and PQ
phases).

**Parameters**

| Name | Description |
| --- | --- |
| `kmeansNIters` |  |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:192`_

### withVqKmeansTrainsetFraction

```java
public Builder withVqKmeansTrainsetFraction(double vqKmeansTrainsetFraction)
```

Sets the fraction of data to use during iterative kmeans building (VQ phase).

When zero, an optimal value is selected using a heuristic.

**Parameters**

| Name | Description |
| --- | --- |
| `vqKmeansTrainsetFraction` |  |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:205`_

### withPqKmeansTrainsetFraction

```java
public Builder withPqKmeansTrainsetFraction(double pqKmeansTrainsetFraction)
```

Sets the fraction of data to use during iterative kmeans building (PQ phase).

When zero, an optimal value is selected using a heuristic.

**Parameters**

| Name | Description |
| --- | --- |
| `pqKmeansTrainsetFraction` |  |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:218`_

### build

```java
public CagraCompressionParams build()
```

Builds an instance of `CagraCompressionParams`.

**Returns**

an instance of `CagraCompressionParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:228`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraCompressionParams.java:12`_
