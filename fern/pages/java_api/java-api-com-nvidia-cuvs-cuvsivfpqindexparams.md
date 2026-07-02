---
slug: api-reference/java-api-com-nvidia-cuvs-cuvsivfpqindexparams
---

# CuVSIvfPqIndexParams

_Java package: `com.nvidia.cuvs`_

```java
public class CuVSIvfPqIndexParams
```

## Public Members

### getMetric

```java
public CuvsDistanceType getMetric()
```

Gets the distance type.

**Returns**

the distance type

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:143`_

### getCodebookKind

```java
public CodebookGen getCodebookKind()
```

Gets how PQ codebooks are created

**Returns**

how PQ codebooks are created

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:152`_

### getMetricArg

```java
public float getMetricArg()
```

Gets the argument used by some distance metrics

**Returns**

the argument used by some distance metrics

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:161`_

### getKmeansTrainsetFraction

```java
public double getKmeansTrainsetFraction()
```

Gets the fraction of data to use during iterative kmeans building

**Returns**

the fraction of data to use during iterative kmeans building

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:170`_

### getnLists

```java
public int getnLists()
```

Gets the number of inverted lists (clusters)

**Returns**

the number of inverted lists (clusters)

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:179`_

### getKmeansNIters

```java
public int getKmeansNIters()
```

Gets the number of iterations searching for kmeans centers

**Returns**

the number of iterations searching for kmeans centers

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:188`_

### getPqBits

```java
public int getPqBits()
```

Gets the bit length of the vector element after compression by PQ

**Returns**

the bit length of the vector element after compression by PQ

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:197`_

### getPqDim

```java
public int getPqDim()
```

Gets the dimensionality of the vector after compression by PQ

**Returns**

the dimensionality of the vector after compression by PQ

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:206`_

### isAddDataOnBuild

```java
public boolean isAddDataOnBuild()
```

Gets whether the dataset content is added to the index

**Returns**

whether the dataset content is added to the index

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:215`_

### isForceRandomRotation

```java
public boolean isForceRandomRotation()
```

Gets the random rotation matrix on the input data and queries

**Returns**

the random rotation matrix on the input data and queries

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:224`_

### isConservativeMemoryAllocation

```java
public boolean isConservativeMemoryAllocation()
```

Gets if conservative allocation behavior is set

**Returns**

if conservative allocation behavior is set

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:233`_

### getMaxTrainPointsPerPqCode

```java
public int getMaxTrainPointsPerPqCode()
```

Gets whether max number of data points to use per PQ code during PQ codebook
training is set

**Returns**

whether max number of data points to use per PQ code during PQ codebook training is set

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:244`_

### withMetric

```java
public Builder withMetric(CuvsDistanceType metric)
```

Sets the distance type.

**Parameters**

| Name | Description |
| --- | --- |
| `metric` | distance type |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:389`_

### withMetricArg

```java
public Builder withMetricArg(float metricArg)
```

Sets the argument used by some distance metrics.

**Parameters**

| Name | Description |
| --- | --- |
| `metricArg` | argument used by some distance metrics |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:400`_

### withAddDataOnBuild

```java
public Builder withAddDataOnBuild(boolean addDataOnBuild)
```

Sets whether to add the dataset content to the index.

**Parameters**

| Name | Description |
| --- | --- |
| `addDataOnBuild` | whether to add the dataset content to the index |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:411`_

### withNLists

```java
public Builder withNLists(int nLists)
```

Sets the number of inverted lists (clusters)

**Parameters**

| Name | Description |
| --- | --- |
| `nLists` | number of inverted lists (clusters) |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:422`_

### withKmeansNIters

```java
public Builder withKmeansNIters(int kmeansNIters)
```

Sets the number of iterations searching for kmeans centers

**Parameters**

| Name | Description |
| --- | --- |
| `kmeansNIters` | number of iterations searching for kmeans centers |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:433`_

### withKmeansTrainsetFraction

```java
public Builder withKmeansTrainsetFraction(double kmeansTrainsetFraction)
```

Sets the fraction of data to use during iterative kmeans building.

**Parameters**

| Name | Description |
| --- | --- |
| `kmeansTrainsetFraction` | fraction of data to use during iterative kmeans building |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:445`_

### withPqBits

```java
public Builder withPqBits(int pqBits)
```

Sets the bit length of the vector element after compression by PQ.

**Parameters**

| Name | Description |
| --- | --- |
| `pqBits` | bit length of the vector element after compression by PQ |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:456`_

### withPqDim

```java
public Builder withPqDim(int pqDim)
```

Sets the dimensionality of the vector after compression by PQ.

**Parameters**

| Name | Description |
| --- | --- |
| `pqDim` | dimensionality of the vector after compression by PQ |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:467`_

### withCodebookKind

```java
public Builder withCodebookKind(CodebookGen codebookKind)
```

Sets how PQ codebooks are created.

**Parameters**

| Name | Description |
| --- | --- |
| `codebookKind` | how PQ codebooks are created |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:478`_

### withForceRandomRotation

```java
public Builder withForceRandomRotation(boolean forceRandomRotation)
```

Sets the random rotation matrix on the input data and queries.

**Parameters**

| Name | Description |
| --- | --- |
| `forceRandomRotation` | random rotation matrix on the input data and queries |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:490`_

### withConservativeMemoryAllocation

```java
public Builder withConservativeMemoryAllocation(boolean conservativeMemoryAllocation)
```

Sets the conservative allocation behavior

**Parameters**

| Name | Description |
| --- | --- |
| `conservativeMemoryAllocation` | conservative allocation behavior |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:501`_

### withMaxTrainPointsPerPqCode

```java
public Builder withMaxTrainPointsPerPqCode(int maxTrainPointsPerPqCode)
```

Sets the max number of data points to use per PQ code during PQ codebook
training

**Parameters**

| Name | Description |
| --- | --- |
| `maxTrainPointsPerPqCode` | max number of data points to use per PQ code during PQ codebook training |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:514`_

### build

```java
public CuVSIvfPqIndexParams build()
```

Builds an instance of `CuVSIvfPqIndexParams`.

**Returns**

an instance of `CuVSIvfPqIndexParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:524`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqIndexParams.java:10`_
