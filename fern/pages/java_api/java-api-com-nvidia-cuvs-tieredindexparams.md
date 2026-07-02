---
slug: api-reference/java-api-com-nvidia-cuvs-tieredindexparams
---

# TieredIndexParams

_Java package: `com.nvidia.cuvs`_

```java
public final class TieredIndexParams
```

Configuration parameters for building a `TieredIndex`.
Only CAGRA is currently supported as the underlying ANN algorithm.

## Public Members

### product

```java
L2, /** Inner product (cosine similarity) distance metric */ INNER_PRODUCT } private final Metric metric
```

L2 (Euclidean) distance metric

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexParams.java:20`_

### TieredIndexParams

```java
private TieredIndexParams(Builder builder)
```

Private constructor used by the Builder.

**Parameters**

| Name | Description |
| --- | --- |
| `builder` | The Builder instance containing the configuration |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexParams.java:35`_

### getMetric

```java
public Metric getMetric()
```

Returns the distance metric used for similarity computation.

**Returns**

The `Metric` (L2 or Inner Product)

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexParams.java:47`_

### getMinAnnRows

```java
public int getMinAnnRows()
```

Returns the minimum number of rows required to use the ANN algorithm.

**Returns**

The minimum row count threshold for ANN algorithm usage

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexParams.java:56`_

### isCreateAnnIndexOnExtend

```java
public boolean isCreateAnnIndexOnExtend()
```

Returns whether to create an ANN index when extending the dataset.

**Returns**

true if ANN index should be created on extend, false otherwise

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexParams.java:65`_

### getCagraParams

```java
public CagraIndexParams getCagraParams()
```

Returns the CAGRA-specific parameters for the ANN algorithm.

**Returns**

The `CagraIndexParams` configuration, or null if not using CAGRA

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexParams.java:75`_

### newBuilder

```java
public static Builder newBuilder()
```

Creates a new Builder for constructing TieredIndexParams.

**Returns**

A new Builder instance

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexParams.java:84`_

### metric

```java
public Builder metric(Metric metric)
```

Sets the distance metric for similarity computation.

**Parameters**

| Name | Description |
| --- | --- |
| `metric` | The `Metric` to use (L2 or Inner Product) |

**Returns**

This Builder instance for method chaining

**Throws**

| Type | Description |
| --- | --- |
| `NullPointerException` | if metric is null |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexParams.java:104`_

### minAnnRows

```java
public Builder minAnnRows(int minAnnRows)
```

Sets the minimum number of rows required to use the ANN algorithm.

**Parameters**

| Name | Description |
| --- | --- |
| `minAnnRows` | The minimum row count threshold (must be positive) |

**Returns**

This Builder instance for method chaining

**Throws**

| Type | Description |
| --- | --- |
| `IllegalArgumentException` | if minAnnRows is not positive |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexParams.java:116`_

### createAnnIndexOnExtend

```java
public Builder createAnnIndexOnExtend(boolean val)
```

Sets whether to create an ANN index when extending the dataset.

**Parameters**

| Name | Description |
| --- | --- |
| `val` | true to create ANN index on extend, false otherwise |

**Returns**

This Builder instance for method chaining

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexParams.java:130`_

### withCagraParams

```java
public Builder withCagraParams(CagraIndexParams params)
```

Sets the CAGRA-specific parameters for the ANN algorithm.

**Parameters**

| Name | Description |
| --- | --- |
| `params` | The `CagraIndexParams` configuration for CAGRA algorithm |

**Returns**

This Builder instance for method chaining

**Throws**

| Type | Description |
| --- | --- |
| `NullPointerException` | if params is null |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexParams.java:143`_

### build

```java
public TieredIndexParams build()
```

Builds and returns a `TieredIndexParams` instance with the
configured parameters.

**Returns**

A new TieredIndexParams instance

**Throws**

| Type | Description |
| --- | --- |
| `IllegalStateException` | if CAGRA params are required but not provided |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexParams.java:156`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/TieredIndexParams.java:14`_
