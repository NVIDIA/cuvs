---
slug: api-reference/java-api-com-nvidia-cuvs-cuvsivfpqparams
---

# CuVSIvfPqParams

_Java package: `com.nvidia.cuvs`_

```java
public class CuVSIvfPqParams
```

## Public Members

### getIndexParams

```java
public CuVSIvfPqIndexParams getIndexParams()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqParams.java:29`_

### getSearchParams

```java
public CuVSIvfPqSearchParams getSearchParams()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqParams.java:37`_

### getRefinementRate

```java
public float getRefinementRate()
```

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqParams.java:45`_

### Builder

```java
private CuVSIvfPqIndexParams cuVSIvfPqIndexParams = new CuVSIvfPqIndexParams.Builder().build()
```

CuVS IVF_PQ index parameters

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqParams.java:66`_

### Builder

```java
private CuVSIvfPqSearchParams cuVSIvfPqSearchParams = new CuVSIvfPqSearchParams.Builder().build()
```

CuVS IVF_PQ search parameters

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqParams.java:69`_

### withCuVSIvfPqIndexParams

```java
public Builder withCuVSIvfPqIndexParams(CuVSIvfPqIndexParams cuVSIvfPqIndexParams)
```

Sets the CuVS IVF_PQ index parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `cuVSIvfPqIndexParams` | the CuVS IVF_PQ index parameters |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqParams.java:83`_

### withCuVSIvfPqSearchParams

```java
public Builder withCuVSIvfPqSearchParams(CuVSIvfPqSearchParams cuVSIvfPqSearchParams)
```

Sets the CuVS IVF_PQ search parameters.

**Parameters**

| Name | Description |
| --- | --- |
| `cuVSIvfPqSearchParams` | the CuVS IVF_PQ search parameters |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqParams.java:94`_

### withRefinementRate

```java
public Builder withRefinementRate(float refinementRate)
```

Sets the refinement rate, default 2.0.

**Parameters**

| Name | Description |
| --- | --- |
| `refinementRate` | the refinement rate |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqParams.java:105`_

### build

```java
public CuVSIvfPqParams build()
```

Builds an instance of `CuVSIvfPqParams`.

**Returns**

an instance of `CuVSIvfPqParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqParams.java:115`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqParams.java:7`_
