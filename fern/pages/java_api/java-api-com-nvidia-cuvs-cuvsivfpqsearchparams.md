---
slug: api-reference/java-api-com-nvidia-cuvs-cuvsivfpqsearchparams
---

# CuVSIvfPqSearchParams

_Java package: `com.nvidia.cuvs`_

```java
public class CuVSIvfPqSearchParams
```

## Public Members

### getnProbes

```java
public int getnProbes()
```

Gets the number of clusters to search

**Returns**

the number of clusters to search

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqSearchParams.java:71`_

### getLutDtype

```java
public CudaDataType getLutDtype()
```

Gets the data type of look up table to be created dynamically at search time

**Returns**

the data type of look up table to be created dynamically at search time

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqSearchParams.java:81`_

### getInternalDistanceDtype

```java
public CudaDataType getInternalDistanceDtype()
```

Gets the storage data type for distance/similarity computed at search time

**Returns**

the storage data type for distance/similarity computed at search time

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqSearchParams.java:90`_

### getPreferredShmemCarveout

```java
public double getPreferredShmemCarveout()
```

Gets the preferred fraction of SM's unified memory / L1 cache to be used as
shared memory

**Returns**

the preferred fraction of SM's unified memory / L1 cache to be used as shared memory

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqSearchParams.java:101`_

### withNProbes

```java
public Builder withNProbes(int nProbes)
```

Sets the number of clusters to search.

**Parameters**

| Name | Description |
| --- | --- |
| `nProbes` | the number of clusters to search |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqSearchParams.java:175`_

### withLutDtype

```java
public Builder withLutDtype(CudaDataType lutDtype)
```

Sets the the data type of look up table to be created dynamically at search
time.

**Parameters**

| Name | Description |
| --- | --- |
| `lutDtype` | the data type of look up table to be created dynamically at search time |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqSearchParams.java:188`_

### withInternalDistanceDtype

```java
public Builder withInternalDistanceDtype(CudaDataType internalDistanceDtype)
```

Sets the storage data type for distance/similarity computed at search time.

**Parameters**

| Name | Description |
| --- | --- |
| `internalDistanceDtype` | storage data type for distance/similarity computed at search time |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqSearchParams.java:200`_

### withPreferredShmemCarveout

```java
public Builder withPreferredShmemCarveout(double preferredShmemCarveout)
```

Sets the preferred fraction of SM's unified memory / L1 cache to be used as
shared memory.

**Parameters**

| Name | Description |
| --- | --- |
| `preferredShmemCarveout` | preferred fraction of SM's unified memory / L1 cache to be used as shared memory |

**Returns**

an instance of Builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqSearchParams.java:213`_

### build

```java
public CuVSIvfPqSearchParams build()
```

Builds an instance of `CuVSIvfPqSearchParams`.

**Returns**

an instance of `CuVSIvfPqSearchParams`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqSearchParams.java:223`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSIvfPqSearchParams.java:9`_
