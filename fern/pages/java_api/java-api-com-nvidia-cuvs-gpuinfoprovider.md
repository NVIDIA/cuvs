---
slug: api-reference/java-api-com-nvidia-cuvs-gpuinfoprovider
---

# GPUInfoProvider

_Java package: `com.nvidia.cuvs`_

```java
public interface GPUInfoProvider
```

## Public Members

### availableGPUs

```java
List<GPUInfo> availableGPUs()
```

Gets all the available GPUs

**Returns**

a list of `GPUInfo` objects with GPU details

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/GPUInfoProvider.java:20`_

### compatibleGPUs

```java
List<GPUInfo> compatibleGPUs()
```

Get the list of compatible GPUs based on compute capability &gt;= 7.0 and total
memory &gt;= 8GB

**Returns**

a list of compatible GPUs. See `GPUInfo`

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/GPUInfoProvider.java:28`_

### getCurrentInfo

```java
CuVSResourcesInfo getCurrentInfo(CuVSResources resources)
```

Gets memory information relative to a `CuVSResources`

**Parameters**

| Name | Description |
| --- | --- |
| `resources` | from which to obtain memory information |

**Returns**

a `CuVSResourcesInfo` record containing the memory information

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/GPUInfoProvider.java:35`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/GPUInfoProvider.java:9`_
