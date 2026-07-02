---
slug: api-reference/java-api-com-nvidia-cuvs-cuvsresourcesinfo
---

# CuVSResourcesInfo

_Java package: `com.nvidia.cuvs`_

```java
public record CuVSResourcesInfo(long freeDeviceMemoryInBytes, long totalDeviceMemoryInBytes)
```

Contains performance-related information associated to a `CuVSResources` and its GPU.
Can be extended to report different types of GPU memory linked to the resources,
e.g. the type and capacity of the underlying RMM `device_memory_resource`

**Parameters**

| Name | Description |
| --- | --- |
| `freeDeviceMemoryInBytes` | free memory in bytes, as reported by the device driver |
| `totalDeviceMemoryInBytes` | total device memory in bytes |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSResourcesInfo.java:15`_
