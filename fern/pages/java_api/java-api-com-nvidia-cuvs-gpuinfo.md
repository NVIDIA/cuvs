---
slug: api-reference/java-api-com-nvidia-cuvs-gpuinfo
---

# GPUInfo

_Java package: `com.nvidia.cuvs`_

```java
public record GPUInfo( int gpuId, String name, long totalDeviceMemoryInBytes, int computeCapabilityMajor, int computeCapabilityMinor, boolean supportsConcurrentCopy, boolean supportsConcurrentKernels)
```

Contains GPU information

**Parameters**

| Name | Description |
| --- | --- |
| `gpuId` | id of the GPU starting from 0 |
| `name` | ASCII string identifying device |
| `totalDeviceMemoryInBytes` | total device memory in bytes |
| `computeCapabilityMajor` | the compute capability of the device (major) |
| `computeCapabilityMinor` | the compute capability of the device (minor) |
| `supportsConcurrentCopy` | whether the device can concurrently copy memory between host and device while executing a kernel |
| `supportsConcurrentKernels` | whether the device supports executing multiple kernels within the same context simultaneously |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/GPUInfo.java:20`_
