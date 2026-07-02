---
slug: api-reference/java-api-com-nvidia-cuvs-cuvsdevicematrix
---

# CuVSDeviceMatrix

_Java package: `com.nvidia.cuvs`_

```java
public interface CuVSDeviceMatrix extends CuVSMatrix
```

A Dataset implementation backed by device (GPU) memory.

## Public Members

### toHost

```java
default CuVSHostMatrix toHost()
```

Returns a new host matrix with data from this device matrix.
The returned host matrix will need to be managed by the caller, which will be
responsible to call `CuVSMatrix#close()` to free its resources when done.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSDeviceMatrix.java:16`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CuVSDeviceMatrix.java:10`_
