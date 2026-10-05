---
slug: api-reference/java-api-com-nvidia-cuvs-bbqquantizer
---

# BbqQuantizer

_Java package: `com.nvidia.cuvs`_

```java
public final class BbqQuantizer
```

Caller-owned tensors describing one encoded BBQ dataset representation.

The index stores views over these matrices rather than copying them, so they must stay open
for as long as any index built from them is in use.

## Public Members

### withCentroidNormSq

```java
public Builder withCentroidNormSq(float value)
```

Sets the squared L2 norm of the centroid. Required: there is no meaningful default, and
leaving it at zero skews inner-product and cosine distances without reporting an error.

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BbqQuantizer.java:185`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/BbqQuantizer.java:15`_
