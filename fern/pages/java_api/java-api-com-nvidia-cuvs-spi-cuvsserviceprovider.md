---
slug: api-reference/java-api-com-nvidia-cuvs-spi-cuvsserviceprovider
---

# CuVSServiceProvider

_Java package: `com.nvidia.cuvs.spi`_

```java
public abstract class CuVSServiceProvider
```

Service-provider class for \{@linkplain CuVSProvider\}.

## Public Members

### get

```java
public abstract CuVSProvider get(CuVSProvider builtinProvider)
```

Initialize and return an `CuVSProvider` provided by this provider.

**Parameters**

| Name | Description |
| --- | --- |
| `builtinProvider` | the built-in provider. |

**Returns**

the CuVSProvider provided by this provider

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSServiceProvider.java:22`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/spi/CuVSServiceProvider.java:16`_
