---
slug: api-reference/java-api-com-nvidia-cuvs-cagramergeparams
---

# CagraMergeParams

_Java package: `com.nvidia.cuvs`_

```java
public class CagraMergeParams
```

## Public Members

### CagraMergeParams

```java
private CagraMergeParams(CagraIndexParams outputIndexParams, MergeStrategy strategy)
```

Constructs a CagraMergeParams with the given output index parameters and merge strategy.

**Parameters**

| Name | Description |
| --- | --- |
| `outputIndexParams` | Index parameters for the output index |
| `strategy` | Merge strategy to use |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraMergeParams.java:17`_

### getOutputIndexParams

```java
public CagraIndexParams getOutputIndexParams()
```

Gets the index parameters for the output index.

**Returns**

Index parameters to use for the output index

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraMergeParams.java:27`_

### getStrategy

```java
public MergeStrategy getStrategy()
```

Gets the merge strategy to use.

**Returns**

The merge strategy

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraMergeParams.java:36`_

### withOutputIndexParams

```java
public Builder withOutputIndexParams(CagraIndexParams outputIndexParams)
```

Sets the index parameters for the output index.

**Parameters**

| Name | Description |
| --- | --- |
| `outputIndexParams` | Index parameters to use for the output index |

**Returns**

This builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraMergeParams.java:68`_

### withStrategy

```java
public Builder withStrategy(MergeStrategy strategy)
```

Sets the merge strategy.

**Parameters**

| Name | Description |
| --- | --- |
| `strategy` | The merge strategy to use |

**Returns**

This builder

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraMergeParams.java:79`_

### build

```java
public CagraMergeParams build()
```

Builds the `CagraMergeParams` object.

**Returns**

The built parameters

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraMergeParams.java:89`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/CagraMergeParams.java:7`_
