---
slug: api-reference/java-api-com-nvidia-cuvs-rowview
---

# RowView

_Java package: `com.nvidia.cuvs`_

```java
public interface RowView
```

Represent a contiguous list of elements backed by off-heap memory.

## Public Members

### getAsInt

```java
int getAsInt(long index)
```

Returns the integer element at the given position. Asserts that the
data type of the dataset on top of which this view is instantiates is
`CuVSMatrix.DataType#INT`

**Parameters**

| Name | Description |
| --- | --- |
| `index` | the element index |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/RowView.java:22`_

### getAsFloat

```java
float getAsFloat(long index)
```

Returns the integer element at the given position. Asserts that the
data type of the dataset on top of which this view is instantiates is
`CuVSMatrix.DataType#FLOAT`

**Parameters**

| Name | Description |
| --- | --- |
| `index` | the element index |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/RowView.java:31`_

### getAsByte

```java
byte getAsByte(long index)
```

Returns the integer element at the given position. Asserts that the
data type of the dataset on top of which this view is instantiates is
`CuVSMatrix.DataType#BYTE`

**Parameters**

| Name | Description |
| --- | --- |
| `index` | the element index |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/RowView.java:40`_

### toArray

```java
void toArray(int[] array)
```

Copies the content of this row to an on-heap Java array.

**Parameters**

| Name | Description |
| --- | --- |
| `array` | the destination array. Must be of length `RowView#size()` or bigger. |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/RowView.java:47`_

### toArray

```java
void toArray(float[] array)
```

Copies the content of this row to an on-heap Java array.

**Parameters**

| Name | Description |
| --- | --- |
| `array` | the destination array. Must be of length `RowView#size()` or bigger. |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/RowView.java:54`_

### toArray

```java
void toArray(byte[] array)
```

Copies the content of this row to an on-heap Java array.

**Parameters**

| Name | Description |
| --- | --- |
| `array` | the destination array. Must be of length `RowView#size()` or bigger. |

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/RowView.java:61`_

_Source: `java/cuvs-java/src/main/java/com/nvidia/cuvs/RowView.java:12`_
