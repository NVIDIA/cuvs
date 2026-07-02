---
slug: api-reference/cpp-api-common-types-copy-serialization-and-utility-apis
---

# Copy, Serialization, and Utility APIs

<a id="raft-copy"></a>
### raft::copy

_Source header: `raft/core/copy.hpp`_

Asynchronously copies elements between compatible memory locations.

```cpp
template <typename OutputIterator, typename InputIterator, typename SizeType>
void copy(OutputIterator dst, InputIterator src, SizeType n,
          rmm::cuda_stream_view stream);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `dst` | `OutputIterator` | Destination pointer or iterator. |
| `src` | `InputIterator` | Source pointer or iterator. |
| `n` | `SizeType` | Number of elements to copy. |
| `stream` | `rmm::cuda_stream_view` | CUDA stream used for the copy. |

**Returns**

`void`

<a id="raft-copy-matrix"></a>
### raft::copy_matrix

_Source header: `raft/core/copy.hpp`_

Copies a dense matrix between compatible matrix views.

```cpp
template <typename OutputView, typename InputView>
void copy_matrix(OutputView dst, InputView src, rmm::cuda_stream_view stream);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `dst` | `OutputView` | Destination matrix view. |
| `src` | `InputView` | Source matrix view. |
| `stream` | `rmm::cuda_stream_view` | CUDA stream used for the copy. |

**Returns**

`void`

<a id="raft-update-device"></a>
### raft::update_device

_Source header: `raft/core/copy.hpp`_

Convenience helper for copying host data to device memory.

```cpp
template <typename DevicePointer, typename HostPointer, typename SizeType>
void update_device(DevicePointer dst, HostPointer src, SizeType n,
                   rmm::cuda_stream_view stream);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `dst` | `DevicePointer` | Destination device pointer. |
| `src` | `HostPointer` | Source host pointer. |
| `n` | `SizeType` | Number of elements to copy. |
| `stream` | `rmm::cuda_stream_view` | CUDA stream used for the copy. |

**Returns**

`void`

<a id="raft-update-host"></a>
### raft::update_host

_Source header: `raft/core/copy.hpp`_

Convenience helper for copying device data to host memory.

```cpp
template <typename HostPointer, typename DevicePointer, typename SizeType>
void update_host(HostPointer dst, DevicePointer src, SizeType n,
                 rmm::cuda_stream_view stream);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `dst` | `HostPointer` | Destination host pointer. |
| `src` | `DevicePointer` | Source device pointer. |
| `n` | `SizeType` | Number of elements to copy. |
| `stream` | `rmm::cuda_stream_view` | CUDA stream used for the copy. |

**Returns**

`void`

<a id="raft-serialize-mdspan"></a>
### raft::serialize_mdspan

_Source header: `raft/core/serialize.hpp`_

Serializes array contents from an mdspan-like view.

```cpp
template <typename View>
void serialize_mdspan(std::ostream& os, View view);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `os` | `std::ostream&` | Output stream. |
| `view` | `View` | View whose contents should be serialized. |

**Returns**

`void`

<a id="raft-deserialize-mdspan"></a>
### raft::deserialize_mdspan

_Source header: `raft/core/serialize.hpp`_

Deserializes array contents into an mdspan-like view.

```cpp
template <typename View>
void deserialize_mdspan(std::istream& is, View view);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `is` | `std::istream&` | Input stream. |
| `view` | `View` | Destination view. |

**Returns**

`void`

<a id="raft-ceildiv"></a>
### raft::ceildiv

_Source header: `raft/util/integer_utils.hpp`_

Computes integer division rounded up.

```cpp
template <typename T>
T ceildiv(T dividend, T divisor);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `dividend` | `T` | Dividend. |
| `divisor` | `T` | Divisor. |

**Returns**

`T`

<a id="raft-round-up-safe"></a>
### raft::round_up_safe

_Source header: `raft/util/integer_utils.hpp`_

Rounds an integer value up to a requested multiple.

```cpp
template <typename T>
T round_up_safe(T value, T multiple);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `value` | `T` | Value to round. |
| `multiple` | `T` | Multiple to round to. |

**Returns**

`T`
