---
slug: api-reference/cpp-api-common-types-dense-view-factories
---

# Dense View Factories

<a id="raft-make-device-matrix-view"></a>
### raft::make_device_matrix_view

_Source header: `raft/core/device_mdspan.hpp`_

Constructs a device matrix view from a pointer and shape.

```cpp
template <typename ElementType, typename IndexType, typename LayoutPolicy>
auto make_device_matrix_view(ElementType* ptr, IndexType rows, IndexType cols);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `ptr` | `ElementType*` | Pointer to device-accessible matrix storage. |
| `rows` | `IndexType` | Number of rows. |
| `cols` | `IndexType` | Number of columns. |

**Returns**

`raft::device_matrix_view<ElementType, IndexType, LayoutPolicy>`

<a id="raft-make-device-vector-view"></a>
### raft::make_device_vector_view

_Source header: `raft/core/device_mdspan.hpp`_

Constructs a device vector view from a pointer and size.

```cpp
template <typename ElementType, typename IndexType, typename LayoutPolicy>
auto make_device_vector_view(ElementType* ptr, IndexType size);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `ptr` | `ElementType*` | Pointer to device-accessible vector storage. |
| `size` | `IndexType` | Number of elements. |

**Returns**

`raft::device_vector_view<ElementType, IndexType, LayoutPolicy>`

<a id="raft-make-host-matrix-view"></a>
### raft::make_host_matrix_view

_Source header: `raft/core/host_mdspan.hpp`_

Constructs a host matrix view from a pointer and shape.

```cpp
template <typename ElementType, typename IndexType, typename LayoutPolicy>
auto make_host_matrix_view(ElementType* ptr, IndexType rows, IndexType cols);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `ptr` | `ElementType*` | Pointer to host matrix storage. |
| `rows` | `IndexType` | Number of rows. |
| `cols` | `IndexType` | Number of columns. |

**Returns**

`raft::host_matrix_view<ElementType, IndexType, LayoutPolicy>`

<a id="raft-make-host-vector-view"></a>
### raft::make_host_vector_view

_Source header: `raft/core/host_mdspan.hpp`_

Constructs a host vector view from a pointer and size.

```cpp
template <typename ElementType, typename IndexType, typename LayoutPolicy>
auto make_host_vector_view(ElementType* ptr, IndexType size);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `ptr` | `ElementType*` | Pointer to host vector storage. |
| `size` | `IndexType` | Number of elements. |

**Returns**

`raft::host_vector_view<ElementType, IndexType, LayoutPolicy>`

<a id="raft-make-const-mdspan"></a>
### raft::make_const_mdspan

_Source header: `raft/core/mdspan.hpp`_

Converts a mutable mdspan-like view into a const view.

```cpp
template <typename View>
auto make_const_mdspan(View view);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `view` | `View` | View to convert to a const view. |

**Returns**

`View with const element type`
