---
slug: api-reference/cpp-api-common-types-dense-array-views
---

# Dense Array Views

Dense array views describe memory owned somewhere else. They carry the pointer, shape, layout, memory-space accessor, and constness needed by NVIDIA cuVS C++ APIs without allocating or freeing data.

<a id="raft-mdspan"></a>
### raft::mdspan

_Source header: `raft/core/mdspan.hpp`_

Generic multi-dimensional non-owning view.

```cpp
template <typename ElementType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
class mdspan;
```

<a id="raft-mdspan-data-handle"></a>
#### raft::mdspan::data_handle

Returns the pointer held by the non-owning view.

```cpp
element_type* data_handle() const;
```

**Returns**

`element_type*`

<a id="raft-mdspan-extents"></a>
#### raft::mdspan::extents

Returns the extents object that describes the view shape.

```cpp
extents_type extents() const;
```

**Returns**

`extents_type`

<a id="raft-mdspan-extent"></a>
#### raft::mdspan::extent

Returns the size of one rank of the view.

```cpp
index_type extent(std::size_t r) const noexcept;
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `r` | `std::size_t` | Rank to query. |

**Returns**

`index_type`

<a id="raft-mdspan-stride"></a>
#### raft::mdspan::stride

Returns the stride for one rank of a strided view.

```cpp
index_type stride(std::size_t r) const;
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `r` | `std::size_t` | Rank to query. |

**Returns**

`index_type`

<a id="raft-mdspan-size"></a>
#### raft::mdspan::size

Returns the total number of elements described by the view.

```cpp
size_type size() const;
```

**Returns**

`size_type`

<a id="raft-mdspan-empty"></a>
#### raft::mdspan::empty

Reports whether the view describes zero elements.

```cpp
bool empty() const;
```

**Returns**

`bool`

<a id="raft-mdspan-operator-call"></a>
#### raft::mdspan::operator()

Indexes into the view with one coordinate per rank.

```cpp
template <typename... IndexType>
reference operator()(IndexType... indices) const;
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `indices` | `IndexType...` | Coordinates into the view, one per rank. |

**Returns**

`reference`

<a id="raft-device-mdspan"></a>
### raft::device_mdspan

_Source header: `raft/core/device_mdspan.hpp`_

Non-owning view over device-accessible memory.

```cpp
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = cuda::std::default_accessor<ElementType>>
using device_mdspan =
  mdspan<ElementType, Extents, LayoutPolicy, device_accessor<AccessorPolicy>>;
```

<a id="raft-host-mdspan"></a>
### raft::host_mdspan

_Source header: `raft/core/host_mdspan.hpp`_

Non-owning view over host memory.

```cpp
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = cuda::std::default_accessor<ElementType>>
using host_mdspan =
  mdspan<ElementType, Extents, LayoutPolicy, host_accessor<AccessorPolicy>>;
```

<a id="raft-device-matrix-view"></a>
### raft::device_matrix_view

_Source header: `raft/core/device_mdspan.hpp`_

Common device view alias for matrix arguments.

```cpp
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using device_matrix_view =
  device_mdspan<ElementType, matrix_extent<IndexType>, LayoutPolicy>;
```

<a id="raft-device-vector-view"></a>
### raft::device_vector_view

_Source header: `raft/core/device_mdspan.hpp`_

Common device view alias for vector arguments.

```cpp
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using device_vector_view =
  device_mdspan<ElementType, vector_extent<IndexType>, LayoutPolicy>;
```

<a id="raft-device-scalar-view"></a>
### raft::device_scalar_view

_Source header: `raft/core/device_mdspan.hpp`_

Common device view alias for scalar arguments.

```cpp
template <typename ElementType, typename IndexType = std::uint32_t>
using device_scalar_view = device_mdspan<ElementType, scalar_extent<IndexType>>;
```

<a id="raft-host-matrix-view"></a>
### raft::host_matrix_view

_Source header: `raft/core/host_mdspan.hpp`_

Common host view alias for matrix arguments.

```cpp
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using host_matrix_view =
  host_mdspan<ElementType, matrix_extent<IndexType>, LayoutPolicy>;
```

<a id="raft-host-vector-view"></a>
### raft::host_vector_view

_Source header: `raft/core/host_mdspan.hpp`_

Common host view alias for vector arguments.

```cpp
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using host_vector_view =
  host_mdspan<ElementType, vector_extent<IndexType>, LayoutPolicy>;
```

<a id="raft-host-scalar-view"></a>
### raft::host_scalar_view

_Source header: `raft/core/host_mdspan.hpp`_

Common host view alias for scalar arguments.

```cpp
template <typename ElementType, typename IndexType = std::uint32_t>
using host_scalar_view = host_mdspan<ElementType, scalar_extent<IndexType>>;
```

<a id="raft-span"></a>
### raft::span

_Source header: `raft/core/span.hpp`_

Lightweight one-dimensional non-owning view. NVIDIA cuVS public APIs usually prefer `device_vector_view` and `host_vector_view` for one-dimensional buffers.

```cpp
template <typename ElementType, std::size_t Extent>
class span;
```

<a id="raft-span-data"></a>
#### raft::span::data

Returns the pointer held by the span.

```cpp
element_type* data() const;
```

**Returns**

`element_type*`

<a id="raft-span-size"></a>
#### raft::span::size

Returns the number of elements in the span.

```cpp
size_type size() const;
```

**Returns**

`size_type`

<a id="raft-span-empty"></a>
#### raft::span::empty

Reports whether the span contains zero elements.

```cpp
bool empty() const;
```

**Returns**

`bool`

<a id="raft-span-operator-subscript"></a>
#### raft::span::operator[]

Indexes into the span.

```cpp
reference operator[](size_type idx) const;
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `idx` | `size_type` | Element index. |

**Returns**

`reference`

<a id="raft-span-begin"></a>
#### raft::span::begin

Returns an iterator to the first element.

```cpp
iterator begin() const;
```

**Returns**

`iterator`

<a id="raft-span-end"></a>
#### raft::span::end

Returns an iterator one past the last element.

```cpp
iterator end() const;
```

**Returns**

`iterator`
