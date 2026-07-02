---
slug: api-reference/cpp-api-common-types-owning-dense-arrays
---

# Owning Dense Arrays

Owning arrays allocate storage and release it when the object is destroyed. They are commonly used in examples, tests, index objects, and user code that needs RAFT to allocate inputs, outputs, or staging buffers.

<a id="raft-mdarray"></a>
### raft::mdarray

_Source header: `raft/core/mdarray.hpp`_

Generic owning multi-dimensional array.

```cpp
template <typename ElementType, typename Extents, typename LayoutPolicy,
          typename ContainerPolicy>
class mdarray;
```

<a id="raft-mdarray-view"></a>
#### raft::mdarray::view

Returns an mdspan view over the owned storage.

```cpp
mdspan_type view();
const_mdspan_type view() const;
```

**Returns**

`mdspan_type or const_mdspan_type`

<a id="raft-mdarray-data-handle"></a>
#### raft::mdarray::data_handle

Returns the pointer to the owned storage.

```cpp
element_type* data_handle();
element_type const* data_handle() const;
```

**Returns**

`element_type* or element_type const*`

<a id="raft-mdarray-extents"></a>
#### raft::mdarray::extents

Returns the extents object that describes the array shape.

```cpp
extents_type extents() const;
```

**Returns**

`extents_type`

<a id="raft-mdarray-extent"></a>
#### raft::mdarray::extent

Returns the size of one rank of the array.

```cpp
index_type extent(std::size_t r) const noexcept;
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `r` | `std::size_t` | Rank to query. |

**Returns**

`index_type`

<a id="raft-mdarray-size"></a>
#### raft::mdarray::size

Returns the total number of elements in the array.

```cpp
size_type size() const;
```

**Returns**

`size_type`

<a id="raft-mdarray-operator-call"></a>
#### raft::mdarray::operator()

Indexes into the array. For device arrays, use this sparingly because element access may require device-host movement.

```cpp
template <typename... IndexType>
reference operator()(IndexType... indices);
template <typename... IndexType>
const_reference operator()(IndexType... indices) const;
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `indices` | `IndexType...` | Coordinates into the array, one per rank. |

**Returns**

`reference or const_reference`

<a id="raft-device-mdarray"></a>
### raft::device_mdarray

_Source header: `raft/core/device_mdarray.hpp`_

Owning array in device-accessible memory.

```cpp
template <typename ElementType, typename Extents,
          typename LayoutPolicy    = layout_c_contiguous,
          typename ContainerPolicy = device_container_policy<ElementType>>
using device_mdarray =
  mdarray<ElementType, Extents, LayoutPolicy, device_accessor<ContainerPolicy>>;
```

<a id="raft-host-mdarray"></a>
### raft::host_mdarray

_Source header: `raft/core/host_mdarray.hpp`_

Owning array in host memory.

```cpp
template <typename ElementType, typename Extents,
          typename LayoutPolicy    = layout_c_contiguous,
          typename ContainerPolicy = host_container_policy<ElementType>>
using host_mdarray =
  mdarray<ElementType, Extents, LayoutPolicy, host_accessor<ContainerPolicy>>;
```

<a id="raft-device-matrix"></a>
### raft::device_matrix

_Source header: `raft/core/device_mdarray.hpp`_

Owning device matrix alias used for datasets, outputs, and temporary storage.

```cpp
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using device_matrix = device_mdarray<ElementType, matrix_extent<IndexType>, LayoutPolicy>;
```

<a id="raft-device-vector"></a>
### raft::device_vector

_Source header: `raft/core/device_mdarray.hpp`_

Owning device vector alias used for outputs and temporary storage.

```cpp
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using device_vector = device_mdarray<ElementType, vector_extent<IndexType>, LayoutPolicy>;
```

<a id="raft-host-matrix"></a>
### raft::host_matrix

_Source header: `raft/core/host_mdarray.hpp`_

Owning host matrix alias used for CPU-resident data and staging.

```cpp
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using host_matrix = host_mdarray<ElementType, matrix_extent<IndexType>, LayoutPolicy>;
```

<a id="raft-host-vector"></a>
### raft::host_vector

_Source header: `raft/core/host_mdarray.hpp`_

Owning host vector alias used for CPU-resident data and staging.

```cpp
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using host_vector = host_mdarray<ElementType, vector_extent<IndexType>, LayoutPolicy>;
```
