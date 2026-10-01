---
slug: api-reference/cpp-api-common-types-owning-array-factories
---

# Owning Array Factories

<a id="raft-make-device-matrix"></a>
### raft::make_device_matrix

_Source header: `raft/core/device_mdarray.hpp`_

Allocates an owning device matrix.

```cpp
template <typename ElementType, typename IndexType, typename LayoutPolicy>
auto make_device_matrix(raft::resources const& res, IndexType rows, IndexType cols);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `res` | `raft::resources const&` | Resources object used for allocation. |
| `rows` | `IndexType` | Number of rows. |
| `cols` | `IndexType` | Number of columns. |

**Returns**

`raft::device_matrix<ElementType, IndexType, LayoutPolicy>`

<a id="raft-make-device-vector"></a>
### raft::make_device_vector

_Source header: `raft/core/device_mdarray.hpp`_

Allocates an owning device vector.

```cpp
template <typename ElementType, typename IndexType, typename LayoutPolicy>
auto make_device_vector(raft::resources const& res, IndexType size);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `res` | `raft::resources const&` | Resources object used for allocation. |
| `size` | `IndexType` | Number of elements. |

**Returns**

`raft::device_vector<ElementType, IndexType, LayoutPolicy>`

<a id="raft-make-host-matrix"></a>
### raft::make_host_matrix

_Source header: `raft/core/host_mdarray.hpp`_

Allocates an owning host matrix.

```cpp
template <typename ElementType, typename IndexType, typename LayoutPolicy>
auto make_host_matrix(IndexType rows, IndexType cols);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `rows` | `IndexType` | Number of rows. |
| `cols` | `IndexType` | Number of columns. |

**Returns**

`raft::host_matrix<ElementType, IndexType, LayoutPolicy>`

<a id="raft-make-host-vector"></a>
### raft::make_host_vector

_Source header: `raft/core/host_mdarray.hpp`_

Allocates an owning host vector.

```cpp
template <typename ElementType, typename IndexType, typename LayoutPolicy>
auto make_host_vector(IndexType size);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `size` | `IndexType` | Number of elements. |

**Returns**

`raft::host_vector<ElementType, IndexType, LayoutPolicy>`
