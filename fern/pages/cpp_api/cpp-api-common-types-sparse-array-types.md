---
slug: api-reference/cpp-api-common-types-sparse-array-types
---

# Sparse Array Types

Sparse RAFT types describe both the sparsity pattern and the values for APIs that accept sparse feature matrices or graph-style connectivity data.

<a id="raft-device-csr-matrix"></a>
### raft::device_csr_matrix

_Source header: `raft/core/device_csr_matrix.hpp`_

Owning compressed sparse row matrix in device memory.

```cpp
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T> typename ContainerPolicy = device_container_policy,
          SparsityType sparsity_type = SparsityType::OWNING>
using device_csr_matrix =
  csr_matrix<ElementType, IndptrType, IndicesType,
             NZType, true, ContainerPolicy, sparsity_type>;
```

<a id="raft-device-csr-matrix-initialize-sparsity"></a>
#### raft::device_csr_matrix::initialize_sparsity

Initializes or changes the number of nonzero entries when the matrix owns its sparsity.

```cpp
void initialize_sparsity(NNZType nnz);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `nnz` | `NNZType` | Number of nonzero entries. |

**Returns**

`void`

<a id="raft-device-csr-matrix-get-elements"></a>
#### raft::device_csr_matrix::get_elements

Returns a span over the nonzero values.

```cpp
raft::device_span<ElementType> get_elements();
```

**Returns**

`raft::device_span<ElementType>`

<a id="raft-device-csr-matrix-structure-view"></a>
#### raft::device_csr_matrix::structure_view

Returns a non-owning view of the CSR sparsity structure. The returned view exposes the row offsets and column indices.

```cpp
structure_view_type structure_view();
```

**Returns**

`structure_view_type`

<a id="raft-device-csr-matrix-view-method"></a>
#### raft::device_csr_matrix::view

Returns a sparsity-preserving non-owning view of the sparse matrix.

```cpp
view_type view();
```

**Returns**

`view_type`

<a id="raft-device-csr-matrix-view"></a>
### raft::device_csr_matrix_view

_Source header: `raft/core/device_csr_matrix.hpp`_

Non-owning compressed sparse row matrix view over device memory.

```cpp
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType>
using device_csr_matrix_view =
  csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, true>;
```

<a id="raft-device-csr-matrix-view-get-elements"></a>
#### raft::device_csr_matrix_view::get_elements

Returns a span over the nonzero values.

```cpp
raft::device_span<ElementType> get_elements();
```

**Returns**

`raft::device_span<ElementType>`

<a id="raft-device-csr-matrix-view-structure-view"></a>
#### raft::device_csr_matrix_view::structure_view

Returns a non-owning view of the CSR sparsity structure.

```cpp
structure_view_type structure_view();
```

**Returns**

`structure_view_type`

<a id="raft-device-coo-matrix"></a>
### raft::device_coo_matrix

_Source header: `raft/core/device_coo_matrix.hpp`_

Owning coordinate sparse matrix in device memory.

```cpp
template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType,
          template <typename T> typename ContainerPolicy = device_container_policy,
          SparsityType sparsity_type = SparsityType::OWNING>
using device_coo_matrix =
  coo_matrix<ElementType, RowType, ColType,
             NZType, true, ContainerPolicy, sparsity_type>;
```

<a id="raft-device-coo-matrix-initialize-sparsity"></a>
#### raft::device_coo_matrix::initialize_sparsity

Initializes or changes the number of nonzero entries when the matrix owns its sparsity.

```cpp
void initialize_sparsity(NNZType nnz);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `nnz` | `NNZType` | Number of nonzero entries. |

**Returns**

`void`

<a id="raft-device-coo-matrix-get-elements"></a>
#### raft::device_coo_matrix::get_elements

Returns a span over the nonzero values.

```cpp
raft::device_span<ElementType> get_elements();
```

**Returns**

`raft::device_span<ElementType>`

<a id="raft-device-coo-matrix-structure-view"></a>
#### raft::device_coo_matrix::structure_view

Returns a non-owning view of the COO sparsity structure. The returned view exposes the row and column coordinate arrays.

```cpp
structure_view_type structure_view();
```

**Returns**

`structure_view_type`

<a id="raft-device-coo-matrix-view-method"></a>
#### raft::device_coo_matrix::view

Returns a sparsity-preserving non-owning view of the sparse matrix.

```cpp
view_type view();
```

**Returns**

`view_type`

<a id="raft-device-coo-matrix-view"></a>
### raft::device_coo_matrix_view

_Source header: `raft/core/device_coo_matrix.hpp`_

Non-owning coordinate sparse matrix view over device memory.

```cpp
template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType>
using device_coo_matrix_view =
  coo_matrix_view<ElementType, RowType, ColType, NZType, true>;
```

<a id="raft-device-coo-matrix-view-get-elements"></a>
#### raft::device_coo_matrix_view::get_elements

Returns a span over the nonzero values.

```cpp
raft::device_span<ElementType> get_elements();
```

**Returns**

`raft::device_span<ElementType>`

<a id="raft-device-coo-matrix-view-structure-view"></a>
#### raft::device_coo_matrix_view::structure_view

Returns a non-owning view of the COO sparsity structure.

```cpp
structure_view_type structure_view();
```

**Returns**

`structure_view_type`

<a id="raft-make-device-csr-matrix"></a>
### raft::make_device_csr_matrix

_Source header: `raft/core/device_csr_matrix.hpp`_

Allocates an owning CSR matrix.

```cpp
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType = uint64_t>
auto make_device_csr_matrix(raft::resources const& handle,
                            IndptrType n_rows,
                            IndicesType n_cols,
                            NZType nnz = 0);

template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType = uint64_t>
auto make_device_csr_matrix(
  raft::resources const& handle,
  device_compressed_structure_view<IndptrType,
                                   IndicesType,
                                   NZType> structure);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `handle` | `raft::resources const&` | Resources object used for allocation. |
| `n_rows` | `IndptrType` | Number of rows. |
| `n_cols` | `IndicesType` | Number of columns. |
| `nnz` | `NZType` | Number of nonzero entries when known. |
| `structure` | `device_compressed_structure_view<IndptrType, IndicesType, NZType>` | Existing CSR sparsity structure for sparsity-preserving matrices. |

**Returns**

`raft::device_csr_matrix<ElementType, IndptrType, IndicesType, NZType>`

<a id="raft-make-device-coo-matrix"></a>
### raft::make_device_coo_matrix

_Source header: `raft/core/device_coo_matrix.hpp`_

Allocates an owning COO matrix.

```cpp
template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType>
auto make_device_coo_matrix(raft::resources const& handle,
                            RowType n_rows,
                            ColType n_cols,
                            NZType nnz = 0);

template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType>
auto make_device_coo_matrix(
  raft::resources const& handle,
  device_coordinate_structure_view<RowType,
                                   ColType,
                                   NZType> structure);
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `handle` | `raft::resources const&` | Resources object used for allocation. |
| `n_rows` | `RowType` | Number of rows. |
| `n_cols` | `ColType` | Number of columns. |
| `nnz` | `NZType` | Number of nonzero entries when known. |
| `structure` | `device_coordinate_structure_view<RowType, ColType, NZType>` | Existing COO sparsity structure for sparsity-preserving matrices. |

**Returns**

`raft::device_coo_matrix<ElementType, RowType, ColType, NZType>`
