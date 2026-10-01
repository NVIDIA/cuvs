---
slug: api-reference/cpp-api-common-types-layouts-and-extents
---

# Layouts and Extents

RAFT layout and extent types make array shape and memory layout explicit in NVIDIA cuVS public signatures.

<a id="raft-row-major"></a>
### raft::row_major

_Source header: `raft/core/mdspan_types.hpp`_

Matrix layout tag for row-major storage.

```cpp
using cuda::std::layout_right;
using row_major = layout_right;
```

<a id="raft-col-major"></a>
### raft::col_major

_Source header: `raft/core/mdspan_types.hpp`_

Matrix layout tag for column-major storage.

```cpp
using cuda::std::layout_left;
using col_major = layout_left;
```

<a id="raft-layout-c-contiguous"></a>
### raft::layout_c_contiguous

_Source header: `raft/core/mdspan_types.hpp`_

Layout tag for C-contiguous memory.

```cpp
using cuda::std::layout_right;
using layout_c_contiguous = layout_right;
```

<a id="raft-layout-f-contiguous"></a>
### raft::layout_f_contiguous

_Source header: `raft/core/mdspan_types.hpp`_

Layout tag for Fortran-contiguous memory.

```cpp
using cuda::std::layout_left;
using layout_f_contiguous = layout_left;
```

<a id="raft-layout-stride"></a>
### raft::layout_stride

_Source header: `raft/core/mdspan_types.hpp`_

Layout tag for strided memory.

```cpp
using cuda::std::layout_stride;
```

<a id="raft-scalar-extent"></a>
### raft::scalar_extent

_Source header: `raft/core/mdspan_types.hpp`_

Convenience extent alias for zero-dimensional scalar values.

```cpp
template <typename IndexType>
using scalar_extent = cuda::std::extents<IndexType, 1>;
```

<a id="raft-matrix-extent"></a>
### raft::matrix_extent

_Source header: `raft/core/mdspan_types.hpp`_

Convenience extent alias for two-dimensional matrices.

```cpp
template <typename IndexType>
using matrix_extent = cuda::std::extents<IndexType,
                                         raft::dynamic_extent,
                                         raft::dynamic_extent>;
```

<a id="raft-vector-extent"></a>
### raft::vector_extent

_Source header: `raft/core/mdspan_types.hpp`_

Convenience extent alias for one-dimensional vectors.

```cpp
template <typename IndexType>
using vector_extent = cuda::std::extents<IndexType,
                                         raft::dynamic_extent>;
```

<a id="raft-extents"></a>
### raft::extents

_Source header: `raft/core/mdspan_types.hpp`_

Generic extent descriptor for static and dynamic dimensions.

```cpp
using cuda::std::extents;
```

<a id="raft-dynamic-extent"></a>
### raft::dynamic_extent

_Source header: `raft/core/mdspan_types.hpp`_

Sentinel used for dimensions whose size is known at runtime.

```cpp
using cuda::std::dynamic_extent;
```
