---
slug: api-reference/cpp-api-core-dataset-sparse
---

# Sparse

_Source header: `cuvs/core/dataset/sparse.hpp`_

## Types

<a id="sparse-spec"></a>
### sparse_spec

Sparse dataset specification implemented via the raft sparse matrix hierarchy. `SparseLayoutT`

selects the representation (`csr_layout`, `coo_layout`).

NOTE: only the memory space of the container policy is used here; raft's sparse types bind the policy themselves, separately for the values, the offsets and the indices.

```cpp
template <typename SparseLayoutT, typename ContainerPolicy>
struct sparse_spec;
```

<a id="csr-layout"></a>
### csr_layout

CSR representation: one compressed row per vector of the dataset.

NOTE: the row offsets are 64-bit for the same reason the dense extents are: the number of non-zeros is not bounded by the index type. The column indices are feature ids, so they are bounded by the dataset dimension rather than by its size.

```cpp
struct csr_layout;
```

<a id="coo-layout"></a>
### coo_layout

COO representation: one (row, column, value) triple per non-zero.

NOTE: unlike CSR, the row component is an index into the dataset rather than an offset into the non-zeros, hence the dataset index type.

```cpp
struct coo_layout;
```
