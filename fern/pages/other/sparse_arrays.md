# Sparse Arrays

Sparse arrays store only the entries that are present, rather than storing every value in a mostly empty matrix. They are useful for high-dimensional feature vectors, graph connectivity, and other data where most positions are zero.

Most NVIDIA cuVS APIs use dense arrays, and NVIDIA cuVS contains growing support for sparse representations and sparse inputs. For dense datasets, queries, and outputs, see [Dense Arrays](/user-guide/api-guides/core-types/array-types/dense-arrays).

<Note>Use the specific API page to confirm whether sparse input is supported. Passing a sparse matrix to an API that expects a dense matrix is not valid, even when the logical shape is the same.</Note>

## Sparse formats

NVIDIA cuVS sparse C++ APIs commonly use two formats:

- Compressed Sparse Row (CSR): stores one row at a time using row offsets, column indices, and values. CSR is a natural fit for sparse feature matrices where each row is a vector.
- Coordinate (COO): stores each nonzero entry as a row index, column index, and value. COO is a natural fit for graph edges and connectivity matrices.

## Sparse arrays in C++

Sparse arrays are usually described in two pieces:

- Sparsity: the structure that says which entries exist.
- Values: the stored data for those entries.

The sparsity is not the same thing as the sparse matrix. For example, a CSR matrix has row offsets and column indices that describe the sparsity pattern, plus a values buffer that stores the nonzero values. Two sparse matrices can share the same sparsity pattern while storing different values.

### Sparsity ownership

This distinction matters for ownership. Sparsity-owning types allocate and own the structural buffers that describe the sparse pattern, such as CSR row offsets and column indices or COO row and column coordinates. Sparsity-preserving views keep that structure owned somewhere else and only describe it to NVIDIA cuVS or RAFT.

Both forms are useful:

- Use an owning sparse matrix when RAFT or NVIDIA cuVS should allocate the sparse structure and values.
- Use a sparse view when the row offsets, indices, coordinates, or values already exist in device memory.
- Use a shape-only owning sparse matrix when an algorithm will discover the sparsity pattern. This lets the algorithm allocate the required sparse buffers while the completed matrix is still owned by the caller after the algorithm returns.

### Creating CSR matrices

Use CSR for sparse row-major matrices. CSR stores `n_rows + 1` row offsets, `nnz` column indices, and `nnz` values.

#### Owning CSR with known sparsity

```cpp
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/resources.hpp>

raft::device_resources res;

int n_rows = 100000;
int n_cols = 10000;
int nnz = estimate_or_load_nnz();

auto csr = raft::make_device_csr_matrix<float, int, int, int>(
    res, n_rows, n_cols, nnz);

fill_csr_values_and_structure(csr);

auto csr_view = csr.view();
```

#### Owning CSR with sparsity discovered later

Sometimes the shape of a sparse matrix is known, but the exact number of nonzero entries is not known until an algorithm runs. In that case, create the owning matrix with only the logical shape and initialize the sparsity once `nnz` is known.

```cpp
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/resources.hpp>

raft::device_resources res;

int n_rows = 100000;
int n_cols = 10000;

auto csr = raft::make_device_csr_matrix<float, int, int, int>(
    res, n_rows, n_cols);

int nnz = compute_or_count_nonzeros();
csr.initialize_sparsity(nnz);

fill_csr_values_and_structure(csr);
```

This pattern is important when the caller needs to own the final matrix, but the algorithm is the only code that can know how much sparse storage is needed.

#### CSR view over existing buffers

Use a CSR view when values, row offsets, and column indices are already allocated on the device. The view does not own those buffers.

```cpp
#include <raft/core/device_csr_matrix.hpp>

int n_rows = get_n_rows();
int n_cols = get_n_cols();
int nnz = get_nnz();

float const* values = get_device_values();
int const* row_offsets = get_device_row_offsets();
int const* column_indices = get_device_column_indices();

auto structure = raft::make_device_compressed_structure_view<int, int, int>(
    row_offsets,
    column_indices,
    n_rows,
    n_cols,
    nnz);

auto csr = raft::make_device_csr_matrix_view<const float>(
    values,
    structure);
```

### Creating COO matrices

Use COO for sparse coordinate data. COO stores `nnz` row indices, `nnz` column indices, and `nnz` values.

#### Owning COO with known sparsity

Use an owning COO matrix when the shape and number of entries are known up front.

```cpp
#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/resources.hpp>

raft::device_resources res;

int n_rows = 100000;
int n_cols = 100000;
int nnz = 1200000;

auto coo = raft::make_device_coo_matrix<float, int, int, int>(
    res, n_rows, n_cols, nnz);

fill_coo_values_and_structure(coo);

auto coo_view = coo.view();
```

#### Owning COO with sparsity discovered later

Some algorithms produce sparse coordinate data but do not know `nnz` until they run. In those cases, create an owning COO matrix with the logical shape and let the algorithm initialize the sparse structure inside the caller-owned object.

```cpp
#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/resources.hpp>

raft::device_resources res;

int n_rows = 100000;
int n_cols = 100000;

auto coo = raft::make_device_coo_matrix<float, int, int, int>(
    res, n_rows, n_cols);

produce_sparse_coordinates(res, coo);

auto nnz = coo.structure_view().get_nnz();
```

#### COO view over existing buffers

Use a COO view when row indices, column indices, and values already exist in device memory. The view preserves the existing sparsity ownership.

```cpp
#include <raft/core/device_coo_matrix.hpp>

int n_rows = get_n_rows();
int n_cols = get_n_cols();
int nnz = get_nnz();

float const* values = get_device_values();
int const* rows = get_device_rows();
int const* cols = get_device_cols();

auto structure = raft::make_device_coordinate_structure_view<int, int, int>(
    rows,
    cols,
    n_rows,
    n_cols,
    nnz);

auto coo = raft::make_device_coo_matrix_view<const float>(
    values,
    structure);
```

### Choosing C++ sparse array types

| Type | Ownership model | Typical use |
| --- | --- | --- |
| `raft::device_csr_matrix_view` | Preserves external CSR sparsity and values | GPU CSR inputs already allocated by RAFT, RMM, or another CUDA-aware library. |
| `raft::device_csr_matrix` | Owns CSR sparsity and values | Owning GPU CSR matrices when `nnz` is known up front or discovered later with `initialize_sparsity()`. |
| `raft::device_coo_matrix_view` | Preserves external COO sparsity and values | GPU COO connectivity data already allocated elsewhere. |
| `raft::device_coo_matrix` | Owns COO sparsity and values | Owning GPU COO matrices when coordinate data should be allocated and owned by the caller. |
| `raft::device_compressed_structure_view` | Preserves external CSR sparsity only | CSR row-offset and column-index structure used to construct a CSR matrix view. |
| `raft::device_coordinate_structure_view` | Preserves external COO sparsity only | COO row-index and column-index structure used to construct a COO matrix view. |

## Using sparse arrays safely

Check the API page before using sparse inputs. Sparse support is not interchangeable with dense support, and each sparse API documents the expected format.

Keep the values, row offsets, column indices, and COO coordinate buffers alive for as long as a sparse view is used. Sparse views do not own memory.

Make sure the logical dimensions match across inputs. For example, two sparse matrices used together must agree on the dimensions expected by the API.

Remember that sparse inputs can still produce dense outputs. Check each API guide for the expected output format and allocation requirements.
