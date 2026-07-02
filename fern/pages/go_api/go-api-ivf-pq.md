---
slug: api-reference/go-api-ivf-pq
---

# Ivf Pq Package

_Go package: `ivf_pq`_

_Sources: `go/ivf_pq`_

## Constants

### InternalDistance_Float32 Constants

```go
const (
InternalDistance_Float32 internalDistanceDtype = iota
InternalDistance_Float64
)
```

_Source: `go/ivf_pq/search_params.go:43`_

### Lut_Uint8 Constants

```go
const (
Lut_Uint8 lutDtype = iota
Lut_Uint16
Lut_Uint32
Lut_Uint64
Lut_Int8
Lut_Int16
Lut_Int32
Lut_Int64
)
```

_Source: `go/ivf_pq/search_params.go:19`_

### Subspace Constants

```go
const (
Subspace codebookKind = iota
Cluster
)
```

_Source: `go/ivf_pq/index_params.go:18`_

## Variables

### CInternalDistanceDtypes

```go
var CInternalDistanceDtypes = map[internalDistanceDtype]int{
```

_Source: `go/ivf_pq/search_params.go:48`_

## Types

### IndexParams

```go
type IndexParams struct {
	// contains filtered or unexported fields
}
```

_Source: `go/ivf_pq/index_params.go:12`_

### IvfPqIndex

```go
type IvfPqIndex struct {
	// contains filtered or unexported fields
}
```

IVF PQ Index

_Source: `go/ivf_pq/ivf_pq.go:14`_

### SearchParams

```go
type SearchParams struct {
	// contains filtered or unexported fields
}
```

Supplemental parameters to search IVF PQ Index

_Source: `go/ivf_pq/search_params.go:13`_

## Functions

### BuildIndex

```go
func BuildIndex[T any](Resources cuvs.Resource, params *IndexParams, dataset *cuvs.Tensor[T], index *IvfPqIndex) error
```

Builds an IvfPqIndex from the dataset for efficient search.

#### Arguments

* `Resources` - Resources to use
* `params` - Parameters for building the index
* `dataset` - A row-major Tensor on either the host or device to index
* `index` - IvfPqIndex to build

_Source: `go/ivf_pq/ivf_pq.go:39`_

### CreateIndex

```go
func CreateIndex(params *IndexParams, dataset *cuvs.Tensor[float32]) (*IvfPqIndex, error)
```

Creates a new empty IvfPqIndex

_Source: `go/ivf_pq/ivf_pq.go:20`_

### CreateIndexParams

```go
func CreateIndexParams() (*IndexParams, error)
```

Creates a new IndexParams

_Source: `go/ivf_pq/index_params.go:29`_

### CreateSearchParams

```go
func CreateSearchParams() (*SearchParams, error)
```

Creates a new SearchParams

_Source: `go/ivf_pq/search_params.go:54`_

### SearchIndex

```go
func SearchIndex[T any](Resources cuvs.Resource, params *SearchParams, index *IvfPqIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[int64], distances *cuvs.Tensor[T]) error
```

Perform a Approximate Nearest Neighbors search on the Index

#### Arguments

* `Resources` - Resources to use
* `params` - Parameters to use in searching the index
* `index` - IvfPqIndex to search
* `queries` - A tensor in device memory to query for
* `neighbors` - Tensor in device memory that receives the indices of the nearest neighbors
* `distances` - Tensor in device memory that receives the distances of the nearest neighbors

_Source: `go/ivf_pq/ivf_pq.go:67`_

## Methods

### IndexParams.Close

```go
func (p *IndexParams) Close() error
```

Destroys IndexParams

_Source: `go/ivf_pq/index_params.go:143`_

### IndexParams.SetAddDataOnBuild

```go
func (p *IndexParams) SetAddDataOnBuild(add_data_on_build bool) (*IndexParams, error)
```

After training the coarse and fine quantizers, we will populate
the index with the dataset if add_data_on_build == true, otherwise
the index is left empty, and the extend method can be used
to add new vectors to the index.

_Source: `go/ivf_pq/index_params.go:133`_

### IndexParams.SetCodebookKind

```go
func (p *IndexParams) SetCodebookKind(codebook_kind codebookKind) (*IndexParams, error)
```

_Source: `go/ivf_pq/index_params.go:98`_

### IndexParams.SetForceRandomRotation

```go
func (p *IndexParams) SetForceRandomRotation(force_random_rotation bool) (*IndexParams, error)
```

Apply a random rotation matrix on the input data and queries even
if `dim % pq_dim == 0`. Note: if `dim` is not multiple of `pq_dim`,
a random rotation is always applied to the input data and queries
to transform the working space from `dim` to `rot_dim`, which may
be slightly larger than the original space and and is a multiple
of `pq_dim` (`rot_dim % pq_dim == 0`). However, this transform is
not necessary when `dim` is multiple of `pq_dim` (`dim == rot_dim`,
hence no need in adding "extra" data columns / features). By
default, if `dim == rot_dim`, the rotation transform is
initialized with the identity matrix. When
`force_random_rotation == True`, a random orthogonal transform

_Source: `go/ivf_pq/index_params.go:120`_

### IndexParams.SetKMeansNIters

```go
func (p *IndexParams) SetKMeansNIters(kmeans_n_iters uint32) (*IndexParams, error)
```

The number of iterations searching for kmeans centers during index building.

_Source: `go/ivf_pq/index_params.go:65`_

### IndexParams.SetKMeansTrainsetFraction

```go
func (p *IndexParams) SetKMeansTrainsetFraction(kmeans_trainset_fraction float64) (*IndexParams, error)
```

If kmeans_trainset_fraction is less than 1, then the dataset is
subsampled, and only n_samples * kmeans_trainset_fraction rows
are used for training.

_Source: `go/ivf_pq/index_params.go:73`_

### IndexParams.SetMetric

```go
func (p *IndexParams) SetMetric(metric cuvs.Distance) (*IndexParams, error)
```

Distance Type to use for building the index

_Source: `go/ivf_pq/index_params.go:47`_

### IndexParams.SetMetricArg

```go
func (p *IndexParams) SetMetricArg(metric_arg float32) (*IndexParams, error)
```

Metric argument for Minkowski distances - set to 2.0 if not applicable

_Source: `go/ivf_pq/index_params.go:59`_

### IndexParams.SetNLists

```go
func (p *IndexParams) SetNLists(n_lists uint32) (*IndexParams, error)
```

The number of clusters used in the coarse quantizer.

_Source: `go/ivf_pq/index_params.go:41`_

### IndexParams.SetPQBits

```go
func (p *IndexParams) SetPQBits(pq_bits uint32) (*IndexParams, error)
```

The bit length of the vector element after quantization.

_Source: `go/ivf_pq/index_params.go:79`_

### IndexParams.SetPQDim

```go
func (p *IndexParams) SetPQDim(pq_dim uint32) (*IndexParams, error)
```

The dimensionality of a the vector after product quantization.
When zero, an optimal value is selected using a heuristic. Note
pq_dim * pq_bits must be a multiple of 8. Hint: a smaller 'pq_dim'
results in a smaller index size and better search performance, but
lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number,
but multiple of 8 are desirable for good performance. If 'pq_bits'
is not 8, 'pq_dim' should be a multiple of 8. For good performance,
it is desirable that 'pq_dim' is a multiple of 32. Ideally,
'pq_dim' should be also a divisor of the dataset dim.

_Source: `go/ivf_pq/index_params.go:93`_

### IvfPqIndex.Close

```go
func (index *IvfPqIndex) Close() error
```

Destroys the IvfPqIndex

_Source: `go/ivf_pq/ivf_pq.go:49`_

### SearchParams.Close

```go
func (p *SearchParams) Close() error
```

Destroys SearchParams

_Source: `go/ivf_pq/search_params.go:101`_

### SearchParams.SetInternalDistanceDtype

```go
func (p *SearchParams) SetInternalDistanceDtype(internal_distance_dtype internalDistanceDtype) (*SearchParams, error)
```

Storage data type for distance/similarity computation.

_Source: `go/ivf_pq/search_params.go:89`_

### SearchParams.SetLutDtype

```go
func (p *SearchParams) SetLutDtype(lut_dtype lutDtype) (*SearchParams, error)
```

Data type of look up table to be created dynamically at search
time. The use of low-precision types reduces the amount of shared
memory required at search time, so fast shared memory kernels can
be used even for datasets with large dimansionality. Note that
the recall is slightly degraded when low-precision type is
selected.

_Source: `go/ivf_pq/search_params.go:77`_

### SearchParams.SetNProbes

```go
func (p *SearchParams) SetNProbes(n_probes uint32) (*SearchParams, error)
```

The number of clusters to search.

_Source: `go/ivf_pq/search_params.go:66`_
