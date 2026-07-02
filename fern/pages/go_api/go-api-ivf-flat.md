---
slug: api-reference/go-api-ivf-flat
---

# Ivf Flat Package

_Go package: `ivf_flat`_

_Sources: `go/ivf_flat`_

## Types

### IndexParams

```go
type IndexParams struct {
	// contains filtered or unexported fields
}
```

Supplemental parameters to build IVF Flat Index

_Source: `go/ivf_flat/index_params.go:13`_

### IvfFlatIndex

```go
type IvfFlatIndex struct {
	// contains filtered or unexported fields
}
```

IVF Flat Index

_Source: `go/ivf_flat/ivf_flat.go:14`_

### SearchParams

```go
type SearchParams struct {
	// contains filtered or unexported fields
}
```

_Source: `go/ivf_flat/search_params.go:10`_

## Functions

### BuildIndex

```go
func BuildIndex[T any](Resources cuvs.Resource, params *IndexParams, dataset *cuvs.Tensor[T], index *IvfFlatIndex) error
```

Builds an IvfFlatIndex from the dataset for efficient search.

#### Arguments

* `Resources` - Resources to use
* `params` - Parameters for building the index
* `dataset` - A row-major Tensor on either the host or device to index
* `index` - IvfFlatIndex to build

_Source: `go/ivf_flat/ivf_flat.go:38`_

### CreateIndex

```go
func CreateIndex[T any](params *IndexParams, dataset *cuvs.Tensor[T]) (*IvfFlatIndex, error)
```

Creates a new empty IvfFlatIndex

_Source: `go/ivf_flat/ivf_flat.go:20`_

### CreateIndexParams

```go
func CreateIndexParams() (*IndexParams, error)
```

Creates a new IndexParams

_Source: `go/ivf_flat/index_params.go:18`_

### CreateSearchParams

```go
func CreateSearchParams() (*SearchParams, error)
```

Creates a new SearchParams

_Source: `go/ivf_flat/search_params.go:15`_

### GetCenters

```go
func GetCenters[T any](index *IvfFlatIndex, centers *cuvs.Tensor[T]) error
```

_Source: `go/ivf_flat/ivf_flat.go:108`_

### GetDim

```go
func GetDim(index *IvfFlatIndex) (dim int64, err error)
```

_Source: `go/ivf_flat/ivf_flat.go:93`_

### GetNLists

```go
func GetNLists(index *IvfFlatIndex) (nlist int64, err error)
```

_Source: `go/ivf_flat/ivf_flat.go:78`_

### SearchIndex

```go
func SearchIndex[T any](Resources cuvs.Resource, params *SearchParams, index *IvfFlatIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[int64], distances *cuvs.Tensor[T]) error
```

Perform a Approximate Nearest Neighbors search on the Index

#### Arguments

* `Resources` - Resources to use
* `params` - Parameters to use in searching the index
* `index` - IvfFlatIndex to search
* `queries` - A tensor in device memory to query for
* `neighbors` - Tensor in device memory that receives the indices of the nearest neighbors
* `distances` - Tensor in device memory that receives the distances of the nearest neighbors

_Source: `go/ivf_flat/ivf_flat.go:66`_

## Methods

### IndexParams.Close

```go
func (p *IndexParams) Close() error
```

Destroys IndexParams

_Source: `go/ivf_flat/index_params.go:81`_

### IndexParams.SetAddDataOnBuild

```go
func (p *IndexParams) SetAddDataOnBuild(add_data_on_build bool) (*IndexParams, error)
```

After training the coarse and fine quantizers, we will populate
the index with the dataset if add_data_on_build == true, otherwise
the index is left empty, and the extend method can be used
to add new vectors to the index.

_Source: `go/ivf_flat/index_params.go:71`_

### IndexParams.SetKMeansNIters

```go
func (p *IndexParams) SetKMeansNIters(kmeans_n_iters uint32) (*IndexParams, error)
```

The number of iterations searching for kmeans centers during index building.

_Source: `go/ivf_flat/index_params.go:54`_

### IndexParams.SetKMeansTrainsetFraction

```go
func (p *IndexParams) SetKMeansTrainsetFraction(kmeans_trainset_fraction float64) (*IndexParams, error)
```

If kmeans_trainset_fraction is less than 1, then the dataset is
subsampled, and only n_samples * kmeans_trainset_fraction rows
are used for training.

_Source: `go/ivf_flat/index_params.go:62`_

### IndexParams.SetMetric

```go
func (p *IndexParams) SetMetric(metric cuvs.Distance) (*IndexParams, error)
```

Distance Type to use for building the index

_Source: `go/ivf_flat/index_params.go:36`_

### IndexParams.SetMetricArg

```go
func (p *IndexParams) SetMetricArg(metric_arg float32) (*IndexParams, error)
```

Metric argument for Minkowski distances - set to 2.0 if not applicable

_Source: `go/ivf_flat/index_params.go:48`_

### IndexParams.SetNLists

```go
func (p *IndexParams) SetNLists(n_lists uint32) (*IndexParams, error)
```

The number of clusters used in the coarse quantizer.

_Source: `go/ivf_flat/index_params.go:30`_

### IvfFlatIndex.Close

```go
func (index *IvfFlatIndex) Close() error
```

Destroys the IvfFlatIndex

_Source: `go/ivf_flat/ivf_flat.go:48`_

### SearchParams.Close

```go
func (p *SearchParams) Close() error
```

Destroy SearchParams

_Source: `go/ivf_flat/search_params.go:33`_

### SearchParams.SetNProbes

```go
func (p *SearchParams) SetNProbes(n_probes uint32) (*SearchParams, error)
```

The number of clusters to search.

_Source: `go/ivf_flat/search_params.go:27`_
