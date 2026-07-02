---
slug: api-reference/go-api-brute-force
---

# Brute Force Package

_Go package: `brute_force`_

_Sources: `go/brute_force`_

## Types

### BruteForceIndex

```go
type BruteForceIndex struct {
	// contains filtered or unexported fields
}
```

Brute Force KNN Index

_Source: `go/brute_force/brute_force.go:14`_

## Functions

### BuildIndex

```go
func BuildIndex[T any](Resources cuvs.Resource, Dataset *cuvs.Tensor[T], metric cuvs.Distance, metric_arg float32, index *BruteForceIndex) error
```

Builds a new Brute Force KNN Index from the dataset for efficient search.

#### Arguments

* `Resources` - Resources to use
* `Dataset` - A row-major matrix on either the host or device to index
* `metric` - Distance type to use for building the index
* `metric_arg` - Value of `p` for Minkowski distances - set to 2.0 if not applicable

_Source: `go/brute_force/brute_force.go:48`_

### CreateIndex

```go
func CreateIndex() (*BruteForceIndex, error)
```

Creates a new empty Brute Force KNN Index

_Source: `go/brute_force/brute_force.go:20`_

### SearchIndex

```go
func SearchIndex[T any](resources cuvs.Resource, index BruteForceIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[int64], distances *cuvs.Tensor[float32]) error
```

Perform a Nearest Neighbors search on the Index

#### Arguments

* `Resources` - Resources to use
* `queries` - Tensor in device memory to query for
* `neighbors` - Tensor in device memory that receives the indices of the nearest neighbors
* `distances` - Tensor in device memory that receives the distances of the nearest neighbors

_Source: `go/brute_force/brute_force.go:72`_

## Methods

### BruteForceIndex.Close

```go
func (index *BruteForceIndex) Close() error
```

Destroys the Brute Force KNN Index

_Source: `go/brute_force/brute_force.go:32`_
