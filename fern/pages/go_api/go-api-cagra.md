---
slug: api-reference/go-api-cagra
---

# Cagra Package

_Go package: `cagra`_

_Sources: `go/cagra`_

## Constants

### BuildAlgo Constants

```go
const (
IvfPq BuildAlgo = iota
NnDescent
AutoSelect
)
```

_Source: `go/cagra/index_params.go:23`_

### HashmapMode Constants

```go
const (
HashmapModeHash HashmapMode = iota
HashmapModeSmall
HashmapModeAuto
)
```

_Source: `go/cagra/search_params.go:28`_

### SearchAlgo Constants

```go
const (
SearchAlgoSingleCta SearchAlgo = iota
SearchAlgoMultiCta
SearchAlgoMultiKernel
SearchAlgoAuto
)
```

_Source: `go/cagra/search_params.go:19`_

## Types

### BuildAlgo

```go
type BuildAlgo int
```

_Source: `go/cagra/index_params.go:21`_

### CagraIndex

```go
type CagraIndex struct {
	// contains filtered or unexported fields
}
```

Cagra ANN Index

_Source: `go/cagra/cagra.go:14`_

### CompressionParams

```go
type CompressionParams struct {
	// contains filtered or unexported fields
}
```

Supplemental parameters to build CAGRA Index

_Source: `go/cagra/index_params.go:17`_

### ExtendParams

```go
type ExtendParams struct {
	// contains filtered or unexported fields
}
```

Parameters to extend CAGRA Index

_Source: `go/cagra/extend_params.go:11`_

### HashmapMode

```go
type HashmapMode int
```

_Source: `go/cagra/search_params.go:26`_

### IndexParams

```go
type IndexParams struct {
	// contains filtered or unexported fields
}
```

_Source: `go/cagra/index_params.go:12`_

### SearchAlgo

```go
type SearchAlgo int
```

_Source: `go/cagra/search_params.go:17`_

### SearchParams

```go
type SearchParams struct {
	// contains filtered or unexported fields
}
```

Supplemental parameters to search CAGRA Index

_Source: `go/cagra/search_params.go:13`_

## Functions

### BuildIndex

```go
func BuildIndex[T any](Resources cuvs.Resource, params *IndexParams, dataset *cuvs.Tensor[T], index *CagraIndex) error
```

Builds a new Index from the dataset for efficient search.

#### Arguments

* `Resources` - Resources to use
* `params` - Parameters for building the index
* `dataset` - A row-major Tensor on either the host or device to index
* `index` - CagraIndex to build

_Source: `go/cagra/cagra.go:38`_

### CreateCompressionParams

```go
func CreateCompressionParams() (*CompressionParams, error)
```

Creates a new CompressionParams

_Source: `go/cagra/index_params.go:36`_

### CreateExtendParams

```go
func CreateExtendParams() (*ExtendParams, error)
```

Creates a new ExtendParams

_Source: `go/cagra/extend_params.go:16`_

### CreateIndex

```go
func CreateIndex() (*CagraIndex, error)
```

Creates a new empty Cagra Index

_Source: `go/cagra/cagra.go:20`_

### CreateIndexParams

```go
func CreateIndexParams() (*IndexParams, error)
```

Creates a new IndexParams

_Source: `go/cagra/index_params.go:99`_

### CreateSearchParams

```go
func CreateSearchParams() (*SearchParams, error)
```

Creates a new SearchParams

_Source: `go/cagra/search_params.go:35`_

### ExtendIndex

```go
func ExtendIndex[T any](Resources cuvs.Resource, params *ExtendParams, additional_dataset *cuvs.Tensor[T], index *CagraIndex) error
```

Extends the index with additional data

#### Arguments

* `Resources` - Resources to use
* `params` - Parameters for extending the index
* `additional_dataset` - A row-major Tensor on the device to extend the index with
* `index` - CagraIndex to extend

_Source: `go/cagra/cagra.go:55`_

### SearchIndex

```go
func SearchIndex[T any](Resources cuvs.Resource, params *SearchParams, index *CagraIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[uint32], distances *cuvs.Tensor[T], allowList []uint32) error
```

Perform a Approximate Nearest Neighbors search on the Index

#### Arguments

* `Resources` - Resources to use
* `params` - Parameters to use in searching the index
* `queries` - A tensor in device memory to query for
* `neighbors` - Tensor in device memory that receives the indices of the nearest neighbors
* `distances` - Tensor in device memory that receives the distances of the nearest neighbors
* `allowList` - List of indices to allow in the search, if nil, no filtering is applied

_Source: `go/cagra/cagra.go:85`_

## Methods

### CagraIndex.Close

```go
func (index *CagraIndex) Close() error
```

Destroys the Cagra Index

_Source: `go/cagra/cagra.go:67`_

### CompressionParams.SetKMeansNIters

```go
func (p *CompressionParams) SetKMeansNIters(kmeans_n_iters uint32) (*CompressionParams, error)
```

The number of iterations searching for kmeans centers (both VQ & PQ
phases).

_Source: `go/cagra/index_params.go:76`_

### CompressionParams.SetPQBits

```go
func (p *CompressionParams) SetPQBits(pq_bits uint32) (*CompressionParams, error)
```

The bit length of the vector element after compression by PQ.

_Source: `go/cagra/index_params.go:52`_

### CompressionParams.SetPQDim

```go
func (p *CompressionParams) SetPQDim(pq_dim uint32) (*CompressionParams, error)
```

The dimensionality of the vector after compression by PQ. When zero,
an optimal value is selected using a heuristic.

_Source: `go/cagra/index_params.go:60`_

### CompressionParams.SetPQKMeansTrainsetFraction

```go
func (p *CompressionParams) SetPQKMeansTrainsetFraction(pq_kmeans_trainset_fraction float64) (*CompressionParams, error)
```

The fraction of data to use during iterative kmeans building (PQ
phase). When zero, an optimal value is selected using a heuristic.

_Source: `go/cagra/index_params.go:92`_

### CompressionParams.SetVQKMeansTrainsetFraction

```go
func (p *CompressionParams) SetVQKMeansTrainsetFraction(vq_kmeans_trainset_fraction float64) (*CompressionParams, error)
```

The fraction of data to use during iterative kmeans building (VQ
phase). When zero, an optimal value is selected using a heuristic.

_Source: `go/cagra/index_params.go:84`_

### CompressionParams.SetVQNCenters

```go
func (p *CompressionParams) SetVQNCenters(vq_n_centers uint32) (*CompressionParams, error)
```

Vector Quantization (VQ) codebook size - number of "coarse cluster
centers". When zero, an optimal value is selected using a heuristic.

_Source: `go/cagra/index_params.go:68`_

### ExtendParams.Close

```go
func (p *ExtendParams) Close() error
```

_Source: `go/cagra/extend_params.go:40`_

### ExtendParams.SetMaxChunkSize

```go
func (p *ExtendParams) SetMaxChunkSize(max_chunk_size uint32) (*ExtendParams, error)
```

The additional dataset is divided into chunks and added to the graph.
This is the knob to adjust the tradeoff between the recall and operation throughput.
Large chunk sizes can result in high throughput, but use more
working memory (O(max_chunk_size*degree^2)).
This can also degrade recall because no edges are added between the nodes in the same chunk.
Auto select when 0.

_Source: `go/cagra/extend_params.go:35`_

### IndexParams.Close

```go
func (p *IndexParams) Close() error
```

Destroys IndexParams

_Source: `go/cagra/index_params.go:152`_

### IndexParams.SetBuildAlgo

```go
func (p *IndexParams) SetBuildAlgo(build_algo BuildAlgo) (*IndexParams, error)
```

ANN algorithm to build knn graph

_Source: `go/cagra/index_params.go:126`_

### IndexParams.SetCompression

```go
func (p *IndexParams) SetCompression(compression *CompressionParams) (*IndexParams, error)
```

Compression parameters

_Source: `go/cagra/index_params.go:145`_

### IndexParams.SetGraphDegree

```go
func (p *IndexParams) SetGraphDegree(intermediate_graph_degree uintptr) (*IndexParams, error)
```

Degree of output graph

_Source: `go/cagra/index_params.go:119`_

### IndexParams.SetIntermediateGraphDegree

```go
func (p *IndexParams) SetIntermediateGraphDegree(intermediate_graph_degree uintptr) (*IndexParams, error)
```

Degree of input graph for pruning

_Source: `go/cagra/index_params.go:113`_

### IndexParams.SetNNDescentNiter

```go
func (p *IndexParams) SetNNDescentNiter(nn_descent_niter uint32) (*IndexParams, error)
```

Number of iterations to run if building with NN_DESCENT

_Source: `go/cagra/index_params.go:138`_

### SearchParams.Close

```go
func (p *SearchParams) Close() error
```

Destroys SearchParams

_Source: `go/cagra/search_params.go:157`_

### SearchParams.SetAlgo

```go
func (p *SearchParams) SetAlgo(algo SearchAlgo) (*SearchParams, error)
```

Which search implementation to use.

_Source: `go/cagra/search_params.go:67`_

### SearchParams.SetHashmapMaxFillRate

```go
func (p *SearchParams) SetHashmapMaxFillRate(hashmap_max_fill_rate float32) (*SearchParams, error)
```

Upper limit of hashmap fill rate. More than 0.1, less than 0.9.

_Source: `go/cagra/search_params.go:139`_

### SearchParams.SetHashmapMinBitlen

```go
func (p *SearchParams) SetHashmapMinBitlen(hashmap_min_bitlen uintptr) (*SearchParams, error)
```

Lower limit of hashmap bit length. More than 8.

_Source: `go/cagra/search_params.go:133`_

### SearchParams.SetHashmapMode

```go
func (p *SearchParams) SetHashmapMode(hashmap_mode HashmapMode) (*SearchParams, error)
```

Hashmap type. Auto selection when AUTO.

_Source: `go/cagra/search_params.go:113`_

### SearchParams.SetItopkSize

```go
func (p *SearchParams) SetItopkSize(itopk_size uintptr) (*SearchParams, error)
```

Number of intermediate search results retained during the search.
This is the main knob to adjust trade off between accuracy and search speed.
Higher values improve the search accuracy

_Source: `go/cagra/search_params.go:55`_

### SearchParams.SetMaxIterations

```go
func (p *SearchParams) SetMaxIterations(max_iterations uintptr) (*SearchParams, error)
```

Upper limit of search iterations. Auto select when 0.

_Source: `go/cagra/search_params.go:61`_

### SearchParams.SetMaxQueries

```go
func (p *SearchParams) SetMaxQueries(max_queries uintptr) (*SearchParams, error)
```

Maximum number of queries to search at the same time (batch size). Auto select when 0

_Source: `go/cagra/search_params.go:47`_

### SearchParams.SetMinIterations

```go
func (p *SearchParams) SetMinIterations(min_iterations uintptr) (*SearchParams, error)
```

Lower limit of search iterations.

_Source: `go/cagra/search_params.go:95`_

### SearchParams.SetNumRandomSamplings

```go
func (p *SearchParams) SetNumRandomSamplings(num_random_samplings uint32) (*SearchParams, error)
```

Number of iterations of initial random seed node selection. 1 or more.

_Source: `go/cagra/search_params.go:145`_

### SearchParams.SetRandXorMask

```go
func (p *SearchParams) SetRandXorMask(rand_xor_mask uint64) (*SearchParams, error)
```

Bit mask used for initial random seed node selection.

_Source: `go/cagra/search_params.go:151`_

### SearchParams.SetSearchWidth

```go
func (p *SearchParams) SetSearchWidth(search_width uintptr) (*SearchParams, error)
```

How many nodes to search at once. Auto select when 0.

_Source: `go/cagra/search_params.go:101`_

### SearchParams.SetTeamSize

```go
func (p *SearchParams) SetTeamSize(team_size uintptr) (*SearchParams, error)
```

Number of threads used to calculate a single distance. 4, 8, 16, or 32.

_Source: `go/cagra/search_params.go:89`_

### SearchParams.SetThreadBlockSize

```go
func (p *SearchParams) SetThreadBlockSize(thread_block_size uintptr) (*SearchParams, error)
```

Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0.

_Source: `go/cagra/search_params.go:107`_
