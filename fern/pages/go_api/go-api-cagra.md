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

_Source: `go/cagra/index_params.go:29`_

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

_Source: `go/cagra/index_params.go:27`_

### CagraIndex

```go
type CagraIndex struct {
	// contains filtered or unexported fields
}
```

Cagra ANN Index

_Source: `go/cagra/cagra.go:15`_

### CompressionParams

```go
type CompressionParams struct {
	// contains filtered or unexported fields
}
```

CompressionParams holds PQ training parameters for CAGRA-Q.

_Source: `go/cagra/index_params.go:18`_

### DatasetHandle

```go
type DatasetHandle interface {
	datasetHandle() C.cuvsDataset_t
}
```

DatasetHandle is any CAGRA dataset handle accepted by UpdateDataset
(device-padded or device PQ).

_Source: `go/cagra/cagra.go:37`_

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

_Source: `go/cagra/index_params.go:13`_

### PaddedDataset

```go
type PaddedDataset struct {
	// contains filtered or unexported fields
}
```

Owning padded dataset handle for explicit CAGRA dataset management.

_Source: `go/cagra/cagra.go:21`_

### PaddedDatasetHandle

```go
type PaddedDatasetHandle interface {
	datasetHandle() C.cuvsDataset_t
}
```

PaddedDatasetHandle is an owning padded dataset or non-owning padded dataset view.

_Source: `go/cagra/cagra.go:31`_

### PaddedDatasetView

```go
type PaddedDatasetView struct {
	// contains filtered or unexported fields
}
```

Non-owning padded dataset view handle.

_Source: `go/cagra/cagra.go:42`_

### PqDataset

```go
type PqDataset struct {
	// contains filtered or unexported fields
}
```

Owning PQ dataset handle for CAGRA-Q search.

_Source: `go/cagra/cagra.go:26`_

### ProductQuantizerParams

```go
type ProductQuantizerParams struct {
	// contains filtered or unexported fields
}
```

ProductQuantizerParams holds product quantizer training parameters.

_Source: `go/cagra/index_params.go:23`_

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

### StandardDatasetView

```go
type StandardDatasetView struct {
	// contains filtered or unexported fields
}
```

Non-owning standard dataset view handle.

_Source: `go/cagra/cagra.go:47`_

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

_Source: `go/cagra/cagra.go:281`_

### CreateCompressionParams

```go
func CreateCompressionParams() (*CompressionParams, error)
```

CreateCompressionParams creates PQ compression params with library defaults.

_Source: `go/cagra/index_params.go:42`_

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

_Source: `go/cagra/cagra.go:263`_

### CreateIndexParams

```go
func CreateIndexParams() (*IndexParams, error)
```

Creates a new IndexParams

_Source: `go/cagra/index_params.go:173`_

### CreateProductQuantizerParams

```go
func CreateProductQuantizerParams() (*ProductQuantizerParams, error)
```

CreateProductQuantizerParams creates product quantizer params with library defaults.

_Source: `go/cagra/index_params.go:107`_

### CreateSearchParams

```go
func CreateSearchParams() (*SearchParams, error)
```

Creates a new SearchParams

_Source: `go/cagra/search_params.go:35`_

### ExtendIndex

```go
func ExtendIndex(Resources cuvs.Resource, params *ExtendParams, extended_dataset PaddedDatasetHandle, newStartRow int64, index *CagraIndex) error
```

Extends the index with a caller-owned pre-concatenated padded dataset.

#### Arguments

* `Resources` - Resources to use
* `params` - Parameters for extending the index
* `extended_dataset` - Caller-owned padded dataset already containing old \|\| new rows
* `newStartRow` - Row index where the additional vectors begin (must equal current index size)
* `index` - CagraIndex to extend

_Source: `go/cagra/cagra.go:332`_

### MakePaddedDataset

```go
func MakePaddedDataset[T any](Resources cuvs.Resource, dataset *cuvs.Tensor[T]) (*PaddedDataset, error)
```

MakePaddedDataset creates an owning padded dataset from a tensor.
Memory residency is inferred from the tensor device type.

_Source: `go/cagra/cagra.go:96`_

### MakePaddedDatasetView

```go
func MakePaddedDatasetView[T any](Resources cuvs.Resource, dataset *cuvs.Tensor[T]) (*PaddedDatasetView, error)
```

MakePaddedDatasetView creates a non-owning padded dataset view from a tensor.
Memory residency is inferred from the tensor.

_Source: `go/cagra/cagra.go:121`_

### MakePqDataset

```go
func MakePqDataset(Resources cuvs.Resource, source PaddedDatasetHandle, params *ProductQuantizerParams) (*PqDataset, error)
```

MakePqDataset trains an owning device PQ dataset (CAGRA-Q) from a device-padded source.
params may be nil to use library defaults. Keep the returned dataset alive while any index uses it.

_Source: `go/cagra/cagra.go:221`_

### MakeStandardDatasetView

```go
func MakeStandardDatasetView[T any](Resources cuvs.Resource, dataset *cuvs.Tensor[T]) (*StandardDatasetView, error)
```

MakeStandardDatasetView creates a non-owning standard dataset view from a tensor.
Memory residency is inferred from the tensor.

_Source: `go/cagra/cagra.go:171`_

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

_Source: `go/cagra/cagra.go:372`_

### UpdateDataset

```go
func UpdateDataset(Resources cuvs.Resource, dataset DatasetHandle, index *CagraIndex) error
```

UpdateDataset updates any CAGRA index layout with a caller-provided device
padded or PQ dataset/view and leaves the same handle search-ready.

_Source: `go/cagra/cagra.go:201`_

## Methods

### CagraIndex.Close

```go
func (index *CagraIndex) Close() error
```

Destroys the Cagra Index

_Source: `go/cagra/cagra.go:354`_

### CompressionParams.Close

```go
func (p *CompressionParams) Close() error
```

Close destroys CompressionParams.

_Source: `go/cagra/index_params.go:94`_

### CompressionParams.SetKMeansNIters

```go
func (p *CompressionParams) SetKMeansNIters(kmeans_n_iters uint32) (*CompressionParams, error)
```

SetKMeansNIters sets kmeans iterations for VQ and PQ phases.

_Source: `go/cagra/index_params.go:76`_

### CompressionParams.SetPQBits

```go
func (p *CompressionParams) SetPQBits(pq_bits uint32) (*CompressionParams, error)
```

SetPQBits sets the bit length of the vector element after PQ compression.

_Source: `go/cagra/index_params.go:58`_

### CompressionParams.SetPQDim

```go
func (p *CompressionParams) SetPQDim(pq_dim uint32) (*CompressionParams, error)
```

SetPQDim sets the dimensionality after PQ compression (0 = heuristic).

_Source: `go/cagra/index_params.go:64`_

### CompressionParams.SetPQKMeansTrainsetFraction

```go
func (p *CompressionParams) SetPQKMeansTrainsetFraction(pq_kmeans_trainset_fraction float64) (*CompressionParams, error)
```

SetPQKMeansTrainsetFraction sets the PQ kmeans trainset fraction (0 = heuristic).

_Source: `go/cagra/index_params.go:88`_

### CompressionParams.SetVQKMeansTrainsetFraction

```go
func (p *CompressionParams) SetVQKMeansTrainsetFraction(vq_kmeans_trainset_fraction float64) (*CompressionParams, error)
```

SetVQKMeansTrainsetFraction sets the VQ kmeans trainset fraction (0 = heuristic).

_Source: `go/cagra/index_params.go:82`_

### CompressionParams.SetVQNCenters

```go
func (p *CompressionParams) SetVQNCenters(vq_n_centers uint32) (*CompressionParams, error)
```

SetVQNCenters sets the VQ codebook size (0 = heuristic).

_Source: `go/cagra/index_params.go:70`_

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

_Source: `go/cagra/index_params.go:219`_

### IndexParams.SetBuildAlgo

```go
func (p *IndexParams) SetBuildAlgo(build_algo BuildAlgo) (*IndexParams, error)
```

ANN algorithm to build knn graph

_Source: `go/cagra/index_params.go:200`_

### IndexParams.SetGraphDegree

```go
func (p *IndexParams) SetGraphDegree(intermediate_graph_degree uintptr) (*IndexParams, error)
```

Degree of output graph

_Source: `go/cagra/index_params.go:193`_

### IndexParams.SetIntermediateGraphDegree

```go
func (p *IndexParams) SetIntermediateGraphDegree(intermediate_graph_degree uintptr) (*IndexParams, error)
```

Degree of input graph for pruning

_Source: `go/cagra/index_params.go:187`_

### IndexParams.SetNNDescentNiter

```go
func (p *IndexParams) SetNNDescentNiter(nn_descent_niter uint32) (*IndexParams, error)
```

Number of iterations to run if building with NN_DESCENT

_Source: `go/cagra/index_params.go:212`_

### PaddedDataset.Close

```go
func (dataset *PaddedDataset) Close() error
```

Destroys an owning padded dataset handle.

_Source: `go/cagra/cagra.go:144`_

### PaddedDatasetView.Close

```go
func (view *PaddedDatasetView) Close() error
```

Destroys a padded dataset view handle.

_Source: `go/cagra/cagra.go:157`_

### PqDataset.Close

```go
func (dataset *PqDataset) Close() error
```

Close destroys an owning PQ dataset handle.

_Source: `go/cagra/cagra.go:250`_

### ProductQuantizerParams.Close

```go
func (p *ProductQuantizerParams) Close() error
```

Close destroys ProductQuantizerParams.

_Source: `go/cagra/index_params.go:160`_

### ProductQuantizerParams.SetKMeansNIters

```go
func (p *ProductQuantizerParams) SetKMeansNIters(value uint32) *ProductQuantizerParams
```

_Source: `go/cagra/index_params.go:144`_

### ProductQuantizerParams.SetMaxTrainPointsPerPQCode

```go
func (p *ProductQuantizerParams) SetMaxTrainPointsPerPQCode(value uint32) *ProductQuantizerParams
```

_Source: `go/cagra/index_params.go:149`_

### ProductQuantizerParams.SetMaxTrainPointsPerVQCluster

```go
func (p *ProductQuantizerParams) SetMaxTrainPointsPerVQCluster(value uint32) *ProductQuantizerParams
```

_Source: `go/cagra/index_params.go:154`_

### ProductQuantizerParams.SetPQBits

```go
func (p *ProductQuantizerParams) SetPQBits(value uint32) *ProductQuantizerParams
```

_Source: `go/cagra/index_params.go:119`_

### ProductQuantizerParams.SetPQDim

```go
func (p *ProductQuantizerParams) SetPQDim(value uint32) *ProductQuantizerParams
```

_Source: `go/cagra/index_params.go:124`_

### ProductQuantizerParams.SetUseSubspaces

```go
func (p *ProductQuantizerParams) SetUseSubspaces(value bool) *ProductQuantizerParams
```

_Source: `go/cagra/index_params.go:129`_

### ProductQuantizerParams.SetUseVQ

```go
func (p *ProductQuantizerParams) SetUseVQ(value bool) *ProductQuantizerParams
```

_Source: `go/cagra/index_params.go:134`_

### ProductQuantizerParams.SetVQNCenters

```go
func (p *ProductQuantizerParams) SetVQNCenters(value uint32) *ProductQuantizerParams
```

_Source: `go/cagra/index_params.go:139`_

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

### StandardDatasetView.Close

```go
func (view *StandardDatasetView) Close() error
```

Destroys a standard dataset view handle.

_Source: `go/cagra/cagra.go:187`_
