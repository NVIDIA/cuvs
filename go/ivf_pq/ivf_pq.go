package ivf_pq

// #include <cuvs/neighbors/ivf_pq.h>
import "C"

import (
	"errors"
	"unsafe"

	cuvs "github.com/nvidia/cuvs/go"
)

// IVF PQ Index
type IvfPqIndex struct {
	index   C.cuvsIvfPqIndex_t
	trained bool
}

// Creates a new empty IvfPqIndex
func CreateIndex(params *IndexParams, dataset *cuvs.Tensor[float32]) (*IvfPqIndex, error) {
	var index C.cuvsIvfPqIndex_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqIndexCreate(&index)))
	if err != nil {
		return nil, err
	}

	return &IvfPqIndex{index: index}, nil
}

// Builds an IvfPqIndex from the dataset for efficient search.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - Parameters for building the index
// * `dataset` - A row-major Tensor on either the host or device to index
// * `index` - IvfPqIndex to build
func BuildIndex[T any](Resources cuvs.Resource, params *IndexParams, dataset *cuvs.Tensor[T], index *IvfPqIndex) error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqBuild(C.ulong(Resources.Resource), params.params, (*C.DLManagedTensor)(unsafe.Pointer(dataset.C_tensor)), index.index)))
	if err != nil {
		return err
	}
	index.trained = true
	return nil
}

// Destroys the IvfPqIndex
func (index *IvfPqIndex) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqIndexDestroy(index.index)))
	if err != nil {
		return err
	}
	return nil
}

// Perform a Approximate Nearest Neighbors search on the Index
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - Parameters to use in searching the index
// * `index` - IvfPqIndex to search
// * `queries` - A tensor in device memory to query for
// * `neighbors` - Tensor in device memory that receives the indices of the nearest neighbors
// * `distances` - Tensor in device memory that receives the distances of the nearest neighbors
func SearchIndex[T any](Resources cuvs.Resource, params *SearchParams, index *IvfPqIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[int64], distances *cuvs.Tensor[T]) error {
	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqSearch(C.cuvsResources_t(Resources.Resource), params.params, index.index, (*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)))))
}
