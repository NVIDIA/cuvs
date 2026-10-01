package brute_force

// #include <cuvs/neighbors/brute_force.h>
import "C"

import (
	"errors"
	"unsafe"

	cuvs "github.com/nvidia/cuvs/go"
)

// Brute Force KNN Index
type BruteForceIndex struct {
	index   C.cuvsBruteForceIndex_t
	trained bool
}

// Creates a new empty Brute Force KNN Index
func CreateIndex() (*BruteForceIndex, error) {
	var index C.cuvsBruteForceIndex_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceIndexCreate(&index)))
	if err != nil {
		return nil, err
	}

	return &BruteForceIndex{index: index, trained: false}, nil
}

// Destroys the Brute Force KNN Index
func (index *BruteForceIndex) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceIndexDestroy(index.index)))
	if err != nil {
		return err
	}
	return nil
}

// Builds a new Brute Force KNN Index from the dataset for efficient search.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `Dataset` - A row-major matrix on either the host or device to index
// * `metric` - Distance type to use for building the index
// * `metric_arg` - Value of `p` for Minkowski distances - set to 2.0 if not applicable
func BuildIndex[T any](Resources cuvs.Resource, Dataset *cuvs.Tensor[T], metric cuvs.Distance, metric_arg float32, index *BruteForceIndex) error {
	CMetric, exists := cuvs.CDistances[metric]

	if !exists {
		return errors.New("cuvs: invalid distance metric")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceBuild(C.cuvsResources_t(Resources.Resource), (*C.DLManagedTensor)(unsafe.Pointer(Dataset.C_tensor)), C.cuvsDistanceType(CMetric), C.float(metric_arg), index.index)))
	if err != nil {
		return err
	}
	index.trained = true

	return nil
}

// Perform a Nearest Neighbors search on the Index
//
// # Arguments
//
// * `Resources` - Resources to use
// * `queries` - Tensor in device memory to query for
// * `neighbors` - Tensor in device memory that receives the indices of the nearest neighbors
// * `distances` - Tensor in device memory that receives the distances of the nearest neighbors
func SearchIndex[T any](resources cuvs.Resource, index BruteForceIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[int64], distances *cuvs.Tensor[float32]) error {
	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	prefilter := C.cuvsFilter{
		addr:  0,
		_type: C.NO_FILTER,
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsBruteForceSearch(C.ulong(resources.Resource), index.index, (*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)), prefilter)))

	return err
}
