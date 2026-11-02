package cagra

// #include <cuvs/neighbors/cagra.h>
import "C"

import (
	"errors"
	"unsafe"

	cuvs "github.com/rapidsai/cuvs/go"
)

// Cagra ANN Index
type CagraIndex struct {
	index   C.cuvsCagraIndex_t
	trained bool
}

// Owning padded dataset handle for explicit CAGRA dataset management.
type PaddedDataset struct {
	dataset C.cuvsDataset_t
}

// PaddedDatasetHandle is an owning padded dataset or non-owning padded dataset view.
type PaddedDatasetHandle interface {
	datasetHandle() C.cuvsDataset_t
}

// Non-owning padded dataset view handle.
type PaddedDatasetView struct {
	view C.cuvsDataset_t
}

// Non-owning standard dataset view handle.
type StandardDatasetView struct {
	view C.cuvsDataset_t
}

// Matches C++ cagra_required_row_width (16-byte default alignment).
func cagraRequiredRowWidth(logicalColumns int64, sizeofValue int) int64 {
	alignBytes := 16
	lcm := lcm(alignBytes, sizeofValue)
	bytes := logicalColumns * int64(sizeofValue)
	rounded := ((bytes + int64(lcm) - 1) / int64(lcm)) * int64(lcm)
	return rounded / int64(sizeofValue)
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func lcm(a, b int) int {
	return a / gcd(a, b) * b
}

func isCagraPaddedTensor(tensor *C.DLManagedTensor, sizeofValue int) bool {
	dl := tensor.dl_tensor
	if dl.ndim != 2 || dl.shape == nil {
		return false
	}
	shape := unsafe.Slice((*C.int64_t)(unsafe.Pointer(dl.shape)), 2)
	logicalColumns := int64(shape[1])
	actualRowWidth := logicalColumns
	if dl.strides != nil {
		strides := unsafe.Slice((*C.int64_t)(unsafe.Pointer(dl.strides)), 2)
		actualRowWidth = int64(strides[0])
	}
	return actualRowWidth == cagraRequiredRowWidth(logicalColumns, sizeofValue)
}

func datasetMemType(tensor *C.DLManagedTensor) C.cuvsDatasetMemType_t {
	deviceType := tensor.dl_tensor.device.device_type
	if deviceType == C.kDLCUDA || deviceType == C.kDLCUDAManaged {
		return C.CUVS_DATASET_MEM_TYPE_DEVICE
	}
	return C.CUVS_DATASET_MEM_TYPE_HOST
}

// MakePaddedDataset creates an owning padded dataset from a tensor.
// Memory residency is inferred from the tensor device type.
func MakePaddedDataset[T any](Resources cuvs.Resource, dataset *cuvs.Tensor[T]) (*PaddedDataset, error) {
	if dataset == nil || dataset.C_tensor == nil {
		return nil, errors.New("dataset is nil")
	}
	datasetTensor := (*C.DLManagedTensor)(unsafe.Pointer(dataset.C_tensor))
	memType := datasetMemType(datasetTensor)
	var paddedDataset C.cuvsDataset_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetMakePadded(
		C.cuvsResources_t(Resources.Resource), datasetTensor, memType, &paddedDataset,
	)))
	if err != nil {
		return nil, err
	}
	return &PaddedDataset{dataset: paddedDataset}, nil
}

func (dataset *PaddedDataset) datasetHandle() C.cuvsDataset_t {
	if dataset == nil {
		return nil
	}
	return dataset.dataset
}

// MakePaddedDatasetView creates a non-owning padded dataset view from a tensor.
// Memory residency is inferred from the tensor.
func MakePaddedDatasetView[T any](Resources cuvs.Resource, dataset *cuvs.Tensor[T]) (*PaddedDatasetView, error) {
	if dataset == nil || dataset.C_tensor == nil {
		return nil, errors.New("dataset is nil")
	}
	datasetTensor := (*C.DLManagedTensor)(unsafe.Pointer(dataset.C_tensor))
	var paddedView C.cuvsDataset_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetMakePaddedView(
		C.cuvsResources_t(Resources.Resource), datasetTensor, &paddedView,
	)))
	if err != nil {
		return nil, err
	}
	return &PaddedDatasetView{view: paddedView}, nil
}

func (view *PaddedDatasetView) datasetHandle() C.cuvsDataset_t {
	if view == nil {
		return nil
	}
	return view.view
}

// Destroys an owning padded dataset handle.
func (dataset *PaddedDataset) Close() error {
	if dataset == nil || dataset.dataset == nil {
		return nil
	}
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetDestroy(dataset.dataset)))
	if err != nil {
		return err
	}
	dataset.dataset = nil
	return nil
}

// Destroys a padded dataset view handle.
func (view *PaddedDatasetView) Close() error {
	if view == nil || view.view == nil {
		return nil
	}
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetDestroy(view.view)))
	if err != nil {
		return err
	}
	view.view = nil
	return nil
}

// MakeStandardDatasetView creates a non-owning standard dataset view from a tensor.
// Memory residency is inferred from the tensor.
func MakeStandardDatasetView[T any](Resources cuvs.Resource, dataset *cuvs.Tensor[T]) (*StandardDatasetView, error) {
	if dataset == nil || dataset.C_tensor == nil {
		return nil, errors.New("dataset is nil")
	}
	datasetTensor := (*C.DLManagedTensor)(unsafe.Pointer(dataset.C_tensor))
	var standardView C.cuvsDataset_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetMakeStandardView(
		C.cuvsResources_t(Resources.Resource), datasetTensor, &standardView,
	)))
	if err != nil {
		return nil, err
	}
	return &StandardDatasetView{view: standardView}, nil
}

// Destroys a standard dataset view handle.
func (view *StandardDatasetView) Close() error {
	if view == nil || view.view == nil {
		return nil
	}
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetDestroy(view.view)))
	if err != nil {
		return err
	}
	view.view = nil
	return nil
}

// UpdateDataset updates any CAGRA index layout with a caller-provided padded
// dataset or view and leaves the same handle search-ready.
func UpdateDataset(Resources cuvs.Resource, paddedDataset PaddedDatasetHandle, index *CagraIndex) error {
	if !index.trained {
		return errors.New("index needs to be built before attaching dataset")
	}
	if paddedDataset == nil || paddedDataset.datasetHandle() == nil {
		return errors.New("padded dataset is nil")
	}
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraUpdateDataset(
		C.cuvsResources_t(Resources.Resource),
		paddedDataset.datasetHandle(),
		index.index,
	)))
	if err != nil {
		return err
	}
	return nil
}

// Creates a new empty Cagra Index
func CreateIndex() (*CagraIndex, error) {
	var index C.cuvsCagraIndex_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraIndexCreate(&index)))
	if err != nil {
		return nil, err
	}

	return &CagraIndex{index: index}, nil
}

// Builds a new Index from the dataset for efficient search.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - Parameters for building the index
// * `dataset` - A row-major Tensor on either the host or device to index
// * `index` - CagraIndex to build
func BuildIndex[T any](Resources cuvs.Resource, params *IndexParams, dataset *cuvs.Tensor[T], index *CagraIndex) error {
	datasetTensor := (*C.DLManagedTensor)(unsafe.Pointer(dataset.C_tensor))

	var zero T
	sizeofValue := int(unsafe.Sizeof(zero))
	isPadded := isCagraPaddedTensor(datasetTensor, sizeofValue)

	var datasetView C.cuvsDataset_t
	defer func() {
		if datasetView != nil {
			_ = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetDestroy(datasetView)))
		}
	}()

	var err error
	if isPadded {
		err = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetMakePaddedView(
			C.cuvsResources_t(Resources.Resource), datasetTensor, &datasetView,
		)))
	} else {
		err = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetMakeStandardView(
			C.cuvsResources_t(Resources.Resource), datasetTensor, &datasetView,
		)))
	}
	if err != nil {
		return err
	}

	err = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraBuild(
		C.cuvsResources_t(Resources.Resource),
		params.params,
		datasetView,
		index.index,
	)))
	if err != nil {
		return err
	}

	index.trained = true
	return nil
}

// Extends the index with a caller-owned pre-concatenated padded dataset.
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - Parameters for extending the index
// * `extended_dataset` - Caller-owned padded dataset already containing old || new rows
// * `newStartRow` - Row index where the additional vectors begin (must equal current index size)
// * `index` - CagraIndex to extend
func ExtendIndex(Resources cuvs.Resource, params *ExtendParams, extended_dataset PaddedDatasetHandle, newStartRow int64, index *CagraIndex) error {
	if !index.trained {
		return errors.New("index needs to be built before calling extend")
	}
	if extended_dataset == nil || extended_dataset.datasetHandle() == nil {
		return errors.New("extended_dataset is nil")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraExtend(
		C.cuvsResources_t(Resources.Resource),
		params.params,
		extended_dataset.datasetHandle(),
		C.int64_t(newStartRow),
		index.index,
	)))
	if err != nil {
		return err
	}
	return nil
}

// Destroys the Cagra Index
func (index *CagraIndex) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraIndexDestroy(index.index)))
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
// * `queries` - A tensor in device memory to query for
// * `neighbors` - Tensor in device memory that receives the indices of the nearest neighbors
// * `distances` - Tensor in device memory that receives the distances of the nearest neighbors
// * `allowList` - List of indices to allow in the search, if nil, no filtering is applied
func SearchIndex[T any](Resources cuvs.Resource, params *SearchParams, index *CagraIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[uint32], distances *cuvs.Tensor[T], allowList []uint32) error {
	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}

	var filter C.cuvsFilter
	bitset := createBitset(allowList)
	allowListTensor, err := cuvs.NewVector[uint32](bitset)
	if err != nil {
		return err
	}
	defer allowListTensor.Close()
	_, err = allowListTensor.ToDevice(&Resources)
	if err != nil {
		return err
	}
	if allowList == nil {
		filter = C.cuvsFilter{
			_type: C.NO_FILTER,
			addr:  C.uintptr_t(0),
		}
	} else {
		filter = C.cuvsFilter{
			_type: C.BITSET,
			addr:  C.uintptr_t(uintptr(unsafe.Pointer(allowListTensor.C_tensor))),
		}
	}
	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraSearch(C.cuvsResources_t(Resources.Resource), params.params, index.index, (*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)), filter)))
}

func createBitset(allowList []uint32) []uint32 {
	// Calculate size needed for the bitset array
	// Each uint32 handles 32 bits, so we divide the max ID by 32 (shift right by 5)
	maxID := uint32(0)
	for _, id := range allowList {
		if id > maxID {
			maxID = id
		}
	}
	size := (maxID >> 5) + 1 // Division by 32, add 1 to handle remainder
	bitset := make([]uint32, size)
	for _, id := range allowList {
		// Calculate which uint32 in our array (divide by 32)
		arrayIndex := id >> 5
		// Calculate bit position within that uint32 (mod 32)
		bitPosition := id & 31 // equivalent to id % 32
		// Set the bit
		bitset[arrayIndex] |= 1 << bitPosition
	}
	return bitset
}
