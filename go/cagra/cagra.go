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
	dataset C.cuvsDatasetPadded_t
}

// Non-owning padded dataset view handle.
type PaddedDatasetView struct {
	view C.cuvsDatasetPaddedView_t
}

// Non-owning standard dataset view handle.
type StandardDatasetView struct {
	view C.cuvsDatasetStandardView_t
}

// Creates an owning device padded dataset from a device tensor.
func MakeDevicePaddedDataset[T any](Resources cuvs.Resource, dataset *cuvs.Tensor[T]) (*PaddedDataset, error) {
	if dataset == nil || dataset.C_tensor == nil {
		return nil, errors.New("dataset is nil")
	}
	datasetTensor := (*C.DLManagedTensor)(unsafe.Pointer(dataset.C_tensor))
	var paddedDataset C.cuvsDatasetPadded_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetMakeDevicePadded(
		C.cuvsResources_t(Resources.Resource), datasetTensor, &paddedDataset,
	)))
	if err != nil {
		return nil, err
	}
	return &PaddedDataset{dataset: paddedDataset}, nil
}

// Creates a non-owning padded view from an owning padded dataset.
func MakeViewFromOwningPadded(paddedDataset *PaddedDataset) (*PaddedDatasetView, error) {
	if paddedDataset == nil || paddedDataset.dataset == nil {
		return nil, errors.New("paddedDataset is nil")
	}
	var paddedView C.cuvsDatasetPaddedView_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetMakeViewFromOwningPadded(
		paddedDataset.dataset, &paddedView,
	)))
	if err != nil {
		return nil, err
	}
	return &PaddedDatasetView{view: paddedView}, nil
}

// Creates a non-owning device padded dataset view from a device tensor.
func MakeDevicePaddedDatasetView[T any](Resources cuvs.Resource, dataset *cuvs.Tensor[T]) (*PaddedDatasetView, error) {
	if dataset == nil || dataset.C_tensor == nil {
		return nil, errors.New("dataset is nil")
	}
	datasetTensor := (*C.DLManagedTensor)(unsafe.Pointer(dataset.C_tensor))
	var paddedView C.cuvsDatasetPaddedView_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetMakeDevicePaddedView(
		C.cuvsResources_t(Resources.Resource), datasetTensor, &paddedView,
	)))
	if err != nil {
		return nil, err
	}
	return &PaddedDatasetView{view: paddedView}, nil
}

// Destroys an owning padded dataset handle.
func (dataset *PaddedDataset) Close() error {
	if dataset == nil || dataset.dataset == nil {
		return nil
	}
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetPaddedDestroy(dataset.dataset)))
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
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetPaddedViewDestroy(view.view)))
	if err != nil {
		return err
	}
	view.view = nil
	return nil
}

// Creates a non-owning device standard dataset view from a device tensor.
func MakeDeviceStandardDatasetView[T any](Resources cuvs.Resource, dataset *cuvs.Tensor[T]) (*StandardDatasetView, error) {
	if dataset == nil || dataset.C_tensor == nil {
		return nil, errors.New("dataset is nil")
	}
	datasetTensor := (*C.DLManagedTensor)(unsafe.Pointer(dataset.C_tensor))
	var standardView C.cuvsDatasetStandardView_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetMakeDeviceStandardView(
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
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetStandardViewDestroy(view.view)))
	if err != nil {
		return err
	}
	view.view = nil
	return nil
}

// Attaches a caller-provided device padded dataset view and converts the index
// in-place to a search-ready padded-device layout.
func AttachDataset(Resources cuvs.Resource, paddedView *PaddedDatasetView, index *CagraIndex) error {
	if !index.trained {
		return errors.New("index needs to be built before attaching dataset")
	}
	if paddedView == nil || paddedView.view == nil {
		return errors.New("device padded dataset view is nil")
	}
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraAttachDataset(
		C.cuvsResources_t(Resources.Resource),
		paddedView.view,
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
	var viewKind C.cuvsDatasetViewKind_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraGetDatasetViewKind(
		datasetTensor,
		&viewKind,
	)))
	if err != nil {
		return err
	}

	var paddedView C.cuvsDatasetPaddedView_t
	var standardView C.cuvsDatasetStandardView_t
	defer func() {
		if paddedView != nil {
			_ = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetPaddedViewDestroy(paddedView)))
		}
		if standardView != nil {
			_ = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetStandardViewDestroy(standardView)))
		}
	}()

	switch viewKind {
	case C.CUVS_DATASET_VIEW_KIND_DEVICE_PADDED:
		err = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetMakeDevicePaddedView(
			C.cuvsResources_t(Resources.Resource), datasetTensor, &paddedView,
		)))
		if err != nil {
			return err
		}
		err = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraBuildDevicePadded(
			C.cuvsResources_t(Resources.Resource), params.params, paddedView, index.index,
		)))
	case C.CUVS_DATASET_VIEW_KIND_DEVICE_STANDARD:
		err = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetMakeDeviceStandardView(
			C.cuvsResources_t(Resources.Resource), datasetTensor, &standardView,
		)))
		if err != nil {
			return err
		}
		err = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraBuildDeviceStandard(
			C.cuvsResources_t(Resources.Resource), params.params, standardView, index.index,
		)))
	case C.CUVS_DATASET_VIEW_KIND_HOST_PADDED:
		err = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetMakeHostPaddedView(
			C.cuvsResources_t(Resources.Resource), datasetTensor, &paddedView,
		)))
		if err != nil {
			return err
		}
		err = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraBuildHostPadded(
			C.cuvsResources_t(Resources.Resource), params.params, paddedView, index.index,
		)))
	case C.CUVS_DATASET_VIEW_KIND_HOST_STANDARD:
		err = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDatasetMakeHostStandardView(
			C.cuvsResources_t(Resources.Resource), datasetTensor, &standardView,
		)))
		if err != nil {
			return err
		}
		err = cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraBuildHostStandard(
			C.cuvsResources_t(Resources.Resource), params.params, standardView, index.index,
		)))
	default:
		return errors.New("unsupported dataset view kind from cuvsCagraGetDatasetViewKind")
	}
	if err != nil {
		return err
	}

	index.trained = true
	return nil
}

// Extends the index with additional data
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - Parameters for extending the index
// * `additional_dataset` - Explicit padded dataset view to extend the index with
// * `extended_dataset` - Caller-owned writable device padded dataset view receiving extended rows
// * `index` - CagraIndex to extend
func ExtendIndex(Resources cuvs.Resource, params *ExtendParams, additional_dataset *PaddedDatasetView, extended_dataset *PaddedDatasetView, index *CagraIndex) error {
	if !index.trained {
		return errors.New("index needs to be built before calling extend")
	}
	if additional_dataset == nil || additional_dataset.view == nil {
		return errors.New("additional_dataset padded view is nil")
	}
	if extended_dataset == nil || extended_dataset.view == nil {
		return errors.New("extended_dataset padded view is nil")
	}

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraExtend(
		C.cuvsResources_t(Resources.Resource),
		params.params,
		additional_dataset.view,
		extended_dataset.view,
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
