package cuvs

// #include <stdlib.h>
// #include <dlpack/dlpack.h>
// #include <cuvs/core/c_api.h>
// #include <cuda_runtime_api.h>
import "C"

import (
	"errors"
	"strconv"
	"unsafe"
)

type TensorNumberType interface {
	int64 | uint32 | float32
}

// ManagedTensor is a wrapper around a dlpack DLManagedTensor object.
// This lets you pass matrices in device or host memory into cuvs.
type Tensor[T any] struct {
	C_tensor *C.DLManagedTensor
	shape    []int64
}

// Creates a new Tensor on the host and copies the data into it.
func NewTensor[T TensorNumberType](data [][]T) (Tensor[T], error) {
	if len(data) == 0 || len(data[0]) == 0 {
		return Tensor[T]{}, errors.New("empty data")
	}

	dtype := getDLDataType[T]()

	totalElements := len(data) * len(data[0])

	dataPtr := C.malloc(C.size_t(totalElements * int(unsafe.Sizeof(T(0)))))
	if dataPtr == nil {
		return Tensor[T]{}, errors.New("data memory allocation failed")
	}

	dataSlice := unsafe.Slice((*T)(dataPtr), totalElements)
	flattenData(data, dataSlice)

	shapePtr := C.malloc(C.size_t(2 * int(unsafe.Sizeof(C.int64_t(0)))))
	if shapePtr == nil {
		C.free(dataPtr)
		return Tensor[T]{}, errors.New("shape memory allocation failed")
	}

	shapeSlice := unsafe.Slice((*C.int64_t)(shapePtr), 2)
	shapeSlice[0] = C.int64_t(len(data))
	shapeSlice[1] = C.int64_t(len(data[0]))

	// Create DLManagedTensor
	dlm := (*C.DLManagedTensor)(C.malloc(C.size_t(unsafe.Sizeof(C.DLManagedTensor{}))))
	if dlm == nil {
		return Tensor[T]{}, errors.New("tensor allocation failed")
	}

	dlm.dl_tensor.data = dataPtr
	dlm.dl_tensor.device = C.DLDevice{
		device_type: C.DLDeviceType(C.kDLCPU),
		device_id:   0,
	}
	dlm.dl_tensor.dtype = dtype
	dlm.dl_tensor.ndim = 2
	dlm.dl_tensor.shape = (*C.int64_t)(shapePtr)
	dlm.dl_tensor.strides = nil
	dlm.dl_tensor.byte_offset = 0
	dlm.manager_ctx = nil
	dlm.deleter = nil

	return Tensor[T]{
		C_tensor: dlm,
		shape:    []int64{int64(len(data)), int64(len(data[0]))},
	}, nil
}

func NewVector[T TensorNumberType](data []T) (Tensor[T], error) {
	if len(data) == 0 {
		return Tensor[T]{}, errors.New("empty data")
	}

	dtype := getDLDataType[T]()

	totalElements := len(data)

	dataPtr := C.malloc(C.size_t(totalElements * int(unsafe.Sizeof(T(0)))))
	if dataPtr == nil {
		return Tensor[T]{}, errors.New("data memory allocation failed")
	}

	dataSlice := unsafe.Slice((*T)(dataPtr), totalElements)
	copy(dataSlice, data)

	shapePtr := C.malloc(C.size_t(int(unsafe.Sizeof(C.int64_t(0)))))
	if shapePtr == nil {
		C.free(dataPtr)
		return Tensor[T]{}, errors.New("shape memory allocation failed")
	}

	shapeSlice := unsafe.Slice((*C.int64_t)(shapePtr), 1)
	shapeSlice[0] = C.int64_t(len(data))

	// Create DLManagedTensor
	dlm := (*C.DLManagedTensor)(C.malloc(C.size_t(unsafe.Sizeof(C.DLManagedTensor{}))))
	if dlm == nil {
		return Tensor[T]{}, errors.New("tensor allocation failed")
	}

	dlm.dl_tensor.data = dataPtr
	dlm.dl_tensor.device = C.DLDevice{
		device_type: C.DLDeviceType(C.kDLCPU),
		device_id:   0,
	}
	dlm.dl_tensor.dtype = dtype
	dlm.dl_tensor.ndim = 1
	dlm.dl_tensor.shape = (*C.int64_t)(shapePtr)
	dlm.dl_tensor.strides = nil
	dlm.dl_tensor.byte_offset = 0
	dlm.manager_ctx = nil
	dlm.deleter = nil

	return Tensor[T]{
		C_tensor: dlm,
		shape:    []int64{int64(len(data))},
	}, nil
}

// Creates a new Tensor with uninitialized data on the current device.
func NewTensorOnDevice[T TensorNumberType](res *Resource, shape []int64) (Tensor[T], error) {
	if len(shape) < 2 {
		return Tensor[T]{}, errors.New("shape must be at least 2")
	}

	shapePtr := C.malloc(C.size_t(len(shape) * int(unsafe.Sizeof(C.int64_t(0)))))
	if shapePtr == nil {
		return Tensor[T]{}, errors.New("shape memory allocation failed")
	}

	shapeSlice := unsafe.Slice((*C.int64_t)(shapePtr), len(shape))
	for i, dim := range shape {
		shapeSlice[i] = C.int64_t(dim)
	}

	dlm := (*C.DLManagedTensor)(C.malloc(C.size_t(unsafe.Sizeof(C.DLManagedTensor{}))))
	if dlm == nil {
		return Tensor[T]{}, errors.New("tensor allocation failed")
	}
	dtype := getDLDataType[T]()

	var deviceDataPtr unsafe.Pointer
	bytes := calculateBytes(shape, dtype)
	err := CheckCuvs(CuvsError(C.cuvsRMMAlloc(res.Resource, &deviceDataPtr, C.size_t(bytes))))
	if err != nil {
		C.free(unsafe.Pointer(dlm))
		C.free(unsafe.Pointer(shapePtr))
		return Tensor[T]{}, err
	}

	dlm.dl_tensor.data = deviceDataPtr
	dlm.dl_tensor.device = C.DLDevice{
		device_type: C.DLDeviceType(C.kDLCUDA),
		device_id:   0,
	}
	dlm.dl_tensor.dtype = dtype
	dlm.dl_tensor.ndim = C.int(len(shape))
	dlm.dl_tensor.shape = (*C.int64_t)(shapePtr)
	dlm.dl_tensor.strides = nil
	dlm.dl_tensor.byte_offset = 0
	dlm.manager_ctx = nil
	dlm.deleter = nil

	shapeCopy := make([]int64, len(shape))
	copy(shapeCopy, shape)

	return Tensor[T]{
		C_tensor: dlm,
		shape:    shapeCopy,
	}, nil
}

// Destroys Tensor, freeing the memory it was allocated on.
func (t *Tensor[T]) Close() error {
	if t.C_tensor.dl_tensor.device.device_type == C.kDLCUDA {
		bytes := t.sizeInBytes()
		res, err := NewResource(nil)
		if err != nil {
			return err
		}
		err = CheckCuvs(CuvsError(C.cuvsRMMFree(res.Resource, t.C_tensor.dl_tensor.data, C.size_t(bytes))))

		return err
	} else if t.C_tensor.dl_tensor.device.device_type == C.kDLCPU {
		if t.C_tensor.dl_tensor.data != nil {
			C.free(t.C_tensor.dl_tensor.data)
			t.C_tensor.dl_tensor.data = nil
		}
	}

	if t.C_tensor.dl_tensor.shape != nil {
		C.free(unsafe.Pointer(t.C_tensor.dl_tensor.shape))
		t.C_tensor.dl_tensor.shape = nil
	}

	if t.C_tensor != nil {
		C.free(unsafe.Pointer(t.C_tensor))
		t.C_tensor = nil
	}

	t.C_tensor = nil
	return nil
}

// Transfers the data in the Tensor to the device.
func (t *Tensor[T]) ToDevice(res *Resource) (*Tensor[T], error) {
	bytes := t.sizeInBytes()

	var DeviceDataPointer unsafe.Pointer

	err := CheckCuvs(CuvsError(C.cuvsRMMAlloc(res.Resource, &DeviceDataPointer, C.size_t(bytes))))
	if err != nil {
		return nil, err
	}

	err = CheckCuda(
		C.cudaMemcpy(
			DeviceDataPointer,
			t.C_tensor.dl_tensor.data,
			C.size_t(bytes),
			C.cudaMemcpyHostToDevice,
		))
	if err != nil {
		C.cuvsRMMFree(res.Resource, DeviceDataPointer, C.size_t(bytes))
		return nil, err
	}
	t.C_tensor.dl_tensor.device.device_type = C.kDLCUDA
	t.C_tensor.dl_tensor.data = DeviceDataPointer

	return t, nil
}

// Returns the shape of the Tensor.
func (t *Tensor[T]) Shape() []int64 {
	return t.shape
}

// Expands the Tensor by adding newData to the end of the current data.
// The Tensor must be on the device.
func (t *Tensor[T]) Expand(res *Resource, newData [][]T) (*Tensor[T], error) {
	if t.C_tensor.dl_tensor.device.device_type != C.kDLCUDA {
		return &Tensor[T]{}, errors.New("Tensor must be on GPU")
	}

	newShape := []int64{int64(len(newData)), int64(len(newData[0]))}

	flatData := make([]T, len(newData)*len(newData[0]))
	for i := range newData {
		for j := range newData[i] {
			flatData[i*len(newData[0])+j] = newData[i][j]
		}
	}

	old_shape := unsafe.Slice((*int64)(unsafe.Pointer(t.C_tensor.dl_tensor.shape)), 2)

	if old_shape[1] != newShape[1] {
		return &Tensor[T]{}, errors.New("new shape must be same as old shape, old shape: " + strconv.Itoa(int(old_shape[1])) + ", new shape: " + strconv.Itoa(int(newShape[1])))
	}

	newDataSize := newShape[0] * newShape[1] * int64(t.C_tensor.dl_tensor.dtype.bits) / 8

	bytes := t.sizeInBytes()

	var NewDeviceDataPointer unsafe.Pointer

	err := CheckCuvs(CuvsError(C.cuvsRMMAlloc(res.Resource, &NewDeviceDataPointer, C.size_t(bytes+newDataSize))))
	if err != nil {
		return nil, err
	}

	err = CheckCuda(
		C.cudaMemcpy(
			NewDeviceDataPointer,
			t.C_tensor.dl_tensor.data,
			C.size_t(bytes),
			C.cudaMemcpyDeviceToDevice,
		))
	if err != nil {
		C.cuvsRMMFree(res.Resource, NewDeviceDataPointer, C.size_t(bytes+newDataSize))
		return nil, err
	}

	err = CheckCuda(
		C.cudaMemcpy(
			unsafe.Pointer(uintptr(NewDeviceDataPointer)+uintptr(bytes)),
			unsafe.Pointer(&flatData[0]),
			C.size_t(newDataSize),
			C.cudaMemcpyHostToDevice,
		))
	if err != nil {
		C.cuvsRMMFree(res.Resource, NewDeviceDataPointer, C.size_t(bytes+newDataSize))
		return nil, err
	}

	err = CheckCuvs(CuvsError(
		C.cuvsRMMFree(res.Resource, t.C_tensor.dl_tensor.data, C.size_t(bytes))))
	if err != nil {
		return nil, err
	}

	shape := make([]int64, 2)
	shape[0] = int64(*t.C_tensor.dl_tensor.shape) + int64(len(newData))

	shape[1] = newShape[1]

	t.shape = shape

	t.C_tensor.dl_tensor.data = NewDeviceDataPointer
	t.C_tensor.dl_tensor.shape = (*C.int64_t)(unsafe.Pointer(&shape[0]))

	return t, nil
}

// Transfers the data in the Tensor to the host.
func (t *Tensor[T]) ToHost(res *Resource) (*Tensor[T], error) {
	bytes := t.sizeInBytes()

	addr := (C.malloc(C.size_t(bytes)))
	if addr == nil {
		return nil, errors.New("memory allocation failed")
	}

	err := CheckCuda(
		C.cudaMemcpy(
			addr,
			t.C_tensor.dl_tensor.data,
			C.size_t(bytes),
			C.cudaMemcpyDeviceToHost,
		))
	if err != nil {
		return nil, err
	}

	err = CheckCuvs(CuvsError(
		C.cuvsRMMFree(res.Resource, t.C_tensor.dl_tensor.data, C.size_t(bytes))))
	if err != nil {
		return nil, err
	}

	t.C_tensor.dl_tensor.device.device_type = C.kDLCPU
	t.C_tensor.dl_tensor.data = addr

	return t, nil
}

// Returns a slice of the data in the Tensor.
// The Tensor must be on the host.
func (t *Tensor[T]) Slice() ([][]T, error) {
	if t.C_tensor.dl_tensor.device.device_type != C.kDLCPU {
		return nil, errors.New("Tensor must be on CPU")
	}

	flatData := unsafe.Slice((*T)(t.C_tensor.dl_tensor.data), t.shape[0]*t.shape[1])

	data := make([][]T, t.shape[0])
	for i := range data {
		data[i] = make([]T, t.shape[1])
		for j := range data[i] {
			data[i][j] = flatData[i*int(t.shape[1])+j]
		}
	}

	return data, nil
}

func getDLDataType[T TensorNumberType]() C.DLDataType {
	var zero T
	switch any(zero).(type) {
	case int64:
		return C.DLDataType{
			bits:  C.uchar(64),
			lanes: C.ushort(1),
			code:  C.kDLInt,
		}
	case uint32:
		return C.DLDataType{
			bits:  C.uchar(32),
			lanes: C.ushort(1),
			code:  C.kDLUInt,
		}
	case float32:
		return C.DLDataType{
			bits:  C.uchar(32),
			lanes: C.ushort(1),
			code:  C.kDLFloat,
		}
	}
	panic("unreachable") // Go compiler ensures this is unreachable
}

func flattenData[T TensorNumberType](data [][]T, dest []T) {
	cols := len(data[0])
	for i, row := range data {
		copy(dest[i*cols:], row)
	}
}

func (t *Tensor[T]) sizeInBytes() int64 {
	return calculateBytes(t.shape, t.C_tensor.dl_tensor.dtype)
}

func calculateBytes(shape []int64, dtype C.DLDataType) int64 {
	bytes := int64(1)
	for dim := range shape {
		bytes *= (shape[dim])
	}

	bytes *= int64(dtype.bits) / 8

	return bytes
}
