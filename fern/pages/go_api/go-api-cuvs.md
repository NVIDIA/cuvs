---
slug: api-reference/go-api-cuvs
---

# cuVS Package

_Go package: `cuvs`_

_Sources: `go`_

## Constants

### CuvsMemoryNew Constants

```go
const (
CuvsMemoryNew = iota
CuvsMemoryRelease
)
```

_Source: `go/memory_resource.go:12`_

### Supported Distance Metrics

```go
const (
DistanceL2 Distance = iota
DistanceSQEuclidean
DistanceEuclidean
DistanceL1
DistanceCityblock
DistanceInnerProduct
DistanceChebyshev
DistanceCanberra
DistanceCosine
DistanceLp
DistanceCorrelation
DistanceJaccard
DistanceHellinger
DistanceBrayCurtis
DistanceJensenShannon
DistanceHamming
DistanceKLDivergence
DistanceMinkowski
DistanceRusselRao
DistanceDice
)
```

Supported distance metrics

_Source: `go/distance.go:14`_

## Variables

### CDistances

```go
var CDistances = map[Distance]int{
```

Maps cuvs Go distances to C distances

_Source: `go/distance.go:38`_

### DeviceDataPointer

```go
var DeviceDataPointer unsafe.Pointer
```

_Source: `go/dlpack.go:219`_

### NewDeviceDataPointer

```go
var NewDeviceDataPointer unsafe.Pointer
```

_Source: `go/dlpack.go:274`_

## Types

### CuvsError

```go
type CuvsError C.cuvsError_t
```

_Source: `go/exceptions.go:7`_

### CuvsMemoryCommand

```go
type CuvsMemoryCommand int
```

_Source: `go/memory_resource.go:10`_

### CuvsPoolMemory

```go
type CuvsPoolMemory struct {
	// contains filtered or unexported fields
}
```

_Source: `go/memory_resource.go:17`_

### Distance

```go
type Distance int
```

_Source: `go/distance.go:11`_

### Resource

```go
type Resource struct {
	Resource C.cuvsResources_t
}
```

Resources are objects that are shared between function calls,
and includes things like CUDA streams, cuBLAS handles and other
resources that are expensive to create.

_Source: `go/resources.go:11`_

### Tensor

```go
type Tensor[T any] struct {
	C_tensor *C.DLManagedTensor
}
```

ManagedTensor is a wrapper around a dlpack DLManagedTensor object.
This lets you pass matrices in device or host memory into cuvs.

_Source: `go/dlpack.go:21`_

### TensorNumberType

```go
type TensorNumberType interface {
	int64 | uint32 | float32
}
```

_Source: `go/dlpack.go:15`_

## Functions

### CheckCuda

```go
func CheckCuda(error C.cudaError_t) error
```

Wrapper function to convert cuda error to Go error

_Source: `go/exceptions.go:18`_

### CheckCuvs

```go
func CheckCuvs(error CuvsError) error
```

Wrapper function to convert cuvs error to Go error

_Source: `go/exceptions.go:10`_

### Example

```go
func Example() error
```

_Source: `go/memory_resource.go:81`_

### NewCuvsPoolMemory

```go
func NewCuvsPoolMemory(initial_pool_size_percent int, max_pool_size_percent int, managed bool) (*CuvsPoolMemory, error)
```

Creates new CuvsPoolMemory struct
initial_pool_size_percent is the initial size of the pool in percent of total available device memory
max_pool_size_percent is the maximum size of the pool in percent of total available device memory
managed is whether to use CUDA managed memory

_Source: `go/memory_resource.go:29`_

### NewResource

```go
func NewResource(stream C.cudaStream_t) (Resource, error)
```

Returns a new Resource object

_Source: `go/resources.go:16`_

### NewTensor

```go
func NewTensor[T TensorNumberType](data [][]T) (Tensor[T], error)
```

Creates a new Tensor on the host and copies the data into it.

_Source: `go/dlpack.go:27`_

### NewTensorOnDevice

```go
func NewTensorOnDevice[T TensorNumberType](res *Resource, shape []int64) (Tensor[T], error)
```

Creates a new Tensor with uninitialized data on the current device.

_Source: `go/dlpack.go:131`_

### NewVector

```go
func NewVector[T TensorNumberType](data []T) (Tensor[T], error)
```

_Source: `go/dlpack.go:79`_

### PairwiseDistance

```go
func PairwiseDistance[T any](Resources Resource, x *Tensor[T], y *Tensor[T], distances *Tensor[float32], metric Distance, metric_arg float32) error
```

Computes the pairwise distance between two vectors.

_Source: `go/distance.go:62`_

## Methods

### CuvsPoolMemory.Close

```go
func (m *CuvsPoolMemory) Close() error
```

Disables pool memory

_Source: `go/memory_resource.go:73`_

### Resource.Close

```go
func (r Resource) Close() error
```

_Source: `go/resources.go:50`_

### Resource.GetCudaStream

```go
func (r Resource) GetCudaStream() (C.cudaStream_t, error)
```

Gets the current cuda stream

_Source: `go/resources.go:39`_

### Resource.Sync

```go
func (r Resource) Sync() error
```

Syncs the current cuda stream

_Source: `go/resources.go:34`_

### Tensor.Close

```go
func (t *Tensor[T]) Close() error
```

Destroys Tensor, freeing the memory it was allocated on.

_Source: `go/dlpack.go:184`_

### Tensor.Expand

```go
func (t *Tensor[T]) Expand(res *Resource, newData [][]T) (*Tensor[T], error)
```

Expands the Tensor by adding newData to the end of the current data.
The Tensor must be on the device.

_Source: `go/dlpack.go:250`_

### Tensor.Shape

```go
func (t *Tensor[T]) Shape() []int64
```

Returns the shape of the Tensor.

_Source: `go/dlpack.go:244`_

### Tensor.Slice

```go
func (t *Tensor[T]) Slice() ([][]T, error)
```

Returns a slice of the data in the Tensor.
The Tensor must be on the host.

_Source: `go/dlpack.go:358`_

### Tensor.ToDevice

```go
func (t *Tensor[T]) ToDevice(res *Resource) (*Tensor[T], error)
```

Transfers the data in the Tensor to the device.

_Source: `go/dlpack.go:216`_

### Tensor.ToHost

```go
func (t *Tensor[T]) ToHost(res *Resource) (*Tensor[T], error)
```

Transfers the data in the Tensor to the host.

_Source: `go/dlpack.go:325`_
