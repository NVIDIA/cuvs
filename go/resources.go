package cuvs

// #include <cuvs/core/c_api.h>
import "C"

type cuvsResource C.cuvsResources_t

// Resources are objects that are shared between function calls,
// and includes things like CUDA streams, cuBLAS handles and other
// resources that are expensive to create.
type Resource struct {
	Resource C.cuvsResources_t
}

// Returns a new Resource object
func NewResource(stream C.cudaStream_t) (Resource, error) {
	res := C.cuvsResources_t(0)
	err := CheckCuvs(CuvsError(C.cuvsResourcesCreate(&res)))
	if err != nil {
		return Resource{}, err
	}

	if stream != nil {
		err := CheckCuvs(CuvsError(C.cuvsStreamSet(res, stream)))
		if err != nil {
			return Resource{}, err
		}
	}

	return Resource{Resource: res}, nil
}

// Syncs the current cuda stream
func (r Resource) Sync() error {
	return CheckCuvs(CuvsError(C.cuvsStreamSync(r.Resource)))
}

// Gets the current cuda stream
func (r Resource) GetCudaStream() (C.cudaStream_t, error) {
	var stream C.cudaStream_t

	err := CheckCuvs(CuvsError(C.cuvsStreamGet(r.Resource, &stream)))
	if err != nil {
		return C.cudaStream_t(nil), err
	}

	return stream, nil
}

// Destroys the Resource. Calling Close more than once is a no-op.
// Copies of a Resource share the same underlying handle and must not be used
// after any of them has been closed.
func (r *Resource) Close() error {
	if r == nil || r.Resource == 0 {
		return nil
	}
	err := CheckCuvs(CuvsError(C.cuvsResourcesDestroy(r.Resource)))
	if err != nil {
		return err
	}
	r.Resource = 0

	return nil
}
