package ivf_pq

// #include <cuvs/neighbors/ivf_pq.h>
import "C"

import (
	"errors"

	cuvs "github.com/nvidia/cuvs/go"
)

// Supplemental parameters to search IVF PQ Index
type SearchParams struct {
	params C.cuvsIvfPqSearchParams_t
}

type lutDtype int

const (
	Lut_Uint8 lutDtype = iota
	Lut_Uint16
	Lut_Uint32
	Lut_Uint64
	Lut_Int8
	Lut_Int16
	Lut_Int32
	Lut_Int64
)

var cLutDtypes = map[lutDtype]int{
	Lut_Uint8:  C.CUDA_R_8U,
	Lut_Uint16: C.CUDA_R_16U,
	Lut_Uint32: C.CUDA_R_32U,
	Lut_Uint64: C.CUDA_R_64U,
	Lut_Int8:   C.CUDA_R_8I,
	Lut_Int16:  C.CUDA_R_16I,
	Lut_Int32:  C.CUDA_R_32I,
	Lut_Int64:  C.CUDA_R_64I,
}

type internalDistanceDtype int

const (
	InternalDistance_Float32 internalDistanceDtype = iota
	InternalDistance_Float64
)

var CInternalDistanceDtypes = map[internalDistanceDtype]int{
	InternalDistance_Float32: C.CUDA_R_32F,
	InternalDistance_Float64: C.CUDA_R_64F,
}

// Creates a new SearchParams
func CreateSearchParams() (*SearchParams, error) {
	var params C.cuvsIvfPqSearchParams_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqSearchParamsCreate(&params)))
	if err != nil {
		return nil, err
	}

	return &SearchParams{params: params}, nil
}

// The number of clusters to search.
func (p *SearchParams) SetNProbes(n_probes uint32) (*SearchParams, error) {
	p.params.n_probes = C.uint32_t(n_probes)
	return p, nil
}

// Data type of look up table to be created dynamically at search
// time. The use of low-precision types reduces the amount of shared
// memory required at search time, so fast shared memory kernels can
// be used even for datasets with large dimansionality. Note that
// the recall is slightly degraded when low-precision type is
// selected.
func (p *SearchParams) SetLutDtype(lut_dtype lutDtype) (*SearchParams, error) {
	CLutDtype, exists := cLutDtypes[lutDtype(lut_dtype)]

	if !exists {
		return nil, errors.New("cuvs: invalid lut_dtype")
	}
	p.params.lut_dtype = C.cudaDataType_t(CLutDtype)

	return p, nil
}

// Storage data type for distance/similarity computation.
func (p *SearchParams) SetInternalDistanceDtype(internal_distance_dtype internalDistanceDtype) (*SearchParams, error) {
	CInternalDistanceDtype, exists := CInternalDistanceDtypes[internalDistanceDtype(internal_distance_dtype)]

	if !exists {
		return nil, errors.New("cuvs: invalid internal_distance_dtype")
	}
	p.params.internal_distance_dtype = C.cudaDataType_t(CInternalDistanceDtype)

	return p, nil
}

// Destroys SearchParams
func (p *SearchParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqSearchParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	return nil
}
