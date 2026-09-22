package ivf_flat

// #include <cuvs/neighbors/ivf_flat.h>
import "C"

import (
	cuvs "github.com/nvidia/cuvs/go"
)

type SearchParams struct {
	params C.cuvsIvfFlatSearchParams_t
}

// Creates a new SearchParams
func CreateSearchParams() (*SearchParams, error) {
	var params C.cuvsIvfFlatSearchParams_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfFlatSearchParamsCreate(&params)))
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

// Destroy SearchParams
func (p *SearchParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfFlatSearchParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	return nil
}
