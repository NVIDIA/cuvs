package cagra

// #include <cuvs/neighbors/cagra.h>
import "C"

import (
	cuvs "github.com/nvidia/cuvs/go"
)

// Parameters to extend CAGRA Index
type ExtendParams struct {
	params C.cuvsCagraExtendParams_t
}

// Creates a new ExtendParams
func CreateExtendParams() (*ExtendParams, error) {
	var params C.cuvsCagraExtendParams_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraExtendParamsCreate(&params)))
	if err != nil {
		return nil, err
	}

	ExtendParams := &ExtendParams{params: params}

	return ExtendParams, nil
}

// The additional dataset is divided into chunks and added to the graph.
// This is the knob to adjust the tradeoff between the recall and operation throughput.
// Large chunk sizes can result in high throughput, but use more
// working memory (O(max_chunk_size*degree^2)).
// This can also degrade recall because no edges are added between the nodes in the same chunk.
// Auto select when 0.
func (p *ExtendParams) SetMaxChunkSize(max_chunk_size uint32) (*ExtendParams, error) {
	p.params.max_chunk_size = C.uint32_t(max_chunk_size)
	return p, nil
}

func (p *ExtendParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraExtendParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	return nil
}
