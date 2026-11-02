package cagra

// #include <cuvs/neighbors/cagra.h>
import "C"

import (
	"errors"

	cuvs "github.com/rapidsai/cuvs/go"
)

type IndexParams struct {
	params C.cuvsCagraIndexParams_t
}

type BuildAlgo int

const (
	IvfPq BuildAlgo = iota
	NnDescent
	AutoSelect
)

var cBuildAlgos = map[BuildAlgo]int{
	IvfPq:      C.IVF_PQ,
	NnDescent:  C.NN_DESCENT,
	AutoSelect: C.AUTO_SELECT,
}

// Creates a new IndexParams
func CreateIndexParams() (*IndexParams, error) {
	var params C.cuvsCagraIndexParams_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraIndexParamsCreate(&params)))
	if err != nil {
		return nil, err
	}

	IndexParams := &IndexParams{params: params}

	return IndexParams, nil
}

// Degree of input graph for pruning
func (p *IndexParams) SetIntermediateGraphDegree(intermediate_graph_degree uintptr) (*IndexParams, error) {
	p.params.intermediate_graph_degree = C.size_t(intermediate_graph_degree)
	return p, nil
}

// Degree of output graph
func (p *IndexParams) SetGraphDegree(intermediate_graph_degree uintptr) (*IndexParams, error) {
	p.params.graph_degree = C.size_t(intermediate_graph_degree)

	return p, nil
}

// ANN algorithm to build knn graph
func (p *IndexParams) SetBuildAlgo(build_algo BuildAlgo) (*IndexParams, error) {
	CBuildAlgo, exists := cBuildAlgos[build_algo]

	if !exists {
		return nil, errors.New("cuvs: invalid build_algo")
	}
	p.params.build_algo = uint32(CBuildAlgo)

	return p, nil
}

// Number of iterations to run if building with NN_DESCENT
func (p *IndexParams) SetNNDescentNiter(nn_descent_niter uint32) (*IndexParams, error) {
	p.params.nn_descent_niter = C.ulong(nn_descent_niter)

	return p, nil
}

// Destroys IndexParams
func (p *IndexParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraIndexParamsDestroy(p.params)))
	if err != nil {
		return err
	}

	return nil
}
