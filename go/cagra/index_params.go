package cagra

// #include <cuvs/neighbors/cagra.h>
import "C"

import (
	"errors"

	cuvs "github.com/nvidia/cuvs/go"
)

type IndexParams struct {
	params C.cuvsCagraIndexParams_t
}

// CompressionParams holds PQ training parameters for CAGRA-Q.
type CompressionParams struct {
	params C.cuvsCagraCompressionParams_t
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

// CreateCompressionParams creates PQ compression params with library defaults.
func CreateCompressionParams() (*CompressionParams, error) {
	var params C.cuvsCagraCompressionParams_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraCompressionParamsCreate(&params)))
	if err != nil {
		return nil, err
	}

	if params == nil {
		return nil, errors.New("memory allocation failed")
	}

	return &CompressionParams{params: params}, nil
}

// SetPQBits sets the bit length of the vector element after PQ compression.
func (p *CompressionParams) SetPQBits(pq_bits uint32) (*CompressionParams, error) {
	p.params.pq_bits = C.uint32_t(pq_bits)
	return p, nil
}

// SetPQDim sets the dimensionality after PQ compression (0 = heuristic).
func (p *CompressionParams) SetPQDim(pq_dim uint32) (*CompressionParams, error) {
	p.params.pq_dim = C.uint32_t(pq_dim)
	return p, nil
}

// SetVQNCenters sets the VQ codebook size (0 = heuristic).
func (p *CompressionParams) SetVQNCenters(vq_n_centers uint32) (*CompressionParams, error) {
	p.params.vq_n_centers = C.uint32_t(vq_n_centers)
	return p, nil
}

// SetKMeansNIters sets kmeans iterations for VQ and PQ phases.
func (p *CompressionParams) SetKMeansNIters(kmeans_n_iters uint32) (*CompressionParams, error) {
	p.params.kmeans_n_iters = C.uint32_t(kmeans_n_iters)
	return p, nil
}

// SetVQKMeansTrainsetFraction sets the VQ kmeans trainset fraction (0 = heuristic).
func (p *CompressionParams) SetVQKMeansTrainsetFraction(vq_kmeans_trainset_fraction float64) (*CompressionParams, error) {
	p.params.vq_kmeans_trainset_fraction = C.double(vq_kmeans_trainset_fraction)
	return p, nil
}

// SetPQKMeansTrainsetFraction sets the PQ kmeans trainset fraction (0 = heuristic).
func (p *CompressionParams) SetPQKMeansTrainsetFraction(pq_kmeans_trainset_fraction float64) (*CompressionParams, error) {
	p.params.pq_kmeans_trainset_fraction = C.double(pq_kmeans_trainset_fraction)
	return p, nil
}

// Close destroys CompressionParams.
func (p *CompressionParams) Close() error {
	if p == nil || p.params == nil {
		return nil
	}
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraCompressionParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	p.params = nil
	return nil
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
