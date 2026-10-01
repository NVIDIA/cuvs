package ivf_pq

// #include <cuvs/neighbors/ivf_pq.h>
import "C"

import (
	"errors"

	cuvs "github.com/nvidia/cuvs/go"
)

type IndexParams struct {
	params C.cuvsIvfPqIndexParams_t
}

type codebookKind int

const (
	Subspace codebookKind = iota
	Cluster
)

var cCodebookKinds = map[codebookKind]int{
	Subspace: C.CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE,
	Cluster:  C.CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER,
}

// Creates a new IndexParams
func CreateIndexParams() (*IndexParams, error) {
	var params C.cuvsIvfPqIndexParams_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqIndexParamsCreate(&params)))
	if err != nil {
		return nil, err
	}

	return &IndexParams{params: params}, nil
}

// The number of clusters used in the coarse quantizer.
func (p *IndexParams) SetNLists(n_lists uint32) (*IndexParams, error) {
	p.params.n_lists = C.uint32_t(n_lists)
	return p, nil
}

// Distance Type to use for building the index
func (p *IndexParams) SetMetric(metric cuvs.Distance) (*IndexParams, error) {
	CMetric, exists := cuvs.CDistances[metric]

	if !exists {
		return nil, errors.New("cuvs: invalid distance metric")
	}
	p.params.metric = C.cuvsDistanceType(CMetric)

	return p, nil
}

// Metric argument for Minkowski distances - set to 2.0 if not applicable
func (p *IndexParams) SetMetricArg(metric_arg float32) (*IndexParams, error) {
	p.params.metric_arg = C.float(metric_arg)
	return p, nil
}

// The number of iterations searching for kmeans centers during index building.
func (p *IndexParams) SetKMeansNIters(kmeans_n_iters uint32) (*IndexParams, error) {
	p.params.kmeans_n_iters = C.uint32_t(kmeans_n_iters)
	return p, nil
}

// If kmeans_trainset_fraction is less than 1, then the dataset is
// subsampled, and only n_samples * kmeans_trainset_fraction rows
// are used for training.
func (p *IndexParams) SetKMeansTrainsetFraction(kmeans_trainset_fraction float64) (*IndexParams, error) {
	p.params.kmeans_trainset_fraction = C.double(kmeans_trainset_fraction)
	return p, nil
}

// The bit length of the vector element after quantization.
func (p *IndexParams) SetPQBits(pq_bits uint32) (*IndexParams, error) {
	p.params.pq_bits = C.uint32_t(pq_bits)
	return p, nil
}

// The dimensionality of a the vector after product quantization.
// When zero, an optimal value is selected using a heuristic. Note
// pq_dim * pq_bits must be a multiple of 8. Hint: a smaller 'pq_dim'
// results in a smaller index size and better search performance, but
// lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number,
// but multiple of 8 are desirable for good performance. If 'pq_bits'
// is not 8, 'pq_dim' should be a multiple of 8. For good performance,
// it is desirable that 'pq_dim' is a multiple of 32. Ideally,
// 'pq_dim' should be also a divisor of the dataset dim.
func (p *IndexParams) SetPQDim(pq_dim uint32) (*IndexParams, error) {
	p.params.pq_dim = C.uint32_t(pq_dim)
	return p, nil
}

func (p *IndexParams) SetCodebookKind(codebook_kind codebookKind) (*IndexParams, error) {
	CCodebookKind, exists := cCodebookKinds[codebook_kind]

	if !exists {
		return nil, errors.New("cuvs: invalid codebook_kind")
	}
	p.params.codebook_kind = uint32(CCodebookKind)

	return p, nil
}

// Apply a random rotation matrix on the input data and queries even
// if `dim % pq_dim == 0`. Note: if `dim` is not multiple of `pq_dim`,
// a random rotation is always applied to the input data and queries
// to transform the working space from `dim` to `rot_dim`, which may
// be slightly larger than the original space and and is a multiple
// of `pq_dim` (`rot_dim % pq_dim == 0`). However, this transform is
// not necessary when `dim` is multiple of `pq_dim` (`dim == rot_dim`,
// hence no need in adding "extra" data columns / features). By
// default, if `dim == rot_dim`, the rotation transform is
// initialized with the identity matrix. When
// `force_random_rotation == True`, a random orthogonal transform
func (p *IndexParams) SetForceRandomRotation(force_random_rotation bool) (*IndexParams, error) {
	if force_random_rotation {
		p.params.force_random_rotation = C._Bool(true)
	} else {
		p.params.force_random_rotation = C._Bool(false)
	}
	return p, nil
}

// After training the coarse and fine quantizers, we will populate
// the index with the dataset if add_data_on_build == true, otherwise
// the index is left empty, and the extend method can be used
// to add new vectors to the index.
func (p *IndexParams) SetAddDataOnBuild(add_data_on_build bool) (*IndexParams, error) {
	if add_data_on_build {
		p.params.add_data_on_build = C._Bool(true)
	} else {
		p.params.add_data_on_build = C._Bool(false)
	}
	return p, nil
}

// Destroys IndexParams
func (p *IndexParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfPqIndexParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	return nil
}
