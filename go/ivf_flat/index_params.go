package ivf_flat

// #include <cuvs/neighbors/ivf_flat.h>
import "C"

import (
	"errors"

	cuvs "github.com/nvidia/cuvs/go"
)

// Supplemental parameters to build IVF Flat Index
type IndexParams struct {
	params C.cuvsIvfFlatIndexParams_t
}

// Creates a new IndexParams
func CreateIndexParams() (*IndexParams, error) {
	var params C.cuvsIvfFlatIndexParams_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfFlatIndexParamsCreate(&params)))
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
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsIvfFlatIndexParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	return nil
}
