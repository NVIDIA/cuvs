package cagra

// #include <cuvs/neighbors/cagra.h>
import "C"

import (
	"errors"

	cuvs "github.com/nvidia/cuvs/go"
)

// Parameters controlling how physical CAGRA indices are merged.
type MergeParams struct {
	params C.cuvsCagraMergeParams_t
}

// Algorithm used to merge physical CAGRA indices.
type MergeAlgo int

const (
	MergeAuto MergeAlgo = iota
	MergeFastener
	MergeRebuild
)

var cMergeAlgos = map[MergeAlgo]int{
	MergeAuto:     C.CUVS_CAGRA_MERGE_AUTO,
	MergeFastener: C.CUVS_CAGRA_MERGE_FASTENER,
	MergeRebuild:  C.CUVS_CAGRA_MERGE_REBUILD,
}

// Creates a new MergeParams, populated with AUTO defaults.
func CreateMergeParams() (*MergeParams, error) {
	var params C.cuvsCagraMergeParams_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraMergeParamsCreate(&params)))
	if err != nil {
		return nil, err
	}

	MergeParams := &MergeParams{params: params}

	return MergeParams, nil
}

// Algorithm used to merge the physical CAGRA indices.
func (p *MergeParams) SetAlgo(algo MergeAlgo) (*MergeParams, error) {
	CMergeAlgo, exists := cMergeAlgos[algo]

	if !exists {
		return nil, errors.New("cuvs: invalid merge algo")
	}
	p.params.algo = uint32(CMergeAlgo)

	return p, nil
}

// Number of levels used by the merge algorithm.
func (p *MergeParams) SetLevels(levels uint32) (*MergeParams, error) {
	p.params.levels = C.uint32_t(levels)
	return p, nil
}

// Fanout of the root level.
func (p *MergeParams) SetRootFanout(root_fanout uint32) (*MergeParams, error) {
	p.params.root_fanout = C.uint32_t(root_fanout)
	return p, nil
}

// Fanout of the lower levels.
func (p *MergeParams) SetLowerFanout(lower_fanout uint32) (*MergeParams, error) {
	p.params.lower_fanout = C.uint32_t(lower_fanout)
	return p, nil
}

// Fraction of points selected as leaders.
func (p *MergeParams) SetLeaderFraction(leader_fraction float64) (*MergeParams, error) {
	p.params.leader_fraction = C.double(leader_fraction)
	return p, nil
}

// Maximum number of leaders.
func (p *MergeParams) SetMaxLeaders(max_leaders uint32) (*MergeParams, error) {
	p.params.max_leaders = C.uint32_t(max_leaders)
	return p, nil
}

// Size of the leaf partitions.
func (p *MergeParams) SetLeafSize(leaf_size uint32) (*MergeParams, error) {
	p.params.leaf_size = C.uint32_t(leaf_size)
	return p, nil
}

// Degree used within the leaf partitions.
func (p *MergeParams) SetLeafDegree(leaf_degree uint32) (*MergeParams, error) {
	p.params.leaf_degree = C.uint32_t(leaf_degree)
	return p, nil
}

// Destroys MergeParams
func (p *MergeParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraMergeParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	return nil
}
