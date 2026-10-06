package cagra

// #include <cuvs/neighbors/cagra.h>
import "C"

import (
	"errors"

	cuvs "github.com/nvidia/cuvs/go"
)

// Supplemental parameters to search CAGRA Index
type SearchParams struct {
	params C.cuvsCagraSearchParams_t
}

type SearchAlgo int

const (
	SearchAlgoSingleCta SearchAlgo = iota
	SearchAlgoMultiCta
	SearchAlgoMultiKernel
	SearchAlgoAuto
)

type HashmapMode int

const (
	HashmapModeHash HashmapMode = iota
	HashmapModeSmall
	HashmapModeAuto
)

// Creates a new SearchParams
func CreateSearchParams() (*SearchParams, error) {
	var params C.cuvsCagraSearchParams_t

	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraSearchParamsCreate(&params)))
	if err != nil {
		return nil, err
	}

	return &SearchParams{params: params}, nil
}

// Maximum number of queries to search at the same time (batch size). Auto select when 0
func (p *SearchParams) SetMaxQueries(max_queries uintptr) (*SearchParams, error) {
	p.params.max_queries = C.size_t(max_queries)
	return p, nil
}

// Number of intermediate search results retained during the search.
// This is the main knob to adjust trade off between accuracy and search speed.
// Higher values improve the search accuracy
func (p *SearchParams) SetItopkSize(itopk_size uintptr) (*SearchParams, error) {
	p.params.itopk_size = C.size_t(itopk_size)
	return p, nil
}

// Upper limit of search iterations. Auto select when 0.
func (p *SearchParams) SetMaxIterations(max_iterations uintptr) (*SearchParams, error) {
	p.params.max_iterations = C.size_t(max_iterations)
	return p, nil
}

// Which search implementation to use.
func (p *SearchParams) SetAlgo(algo SearchAlgo) (*SearchParams, error) {
	CAlgo := C.SINGLE_CTA

	switch algo {
	case SearchAlgoSingleCta:
		CAlgo = C.SINGLE_CTA
	case SearchAlgoMultiCta:
		CAlgo = C.MULTI_CTA
	case SearchAlgoMultiKernel:
		CAlgo = C.MULTI_KERNEL
	case SearchAlgoAuto:
		CAlgo = C.AUTO
	default:
		return nil, errors.New("unsupported algo")
	}

	p.params.algo = uint32(CAlgo)

	return p, nil
}

// Number of threads used to calculate a single distance. 4, 8, 16, or 32.
func (p *SearchParams) SetTeamSize(team_size uintptr) (*SearchParams, error) {
	p.params.team_size = C.size_t(team_size)
	return p, nil
}

// Lower limit of search iterations.
func (p *SearchParams) SetMinIterations(min_iterations uintptr) (*SearchParams, error) {
	p.params.min_iterations = C.size_t(min_iterations)
	return p, nil
}

// How many nodes to search at once. Auto select when 0.
func (p *SearchParams) SetSearchWidth(search_width uintptr) (*SearchParams, error) {
	p.params.search_width = C.size_t(search_width)
	return p, nil
}

// Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0.
func (p *SearchParams) SetThreadBlockSize(thread_block_size uintptr) (*SearchParams, error) {
	p.params.thread_block_size = C.size_t(thread_block_size)
	return p, nil
}

// Hashmap type. Auto selection when AUTO.
func (p *SearchParams) SetHashmapMode(hashmap_mode HashmapMode) (*SearchParams, error) {
	CHashMode := C.AUTO_HASH

	switch hashmap_mode {
	case HashmapModeHash:
		CHashMode = C.HASH
	case HashmapModeSmall:
		CHashMode = C.SMALL
	case HashmapModeAuto:
		CHashMode = C.AUTO_HASH
	default:
		return nil, errors.New("unsupported hashmap_mode")
	}

	p.params.hashmap_mode = uint32(CHashMode)

	return p, nil
}

// Lower limit of hashmap bit length. More than 8.
func (p *SearchParams) SetHashmapMinBitlen(hashmap_min_bitlen uintptr) (*SearchParams, error) {
	p.params.hashmap_min_bitlen = C.size_t(hashmap_min_bitlen)
	return p, nil
}

// Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
func (p *SearchParams) SetHashmapMaxFillRate(hashmap_max_fill_rate float32) (*SearchParams, error) {
	p.params.hashmap_max_fill_rate = C.float(hashmap_max_fill_rate)
	return p, nil
}

// Number of iterations of initial random seed node selection. 1 or more.
func (p *SearchParams) SetNumRandomSamplings(num_random_samplings uint32) (*SearchParams, error) {
	p.params.num_random_samplings = C.uint32_t(num_random_samplings)
	return p, nil
}

// Bit mask used for initial random seed node selection.
func (p *SearchParams) SetRandXorMask(rand_xor_mask uint64) (*SearchParams, error) {
	p.params.rand_xor_mask = C.uint64_t(rand_xor_mask)
	return p, nil
}

// Destroys SearchParams
func (p *SearchParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsCagraSearchParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	return nil
}
