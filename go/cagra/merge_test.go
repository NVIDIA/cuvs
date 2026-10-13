package cagra

import (
	"math/rand/v2"
	"testing"

	cuvs "github.com/nvidia/cuvs/go"
)

func TestCagraMerge(t *testing.T) {
	const (
		nDataPoints1 = 256
		nDataPoints2 = 256
		nFeatures    = 16
		nQueries     = 4
		k            = 4
		epsilon      = 0.001
	)
	r := rand.New(rand.NewPCG(7, 0))

	resource, _ := cuvs.NewResource(nil)
	defer resource.Close()

	// Build two disjoint datasets.
	dataset1 := make([][]float32, nDataPoints1)
	for i := range dataset1 {
		dataset1[i] = make([]float32, nFeatures)
		for j := range dataset1[i] {
			dataset1[i][j] = r.Float32()
		}
	}
	dataset2 := make([][]float32, nDataPoints2)
	for i := range dataset2 {
		dataset2[i] = make([]float32, nFeatures)
		for j := range dataset2[i] {
			dataset2[i][j] = r.Float32()
		}
	}

	// The merged dataset the caller must build: dataset1 || dataset2, in the
	// same order the indices will be passed to MergeIndex.
	mergedRaw := make([][]float32, 0, nDataPoints1+nDataPoints2)
	mergedRaw = append(mergedRaw, dataset1...)
	mergedRaw = append(mergedRaw, dataset2...)

	tensor1, err := cuvs.NewTensor(dataset1)
	if err != nil {
		t.Fatalf("error creating dataset1 tensor: %v", err)
	}
	defer tensor1.Close()
	tensor2, err := cuvs.NewTensor(dataset2)
	if err != nil {
		t.Fatalf("error creating dataset2 tensor: %v", err)
	}
	defer tensor2.Close()

	if _, err := tensor1.ToDevice(&resource); err != nil {
		t.Fatalf("error moving dataset1 to device: %v", err)
	}
	if _, err := tensor2.ToDevice(&resource); err != nil {
		t.Fatalf("error moving dataset2 to device: %v", err)
	}

	indexParams, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("error creating index params: %v", err)
	}
	defer indexParams.Close()

	index1, err := CreateIndex()
	if err != nil {
		t.Fatalf("error creating index1: %v", err)
	}
	defer index1.Close()
	index2, err := CreateIndex()
	if err != nil {
		t.Fatalf("error creating index2: %v", err)
	}
	defer index2.Close()

	if err := BuildIndex(resource, indexParams, &tensor1, index1); err != nil {
		t.Fatalf("error building index1: %v", err)
	}
	if err := BuildIndex(resource, indexParams, &tensor2, index2); err != nil {
		t.Fatalf("error building index2: %v", err)
	}

	if err := resource.Sync(); err != nil {
		t.Fatalf("error syncing resource: %v", err)
	}

	mergedTensor, err := cuvs.NewTensor(mergedRaw)
	if err != nil {
		t.Fatalf("error creating merged tensor: %v", err)
	}
	defer mergedTensor.Close()
	if _, err := mergedTensor.ToDevice(&resource); err != nil {
		t.Fatalf("error moving merged dataset to device: %v", err)
	}

	mergedDataset, err := MakePaddedDatasetAuto(resource, &mergedTensor)
	if err != nil {
		t.Fatalf("error making padded merged dataset: %v", err)
	}
	defer mergedDataset.Close()

	// Unfiltered merge: offsets are just the cumulative row counts.
	offsets := []int64{0, nDataPoints1, nDataPoints1 + nDataPoints2}

	mergedIndex, err := CreateIndex()
	if err != nil {
		t.Fatalf("error creating merged index: %v", err)
	}
	defer mergedIndex.Close()

	if err := MergeIndex(resource, indexParams, []*CagraIndex{index1, index2}, mergedDataset, offsets, nil, mergedIndex); err != nil {
		t.Fatalf("error merging indices: %v", err)
	}

	if err := resource.Sync(); err != nil {
		t.Fatalf("error syncing resource: %v", err)
	}

	// Query with points from both halves; each should find itself as the
	// nearest neighbor in the merged index.
	queries := make([][]float32, 0, 2*nQueries)
	queries = append(queries, dataset1[:nQueries]...)
	queries = append(queries, dataset2[:nQueries]...)
	expected := []uint32{0, 1, 2, 3, nDataPoints1, nDataPoints1 + 1, nDataPoints1 + 2, nDataPoints1 + 3}

	queriesTensor, err := cuvs.NewTensor(queries)
	if err != nil {
		t.Fatalf("error creating queries tensor: %v", err)
	}
	defer queriesTensor.Close()
	if _, err := queriesTensor.ToDevice(&resource); err != nil {
		t.Fatalf("error moving queries to device: %v", err)
	}

	neighbors, err := cuvs.NewTensorOnDevice[uint32](&resource, []int64{int64(len(queries)), int64(k)})
	if err != nil {
		t.Fatalf("error creating neighbors tensor: %v", err)
	}
	defer neighbors.Close()

	distances, err := cuvs.NewTensorOnDevice[float32](&resource, []int64{int64(len(queries)), int64(k)})
	if err != nil {
		t.Fatalf("error creating distances tensor: %v", err)
	}
	defer distances.Close()

	searchParams, err := CreateSearchParams()
	if err != nil {
		t.Fatalf("error creating search params: %v", err)
	}
	defer searchParams.Close()

	if err := SearchIndex(resource, searchParams, mergedIndex, &queriesTensor, &neighbors, &distances, nil); err != nil {
		t.Fatalf("error searching merged index: %v", err)
	}

	if _, err := neighbors.ToHost(&resource); err != nil {
		t.Fatalf("error moving neighbors to host: %v", err)
	}
	if _, err := distances.ToHost(&resource); err != nil {
		t.Fatalf("error moving distances to host: %v", err)
	}
	if err := resource.Sync(); err != nil {
		t.Fatalf("error syncing resource: %v", err)
	}

	neighborsSlice, err := neighbors.Slice()
	if err != nil {
		t.Fatalf("error getting neighbors slice: %v", err)
	}
	distancesSlice, err := distances.Slice()
	if err != nil {
		t.Fatalf("error getting distances slice: %v", err)
	}

	for i := range neighborsSlice {
		if neighborsSlice[i][0] != expected[i] {
			t.Error("wrong neighbor, expected", expected[i], "got", neighborsSlice[i][0])
		}
		if distancesSlice[i][0] >= epsilon || distancesSlice[i][0] <= -epsilon {
			t.Error("distance should be close to 0, got", distancesSlice[i][0])
		}
	}
}

func TestMergedDatasetOffsetsUnfiltered(t *testing.T) {
	const (
		nDataPoints1 = 32
		nDataPoints2 = 48
		nFeatures    = 8
	)
	r := rand.New(rand.NewPCG(11, 0))

	resource, _ := cuvs.NewResource(nil)
	defer resource.Close()

	dataset1 := make([][]float32, nDataPoints1)
	for i := range dataset1 {
		dataset1[i] = make([]float32, nFeatures)
		for j := range dataset1[i] {
			dataset1[i][j] = r.Float32()
		}
	}
	dataset2 := make([][]float32, nDataPoints2)
	for i := range dataset2 {
		dataset2[i] = make([]float32, nFeatures)
		for j := range dataset2[i] {
			dataset2[i][j] = r.Float32()
		}
	}

	tensor1, err := cuvs.NewTensor(dataset1)
	if err != nil {
		t.Fatalf("error creating dataset1 tensor: %v", err)
	}
	defer tensor1.Close()
	tensor2, err := cuvs.NewTensor(dataset2)
	if err != nil {
		t.Fatalf("error creating dataset2 tensor: %v", err)
	}
	defer tensor2.Close()

	if _, err := tensor1.ToDevice(&resource); err != nil {
		t.Fatalf("error moving dataset1 to device: %v", err)
	}
	if _, err := tensor2.ToDevice(&resource); err != nil {
		t.Fatalf("error moving dataset2 to device: %v", err)
	}

	indexParams, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("error creating index params: %v", err)
	}
	defer indexParams.Close()

	index1, err := CreateIndex()
	if err != nil {
		t.Fatalf("error creating index1: %v", err)
	}
	defer index1.Close()
	index2, err := CreateIndex()
	if err != nil {
		t.Fatalf("error creating index2: %v", err)
	}
	defer index2.Close()

	if err := BuildIndex(resource, indexParams, &tensor1, index1); err != nil {
		t.Fatalf("error building index1: %v", err)
	}
	if err := BuildIndex(resource, indexParams, &tensor2, index2); err != nil {
		t.Fatalf("error building index2: %v", err)
	}

	if err := resource.Sync(); err != nil {
		t.Fatalf("error syncing resource: %v", err)
	}

	offsets, err := MergedDatasetOffsets(resource, []*CagraIndex{index1, index2}, nil)
	if err != nil {
		t.Fatalf("error computing merged dataset offsets: %v", err)
	}

	expected := []int64{0, nDataPoints1, nDataPoints1 + nDataPoints2}
	if len(offsets) != len(expected) {
		t.Fatalf("expected %d offsets, got %d", len(expected), len(offsets))
	}
	for i := range expected {
		if offsets[i] != expected[i] {
			t.Errorf("offsets[%d]: expected %d, got %d", i, expected[i], offsets[i])
		}
	}
}
