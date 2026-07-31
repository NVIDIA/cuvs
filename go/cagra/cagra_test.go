package cagra

import (
	"math/rand/v2"
	"testing"

	cuvs "github.com/rapidsai/cuvs/go"
)

func TestCagra(t *testing.T) {
	const (
		nDataPoints = 1024
		nFeatures   = 16
		nQueries    = 4
		k           = 4
		epsilon     = 0.001
	)
	r := rand.New(rand.NewPCG(42, 0))

	resource, _ := cuvs.NewResource(nil)
	defer resource.Close()

	testDataset := make([][]float32, nDataPoints)
	for i := range testDataset {
		testDataset[i] = make([]float32, nFeatures)
		for j := range testDataset[i] {
			testDataset[i][j] = r.Float32()
		}
	}

	dataset, err := cuvs.NewTensor(testDataset)
	if err != nil {
		t.Fatalf("error creating dataset tensor: %v", err)
	}
	defer dataset.Close()

	indexParams, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("error creating index params: %v", err)
	}
	defer indexParams.Close()

	index, err := CreateIndex()
	if err != nil {
		t.Fatalf("error creating index: %v", err)
	}
	defer index.Close()

	// Use the first 4 points from the dataset as queries : will test that we get them back
	// as their own nearest neighbor
	queries, err := cuvs.NewTensor(testDataset[:nQueries])
	if err != nil {
		t.Fatalf("error creating queries tensor: %v", err)
	}
	defer queries.Close()

	neighbors, err := cuvs.NewTensorOnDevice[uint32](&resource, []int64{int64(nQueries), int64(k)})
	if err != nil {
		t.Fatalf("error creating neighbors tensor: %v", err)
	}
	defer neighbors.Close()

	distances, err := cuvs.NewTensorOnDevice[float32](&resource, []int64{int64(nQueries), int64(k)})
	if err != nil {
		t.Fatalf("error creating distances tensor: %v", err)
	}
	defer distances.Close()

	if _, err := dataset.ToDevice(&resource); err != nil {
		t.Fatalf("error moving dataset to device: %v", err)
	}

	if err := BuildIndex(resource, indexParams, &dataset, index); err != nil {
		t.Fatalf("error building index: %v", err)
	}

	if err := resource.Sync(); err != nil {
		t.Fatalf("error syncing resource: %v", err)
	}

	if _, err := queries.ToDevice(&resource); err != nil {
		t.Fatalf("error moving queries to device: %v", err)
	}

	SearchParams, err := CreateSearchParams()
	if err != nil {
		t.Fatalf("error creating search params: %v", err)
	}
	defer SearchParams.Close()

	err = SearchIndex(resource, SearchParams, index, &queries, &neighbors, &distances, nil)
	if err != nil {
		t.Fatalf("error searching index: %v", err)
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

	for i := range neighborsSlice {
		if neighborsSlice[i][0] != uint32(i) {
			t.Error("wrong neighbor, expected", i, "got", neighborsSlice[i][0])
		}
	}

	distancesSlice, err := distances.Slice()
	if err != nil {
		t.Fatalf("error getting distances slice: %v", err)
	}

	for i := range distancesSlice {
		if distancesSlice[i][0] >= epsilon || distancesSlice[i][0] <= -epsilon {
			t.Error("distance should be close to 0, got", distancesSlice[i][0])
		}
	}
}

func TestCagraFiltering(t *testing.T) {
	const (
		nDataPoints  = 1024
		nFilteredOut = 512
		nFeatures    = 16
		nQueries     = 4
		k            = 4
		epsilon      = 0.001
	)

	resource, _ := cuvs.NewResource(nil)
	defer resource.Close()

	testDataset := make([][]float32, nDataPoints)
	for i := range testDataset {
		testDataset[i] = make([]float32, nFeatures)
		for j := range testDataset[i] {
			testDataset[i][j] = rand.Float32()
		}
	}

	dataset, err := cuvs.NewTensor(testDataset)
	if err != nil {
		t.Fatalf("error creating dataset tensor: %v", err)
	}
	defer dataset.Close()

	indexParams, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("error creating index params: %v", err)
	}
	defer indexParams.Close()

	index, _ := CreateIndex()
	defer index.Close()

	// Test queries: first 4 points (should be found without filter)
	queries1, _ := cuvs.NewTensor(testDataset[:nQueries])
	defer queries1.Close()

	// Test queries: points 512-515 (should be found with filter, not found without)
	queries2, _ := cuvs.NewTensor(testDataset[nFilteredOut:(nFilteredOut + nQueries)])
	defer queries2.Close()

	neighbors, err := cuvs.NewTensorOnDevice[uint32](&resource, []int64{int64(nQueries), int64(k)})
	if err != nil {
		t.Fatalf("error creating neighbors tensor: %v", err)
	}
	defer neighbors.Close()

	distances, err := cuvs.NewTensorOnDevice[float32](&resource, []int64{int64(nQueries), int64(k)})
	if err != nil {
		t.Fatalf("error creating distances tensor: %v", err)
	}
	defer distances.Close()

	if _, err := dataset.ToDevice(&resource); err != nil {
		t.Fatalf("error moving dataset to device: %v", err)
	}

	if err := BuildIndex(resource, indexParams, &dataset, index); err != nil {
		t.Fatalf("error building index: %v", err)
	}

	if err := resource.Sync(); err != nil {
		t.Fatalf("error syncing resource: %v", err)
	}

	SearchParams, err := CreateSearchParams()
	if err != nil {
		t.Fatalf("error creating search params: %v", err)
	}
	defer SearchParams.Close()

	// Step 1: Search without filter - first 4 points should find themselves
	if _, err := queries1.ToDevice(&resource); err != nil {
		t.Fatalf("error moving queries1 to device: %v", err)
	}

	err = SearchIndex(resource, SearchParams, index, &queries1, &neighbors, &distances, nil)
	if err != nil {
		t.Fatalf("error searching index without filter: %v", err)
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

	// Verify first 4 points found themselves without filter
	neighborsSlice, err := neighbors.Slice()
	if err != nil {
		t.Fatalf("error getting neighbors slice: %v", err)
	}

	for i := range neighborsSlice {
		if neighborsSlice[i][0] != uint32(i) {
			t.Error("without filter: wrong neighbor, expected", i, "got", neighborsSlice[i][0])
		}
	}

	// Step 2: Search with filter excluding first half - first 4 points should not be found
	allowList := make([]uint32, nDataPoints-nFilteredOut)
	for i := range allowList {
		allowList[i] = uint32(i + nFilteredOut)
	}

	if _, err := queries1.ToDevice(&resource); err != nil {
		t.Fatalf("error moving queries1 back to device: %v", err)
	}

	if _, err := neighbors.ToDevice(&resource); err != nil {
		t.Fatalf("error moving neighbors back to device: %v", err)
	}

	if _, err := distances.ToDevice(&resource); err != nil {
		t.Fatalf("error moving distances back to device: %v", err)
	}

	err = SearchIndex(resource, SearchParams, index, &queries1, &neighbors, &distances, allowList)
	if err != nil {
		t.Fatalf("error searching index with filter: %v", err)
	}

	if _, err := neighbors.ToHost(&resource); err != nil {
		t.Fatalf("error moving neighbors to host: %v", err)
	}

	if err := resource.Sync(); err != nil {
		t.Fatalf("error syncing resource: %v", err)
	}

	// Verify first 4 points are not in results when filtered
	neighborsSlice, err = neighbors.Slice()
	if err != nil {
		t.Fatalf("error getting neighbors slice: %v", err)
	}

	for i := range neighborsSlice {
		for j := range neighborsSlice[i] {
			if neighborsSlice[i][j] < uint32(nFilteredOut) {
				t.Error("with filter: found point that should be filtered out:", neighborsSlice[i][j])
			}
		}
	}

	// Step 3: Search points 512-515 with filter - they should find themselves
	if _, err := queries2.ToDevice(&resource); err != nil {
		t.Fatalf("error moving queries2 to device: %v", err)
	}

	if _, err := neighbors.ToDevice(&resource); err != nil {
		t.Fatalf("error moving neighbors back to device: %v", err)
	}

	if _, err := distances.ToDevice(&resource); err != nil {
		t.Fatalf("error moving distances back to device: %v", err)
	}

	err = SearchIndex(resource, SearchParams, index, &queries2, &neighbors, &distances, allowList)
	if err != nil {
		t.Fatalf("error searching index with filter for second query set: %v", err)
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

	neighborsSlice, err = neighbors.Slice()
	if err != nil {
		t.Fatalf("error getting neighbors slice: %v", err)
	}

	distancesSlice, err := distances.Slice()
	if err != nil {
		t.Fatalf("error getting distances slice: %v", err)
	}

	// Verify points 512-515 find themselves when filtered to second half
	for i := range neighborsSlice {
		expectedID := uint32(i + nFilteredOut)
		if neighborsSlice[i][0] != expectedID {
			t.Error("with filter: wrong neighbor for filtered query, expected", expectedID, "got", neighborsSlice[i][0])
		}
		if distancesSlice[i][0] >= epsilon || distancesSlice[i][0] <= -epsilon {
			t.Error("with filter: distance should be close to 0 for filtered query, got", distancesSlice[i][0])
		}
	}
}
