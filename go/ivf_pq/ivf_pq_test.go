package ivf_pq

import (
	"math/rand/v2"
	"testing"

	cuvs "github.com/nvidia/cuvs/go"
)

func TestIvfPq(t *testing.T) {
	const (
		nDataPoints = 1024
		nFeatures   = 16
		nQueries    = 4
		k           = 4
		epsilon     = 0.001
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

	index, _ := CreateIndex(indexParams, &dataset)
	defer index.Close()

	// use the first 4 points from the dataset as queries : will test that we get them back
	// as their own nearest neighbor
	queries, _ := cuvs.NewTensor(testDataset[:nQueries])
	defer queries.Close()

	neighbors, err := cuvs.NewTensorOnDevice[int64](&resource, []int64{int64(nQueries), int64(k)})
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

	err = SearchIndex(resource, SearchParams, index, &queries, &neighbors, &distances)
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
		println(neighborsSlice[i][0])
		if neighborsSlice[i][0] != int64(i) {
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
