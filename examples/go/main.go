package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	cuvs "github.com/nvidia/cuvs/go"
	"github.com/nvidia/cuvs/go/cagra"
)

func main() {
	// Initialize resources
	resources, err := cuvs.NewResource(nil)
	if err != nil {
		log.Fatalf("Failed to create resources: %v", err)
	}
	defer resources.Close()

	// Dataset
	const (
		nDatapoints = 65536
		nFeatures   = 512
		nQueries    = 4
		k           = 10
	)

	// Create random dataset
	rand.Seed(time.Now().UnixNano())
	dataset := make([][]float32, nDatapoints)
	for i := range dataset {
		dataset[i] = make([]float32, nFeatures)
		for j := range dataset[i] {
			dataset[i][j] = rand.Float32()
		}
	}

	// Create tensor from dataset
	datasetTensor, err := cuvs.NewTensor(dataset)
	if err != nil {
		log.Fatalf("Failed to create dataset tensor: %v", err)
	}
	defer datasetTensor.Close()

	// Move dataset to GPU
	if _, err := datasetTensor.ToDevice(&resources); err != nil {
		log.Fatalf("Failed to move dataset to GPU: %v", err)
	}

	// Create and configure CAGRA index
	indexParams, err := cagra.CreateIndexParams()
	if err != nil {
		log.Fatalf("Failed to create index params: %v", err)
	}
	defer indexParams.Close()

	index, err := cagra.CreateIndex()
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}
	defer index.Close()

	// Build the index
	fmt.Printf("Building index for %d vectors with %d dimensions...\n", nDatapoints, nFeatures)
	if err := cagra.BuildIndex(resources, indexParams, &datasetTensor, index); err != nil {
		log.Fatalf("Failed to build index: %v", err)
	}

	// Create query tensor (using first few vectors as queries)
	queries, err := cuvs.NewTensor(dataset[:nQueries])
	if err != nil {
		log.Fatalf("Failed to create queries tensor: %v", err)
	}
	defer queries.Close()

	// Move queries to GPU
	if _, err := queries.ToDevice(&resources); err != nil {
		log.Fatalf("Failed to move queries to GPU: %v", err)
	}

	// Create tensors for results
	neighbors, err := cuvs.NewTensorOnDevice[uint32](&resources, []int64{int64(nQueries), int64(k)})
	if err != nil {
		log.Fatalf("Failed to create neighbors tensor: %v", err)
	}
	defer neighbors.Close()

	distances, err := cuvs.NewTensorOnDevice[float32](&resources, []int64{int64(nQueries), int64(k)})
	if err != nil {
		log.Fatalf("Failed to create distances tensor: %v", err)
	}
	defer distances.Close()

	// Create search parameters
	searchParams, err := cagra.CreateSearchParams()
	if err != nil {
		log.Fatalf("Failed to create search params: %v", err)
	}
	defer searchParams.Close()

	// Perform the search
	fmt.Printf("Searching for %d nearest neighbors for %d queries...\n", k, nQueries)
	if err := cagra.SearchIndex(resources, searchParams, index, &queries, &neighbors, &distances, nil); err != nil {
		log.Fatalf("Failed to search index: %v", err)
	}

	// Get results
	if _, err := neighbors.ToHost(&resources); err != nil {
		log.Fatalf("Failed to move neighbors to host: %v", err)
	}
	if _, err := distances.ToHost(&resources); err != nil {
		log.Fatalf("Failed to move distances to host: %v", err)
	}
	resources.Sync()

	neighborsResult, err := neighbors.Slice()
	if err != nil {
		log.Fatalf("Failed to get neighbors result: %v", err)
	}
	distancesResult, err := distances.Slice()
	if err != nil {
		log.Fatalf("Failed to get distances result: %v", err)
	}

	// Print results
	fmt.Println("\nSearch Results:")
	for i := 0; i < nQueries; i++ {
		fmt.Printf("\nQuery %d:\n", i)
		fmt.Printf("Neighbors: %v\n", neighborsResult[i])
		fmt.Printf("Distances: %v\n", distancesResult[i])
	}
}
