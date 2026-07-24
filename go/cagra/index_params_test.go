package cagra

import (
	"testing"
)

func TestCreateIndexParams(t *testing.T) {
	params, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("Failed to create IndexParams: %v", err)
	}
	defer params.Close()

	if params == nil {
		t.Fatal("CreateIndexParams returned nil params")
	}

	if params.params == nil {
		t.Fatal("IndexParams internal params are nil")
	}
}

func TestIndexParamsSetIntermediateGraphDegree(t *testing.T) {
	params, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("Failed to create IndexParams: %v", err)
	}
	defer params.Close()

	testCases := []struct {
		name  string
		value uintptr
	}{
		{"Small degree", 32},
		{"Medium degree", 64},
		{"Large degree", 128},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetIntermediateGraphDegree(tc.value)
			if err != nil {
				t.Errorf("SetIntermediateGraphDegree failed: %v", err)
			}
			if result != params {
				t.Error("SetIntermediateGraphDegree should return the same params instance")
			}
			if uintptr(params.params.intermediate_graph_degree) != tc.value {
				t.Errorf("Expected intermediate_graph_degree %d, got %d",
					tc.value, params.params.intermediate_graph_degree)
			}
		})
	}
}

func TestIndexParamsSetGraphDegree(t *testing.T) {
	params, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("Failed to create IndexParams: %v", err)
	}
	defer params.Close()

	testCases := []struct {
		name  string
		value uintptr
	}{
		{"Small degree", 16},
		{"Medium degree", 32},
		{"Large degree", 64},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetGraphDegree(tc.value)
			if err != nil {
				t.Errorf("SetGraphDegree failed: %v", err)
			}
			if result != params {
				t.Error("SetGraphDegree should return the same params instance")
			}
			if uintptr(params.params.graph_degree) != tc.value {
				t.Errorf("Expected graph_degree %d, got %d",
					tc.value, params.params.graph_degree)
			}
		})
	}
}

func TestIndexParamsSetBuildAlgo(t *testing.T) {
	params, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("Failed to create IndexParams: %v", err)
	}
	defer params.Close()

	testCases := []struct {
		name          string
		algo          BuildAlgo
		expectedError bool
	}{
		{"IVF_PQ algorithm", IvfPq, false},
		{"NN_DESCENT algorithm", NnDescent, false},
		{"AUTO_SELECT algorithm", AutoSelect, false},
		{"Invalid algorithm", BuildAlgo(999), true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetBuildAlgo(tc.algo)

			if tc.expectedError {
				if err == nil {
					t.Error("Expected error for invalid build_algo, got none")
				}
			} else {
				if err != nil {
					t.Errorf("SetBuildAlgo failed: %v", err)
				}
				if result != params {
					t.Error("SetBuildAlgo should return the same params instance")
				}
			}
		})
	}
}

func TestIndexParamsSetNNDescentNiter(t *testing.T) {
	params, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("Failed to create IndexParams: %v", err)
	}
	defer params.Close()

	testCases := []struct {
		name  string
		value uint32
	}{
		{"Few iterations", 10},
		{"Default iterations", 20},
		{"Many iterations", 50},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetNNDescentNiter(tc.value)
			if err != nil {
				t.Errorf("SetNNDescentNiter failed: %v", err)
			}
			if result != params {
				t.Error("SetNNDescentNiter should return the same params instance")
			}
			if uint32(params.params.nn_descent_niter) != tc.value {
				t.Errorf("Expected nn_descent_niter %d, got %d",
					tc.value, params.params.nn_descent_niter)
			}
		})
	}
}

func TestIndexParamsClose(t *testing.T) {
	params, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("Failed to create IndexParams: %v", err)
	}

	err = params.Close()
	if err != nil {
		t.Errorf("Close() failed: %v", err)
	}
}

func TestBuildAlgoConstants(t *testing.T) {
	algos := []BuildAlgo{IvfPq, NnDescent, AutoSelect}

	for _, algo := range algos {
		_, exists := cBuildAlgos[algo]
		if !exists {
			t.Errorf("BuildAlgo %v not found in cBuildAlgos map", algo)
		}
	}
}
