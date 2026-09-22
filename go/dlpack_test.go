package cuvs_test

import (
	"math/rand"
	"reflect"
	"testing"
	"time"

	cuvs "github.com/nvidia/cuvs/go"
)

func TestDlPack(t *testing.T) {
	resource, _ := cuvs.NewResource(nil)
	rand.Seed(time.Now().UnixNano())
	NDataPoints := 256
	NFeatures := 16
	TestDataset := make([][]float32, NDataPoints)
	for i := range TestDataset {
		TestDataset[i] = make([]float32, NFeatures)
		for j := range TestDataset[i] {
			TestDataset[i][j] = float32(i)
		}
	}

	dataset, err := cuvs.NewTensor(TestDataset[:127])
	if err != nil {
		t.Fatal(err)
	}

	_, err = dataset.ToDevice(&resource)
	if err != nil {
		t.Fatal(err)
	}

	_, err = dataset.Expand(&resource, TestDataset[127:])
	if err != nil {
		t.Fatal(err)
	}

	_, err = dataset.ToHost(&resource)
	if err != nil {
		t.Fatal(err)
	}

	arr, err := dataset.Slice()
	if err != nil {
		t.Fatal(err)
	}

	for i := range arr {
		for j := range arr[i] {
			if arr[i][j] != TestDataset[i][j] {
				t.Errorf("slices don't match at [%d][%d], expected %f, got %f",
					i, j, TestDataset[i][j], arr[i][j])
			}
		}
	}
}

func TestShape(t *testing.T) {
	// Test cases with different shapes
	testCases := []struct {
		rows int
		cols int
		name string
	}{
		{10, 5, "small matrix"},
		{100, 20, "medium matrix"},
		{1000, 50, "large matrix"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create test data
			data := make([][]float32, tc.rows)
			for i := range data {
				data[i] = make([]float32, tc.cols)
				for j := range data[i] {
					data[i][j] = float32(i * j)
				}
			}

			// Create tensor
			tensor, err := cuvs.NewTensor(data)
			if err != nil {
				t.Fatalf("failed to create tensor: %v", err)
			}
			defer tensor.Close()

			// Check shape
			shape := tensor.Shape()
			expectedShape := []int64{int64(tc.rows), int64(tc.cols)}
			if !reflect.DeepEqual(shape, expectedShape) {
				t.Errorf("incorrect shape: got %v, want %v", shape, expectedShape)
			}
		})
	}
}

func TestEmptyTensor(t *testing.T) {
	// Test creating tensor with empty data
	_, err := cuvs.NewTensor([][]float32{})
	if err == nil {
		t.Error("expected error when creating tensor with empty data, got nil")
	}
}

func TestDeviceOperations(t *testing.T) {
	resource, err := cuvs.NewResource(nil)
	if err != nil {
		t.Fatal(err)
	}

	// Create test data
	data := make([][]float32, 10)
	for i := range data {
		data[i] = make([]float32, 5)
		for j := range data[i] {
			data[i][j] = float32(i * j)
		}
	}

	// Test device transfer operations
	t.Run("device transfer", func(t *testing.T) {
		tensor, err := cuvs.NewTensor(data)
		if err != nil {
			t.Fatal(err)
		}
		defer tensor.Close()

		// Transfer to device
		deviceTensor, err := tensor.ToDevice(&resource)
		if err != nil {
			t.Fatalf("failed to transfer to device: %v", err)
		}

		// Transfer back to host
		hostTensor, err := deviceTensor.ToHost(&resource)
		if err != nil {
			t.Fatalf("failed to transfer back to host: %v", err)
		}

		// Verify data
		result, err := hostTensor.Slice()
		if err != nil {
			t.Fatalf("failed to slice tensor: %v", err)
		}

		for i := range data {
			for j := range data[i] {
				if result[i][j] != data[i][j] {
					t.Errorf("data mismatch at [%d][%d]: got %f, want %f",
						i, j, result[i][j], data[i][j])
				}
			}
		}
	})
}

func TestDifferentDataTypes(t *testing.T) {
	// Test int64 tensor
	t.Run("int64", func(t *testing.T) {
		data := [][]int64{{1, 2}, {3, 4}}
		tensor, err := cuvs.NewTensor(data)
		if err != nil {
			t.Fatal(err)
		}
		defer tensor.Close()

		shape := tensor.Shape()
		if !reflect.DeepEqual(shape, []int64{2, 2}) {
			t.Errorf("incorrect shape for int64 tensor: got %v, want [2 2]", shape)
		}
	})

	// Test uint32 tensor
	t.Run("uint32", func(t *testing.T) {
		data := [][]uint32{{1, 2}, {3, 4}}
		tensor, err := cuvs.NewTensor(data)
		if err != nil {
			t.Fatal(err)
		}
		defer tensor.Close()

		shape := tensor.Shape()
		if !reflect.DeepEqual(shape, []int64{2, 2}) {
			t.Errorf("incorrect shape for uint32 tensor: got %v, want [2 2]", shape)
		}
	})
}
