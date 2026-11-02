package cuvs_test

import (
	"testing"

	cuvs "github.com/nvidia/cuvs/go"
)

func TestMemoryResource(t *testing.T) {
	mem, err := cuvs.NewCuvsPoolMemory(60, 100, false)
	if err != nil {
		t.Fatal("Failed to create memory resource:", err)
	}

	err = mem.Close()
	if err != nil {
		t.Fatal("Failed to close memory resource:", err)
	}
}
