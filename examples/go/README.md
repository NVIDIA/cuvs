# cuVS Go Bindings

This package provides Go bindings for the cuVS (CUDA Vector Search) library.

## Prerequisites

The required dependencies can be installed with a simple command (which creates your build environment):

```bash
conda env create --name go -f conda/environments/go_cuda-133_arch-$(uname -m).yaml
conda activate go
```
You may prefer to use `mamba`, as it provides significant speedup over `conda`.

## Installation

1. Set up the required environment variables:
```bash
export CGO_CFLAGS="-I${CONDA_PREFIX}/include"
export CGO_LDFLAGS="-L${CONDA_PREFIX}/lib -lcudart_static -ldl -lrt -lcuvs -lcuvs_c"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CC=clang
```

2. Install the Go module:
```bash
go get github.com/rapidsai/cuvs/go@v26.08.00 # 25.02.00 being your desired version, selected from https://github.com/rapidsai/cuvs/tags
```
Then you can build your project with the usual `go build`.

Note: The installation will fail if the C libraries are not properly installed and the environment variables are not set correctly, as this module requires CGO compilation.

## Example Usage

```go
package main

import (
    "github.com/rapidsai/cuvs/go"
    "github.com/rapidsai/cuvs/go/cagra"
)

func main() {
    // Example code showing how to use the library
}
```
See [main.go](./main.go) for an example implementation.
