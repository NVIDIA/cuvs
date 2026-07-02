# Interoperability

## DLPack (C)

Approximate nearest neighbor (ANN) indexes provide an interface to build and search an index via a C API. [DLPack v0.8](https://github.com/dmlc/dlpack/blob/main/README.md), a tensor interface framework, is used as the standard to interact with our C API.

Representing a tensor with DLPack is simple, as it is a POD struct that stores information about the tensor at runtime. At the moment, `DLManagedTensor` from DLPack v0.8 is compatible with out C API however we will soon upgrade to `DLManagedTensorVersioned` from DLPack v1.0 as it will help us maintain ABI and API compatibility.

Here's an example on how to represent device memory using `DLManagedTensor`:

```c
#include <dlpack/dlpack.h>

// Create data representation in host memory
float dataset[2][1] = {{0.2, 0.1}};
// copy data to device memory
float *dataset_dev;
cuvsRMMAlloc(&dataset_dev, sizeof(float) * 2 * 1);
cudaMemcpy(dataset_dev, dataset, sizeof(float) * 2 * 1, cudaMemcpyDefault);

// Use DLPack for representing the data as a tensor
DLManagedTensor dataset_tensor;
dataset_tensor.dl_tensor.data               = dataset;
dataset_tensor.dl_tensor.device.device_type = kDLCUDA;
dataset_tensor.dl_tensor.ndim               = 2;
dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
dataset_tensor.dl_tensor.dtype.bits         = 32;
dataset_tensor.dl_tensor.dtype.lanes        = 1;
int64_t dataset_shape[2]                    = {2, 1};
dataset_tensor.dl_tensor.shape              = dataset_shape;
dataset_tensor.dl_tensor.strides            = nullptr;

// free memory after use
cuvsRMMFree(dataset_dev);
```

Please refer to [NVIDIA cuVS C API documentation](/api-reference/c-api-documentation) to learn more.

## Multi-dimensional span (C++)

NVIDIA cuVS is built on top of the GPU-accelerated machine learning and data mining primitives in the [RAFT](https://github.com/rapidsai/raft) library. Most of the C++ APIs in NVIDIA cuVS accept [mdspan](https://arxiv.org/abs/2010.06474) multi-dimensional array view for representing data in higher dimensions similar to the `ndarray` in the Numpy Python library. RAFT also contains the corresponding owning `mdarray` structure, which simplifies the allocation and management of multi-dimensional data in both host and device (GPU) memory.

The `mdarray` is an owning object that forms a convenience layer over RMM and can be constructed in RAFT using a number of different helper functions:

```c++
#include <raft/core/device_mdarray.hpp>

int n_rows = 10;
int n_cols = 10;

auto scalar = raft::make_device_scalar<float>(handle, 1.0);
auto vector = raft::make_device_vector<float>(handle, n_cols);
auto matrix = raft::make_device_matrix<float>(handle, n_rows, n_cols);
```

The `mdspan` is a lightweight non-owning view that can wrap around any pointer, maintaining shape, layout, and indexing information for accessing elements.

We can construct `mdspan` instances directly from the above `mdarray` instances:

```c++
// Scalar mdspan on device
auto scalar_view = scalar.view();

// Vector mdspan on device
auto vector_view = vector.view();

// Matrix mdspan on device
auto matrix_view = matrix.view();
```

Since the `mdspan` is just a lightweight wrapper, we can also construct it from the underlying data handles in the `mdarray` instances above. We use the extent to get information about the `mdarray` or `mdspan`'s shape.

```c++
#include <raft/core/device_mdspan.hpp>

auto scalar_view = raft::make_device_scalar_view(scalar.data_handle());
auto vector_view = raft::make_device_vector_view(vector.data_handle(), vector.extent(0));
auto matrix_view = raft::make_device_matrix_view(matrix.data_handle(), matrix.extent(0), matrix.extent(1));
```

Of course, RAFT's `mdspan`/`mdarray` APIs aren't just limited to the `device`. You can also create `host` variants:

```c++
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>

int n_rows = 10;
int n_cols = 10;

auto scalar = raft::make_host_scalar<float>(handle, 1.0);
auto vector = raft::make_host_vector<float>(handle, n_cols);
auto matrix = raft::make_host_matrix<float>(handle, n_rows, n_cols);

auto scalar_view = raft::make_host_scalar_view(scalar.data_handle());
auto vector_view = raft::make_host_vector_view(vector.data_handle(), vector.extent(0));
auto matrix_view = raft::make_host_matrix_view(matrix.data_handle(), matrix.extent(0), matrix.extent(1));
```

Please refer to RAFT's [mdspan documentation](https://docs.rapids.ai/api/raft/stable/cpp_api/mdspan/) to learn more.

## CUDA array interface (Python)

The Python APIs accept objects that expose the CUDA Array Interface, which enables interoperability with GPU array libraries such as CuPy, Numba, PyTorch, and TensorFlow without copying device memory when compatible layouts are used.

See [Using APIs in Python](/getting-started/using-cu-vs-ap-is#python) for examples of building and searching NVIDIA cuVS indexes from Python arrays.
